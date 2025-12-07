import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import os
from tqdm.auto import tqdm
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import Dataset, DataLoader

# ==========================================
# 1. 설정 (Configuration)
# ==========================================
# 평가를 수행할 데이터와 학습된 모델의 경로를 지정합니다.
DATA_PATH = "/content/merge_data_labeled.csv"
MODEL_PATH = "/content/QuoteMiningDetection/model/framing_classifier/classifier_best.bin"

# ==========================================
# 2. Dataset Class
# ==========================================
class FramingDataset(Dataset):
    """
    평가 데이터를 모델 입력(Input) 형태로 변환하는 데이터셋 클래스.
    
    데이터 내에 'distorted' 컬럼 존재 여부에 따라
    - 단일 문장 입력 (Single Sequence)
    - 문장 쌍 입력 (Sentence Pair: Original vs Distorted)
    방식을 자동으로 결정합니다.
    """
    def __init__(self, df, tokenizer, max_len=256):
        self.df = df.reset_index(drop=True) # 인덱스 재정렬로 접근 오류 방지
        self.tokenizer = tokenizer
        self.max_len = max_len
        
        # 'distorted' 컬럼 유무로 입력 방식 결정 (Pair Classification 여부)
        self.use_pair = "distorted" in self.df.columns

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        article = str(row["article_text"])
        label = int(row["label"])
        
        # [Case 1] 문장 쌍(Pair) 입력인 경우
        # 예: 원문과 왜곡된 문장을 비교하여 판단해야 하는 모델
        if self.use_pair:
            original = str(row["distorted"])
            encoding = self.tokenizer(
                original, 
                article, 
                truncation=True, 
                padding="max_length", 
                max_length=self.max_len, 
                return_tensors="pt"
            )
        # [Case 2] 단일 문장 입력인 경우
        # 일반적인 텍스트 분류
        else:
            encoding = self.tokenizer(
                article, 
                truncation=True, 
                padding="max_length", 
                max_length=self.max_len, 
                return_tensors="pt"
            )

        # 모델에 들어갈 딕셔너리 생성 (Batch 차원 1 제거: squeeze)
        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
        }
        return item

# ==========================================
# 3. Model Class
# ==========================================
class FramingClassifier(nn.Module):
    """
    학습된 모델과 동일한 구조를 가진 분류기 클래스.
    가중치(weights)를 로드하기 위해서는 구조가 학습 코드와 100% 일치해야 합니다.
    """
    def __init__(self, backbone_name="roberta-base", num_classes=2, dropout_rate=0.1):
        super().__init__()
        # Pre-trained Backbone (RoBERTa)
        self.backbone = AutoModel.from_pretrained(backbone_name)
        
        self.hidden_size = 768 
        self.dropout = nn.Dropout(dropout_rate)
        
        # Classifier Head (Hidden Size -> Class 개수)
        self.classifier = nn.Linear(self.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        # Backbone 통과
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        
        # [Pooling Strategy]
        # BERT 계열은 pooler_output을, 그 외에는 CLS 토큰(first token)을 사용
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            x = outputs.pooler_output
        else:
            x = outputs.last_hidden_state[:, 0, :] # [CLS] token extraction
            
        x = self.dropout(x)
        logits = self.classifier(x) # 최종 예측값 (Logits)
        return logits

# ==========================================
# 4. Main Evaluation Logic
# ==========================================
def evaluate():
    """
    저장된 모델(.bin)을 로드하여 Test Set에 대한 성능을 평가하는 함수
    """
    # GPU 사용 가능 여부 확인
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {DEVICE}")
    print(f"[INFO] Loading Model from: {MODEL_PATH}")
    
    if not os.path.exists(MODEL_PATH):
        print(f"!!! 오류: 모델 파일이 {MODEL_PATH}에 없습니다.")
        return

    # --- 1. 데이터 로드 및 전처리 ---
    print("[INFO] Loading Data & Tokenizer...")
    if DATA_PATH.endswith('.csv'):
        df = pd.read_csv(DATA_PATH)
    else:
        df = pd.read_pickle(DATA_PATH)

    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    
    # [중요] Test Set 분리
    # 학습 시 사용한 random_state와 동일한 값을 사용하여
    # 학습에 사용되지 않은 '순수한 테스트 데이터'를 복원합니다.
    _, df_test = train_test_split(
        df, 
        test_size=0.2, 
        random_state=0,  # 학습 코드와 동일한 Seed 유지 필수
        stratify=df['label'] # 레이블 비율 유지
    )
    
    test_dataset = FramingDataset(df_test, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False) # 평가는 셔플 불필요
    
    # --- 2. 모델 로드 및 초기화 ---
    model = FramingClassifier(backbone_name="roberta-base")
    
    # 저장된 학습 가중치(State Dict) 로드
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    
    model.to(DEVICE)
    model.eval() # 평가 모드 전환 (Dropout 비활성화 등)
    
    # --- 3. 추론(Inference) 수행 ---
    print(f"[INFO] Running Inference on {len(df_test)} samples...")
    y_true, y_pred, y_probs = [], [], []
    
    # Gradient 계산 비활성화로 메모리 절약 및 속도 향상
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['label'].to(DEVICE)
            
            # 모델 예측
            logits = model(input_ids, attention_mask)
            
            # 확률 및 예측 클래스 도출
            probs = F.softmax(logits, dim=1)      # Softmax로 확률 변환
            preds = torch.argmax(logits, dim=1)   # 가장 높은 확률의 인덱스 추출
            
            # 결과 저장 (CPU로 이동 후 리스트 변환)
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())
            y_probs.extend(probs[:, 1].cpu().tolist()) # AUC 계산용 (Class 1 확률)
            
    # --- 4. 성능 지표 계산 (Metrics) ---
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro') # 클래스별 균형을 고려한 F1
    
    try:
        auc = roc_auc_score(y_true, y_probs)
    except:
        auc = 0.0
        
    # --- 5. 최종 결과 출력 ---
    print("\n" + "="*40)
    print("===== Final Evaluation Result =====")
    print("="*40)
    print(f"Accuracy    : {acc:.4f}")
    print(f"F1 Macro    : {f1_macro:.4f}")
    print(f"AUC Score   : {auc:.4f}")
    
    print("\n----- Per-class metrics -----")
    # 클래스별(0: 정상, 1: 왜곡 등) 상세 지표 출력
    label_ids = [0, 1]
    
    prec_per_class = precision_score(y_true, y_pred, labels=label_ids, average=None, zero_division=0)
    rec_per_class = recall_score(y_true, y_pred, labels=label_ids, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, labels=label_ids, average=None, zero_division=0)
    
    for i in label_ids:
        print(f"[{i}] Class {i}")
        print(f"  Precision : {prec_per_class[i]:.4f}")
        print(f"  Recall    : {rec_per_class[i]:.4f}")
        print(f"  F1-Score  : {f1_per_class[i]:.4f}")
    print("="*40)

if __name__ == "__main__":
    evaluate()

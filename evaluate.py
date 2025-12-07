import os
import argparse
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score

# ---------------------------------------------------------
# Custom Modules
# ---------------------------------------------------------
from dataset import FramingDataset
from model import FramingClassifier
from utils import load_backbone_and_tokenizer

def evaluate(args):
    """
    학습된 모델 가중치(.bin)를 로드하여 검증/테스트 데이터셋에 대한 성능을 평가합니다.
    
    Process:
        1. 데이터 로드 및 전처리 (학습 시와 동일한 환경 구성)
        2. 테스트 데이터 분리 (Random Seed 고정으로 데이터 누수 방지)
        3. 저장된 모델 가중치 로드
        4. 추론(Inference) 및 성능 지표(Metric) 산출
    """
    
    # ---------------------------------------------------------
    # 1. 환경 설정 (Device Setup)
    # ---------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")
    
    # 모델 파일 존재 확인
    if not os.path.exists(args.model_path):
        print(f"!!! 오류: 모델 파일이 없습니다 -> {args.model_path}")
        return

    # ---------------------------------------------------------
    # 2. 데이터 로드 (Data Loading)
    # ---------------------------------------------------------
    print(f"[INFO] Loading Data: {args.data_path}")
    try:
        # 피클(.pkl) 또는 CSV 파일 로드 지원
        if args.data_path.endswith('.pkl'):
            df = pd.read_pickle(args.data_path)
        else:
            df = pd.read_csv(args.data_path)
    except:
        raise ValueError("데이터 파일 경로를 확인해주세요.")

    # ---------------------------------------------------------
    # 3. 토크나이저 로드 (Load Tokenizer)
    # ---------------------------------------------------------
    # 학습 때 사용한 모델(Backbone)과 동일한 토크나이저
    dummy_args = argparse.Namespace(backbone=args.backbone, hf_model_name=args.hf_model_name)
    _, tokenizer = load_backbone_and_tokenizer(dummy_args)
    
    # ---------------------------------------------------------
    # 4. 테스트 셋 분리 (Split Test Set)
    # ---------------------------------------------------------
    # 학습 때와 동일한 seed(0)를 사용하여 테스트 셋을 정확히 복원
    _, df_test = train_test_split(
        df, 
        test_size=args.val_ratio, 
        random_state=args.split_seed, 
        stratify=df['label']
    )
    
    print(f"[INFO] Test samples: {len(df_test)}")

    # ---------------------------------------------------------
    # 5. 데이터셋 & 로더 준비 (Dataset & DataLoader)
    # ---------------------------------------------------------
    test_dataset = FramingDataset(df_test, tokenizer, args.max_len)
    # 평가 시에는 순서를 섞을 필요가 없으므로 shuffle=False
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # ---------------------------------------------------------
    # 6. 모델 초기화 및 가중치 로드 (Load Weights)
    # ---------------------------------------------------------
    print(f"[INFO] Loading Model Weights from: {args.model_path}")
    
    # 1) 모델의 구조(Architecture)를 먼저 생성합니다.
    dummy_backbone, _ = load_backbone_and_tokenizer(dummy_args) # 구조 생성을 위한 백본 로드
    model = FramingClassifier(dummy_backbone, num_classes=args.num_classes)
    
    # 2) 저장된 학습 가중치(State Dict)를 덮어씌웁니다.
    state_dict = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state_dict)
    
    model.to(device)
    model.eval()   # 평가 모드 전환 (Dropout, BatchNorm 등의 동작 변경)

    # ---------------------------------------------------------
    # 7. 추론 수행 (Inference)
    # ---------------------------------------------------------
    y_true, y_pred, y_probs = [], [], []

    # Gradients 계산을 비활성화하여 메모리 사용량 절약 및 속도 향상
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            # 데이터를 GPU(또는 해당 디바이스)로 이동
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            # 토큰 타입 아이디 (필요한 경우)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)

            # 모델 Forward
            logits = model(input_ids, attention_mask, token_type_ids)

            # 결과 산출
            probs = F.softmax(logits, dim=1)      # 확률
            preds = torch.argmax(logits, dim=1)   # 예측 클래스

            # 결과 저장 (CPU로 이동 후 리스트 변환)
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())
            y_probs.extend(probs[:, 1].cpu().tolist()) # Class 1 확률

    # ---------------------------------------------------------
    # 8. 성능 평가 및 출력 (Evaluation Metrics)
    # ---------------------------------------------------------
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro') # 클래스 불균형을 고려하여 macro average 권장
    try:
        auc = roc_auc_score(y_true, y_probs)
    except:
        auc = 0.0  # 클래스가 하나뿐일 경우 예외 처리

    print("\n" + "="*40)
    print("===== Final Evaluation Result =====")
    print("="*40)
    print(f"Accuracy    : {acc:.4f}")
    print(f"F1 Macro    : {f1:.4f}")
    print(f"AUC Score   : {auc:.4f}")

    print("\n----- Per-class metrics -----")
    label_ids = [0, 1]
    prec = precision_score(y_true, y_pred, labels=label_ids, average=None, zero_division=0)
    rec = recall_score(y_true, y_pred, labels=label_ids, average=None, zero_division=0)
    f1_cls = f1_score(y_true, y_pred, labels=label_ids, average=None, zero_division=0)

    for i in label_ids:
        print(f"[Class {i}] Precision: {prec[i]:.4f} | Recall: {rec[i]:.4f} | F1: {f1_cls[i]:.4f}")
    print("="*40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="학습된 모델 평가 스크립트")
    
    # 데이터 경로 및 모델 경로 설정
    parser.add_argument("--data_path", type=str, default="data/dataset.csv", help="평가할 데이터셋 파일 경로 (.csv))
    parser.add_argument("--model_path", type=str, default="./model_result/classifier_best.bin", help="저장된 모델 가중치 파일 경로")
    
    # 모델 하이퍼파라미터
    parser.add_argument("--backbone", type=str, default="hf", help="모델 백본 타입")
    parser.add_argument("--hf_model_name", type=str, default="roberta-base", help="HuggingFace 모델 이름")
    parser.add_argument("--max_len", type=int, default=256, help="토큰 최대 길이")
    parser.add_argument("--num_classes", type=int, default=2, help="분류할 클래스 개수")
    parser.add_argument("--batch_size", type=int, default=32, help="평가 배치 사이즈")
    
    # 데이터 분할 시드 (Test Set 복원을 위해 필수)
    parser.add_argument("--val_ratio", type=float, default=0.2, help="전체 데이터 중 테스트 셋 비율")
    parser.add_argument("--split_seed", type=int, default=0, help="Train/Test 분리 시 사용한 Random Seed")

    args = parser.parse_args()
    evaluate(args)


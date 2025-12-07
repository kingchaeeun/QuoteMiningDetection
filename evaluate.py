import os
import argparse
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score

# ★★★ 우리가 만든 모듈에서 가져오기 (핵심) ★★★
from dataset import FramingDataset
from model import FramingClassifier
from utils import load_backbone_and_tokenizer

def evaluate(args):
    # 1. 디바이스 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")
    
    # 모델 파일 존재 확인
    if not os.path.exists(args.model_path):
        print(f"!!! 오류: 모델 파일이 없습니다 -> {args.model_path}")
        return

    # 2. 데이터 로드
    print(f"[INFO] Loading Data: {args.data_path}")
    try:
        if args.data_path.endswith('.pkl'):
            df = pd.read_pickle(args.data_path)
        else:
            df = pd.read_csv(args.data_path)
    except:
        raise ValueError("데이터 파일 경로를 확인해주세요.")

    # 3. 토크나이저 로드 (학습 때와 똑같은 것 사용)
    # 평가 시 backbone 모델 객체 자체는 필요 없지만, 토크나이저는 필요하므로 로드 함수 재사용
    # (단, args.backbone 등이 train 때와 같아야 함)
    dummy_args = argparse.Namespace(backbone=args.backbone, hf_model_name=args.hf_model_name)
    backbone, tokenizer = load_backbone_and_tokenizer(dummy_args)

    # 4. 테스트 셋 분리
    # 학습 때와 동일한 seed(0)를 사용하여 테스트 셋을 정확히 복원
    _, df_test = train_test_split(
        df, 
        test_size=args.val_ratio, 
        random_state=args.split_seed, 
        stratify=df['label']
    )
    
    print(f"[INFO] Test samples: {len(df_test)}")

    # 5. 데이터셋 & 로더 준비
    test_dataset = FramingDataset(df_test, tokenizer, args.max_len)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # 6. 모델 초기화 및 가중치 로드
    # 껍데기(구조) 생성
    model = FramingClassifier(backbone, num_classes=args.num_classes)
    
    # 저장된 가중치 불러오기
    print(f"[INFO] Loading Weights from {args.model_path}")
    state_dict = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state_dict)
    
    model.to(device)
    model.eval()

    # 7. 추론 (Inference)
    y_true, y_pred, y_probs = [], [], []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            # 토큰 타입 아이디 (필요한 경우)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)

            # 모델 예측
            logits = model(input_ids, attention_mask, token_type_ids)
            
            probs = F.softmax(logits, dim=1)      # 확률
            preds = torch.argmax(logits, dim=1)   # 예측 클래스

            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())
            y_probs.extend(probs[:, 1].cpu().tolist()) # Class 1 확률

    # 8. 성능 평가 출력
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    try:
        auc = roc_auc_score(y_true, y_probs)
    except:
        auc = 0.0

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
    parser = argparse.ArgumentParser()
    
    # 평가 데이터 경로
    parser.add_argument("--data_path", type=str, default="dataset.csv")
    
    # 학습된 모델 경로 (train.py가 저장한 경로와 맞춰주세요)
    parser.add_argument("--model_path", type=str, default="./model_result/classifier_best.bin")
    
    # 모델 설정 (학습 때와 같아야 함)
    parser.add_argument("--backbone", type=str, default="hf")
    parser.add_argument("--hf_model_name", type=str, default="roberta-base")
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=32)
    
    # 데이터 분할 시드 (학습 때와 같아야 정확한 테스트셋 분리가 됨)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--split_seed", type=int, default=0)

    args = parser.parse_args()
    evaluate(args)

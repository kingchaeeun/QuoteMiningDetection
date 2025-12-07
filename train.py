import os
import argparse
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ---------------------------------------------------------
# Custom Modules
# (프로젝트 모듈화: 데이터, 모델, 유틸리티 분리)
# ---------------------------------------------------------
from dataset import FramingDataset
from model import FramingClassifier
from utils import set_seed, AverageMeter, load_backbone_and_tokenizer

def save_inference_results(model, tokenizer, valid_df, device, args):
    """
    학습이 완료된(Best) 모델을 사용하여 검증 데이터의 추론 결과를 CSV로 저장합니다.
    """
    print("\n" + "="*50)
    print("[Inference] 결과 파일 생성 중...")
    
    model.eval()   # 평가 모드 (Dropout 비활성화)
    
    pred_labels = []
    prob_distorted = [] # '왜곡(Class 1)'일 확률

    # Gradient 계산 비활성화 (메모리 절약)
    with torch.no_grad():
        for i, row in tqdm(valid_df.iterrows(), total=len(valid_df), desc="Calculating Probabilities"):
            article = str(row["article_text"])
            
            # -----------------------------------------------------
            # 데이터셋 클래스와 동일한 방식의 토큰화 수행
            # -----------------------------------------------------
            if "distorted" in valid_df.columns:
                original = str(row["distorted"])
                encoding = tokenizer(original, article, truncation=True, padding="max_length", max_length=args.max_len, return_tensors="pt")
            else:
                encoding = tokenizer(article, truncation=True, padding="max_length", max_length=args.max_len, return_tensors="pt")
            
            # GPU 이동
            input_ids = encoding["input_ids"].to(device)
            attention_mask = encoding["attention_mask"].to(device)
            token_type_ids = encoding.get("token_type_ids")
            if token_type_ids is not None: token_type_ids = token_type_ids.to(device)

            # 모델 예측
            logits = model(input_ids, attention_mask, token_type_ids)
            probs = F.softmax(logits, dim=1)

            # 결과 저장
            prob_distorted.append(probs[0][1].item())
            pred_labels.append(torch.argmax(logits, dim=1).item())
            
    # DataFrame에 결과 컬럼 추가
    valid_df['pred_label'] = pred_labels
    valid_df['prob_distorted'] = prob_distorted

    # 파일 저장
    save_path = os.path.join(args.save_dir, "inference_result.csv")
    valid_df.to_csv(save_path, index=False, encoding='utf-8-sig')
    print(f"저장 완료: {save_path}")

def main(args):
    """
    Main Training Loop
    과정: Setup -> Data Load -> Model Init -> Train/Valid Loop -> Save Best Model
    """
    # ---------------------------------------------------------
    # 1. 초기 설정 (Setup)
    # ---------------------------------------------------------
    os.makedirs(args.save_dir, exist_ok=True)
    
    # [중요] 재현성 확보를 위해 Random Seed 고정
    set_seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    # ---------------------------------------------------------
    # 2. 데이터 로드 및 분할 (Data Preparation)
    # ---------------------------------------------------------
    print(f"[INFO] Loading data: {args.data_path}")
    try:
        if args.data_path.endswith('.pkl'):
            df = pd.read_pickle(args.data_path)
        else:
            df = pd.read_csv(args.data_path)
    except Exception as e:
        raise ValueError(f"데이터 로드 실패: {e}")

    # Stratified Split
    # 레이블(0/1)의 비율을 유지하면서 Train/Valid를 나눕니다.
    # 데이터 불균형이 있을 때 필수적인 옵션입니다.
    train_df, valid_df = train_test_split(
        df, 
        test_size=args.val_ratio, 
        random_state=args.split_seed, 
        stratify=df["label"]
    )
    print(f"[INFO] Train Size: {len(train_df)} | Valid Size: {len(valid_df)}")

    # ---------------------------------------------------------
    # 3. 모델 및 토크나이저 준비 (Model Initialization)
    # ---------------------------------------------------------
    # utils.py의 헬퍼 함수를 통해 일관된 설정으로 로드
    backbone, tokenizer = load_backbone_and_tokenizer(args)

    model = FramingClassifier(backbone, num_classes=args.num_classes)
    model = model.to(device)

    # ---------------------------------------------------------
    # 4. 데이터셋 & 로더 구성 (DataLoader)
    # ---------------------------------------------------------
    train_dataset = FramingDataset(train_df, tokenizer, args.max_len)
    valid_dataset = FramingDataset(valid_df, tokenizer, args.max_len)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # ---------------------------------------------------------
    # 5. 학습 최적화 설정 (Optimizer & Loss)
    # ---------------------------------------------------------
    criterion = nn.CrossEntropyLoss().to(device)
    # AdamW: Adam에 Weight Decay를 적용하여 일반화 성능을 높인 Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    # ---------------------------------------------------------
    # 6. 학습 루프 (Training Loop)
    # ---------------------------------------------------------
    best_loss = float("inf")
    train_loss_hist, valid_loss_hist, valid_acc_hist = [], [], []

    print("[INFO] Start Training...")
    for epoch in range(1, args.epochs + 1):
        # === [Train Phase] ===
        model.train()
        train_meter = AverageMeter()  # Loss 평균 계산 유틸
        tbar = tqdm(train_loader, desc=f"Epoch {epoch} Train")
        
        for batch in tbar:
            # 1) 데이터 GPU 이동
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None: 
                token_type_ids = token_type_ids.to(device)
            
            # 2) Forward & Loss
            optimizer.zero_grad()
            logits = model(input_ids, attention_mask, token_type_ids)
            loss = criterion(logits, labels)

            # 3) Backward & Step
            loss.backward()
            optimizer.step()

            # 4) 기록
            train_meter.update(loss.item(), input_ids.size(0))
            tbar.set_postfix(loss=f"{train_meter.avg:.4f}")

        # === [Valid Phase] ===
        model.eval()
        val_meter = AverageMeter()
        preds_list, labels_list = [], []
        
        with torch.no_grad():
            for batch in tqdm(valid_loader, desc=f"Epoch {epoch} Valid"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)
                token_type_ids = batch.get("token_type_ids")
                if token_type_ids is not None: 
                    token_type_ids = token_type_ids.to(device)

                logits = model(input_ids, attention_mask, token_type_ids)
                loss = criterion(logits, labels)
                
                val_meter.update(loss.item(), input_ids.size(0))
                
                # 정확도 계산을 위한 예측값 수집
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                preds_list.extend(preds)
                labels_list.extend(labels.cpu().numpy())

        # 성능 지표 출력
        val_acc = accuracy_score(labels_list, preds_list)
        print(f"Epoch {epoch} | Train Loss: {train_meter.avg:.4f} | Val Loss: {val_meter.avg:.4f} | Val Acc: {val_acc:.4f}")

        # 로그 저장
        train_loss_hist.append(train_meter.avg)
        valid_loss_hist.append(val_meter.avg)
        valid_acc_hist.append(val_acc)

        # === [Model Saving] ===
        # Validation Loss가 개선되었을 때만 모델 저장 (Overfitting 방지)
        if val_meter.avg < best_loss:
            best_loss = val_meter.avg
            save_target = os.path.join(args.save_dir, "classifier_best.bin")
            torch.save(model.state_dict(), save_target)
            print(f"★ Best Model Updated (Loss: {best_loss:.4f}) -> Saved to {save_target}")
    
    # ---------------------------------------------------------
    # 7. 학습 종료 후 시각화 및 최종 저장
    # ---------------------------------------------------------
    # Loss Curve 시각화 (학습 추이 확인용)
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss_hist, label='Train Loss')
    plt.plot(valid_loss_hist, label='Valid Loss')
    plt.title("Training & Validation Loss Curve")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(args.save_dir, "loss_curve.png"))
    print(f"\n[INFO] Loss Curve saved.")
    
    # Best Model을 다시 로드하여 최종 추론 파일 생성
    print("[INFO] Loading Best Model for Final Inference...")
    model.load_state_dict(torch.load(os.path.join(args.save_dir, "classifier_best.bin"), map_location=device))
    save_inference_results(model, tokenizer, valid_df, device, args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RoBERTa Framing Detection Training Script")

    # [Data Params]
    parser.add_argument("--data_path", type=str, default="data/dataset.csv", help="학습 데이터 경로 (.csv/.pkl)")
    parser.add_argument("--save_dir", type=str, default="./model_result", help="모델 및 로그 저장 디렉토리")
    
    # [Model Params]
    parser.add_argument("--backbone", type=str, default="hf", help="Backbone 타입 (hf)")
    parser.add_argument("--hf_model_name", type=str, default="roberta-base", help="HuggingFace 모델명")
    parser.add_argument("--max_len", type=int, default=256, help="입력 시퀀스 최대 길이")
    parser.add_argument("--num_classes", type=int, default=2, help="분류 클래스 수")
    
    # [Training Params]
    parser.add_argument("--epochs", type=int, default=5, help="총 학습 에폭 수")
    parser.add_argument("--batch_size", type=int, default=16, help="배치 크기")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning Rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="AdamW Weight Decay")
    
    # [Split & Seed]
    parser.add_argument("--val_ratio", type=float, default=0.2, help="검증 데이터 비율")
    parser.add_argument("--seed", type=int, default=42, help="전역 Random Seed")
    parser.add_argument("--split_seed", type=int, default=0, help="Data Split 전용 Seed")
    
    # [System]
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader 워커 수")
    
    args = parser.parse_args()
    main(args)


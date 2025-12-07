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

# 분리된 모듈 임포트
from dataset import FramingDataset
from model import FramingClassifier
from utils import set_seed, AverageMeter, load_backbone_and_tokenizer

def save_inference_results(model, tokenizer, valid_df, device, args):
    """검증 데이터에 대한 확률(Probability) 계산 및 CSV 저장"""
    print("\n" + "="*50)
    print("💾 [Inference] 결과 파일 생성 중...")
    
    model.eval()
    pred_labels = []
    prob_distorted = [] # 왜곡 확률(%)
    
    with torch.no_grad():
        for i, row in tqdm(valid_df.iterrows(), total=len(valid_df), desc="Calculating Probabilities"):
            article = str(row["article_text"])
            
            # 토큰화
            if "distorted" in valid_df.columns:
                original = str(row["distorted"])
                encoding = tokenizer(original, article, truncation=True, padding="max_length", max_length=args.max_len, return_tensors="pt")
            else:
                encoding = tokenizer(article, truncation=True, padding="max_length", max_length=args.max_len, return_tensors="pt")
            
            input_ids = encoding["input_ids"].to(device)
            attention_mask = encoding["attention_mask"].to(device)
            token_type_ids = encoding.get("token_type_ids")
            if token_type_ids is not None: token_type_ids = token_type_ids.to(device)
            
            logits = model(input_ids, attention_mask, token_type_ids)
            probs = F.softmax(logits, dim=1)
            
            # Class 1 (왜곡) 확률 저장
            prob_distorted.append(probs[0][1].item())
            pred_labels.append(torch.argmax(logits, dim=1).item())

    valid_df['pred_label'] = pred_labels
    valid_df['prob_distorted'] = prob_distorted
    
    save_path = os.path.join(args.save_dir, "inference_result.csv")
    valid_df.to_csv(save_path, index=False, encoding='utf-8-sig')
    print(f"✅ 저장 완료: {save_path}")

def main(args):
    # 1. 설정 및 초기화
    os.makedirs(args.save_dir, exist_ok=True)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    # 2. 데이터 로드
    print(f"[INFO] Loading data: {args.data_path}")
    try:
        if args.data_path.endswith('.pkl'):
            df = pd.read_pickle(args.data_path)
        else:
            # csv 기본
            df = pd.read_csv(args.data_path)
    except Exception as e:
        print(f"Error: {e}")
        raise ValueError("Check file path/format.")

    # Stratified Split
    train_df, valid_df = train_test_split(
        df, test_size=args.val_ratio, random_state=args.split_seed, stratify=df["label"]
    )

    # 3. 모델 및 토크나이저 준비 (roberta-base)
    backbone, tokenizer = load_backbone_and_tokenizer(args)
    model = FramingClassifier(backbone, num_classes=args.num_classes)
    model = model.to(device)

    # 4. 데이터셋 & 로더
    train_dataset = FramingDataset(train_df, tokenizer, args.max_len)
    valid_dataset = FramingDataset(valid_df, tokenizer, args.max_len)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # 5. 학습 설정
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    # 6. 학습 루프
    best_loss = float("inf")
    train_loss_hist, valid_loss_hist, valid_acc_hist = [], [], []

    print("[INFO] Start Training...")
    for epoch in range(1, args.epochs + 1):
        # --- Train ---
        model.train()
        train_meter = AverageMeter()
        tbar = tqdm(train_loader, desc=f"Epoch {epoch} Train")
        
        for batch in tbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None: token_type_ids = token_type_ids.to(device)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask, token_type_ids)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_meter.update(loss.item(), input_ids.size(0))
            tbar.set_postfix(loss=f"{train_meter.avg:.4f}")

        # --- Valid ---
        model.eval()
        val_meter = AverageMeter()
        preds_list, labels_list = [], []
        
        with torch.no_grad():
            for batch in tqdm(valid_loader, desc=f"Epoch {epoch} Valid"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)
                token_type_ids = batch.get("token_type_ids")
                if token_type_ids is not None: token_type_ids = token_type_ids.to(device)

                logits = model(input_ids, attention_mask, token_type_ids)
                loss = criterion(logits, labels)
                val_meter.update(loss.item(), input_ids.size(0))
                
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                preds_list.extend(preds)
                labels_list.extend(labels.cpu().numpy())

        val_acc = accuracy_score(labels_list, preds_list)
        print(f"Epoch {epoch} | Train Loss: {train_meter.avg:.4f} | Val Loss: {val_meter.avg:.4f} | Val Acc: {val_acc:.4f}")

        # 기록 및 저장
        train_loss_hist.append(train_meter.avg)
        valid_loss_hist.append(val_meter.avg)
        valid_acc_hist.append(val_acc)

        if val_meter.avg < best_loss:
            best_loss = val_meter.avg
            torch.save(model.state_dict(), os.path.join(args.save_dir, "classifier_best.bin"))
            print("★ Best Model Saved")

    # 7. 그래프 저장 및 추론 결과 생성
    plt.plot(train_loss_hist, label='Train Loss')
    plt.plot(valid_loss_hist, label='Valid Loss')
    plt.legend()
    plt.savefig(os.path.join(args.save_dir, "loss_curve.png"))
    
    # Best Model 로드 후 추론 결과 저장
    model.load_state_dict(torch.load(os.path.join(args.save_dir, "classifier_best.bin"), map_location=device))
    save_inference_results(model, tokenizer, valid_df, device, args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # [설정] 기본값을 roberta-base로 변경 완료
    parser.add_argument("--data_path", type=str, default="dataset.csv")
    parser.add_argument("--backbone", type=str, default="hf")  # 'kobert' -> 'hf'
    parser.add_argument("--hf_model_name", type=str, default="roberta-base") # 모델명 지정
    parser.add_argument("--save_dir", type=str, default="./model_result")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split_seed", type=int, default=0)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--num_classes", type=int, default=2)
    
    args = parser.parse_args()
    main(args)

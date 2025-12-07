#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
[Project] Quote Mining Detection - Classification Training Script

이 스크립트는 데이터 로드, 전처리, 모델 정의, 학습 및 검증 과정을 하나로 통합한
All-in-One 학습 파일입니다.

주요 기능:
1. FramingDataset: CSV 데이터를 로드하여 BERT 입력 형식으로 변환 (Pair/Single 모드 지원)
2. FramingClassifier: Pre-trained Backbone(KoBERT/RoBERTa) + Classifier Head 구조 정의
3. Train Loop: CrossEntropy Loss를 사용한 학습 및 검증, Best Model 저장
4. Visualization: 학습 종료 후 Loss 및 Accuracy 곡선 시각화

사용법 예시:
1) 기본 실행 (dataset.csv 자동 로드):
   python train_classifier.py

2) 옵션 지정 실행:
   python train_classifier.py --backbone hf --hf_model_name "roberta-base" --epochs 5
"""

import os
import argparse
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from util import set_seed, AverageMeter
from kobert_transformers import get_kobert_model, get_tokenizer
from transformers import AutoModel, AutoTokenizer

## epoch 기록 위한 리스트 선언
train_loss_history = []
valid_loss_history = []
valid_acc_history = [] # 정확도 기록 추가

# =========================
# Dataset
# =========================
class FramingDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_len: int = 256):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len

        # 분류 학습이므로 label 필수
        required = {"article_text", "label"} 
        # distorted(original) 컬럼이 있으면 같이 쓰고, 없으면 article_text만 씀 (유연성 확보)
        self.use_pair = "distorted" in self.df.columns
        
        if not required.issubset(self.df.columns):
            raise ValueError(f"Dataset missing required columns: {required}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        article = str(row["article_text"])
        label = int(row["label"])
        
        # 두 문장(Pair) 입력인지 단일 문장 입력인지 확인
        if self.use_pair:
            original = str(row["distorted"])
            # Tokenizer에 두 문장을 넣으면 [CLS] A [SEP] B [SEP] 형태로 만들어줌
            encoding = self.tokenizer(
                original,
                article,
                truncation=True,
                padding="max_length",
                max_length=self.max_len,
                return_tensors="pt",
            )
        else:
            encoding = self.tokenizer(
                article,
                truncation=True,
                padding="max_length",
                max_length=self.max_len,
                return_tensors="pt",
            )

        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
        }
        
        # BERT 계열(KoBERT)은 token_type_ids가 필요할 수 있음 (RoBERTa는 무시됨)
        if "token_type_ids" in encoding:
            item["token_type_ids"] = encoding["token_type_ids"].squeeze(0)
            
        return item


# =========================
# Classifier Model (변경됨)
# =========================
class FramingClassifier(nn.Module):
    def __init__(self, backbone, num_classes=2, dropout_rate=0.1):
        super().__init__()
        self.backbone = backbone
        
        # Backbone의 hidden_size 자동 감지
        if hasattr(backbone.config, 'hidden_size'):
            self.hidden_size = backbone.config.hidden_size
        else:
            self.hidden_size = 768 # Default for BERT/RoBERTa base
            
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        # Backbone Forward
        # token_type_ids는 모델이 지원할 때만 전달
        if token_type_ids is not None and "token_type_ids" in self.backbone.forward.__code__.co_varnames:
            outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        else:
            outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        
        # CLS 토큰 추출 (Pooler Output or First Token)
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            cls_token = outputs.pooler_output
        else:
            cls_token = outputs.last_hidden_state[:, 0, :]
            
        x = self.dropout(cls_token)
        logits = self.classifier(x)
        
        return logits


# =========================
# Backbone Loader
# =========================
def load_backbone_and_tokenizer(args):
    if args.backbone == "kobert":
        print("[INFO] Using KoBERT backbone")
        backbone = get_kobert_model()
        tokenizer = get_tokenizer()
        return backbone, tokenizer

    elif args.backbone == "hf":
        if args.hf_model_name is None:
            raise ValueError("--hf_model_name must be provided for HF backbone")

        print(f"[INFO] Using HF backbone: {args.hf_model_name}")
        tokenizer = AutoTokenizer.from_pretrained(args.hf_model_name)
        backbone = AutoModel.from_pretrained(args.hf_model_name)

        return backbone, tokenizer

    else:
        raise ValueError(f"Unknown backbone type: {args.backbone}")


# =========================
# Training Function (변경됨)
# =========================
def train_classification(args):
    os.makedirs(args.save_dir, exist_ok=True)
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    # ---------- Load Data ----------
    if args.data_path.endswith(".csv"):
        df = pd.read_csv(args.data_path)
    elif args.data_path.endswith(".pkl"):
        df = pd.read_pickle(args.data_path)
    else:
        raise ValueError("Dataset must be .csv or .pkl")

    if "label" not in df.columns:
        raise ValueError("Dataset must contain column 'label'")

    train_df, valid_df = train_test_split(
        df,
        test_size=args.val_ratio,
        random_state=args.split_seed,
        stratify=df["label"],
    )
    
    print(f"[INFO] Train samples: {len(train_df)} | Valid samples: {len(valid_df)}")

    # ---------- Backbone ----------
    backbone, tokenizer = load_backbone_and_tokenizer(args)

    # 모델 초기화 (Classifier)
    model = FramingClassifier(backbone, num_classes=args.num_classes)
    model = model.to(device)

    # ---------- Dataset ----------
    train_dataset = FramingDataset(train_df, tokenizer, args.max_len)
    valid_dataset = FramingDataset(valid_df, tokenizer, args.max_len)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # ---------- Loss / Optimizer ----------
    criterion = nn.CrossEntropyLoss().to(device) # 분류 손실 함수

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    # ---------- Training Loop ----------
    best_loss = float("inf")
    print("[INFO] Start Classification Training")

    for epoch in range(1, args.epochs + 1):
        # === Train ===
        model.train()
        train_meter = AverageMeter()
        
        tbar = tqdm(train_loader, desc=f"[Epoch {epoch}] Train")

        for batch in tbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)

            # Forward
            logits = model(input_ids, attention_mask, token_type_ids)
            loss = criterion(logits, labels)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_meter.update(loss.item(), input_ids.size(0))
            tbar.set_postfix(loss=f"{train_meter.avg:.4f}")

        # === Validation ===
        model.eval()
        val_meter = AverageMeter()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(valid_loader, desc=f"[Epoch {epoch}] Valid"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)
                
                token_type_ids = batch.get("token_type_ids")
                if token_type_ids is not None:
                    token_type_ids = token_type_ids.to(device)

                logits = model(input_ids, attention_mask, token_type_ids)
                loss = criterion(logits, labels)
                
                val_meter.update(loss.item(), input_ids.size(0))
                
                # Accuracy 계산용
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())

        # Calculate Accuracy
        val_acc = accuracy_score(all_labels, all_preds)
        
        print(f"Epoch {epoch} | Train Loss: {train_meter.avg:.4f} | Val Loss: {val_meter.avg:.4f} | Val Acc: {val_acc:.4f}")
        
        ## 기록
        train_loss_history.append(train_meter.avg)
        valid_loss_history.append(val_meter.avg)
        valid_acc_history.append(val_acc)

        # Save Last Model
        last_path = os.path.join(args.save_dir, "classifier_last.bin")
        torch.save(model.state_dict(), last_path)

        # Save Best Model (Loss 기준, 필요시 Acc 기준으로 변경 가능)
        if val_meter.avg < best_loss:
            best_loss = val_meter.avg
            best_path = os.path.join(args.save_dir, "classifier_best.bin")
            torch.save(model.state_dict(), best_path)
            print(f"[INFO] Best model saved → {best_path}")

    print("[INFO] Training completed.")

    ## 그래프 그리기 (Loss & Accuracy)
    import matplotlib.pyplot as plt
    
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:red'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color=color)
    ax1.plot(train_loss_history, label="Train Loss", color=color, linestyle='-')
    ax1.plot(valid_loss_history, label="Valid Loss", color=color, linestyle='--')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    ax2.set_ylabel('Accuracy', color=color)  # we already handled the x-label with ax1
    ax2.plot(valid_acc_history, label="Valid Acc", color=color, linestyle='-')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.legend(loc='upper right')

    plt.title("Training Loss and Validation Accuracy")
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig(os.path.join(args.save_dir, "training_curve.png"))
    print("[INFO] training_curve.png saved.")


# =========================
# CLI Arguments
# =========================
def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--val_ratio", type=float, default=0.2)
    p.add_argument("--max_len", type=int, default=256)

    # Backbone
    p.add_argument("--backbone", choices=["kobert", "hf"], default="kobert")
    p.add_argument("--hf_model_name", type=str, default=None)

    # Training
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--split_seed", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--learning_rate", type=float, default=2e-5) # Fine-tuning은 LR을 낮게 잡는게 보통
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=2)

    # Classification Specific
    p.add_argument("--num_classes", type=int, default=2, help="Number of classes (e.g. 2 or 4)")

    # Save dir
    p.add_argument("--save_dir", type=str, default="./model/framing_classifier/")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_classification(args)


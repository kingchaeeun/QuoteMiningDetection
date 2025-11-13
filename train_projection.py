#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
train_projection.py

- KoBERT 또는 HuggingFace(KPF 등) 백본 선택
- original + article_text + label(4-class) 기반 Supervised Contrastive Learning
- Projection Layer 학습 후 Encoder 저장

KoBERT 사용시:
python train_projection.py --backbone kobert

KPF 사용시:
python train_projection.py --backbone hf --hf_model_name "KPF/KPF-bert-ner"

마찬가지로 data_path도 cli 기반으로 전달하게 해뒀음
ex. python train_projection.py \
    --backbone kobert \
    --data_path "./data/input_dataset.csv"
ex. python train_projection.py \
    --backbone hf \
    --hf_model_name "KPF/KPF-bert-ner" \
    --data_path "./data/framing_dataset.pkl"


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

from util import set_seed, AverageMeter
from kobert_transformers import get_kobert_model, get_tokenizer

from transformers import AutoModel, AutoTokenizer

## epoch 기록 위한 리스트 선언
train_loss_history = []
valid_loss_history = []


# =========================
# Dataset
# =========================
class FramingDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_len: int = 256):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len

        required = {"original", "article_text", "label"}
        if not required.issubset(self.df.columns):
            raise ValueError(f"Dataset missing required columns: {required}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        original = str(row["original"])
        article = str(row["article_text"])
        label = int(row["label"])

        encoding = self.tokenizer(
            original,
            article,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
        }


# =========================
# Supervised Contrastive Loss
# =========================
class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        device = features.device
        labels = labels.contiguous().view(-1, 1)
        batch_size = features.size(0)

        # same label = positive
        mask = torch.eq(labels, labels.T).float().to(device)

        logits = torch.div(features @ features.T, self.temperature)

        # remove self-contrast
        logits_mask = torch.ones_like(mask) - torch.eye(batch_size, device=device)
        mask = mask * logits_mask

        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-12)

        return -mean_log_prob_pos.mean()


# =========================
# Encoder (Backbone + Projection)
# =========================
class FramingEncoder(nn.Module):
    def __init__(self, backbone, hidden_size=100):
        super().__init__()
        self.backbone = backbone

        backbone_dim = backbone.config.hidden_size
        self.projection = nn.Sequential(
            nn.Linear(backbone_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled = out.pooler_output
        z = self.projection(pooled)
        return F.normalize(z, dim=-1)


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
# Training
# =========================
def train_projection(args):
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

    # ---------- Backbone ----------
    backbone, tokenizer = load_backbone_and_tokenizer(args)

    encoder = FramingEncoder(backbone, hidden_size=args.projection_hidden_size)
    encoder = encoder.to(device)

    if args.freeze_backbone:
        print("[INFO] Freezing backbone; training projection only")
        for p in encoder.backbone.parameters():
            p.requires_grad = False

    # ---------- Dataset ----------
    train_dataset = FramingDataset(train_df, tokenizer, args.max_len)
    valid_dataset = FramingDataset(valid_df, tokenizer, args.max_len)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)

    # ---------- Loss / Optimizer ----------
    criterion = SupConLoss(args.temperature).to(device)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, encoder.parameters()),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    # ---------- Training Loop ----------
    best_loss = float("inf")
    print("[INFO] Start Contrastive Learning")

    for epoch in range(1, args.epochs + 1):
        encoder.train()
        meter = AverageMeter()

        tbar = tqdm(train_loader, desc=f"[Epoch {epoch}] Train")

        for batch in tbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            z = encoder(input_ids, attention_mask)
            loss = criterion(z, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            meter.update(loss.item(), input_ids.size(0))
            tbar.set_postfix(loss=f"{meter.avg:.4f}")

        # ---------- Validation ----------
        encoder.eval()
        val_meter = AverageMeter()

        with torch.no_grad():
            for batch in tqdm(valid_loader, desc=f"[Epoch {epoch}] Valid"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)

                z = encoder(input_ids, attention_mask)
                loss = criterion(z, labels)
                val_meter.update(loss.item(), input_ids.size(0))

        print(f"Epoch {epoch} | Train Loss: {meter.avg:.4f} | Val Loss: {val_meter.avg:.4f}")
        ## epoch 기록 코드 추가
        train_loss_history.append(meter.avg)
        valid_loss_history.append(val_meter.avg)

        # Save last
        last_path = os.path.join(args.save_dir, "projection_encoder_last.bin")
        torch.save(encoder.state_dict(), last_path)

        # Save best
        if val_meter.avg < best_loss:
            best_loss = val_meter.avg
            best_path = os.path.join(args.save_dir, "projection_encoder_best.bin")
            torch.save(encoder.state_dict(), best_path)
            print(f"[INFO] Best model saved → {best_path}")

    print("[INFO] Training completed.")

    ## 그래프 그리는 코드 추가
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 5))
    plt.plot(train_loss_history, label="Train Loss")
    plt.plot(valid_loss_history, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Projection Layer Contrastive Learning Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.show()


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
    p.add_argument("--learning_rate", type=float, default=3e-5)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=2)

    # Projection Layer
    p.add_argument("--projection_hidden_size", type=int, default=100)
    p.add_argument("--temperature", type=float, default=0.07)
    p.add_argument("--freeze_backbone", action="store_true")

    # Save dir
    p.add_argument("--save_dir", type=str, default="./model/projection_encoder/")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_projection(args)

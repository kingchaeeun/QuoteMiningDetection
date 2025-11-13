# backbone 옵션: 커맨드 단에서 처리
# # KoBERT 백본 사용
# python train_projection.py --backbone kobert
#
# # KPF(HuggingFace) 백본 사용
# python train_projection.py --backbone hf --hf_model_name kpf-multilingual-base



#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
train_projection.py

- KoBERT(혹은 HuggingFace 백본)을 기반으로
- original + article_text + 4-class(label) 데이터셋을 이용하여
- Projection Layer를 Supervised Contrastive Learning으로 학습하고
- 프레이밍 임베딩 Encoder를 저장하는 스크립트.

데이터셋 포맷 (CSV 또는 PKL 가정):
    id, original, article_text, label
label: 0 = 정상, 1 = Topic, 2 = Lexical, 3 = Narrative
"""

import os
import argparse
from types import SimpleNamespace

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split

from util import set_seed, AverageMeter          # ✅ 레포 util 재사용
from kobert_transformers import get_kobert_model, get_tokenizer  # ✅ 레포와 동일

from transformers import AutoModel, AutoTokenizer


# =========================
#  Dataset
# =========================

class FramingDataset(Dataset):
    """
    columns: id, original, article_text, label
    original 과 article_text 두 문장을 하나의 입력으로 사용:
        [CLS] original [SEP] article_text [SEP]
    """
    def __init__(self, df: pd.DataFrame, tokenizer, max_len: int = 256):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len

        required_cols = {"original", "article_text", "label"}
        missing = required_cols - set(self.df.columns)
        if missing:
            raise ValueError(f"DataFrame is missing required columns: {missing}")

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

        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
        }
        return item


# =========================
#  SupConLoss
# =========================

class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss
    reference: https://arxiv.org/abs/2004.11362
    - features: [batch, dim]
    - labels: [batch]
    """
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        device = features.device
        batch_size = features.size(0)

        # [B, 1]
        labels = labels.contiguous().view(-1, 1)
        # [B, B] 같은 라벨이면 1, 아니면 0
        mask = torch.eq(labels, labels.T).float().to(device)

        # 유사도 행렬 (cosine 대신 dot/temperature 사용)
        logits = torch.div(features @ features.T, self.temperature)

        # 자기 자신 마스크 제거
        logits_mask = torch.ones_like(mask) - torch.eye(batch_size, device=device)
        mask = mask * logits_mask

        # 안정적인 softmax를 위한 정규화
        logits = logits - torch.max(logits, dim=1, keepdim=True)[0]
        exp_logits = torch.exp(logits) * logits_mask

        # log_prob = log( p_pos / p_all )
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

        # 양성(같은 라벨)들에 대한 평균 log_prob
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-12)

        # loss = - E[mean_log_prob_pos]
        loss = -mean_log_prob_pos.mean()
        return loss


# =========================
#  Encoder (Backbone + Projection)
# =========================

class FramingEncoder(nn.Module):
    """
    Backbone(BERT 계열) + Projection Layer
    - backbone: HuggingFace/BERT-style 모델 (pooler_output 사용)
    - projection: 768 -> hidden_size -> hidden_size
    """
    def __init__(self, backbone: nn.Module, hidden_size: int = 100):
        super().__init__()
        self.backbone = backbone
        # 768은 KoBERT 기준, 다른 모델은 config.hidden_size 사용 가능
        backbone_hidden = getattr(backbone.config, "hidden_size", 768)
        self.projection = nn.Sequential(
            nn.Linear(backbone_hidden, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, input_ids, attention_mask):
        out = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        # [batch, hidden]
        pooled = out.pooler_output
        z = self.projection(pooled)
        # contrastive 학습용 정규화
        z = F.normalize(z, dim=-1)
        return z


# =========================
#  Backbone Loader
# =========================

def load_backbone_and_tokenizer(args):
    """
    args.backbone:
        - 'kobert' : 기존 레포의 KoBERT 사용
        - 'hf'     : HuggingFace model_name 사용 (args.hf_model_name)
    """
    if args.backbone == "kobert":
        backbone = get_kobert_model()
        tokenizer = get_tokenizer()
    elif args.backbone == "hf": # TODO 사용할 hf 모델 이름으로 바꿔 넣기
        if args.hf_model_name is None:
            raise ValueError("When backbone='hf', --hf_model_name must be set.")
        tokenizer = AutoTokenizer.from_pretrained(args.hf_model_name)
        backbone = AutoModel.from_pretrained(args.hf_model_name)
    else:
        raise ValueError(f"Unknown backbone type: {args.backbone}")

    return backbone, tokenizer


# =========================
#  Train / Eval
# =========================

def train_projection(args):
    # -----------------
    # 준비
    # -----------------
    os.makedirs(args.save_dir, exist_ok=True)
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # -----------------
    # 데이터 로드
    # -----------------
    if args.data_path.endswith(".csv"):
        df = pd.read_csv(args.data_path)
    elif args.data_path.endswith(".pkl") or args.data_path.endswith(".pickle"):
        df = pd.read_pickle(args.data_path)
    else:
        raise ValueError("data_path must be .csv or .pkl/.pickle")

    if "label" not in df.columns:
        raise ValueError("Dataset must contain 'label' column (0=정상, 1=Topic, 2=Lexical, 3=Narrative).")

    # train / valid split
    train_df, valid_df = train_test_split(
        df,
        test_size=args.val_ratio,
        random_state=args.split_seed,
        stratify=df["label"],
    )

    # -----------------
    # Backbone + Tokenizer
    # -----------------
    backbone, tokenizer = load_backbone_and_tokenizer(args)
    encoder = FramingEncoder(backbone=backbone, hidden_size=args.projection_hidden_size)

    if args.freeze_backbone:
        for p in encoder.backbone.parameters():
            p.requires_grad = False
        print("[INFO] Backbone parameters are frozen. Only projection layer will be trained.")
    else:
        print("[INFO] Backbone + projection will be trained.")

    encoder = encoder.to(device)

    # -----------------
    # DataLoader
    # -----------------
    train_dataset = FramingDataset(train_df, tokenizer, max_len=args.max_len)
    valid_dataset = FramingDataset(valid_df, tokenizer, max_len=args.max_len)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # -----------------
    # Loss / Optimizer
    # -----------------
    criterion = SupConLoss(temperature=args.temperature).to(device)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, encoder.parameters()),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    # -----------------
    # Training Loop
    # -----------------
    print("[INFO] Start training projection layer (Supervised Contrastive Learning)")
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        encoder.train()
        train_loss_meter = AverageMeter()

        tbar = tqdm(train_loader, desc=f"[Epoch {epoch}/{args.epochs}] Train", ncols=100)
        for batch in tbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            z = encoder(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(z, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_meter.update(loss.item(), input_ids.size(0))
            tbar.set_postfix(loss=f"{train_loss_meter.avg:.4f}")

        # -----------------
        # Validation
        # -----------------
        encoder.eval()
        val_loss_meter = AverageMeter()
        with torch.no_grad():
            vbar = tqdm(valid_loader, desc=f"[Epoch {epoch}/{args.epochs}] Valid", ncols=100)
            for batch in vbar:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)

                z = encoder(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(z, labels)

                val_loss_meter.update(loss.item(), input_ids.size(0))
                vbar.set_postfix(loss=f"{val_loss_meter.avg:.4f}")

        print(f"[Epoch {epoch}] Train Loss: {train_loss_meter.avg:.4f} | Val Loss: {val_loss_meter.avg:.4f}")

        # -----------------
        # Checkpoint 저장
        # -----------------
        is_best = val_loss_meter.avg < best_val_loss
        if is_best:
            best_val_loss = val_loss_meter.avg
            save_path = os.path.join(args.save_dir, "projection_encoder_best.bin")
            torch.save(encoder.state_dict(), save_path)
            print(f"[INFO] Best model updated. Saved to: {save_path}")

        # 마지막 epoch도 저장 (옵션)
        last_path = os.path.join(args.save_dir, "projection_encoder_last.bin")
        torch.save(encoder.state_dict(), last_path)

    print("[INFO] Training finished.")
    print(f"[INFO] Best validation loss: {best_val_loss:.4f}")


# =========================
#  Main
# =========================

def parse_args():
    parser = argparse.ArgumentParser(description="Train Projection Layer for Framing Embedding (Supervised Contrastive Learning)")

    # Data
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to dataset file (.csv or .pkl) with columns: id, original, article_text, label")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Validation ratio")
    parser.add_argument("--max_len", type=int, default=256, help="Max sequence length")

    # Backbone
    parser.add_argument("--backbone", type=str, default="kobert",
                        choices=["kobert", "hf"],
                        help="Backbone type: 'kobert' or 'hf'(HuggingFace)")
    parser.add_argument("--hf_model_name", type=str, default=None,
                        help="HuggingFace model name (used when backbone='hf'), e.g., 'klue/roberta-base'")

    # Training
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--split_seed", type=int, default=0, help="Random seed for train/valid split")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=3e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--num_workers", type=int, default=2, help="Num workers for DataLoader")

    # Projection
    parser.add_argument("--projection_hidden_size", type=int, default=100,
                        help="Hidden size of projection layer (output embedding dim)")
    parser.add_argument("--temperature", type=float, default=0.07,
                        help="Temperature for SupConLoss")
    parser.add_argument("--freeze_backbone", action="store_true",
                        help="Freeze backbone and train only projection layer")

    # Save
    parser.add_argument("--save_dir", type=str, default="./model/projection_encoder/",
                        help="Directory to save trained encoder")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    train_projection(args)

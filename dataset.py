import torch
import pandas as pd
from torch.utils.data import Dataset

class FramingDataset(Dataset):
    """
    CSV 데이터를 읽어서 RoBERTa 모델 입력용 Tensor로 변환하는 클래스
    """
    def __init__(self, df: pd.DataFrame, tokenizer, max_len: int = 256):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len
        
        # 'distorted' 컬럼이 있으면 Pair(두 문장) 모드, 없으면 Single(한 문장) 모드
        self.use_pair = "distorted" in self.df.columns
        
        # 필수 컬럼 확인
        required = {"article_text", "label"}
        if not required.issubset(self.df.columns):
            raise ValueError(f"필수 컬럼 누락: {required}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        article = str(row["article_text"])
        label = int(row["label"])
        
        # 1. 토큰화 (Pair vs Single)
        if self.use_pair:
            original = str(row["distorted"])
            # RoBERTa Tokenizer가 알아서 [CLS]..[SEP]..[SEP] 형태로 처리함
            encoding = self.tokenizer(
                original, article, 
                truncation=True, padding="max_length", max_length=self.max_len, return_tensors="pt"
            )
        else:
            encoding = self.tokenizer(
                article, 
                truncation=True, padding="max_length", max_length=self.max_len, return_tensors="pt"
            )

        # 2. 텐서 변환 (배치 차원 제거 squeeze)
        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
        }
        
        # RoBERTa는 token_type_ids가 없을 수 있음 (있을 때만 추가)
        if "token_type_ids" in encoding:
            item["token_type_ids"] = encoding["token_type_ids"].squeeze(0)
            
        return item

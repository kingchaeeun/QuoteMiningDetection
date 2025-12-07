import torch
import pandas as pd
from torch.utils.data import Dataset

class FramingDataset(Dataset):
"""
    CSV 데이터를 읽어서 RoBERTa 모델 입력용 Tensor로 변환하는 클래스입니다.
    
    기능:
        - 문장 쌍(Sentence Pair) 학습을 전제로 합니다.
        - 입력: 'distorted'(변형된 인용문) + 'article_text'(기사 원문)
        - 출력: 모델 학습에 필요한 input_ids, attention_mask, label 텐서
    """
    def __init__(self, df: pd.DataFrame, tokenizer, max_len: int = 256):
        """
        Args:
            df (pd.DataFrame): 전처리된 데이터프레임. 
                               필수 컬럼: 'distorted', 'article_text', 'label'
            tokenizer: HuggingFace Transformers Tokenizer (예: RoBERTaTokenizer)
            max_len (int): 토큰화 시 적용할 최대 시퀀스 길이 (Default: 256)
        """
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len

        # ---------------------------------------------------------
        # 학습 모드 자동 감지 (Pair vs Single)
        # ---------------------------------------------------------
        self.use_pair = "distorted" in self.df.columns
        
        # ---------------------------------------------------------
        # 데이터 유효성 검사
        # --------------------------------------------------------
        required = {"article_text", "distorted", "label"}
        if not required.issubset(self.df.columns):
            # 어떤 컬럼이 없는지 명시적으로 에러 메시지에 띄워 디버깅을 돕습니다.
            missing = required_columns - set(self.df.columns)
            raise ValueError(f"필수 컬럼 누락: {required}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """
        주어진 인덱스(idx)의 데이터를 문장 쌍(Pair) 형태로 토큰화하여 반환합니다.
        """
        row = self.df.iloc[idx]

        # 데이터 추출 (모든 텍스트는 문자열로 변환하여 안전성 확보)
        article = str(row["article_text"])
        label = int(row["label"])
        
        # ---------------------------------------------------------
        # 1. 토큰화 진행 (Tokenization)
        # ---------------------------------------------------------
        if self.use_pair:
            # Case A: 문장 쌍 (Pair) 입력 모드
            # 'distorted'(왜곡된 문장/원문)와 'article'(기사 문장)을 함께 입력합니다.
            # Tokenizer가 자동으로 [CLS] 문장1 [SEP] 문장2 [SEP] 형태로 구성합니다.
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

        # ---------------------------------------------------------
        # 2. 텐서 차원 조정 및 결과 구성
        # ---------------------------------------------------------
        # tokenizer 결과는 (Batch=1, Seq_Len) 형태이므로 squeeze(0)하여
        # (Seq_Len,) 벡터 형태로 변환합니다. (DataLoader가 배치 차원 추가 예정)
        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
        }
        
        # BERT 등 token_type_ids(Segment ID)를 사용하는 모델을 위한 호환성 유지
        if "token_type_ids" in encoding:
            item["token_type_ids"] = encoding["token_type_ids"].squeeze(0)
            
        return item


from torch.utils.data import Dataset, DataLoader, TensorDataset
from tqdm.auto import tqdm
import torch
import numpy as np
from util import most_sim

class Contextomized_Detection_Dataset(Dataset):
    def __init__(self, args, title_texts, body_texts, label, max_seq=85):
        self.tokenizer = args.tokenizer
        self.title = []
        self.body = []
        self.label = []
        self.max_seq = max_seq
        self.max_len = args.max_len
        self.body_len = []
        
        assert len(title_texts) == len(body_texts) 
        
        # tqdm으로 진행상황 표시
        print("Tokenizing data...")
        for idx in tqdm(range(len(title_texts))):
            title = title_texts[idx]
            body = body_texts[idx]
            
            # 1. Title Tokenization
            title_input = self.tokenizer(title, padding='max_length', truncation=True,
                                         max_length=self.max_len, return_tensors='pt')
            title_input['input_ids'] = torch.squeeze(title_input['input_ids'])
            title_input['attention_mask'] = torch.squeeze(title_input['attention_mask'])
            
            # RoBERTa 등에서는 token_type_ids가 없을 수 있음 (KeyError 방지)
            if 'token_type_ids' in title_input:
                 title_input['token_type_ids'] = torch.squeeze(title_input['token_type_ids'])
            
            
            # 2. Body Tokenization
            body_input = self.tokenizer(body, padding='max_length', truncation=True,
                                        max_length=self.max_len, return_tensors='pt')
            
            # body_len 저장 (현재 구조상 1이 들어감)
            self.body_len.append(len(body_input['input_ids']))
            
            # 빈 매트릭스 생성 (max_seq x max_len)
            b_input = np.zeros((self.max_seq, self.max_len))
            b_att = np.zeros((self.max_seq, self.max_len))
            b_token = np.zeros((self.max_seq, self.max_len))
            
            # 값 채워넣기
            b_input[:len(body_input['input_ids'])] = body_input['input_ids']
            b_att[:len(body_input['attention_mask'])] = body_input['attention_mask']
            
            # [수정된 부분] token_type_ids 처리
            if 'token_type_ids' in body_input:
                b_token[:len(body_input['token_type_ids'])] = body_input['token_type_ids']
            else:
                # RoBERTa는 token_type_ids가 없으므로 0으로 둠 (이미 np.zeros라 생략 가능하지만 명시)
                # 필요하다면 아래처럼 0으로 채움 (여기선 b_token이 이미 0이라 생략해도 됨)
                pass 
            
            # Tensor 변환 및 Squeeze
            # 주의: 여기서 squeeze는 차원이 1인 경우만 제거하므로 (85, 512)는 유지됨
            b_input = torch.Tensor(b_input)
            b_att = torch.Tensor(b_att)
            b_token = torch.Tensor(b_token) # token_type_ids용
            
            body_input['input_ids'] = b_input
            body_input['attention_mask'] = b_att
            body_input['token_type_ids'] = b_token
            
            self.title.append(title_input)
            self.body.append(body_input)
            self.label.append(label[idx])
            
    def __len__(self):
        return len(self.title)
    
    def __getitem__(self, idx):
        return self.title[idx], self.body[idx], self.body_len[idx], torch.tensor(self.label[idx], dtype=torch.long)
    

# ============================================================
# DataLoader 생성 함수
# ============================================================

def create_data_loader(args, df, shuffle, drop_last):
    # 데이터 추출
    title_texts = df.article_text.to_numpy()    # 제목 (없어서 본문 재사용)
    body_texts = df.article_text.to_numpy()     # 본문
    labels = df.label.to_numpy()
    
    # max_seq 계산 (가장 긴 문단 개수 등, 여기선 단순 길이 기반 추정인 듯)
    # 실제로는 문장 분리 로직에 따라 달라질 수 있음
    # 안전하게 85로 고정하거나 계산
    max_seq = 85 
    if not df.empty:
         # 예시: 임의 계산 (필요시 수정)
         pass

    # Dataset 생성
    cd = Contextomized_Detection_Dataset(
        args, 
        title_texts=title_texts,
        body_texts=body_texts,
        label=labels,
        max_seq=max_seq,
    )

    # DataLoader 생성
    loader = DataLoader(
        cd,
        batch_size=args.batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=args.num_workers,
    )
    return loader


def make_tensorloader(args, encoder, data_loader, train=False):
    output = []
    labels = []
    
    encoder.eval()
    with torch.no_grad():
        for title, body, body_len, label in tqdm(data_loader, desc="Encoding"):
            
            # Title
            title_id = title['input_ids'].to(args.device).long()
            title_at = title['attention_mask'].to(args.device).long()
            
            # Body (Multi-sentence structure handling)
            b_ids = []
            b_atts = []
            
            # Batch 내 각 샘플에 대해 실제 길이만큼 잘라서 가져옴
            for b in range(len(body_len)):
                i = body_len[b] # 유효 문장 개수
                # body['input_ids'] shape: [Batch, Max_Seq, Max_Len]
                # 여기서 [b][:i]는 해당 배치의 b번째 샘플의 0~i번째 문장들을 가져옴
                b_id = body['input_ids'][b][:i].to(args.device).long()
                b_at = body['attention_mask'][b][:i].to(args.device).long()
                
                b_ids.append(b_id)
                b_atts.append(b_at)
            
            body_ids = torch.cat(b_ids, dim=0)
            body_atts = torch.cat(b_atts, dim=0)

            # Encoder Forward
            # (Title과 Body를 합쳐서 한 번에 인코딩하거나 구조에 따라 다름)
            outs = encoder(
                        input_ids = torch.cat([title_id, body_ids]), 
                        attention_mask = torch.cat([title_at, body_atts]),
            )

            # Similarity Calculation (util.most_sim)
            s1, s2 = most_sim(outs, args.batch_size, body_len)

            # Feature Concatenation for Classifier
            s = torch.cat([s1, s2, abs(s1-s2), s1*s2], dim=1)
            
            output.append(s)
            labels.append(label)
            
        # 전체 데이터 통합
        output = torch.cat(output, dim=0).contiguous().squeeze()
        labels = torch.cat(labels)

    # 최종 TensorDataset 반환
    linear_ds = TensorDataset(output, labels)
    linear_loader = DataLoader(linear_ds, batch_size=args.batch_size, shuffle=train, drop_last=True)
    
    return linear_loader

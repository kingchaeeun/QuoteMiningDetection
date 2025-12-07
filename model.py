import torch.nn as nn

class FramingClassifier(nn.Module):
"""
    사전 학습된 언어 모델(Pre-trained Backbone) 위에 분류 레이어(Classification Head)를 부착한 모델입니다.
    
    구조:
        [Input Text] -> [Backbone (e.g., RoBERTa)] -> [CLS Token Pooling] -> [Dropout] -> [Linear Head] -> [Logits]
    
    특징:
        - 다양한 Backbone(BERT, RoBERTa 등)의 hidden_size를 자동으로 감지하여 호환성을 높였습니다.
        - 모델마다 다른 입력 인자(token_type_ids)나 출력 형태(pooler_output)를 유연하게 처리합니다.
    """
    def __init__(self, backbone, num_classes=2, dropout_rate=0.1):
        """
        Args:
            backbone: HuggingFace Transformers의 PreTrainedModel 객체
            num_classes (int): 분류할 클래스 개수 (Default: 2)
            dropout_rate (float): Overfitting 방지를 위한 Dropout 비율
        """
        super().__init__()
        self.backbone = backbone
        
        # ---------------------------------------------------------
        # Backbone의 출력 차원(Hidden Size) 자동 감지
        # ---------------------------------------------------------
        # config 객체를 통해 차원을 동적으로 가져와 하드코딩을 방지합니다.
        if hasattr(backbone.config, 'hidden_size'):
            self.hidden_size = backbone.config.hidden_size
        else:
            self.hidden_size = 768  # Fallback

        # ---------------------------------------------------------
        # Classification Head 정의
        # ---------------------------------------------------------
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        """
        모델의 Forward Pass를 수행합니다.
        
        Returns:
            logits (torch.Tensor): Softmax 적용 전의 raw score (Batch_Size, Num_Classes)
        """
        
        # ---------------------------------------------------------
        # 1. Backbone Forwarding (모델별 호환성 처리)
        # ---------------------------------------------------------
        # BERT는 'token_type_ids'(Segment ID)가 필요하지만, RoBERTa는 필요하지 않습니다.
        # forward 메서드의 인자를 검사(inspect)하여 에러 없이 유연하게 처리합니다.
        backbone_args = {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }
        
        # 백본 모델이 token_type_ids를 인자로 받는지 확인
        if token_type_ids is not None and \
           "token_type_ids" in self.backbone.forward.__code__.co_varnames:
            backbone_args["token_type_ids"] = token_type_ids

        # Unpacking(**)을 사용하여 인자 전달
        outputs = self.backbone(**backbone_args)
        
        # ---------------------------------------------------------
        # 2. Pooling Strategy (CLS 토큰 추출)
        # ---------------------------------------------------------
        # 문장 전체의 의미를 대표하는 [CLS] 토큰의 벡터를 추출합니다.
        # 모델에 따라 pooler_output을 제공하거나 제공하지 않을 수 있어 분기 처리합니다.
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            # BERT 계열: Pooler layer(Linear+Tanh)를 거친 CLS 벡터 사용
            cls_token = outputs.pooler_output
        else:
            # RoBERTa/DistilBERT 등: Last Hidden State의 첫 번째 토큰([CLS])을 직접 슬라이싱
            # shape: (Batch_Size, Seq_Len, Hidden_Size) -> (Batch_Size, Hidden_Size)
            cls_token = outputs.last_hidden_state[:, 0, :]

        # ---------------------------------------------------------
        # 3. Classification Head
        # ---------------------------------------------------------
        x = self.dropout(cls_token)
        logits = self.classifier(x)
        
        return logits


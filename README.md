Markdown# 🔍 Quote Mining Detection (인용구 왜곡 탐지 프로젝트)

이 저장소는 뉴스 기사 내 인용구가 원문과 비교하여 왜곡되었는지 여부를 탐지하는 **딥러닝 분류 모델(Classification Model)** 프로젝트입니다.
**RoBERTa-base** 기반의 사전 학습 모델을 사용하여, 문맥을 분석하고 인용구의 **정상(Original)** 또는 **왜곡(Distorted)** 여부를 이진 분류합니다.

## 📂 Repository Structure (폴더 구조)

프로젝트는 유지보수가 쉽도록 기능별로 모듈화되어 있습니다.

```bash
QuoteMiningDetection/
├── data/                   # [데이터 폴더]
│   └── dataset.csv         # 학습 및 검증에 사용할 데이터 파일
│
├── dataset.py              # [모듈] 데이터 로드 및 전처리 (Dataset 클래스)
├── model.py                # [모듈] 모델 아키텍처 정의 (Classifier 클래스)
├── utils.py                # [모듈] 시드 고정, 로깅 등 유틸리티 함수
│
├── train.py                # [실행] 모델 학습 메인 스크립트
├── evaluate.py             # [실행] 모델 평가 메인 스크립트
│
└── README.md               # 프로젝트 설명서
🚀 Getting Started (시작하기)1. 환경 설정 (Requirements)이 프로젝트를 실행하기 위해 필요한 라이브러리를 설치합니다.(Python 3.8+ 환경 권장)Bashpip install torch transformers pandas scikit-learn tqdm matplotlib
2. 데이터 준비 (Data Preparation)data/ 폴더 내에 dataset.csv (또는 .pkl) 파일을 위치시켜 주세요.데이터셋은 반드시 아래 컬럼을 포함해야 합니다.컬럼명필수 여부설명예시article_text✅ 필수분석할 텍스트 (Target)"경제 성장률이 하락세라고 밝혔다."label✅ 필수정답 레이블 (0: 정상, 1: 왜곡)1distorted(선택)비교할 원문 텍스트 (Source)"경제 성장률이 둔화될 가능성이 있다."Tip: distorted 컬럼이 있으면 모델이 두 문장을 쌍(Pair)으로 입력받아 더 정밀하게 비교하며, 없으면 article_text만 보고 판단합니다.💻 Usage (사용 방법)1. 모델 학습 (Training)train.py를 실행하면 데이터셋을 로드하고 모델 학습을 시작합니다.학습이 완료되면 가장 성능이 좋은 모델이 자동으로 저장됩니다.기본 실행:Bashpython train.py
(기본적으로 data/dataset.csv를 읽고 roberta-base 모델로 학습합니다.)옵션 지정 실행 예시:Bashpython train.py \
    --data_path "data/dataset.csv" \
    --hf_model_name "roberta-base" \
    --epochs 5 \
    --batch_size 16
학습 결과물 (model_result/ 폴더 생성):classifier_best.bin: 학습된 최고 성능 모델 체크포인트loss_curve.png: 학습 진행 상황(Loss, Accuracy) 그래프inference_result.csv: 검증 데이터에 대한 **예측 확률(%)**이 포함된 결과표2. 모델 평가 (Evaluation)학습된 모델(classifier_best.bin)을 불러와 테스트 데이터셋에 대한 최종 성능을 측정합니다.Bashpython evaluate.py
(평가 시에는 학습 때와 동일한 시드(split_seed=0)를 사용하여, 학습에 쓰이지 않은 데이터를 자동으로 분리해 테스트합니다.)📊 Model ArchitectureBackbone: roberta-base (Hugging Face Transformers)Architecture:Input Encoding: Tokenization (Single or Pair)Encoder: RoBERTa-base layersPooling: CLS Token EmbeddingClassifier Head:Linear Layer (Hidden Size -> Hidden Size)Dropout (0.1)Output Layer (Hidden Size -> 2 Classes)Loss Function: CrossEntropyLossOptimizer: AdamW📈 Results Example학습이 완료되면 아래와 같은 성능 지표와 추론 결과를 확인할 수 있습니다.Accuracy: 9X.XX%F1-Macro: 0.XXInference Output Example:Plaintext[Sample]
▶ 정답 라벨: 왜곡(Distorted)
▶ 모델 예측: 왜곡
▶ 확률 분석: 정상 12.5% vs 왜곡 87.5%
✅ 정답입니다!

# Quote Mining Detection (인용구 왜곡 탐지 프로젝트)

이 저장소는 뉴스 기사 내 인용구가 원문과 비교하여 왜곡되었는지 여부를 탐지하는 **딥러닝 분류 모델(Classification Model)** 프로젝트입니다.
**RoBERTa-base** 기반의 사전 학습 모델을 사용하여, 문맥을 분석하고 인용구의 **정상(Original)** 또는 **왜곡(Distorted)** 여부를 이진 분류합니다.

## Repository Structure (폴더 구조)

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

Getting Started (시작하기)
1. 환경 설정 (Requirements)
이 프로젝트를 실행하기 위해 필요한 라이브러리를 설치합니다. (Python 3.8+ 환경 권장)

pip install torch transformers pandas scikit-learn tqdm matplotlib

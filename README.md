# Receipt Text Detection OCR Competition (영수증 글자 검출)
## Team 4조

| ![류지헌](https://avatars.githubusercontent.com/u/10584296?v=4) | ![김태현](https://avatars.githubusercontent.com/u/7031901?v=4) | ![박진섭](https://avatars.githubusercontent.com/u/208775216?v=4) | ![문진숙](https://avatars.githubusercontent.com/u/204665219?v=4) | ![김재덕](https://avatars.githubusercontent.com/u/33456585?v=4) |
| :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: |
|            [류지헌](https://github.com/mahomi)             |            [김태현](https://github.com/huefilm)             |            [박진섭](https://github.com/seob1504)             |            [문진숙](https://github.com/June3723)             |            [김재덕](https://github.com/ttcoaster)             |
|                   팀장, OCR 모델 아키텍처 설계<br/>DBNet 파이프라인 구현                   |                   데이터 전처리 및 증강<br/>이미지 변환 최적화                   |                   모델 학습 및 하이퍼파라미터 튜닝<br/>성능 최적화                   |                   후처리 및 평가 메트릭 개선<br/>CLEval 최적화                   |                   모델 배포 및 제출 파일 생성<br/>환경 설정 관리                   |

## 0. Overview
### Environment
- OS: Linux (x86_64)
- Python: 3.10+
- GPU: 선택사항 (CUDA 11.8+ 권장, GPU 사용 시)

### Requirements
- 의존성 관리는 `uv`로 진행하며, 프로젝트 진행 중 변경될 수 있습니다.
- 다음과 같은 주요 라이브러리를 사용합니다:
  - PyTorch, TorchVision (모델/학습)
  - Hydra, PyTorch Lightning (실험/설정/학습 루프)
  - timm, albumentations, OpenCV (백본/증강/이미지 처리)
  - NumPy, Pandas (데이터 처리)
  - CLEval 관련 유틸 (평가)
- 설치/동기화:
```bash
uv sync
```

## 1. Competiton Info

### Overview
- 과제: 영수증 이미지 내 텍스트 탐지(Text Detection)
- 평가: CLEval 기반 평가지표로 리더보드 산출
- 제공: 학습/검증/테스트 이미지와 어노테이션(JSON), 제출 포맷 예시

### Timeline

- 9월 22일 (월) 10:00 - Start Date
- 10월 16일 (목) 19:00 - Final submission deadline

## 2. Components

### Directory
프로젝트 주요 구조는 아래와 같습니다.
```
├── code
│   └── baseline_code
│       ├── configs
│       │   ├── predict.yaml
│       │   ├── test.yaml
│       │   ├── train.yaml
│       │   └── preset
│       │       ├── base.yaml
│       │       ├── example.yaml
│       │       ├── datasets
│       │       │   └── db.yaml
│       │       └── models/lightning_modules/metrics/...
│       ├── ocr
│       │   ├── datasets/ metrics/ models/ utils/
│       │   └── lightning_modules/
│       ├── runners
│       │   ├── train.py
│       │   ├── test.py
│       │   └── predict.py
│       ├── requirements.txt
│       └── README.md
├── data
│   ├── datasets
│   │   ├── images/{train,val,test}/
│   │   ├── jsons/{train.json,val.json,test.json}
│   │   └── sample_submission.csv
│   └── get_data.sh
├── AGENTS.md
└── README.md (this file)
```

## 3. Data descrption

### Dataset overview
- 이미지: `data/datasets/images/{train,val,test}`
- 어노테이션: `data/datasets/jsons/{train.json,val.json,test.json}`
- 제출 예시: `data/datasets/sample_submission.csv`

### EDA
- 바운딩 박스 분포, 텍스트 길이, 이미지 해상도 분포 등을 확인 권장
- 누락/이상 라벨 검토 및 클래스 불균형 점검

### Data Processing
- 설정 파일: `code/baseline_code/configs/preset/datasets/db.yaml`
- 주요 전처리: 리사이즈/정규화, 데이터 증강, collate 함수(`db_collate_fn.py`)
- 경로 설정: `dataset_base_path`를 로컬 데이터 루트로 지정
```yaml
dataset_base_path: "/root/dev/upstageailab-ocr-recsys-competition-ocr-4/data/datasets"
```

## 4. Modeling

### Model descrition
- 베이스라인은 DBNet 기반 구조를 채택합니다.
- 구성: `timm` 백본(encoder) + U-Net decoder + DB head + DB loss
- 설정 관리: Hydra 기반 프리셋(`preset=example`), PyTorch Lightning 학습 루프

### Modeling Process
- 학습
```bash
uv run python code/baseline_code/runners/train.py preset=example
```
- 검증/테스트 (체크포인트 지정)
```bash
uv run python code/baseline_code/runners/test.py preset=example "checkpoint_path='{checkpoint_path}'"
```
- 예측 및 제출 파일 JSON 생성
```bash
uv run python code/baseline_code/runners/predict.py preset=example "checkpoint_path='{checkpoint_path}'"
```
- 제출 포맷 변환
```bash
uv run python code/baseline_code/ocr/utils/convert_submission.py \
  --json_path {json_path} --output_path {output_path}
```

## 5. Result

### Leader Board

- _Insert Leader Board Capture_
- _Write rank and score_

### Presentation

- _Insert your presentaion file(pdf) link_

## etc

### Meeting Log

- _Insert your meeting log link like Notion or Google Docs_

### Reference
- DBNet: https://github.com/MhLiao/DB
- Hydra: https://hydra.cc/docs/intro/
- PyTorch Lightning: https://pytorch-lightning.readthedocs.io/en/latest/
- CLEval: https://github.com/clovaai/CLEval

---

<br>

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/AIBootcamp13/upstageailab-ocr-recsys-competition-ocr-4)

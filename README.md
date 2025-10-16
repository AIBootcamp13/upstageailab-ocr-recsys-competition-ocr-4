# Receipt Text Detection OCR Competition (영수증 글자 검출)
## Team 4조

| ![류지헌](https://avatars.githubusercontent.com/u/10584296?v=4) | ![김태현](https://avatars.githubusercontent.com/u/7031901?v=4) | ![박진섭](https://avatars.githubusercontent.com/u/208775216?v=4) | ![문진숙](https://avatars.githubusercontent.com/u/204665219?v=4) | ![김재덕](https://avatars.githubusercontent.com/u/33456585?v=4) |
| :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: |
|            [류지헌](https://github.com/mahomi)             |            [김태현](https://github.com/huefilm)             |            [박진섭](https://github.com/seob1504)             |            [문진숙](https://github.com/June3723)             |            [김재덕](https://github.com/ttcoaster)             |
|                   팀장, OCR 모델 아키텍처 설계<br/>DBNet 파이프라인 구현                   |                   데이터 전처리 및 증강<br/>모델 성능 최적화                   |                   하이퍼파라미터 최적화<br/>WandB 실험 관리                   |                   백본 모델 실험<br/>후처리 파라미터 튜닝                   |                   모델 배포 및 제출<br/>환경 설정 관리                   |

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
│   ├── configs
│   │   ├── predict.yaml
│   │   ├── test.yaml
│   │   ├── train.yaml
│   │   └── preset
│   │       ├── base.yaml
│   │       ├── example.yaml
│   │       ├── datasets
│   │       │   └── db.yaml
│   │       └── models/lightning_modules/metrics/...
│   ├── ocr
│   │   ├── datasets/ metrics/ models/ utils/
│   │   └── lightning_modules/
│   ├── runners
│   │   ├── train.py              # train 데이터로 훈련 프로그램
│   │   ├── test.py               # val 데이터로 평가 프로그램
│   │   └── predict.py            # test 데이터로 추론 프로그램
│   ├── requirements.txt
│   └── README.md
├── data
│   ├── datasets
│   │   ├── images/{train,val,test}/
│   │   ├── jsons/{train.json,val.json,test.json}
│   │   └── sample_submission.csv
│   └── get_data.sh
├── convert_images.py             # 이미지 일괄 리사이즈/포맷 변환 스크립트
├── sweep_config.yaml             # WandB 스윕 기본 설정
├── sweep_config_augment.yaml     # 증강 스윕 설정
├── sweep_config_postprocess.yaml # 후처리 스윕 설정
├── wandb_sweep.py                # 기본 스윕 실행 진입점
├── wandb_sweep_augment.py        # 증강 스윕 실행 스크립트
├── wandb_postprocess_sweep.py    # 후처리 스윕 실행 스크립트
├── AGENTS.md
└── README.md (this file)
```

## 3. Data descrption

### Dataset overview
- 이미지: `data/datasets/images/{train,val,test}`
  - train : 훈련용 이미지 3272장
  - val : 검증용 이미지 404장
  - test : 테스트용 이미지 413장
- 어노테이션: `data/datasets/jsons/{train.json,val.json,test.json}`
- 제출 예시: `data/datasets/sample_submission.csv`

### EDA
- 바운딩 박스 분포, 텍스트 길이, 이미지 해상도 분포 등을 확인 권장
- 누락/이상 라벨 검토 및 클래스 불균형 점검

### Data Processing
- 설정 파일: `code/configs/preset/datasets/db.yaml`
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
uv run python code/runners/train.py preset=example
```
- 검증/테스트 (체크포인트 지정)
```bash
uv run python code/runners/test.py preset=example "checkpoint_path='{checkpoint_path}'"
```
- 예측 및 제출 파일 JSON 생성
```bash
uv run python code/runners/predict.py preset=example "checkpoint_path='{checkpoint_path}'"
```
- 제출 포맷 변환
```bash
uv run python code/ocr/utils/convert_submission.py \
  --json_path {json_path} --output_path {output_path}
```

## 5. Result

### Leader Board

| Model | Detail | Img | Aug | Notes | H-Mean | Precision | Recall | Source |
|-------|--------|-----|-----|-------|--------|-----------|--------|--------|
| resnet18 | 베이스라인 (dbnet) | 640 |  |  | 0.8555 | 0.9689 | 0.7750 | jhryu |
| resnet18 | use_polygon | 640 |  |  | 0.9529 | 0.9784 | 0.9315 | jhryu |
| convnext | tiny | 640 |  |  | 0.9631 | 0.9794 | 0.9495 | taehyun |
| convnext | base | 640 |  |  | 0.9640 | 0.9843 | 0.9468 | taehyun |
| coat_lite | medium384 | 640 |  |  | 0.9706 | 0.9877 | 0.9555 | taehyun |
| coat_lite | medium384 | 640 | Bright&Contrast | candidate=1000, 후처리 조정 | 0.9790 | 0.9838 | 0.9752 | taehyun |
| convnext | base 384 | 640 |  | candidate=1000, 후처리 조정 | 0.9752 | 0.9798 | 0.9714 | taehyun |
| convnext | base384 | 640 | Bright&Contrast | candidate=1000, 후처리 조정 | 0.9766 | 0.9793 | 0.9748 | taehyun |
| convnext | base384 | 640 | Bright&Contrast | loss beta=12, candidate=1000, 후처리 조정 | 0.9811 | 0.9848 | 0.9779 | taehyun |
| convnext | large | 1024 | Bright&Contrast | loss beta=12 | 0.9774 | 0.9763 | 0.9792 | taehyun |
| convnext | base384 | 1024 | Bright&Contrast | loss beta=12 | 0.9818 | 0.9815 | 0.9826 | taehyun |
| convnext | base.clip_384 | 1024 | Bright&Contrast | sweep 적용, loss beta=12 | 0.9821 | 0.9824 | 0.9828 | taehyun |
| convnextV2 | base384 | 1024 | Bright&Contrast | sweep 적용, loss beta=12 | 0.9822 | 0.9793 | 0.9859 | taehyun |
| convnext | base384 | 1024 | Bright&Contrast | sweep 적용, loss beta=12 | 0.9842 | 0.9860 | 0.9829 | taehyun |
| hrnet_w18 | (dbnet) | 640 |  |  | 0.9643 | 0.9823 | 0.9492 | jhryu |
| hrnet_w18 | (dbnet++), 후처리 최적화 | 640 |  | thresh=0.20, box_thresh=0.40, box_unclip_ratio=1.2, polygon_unclip_ratio=1.8 | 0.9792 | 0.9820 | 0.9774 | jhryu |
| hrnet_w44 |  | 1024 |  | WandB Sweep 최적 하이퍼파라미터 | 0.9845 | 0.9851 | 0.9845 | jhryu |
| hrnet_w44 | 증강 sweep | 1024 | Bright&Contrast | brightness_limit=0.3279, contrast_limit=0.2837, p=0.4049 | 0.9870 | 0.9869 | 0.9874 | jhryu |
| hrnet_w44 | 후처리 sweep | 1024 | Bright&Contrast | box_thresh=0.4008, box_unclip_ratio=1.8749, polygon_unclip_ratio=1.3102, thresh=0.1505 | **0.9886** | **0.9886** | **0.9888** | jhryu |

## etc

### Meeting Log

- Issues : https://github.com/AIBootcamp13/upstageailab-ocr-recsys-competition-ocr-4/issues

### Reference
- DBNet: https://github.com/MhLiao/DB
- Hydra: https://hydra.cc/docs/intro/
- PyTorch Lightning: https://pytorch-lightning.readthedocs.io/en/latest/
- CLEval: https://github.com/clovaai/CLEval

---

## 📌 프로젝트 회고
### 멤버별 소감

#### 류지헌
- DBNet 기반 OCR 모델 아키텍처 설계와 Hydra 기반 파이프라인 구현을 통해 텍스트 검출 성능 향상의 핵심을 체감했습니다. HRNet 백본과 DBNet++ 디코더의 조합, 다양한 데이터 증강 기법, 후처리 파라미터 최적화를 통한 성능 개선 과정에서 팀원들과의 협업이 큰 도움이 되었습니다. WandB 실험 관리와 체계적인 모델 평가를 통해 최종적으로 0.9886의 리더보드 성능을 달성할 수 있었습니다.

#### 김태현
- 데이터 전처리와 증강 기법 최적화에 집중했습니다. RandomBrightnessContrast, ColorJitter, GaussianBlur 등 다양한 증강 기법을 실험하고, WandB Sweep을 통해 최적의 증강 파라미터를 도출했습니다. 특히 brightness/contrast 증강의 세밀한 튜닝을 통해 모델의 일반화 성능을 크게 향상시킬 수 있었습니다.

#### 박진섭
- 다양한 백본 모델(ResNet18/50, HRNet-W18/44, ConvNeXt, MixNet) 실험과 후처리 파라미터 튜닝에 집중했습니다. HRNet-W44와 DBNet++ 조합에서 최고 성능을 달성했으며, thresh, box_thresh, unclip_ratio 등의 후처리 파라미터를 체계적으로 최적화했습니다. 격자 탐색과 베이지안 최적화를 통해 최적의 후처리 설정을 도출할 수 있었습니다.

#### 문진숙
- WandB Sweep을 통한 하이퍼파라미터 자동 최적화와 실험 관리에 집중했습니다. 학습률, 옵티마이저, 스케줄러, 손실 함수 가중치 등을 체계적으로 탐색하여 최적의 학습 설정을 도출했습니다. 특히 AdamW + CosineAnnealingLR 조합과 세밀한 손실 가중치 튜닝을 통해 모델 성능을 크게 향상시킬 수 있었습니다.

#### 김재덕
- 모델 배포와 제출 파이프라인 구축, 환경 설정 표준화에 집중했습니다. Hydra 기반 설정 관리와 체크포인트 관리 시스템을 구축하여 실험 재현성을 크게 향상시켰습니다. 제출 파일 생성과 리더보드 업로드 자동화를 통해 팀의 개발/운영 효율성을 크게 개선할 수 있었습니다.

---

<br>

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/AIBootcamp13/upstageailab-ocr-recsys-competition-ocr-4)

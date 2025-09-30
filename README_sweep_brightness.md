# WandB Sweep for RandomBrightnessContrast Optimization

베스트 하이퍼파라미터는 고정하고, **RandomBrightnessContrast augmentation** 파라미터만 최적화하는 도구입니다.

## 특징

- **고정된 베스트 파라미터**: hrnet_w44, 이미지 사이즈 1024, 배치 사이즈 2 등 모든 검증된 파라미터 사용
- **최적화 대상**: RandomBrightnessContrast의 brightness_limit, contrast_limit, p 파라미터만
- **빠른 실행**: 다른 모든 설정이 고정되어 있어 augmentation 효과만 집중 분석 가능

## 설치 및 설정

### 1. 의존성 설치

```bash
# 이미 설치되어 있다면 스킵
uv add wandb python-dotenv
```

### 2. 환경변수 설정

`.env` 파일 생성:

```bash
# WandB 설정
WANDB_API_KEY=your_wandb_api_key_here
WANDB_PROJECT=OCR-Brightness-Sweep
WANDB_ENTITY=your_username_or_team

# 데이터셋 경로
DATASET_BASE_PATH=/root/dev/upstageailab-ocr-recsys-competition-ocr-4/data/datasets/

# 선택사항
WANDB_MODE=online  # online/offline/disabled
```

## 사용법

### 방법 1: Python 스크립트 사용 (권장)

```bash
# 1. Sweep 생성
uv run python wandb_sweep_brightness.py --create-sweep

# 2. Agent 실행 (출력된 sweep ID 사용)
wandb agent your_sweep_id

# 또는 한 번에 실행 (20회)
uv run python wandb_sweep_brightness.py --sweep-id your_sweep_id --count 20
```

### 방법 2: YAML 파일 사용

```bash
# 1. Sweep 생성
wandb sweep sweep_config_brightness.yaml

# 2. Agent 실행
wandb agent your_sweep_id
```

### 방법 3: 설정 확인

```bash
# Sweep 설정 미리보기
uv run python wandb_sweep_brightness.py
```

## 최적화 파라미터

### Sweep 대상 (RandomBrightnessContrast)
- `brightness_limit`: 0.1-0.3 (기본값: 0.2)
- `contrast_limit`: 0.1-0.3 (기본값: 0.2)
- `p` (적용 확률): 0.3-0.7 (기본값: 0.5)

### 고정된 베스트 파라미터

#### 모델
- **Encoder**: hrnet_w44
- **Features**: [1,2,3,4]
- **Decoder Channels**: [128,256,512,1024]

#### 이미지 & 배치
- **이미지 사이즈**: 1024x1024
- **배치 사이즈**: 2

#### 후처리
- **thresh**: 0.23105253214239585
- **box_thresh**: 0.4324259445084524
- **box_unclip_ratio**: 1.4745700672729625
- **polygon_unclip_ratio**: 1.9770744341268096

#### Loss Weights
- **negative_ratio**: 2.824132345320219
- **prob_map_loss_weight**: 3.591196851512631
- **thresh_map_loss_weight**: 8.028627860143937
- **binary_map_loss_weight**: 0.6919312670387725

#### 학습 설정
- **Optimizer**: AdamW
- **Learning Rate**: 0.0013358832166152786
- **Weight Decay**: 0.0003571900294890783
- **Scheduler**: CosineAnnealingLR (T_max=10)
- **Max Epochs**: 13

#### CollateFN
- **shrink_ratio**: 0.428584820771695
- **thresh_max**: 0.7506908133484191
- **thresh_min**: 0.33967147700431666

## 예제 실행

### 빠른 테스트 (5회 실행)
```bash
uv run python wandb_sweep_brightness.py --create-sweep
wandb agent your_sweep_id --count 5
```

### 표준 최적화 (20회 실행)
```bash
uv run python wandb_sweep_brightness.py --create-sweep
wandb agent your_sweep_id --count 20
```

### 여러 Agent 병렬 실행
```bash
# 터미널 1
wandb agent your_sweep_id

# 터미널 2
wandb agent your_sweep_id

# 터미널 3
wandb agent your_sweep_id
```

## 결과 분석

### WandB 대시보드에서 확인 가능한 메트릭:
- `val/hmean` (주요 최적화 목표)
- `val/cleval_recall`
- `val/cleval_precision`
- `train/loss`
- `val/loss`
- RandomBrightnessContrast 파라미터 (brightness_limit, contrast_limit, p)

### 베스트 결과 찾기:
```python
import wandb

# WandB에서 베스트 실행 찾기
api = wandb.Api()
runs = api.runs("your_entity/OCR-Brightness-Sweep")
best_run = max(runs, key=lambda run: run.summary.get("val/hmean", 0))

print(f"Best hmean: {best_run.summary['val/hmean']}")
print(f"Best brightness_limit: {best_run.config['brightness_limit']}")
print(f"Best contrast_limit: {best_run.config['contrast_limit']}")
print(f"Best p: {best_run.config['brightness_contrast_p']}")
```

## 문제 해결

### GPU 메모리 부족
배치 사이즈가 2로 고정되어 있지만, GPU 메모리가 부족하면:
```bash
# 더 작은 GPU로 실행
export CUDA_VISIBLE_DEVICES=0

# 또는 wandb_sweep_brightness.py에서 batch_size를 1로 수정
```

### WandB 연결 문제
```bash
# API 키 확인
wandb login

# 오프라인 모드
export WANDB_MODE=offline
```

### 의존성 문제
```bash
# 전체 재설치
uv sync
uv add wandb python-dotenv
```

## 고급 설정

### Custom Sweep 만들기
`sweep_config_brightness.yaml`을 수정하여 파라미터 범위 조정:

```yaml
parameters:
  brightness_limit:
    min: 0.15  # 더 좁은 범위
    max: 0.25

  contrast_limit:
    min: 0.15
    max: 0.25

  brightness_contrast_p:
    min: 0.4
    max: 0.6
```

### Early Termination 설정
성능이 낮은 실험을 조기 종료:

```yaml
early_terminate:
  type: hyperband
  min_iter: 3
  eta: 2
```

## Augmentation 효과 분석

RandomBrightnessContrast는 다음과 같은 효과를 가집니다:

- **brightness_limit**: 이미지 밝기 변화 범위
  - 값이 클수록 더 극단적인 밝기 변화
  - OCR에서는 너무 크면 텍스트 가독성 저하 가능

- **contrast_limit**: 이미지 대비 변화 범위
  - 값이 클수록 더 극단적인 대비 변화
  - 적절한 값은 다양한 조명 조건에 대한 robustness 향상

- **p**: augmentation 적용 확률
  - 값이 클수록 더 자주 적용
  - 너무 크면 원본 데이터 분포와 멀어질 수 있음

## 참고 자료

- [experiments.md](./experiments.md): 전체 실험 결과 및 베스트 파라미터
- [wandb_sweep.py](./wandb_sweep.py): 전체 파라미터 최적화용 sweep
- [WandB Sweep 문서](https://docs.wandb.ai/guides/sweeps)
- [Albumentations 문서](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.RandomBrightnessContrast)
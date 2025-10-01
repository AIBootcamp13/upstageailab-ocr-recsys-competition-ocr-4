# WandB Sweep for Augmentation Optimization

베스트 하이퍼파라미터는 고정하고, **다양한 이미지 증강(augmentation)** 파라미터를 최적화하는 도구입니다.

## 특징

- **고정된 베스트 파라미터**: hrnet_w44, 이미지 사이즈 1024, 배치 사이즈 2 등 모든 검증된 파라미터 사용
- **최적화 대상**: 13개 augmentation의 파라미터와 적용 확률(p)
- **포괄적 탐색**: 색상/밝기, 블러/노이즈, 이미지 품질, 날씨/조명 효과 등 다양한 증강 기법

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
WANDB_PROJECT=OCR-Augment-Sweep
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
uv run python wandb_sweep_augment.py --create-sweep

# 2. Agent 실행 (출력된 sweep ID 사용)
wandb agent your_sweep_id

# 또는 한 번에 실행 (20회)
uv run python wandb_sweep_augment.py --sweep-id your_sweep_id --count 20
```

### 방법 2: YAML 파일 사용

```bash
# 1. Sweep 생성
wandb sweep sweep_config_augment.yaml

# 2. Agent 실행
wandb agent your_sweep_id
```

### 방법 3: 설정 확인

```bash
# Sweep 설정 미리보기
uv run python wandb_sweep_augment.py
```

## 최적화 파라미터

### Sweep 대상 (13개 Augmentation)

#### 1. 색상/밝기 조정
- **RandomBrightnessContrast**: 밝기와 대비 조정
  - `brightness_limit`: 0.1-0.4
  - `contrast_limit`: 0.1-0.4
  - `p`: 0.0-0.5 (0.0 = 미적용)

- **ColorJitter**: 밝기, 대비, 채도, 색조 조정
  - `brightness`, `contrast`, `saturation`: 0.0-0.3
  - `hue`: 0.0-0.15
  - `p`: 0.0-0.5

- **RandomGamma**: 감마 보정
  - `gamma_limit`: [70-90, 110-130]
  - `p`: 0.0-0.5

- **HueSaturationValue**: HSV 색공간 조정
  - `hue_shift_limit`: 0-30
  - `sat_shift_limit`: 0-40
  - `val_shift_limit`: 0-30
  - `p`: 0.0-0.5

#### 2. 블러/노이즈
- **GaussianBlur**: 가우시안 블러
  - `blur_limit`: 3-9 (홀수)
  - `p`: 0.0-0.4

- **MotionBlur**: 모션 블러
  - `blur_limit`: 3-9 (홀수)
  - `p`: 0.0-0.4

- **GaussNoise**: 가우시안 노이즈
  - `var_limit`: [5.0-15.0, 30.0-70.0]
  - `p`: 0.0-0.4

#### 3. 이미지 품질
- **ImageCompression**: JPEG 압축
  - `quality_lower`: 60-80
  - `quality_upper`: 95-100
  - `p`: 0.0-0.4

- **Sharpen**: 선명도 조정
  - `alpha`: [0.1-0.3, 0.4-0.7]
  - `lightness`: [0.3-0.6, 0.8-1.2]
  - `p`: 0.0-0.4

- **Downscale**: 해상도 다운스케일 후 업스케일
  - `scale_min`: 0.6-0.8
  - `scale_max`: 0.85-0.98
  - `p`: 0.0-0.4

#### 4. 날씨/조명 효과
- **RandomShadow**: 그림자 효과
  - `num_shadows`: [1-2, 2-4]
  - `shadow_dimension`: 3-7
  - `p`: 0.0-0.4

- **PlasmaShadow**: 플라즈마 그림자 효과
  - `shadow_intensity_range`: [0.2-0.4, 0.6-0.8]
  - `roughness`: 2.0-5.0
  - `p`: 0.0-0.4

- **RandomFog**: 안개 효과
  - `fog_coef`: [0.05-0.15, 0.2-0.4]
  - `alpha_coef`: 0.05-0.12
  - `p`: 0.0-0.4

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
uv run python wandb_sweep_augment.py --create-sweep
wandb agent your_sweep_id --count 5
```

### 표준 최적화 (20회 실행)
```bash
uv run python wandb_sweep_augment.py --create-sweep
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
- 13개 augmentation의 모든 파라미터 (p 값 포함)

### 베스트 결과 찾기:
```python
import wandb

# WandB에서 베스트 실행 찾기
api = wandb.Api()
runs = api.runs("your_entity/OCR-Augment-Sweep")
best_run = max(runs, key=lambda run: run.summary.get("val/hmean", 0))

print(f"Best hmean: {best_run.summary['val/hmean']}")

# 색상/밝기 조정
print(f"\n=== 색상/밝기 조정 ===")
print(f"RandomBrightnessContrast p: {best_run.config['brightness_contrast_p']}")
print(f"ColorJitter p: {best_run.config['color_jitter_p']}")
print(f"RandomGamma p: {best_run.config['random_gamma_p']}")
print(f"HueSaturationValue p: {best_run.config['hsv_p']}")

# 블러/노이즈
print(f"\n=== 블러/노이즈 ===")
print(f"GaussianBlur p: {best_run.config['gaussian_blur_p']}")
print(f"MotionBlur p: {best_run.config['motion_blur_p']}")
print(f"GaussNoise p: {best_run.config['gauss_noise_p']}")

# 이미지 품질
print(f"\n=== 이미지 품질 ===")
print(f"ImageCompression p: {best_run.config['image_compression_p']}")
print(f"Sharpen p: {best_run.config['sharpen_p']}")
print(f"Downscale p: {best_run.config['downscale_p']}")

# 날씨/조명 효과
print(f"\n=== 날씨/조명 효과 ===")
print(f"RandomShadow p: {best_run.config['random_shadow_p']}")
print(f"PlasmaShadow p: {best_run.config['plasma_shadow_p']}")
print(f"RandomFog p: {best_run.config['random_fog_p']}")
```

## 문제 해결

### GPU 메모리 부족
배치 사이즈가 2로 고정되어 있지만, GPU 메모리가 부족하면:
```bash
# 더 작은 GPU로 실행
export CUDA_VISIBLE_DEVICES=0

# 또는 wandb_sweep_augment.py에서 batch_size를 1로 수정
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
`sweep_config_augment.yaml`을 수정하여 파라미터 범위 조정:

```yaml
parameters:
  # 특정 augmentation에만 집중하고 싶다면 p 값 범위를 조정
  brightness_contrast_p:
    min: 0.2
    max: 0.5

  # 다른 augmentation은 p 값을 낮게 설정
  gaussian_blur_p:
    min: 0.0
    max: 0.2

  motion_blur_p:
    min: 0.0
    max: 0.2
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

### 각 카테고리별 효과

#### 1. 색상/밝기 조정
다양한 조명 조건과 색상 환경에서 텍스트를 감지할 수 있도록 도와줍니다:
- **RandomBrightnessContrast, ColorJitter**: 실내외 다양한 조명 조건 대응
- **RandomGamma**: 디스플레이 감마 차이 보정
- **HueSaturationValue**: 색상 왜곡에 대한 robustness

#### 2. 블러/노이즈
카메라 흔들림, 초점 문제, 센서 노이즈에 대한 robustness:
- **GaussianBlur, MotionBlur**: 카메라 흔들림, 초점 불량 대응
- **GaussNoise**: 저조도 환경의 센서 노이즈 대응

#### 3. 이미지 품질
다양한 품질의 이미지에 대한 robustness:
- **ImageCompression**: JPEG 압축 아티팩트 대응
- **Sharpen**: 선명도가 다른 이미지 처리
- **Downscale**: 다양한 해상도의 이미지 대응

#### 4. 날씨/조명 효과
실제 촬영 환경의 다양한 조건 대응:
- **RandomShadow, PlasmaShadow**: 그림자 효과로 인한 가독성 저하 대응
- **RandomFog**: 안개, 스모그 등의 환경 대응

### 주의사항
- **p 값**: 너무 크면 원본 데이터 분포와 멀어져 오버피팅 가능
- **파라미터 강도**: 너무 극단적인 값은 텍스트 가독성을 저하시킬 수 있음
- **조합 효과**: 여러 augmentation이 동시에 적용될 수 있으므로 p 값의 합에 유의

## 참고 자료

- [experiments.md](./experiments.md): 전체 실험 결과 및 베스트 파라미터
- [wandb_sweep.py](./wandb_sweep.py): 전체 파라미터 최적화용 sweep
- [WandB Sweep 문서](https://docs.wandb.ai/guides/sweeps)
- [Albumentations 문서](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.RandomBrightnessContrast)
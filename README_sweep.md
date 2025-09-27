# WandB Sweep for HRNet OCR Optimization

experiments.md의 베스트 파라미터들을 기반으로 한 체계적인 하이퍼파라미터 최적화 도구입니다.

## 주요 특징

- **HRNet 계열 모델**: hrnet_w18부터 hrnet_w48까지 상위 모델 포함
- **이미지 사이즈 & 배치 사이즈**: 메모리 효율성을 고려한 자동 조정
- **후처리 파라미터**: experiments.md의 베스트 범위 (thresh=0.22, box_thresh=0.42 등) 포함
- **End-to-end 최적화**: 모델부터 후처리까지 전체 파이프라인 최적화

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
WANDB_PROJECT=OCR-HRNet-Sweep
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
python wandb_sweep.py --create-sweep

# 2. Agent 실행 (출력된 sweep ID 사용)
wandb agent your_sweep_id

# 또는 한 번에 실행
python wandb_sweep.py --sweep-id your_sweep_id --count 50
```

### 방법 2: YAML 파일 사용

```bash
# 1. Sweep 생성
wandb sweep sweep_config.yaml

# 2. Agent 실행
wandb agent your_sweep_id
```

### 방법 3: 설정 확인

```bash
# Sweep 설정 미리보기
python wandb_sweep.py
```

## 최적화 파라미터

### HRNet 모델
- `hrnet_w18`: 베스트 성능 (hmean 0.9788)
- `hrnet_w32, w40, w44, w48`: 상위 모델들

### 이미지 사이즈 & 배치 사이즈
- **이미지 사이즈**: 640, 800, 1024, 1280
- **배치 사이즈**: 4, 8, 16, 32 (자동 메모리 조정)

### 후처리 파라미터 (experiments.md 베스트 포함)
- `thresh`: 0.19-0.25 (베스트: 0.22)
- `box_thresh`: 0.40-0.49 (베스트: 0.42)
- `box_unclip_ratio`: 1.2-1.6 (베스트: 1.3)
- `polygon_unclip_ratio`: 1.6-2.0 (베스트: 1.8)

### DB Loss 파라미터
- `negative_ratio`: 2.0-4.0
- `prob_map_loss_weight`: 3.0-7.0
- `thresh_map_loss_weight`: 8.0-12.0
- `binary_map_loss_weight`: 0.5-2.0

### 학습 파라미터
- **Optimizer**: Adam (권장), AdamW
- **Learning Rate**: 0.0005-0.002
- **Weight Decay**: 0.00005-0.0005
- **Scheduler**: StepLR, CosineAnnealingLR
- **Max Epochs**: 8, 10, 12 (10이 최적)

## 예제 실행

### 빠른 테스트 (5회 실행)
```bash
python wandb_sweep.py --create-sweep
wandb agent your_sweep_id --count 5
```

### 전체 최적화 (50회 실행)
```bash
python wandb_sweep.py --create-sweep
wandb agent your_sweep_id --count 50
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

## 메모리 최적화

스크립트가 자동으로 GPU 메모리에 따라 배치 사이즈를 조정합니다:

- **1280px**: 최대 배치 사이즈 4
- **1024px**: 최대 배치 사이즈 8
- **800px**: 최대 배치 사이즈 16
- **640px**: 최대 배치 사이즈 32

## 결과 분석

### WandB 대시보드에서 확인 가능한 메트릭:
- `val/cleval_hmean` (주요 최적화 목표)
- `val/cleval_recall`
- `val/cleval_precision`
- `train/loss`
- `val/loss`
- 학습률, 배치 사이즈 등 하이퍼파라미터

### 베스트 결과 찾기:
```python
import wandb

# WandB에서 베스트 실행 찾기
api = wandb.Api()
runs = api.runs("your_entity/OCR-HRNet-Sweep")
best_run = max(runs, key=lambda run: run.summary.get("val/cleval_hmean", 0))

print(f"Best hmean: {best_run.summary['val/cleval_hmean']}")
print(f"Best config: {best_run.config}")
```

## 문제 해결

### GPU 메모리 부족
```bash
# 작은 이미지 사이즈와 배치 사이즈로 시작
# 스크립트가 자동으로 조정하지만 수동 설정도 가능
export CUDA_VISIBLE_DEVICES=0
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
`sweep_config.yaml`을 수정하여 사용자 정의 파라미터 범위 설정:

```yaml
parameters:
  models.encoder.model_name:
    values: [hrnet_w18, hrnet_w32]  # 특정 모델만 테스트

  image_size:
    values: [800, 1024]  # 특정 사이즈만 테스트
```

### Early Termination 설정
성능이 낮은 실험을 조기 종료:

```yaml
early_terminate:
  type: hyperband
  min_iter: 3
  eta: 2
```

## 참고 자료

- [experiments.md](./experiments.md): 기존 실험 결과 및 베스트 파라미터
- [WandB Sweep 문서](https://docs.wandb.ai/guides/sweeps)
- [env_template.txt](./env_template.txt): 환경변수 설정 예제
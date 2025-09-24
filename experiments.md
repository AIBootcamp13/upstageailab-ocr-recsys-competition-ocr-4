# Experiments Log

## Environment Snapshot
- Date: 2025-09-24
- GPU: NVIDIA GeForce RTX 3090 (24GB, CUDA 12.2, Driver 535.86.10)
- CPU: AMD Ryzen Threadripper 3960X (24C/48T)
- RAM: 256GB

## Experiments

### 2025-09-24 — baseline_v0 (DBNet ResNet18)
- 목적: 파이프라인 검증 및 초기 CLEval 기준점 수집
- 실행 명령: `uv run python code/baseline_code/runners/train.py preset=example dataset_base_path=/root/dev/upstageailab-ocr-recsys-competition-ocr-4/data/datasets/ exp_name=baseline_v0 trainer.max_epochs=1 +trainer.limit_train_batches=0.25 +trainer.limit_val_batches=0.25`
- 주요 설정: epoch 1, train/val 배치 25% 사용, test 전체 평가 수행
- Validation CLEval: recall 0.0201 / precision 0.2100 / hmean 0.0343
- Test CLEval (val split): recall 0.0916 / precision 0.8461 / hmean 0.1553
- 체크포인트: `outputs/baseline_v0/checkpoints/epoch=0-step=51.ckpt`
- 메모: 학습/검증 shuffling 확인, 추후 limit 제거 후 정규 학습 필요

### 2025-09-24 — baseline_full (DBNet ResNet18)
- 목적: 기본 설정(10 epoch) 성능 측정
- 실행 명령: `uv run python code/baseline_code/runners/train.py preset=example dataset_base_path=/root/dev/upstageailab-ocr-recsys-competition-ocr-4/data/datasets/ exp_name=baseline_full`
- 주요 설정: epoch 10, 전체 데이터 사용
- Validation CLEval: recall 0.7861 / precision 0.9648 / hmean 0.8608
- Test CLEval (val split): recall 0.7861 / precision 0.9648 / hmean 0.8608
- 체크포인트: `outputs/baseline_full/checkpoints/epoch=9-step=2050.ckpt`
- 메모: baseline 대비 빠른 수렴, 이후 개선 실험 기준값으로 사용

### 2025-09-24 — resnet50_full (DBNet ResNet50)
- 목적: 상위 백본 적용에 따른 성능 개선 검증
- 실행 명령: `uv run python code/baseline_code/runners/train.py preset=example dataset_base_path=/root/dev/upstageailab-ocr-recsys-competition-ocr-4/data/datasets/ exp_name=resnet50_full`
- 주요 설정: encoder ResNet50 (timm), decoder in_channels [256,512,1024,2048], epoch 10
- Validation CLEval: recall 0.8041 / precision 0.9557 / hmean 0.8638
- Test CLEval (val split): recall 0.8041 / precision 0.9557 / hmean 0.8638
- 체크포인트: `outputs/resnet50_full/checkpoints/epoch=7-step=1640.ckpt`
- 메모: baseline 대비 hmean +0.003, recall 상승, precision 약간 감소

### 2025-09-24 — resnet50_opt (AdamW + Cosine)
- 목적: 최적화기 변경에 따른 학습 안정성 및 성능 비교
- 실행 명령: `uv run python code/baseline_code/runners/train.py preset=example dataset_base_path=/root/dev/upstageailab-ocr-recsys-competition-ocr-4/data/datasets/ exp_name=resnet50_opt`
- 주요 설정: optimizer AdamW(lr=5e-4, weight_decay=0.01), CosineAnnealingLR(T_max=10)
- Validation CLEval: recall 0.7958 / precision 0.9464 / hmean 0.8578
- Test CLEval (val split): recall 0.7958 / precision 0.9464 / hmean 0.8578
- 체크포인트: `outputs/resnet50_opt/checkpoints/epoch=9-step=2050.ckpt`
- 메모: baseline 대비 hmean -0.006, AdamW 적용은 보류

### 2025-09-24 — resnet50_aug2 (Affine + Color aug)
- 목적: 기하/광학 증강을 강화하여 일반화 성능 향상
- 실행 명령: `uv run python code/baseline_code/runners/train.py preset=example dataset_base_path=/root/dev/upstageailab-ocr-recsys-competition-ocr-4/data/datasets/ exp_name=resnet50_aug2`
- 주요 설정: train transform에 Affine(±5% 이동/±10% 스케일/±3° 회전), RandomBrightnessContrast, HueSaturationValue, GaussianBlur 추가
- Validation CLEval: recall 0.8225 / precision 0.9531 / hmean 0.8776
- Test CLEval (val split): recall 0.8225 / precision 0.9531 / hmean 0.8776
- 체크포인트: `outputs/resnet50_aug2/checkpoints/epoch=8-step=1845.ckpt`
- 메모: 기존 대비 hmean +0.013, 특히 recall 향상 두드러짐

### 2025-09-24 — resnet50_aug (Rotate90 + ShiftScale)
- 목적: 강한 기하 증강 조합 실험
- 실행 명령: `uv run python code/baseline_code/runners/train.py preset=example dataset_base_path=/root/dev/upstageailab-ocr-recsys-competition-ocr-4/data/datasets/ exp_name=resnet50_aug`
- 주요 설정: RandomRotate90, ShiftScaleRotate(scale ±0.15, rotate ±5°), ColorJitter 포함
- Validation CLEval: recall 0.7926 / precision 0.9525 / hmean 0.8590
- Test CLEval (val split): recall 0.7926 / precision 0.9525 / hmean 0.8590
- 체크포인트: `outputs/resnet50_aug/checkpoints/epoch=8-step=1845.ckpt`
- 메모: 회전 90° 증강이 레이아웃 훼손 → 폐기

### 2025-09-24 — resnet50_aug2_long (15 epochs)
- 목적: 최적 증강 세팅에서 학습 에폭 확장 효과 확인
- 실행 명령: `uv run python code/baseline_code/runners/train.py preset=example dataset_base_path=/root/dev/upstageailab-ocr-recsys-competition-ocr-4/data/datasets/ exp_name=resnet50_aug2_long trainer.max_epochs=15`
- 주요 설정: resnet50_aug2와 동일, 단 max_epochs=15
- Validation CLEval: recall 0.7659 / precision 0.9695 / hmean 0.8485
- Test CLEval (val split): recall 0.7659 / precision 0.9695 / hmean 0.8485
- 체크포인트: `outputs/resnet50_aug2_long/checkpoints/epoch=13-step=2870.ckpt`
- 메모: 과적합으로 hmean 하락, 10 epoch 유지

### 2025-09-24 — convnext_tiny (실패)
- 목적: ConvNeXt-Tiny 백본 적용 가능성 탐색
- 실행 명령: `uv run python code/baseline_code/runners/train.py preset=example dataset_base_path=/root/dev/upstageailab-ocr-recsys-competition-ocr-4/data/datasets/ exp_name=convnext_tiny trainer.max_epochs=10 models.encoder.model_name=convnext_tiny models.encoder.select_features=[0,1,2,3] models.decoder.in_channels=[96,192,384,768]`
- Validation CLEval: recall 0.3159 / precision 0.9095 / hmean 0.4518
- Test CLEval (val split): recall 0.3159 / precision 0.9095 / hmean 0.4518
- 체크포인트: `outputs/convnext_tiny/checkpoints/epoch=8-step=1845.ckpt`
- 메모: ConvNeXt 백본은 수렴 불안정, 추가 튜닝 필요 → 채택 보류

### 2025-09-24 — baseline_polygon (ResNet18 + use_polygon)
- 목적: 베이스라인 설정으로 되돌리고 polygon 후처리를 활성화한 모델 재학습
- 실행 명령: `uv run python code/baseline_code/runners/train.py preset=example dataset_base_path=/root/dev/upstageailab-ocr-recsys-competition-ocr-4/data/datasets/ exp_name=baseline_polygon`
- 주요 설정: encoder ResNet18, decoder in_channels [64,128,256,512], train 증강 기본값, postprocess.use_polygon=True
- Validation CLEval: recall 0.9270 / precision 0.9741 / hmean 0.9480
- Test CLEval (val split): recall 0.9270 / precision 0.9741 / hmean 0.9480
- 체크포인트: `outputs/baseline_polygon/checkpoints/epoch=9-step=2050.ckpt`
- 제출파일: `outputs/baseline_polygon_pred/submissions/20250924_155253.csv`
- 메모: polygon 사용으로 recall/precision 모두 상승, 리더보드 제출 권장

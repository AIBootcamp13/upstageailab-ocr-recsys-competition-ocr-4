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

### 2025-09-24 — hrnet_full (HRNet-W18)
- 목적: HRNet 백본 적용 시 성능 검증
- 실행 명령: `uv run python code/baseline_code/runners/train.py preset=example dataset_base_path=/root/dev/upstageailab-ocr-recsys-competition-ocr-4/data/datasets/ exp_name=hrnet_full models.encoder.model_name=hrnet_w18 models.encoder.select_features=[1,2,3,4] models.decoder.in_channels=[128,256,512,1024]`
- Validation CLEval: recall 0.9487 / precision 0.9783 / hmean 0.9618
- Test CLEval (val split): recall 0.9487 / precision 0.9783 / hmean 0.9618
- 체크포인트: `outputs/hrnet_full/checkpoints/epoch=9-step=2050.ckpt`
- 메모: 베이스라인 대비 hmean +0.013 (polygon 모델 기준), 최고 성능 갱신

### 2025-09-24 — mixnet_full (MixNet-L)
- 목적: MixNet 백본 적용 성능 확인
- 실행 명령: `uv run python code/baseline_code/runners/train.py preset=example dataset_base_path=/root/dev/upstageailab-ocr-recsys-competition-ocr-4/data/datasets/ exp_name=mixnet_full models.encoder.model_name=mixnet_l models.encoder.select_features=[1,2,3,4] models.decoder.in_channels=[40,56,160,264]`
- Validation CLEval: recall 0.9075 / precision 0.9753 / hmean 0.9379
- Test CLEval (val split): recall 0.9075 / precision 0.9753 / hmean 0.9379
- 체크포인트: `outputs/mixnet_full/checkpoints/epoch=9-step=2050.ckpt`
- 메모: ResNet18 polygon 대비 하락, MixNet은 부적합

### 2025-09-24 — hrnet_full (thresh=0.25, box=0.45)
- 목적: HRNet 모델에 대한 후처리 임계값 튜닝
- 실행 명령: `uv run python code/baseline_code/runners/test.py preset=example dataset_base_path=/root/dev/upstageailab-ocr-recsys-competition-ocr-4/data/datasets/ exp_name=hrnet_eval_thresh "checkpoint_path='outputs/hrnet_full/checkpoints/epoch=9-step=2050.ckpt'" models.encoder.model_name=hrnet_w18 models.encoder.select_features=[1,2,3,4] models.decoder.in_channels=[128,256,512,1024] models.head.postprocess.thresh=0.25 models.head.postprocess.box_thresh=0.45`
- Validation CLEval: recall 0.9778 / precision 0.9765 / hmean 0.9767
- 제출파일: `outputs/hrnet_full_thresh_pred/submissions/20250924_191204.csv`
- 메모: 기본 설정 대비 hmean +0.0149, 현재 최종 제출 후보

### 2025-09-24 — hrnet_postprocess_grid (tune_postprocess.py)
- 목적: HRNet 체크포인트에 대해 `thresh`, `box_thresh` 격자 탐색으로 CLEval 개선 여부 확인
- 실행 명령: `TQDM_DISABLE=1 uv run python code/baseline_code/runners/tune_postprocess.py --checkpoint outputs/hrnet_full/checkpoints/epoch=9-step=2050.ckpt --exp-name=hrnet_tune_grid1 --overrides preset=example dataset_base_path=/root/dev/upstageailab-ocr-recsys-competition-ocr-4/data/datasets/ models.encoder.model_name=hrnet_w18 models.encoder.select_features=[1,2,3,4] models.decoder.in_channels=[128,256,512,1024] --thresh 0.23 0.25 0.27 --box-thresh 0.40 0.43 0.46 0.49`
- 주요 설정: 새 스크립트 `code/baseline_code/runners/tune_postprocess.py` 사용, 기본 unclip 비율 유지, 총 12조합 평가
- Best CLEval (val split): recall 0.9807 / precision 0.9752 / hmean 0.9776 (`thresh=0.23`, `box_thresh=0.43`)
- 메모: 기존 최고 대비 hmean +0.0009, recall 상승 폭이 precision 손실을 상회

### 2025-09-24 — hrnet_full (thresh=0.23, box=0.43)
- 목적: 격자 탐색에서 얻은 최적 후처리 설정으로 추론 성능 재평가
- 실행 명령: `TQDM_DISABLE=1 uv run python code/baseline_code/runners/test.py preset=example dataset_base_path=/root/dev/upstageailab-ocr-recsys-competition-ocr-4/data/datasets/ exp_name=hrnet_eval_thresh_v2 "checkpoint_path='outputs/hrnet_full/checkpoints/epoch=9-step=2050.ckpt'" models.encoder.model_name=hrnet_w18 models.encoder.select_features=[1,2,3,4] models.decoder.in_channels=[128,256,512,1024] models.head.postprocess.thresh=0.23 models.head.postprocess.box_thresh=0.43`
- Validation CLEval: recall 0.9807 / precision 0.9752 / hmean 0.9776
- 제출파일: (생성 안 함)
- 메모: 현재까지 최고 CLEval, 제출용 CSV 생성 시 `predictions/` 폴더 지정 필요

### 2025-09-25 — hrnet_postprocess_grid2 (세밀 탐색)
- 목적: `thresh` 0.22~0.24, `box_thresh` 0.41~0.44 범위 재탐색으로 추가 향상 도모
- 실행 명령: `TQDM_DISABLE=1 uv run python code/baseline_code/runners/tune_postprocess.py --checkpoint outputs/hrnet_full/checkpoints/epoch=9-step=2050.ckpt --exp-name=hrnet_tune_grid2 --overrides preset=example dataset_base_path=/root/dev/upstageailab-ocr-recsys-competition-ocr-4/data/datasets/ models.encoder.model_name=hrnet_w18 models.encoder.select_features=[1,2,3,4] models.decoder.in_channels=[128,256,512,1024] --thresh 0.22 0.23 0.24 --box-thresh 0.41 0.42 0.43 0.44`
- 주요 설정: box/polygon unclip 비율 기본값 유지, 12 조합 반복 평가
- Best CLEval (val split): recall 0.9816 / precision 0.9749 / hmean 0.9779 (`thresh=0.22`, `box_thresh=0.42`)
- 메모: hmean 소폭 상승, 낮은 thresh가 recall 극대화에 기여

### 2025-09-25 — hrnet_postprocess_unclip (unclip 비율 포함 탐색)
- 목적: 최상 조합(thresh=0.22, box=0.42) 기준으로 `box_unclip_ratio`, `polygon_unclip_ratio` 튜닝
- 실행 명령: `TQDM_DISABLE=1 uv run python code/baseline_code/runners/tune_postprocess.py --checkpoint outputs/hrnet_full/checkpoints/epoch=9-step=2050.ckpt --exp-name=hrnet_tune_unclip2 --overrides preset=example dataset_base_path=/root/dev/upstageailab-ocr-recsys-competition-ocr-4/data/datasets/ models.encoder.model_name=hrnet_w18 models.encoder.select_features=[1,2,3,4] models.decoder.in_channels=[128,256,512,1024] --thresh 0.22 --box-thresh 0.42 --box-unclip-ratio 1.3 1.4 1.5 1.6 --polygon-unclip-ratio 1.8 2.0`
- 주요 설정: 8 조합 평가, min_size 기본값 유지
- Best CLEval (val split): recall 0.9808 / precision 0.9775 / hmean 0.9788 (`thresh=0.22`, `box_thresh=0.42`, `box_unclip_ratio=1.3`, `polygon_unclip_ratio=1.8`)
- 메모: Precision 향상이 두드러져 hmean 추가 +0.0012 확보, 현재 최고 기록

### 2025-09-25 — hrnet_full (thresh=0.22, box=0.42, unclip=1.3/1.8)
- 목적: 최종 설정으로 검증/제출 산출물 생성
- 실행 명령: `TQDM_DISABLE=1 uv run python code/baseline_code/runners/test.py preset=example dataset_base_path=/root/dev/upstageailab-ocr-recsys-competition-ocr-4/data/datasets/ exp_name=hrnet_eval_thresh_v3 "checkpoint_path='outputs/hrnet_full/checkpoints/epoch=9-step=2050.ckpt'" models.encoder.model_name=hrnet_w18 models.encoder.select_features=[1,2,3,4] models.decoder.in_channels=[128,256,512,1024] models.head.postprocess.thresh=0.22 models.head.postprocess.box_thresh=0.42 models.head.postprocess.box_unclip_ratio=1.3 models.head.postprocess.polygon_unclip_ratio=1.8`
- Validation CLEval: recall 0.9808 / precision 0.9775 / hmean 0.9788
- 제출파일: `outputs/hrnet_pred_thresh_v3/submissions/20250925_024745.csv`
- 메모: 제출용 CSV는 predict 실행 후 `convert_submission.py`로 변환, 현시점 최고 성능

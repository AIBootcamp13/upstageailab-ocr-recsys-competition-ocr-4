# Experiments Log

## Environment Snapshot
- Date: 2025-09-24
- GPU: NVIDIA GeForce RTX 3090 (24GB, CUDA 12.2, Driver 535.86.10)
- CPU: AMD Ryzen Threadripper 3960X (24C/48T)
- RAM: 256GB

## Experiments

### 2025-09-24 — baseline_v0 (DBNet ResNet18)
- 목적: 파이프라인 검증 및 초기 CLEval 기준점 수집
- 실행 명령: `uv run python code/baseline_code/runners/train.py preset=example exp_name=baseline_v0 trainer.max_epochs=1 +trainer.limit_train_batches=0.25 +trainer.limit_val_batches=0.25`
- 주요 설정: epoch 1, train/val 배치 25% 사용, test 전체 평가 수행
- Validation CLEval: recall 0.0201 / precision 0.2100 / hmean 0.0343
- Test CLEval (val split): recall 0.0916 / precision 0.8461 / hmean 0.1553
- 체크포인트: `outputs/baseline_v0/checkpoints/epoch=0-step=51.ckpt`
- 메모: 학습/검증 shuffling 확인, 추후 limit 제거 후 정규 학습 필요

### 2025-09-24 — baseline_full (DBNet ResNet18)
- 목적: 기본 설정(10 epoch) 성능 측정
- 실행 명령: `uv run python code/baseline_code/runners/train.py preset=example exp_name=baseline_full`
- 주요 설정: epoch 10, 전체 데이터 사용
- Validation CLEval: recall 0.7861 / precision 0.9648 / hmean 0.8608
- Test CLEval (val split): recall 0.7861 / precision 0.9648 / hmean 0.8608
- 체크포인트: `outputs/baseline_full/checkpoints/epoch=9-step=2050.ckpt`
- 메모: baseline 대비 빠른 수렴, 이후 개선 실험 기준값으로 사용

### 2025-09-24 — resnet50_full (DBNet ResNet50)
- 목적: 상위 백본 적용에 따른 성능 개선 검증
- 실행 명령: `uv run python code/baseline_code/runners/train.py preset=example exp_name=resnet50_full`
- 주요 설정: encoder ResNet50 (timm), decoder in_channels [256,512,1024,2048], epoch 10
- Validation CLEval: recall 0.8041 / precision 0.9557 / hmean 0.8638
- Test CLEval (val split): recall 0.8041 / precision 0.9557 / hmean 0.8638
- 체크포인트: `outputs/resnet50_full/checkpoints/epoch=7-step=1640.ckpt`
- 메모: baseline 대비 hmean +0.003, recall 상승, precision 약간 감소

### 2025-09-24 — resnet50_opt (AdamW + Cosine)
- 목적: 최적화기 변경에 따른 학습 안정성 및 성능 비교
- 실행 명령: `uv run python code/baseline_code/runners/train.py preset=example exp_name=resnet50_opt`
- 주요 설정: optimizer AdamW(lr=5e-4, weight_decay=0.01), CosineAnnealingLR(T_max=10)
- Validation CLEval: recall 0.7958 / precision 0.9464 / hmean 0.8578
- Test CLEval (val split): recall 0.7958 / precision 0.9464 / hmean 0.8578
- 체크포인트: `outputs/resnet50_opt/checkpoints/epoch=9-step=2050.ckpt`
- 메모: baseline 대비 hmean -0.006, AdamW 적용은 보류

### 2025-09-24 — resnet50_aug2 (Affine + Color aug)
- 목적: 기하/광학 증강을 강화하여 일반화 성능 향상
- 실행 명령: `uv run python code/baseline_code/runners/train.py preset=example exp_name=resnet50_aug2`
- 주요 설정: train transform에 Affine(±5% 이동/±10% 스케일/±3° 회전), RandomBrightnessContrast, HueSaturationValue, GaussianBlur 추가
- Validation CLEval: recall 0.8225 / precision 0.9531 / hmean 0.8776
- Test CLEval (val split): recall 0.8225 / precision 0.9531 / hmean 0.8776
- 체크포인트: `outputs/resnet50_aug2/checkpoints/epoch=8-step=1845.ckpt`
- 메모: 기존 대비 hmean +0.013, 특히 recall 향상 두드러짐

### 2025-09-24 — resnet50_aug (Rotate90 + ShiftScale)
- 목적: 강한 기하 증강 조합 실험
- 실행 명령: `uv run python code/baseline_code/runners/train.py preset=example exp_name=resnet50_aug`
- 주요 설정: RandomRotate90, ShiftScaleRotate(scale ±0.15, rotate ±5°), ColorJitter 포함
- Validation CLEval: recall 0.7926 / precision 0.9525 / hmean 0.8590
- Test CLEval (val split): recall 0.7926 / precision 0.9525 / hmean 0.8590
- 체크포인트: `outputs/resnet50_aug/checkpoints/epoch=8-step=1845.ckpt`
- 메모: 회전 90° 증강이 레이아웃 훼손 → 폐기

### 2025-09-24 — resnet50_aug2_long (15 epochs)
- 목적: 최적 증강 세팅에서 학습 에폭 확장 효과 확인
- 실행 명령: `uv run python code/baseline_code/runners/train.py preset=example exp_name=resnet50_aug2_long trainer.max_epochs=15`
- 주요 설정: resnet50_aug2와 동일, 단 max_epochs=15
- Validation CLEval: recall 0.7659 / precision 0.9695 / hmean 0.8485
- Test CLEval (val split): recall 0.7659 / precision 0.9695 / hmean 0.8485
- 체크포인트: `outputs/resnet50_aug2_long/checkpoints/epoch=13-step=2870.ckpt`
- 메모: 과적합으로 hmean 하락, 10 epoch 유지

### 2025-09-24 — convnext_tiny (실패)
- 목적: ConvNeXt-Tiny 백본 적용 가능성 탐색
- 실행 명령: `uv run python code/baseline_code/runners/train.py preset=example exp_name=convnext_tiny trainer.max_epochs=10 models.encoder.model_name=convnext_tiny models.encoder.select_features=[0,1,2,3] models.decoder.in_channels=[96,192,384,768]`
- Validation CLEval: recall 0.3159 / precision 0.9095 / hmean 0.4518
- Test CLEval (val split): recall 0.3159 / precision 0.9095 / hmean 0.4518
- 체크포인트: `outputs/convnext_tiny/checkpoints/epoch=8-step=1845.ckpt`
- 메모: ConvNeXt 백본은 수렴 불안정, 추가 튜닝 필요 → 채택 보류

### 2025-09-24 — baseline_polygon (ResNet18 + use_polygon)
- 목적: 베이스라인 설정으로 되돌리고 polygon 후처리를 활성화한 모델 재학습
- 실행 명령: `uv run python code/baseline_code/runners/train.py preset=example exp_name=baseline_polygon`
- 주요 설정: encoder ResNet18, decoder in_channels [64,128,256,512], train 증강 기본값, postprocess.use_polygon=True
- Validation CLEval: recall 0.9270 / precision 0.9741 / hmean 0.9480
- Test CLEval (val split): recall 0.9270 / precision 0.9741 / hmean 0.9480
- 체크포인트: `outputs/baseline_polygon/checkpoints/epoch=9-step=2050.ckpt`
- 제출파일: `outputs/baseline_polygon_pred/submissions/20250924_155253.csv`
- 메모: polygon 사용으로 recall/precision 모두 상승, 리더보드 제출 권장

### 2025-09-24 — hrnet_full (HRNet-W18)
- 목적: HRNet 백본 적용 시 성능 검증
- 실행 명령: `uv run python code/baseline_code/runners/train.py preset=example exp_name=hrnet_full models.encoder.model_name=hrnet_w18 models.encoder.select_features=[1,2,3,4] models.decoder.in_channels=[128,256,512,1024]`
- Validation CLEval: recall 0.9487 / precision 0.9783 / hmean 0.9618
- Test CLEval (val split): recall 0.9487 / precision 0.9783 / hmean 0.9618
- 체크포인트: `outputs/hrnet_full/checkpoints/epoch=9-step=2050.ckpt`
- 메모: 베이스라인 대비 hmean +0.013 (polygon 모델 기준), 최고 성능 갱신

### 2025-09-24 — mixnet_full (MixNet-L)
- 목적: MixNet 백본 적용 성능 확인
- 실행 명령: `uv run python code/baseline_code/runners/train.py preset=example exp_name=mixnet_full models.encoder.model_name=mixnet_l models.encoder.select_features=[1,2,3,4] models.decoder.in_channels=[40,56,160,264]`
- Validation CLEval: recall 0.9075 / precision 0.9753 / hmean 0.9379
- Test CLEval (val split): recall 0.9075 / precision 0.9753 / hmean 0.9379
- 체크포인트: `outputs/mixnet_full/checkpoints/epoch=9-step=2050.ckpt`
- 메모: ResNet18 polygon 대비 하락, MixNet은 부적합

### 2025-09-24 — hrnet_full (thresh=0.25, box=0.45)
- 목적: HRNet 모델에 대한 후처리 임계값 튜닝
- 실행 명령: `uv run python code/baseline_code/runners/test.py preset=example exp_name=hrnet_eval_thresh "checkpoint_path='outputs/hrnet_full/checkpoints/epoch=9-step=2050.ckpt'" models.encoder.model_name=hrnet_w18 models.encoder.select_features=[1,2,3,4] models.decoder.in_channels=[128,256,512,1024] models.head.postprocess.thresh=0.25 models.head.postprocess.box_thresh=0.45`
- Validation CLEval: recall 0.9778 / precision 0.9765 / hmean 0.9767
- 제출파일: `outputs/hrnet_full_thresh_pred/submissions/20250924_191204.csv`
- 메모: 기본 설정 대비 hmean +0.0149, 현재 최종 제출 후보

### 2025-09-24 — hrnet_postprocess_grid (tune_postprocess.py)
- 목적: HRNet 체크포인트에 대해 `thresh`, `box_thresh` 격자 탐색으로 CLEval 개선 여부 확인
- 실행 명령: `TQDM_DISABLE=1 uv run python code/baseline_code/runners/tune_postprocess.py --checkpoint outputs/hrnet_full/checkpoints/epoch=9-step=2050.ckpt --exp-name=hrnet_tune_grid1 --overrides preset=example models.encoder.model_name=hrnet_w18 models.encoder.select_features=[1,2,3,4] models.decoder.in_channels=[128,256,512,1024] --thresh 0.23 0.25 0.27 --box-thresh 0.40 0.43 0.46 0.49`
- 주요 설정: 새 스크립트 `code/baseline_code/runners/tune_postprocess.py` 사용, 기본 unclip 비율 유지, 총 12조합 평가
- Best CLEval (val split): recall 0.9807 / precision 0.9752 / hmean 0.9776 (`thresh=0.23`, `box_thresh=0.43`)
- 메모: 기존 최고 대비 hmean +0.0009, recall 상승 폭이 precision 손실을 상회

### 2025-09-24 — hrnet_full (thresh=0.23, box=0.43)
- 목적: 격자 탐색에서 얻은 최적 후처리 설정으로 추론 성능 재평가
- 실행 명령: `TQDM_DISABLE=1 uv run python code/baseline_code/runners/test.py preset=example exp_name=hrnet_eval_thresh_v2 "checkpoint_path='outputs/hrnet_full/checkpoints/epoch=9-step=2050.ckpt'" models.encoder.model_name=hrnet_w18 models.encoder.select_features=[1,2,3,4] models.decoder.in_channels=[128,256,512,1024] models.head.postprocess.thresh=0.23 models.head.postprocess.box_thresh=0.43`
- Validation CLEval: recall 0.9807 / precision 0.9752 / hmean 0.9776
- 제출파일: (생성 안 함)
- 메모: 현재까지 최고 CLEval, 제출용 CSV 생성 시 `predictions/` 폴더 지정 필요

### 2025-09-25 — hrnet_postprocess_grid2 (세밀 탐색)
- 목적: `thresh` 0.22~0.24, `box_thresh` 0.41~0.44 범위 재탐색으로 추가 향상 도모
- 실행 명령: `TQDM_DISABLE=1 uv run python code/baseline_code/runners/tune_postprocess.py --checkpoint outputs/hrnet_full/checkpoints/epoch=9-step=2050.ckpt --exp-name=hrnet_tune_grid2 --overrides preset=example models.encoder.model_name=hrnet_w18 models.encoder.select_features=[1,2,3,4] models.decoder.in_channels=[128,256,512,1024] --thresh 0.22 0.23 0.24 --box-thresh 0.41 0.42 0.43 0.44`
- 주요 설정: box/polygon unclip 비율 기본값 유지, 12 조합 반복 평가
- Best CLEval (val split): recall 0.9816 / precision 0.9749 / hmean 0.9779 (`thresh=0.22`, `box_thresh=0.42`)
- 메모: hmean 소폭 상승, 낮은 thresh가 recall 극대화에 기여

### 2025-09-25 — hrnet_postprocess_unclip (unclip 비율 포함 탐색)
- 목적: 최상 조합(thresh=0.22, box=0.42) 기준으로 `box_unclip_ratio`, `polygon_unclip_ratio` 튜닝
- 실행 명령: `TQDM_DISABLE=1 uv run python code/baseline_code/runners/tune_postprocess.py --checkpoint outputs/hrnet_full/checkpoints/epoch=9-step=2050.ckpt --exp-name=hrnet_tune_unclip2 --overrides preset=example models.encoder.model_name=hrnet_w18 models.encoder.select_features=[1,2,3,4] models.decoder.in_channels=[128,256,512,1024] --thresh 0.22 --box-thresh 0.42 --box-unclip-ratio 1.3 1.4 1.5 1.6 --polygon-unclip-ratio 1.8 2.0`
- 주요 설정: 8 조합 평가, min_size 기본값 유지
- Best CLEval (val split): recall 0.9808 / precision 0.9775 / hmean 0.9788 (`thresh=0.22`, `box_thresh=0.42`, `box_unclip_ratio=1.3`, `polygon_unclip_ratio=1.8`)
- 메모: Precision 향상이 두드러져 hmean 추가 +0.0012 확보, 현재 최고 기록

### 2025-09-25 — hrnet_full (thresh=0.22, box=0.42, unclip=1.3/1.8)
- 목적: 최종 설정으로 검증/제출 산출물 생성
- 실행 명령: `TQDM_DISABLE=1 uv run python code/baseline_code/runners/test.py preset=example exp_name=hrnet_eval_thresh_v3 "checkpoint_path='outputs/hrnet_full/checkpoints/epoch=9-step=2050.ckpt'" models.encoder.model_name=hrnet_w18 models.encoder.select_features=[1,2,3,4] models.decoder.in_channels=[128,256,512,1024] models.head.postprocess.thresh=0.22 models.head.postprocess.box_thresh=0.42 models.head.postprocess.box_unclip_ratio=1.3 models.head.postprocess.polygon_unclip_ratio=1.8`
- Validation CLEval: recall 0.9808 / precision 0.9775 / hmean 0.9788
- 제출파일: `outputs/hrnet_pred_thresh_v3/submissions/20250925_024745.csv`
- 메모: 제출용 CSV는 predict 실행 후 `convert_submission.py`로 변환, 현시점 최고 성능

### 2025-09-26 — hrnet_dbpp_full (HRNet-W18 + DBNet++)
- 목적: FPNC(+ASF) 디코더가 포함된 DBNet++ 구조로 백본 HRNet-W18 학습
- 실행 명령: `uv run python code/baseline_code/runners/train.py preset=example exp_name=hrnet_dbpp_full models.encoder.model_name=hrnet_w18 models.encoder.select_features=[1,2,3,4] models.decoder.in_channels=[128,256,512,1024]`
- Validation CLEval: recall 0.9515 / precision 0.9787 / hmean 0.9648
- Test CLEval (val split): recall 0.9602 / precision 0.9788 / hmean 0.9684
- 체크포인트: `outputs/2025-09-25/22-35-11/checkpoints/epoch=6-step=1435.ckpt`
- 메모: DBNet++ 전환으로 초기 hmean은 기존 HRNet 대비 -0.0104p, 추후 파인튜닝 및 후처리 탐색 진행

### 2025-09-26 — hrnet_dbpp_postprocess_grid (확장 탐색)
- 목적: DBNet++ 체크포인트 기준 `thresh`, `box_thresh`, unclip 비율 동시 탐색으로 최적 CLEval 도출
- 실행 명령: `TQDM_DISABLE=1 uv run python code/baseline_code/runners/tune_postprocess.py --checkpoint outputs/2025-09-25/22-35-11/checkpoints/epoch=6-step=1435.ckpt --exp-name=hrnet_dbpp_tune_grid --overrides preset=example models.encoder.model_name=hrnet_w18 models.encoder.select_features=[1,2,3,4] models.decoder.in_channels=[128,256,512,1024] --thresh 0.19 0.20 0.21 0.22 0.23 --box-thresh 0.40 0.42 0.44 --box-unclip-ratio 1.2 1.3 1.4 1.5 1.6 --polygon-unclip-ratio 1.6 1.8 2.0`
- Best CLEval (val split): recall 0.9737 / precision 0.9773 / hmean 0.9750 (`thresh=0.20`, `box_thresh=0.40`, `box_unclip_ratio=1.2`, `polygon_unclip_ratio=1.8`)
- 메모: 낮은 threshold·unclip 조합이 recall/precision 균형 확보, 이후 파인튜닝에도 동일 설정 사용

### 2025-09-26 — hrnet_dbpp_full_ft (resume + 장기 학습)
- 목적: DBNet++ 모델을 checkpoint(epoch=6)에서 이어 학습해 수렴 안정화
- 실행 명령: `uv run python code/baseline_code/runners/train.py preset=example exp_name=hrnet_dbpp_full_ft trainer.max_epochs=15 "resume='outputs/2025-09-25/22-35-11/checkpoints/epoch=6-step=1435.ckpt'" models.encoder.model_name=hrnet_w18 models.encoder.select_features=[1,2,3,4] models.decoder.in_channels=[128,256,512,1024]`
- Validation CLEval: recall 0.9568 / precision 0.9793 / hmean 0.9678
- Test CLEval (val split): recall 0.9757 / precision 0.9779 / hmean 0.9763
- 체크포인트: `outputs/2025-09-26/07-26-00/checkpoints/epoch=12-step=2665.ckpt`
- 메모: 재학습 후 테스트 hmean +0.0079p 개선, 최적 후처리 적용 시 추가 상승 확인

### 2025-09-26 — hrnet_dbpp_full (thresh=0.20, box=0.40, unclip=1.2/1.8)
- 목적: 파인튜닝된 DBNet++ 모델에 최적 후처리 파라미터 적용해 최종 성능 산출
- 실행 명령: `TQDM_DISABLE=1 uv run python code/baseline_code/runners/test.py preset=example exp_name=hrnet_dbpp_ft_eval_best "checkpoint_path='outputs/2025-09-26/07-26-00/checkpoints/epoch=12-step=2665.ckpt'" models.encoder.model_name=hrnet_w18 models.encoder.select_features=[1,2,3,4] models.decoder.in_channels=[128,256,512,1024] models.head.postprocess.thresh=0.20 models.head.postprocess.box_thresh=0.40 models.head.postprocess.box_unclip_ratio=1.2 models.head.postprocess.polygon_unclip_ratio=1.8`
- Test CLEval (val split): recall 0.9772 / precision 0.9774 / hmean 0.9769
- 제출파일: `outputs/2025-09-26/10-47-31/submissions/20250926_104804.csv`
- 메모: HRNet + DBNet 원본 대비 hmean -0.0019p, precision 유지·recall 약간 감소, 추가 최적화(optimizer/스케줄) 여지 존재

# monitor='val/loss',
# mode='min',
monitor='val/hmean',
mode='max',

### 2025-09-29 — hrnet_w44_1024_reproduction (WandB Sweep 최적 설정 재현)
- 목적: WandB Sweep에서 도출된 최적 하이퍼파라미터 조합으로 HRNet-W44 모델 재학습
- 실행 명령: `uv run python runners/train.py preset=example models.encoder.model_name=hrnet_w44 models.encoder.select_features=[1,2,3,4] models.decoder.in_channels=[128,256,512,1024] transforms.train_transform.transforms.0.max_size=1024 transforms.train_transform.transforms.1.min_width=1024 transforms.train_transform.transforms.1.min_height=1024 transforms.val_transform.transforms.0.max_size=1024 transforms.val_transform.transforms.1.min_width=1024 transforms.val_transform.transforms.1.min_height=1024 transforms.test_transform.transforms.0.max_size=1024 transforms.test_transform.transforms.1.min_width=1024 transforms.test_transform.transforms.1.min_height=1024 dataloaders.train_dataloader.batch_size=2 dataloaders.val_dataloader.batch_size=2 dataloaders.test_dataloader.batch_size=2 models.head.postprocess.thresh=0.23105253214239585 models.head.postprocess.box_thresh=0.4324259445084524 models.head.postprocess.box_unclip_ratio=1.4745700672729625 models.head.postprocess.polygon_unclip_ratio=1.9770744341268096 models.loss.negative_ratio=2.824132345320219 models.loss.prob_map_loss_weight=3.591196851512631 models.loss.thresh_map_loss_weight=8.028627860143937 models.loss.binary_map_loss_weight=0.6919312670387725 models.head.k=45 models.optimizer._target_=torch.optim.AdamW models.optimizer.lr=0.0013358832166152786 models.optimizer.weight_decay=0.0003571900294890783 models.scheduler._target_=torch.optim.lr_scheduler.CosineAnnealingLR ~models.scheduler.step_size ~models.scheduler.gamma +models.scheduler.T_max=10 trainer.max_epochs=10 collate_fn.shrink_ratio=0.428584820771695 collate_fn.thresh_max=0.7506908133484191 collate_fn.thresh_min=0.33967147700431666 exp_name=hrnet_w44_1024_reproduction wandb=true`
- 주요 설정: HRNet-W44 백본, 1024px 해상도, AdamW+CosineAnnealingLR, 베이지안 최적화된 후처리 파라미터
- Test CLEval (internal): recall 0.9832 / precision 0.9821 / hmean 0.9823
- 리더보드 성과: H-Mean 0.9845 / Precision 0.9851 / Recall 0.9845
- 체크포인트: `outputs/2025-09-29/15-52-48/checkpoints/best_model.ckpt`
- 메모: **현재 최고 성능 달성**, 내부 평가와 리더보드 점수 모두 0.98+ 달성, WandB Sweep 기반 자동 하이퍼파라미터 튜닝의 효과 입증

### 2025-09-29 — hrnet_w44_1024_rbc (epoch 13)
- _target_: albumentations.RandomBrightnessContrast
brightness_limit: 0.2
contrast_limit: 0.2
p: 0.5
       test/hmean            0.984672486782074
     test/precision         0.9834803938865662
       test/recall          0.9864917397499084
Leaderboard:
H-Mean	Precision	Recall
0.9863	0.9868	0.9862

rbc_b32_c28_p40
Leaderboard:
0.9870 / 0.9869 / 0.9874

─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
       Test metric             DataLoader 0
─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
       test/hmean           0.9850713014602661
     test/precision         0.9830842018127441
       test/recall          0.9875101447105408
─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
Best checkpoint epoch: 17
Predicting DataLoader 0: 100%|##########| 413/413 [00:46<00:00,  8.95it/s]
Submission JSON saved to: /root/dev/upstageailab-ocr-recsys-competition-ocr-4-fork/outputs/2025-10-12/16-59-34/submissions/20251012_221528.json
Submission CSV saved to: /root/dev/upstageailab-ocr-recsys-competition-ocr-4-fork/outputs/2025-10-12/16-59-34/submissions/20251012_221528.csv (best epoch: 17)
Submission CSV uploaded as WandB artifact: submission_csv_f2eua7a5










best 증강 sweep
brightness_contrast_p:0.17099417286429375
brightness_limit:0.34146436314408957
color_jitter_brightness:0.1482166002922425
color_jitter_contrast:0.1380738907923448
color_jitter_hue:0.093762861716502
color_jitter_p:0.024452081273023007
color_jitter_saturation:0.03303697914933371
contrast_limit:0.19927640404389452
gamma_limit_lower:82
gamma_limit_upper:128
gaussian_blur_limit:5
gaussian_blur_p:0.1896680736951049
random_gamma_p:0.122251369866415
random_shadow_p:0.0558550781178635
shadow_dimension:6
shadow_num_lower:2
shadow_num_upper:4


-------------------
기본 모델 1에폭
Evaluation: 100%|##########| 404/404 [01:41<00:00,  4.00it/s]num=nmci, val/recall=0.781, val/precision=0.922, val/hmean=0.837]
Evaluation: 100%|##########| 404/404 [01:45<00:00,  3.82it/s]num=nmci, val/recall=0.900, val/precision=0.934, val/hmean=0.913]
Evaluation: 100%|##########| 404/404 [01:45<00:00,  3.83it/s]num=nmci, val/recall=0.927, val/precision=0.944, val/hmean=0.933]
Evaluation: 100%|##########| 404/404 [01:48<00:00,  3.73it/s]num=nmci, val/recall=0.920, val/precision=0.954, val/hmean=0.934]
Evaluation: 100%|##########| 404/404 [01:45<00:00,  3.82it/s]num=nmci, val/recall=0.960, val/precision=0.957, val/hmean=0.957]
Evaluation: 100%|##########| 404/404 [01:44<00:00,  3.85it/s]num=nmci, val/recall=0.961, val/precision=0.956, val/hmean=0.957]
Evaluation: 100%|##########| 404/404 [01:46<00:00,  3.81it/s]num=nmci, val/recall=0.963, val/precision=0.962, val/hmean=0.962]
Evaluation: 100%|##########| 404/404 [01:44<00:00,  3.87it/s]num=nmci, val/recall=0.954, val/precision=0.965, val/hmean=0.958]
Evaluation: 100%|##########| 404/404 [01:47<00:00,  3.76it/s]num=nmci, val/recall=0.956, val/precision=0.964, val/hmean=0.959]
Epoch 9: 100%|##########| 205/205 [03:53<00:00,  0.88it/s, v_num=nmci, val/recall=0.969, val/precision=0.969, val/hmean=0.968]
Using best checkpoint from epoch: 9                                     
Best checkpoint path: /root/dev/upstageailab-ocr-recsys-competition-ocr-4-fork/outputs/2025-10-01/14-06-52/checkpoints/epoch=9-step=2050.ckpt
       test/hmean           0.9680269360542297
     test/precision         0.9691384434700012
       test/recall          0.9685279130935669

RandomShadow
       test/hmean           0.9696398973464966
     test/precision         0.9690210223197937
       test/recall          0.9715850353240967

PlasmaShadow
       test/hmean           0.9623655676841736
     test/precision         0.9629773497581482
       test/recall          0.9636946320533752

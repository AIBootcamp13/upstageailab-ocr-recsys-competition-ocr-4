## EDA
1. GT에 의외의 영역에 박스가 있었다.
2. 손글씨 쓴것도 라벨링 되있다
3. 워터마크도 라벨링 되있다.

## 증강
remove 백그라운드를 한 png이미지로  실험했으나 큰 개선은 없었다.
베이스라인에 적용되어 있는 HorizontalFlip 증강 이외에
- 여러 증강을 wandb sweep 으로 최적화시도
- bright and Constrast 증강만 wandb sweep 으로 최적화 시도
-> 여러 온라인 증강 조합을 시도했으나 bright and Constrast정도가 유효했다.
- 이미지사이즈는 1024에서 가장 유효한 점수가 나왔다

## 모델 선택
- convnext,(tiny,base,base384, large,v2)  coat_lite, hrnet 시도해봤다. hrnet이 가장 성능이 좋았다.
- DBNET => DBNET++ 으로 전환했을대 성능 향상은 미미 했다.
- usePolygon을 했을때 성능이 대폭 향상되었다. (+0.1정도 향상)
- 후처리 파라미터 최적화로 성능 대폭 향상되었다.
### 최종 모델 선택 이유
- HRNet : 특징서술 및 @docs/model_compare.md 참고


----------

## 🎯 슬라이드 1. **EDA (Exploratory Data Analysis)**

### ✅ 주요 관찰 내용

1. **GT(Bounding Box) 이상치 존재**
   “바운딩 박스 분포, 텍스트 길이, 이미지 해상도 분포 등을 확인 권장. 누락/이상 라벨 검토 및 클래스 불균형 점검.”
   실제 점검 결과, 일부 이미지에서 텍스트가 없는 영역에도 GT 박스가 존재했고, OCR 모델이 불필요한 배경을 학습할 가능성이 있었음. [참고 이미지]

2. **손글씨 라벨링 포함**
   데이터셋에는 인쇄체뿐 아니라 손으로 쓴 금액, 서명, 메모 등의 손글씨도 포함되어 있었음 [참고 이미지]

3. **워터마크 라벨링 포함**
   일부 영수증 이미지 하단에는 브랜드명, 상호 로고 등의 워터마크가 라벨링 되어 있었음. [참고 이미지]

---

## 🧩 슬라이드 2. **증강 (Data Augmentation)**

### ⚙️ 실험 과정

* 기본 베이스라인에는 `HorizontalFlip`만 적용되어 있었음.
* `remove background`로 PNG 전처리를 수행했으나 “큰 개선은 없었음.”
* WandB Sweep을 활용해 다양한 증강 조합을 탐색한 결과,
  “brightness_limit: 0.3279, contrast_limit: 0.2837, p: 0.4049” 조합의 `RandomBrightnessContrast` 증강이 가장 유의미한 성능 향상을 보였음.
* 여러 온라인 증강 기법(ColorJitter, GaussianBlur, HueSaturationValue, RandomShadow, ShiftScaleRotate, VerticalFlip 등)을 시도했으나, 심플하게 HorizontalFlip + Brightness/Contrast 조합이 가장 성능이 좋았음.

### 📊 결과 요약

| 모델            | Img Size | 증강                         | H-Mean | Precision | Recall |
| ------------- | -------- | -------------------------- | ------ | --------- | ------ |
| ConvNeXt Base | 1024     | 없음                         | 0.9774 | 0.9763    | 0.9792 |
| ConvNeXt Base | 1024     | Bright&Contrast            | 0.9818 | 0.9815    | 0.9826 |
| ConvNeXt Base | 1024     | Bright&Contrast (Sweep 적용) | 0.9842 | 0.9860    | 0.9829 |
| HRNet W44     | 1024     | 없음                         | 0.9845 | 0.9851    | 0.9845 |
| HRNet W44     | 1024     | Bright&Contrast (Sweep)    | 0.9870 | 0.9869    | 0.9874 |

밝기·대비 증강 적용 전 대비 후 H-Mean이 약 +0.0044 상승하였으며, Precision과 Recall 모두 소폭 향상됨. 이미지 크기 1024에서 가장 안정적이고 높은 점수를 기록함.

---

## 🧠 슬라이드 3. **모델 선택 (Model Selection)**

### 🧱 실험 개요

* 실험한 모델: `ConvNeXt (Tiny/Base/Base384/Large/V2)`, `CoaT-Lite`, `HRNet (W18/W44)`

* 실제 실험에서 HRNet이 가장 높은 성능을 보였음

* DBNet에서 DBNet++로 전환 시 성능 향상은 미미했음 (+0.0041 향상).

* usePolygon을 적용했을 때 성능이 대폭 향상되었음 (+0.1 정도 향상).

### 🧩 최종 모델 선정

* **최종 모델:** HRNet-W44 + DBNet++
* **HRNet의 특징:** 고해상도 특징 맵을 유지하며 여러 해상도의 정보를 동시에 처리함. 병렬 브랜치 구조로 서로 다른 스케일의 피처를 교환해 작은 글자나 복잡한 텍스트 경계에서도 정밀한 검출이 가능함. 공간 정보 손실이 적어 문서나 영수증과 같은 세밀한 구조를 인식하기에 적합함. 또한 DBNet과 결합 시 텍스트 영역의 윤곽 복원과 경계 추정이 안정적이며 전체적인 검출 성능이 뛰어남.
* 후처리 단계에서 polygon unclip 비율 조정 및 threshold sweep을 통해 추가적인 성능 향상을 달성함.

### ⚡ 성능 비교표

| Model      | Detail             | Img  | Aug             | Notes                                                                                  | H-Mean     | Precision  | Recall     |
| ---------- | ------------------ | ---- | --------------- | -------------------------------------------------------------------------------------- | ---------- | ---------- | ---------- |
| resnet18   | 베이스라인 (dbnet)      | 640  |                 |                                                                                        | 0.8555     | 0.9689     | 0.7750     |
| resnet18   | use_polygon        | 640  |                 |                                                                                        | 0.9529     | 0.9784     | 0.9315     |
| convnext   | tiny               | 640  |                 |                                                                                        | 0.9631     | 0.9794     | 0.9495     |
| convnext   | base               | 640  |                 |                                                                                        | 0.9640     | 0.9843     | 0.9468     |
| coat_lite  | medium384          | 640  |                 |                                                                                        | 0.9706     | 0.9877     | 0.9555     |
| coat_lite  | medium384          | 640  | Bright&Contrast | candidate=1000, 후처리 조정                                                                 | 0.9790     | 0.9838     | 0.9752     |
| convnext   | base 384           | 640  |                 | candidate=1000, 후처리 조정                                                                 | 0.9752     | 0.9798     | 0.9714     |
| convnext   | base384            | 640  | Bright&Contrast | candidate=1000, 후처리 조정                                                                 | 0.9766     | 0.9793     | 0.9748     |
| convnext   | base384            | 640  | Bright&Contrast | loss beta=12, candidate=1000, 후처리 조정                                                   | 0.9811     | 0.9848     | 0.9779     |
| convnext   | large              | 1024 | Bright&Contrast | loss beta=12                                                                           | 0.9774     | 0.9763     | 0.9792     |
| convnext   | base384            | 1024 | Bright&Contrast | loss beta=12                                                                           | 0.9818     | 0.9815     | 0.9826     |
| convnext   | base.clip_384      | 1024 | Bright&Contrast | sweep 적용, loss beta=12                                                                 | 0.9821     | 0.9824     | 0.9828     |
| convnextV2 | base384            | 1024 | Bright&Contrast | sweep 적용, loss beta=12                                                                 | 0.9822     | 0.9793     | 0.9859     |
| convnext   | base384            | 1024 | Bright&Contrast | sweep 적용, loss beta=12                                                                 | 0.9842     | 0.9860     | 0.9829     |
| hrnet_w18  | (dbnet)            | 640  |                 |                                                                                        | 0.9643     | 0.9823     | 0.9492     |
| hrnet_w18  | (dbnet++)          | 640  |                 |                                                                                        | 0.9684     | 0.9788     | 0.9602     |
| hrnet_w18  | (dbnet++), 후처리 최적화 | 640  |                 | thresh=0.20, box_thresh=0.40, box_unclip_ratio=1.2, polygon_unclip_ratio=1.8           | 0.9792     | 0.9820     | 0.9774     |
| hrnet_w44  |                    | 1024 |                 | WandB Sweep 최적 하이퍼파라미터                                                                 | 0.9845     | 0.9851     | 0.9845     |
| hrnet_w44  | 증강 sweep           | 1024 | Bright&Contrast | brightness_limit=0.3279, contrast_limit=0.2837, p=0.4049                               | 0.9870     | 0.9869     | 0.9874     |
| hrnet_w44  | 후처리 sweep          | 1024 | Bright&Contrast | box_thresh=0.4008, box_unclip_ratio=1.8749, polygon_unclip_ratio=1.3102, thresh=0.1505 | **0.9886** | **0.9886** | **0.9888** |

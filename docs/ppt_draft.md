## 경진대회 목표 수립

### 주제
Receipt Text Detection | 영수증 글자 검출
영수증 사진에서 글자 위치를 추출하는 태스크를 수행합니다.

### 목표
- **목표 지표**: CLEval 기반 H-Mean 0.98+ 달성 및 상위권 리더보드 진입
- **재현성 확보**: Hydra 설정 관리와 `uv` 기반 환경 동기화로 실험 재현성 보장
- **추론 파이프라인**: 예측 JSON/CSV 자동 생성 및 제출 자동화 구축

### 개요
#### 소개 및 배경 설명
- **과제**: 영수증 이미지 내 텍스트 영역(단어/문장 단위) 탐지
- **접근**: DBNet 계열(Text Detection) 모델을 기반으로 백본/디코더/후처리 조합 최적화
- **핵심 기술**: PyTorch Lightning, Hydra, timm 백본, Albumentations 증강, CLEval 평가
#### 기간 
- 2025.09.22~ 2025.10.17

---

## 경진대회 수행 내용

### 개발 환경 구축
- **대표 라이브러리**: PyTorch, TorchVision, timm, Albumentations, OpenCV, PyTorch Lightning, Hydra, NumPy, Pandas, torchmetrics, Shapely, PyClipper, tqdm, Weights & Biases, CLEval 유틸
- **환경 표준화**: `uv sync`로 의존성 동기화, Python 3.10+ 기준
### 데이터 분석
- **데이터 규모**: train 3,272장 / val 404장 / test 413장
- **분포 점검**: 바운딩 박스 크기/비율, 텍스트 길이, 이미지 해상도 분포 확인
- **품질 점검**: 누락/이상 라벨 검토 및 클래스(텍스트 길이) 불균형 확인
- **검증 전략**: 제공된 `train/val/test` 분할과 CLEval 지표로 일관 평가
### Feature 엔진니어링
- **이미지 증강**: RandomBrightnessContrast, RandomShadow, PlasmaShadow, HueSaturationValue, GaussianBlur 등 조합 탐색
- **해상도 전략**: 640→1024 해상도 확장 시 성능 향상 검증(HRNet-W44, 1024 설정)
- **후처리 튜닝**: `thresh`, `box_thresh`, `box_unclip_ratio`, `polygon_unclip_ratio` 그리드/베이지안 탐색
- **학습 하이퍼파라미터**: AdamW + CosineAnnealingLR 및 손실 가중치 최적화(Sweep)
### 모델 선택 학습 및 평가
- **베이스라인**: DBNet(ResNet18) 파이프라인 검증 후 HRNet 계열로 확장
- **백본 비교**: ResNet50, MixNet, ConvNeXt 계열 실험→ HRNet-W18/44가 안정적 상위 성능
- **DBNet++ 전환**: FPNC(+ASF) 디코더 실험으로 재현성 및 성능 비교, 후처리로 보완
- **최종 베스트**: HRNet-W44(1024) 기반, 후처리/증강/손실 가중치 Sweep 적용
  - 리더보드 최고: **H-Mean 0.9886 / Precision 0.9886 / Recall 0.9888**

---

## 경진대회 회고

### Point 1. 우리 팀의 처음 목표에서 어디까지 도달했는가
- **목표 초과 달성**: H-Mean 0.9+ 목표 대비 리더보드 0.9886 달성
- **재현성 확보**: 설정/환경 표준화로 실험 재현성 및 제출 자동화 구축 완료

### Point 2. 우리 팀이 잘했던 점
- **체계적 탐색**: 후처리 임계값/Unclip 비율 그리드 탐색과 WandB Sweep으로 효율적 최적화
- **협업/기록**: 실험 로그와 체크포인트, 제출 파일 아카이빙으로 빠른 의사결정 지원
- **엔지니어링**: Hydra 설정화, `uv` 기반 환경 동기화, Lightning 루프로 실험 속도/안정성 확보

### Point 3. 협업하면서 아쉬웠던 점
- **아쉬움**: 초기 단계에서 백본/해상도/증강 변수 공간 정의가 넓어 탐색 비용이 컸음. 시간관계상 다양한 파라미터 조합에 대한 체계적인 비교 실험이 부족했던거 같음.
- **향후 계획**: 다양한 모델과 기법을 적용하여 실험

---

## 모델 선정 이유


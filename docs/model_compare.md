백본(ResNet, HRNet, ConvNext) 어느걸 추천하는가?
좋은 질문이에요! 결론부터 드리면 **정확도 최우선이면 HRNet**, **균형·호환성은 ResNet**, **대규모 데이터·최신 트렌드는 ConvNeXt**를 권장합니다. 상황별로 골라보세요.

# 한 줄 가이드

* **HRNet (V2-W32/W48)**: “작은 글자·복잡 배경·곡선 경계”까지 가장 안정적. DBNet/DBNet++와 궁합 최고.
* **ResNet (50/101)**: 구현 쉽고 레퍼런스 많아 튜닝·재현성 좋음. 대회 베이스라인/프로덕션 모두 무난.
* **ConvNeXt (T/S/B)**: 최신 구조로 표현력이 좋고 regularization에 강함. 데이터 충분할수록 성능이 잘 나옴.

# 언제 무엇을 쓸까

## 1) 정확도 최우선(대회/프로덕션 품질용)

* **HRNetV2-W48 → W32** 순으로 추천

  * 장점: 고해상도 특성 맵을 끝까지 유지 → 미세 경계/소형 텍스트 강함
  * 주의: 메모리·연산량 큼(W48는 특히 큼). 배치 사이즈·학습 속도 감수 필요

## 2) 밸런스·재현성(빠르게 성능 뽑기)

* **ResNet50** (여유 있으면 **ResNeXt50**)

  * 장점: 코드/체크포인트 풍부, 디버깅·전이학습 쉬움
  * 팁: FPN 채널을 256으로 맞추고 SyncBN/EMA, BCE+Dice 조합으로 안정화

## 3) 최신 백본·데이터가 넉넉할 때

* **ConvNeXt-T/S**

  * 장점: 스케일 업에 유리, augmentation(Strong aug, MixUp/CutMix)와 궁합 좋음
  * 팁: cosine LR, AdamW, layer scale·stochastic depth와 함께 쓰면 수렴 안정

## 4) 경량·실시간(엣지/CPU·모바일)

* **MobileNetV3-Large / EfficientNet-Lite**(참고)

  * DBNet이나 PAN과 조합 시 속도 효율 좋음
  * 팁: 해상도·unclip/threshold 스윕으로 FP 관리

# 데이터 특성별 추천

* **영수증/문서형(작은 글자, 빽빽)**: **HRNetV2-W32/W48** > ConvNeXt-T > ResNet50
* **자연장면/간판(배경 복잡, 곡선)**: **HRNetV2-W32/W48** ≈ ConvNeXt-S > ResNeXt50
* **라벨 수 적음/증강 의존 큼**: **ResNet50**가 과적합 덜하고 튜닝 용이
* **지연시간 제약**: **MobileNetV3-Large** 백본의 DBNet(+PAN 대안)

# 실전 세팅 예시

* **정확도 지향(권장):** DBNet++ + **HRNetV2-W32**

  * 입력: 짧은 변 1024 멀티스케일
  * 손실: Balanced BCE + Dice
  * 후처리: `prob_thr=0.3~0.4`, `box_thr=0.6~0.7`, `unclip=1.6~2.0` 스윕
  * 학습: Cosine + AdamW, SyncBN, strong aug(Perspective/Rotate/ColorJitter)
* **빠른 베이스라인:** DBNet + **ResNet50**

  * 동일 손실/후처리, 배치 키워서 재현성 확보
* **경량 배포:** DBNet + **MobileNetV3-Large**

  * 입력 축소(짧은 변 736/896), FP16, TensorRT/ONNX 최적화

# 선택 체크리스트(짧게)

* **최고 성능**이 필요하다 → **HRNet**
* **세팅 쉬움·자료 많음** → **ResNet**
* **큰 데이터·최신 성능 곡선** → **ConvNeXt**
* **속도/메모리 제약** → **MobileNetV3**

원하시면 현재 **데이터 예시(장면/문서 비율, 최소 글자 높이 px, 목표 FPS/지연시간, GPU 메모리)**를 알려주세요. 그 정보로 **백본 확정 + 하이퍼 파라미터 그리드**를 바로 잡아 드릴게요.

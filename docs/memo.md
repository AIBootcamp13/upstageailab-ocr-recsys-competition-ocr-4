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

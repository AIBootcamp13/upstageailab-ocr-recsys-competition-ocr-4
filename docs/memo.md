스토리텔링을 붙이면... 
1. GT에 의외의 영역에 박스가 있었다.
2. 손글씨 쓴것도 라벨링 되있다
3. 워터마크도 라벨링 되있다.


remove 백그라운드를 한 png이미지로  실험했으나 큰 개선은 없었다.

여러 온라인 증강중  bright and Constrast정도가 유효했다.
이미지사이즈는 1024에서 가장 유효한 점수가 나왔다
sweep을 통해 파라미터를 최적화했을때 큰개선이 있었다

remove 백그라운드를 한 png이미지로  실험했으나 큰 개선은 없었다.

여러 온라인 증강중  bright and Constrast정도가 유효했다.
이미지사이즈는 1024에서 가장 유효한 점수가 나왔다
sweep을 통해 파라미터를 최적화했을때 큰개선이 있었다

convnext,(tiny,base,base384, large,v2)  coat_lite, hrnet 시도해봤다. hrnet이 가장 성능이 좋았다.

DBNET => DBNET++로 변경후 실험
usePolygon을 했을때 성능이 향상되었다.


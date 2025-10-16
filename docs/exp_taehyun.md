| 모델         | 모델세부          | 증강              | img-size | sweep<br>적용 | loss<br>beta | candidate | thresh/<br>box-thresh | H-Mean | Precision | Recall |
| ---------- | ------------- | --------------- | -------- | ----------- | ------------ | --------- | --------------------- | ------ | --------- | ------ |
| resnet18   |               |                 | 640      |             |              |           |                       |  0.8555  | 0.9689 | 0.7750 |
| convnext   | tiny          |                 | 640      |             |              |           |                       | 0.9631 | 0.9794    | 0.9495 |
| convnext   | base          |                 | 640      |             |              |           |                       | 0.9640 | 0.9843    | 0.9468 |
| coat_lite  | medium384     |                 | 640      |             |              |           |                       | 0.9706 | 0.9877    | 0.9555 |
| coat_lite  | medium384     | Bright&Contrast | 640      |             |              | 1000      | 조정                    | 0.9790 | 0.9838    | 0.9752 |
| convnext   | base 384      |                 | 640      |             |              | 1000      | 조정                    | 0.9752 | 0.9798    | 0.9714 |
| convnext   | base384       | Bright&Contrast | 640      |             |              | 1000      | 조정                    | 0.9766 | 0.9793    | 0.9748 |
| convnext   | base384       | Bright&Contrast | 640      |             | 12           | 1000      | 조정                    | 0.9811 | 0.9848    | 0.9779 |
| convnext   | large         | Bright&Contrast | 1024     |             | 12           | 1000      | 조정                    | 0.9774 | 0.9763    | 0.9792 |
| convnext   | base384       | Bright&Contrast | 1024     |             | 12           | 1000      | 조정                    | 0.9818 | 0.9815    | 0.9826 |
| convnext   | base.clip_384 | Bright&Contrast | 1024     | 적용          | 12           | 1000      | 조정                    | 0.9821 | 0.9824    | 0.9828 |
| convnextV2 | base384       | Bright&Contrast | 1024     | 적용          | 12           | 1000      | 조정                    | 0.9822 | 0.9793    | 0.9859 |
| convnext   | base384       | Bright&Contrast | 1024     | 적용          | 12           | 1000      | 조정                    | 0.9842 | 0.9860    | 0.9829 |
| hrnet_w44   | base384       | Bright&Contrast | 1024     | 적용          | 12           | 1000      | 조정                    | 0.9864 | 0.9870    | 0.9862 |
| hrnet_w48   | base384       | Bright&Contrast | 1024     | 적용          | 12           | 1000      | 조정                    | 0.9860 | 0.9876    | 0.9848 |

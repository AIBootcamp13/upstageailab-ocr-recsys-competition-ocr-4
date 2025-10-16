| model      | image size | 비고 | 내부 점수 hmean | 내부 점수 precision | 내부 점수 recall | LeaderBoard hmean | LeaderBoard precision | LeaderBoard recall |
|-------------|-------------|------|------------------|----------------------|------------------|--------------------|------------------------|--------------------|
| resnet18    | 640         | 베이스라인 (dbnet) | 0.8608 | 0.9648 | 0.7861 | 0.8555 | 0.9689 | 0.7750 |
| resnet18    | 640         | use_polygon | 0.9480 | 0.9741 | 0.9270 | 0.9529 | 0.9784 | 0.9315 |
| hrnet_w18   | 640         | hrnet_w18 (dbnet) | 0.9618 | 0.9783 | 0.9487 | 0.9643 | 0.9823 | 0.9492 |
| hrnet_w18   | 640         | hrnet_w18 (dbnet++) | 0.9684 | 0.9788 | 0.9602 | - | - | - |
| hrnet_w18   | 640         | "hrnet_w18 (dbnet++), 후처리 최적화<br>thresh=0.20<br>box_thresh=0.40<br>box_unclip_ratio=1.2<br>polygon_unclip_ratio=1.8" | 0.9769 | 0.9774 | 0.9772 | 0.9792 | 0.9820 | 0.9774 |
| hrnet_w44   | 1024        | WandB Sweep에서 도출된 최적 하이퍼파라미터 조합으로 HRNet-W44 모델 재학습 | 0.9823 | 0.9821 | 0.9832 | 0.9845 | 0.9851 | 0.9845 |
| hrnet_w44   | 1024        | "증강 sweep<br>RandomBrightnessContrast<br> - brightness_limit: 0.32790290721690396<br> - contrast_limit: 0.2837328575729313<br> - p: 0.4049235865094287" | 0.9851 | 0.9831 | 0.9875 | 0.9870 | 0.9869 | 0.9874 |
| hrnet_w44   | 1024        | "후처리 sweep<br>box_thresh:0.4008455515878936<br>box_unclip_ratio:1.8748531609404064<br>polygon_unclip_ratio:1.3102306936503645<br>thresh:0.15048754740586018" | 0.9868 | 0.9861 | 0.9879 | 0.9886 | 0.9886 | 0.9888 |

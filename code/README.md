# 패스트캠퍼스 Receipt Text Detection Competition


## 환경 설정 및 의존성 설치
### Setup Environments
- 데이터셋 경로 설정
* configs/preset/datasets/db.yaml
```yaml
dataset_base_path: "/data/datasets/"   # Change your path
```
위와 같이 데이터셋 설정 파일에서 데이터셋 경로를 올바르게 지정하여야 합니다.

### Installation
- 의존성 패키지 설치
```bash
pip install -r requirements.txt
```

---
## 모델 및 학습 설정
### 파일 구조
```plaintext
└─── configs
    ├── preset
    │   ├── example.yaml
    │   ├── base.yaml
    │   ├── datasets
    │   │   └── db.yaml
    │   ├── lightning_modules
    │   │   └── base.yaml
    │   ├── metrics
    │   │   └── cleval.yaml
    │   └── models
    │       ├── decoder
    │       │   └── unet.yaml
    │       ├── encoder
    │       │   └── timm_backbone.yaml
    │       ├── head
    │       │   └── db_head.yaml
    │       ├── loss
    │       │   └── db_loss.yaml
    │       ├── postprocess
    │       │   └── base.yaml
    │       └── model_example.yaml
    ├── train.yaml
    ├── test.yaml
    └── predict.yaml
```
### 주요 설정 파일
- train.yaml, test.yaml, predict.yaml : Runner를 실행할 때 필요한 설정값
- preset/example.yaml : 각 모듈의 설정 파일을 지정
- preset/datasets/db.yaml : Dataset, Transform 등 데이터에 관련된 설정값
- preset/lightning_modules/base.yaml : PyTorch Lightning 실행에 관련된 설정값
- preset/metrics/cleval.yaml : CLEval 평가에 관련된 설정값
- preset/models/model_example.yaml : 각 모델 모듈의 설정 파일 및 Optimizer를 지정
- preset/models/* : 모델 구성에 필요한 각각의 모듈에 관련된 설정값

---
## Model Architecture
이 Baseline 코드는 DBNet을 기반으로 작성되었습니다.

### DBNet: Real-time Scene Text Detection with Differentiable Binarization
![DBNet](https://www.researchgate.net/publication/369783176/figure/fig1/AS:11431281137414188@1680649387586/Structure-of-DBNet-DBNet-is-a-novel-network-architecture-for-real-time-scene-text.png)

---
## 모델 평가지표
이번 대회에서는 Text Detection 결과를 평가하기 위해 **CLEval**을 사용합니다.

### CLEval: Character-Level Evaluation for Text Detection and Recognition Tasks
![CLEval](https://github.com/clovaai/CLEval/raw/master/resources/screenshots/explanation.gif)

---
## 실행 방법
### [Examples] preset/example.yaml을 지정하여 실행하는 경우
- Run Training:
```bash
python runners/train.py preset=example
```

- Run Test:
```bash
python runners/test.py preset=example "checkpoint_path='{checkpoint_path}'"
```

- Run Predict and Generate:
```bash
python runners/predict.py preset=example "checkpoint_path='{checkpoint_path}'"
```

---
## 제출파일 포맷 변환 방법
```bash
python ocr/utils/convert_submission.py --json_path {json_path} --output_path {output_path}
```

---
## 결과파일 저장 경로
### 파일 구조
```plaintext
└─── outputs
    └── {exp_name}
        ├── .hydra
        │   ├── overrides.yaml
        │   ├── config.yaml
        │   └── hydra.yaml
        ├── checkpoints
        │   └── epoch={epoch}-step={step}.ckpt
        ├── logs
        │   └── {exp_name}
        │       └── {exp_version}
        │           └── events.out.tfevents.{timestamp}.{hostname}.{pid}.v2
        └── submissions
            └── {timestamp}.json
```
### 주요 파일
- outputs/{exp_name}/submissions/{timestamp}.json : 제출파일
- outputs/{exp_name}/checkpoints/epoch={epoch}-step={step}.ckpt : 학습된 모델의 체크포인트 파일
- outputs/{exp_name}/.hydra/*.yaml : 실행 시 입력한 설정값

---
## 참고자료
- [DBNet](https://github.com/MhLiao/DB)
- [Hydra](https://hydra.cc/docs/intro/)
- [PyTorch Lightning](https://pytorch-lightning.readthedocs.io/en/latest/)
- [CLEval](https://github.com/clovaai/CLEval)

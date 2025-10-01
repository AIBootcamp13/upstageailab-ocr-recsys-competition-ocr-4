import os
import sys
import warnings
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# WandB 환경변수 설정 - 연속 실패 허용 횟수 증가 및 CUDA 메모리 최적화
os.environ['WANDB_AGENT_MAX_INITIAL_FAILURES'] = '30'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

warnings.filterwarnings(
    "ignore",
    message=r"^pkg_resources is deprecated as an API",
)

import lightning.pytorch as pl
import torch
import hydra
import wandb

# WandB API 키로 자동 로그인
wandb_api_key = os.environ.get('WANDB_API_KEY')
if wandb_api_key:
    try:
        wandb.login(key=wandb_api_key)
        print("WandB 로그인 성공")
    except Exception as e:
        print(f"WandB 로그인 실패: {e}")
from hydra.core.hydra_config import HydraConfig
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
)

PROJECT_ROOT = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(PROJECT_ROOT / "code" / "baseline_code"))

from ocr.lightning_modules import get_pl_modules_by_cfg
from ocr.utils.console_logging import setup_console_logging

CONFIG_DIR = os.environ.get('OP_CONFIG_DIR') or 'code/baseline_code/configs'

def convert_sweep_value(value):
    """WandB sweep config 값을 적절한 타입으로 변환"""
    if not isinstance(value, str):
        return value

    # bool 변환
    if value.lower() in ['true', 'false']:
        return value.lower() == 'true'

    # list 변환
    if value.startswith('[') and value.endswith(']'):
        return eval(value)

    # numeric 변환
    if value.replace('.', '').replace('-', '').replace('e', '').replace('+', '').isdigit():
        if '.' in value or 'e' in value.lower():
            return float(value)
        else:
            return int(value)

    return value

def load_sweep_config():
    """sweep_config_brightness.yaml 파일에서 설정 로드"""
    import yaml

    config_path = PROJECT_ROOT / "sweep_config_brightness.yaml"

    if not config_path.exists():
        raise FileNotFoundError(
            f"Sweep config file not found: {config_path}\n"
            f"Please ensure 'sweep_config_brightness.yaml' exists in the project root."
        )

    with open(config_path, 'r', encoding='utf-8') as f:
        sweep_config = yaml.safe_load(f)

    print(f"Loaded sweep config from: {config_path}")
    return sweep_config

def train_with_sweep():
    """WandB sweep agent에서 호출되는 학습 함수 - 베스트 파라미터 고정"""

    # GPU 메모리 정리
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"GPU memory cleared. Available: {torch.cuda.get_device_properties(0).total_memory // 1024**3}GB")

    print("Starting sweep agent for RandomBrightnessContrast optimization")

    # 현재 시간으로 run name 생성
    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    run_name = f"brightness_sweep_{timestamp}"

    # WandB 초기화
    wandb.init(name=run_name)
    sweep_config = wandb.config

    # Hydra 설정 로드
    from hydra import initialize, compose
    from omegaconf import OmegaConf
    with initialize(config_path=CONFIG_DIR, version_base='1.2'):
        config = compose(config_name='train')
        # struct 모드 비활성화 (새로운 키 추가 허용)
        OmegaConf.set_struct(config, False)

    # ==========================================
    # 베스트 파라미터 고정 (모두 하드코딩)
    # ==========================================
    overrides = [
        "preset=example",
        f"dataset_base_path={os.environ.get('DATASET_BASE_PATH', '/root/dev/upstageailab-ocr-recsys-competition-ocr-4/data/datasets/')}",
        "wandb=True",

        # 모델 설정
        "models.encoder.model_name=hrnet_w44",
        "models.encoder.select_features=[1,2,3,4]",
        "models.decoder.in_channels=[128,256,512,1024]",

        # 이미지 사이즈 설정 (1024)
        "transforms.train_transform.transforms.0.max_size=1024",
        "transforms.train_transform.transforms.1.min_width=1024",
        "transforms.train_transform.transforms.1.min_height=1024",
        "transforms.val_transform.transforms.0.max_size=1024",
        "transforms.val_transform.transforms.1.min_width=1024",
        "transforms.val_transform.transforms.1.min_height=1024",
        "transforms.test_transform.transforms.0.max_size=1024",
        "transforms.test_transform.transforms.1.min_width=1024",
        "transforms.test_transform.transforms.1.min_height=1024",

        # 배치 사이즈 (2)
        "dataloaders.train_dataloader.batch_size=2",
        "dataloaders.val_dataloader.batch_size=2",
        "dataloaders.test_dataloader.batch_size=2",

        # 후처리 파라미터
        "models.head.postprocess.thresh=0.23105253214239585",
        "models.head.postprocess.box_thresh=0.4324259445084524",
        "models.head.postprocess.box_unclip_ratio=1.4745700672729625",
        "models.head.postprocess.polygon_unclip_ratio=1.9770744341268096",

        # Loss weights
        "models.loss.negative_ratio=2.824132345320219",
        "models.loss.prob_map_loss_weight=3.591196851512631",
        "models.loss.thresh_map_loss_weight=8.028627860143937",
        "models.loss.binary_map_loss_weight=0.6919312670387725",

        # DBHead k
        "models.head.k=45",

        # Optimizer (AdamW)
        "models.optimizer._target_=torch.optim.AdamW",
        "models.optimizer.lr=0.0013358832166152786",
        "models.optimizer.weight_decay=0.0003571900294890783",

        # Scheduler (CosineAnnealingLR)
        "models.scheduler._target_=torch.optim.lr_scheduler.CosineAnnealingLR",
        "models.scheduler.T_max=10",

        # Epochs
        "trainer.max_epochs=13",

        # CollateFN 파라미터
        "collate_fn.shrink_ratio=0.428584820771695",
        "collate_fn.thresh_max=0.7506908133484191",
        "collate_fn.thresh_min=0.33967147700431666",
    ]

    # ==========================================
    # RandomBrightnessContrast 파라미터만 sweep
    # ==========================================
    brightness_limit = sweep_config.get('brightness_limit', 0.2)
    contrast_limit = sweep_config.get('contrast_limit', 0.2)
    brightness_contrast_p = sweep_config.get('brightness_contrast_p', 0.5)

    # transforms.train_transform.transforms[3]가 RandomBrightnessContrast
    overrides.extend([
        f"transforms.train_transform.transforms.3.brightness_limit={brightness_limit}",
        f"transforms.train_transform.transforms.3.contrast_limit={contrast_limit}",
        f"transforms.train_transform.transforms.3.p={brightness_contrast_p}",
    ])

    print(f"RandomBrightnessContrast params: brightness_limit={brightness_limit}, contrast_limit={contrast_limit}, p={brightness_contrast_p}")

    # exp_name 설정
    sweep_exp_name = f"brightness_sweep_{wandb.run.name}"
    overrides.append(f"exp_name={sweep_exp_name}")

    # Hydra config 업데이트
    from omegaconf import OmegaConf

    # 스케줄러 파라미터 정리 - CosineAnnealingLR 고정이므로 StepLR 파라미터 제거
    if hasattr(config.models.scheduler, 'step_size'):
        OmegaConf.set_struct(config.models.scheduler, False)
        delattr(config.models.scheduler, 'step_size')
        print("Removed step_size from scheduler config")
    if hasattr(config.models.scheduler, 'gamma'):
        OmegaConf.set_struct(config.models.scheduler, False)
        delattr(config.models.scheduler, 'gamma')
        print("Removed gamma from scheduler config")

    # T_max가 없으면 추가 (CosineAnnealingLR 필수 파라미터)
    if not hasattr(config.models.scheduler, 'T_max'):
        OmegaConf.set_struct(config.models.scheduler, False)
        config.models.scheduler.T_max = 10
        print("Added T_max=10 to scheduler config")

    for override in overrides:
        # ~ 또는 + 로 시작하는 특수 overrides는 건너뜀 (Hydra가 직접 처리)
        if override.startswith('~') or override.startswith('+'):
            continue

        if '=' not in override:
            continue

        key, value = override.split('=', 1)

        # 타입 변환
        value = convert_sweep_value(value)

        # OmegaConf를 사용한 안전한 설정
        try:
            keys = key.split('.')
            current = config
            for k in keys[:-1]:
                # 리스트 인덱스 처리 (예: transforms[0])
                if k.isdigit():
                    current = current[int(k)]
                else:
                    if k not in current:
                        current[k] = OmegaConf.create({})
                    current = current[k]

            # 마지막 키 처리
            last_key = keys[-1]
            if last_key.isdigit():
                current[int(last_key)] = value
            else:
                current[last_key] = value
        except Exception as e:
            print(f"Warning: Failed to set {key}={value}: {e}")
            continue

    # 학습 실행
    import tempfile
    run_dir = Path(tempfile.mkdtemp(prefix=f"brightness_sweep_{wandb.run.name}_"))

    # log_dir 직접 설정
    log_dir = run_dir / "logs"
    config.log_dir = str(log_dir)

    log_path = setup_console_logging(log_dir, "train.log")

    start_time = datetime.now()
    print(f"[{start_time:%Y-%m-%d %H:%M:%S}] Training run started. Log file: {log_path}", flush=True)

    try:
        pl.seed_everything(config.get("seed", 42), workers=True)

        model_module, data_module = get_pl_modules_by_cfg(config)

        # Sweep agent의 run을 재사용하므로 finish() 호출하지 않음
        # wandb.finish()를 호출하면 sweep agent의 run이 종료되고,
        # WandbLogger가 새 run을 만들어서 sweep 연결이 끊어짐

        # WandB Logger 사용
        from lightning.pytorch.loggers import WandbLogger
        logger = WandbLogger(
            project=os.environ.get('WANDB_PROJECT', 'OCR-Brightness-Sweep'),
            name=f"{sweep_exp_name}",
            config=dict(sweep_config)
        )

        # checkpoint_dir 직접 설정 (HydraConfig 보간 문제 해결)
        checkpoint_dir = run_dir / "checkpoints"
        config.checkpoint_dir = str(checkpoint_dir)

        # submission_dir 직접 설정 (HydraConfig 보간 문제 해결)
        submission_dir = run_dir / "submissions"
        config.submission_dir = str(submission_dir)

        checkpoint_callback = ModelCheckpoint(
            dirpath=str(checkpoint_dir),
            save_top_k=3,
            monitor='val/hmean',
            mode='max',
        )

        callbacks = [
            LearningRateMonitor(logging_interval='step'),
            checkpoint_callback,
        ]

        trainer = pl.Trainer(
            **config.trainer,
            logger=logger,
            callbacks=callbacks
        )

        trainer.fit(
            model_module,
            data_module,
            ckpt_path=config.get("resume", None),
        )

        # 테스트 실행
        trainer.test(
            model_module,
            data_module,
        )

        # 베스트 체크포인트 로드 및 CSV 제출파일 생성
        best_checkpoint = checkpoint_callback.best_model_path
        if best_checkpoint:
            checkpoint = torch.load(best_checkpoint, map_location='cpu')
            best_epoch = checkpoint.get("epoch")
            state_dict = checkpoint.get("state_dict", checkpoint)
            model_module.load_state_dict(state_dict)

            # CSV 생성을 위한 predict 실행
            trainer.predict(model_module, data_module)

            if getattr(model_module, "last_submission_paths", None):
                paths = model_module.last_submission_paths
                print(f"Submission JSON saved to: {paths['json']}")
                if best_epoch is not None:
                    print(f"Submission CSV saved to: {paths['csv']} (best epoch: {best_epoch})")
                else:
                    print(f"Submission CSV saved to: {paths['csv']} (best epoch metadata unavailable)")

                # CSV 파일을 WandB artifact로 업로드
                try:
                    csv_path = paths['csv']
                    if os.path.exists(csv_path):
                        artifact = wandb.Artifact(
                            name=f"submission_csv_{wandb.run.id}",
                            type="submission",
                            description=f"CSV submission file from epoch {best_epoch if best_epoch is not None else 'unknown'}"
                        )
                        artifact.add_file(csv_path, name="submission.csv")
                        wandb.log_artifact(artifact)
                        print(f"Submission CSV uploaded as WandB artifact: submission_csv_{wandb.run.id}")
                    else:
                        print(f"CSV file not found at: {csv_path}")
                except Exception as e:
                    print(f"Failed to upload CSV as artifact: {e}")
        else:
            print("Model checkpoint was not created; skipping submission generation.")

        # 최종 메트릭 로깅
        if hasattr(model_module, 'test_metrics'):
            test_metrics = model_module.test_metrics
            for key, value in test_metrics.items():
                wandb.log({f"final_test/{key}": value})

    except Exception as e:
        print(f"Training failed: {e}")
        wandb.log({"training_failed": True})
        raise
    finally:
        end_time = datetime.now()
        elapsed = end_time - start_time
        print(f"[{end_time:%Y-%m-%d %H:%M:%S}] Training run finished. Duration: {elapsed}", flush=True)
        # wandb.finish() 제거: sweep agent가 자동으로 run을 관리하므로 명시적 finish() 호출 불필요
        # 명시적으로 finish()를 호출하면 sweep이 조기 종료될 수 있음

def run_sweep():
    """WandB sweep 실행 - sweep_config_brightness.yaml 파일 사용"""
    # train.yaml에서 프로젝트 설정 로드
    from hydra import initialize, compose
    with initialize(config_path=CONFIG_DIR, version_base='1.2'):
        config = compose(config_name='train')

    # sweep_config_brightness.yaml 파일에서 설정 로드
    sweep_config = load_sweep_config()

    # WandB 프로젝트 설정 (YAML 파일에서 가져오되, 환경변수로 오버라이드 가능)
    project_name = os.environ.get('WANDB_PROJECT', sweep_config.get('project', 'OCR-Brightness-Sweep'))
    entity = os.environ.get('WANDB_ENTITY', sweep_config.get('entity', None))

    # Sweep 생성
    sweep_id = wandb.sweep(sweep_config, project=project_name, entity=entity)

    print(f"Created sweep: {sweep_id}")
    print(f"Project: {project_name}")
    print(f"Sweep URL: https://wandb.ai/{entity if entity else 'your_username'}/{project_name}/sweeps/{sweep_id}")
    print(f"Run: wandb agent {sweep_id}")

    return sweep_id

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='WandB Sweep for RandomBrightnessContrast optimization')
    parser.add_argument('--create-sweep', action='store_true', help='Create new sweep')
    parser.add_argument('--sweep-id', type=str, help='Existing sweep ID to join')
    parser.add_argument('--count', type=int, default=20, help='Number of runs')

    args = parser.parse_args()

    if args.create_sweep:
        sweep_id = run_sweep()
        print(f"\nTo start the sweep agent, run:")
        print(f"wandb agent {sweep_id}")
    elif args.sweep_id:
        # train.yaml에서 프로젝트 설정 로드
        from hydra import initialize, compose
        with initialize(config_path=CONFIG_DIR, version_base='1.2'):
            config = compose(config_name='train')

        # Entity와 project 정보 추출
        project_name = os.environ.get('WANDB_PROJECT', 'OCR-Brightness-Sweep')
        entity = os.environ.get('WANDB_ENTITY', None)

        print(f"Starting sweep agent for project: {project_name}")
        wandb.agent(args.sweep_id, train_with_sweep, count=args.count,
                   project=project_name, entity=entity)
    else:
        # 기본 동작: sweep config 출력
        import yaml
        try:
            sweep_config = load_sweep_config()
            print("Sweep configuration from sweep_config_brightness.yaml:")
            print(yaml.dump(sweep_config, default_flow_style=False))
            print("\nTo create a sweep:")
            print("uv run python wandb_sweep_brightness.py --create-sweep")
        except FileNotFoundError as e:
            print(f"Error: {e}")
            sys.exit(1)
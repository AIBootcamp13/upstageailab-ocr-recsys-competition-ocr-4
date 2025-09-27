import os
import sys
import warnings
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

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

# HRNet 모델별 in_channels 매핑 (현재 작동하는 설정 기준)
HRNET_CHANNELS = {
    'hrnet_w18': [128, 256, 512, 1024],   # 모든 HRNet 모델이 동일한 채널 구조 사용
    'hrnet_w32': [128, 256, 512, 1024],   # timm features_only에서 확인된 구조
    'hrnet_w40': [128, 256, 512, 1024],   # select_features=[1,2,3,4] 사용시
    'hrnet_w44': [128, 256, 512, 1024],   # 모든 HRNet 계열이 동일
    'hrnet_w48': [128, 256, 512, 1024],   # 동일한 출력 채널 구조
}

def get_sweep_config():
    """WandB sweep configuration"""
    return {
        'method': 'bayes',
        'metric': {
            'name': 'val/cleval_hmean',
            'goal': 'maximize'
        },
        'parameters': {
            # HRNet 모델 선택
            'models.encoder.model_name': {
                'values': ['hrnet_w18', 'hrnet_w32', 'hrnet_w40', 'hrnet_w44', 'hrnet_w48']
            },

            # 이미지 사이즈 파라미터
            'image_size': {
                'values': [640, 800, 1024, 1280]
            },

            # 배치 사이즈 (이미지 사이즈와 연동)
            'batch_size': {
                'values': [4, 8, 16, 32]
            },

            # 후처리 파라미터 (experiments.md 베스트 범위)
            'models.head.postprocess.thresh': {
                'min': 0.19,
                'max': 0.25
            },
            'models.head.postprocess.box_thresh': {
                'min': 0.40,
                'max': 0.49
            },
            'models.head.postprocess.box_unclip_ratio': {
                'min': 1.2,
                'max': 1.6
            },
            'models.head.postprocess.polygon_unclip_ratio': {
                'min': 1.6,
                'max': 2.0
            },

            # DBLoss 파라미터
            'models.loss.negative_ratio': {
                'min': 2.0,
                'max': 4.0
            },
            'models.loss.prob_map_loss_weight': {
                'min': 3.0,
                'max': 7.0
            },
            'models.loss.thresh_map_loss_weight': {
                'min': 8.0,
                'max': 12.0
            },
            'models.loss.binary_map_loss_weight': {
                'min': 0.5,
                'max': 2.0
            },

            # DBHead 파라미터
            'models.head.k': {
                'values': [40, 45, 50, 55, 60]
            },

            # 학습 파라미터
            'models.optimizer._target_': {
                'values': ['torch.optim.Adam', 'torch.optim.AdamW']
            },
            'models.optimizer.lr': {
                'min': 0.0005,
                'max': 0.002
            },
            'models.optimizer.weight_decay': {
                'min': 0.00005,
                'max': 0.0005
            },

            # 스케줄러
            'models.scheduler._target_': {
                'values': ['torch.optim.lr_scheduler.StepLR', 'torch.optim.lr_scheduler.CosineAnnealingLR']
            },
            'models.scheduler.step_size': {
                'values': [50, 75, 100, 125]  # StepLR용
            },
            'models.scheduler.gamma': {
                'values': [0.1, 0.2, 0.5]  # StepLR용
            },
            'models.scheduler.T_max': {
                'values': [8, 10, 12]  # CosineAnnealingLR용
            },

            # 에포크
            'trainer.max_epochs': {
                'values': [8, 10, 12]
            },

            # CollateFN 파라미터
            'collate_fn.shrink_ratio': {
                'min': 0.3,
                'max': 0.5
            },
            'collate_fn.thresh_min': {
                'min': 0.2,
                'max': 0.4
            },
            'collate_fn.thresh_max': {
                'min': 0.6,
                'max': 0.8
            },
        }
    }

def adjust_batch_size_for_memory(image_size, suggested_batch_size):
    """GPU 메모리에 따른 배치 사이즈 자동 조정"""
    # 더 보수적인 메모리 기반 조정 (HRNet 모델은 메모리 사용량이 많음)
    memory_mapping = {
        1280: 2,  # 더 작게 조정
        1024: 4,  # 더 작게 조정
        800: 8,   # 더 작게 조정
        640: 16,  # 더 작게 조정
    }

    max_batch_size = memory_mapping.get(image_size, 8)
    return min(suggested_batch_size, max_batch_size)

def train_with_sweep():
    """WandB sweep agent에서 호출되는 학습 함수"""

    # WandB 초기화
    wandb.init()
    sweep_config = wandb.config

    # Hydra 설정 로드
    from hydra import initialize, compose
    from omegaconf import OmegaConf
    with initialize(config_path=CONFIG_DIR, version_base='1.2'):
        config = compose(config_name='train')
        # struct 모드 비활성화 (새로운 키 추가 허용)
        OmegaConf.set_struct(config, False)

    # Hydra config 오버라이드
    overrides = []

    # 모델 선택 및 in_channels 자동 설정
    model_name = sweep_config.get('models.encoder.model_name', 'hrnet_w18')
    if model_name in HRNET_CHANNELS:
        in_channels = HRNET_CHANNELS[model_name]
        overrides.extend([
            f"models.encoder.model_name={model_name}",
            f"models.encoder.select_features=[1,2,3,4]",
            f"models.decoder.in_channels={in_channels}"
        ])

    # 이미지 사이즈 설정 - OmegaConf 리스트 접근 방식 사용
    image_size = sweep_config.get('image_size', 640)

    # transforms 직접 수정 (overrides 대신)
    try:
        # train_transform
        if 'transforms' in config and 'train_transform' in config.transforms:
            transforms_list = config.transforms.train_transform.transforms
            if len(transforms_list) > 0:
                transforms_list[0].max_size = image_size
            if len(transforms_list) > 1:
                transforms_list[1].min_width = image_size
                transforms_list[1].min_height = image_size

        # val_transform
        if 'transforms' in config and 'val_transform' in config.transforms:
            transforms_list = config.transforms.val_transform.transforms
            if len(transforms_list) > 0:
                transforms_list[0].max_size = image_size
            if len(transforms_list) > 1:
                transforms_list[1].min_width = image_size
                transforms_list[1].min_height = image_size

        # test_transform
        if 'transforms' in config and 'test_transform' in config.transforms:
            transforms_list = config.transforms.test_transform.transforms
            if len(transforms_list) > 0:
                transforms_list[0].max_size = image_size
            if len(transforms_list) > 1:
                transforms_list[1].min_width = image_size
                transforms_list[1].min_height = image_size

        print(f"Image size set to: {image_size}")
    except Exception as e:
        print(f"Warning: Failed to set image size {image_size}: {e}")

    # 배치 사이즈 조정
    suggested_batch_size = sweep_config.get('batch_size', 16)
    batch_size = adjust_batch_size_for_memory(image_size, suggested_batch_size)
    print(f"Batch size adjustment: image_size={image_size}, suggested={suggested_batch_size}, final={batch_size}")
    overrides.extend([
        f"dataloaders.train_dataloader.batch_size={batch_size}",
        f"dataloaders.val_dataloader.batch_size={batch_size}",
        f"dataloaders.test_dataloader.batch_size={batch_size}",
    ])

    # 기타 파라미터들
    param_mapping = {
        'models.head.postprocess.thresh': 'models.head.postprocess.thresh',
        'models.head.postprocess.box_thresh': 'models.head.postprocess.box_thresh',
        'models.head.postprocess.box_unclip_ratio': 'models.head.postprocess.box_unclip_ratio',
        'models.head.postprocess.polygon_unclip_ratio': 'models.head.postprocess.polygon_unclip_ratio',
        'models.loss.negative_ratio': 'models.loss.negative_ratio',
        'models.loss.prob_map_loss_weight': 'models.loss.prob_map_loss_weight',
        'models.loss.thresh_map_loss_weight': 'models.loss.thresh_map_loss_weight',
        'models.loss.binary_map_loss_weight': 'models.loss.binary_map_loss_weight',
        'models.head.k': 'models.head.k',
        'models.optimizer._target_': 'models.optimizer._target_',
        'models.optimizer.lr': 'models.optimizer.lr',
        'models.optimizer.weight_decay': 'models.optimizer.weight_decay',
        'trainer.max_epochs': 'trainer.max_epochs',
        'collate_fn.shrink_ratio': 'collate_fn.shrink_ratio',
        'collate_fn.thresh_min': 'collate_fn.thresh_min',
        'collate_fn.thresh_max': 'collate_fn.thresh_max',
    }

    for sweep_key, config_key in param_mapping.items():
        if sweep_key in sweep_config:
            overrides.append(f"{config_key}={sweep_config[sweep_key]}")

    # 스케줄러 설정 - 새로운 파라미터 구조 사용
    scheduler_type = sweep_config.get('scheduler_type')
    if scheduler_type:
        if scheduler_type == 'StepLR':
            overrides.append("models.scheduler._target_=torch.optim.lr_scheduler.StepLR")
            if 'step_lr_step_size' in sweep_config:
                overrides.append(f"models.scheduler.step_size={sweep_config['step_lr_step_size']}")
            if 'step_lr_gamma' in sweep_config:
                overrides.append(f"models.scheduler.gamma={sweep_config['step_lr_gamma']}")
        elif scheduler_type == 'CosineAnnealingLR':
            overrides.append("models.scheduler._target_=torch.optim.lr_scheduler.CosineAnnealingLR")
            if 'cosine_lr_T_max' in sweep_config:
                overrides.append(f"models.scheduler.T_max={sweep_config['cosine_lr_T_max']}")

    # exp_name 설정 (기본값에 sweep run name 추가)
    base_exp_name = getattr(config, 'exp_name', 'ocr_training')
    sweep_exp_name = f"{base_exp_name}_sweep_{wandb.run.name}"

    # 기본 설정
    overrides.extend([
        "preset=example",
        f"dataset_base_path={os.environ.get('DATASET_BASE_PATH', '/root/dev/upstageailab-ocr-recsys-competition-ocr-4/data/datasets/')}",
        f"exp_name={sweep_exp_name}",
        "wandb=True"
    ])

    # Hydra config 업데이트
    from omegaconf import OmegaConf
    for override in overrides:
        key, value = override.split('=', 1)

        # 타입 변환 (더 안전한 방식)
        try:
            # 이미 올바른 타입이면 그대로 사용
            if not isinstance(value, str):
                pass
            elif value.lower() in ['true', 'false']:
                value = value.lower() == 'true'
            elif value.startswith('[') and value.endswith(']'):
                value = eval(value)
            elif value.replace('.', '').replace('-', '').isdigit():
                if '.' in value:
                    value = float(value)
                else:
                    value = int(value)
        except Exception as e:
            print(f"Warning: Type conversion failed for {key}={value}: {e}")
            pass

        # OmegaConf를 사용한 안전한 설정
        try:
            keys = key.split('.')
            current = config
            for k in keys[:-1]:
                if k not in current:
                    current[k] = OmegaConf.create({})
                current = current[k]
            current[keys[-1]] = value
        except Exception as e:
            print(f"Warning: Failed to set {key}={value}: {e}")
            continue

    # 학습 실행
    import tempfile
    run_dir = Path(tempfile.mkdtemp(prefix=f"sweep_{wandb.run.name}_"))

    # log_dir 직접 설정 (HydraConfig 보간 문제 해결)
    log_dir = run_dir / "logs"
    config.log_dir = str(log_dir)

    log_path = setup_console_logging(log_dir, "train.log")

    start_time = datetime.now()
    print(f"[{start_time:%Y-%m-%d %H:%M:%S}] Training run started. Log file: {log_path}", flush=True)

    try:
        pl.seed_everything(config.get("seed", 42), workers=True)

        model_module, data_module = get_pl_modules_by_cfg(config)

        # WandB Logger 사용
        from lightning.pytorch.loggers import WandbLogger
        logger = WandbLogger(
            project=os.environ.get('WANDB_PROJECT', getattr(config, 'project_name', 'OCRProject')),
            name=f"{sweep_exp_name}",
            config=dict(sweep_config)
        )

        # checkpoint_dir 직접 설정 (HydraConfig 보간 문제 해결)
        checkpoint_dir = run_dir / "checkpoints"
        config.checkpoint_dir = str(checkpoint_dir)

        checkpoint_callback = ModelCheckpoint(
            dirpath=str(checkpoint_dir),
            save_top_k=3,
            monitor='val/cleval_hmean',
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
        wandb.finish()

def run_sweep():
    """WandB sweep 실행"""
    # train.yaml에서 프로젝트 설정 로드
    from hydra import initialize, compose
    with initialize(config_path=CONFIG_DIR, version_base='1.2'):
        config = compose(config_name='train')

    # WandB 프로젝트 설정 (train.yaml 우선, 환경변수로 오버라이드 가능)
    project_name = os.environ.get('WANDB_PROJECT', getattr(config, 'project_name', 'OCR-HRNet-Sweep'))
    entity = os.environ.get('WANDB_ENTITY', None)

    # Sweep 생성
    sweep_config = get_sweep_config()
    sweep_id = wandb.sweep(sweep_config, project=project_name, entity=entity)

    print(f"Created sweep: {sweep_id}")
    print(f"Project: {project_name}")
    print(f"Sweep URL: https://wandb.ai/{entity if entity else 'your_username'}/{project_name}/sweeps/{sweep_id}")
    print(f"Run: wandb agent {sweep_id}")

    return sweep_id

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='WandB Sweep for OCR HRNet optimization')
    parser.add_argument('--create-sweep', action='store_true', help='Create new sweep')
    parser.add_argument('--sweep-id', type=str, help='Existing sweep ID to join')
    parser.add_argument('--count', type=int, default=50, help='Number of runs')

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

        # Entity와 project 정보 추출 (train.yaml 우선)
        project_name = os.environ.get('WANDB_PROJECT', getattr(config, 'project_name', 'OCR-HRNet-Sweep'))
        entity = os.environ.get('WANDB_ENTITY', None)

        print(f"Starting sweep agent for project: {project_name}")
        wandb.agent(args.sweep_id, train_with_sweep, count=args.count,
                   project=project_name, entity=entity)
    else:
        # 기본 동작: sweep config 출력
        import yaml
        sweep_config = get_sweep_config()
        print("Sweep configuration:")
        print(yaml.dump(sweep_config, default_flow_style=False))
        print("\nTo create a sweep:")
        print("python wandb_sweep.py --create-sweep")
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

# HRNet 모델별 in_channels 매핑 (현재 작동하는 설정 기준)
HRNET_CHANNELS = {
    'hrnet_w18': [128, 256, 512, 1024],   # 모든 HRNet 모델이 동일한 채널 구조 사용
    'hrnet_w32': [128, 256, 512, 1024],   # timm features_only에서 확인된 구조
    'hrnet_w40': [128, 256, 512, 1024],   # select_features=[1,2,3,4] 사용시
    'hrnet_w44': [128, 256, 512, 1024],   # 모든 HRNet 계열이 동일
    'hrnet_w48': [128, 256, 512, 1024],   # 동일한 출력 채널 구조
}

def load_sweep_config():
    """sweep_config.yaml 파일에서 설정 로드"""
    import yaml

    config_path = PROJECT_ROOT / "sweep_config.yaml"

    if not config_path.exists():
        raise FileNotFoundError(
            f"Sweep config file not found: {config_path}\n"
            f"Please ensure 'sweep_config.yaml' exists in the project root."
        )

    with open(config_path, 'r', encoding='utf-8') as f:
        sweep_config = yaml.safe_load(f)

    print(f"Loaded sweep config from: {config_path}")
    return sweep_config

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

def adjust_batch_size_for_memory(image_size, suggested_batch_size):
    """GPU 메모리에 따른 배치 사이즈 자동 조정"""
    # 매우 보수적인 메모리 기반 조정 (OOM 방지를 위해 더 작게 설정)
    memory_mapping = {
        1280: 1,  # 매우 작게 조정
        1024: 2,  # 매우 작게 조정
        800: 4,   # 매우 작게 조정
        640: 8,   # 매우 작게 조정
    }

    max_batch_size = memory_mapping.get(image_size, 4)
    return min(suggested_batch_size, max_batch_size)

def train_with_sweep():
    """WandB sweep agent에서 호출되는 학습 함수"""

    # GPU 메모리 정리
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"GPU memory cleared. Available: {torch.cuda.get_device_properties(0).total_memory // 1024**3}GB")

    print("Starting sweep agent for project: OCRProject")

    # 현재 시간으로 run name 생성
    from datetime import datetime
    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    run_name = f"sweep_{timestamp}"

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

            # RandomBrightnessContrast 파라미터 오버라이드
            rbc_param_mapping = {
                'brightness_limit': 'brightness_limit',
                'contrast_limit': 'contrast_limit',
                'brightness_contrast_p': 'p',
            }
            rbc_overrides = {}
            for sweep_key, transform_key in rbc_param_mapping.items():
                if sweep_key in sweep_config:
                    rbc_overrides[transform_key] = convert_sweep_value(sweep_config[sweep_key])

            if rbc_overrides:
                rbc_transform = None
                for transform_cfg in transforms_list:
                    target = transform_cfg.get('_target_') if hasattr(transform_cfg, 'get') else getattr(transform_cfg, '_target_', None)
                    if target == 'albumentations.RandomBrightnessContrast':
                        rbc_transform = transform_cfg
                        break

                if rbc_transform is not None:
                    OmegaConf.set_struct(rbc_transform, False)
                    for key, value in rbc_overrides.items():
                        rbc_transform[key] = value
                    print(
                        "RandomBrightnessContrast overrides applied:",
                        {k: rbc_transform[k] for k in rbc_overrides.keys()}
                    )
                else:
                    print("RandomBrightnessContrast transform not found; skipping overrides.")

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

    # 기타 파라미터들 - config에 직접 설정 (타입 변환 후)
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
            # 타입 변환 후 config에 직접 설정
            value = convert_sweep_value(sweep_config[sweep_key])
            keys = config_key.split('.')
            current = config
            for k in keys[:-1]:
                if k not in current:
                    current[k] = OmegaConf.create({})
                current = current[k]
            current[keys[-1]] = value

    # 스케줄러 설정 - 완전한 스케줄러 교체
    scheduler_target = sweep_config.get('models.scheduler._target_')
    if scheduler_target:
        config.models.scheduler._target_ = scheduler_target
        if scheduler_target == 'torch.optim.lr_scheduler.StepLR':
            # StepLR로 완전 교체
            # T_max 파라미터 제거
            if hasattr(config.models.scheduler, 'T_max'):
                delattr(config.models.scheduler, 'T_max')
            # StepLR 파라미터 설정
            if 'models.scheduler.step_size' in sweep_config:
                config.models.scheduler.step_size = convert_sweep_value(sweep_config['models.scheduler.step_size'])
            if 'models.scheduler.gamma' in sweep_config:
                config.models.scheduler.gamma = convert_sweep_value(sweep_config['models.scheduler.gamma'])
        elif scheduler_target == 'torch.optim.lr_scheduler.CosineAnnealingLR':
            # CosineAnnealingLR로 완전 교체 - T_max만 설정
            # step_size, gamma 파라미터 제거
            if hasattr(config.models.scheduler, 'step_size'):
                delattr(config.models.scheduler, 'step_size')
            if hasattr(config.models.scheduler, 'gamma'):
                delattr(config.models.scheduler, 'gamma')
            # T_max 설정
            if 'models.scheduler.T_max' in sweep_config:
                config.models.scheduler.T_max = convert_sweep_value(sweep_config['models.scheduler.T_max'])

    # exp_name 설정 (기본값에 timestamp 추가, sweep 중복 방지)
    base_exp_name = getattr(config, 'exp_name', 'ocr_training')
    # wandb.run.name이 이미 "sweep_timestamp" 형식이므로 그대로 사용
    sweep_exp_name = f"{base_exp_name}_{wandb.run.name}"

    # 기본 설정
    overrides.extend([
        "preset=example",
        # f"dataset_base_path={os.environ.get('DATASET_BASE_PATH', '/root/dev/upstageailab-ocr-recsys-competition-ocr-4/data/datasets/')}",
        f"exp_name={sweep_exp_name}",
        "wandb=True"
    ])

    # Hydra config 업데이트 - 나머지 overrides 적용
    from omegaconf import OmegaConf

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

        # Sweep agent의 run을 재사용하므로 finish() 호출하지 않음
        # wandb.finish()를 호출하면 sweep agent의 run이 종료되고,
        # WandbLogger가 새 run을 만들어서 sweep 연결이 끊어짐

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
                        # artifact 생성
                        artifact = wandb.Artifact(
                            name=f"submission_csv_{wandb.run.id}",
                            type="submission",
                            description=f"CSV submission file from epoch {best_epoch if best_epoch is not None else 'unknown'}"
                        )

                        # CSV 파일 추가
                        artifact.add_file(csv_path, name="submission.csv")

                        # artifact 업로드
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
    """WandB sweep 실행"""
    # train.yaml에서 프로젝트 설정 로드
    from hydra import initialize, compose
    with initialize(config_path=CONFIG_DIR, version_base='1.2'):
        config = compose(config_name='train')

    # WandB 프로젝트 설정 (train.yaml 우선, 환경변수로 오버라이드 가능)
    project_name = os.environ.get('WANDB_PROJECT', getattr(config, 'project_name', 'OCR-HRNet-Sweep'))
    entity = os.environ.get('WANDB_ENTITY', None)

    # Sweep 생성
    sweep_config = load_sweep_config()
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
        sweep_config = load_sweep_config()
        print("Sweep configuration:")
        print(yaml.dump(sweep_config, default_flow_style=False))
        print("\nTo create a sweep:")
        print("python wandb_sweep.py --create-sweep")

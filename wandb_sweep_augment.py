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
    """sweep_config_augment.yaml 파일에서 설정 로드"""
    import yaml

    config_path = PROJECT_ROOT / "sweep_config_augment.yaml"

    if not config_path.exists():
        raise FileNotFoundError(
            f"Sweep config file not found: {config_path}\n"
            f"Please ensure 'sweep_config_augment.yaml' exists in the project root."
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

    print("Starting sweep agent for Augmentation optimization")

    # 현재 시간으로 run name 생성
    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    run_name = f"augment_sweep_{timestamp}"

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
        # f"dataset_base_path={os.environ.get('DATASET_BASE_PATH', '/root/dev/upstageailab-ocr-recsys-competition-ocr-4/data/datasets/')}",
        "wandb=True",

        # # 모델 설정
        # "models.encoder.model_name=hrnet_w44",
        # "models.encoder.select_features=[1,2,3,4]",
        # "models.decoder.in_channels=[128,256,512,1024]",

        # # 이미지 사이즈 설정 (1024)
        # "transforms.train_transform.transforms.0.max_size=1024",
        # "transforms.train_transform.transforms.1.min_width=1024",
        # "transforms.train_transform.transforms.1.min_height=1024",
        # "transforms.val_transform.transforms.0.max_size=1024",
        # "transforms.val_transform.transforms.1.min_width=1024",
        # "transforms.val_transform.transforms.1.min_height=1024",
        # "transforms.test_transform.transforms.0.max_size=1024",
        # "transforms.test_transform.transforms.1.min_width=1024",
        # "transforms.test_transform.transforms.1.min_height=1024",

        # # 배치 사이즈 (2)
        # "dataloaders.train_dataloader.batch_size=2",
        # "dataloaders.val_dataloader.batch_size=2",
        # "dataloaders.test_dataloader.batch_size=2",

        # # 후처리 파라미터
        # "models.head.postprocess.thresh=0.23105253214239585",
        # "models.head.postprocess.box_thresh=0.4324259445084524",
        # "models.head.postprocess.box_unclip_ratio=1.4745700672729625",
        # "models.head.postprocess.polygon_unclip_ratio=1.9770744341268096",

        # # Loss weights
        # "models.loss.negative_ratio=2.824132345320219",
        # "models.loss.prob_map_loss_weight=3.591196851512631",
        # "models.loss.thresh_map_loss_weight=8.028627860143937",
        # "models.loss.binary_map_loss_weight=0.6919312670387725",

        # # DBHead k
        # "models.head.k=45",

        # # Optimizer (AdamW)
        # "models.optimizer._target_=torch.optim.AdamW",
        # "models.optimizer.lr=0.0013358832166152786",
        # "models.optimizer.weight_decay=0.0003571900294890783",

        # # Scheduler (CosineAnnealingLR)
        # "models.scheduler._target_=torch.optim.lr_scheduler.CosineAnnealingLR",
        # "models.scheduler.T_max=10",

        # # Epochs
        # "trainer.max_epochs=13",

        # # CollateFN 파라미터
        # "collate_fn.shrink_ratio=0.428584820771695",
        # "collate_fn.thresh_max=0.7506908133484191",
        # "collate_fn.thresh_min=0.33967147700431666",
    ]

    # ==========================================
    # 활성화된 Augmentation 파라미터만 sweep
    # ==========================================

    train_transforms = config.transforms.train_transform.transforms
    transform_index_map = {}
    for idx, transform in enumerate(train_transforms):
        target_name = transform.get('_target_') if hasattr(transform, 'get') else None
        if target_name:
            transform_index_map[target_name] = idx

    def apply_transform_params(target_name: str, params: dict, label: str):
        idx = transform_index_map.get(target_name)
        if idx is None:
            print(f"Skipping augmentation override for {label} ({target_name}) - transform not found in base config")
            return False

        transform_cfg = train_transforms[idx]
        OmegaConf.set_struct(transform_cfg, False)
        for key, value in params.items():
            transform_cfg[key] = value
        return True

    applied_logs = []

    # 1. RandomBrightnessContrast
    brightness_limit = sweep_config.get('brightness_limit', 0.2)
    contrast_limit = sweep_config.get('contrast_limit', 0.2)
    brightness_contrast_p = sweep_config.get('brightness_contrast_p', 0.3)
    if apply_transform_params(
        'albumentations.RandomBrightnessContrast',
        {
            'brightness_limit': brightness_limit,
            'contrast_limit': contrast_limit,
            'p': brightness_contrast_p,
        },
        'RandomBrightnessContrast',
    ):
        applied_logs.append(
            f"RandomBrightnessContrast: brightness_limit={brightness_limit}, contrast_limit={contrast_limit}, p={brightness_contrast_p}"
        )

    # 2. ColorJitter
    # color_jitter_brightness = sweep_config.get('color_jitter_brightness', 0.2)
    # color_jitter_contrast = sweep_config.get('color_jitter_contrast', 0.2)
    # color_jitter_saturation = sweep_config.get('color_jitter_saturation', 0.2)
    # color_jitter_hue = sweep_config.get('color_jitter_hue', 0.1)
    # color_jitter_p = sweep_config.get('color_jitter_p', 0.3)
    # if apply_transform_params(
    #     'albumentations.ColorJitter',
    #     {
    #         'brightness': color_jitter_brightness,
    #         'contrast': color_jitter_contrast,
    #         'saturation': color_jitter_saturation,
    #         'hue': color_jitter_hue,
    #         'p': color_jitter_p,
    #     },
    #     'ColorJitter',
    # ):
    #     applied_logs.append(
    #         f"ColorJitter: brightness={color_jitter_brightness}, contrast={color_jitter_contrast}, saturation={color_jitter_saturation}, hue={color_jitter_hue}, p={color_jitter_p}"
    #     )

    # 3. RandomGamma
    gamma_limit_lower = int(sweep_config.get('gamma_limit_lower', 80))
    gamma_limit_upper = int(sweep_config.get('gamma_limit_upper', 120))
    random_gamma_p = sweep_config.get('random_gamma_p', 0.3)
    if apply_transform_params(
        'albumentations.RandomGamma',
        {
            'gamma_limit': [gamma_limit_lower, gamma_limit_upper],
            'p': random_gamma_p,
        },
        'RandomGamma',
    ):
        applied_logs.append(
            f"RandomGamma: gamma_limit=[{gamma_limit_lower},{gamma_limit_upper}], p={random_gamma_p}"
        )

    # 4. HueSaturationValue
    hue_shift_limit = int(sweep_config.get('hue_shift_limit', 20))
    sat_shift_limit = int(sweep_config.get('sat_shift_limit', 30))
    val_shift_limit = int(sweep_config.get('val_shift_limit', 20))
    hsv_p = sweep_config.get('hsv_p', 0.3)
    if apply_transform_params(
        'albumentations.HueSaturationValue',
        {
            'hue_shift_limit': hue_shift_limit,
            'sat_shift_limit': sat_shift_limit,
            'val_shift_limit': val_shift_limit,
            'p': hsv_p,
        },
        'HueSaturationValue',
    ):
        applied_logs.append(
            f"HueSaturationValue: hue_shift_limit={hue_shift_limit}, sat_shift_limit={sat_shift_limit}, val_shift_limit={val_shift_limit}, p={hsv_p}"
        )

    # 5. GaussianBlur
    gaussian_blur_limit = int(sweep_config.get('gaussian_blur_limit', 5))
    if gaussian_blur_limit % 2 == 0:
        gaussian_blur_limit += 1
    gaussian_blur_p = sweep_config.get('gaussian_blur_p', 0.2)
    if apply_transform_params(
        'albumentations.GaussianBlur',
        {
            'blur_limit': [3, gaussian_blur_limit],
            'p': gaussian_blur_p,
        },
        'GaussianBlur',
    ):
        applied_logs.append(
            f"GaussianBlur: blur_limit=[3,{gaussian_blur_limit}], p={gaussian_blur_p}"
        )

    # 6. MotionBlur
    motion_blur_limit = int(sweep_config.get('motion_blur_limit', 5))
    if motion_blur_limit % 2 == 0:
        motion_blur_limit += 1
    motion_blur_p = sweep_config.get('motion_blur_p', 0.2)
    if apply_transform_params(
        'albumentations.MotionBlur',
        {
            'blur_limit': [3, motion_blur_limit],
            'p': motion_blur_p,
        },
        'MotionBlur',
    ):
        applied_logs.append(
            f"MotionBlur: blur_limit=[3,{motion_blur_limit}], p={motion_blur_p}"
        )

    # 7. GaussNoise
    gauss_noise_std_lower = sweep_config.get('gauss_noise_std_lower', 0.02)
    gauss_noise_std_upper = sweep_config.get('gauss_noise_std_upper', 0.08)
    gauss_noise_p = sweep_config.get('gauss_noise_p', 0.2)
    if apply_transform_params(
        'albumentations.GaussNoise',
        {
            'std_range': [gauss_noise_std_lower, gauss_noise_std_upper],
            'p': gauss_noise_p,
        },
        'GaussNoise',
    ):
        applied_logs.append(
            f"GaussNoise: std_range=[{gauss_noise_std_lower},{gauss_noise_std_upper}], p={gauss_noise_p}"
        )

    # 8. ImageCompression
    compression_quality_lower = int(sweep_config.get('compression_quality_lower', 75))
    compression_quality_upper = int(sweep_config.get('compression_quality_upper', 100))
    image_compression_p = sweep_config.get('image_compression_p', 0.2)
    if apply_transform_params(
        'albumentations.ImageCompression',
        {
            'quality_range': [compression_quality_lower, compression_quality_upper],
            'p': image_compression_p,
        },
        'ImageCompression',
    ):
        applied_logs.append(
            f"ImageCompression: quality_range=[{compression_quality_lower},{compression_quality_upper}], p={image_compression_p}"
        )

    # 9. Sharpen
    sharpen_alpha_lower = sweep_config.get('sharpen_alpha_lower', 0.2)
    sharpen_alpha_upper = sweep_config.get('sharpen_alpha_upper', 0.5)
    sharpen_lightness_lower = sweep_config.get('sharpen_lightness_lower', 0.5)
    sharpen_lightness_upper = sweep_config.get('sharpen_lightness_upper', 1.0)
    sharpen_p = sweep_config.get('sharpen_p', 0.2)
    if apply_transform_params(
        'albumentations.Sharpen',
        {
            'alpha': [sharpen_alpha_lower, sharpen_alpha_upper],
            'lightness': [sharpen_lightness_lower, sharpen_lightness_upper],
            'p': sharpen_p,
        },
        'Sharpen',
    ):
        applied_logs.append(
            f"Sharpen: alpha=[{sharpen_alpha_lower},{sharpen_alpha_upper}], lightness=[{sharpen_lightness_lower},{sharpen_lightness_upper}], p={sharpen_p}"
        )

    # 10. Downscale
    downscale_lower = sweep_config.get('downscale_lower', 0.75)
    downscale_upper = sweep_config.get('downscale_upper', 0.95)
    downscale_p = sweep_config.get('downscale_p', 0.2)
    if apply_transform_params(
        'albumentations.Downscale',
        {
            'scale_range': [downscale_lower, downscale_upper],
            'p': downscale_p,
        },
        'Downscale',
    ):
        applied_logs.append(
            f"Downscale: scale_range=[{downscale_lower},{downscale_upper}], p={downscale_p}"
        )

    # 11. RandomShadow
    # shadow_num_lower = int(sweep_config.get('shadow_num_lower', 1))
    # shadow_num_upper = int(sweep_config.get('shadow_num_upper', 2))
    # shadow_dimension = int(sweep_config.get('shadow_dimension', 5))
    # random_shadow_p = sweep_config.get('random_shadow_p', 0.2)
    # if apply_transform_params(
    #     'albumentations.RandomShadow',
    #     {
    #         'num_shadows_limit': [shadow_num_lower, shadow_num_upper],
    #         'shadow_dimension': shadow_dimension,
    #         'p': random_shadow_p,
    #     },
    #     'RandomShadow',
    # ):
    #     applied_logs.append(
    #         f"RandomShadow: num_shadows_limit=[{shadow_num_lower},{shadow_num_upper}], shadow_dimension={shadow_dimension}, p={random_shadow_p}"
    #     )

    # 12. PlasmaShadow
    plasma_shadow_intensity_lower = sweep_config.get('plasma_shadow_intensity_lower', 0.3)
    plasma_shadow_intensity_upper = sweep_config.get('plasma_shadow_intensity_upper', 0.7)
    plasma_roughness = sweep_config.get('plasma_roughness', 3.0)
    plasma_shadow_p = sweep_config.get('plasma_shadow_p', 0.2)
    if apply_transform_params(
        'albumentations.PlasmaShadow',
        {
            'shadow_intensity_range': [plasma_shadow_intensity_lower, plasma_shadow_intensity_upper],
            'roughness': plasma_roughness,
            'p': plasma_shadow_p,
        },
        'PlasmaShadow',
    ):
        applied_logs.append(
            f"PlasmaShadow: shadow_intensity_range=[{plasma_shadow_intensity_lower},{plasma_shadow_intensity_upper}], roughness={plasma_roughness}, p={plasma_shadow_p}"
        )

    # 13. RandomFog
    fog_coef_lower = sweep_config.get('fog_coef_lower', 0.1)
    fog_coef_upper = sweep_config.get('fog_coef_upper', 0.3)
    fog_alpha_coef = sweep_config.get('fog_alpha_coef', 0.08)
    random_fog_p = sweep_config.get('random_fog_p', 0.2)
    if apply_transform_params(
        'albumentations.RandomFog',
        {
            'fog_coef_range': [fog_coef_lower, fog_coef_upper],
            'alpha_coef': fog_alpha_coef,
            'p': random_fog_p,
        },
        'RandomFog',
    ):
        applied_logs.append(
            f"RandomFog: fog_coef_range=[{fog_coef_lower},{fog_coef_upper}], alpha_coef={fog_alpha_coef}, p={random_fog_p}"
        )

    if applied_logs:
        print("Augmentation params configured:")
        for log_line in applied_logs:
            print(f"  {log_line}")

    # exp_name 설정
    sweep_exp_name = f"augment_sweep_{wandb.run.name}"
    overrides.append(f"exp_name={sweep_exp_name}")

    # Hydra config 업데이트
    from omegaconf import OmegaConf

    # 스케줄러 파라미터 정리 - 스케줄러 타입에 따라 처리
    scheduler_target = config.models.scheduler.get('_target_', '')

    if 'CosineAnnealingLR' in scheduler_target:
        # CosineAnnealingLR인 경우: StepLR 파라미터 제거, T_max 추가
        if hasattr(config.models.scheduler, 'step_size'):
            OmegaConf.set_struct(config.models.scheduler, False)
            delattr(config.models.scheduler, 'step_size')
            print("Removed step_size from scheduler config (CosineAnnealingLR)")
        if hasattr(config.models.scheduler, 'gamma'):
            OmegaConf.set_struct(config.models.scheduler, False)
            delattr(config.models.scheduler, 'gamma')
            print("Removed gamma from scheduler config (CosineAnnealingLR)")

        # T_max가 없으면 추가
        if not hasattr(config.models.scheduler, 'T_max'):
            OmegaConf.set_struct(config.models.scheduler, False)
            config.models.scheduler.T_max = 10
            print("Added T_max=10 to scheduler config (CosineAnnealingLR)")

    elif 'StepLR' in scheduler_target:
        # StepLR인 경우: T_max 제거 (있다면)
        if hasattr(config.models.scheduler, 'T_max'):
            OmegaConf.set_struct(config.models.scheduler, False)
            delattr(config.models.scheduler, 'T_max')
            print("Removed T_max from scheduler config (StepLR)")
    else:
        # 기타 스케줄러: 로그만 출력
        print(f"Scheduler type: {scheduler_target} - no parameter adjustment")

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
            project=os.environ.get('WANDB_PROJECT', 'OCR-Augment-Sweep'),
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

        # # 베스트 체크포인트 로드 및 CSV 제출파일 생성
        # best_checkpoint = checkpoint_callback.best_model_path
        # if best_checkpoint:
        #     checkpoint = torch.load(best_checkpoint, map_location='cpu')
        #     best_epoch = checkpoint.get("epoch")
        #     state_dict = checkpoint.get("state_dict", checkpoint)
        #     model_module.load_state_dict(state_dict)

        #     # CSV 생성을 위한 predict 실행
        #     trainer.predict(model_module, data_module)

        #     if getattr(model_module, "last_submission_paths", None):
        #         paths = model_module.last_submission_paths
        #         print(f"Submission JSON saved to: {paths['json']}")
        #         if best_epoch is not None:
        #             print(f"Submission CSV saved to: {paths['csv']} (best epoch: {best_epoch})")
        #         else:
        #             print(f"Submission CSV saved to: {paths['csv']} (best epoch metadata unavailable)")

        #         # CSV 파일을 WandB artifact로 업로드
        #         try:
        #             csv_path = paths['csv']
        #             if os.path.exists(csv_path):
        #                 artifact = wandb.Artifact(
        #                     name=f"submission_csv_{wandb.run.id}",
        #                     type="submission",
        #                     description=f"CSV submission file from epoch {best_epoch if best_epoch is not None else 'unknown'}"
        #                 )
        #                 artifact.add_file(csv_path, name="submission.csv")
        #                 wandb.log_artifact(artifact)
        #                 print(f"Submission CSV uploaded as WandB artifact: submission_csv_{wandb.run.id}")
        #             else:
        #                 print(f"CSV file not found at: {csv_path}")
        #         except Exception as e:
        #             print(f"Failed to upload CSV as artifact: {e}")
        # else:
        #     print("Model checkpoint was not created; skipping submission generation.")

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
    """WandB sweep 실행 - sweep_config_augment.yaml 파일 사용"""
    # train.yaml에서 프로젝트 설정 로드
    from hydra import initialize, compose
    with initialize(config_path=CONFIG_DIR, version_base='1.2'):
        config = compose(config_name='train')

    # sweep_config_augment.yaml 파일에서 설정 로드
    sweep_config = load_sweep_config()

    # WandB 프로젝트 설정 (YAML 파일에서 가져오되, 환경변수로 오버라이드 가능)
    project_name = os.environ.get('WANDB_PROJECT', sweep_config.get('project', 'OCR-Augment-Sweep'))
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

    parser = argparse.ArgumentParser(description='WandB Sweep for Augmentation optimization')
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
        project_name = os.environ.get('WANDB_PROJECT', 'OCR-Augment-Sweep')
        entity = os.environ.get('WANDB_ENTITY', None)

        print(f"Starting sweep agent for project: {project_name}")
        wandb.agent(args.sweep_id, train_with_sweep, count=args.count,
                   project=project_name, entity=entity)
    else:
        # 기본 동작: sweep config 출력
        import yaml
        try:
            sweep_config = load_sweep_config()
            print("Sweep configuration from sweep_config_augment.yaml:")
            print(yaml.dump(sweep_config, default_flow_style=False))
            print("\nTo create a sweep:")
            print("uv run python wandb_sweep_augment.py --create-sweep")
        except FileNotFoundError as e:
            print(f"Error: {e}")
            sys.exit(1)

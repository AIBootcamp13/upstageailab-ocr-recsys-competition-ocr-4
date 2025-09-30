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
from hydra.core.hydra_config import HydraConfig
from lightning.pytorch.callbacks import (  # noqa
    LearningRateMonitor,
    ModelCheckpoint,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ocr.lightning_modules import get_pl_modules_by_cfg  # noqa: E402
from ocr.utils.console_logging import setup_console_logging  # noqa: E402

CONFIG_DIR = os.environ.get('OP_CONFIG_DIR') or '../configs'


@hydra.main(config_path=CONFIG_DIR, config_name='train', version_base='1.2')
def train(config):
    """
    Train a OCR model using the provided configuration.

    Args:
        `config` (dict): A dictionary containing configuration settings for training.
    """
    run_dir = Path(HydraConfig.get().runtime.output_dir)
    log_dir = Path(getattr(config, "log_dir", run_dir / "logs"))
    log_path = setup_console_logging(log_dir, "train.log")

    start_time = datetime.now()
    print(f"[{start_time:%Y-%m-%d %H:%M:%S}] Training run started. Log file: {log_path}", flush=True)

    best_checkpoint = None
    best_epoch = None

    try:
        pl.seed_everything(config.get("seed", 42), workers=True)

        model_module, data_module = get_pl_modules_by_cfg(config)

        if config.get("wandb"):
            from lightning.pytorch.loggers import WandbLogger as Logger  # noqa: E402
            logger = Logger(config.exp_name,
                            project=config.project_name,
                            config=dict(config),
                            )
        else:
            from lightning.pytorch.loggers.tensorboard import TensorBoardLogger  # noqa: E402
            logger = TensorBoardLogger(
                save_dir=str(log_dir),
                name=config.exp_name,
                version=config.exp_version,
                default_hp_metric=False,
            )

        checkpoint_dir = Path(getattr(config, "checkpoint_dir", run_dir / "checkpoints"))

        checkpoint_callback = ModelCheckpoint(
            dirpath=str(checkpoint_dir),
            save_top_k=3,
            # monitor='val/loss',
            # mode='min',
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

        best_checkpoint = checkpoint_callback.best_model_path
        if best_checkpoint:
            checkpoint = torch.load(best_checkpoint, map_location='cpu')
            best_epoch = checkpoint.get("epoch")
            state_dict = checkpoint.get("state_dict", checkpoint)
            model_module.load_state_dict(state_dict)

        trainer.test(
            model_module,
            data_module,
        )

        if best_checkpoint:
            if best_epoch is not None:
                print(f"Best checkpoint epoch: {best_epoch}")
            trainer.predict(
                model_module,
                data_module,
            )

            if getattr(model_module, "last_submission_paths", None):
                paths = model_module.last_submission_paths
                print(f"Submission JSON saved to: {paths['json']}")
                if best_epoch is not None:
                    print(f"Submission CSV saved to: {paths['csv']} (best epoch: {best_epoch})")
                else:
                    print(f"Submission CSV saved to: {paths['csv']} (best epoch metadata unavailable)")

                # WandB가 활성화된 경우 CSV 파일을 artifact로 업로드
                if config.get("wandb") and hasattr(logger, 'experiment'):
                    try:
                        import wandb
                        csv_path = paths['csv']
                        if os.path.exists(csv_path):
                            # artifact 생성
                            artifact = wandb.Artifact(
                                name=f"submission_csv_{logger.experiment.id}",
                                type="submission",
                                description=f"CSV submission file from epoch {best_epoch if best_epoch is not None else 'unknown'}"
                            )

                            # CSV 파일 추가
                            artifact.add_file(csv_path, name="submission.csv")

                            # artifact 업로드
                            wandb.log_artifact(artifact)
                            print(f"Submission CSV uploaded as WandB artifact: submission_csv_{logger.experiment.id}")
                        else:
                            print(f"CSV file not found at: {csv_path}")
                    except Exception as e:
                        print(f"Failed to upload CSV as artifact: {e}")
        else:
            print("Model checkpoint was not created; skipping submission generation.")
    finally:
        # WandB 정리 (sweep에서 실행되지 않을 때만)
        if config.get("wandb") and not os.environ.get('WANDB_SWEEP_ID'):
            try:
                import wandb
                wandb.finish()
            except Exception:
                pass  # WandB 종료 중 오류 무시

        end_time = datetime.now()
        elapsed = end_time - start_time
        print(f"[{end_time:%Y-%m-%d %H:%M:%S}] Training run finished. Duration: {elapsed}", flush=True)


if __name__ == "__main__":
    train()

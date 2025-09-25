import os
import sys
import warnings
from pathlib import Path
from datetime import datetime

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
            monitor='val/loss',
            mode='min',
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
        else:
            print("Model checkpoint was not created; skipping submission generation.")
    finally:
        end_time = datetime.now()
        elapsed = end_time - start_time
        print(f"[{end_time:%Y-%m-%d %H:%M:%S}] Training run finished. Duration: {elapsed}", flush=True)


if __name__ == "__main__":
    train()

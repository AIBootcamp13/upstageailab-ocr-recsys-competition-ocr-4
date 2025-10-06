import os
import sys
import warnings
import shutil
import re
from pathlib import Path

warnings.filterwarnings(
    "ignore",
    message=r"^pkg_resources is deprecated as an API",
)

import lightning.pytorch as pl
import hydra
from lightning.pytorch.callbacks import (  # noqa
    LearningRateMonitor,
    ModelCheckpoint,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ocr.lightning_modules import get_pl_modules_by_cfg  # noqa: E402

CONFIG_DIR = os.environ.get('OP_CONFIG_DIR') or '../configs'


@hydra.main(config_path=CONFIG_DIR, config_name='train', version_base='1.2')
def train(config):
    """
    Train a OCR model using the provided configuration.

    Args:
        `config` (dict): A dictionary containing configuration settings for training.
    """
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
            save_dir=config.log_dir,
            name=config.exp_name,
            version=config.exp_version,
            default_hp_metric=False,
        )

    checkpoint_path = config.checkpoint_dir

    callbacks = [
        LearningRateMonitor(logging_interval='step'),
        ModelCheckpoint(dirpath=checkpoint_path,
                        filename='{epoch}-{val/hmean:.3f}',
                        save_top_k=3, monitor='val/hmean', mode='max'),
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
    trainer.test(
        model_module,
        data_module,
    )

    # Copy best checkpoint to best folder
    best_dir = Path(checkpoint_path) / "best"
    best_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = list(Path(checkpoint_path).rglob("*.ckpt"))
    if checkpoints:
        def extract_hmean(path):
            match = re.search(r'-([0-9]+\.\d+)', path.name)
            if match:
                return float(match.group(1))
            return float('-inf')
        best_ckpt = max(checkpoints, key=extract_hmean)
        shutil.copy(best_ckpt, best_dir / "model.ckpt")


if __name__ == "__main__":
    train()

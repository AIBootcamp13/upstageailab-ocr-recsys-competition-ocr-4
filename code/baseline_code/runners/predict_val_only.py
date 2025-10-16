import os
import sys
from pathlib import Path

import hydra
from omegaconf import OmegaConf
import lightning.pytorch as pl
import json
from datetime import datetime
from collections import OrderedDict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from ocr.lightning_modules import get_pl_modules_by_cfg

CONFIG_DIR = "/root/dev/upstageailab-ocr-recsys-competition-ocr-4/code/baseline_code/configs"

@hydra.main(config_path=CONFIG_DIR, config_name="predict", version_base="1.2")
def predict_val(config):
    """VAL 데이터셋 예측을 위한 스크립트"""

    # Extract exp_name from checkpoint path
    checkpoint_path = Path(config.checkpoint_path)
    exp_name = checkpoint_path.parent.parent.parent.name

    config.exp_name = exp_name
    print(f"Extracted exp_name: {exp_name}")
    print(f"Predicting VAL dataset with checkpoint: {config.checkpoint_path}")

    # Load dataset config manually - correct absolute path
    dataset_config_path = Path("/root/dev/upstageailab-ocr-recsys-competition-ocr-4/code/baseline_code/configs") / "preset/datasets/db.yaml"
    dataset_cfg = OmegaConf.load(dataset_config_path)
    config.datasets.predict_dataset = dataset_cfg.datasets.val_dataset

    print("Replaced predict_dataset with val_dataset")
    print(f"Using VAL dataset: {config.datasets.predict_dataset.image_path}")

    pl.seed_everything(config.get("seed", 42), workers=True)
    model_module, data_module = get_pl_modules_by_cfg(config)

    trainer = pl.Trainer(logger=False)

    ckpt_path = config.get("checkpoint_path")
    assert ckpt_path is not None, "checkpoint_path must be provided"

    trainer.predict(model_module,
                    data_module,
                    ckpt_path=ckpt_path)

if __name__ == "__main__":
    predict_val()

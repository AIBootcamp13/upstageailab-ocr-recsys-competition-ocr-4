import os
import sys
from pathlib import Path
from datetime import datetime

import lightning.pytorch as pl
import hydra
from hydra.core.hydra_config import HydraConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from ocr.lightning_modules import get_pl_modules_by_cfg  # noqa: E402
from ocr.utils.console_logging import setup_console_logging  # noqa: E402

CONFIG_DIR = os.environ.get('OP_CONFIG_DIR') or '../configs'


@hydra.main(config_path=CONFIG_DIR, config_name='predict', version_base='1.2')
def predict(config):
    """
    Train a OCR model using the provided configuration.

    Args:
        `config` (dict): A dictionary containing configuration settings for predict.
    """
    run_dir = Path(HydraConfig.get().runtime.output_dir)
    log_dir = Path(getattr(config, "log_dir", run_dir / "logs"))
    log_path = setup_console_logging(log_dir, "predict.log")

    start_time = datetime.now()
    print(f"[{start_time:%Y-%m-%d %H:%M:%S}] Predict run started. Log file: {log_path}", flush=True)

    try:
        pl.seed_everything(config.get("seed", 42), workers=True)

        model_module, data_module = get_pl_modules_by_cfg(config)

        trainer = pl.Trainer(logger=False)

        ckpt_path = config.get("checkpoint_path")
        assert ckpt_path, "checkpoint_path must be provided for prediction"

        trainer.predict(model_module,
                        data_module,
                        ckpt_path=ckpt_path,
                        )
    finally:
        end_time = datetime.now()
        elapsed = end_time - start_time
        print(f"[{end_time:%Y-%m-%d %H:%M:%S}] Predict run finished. Duration: {elapsed}", flush=True)


if __name__ == "__main__":
    predict()

import argparse
import itertools
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import lightning.pytorch as pl
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ocr.lightning_modules import get_pl_modules_by_cfg  # noqa: E402

DEFAULT_CONFIG_DIR = os.environ.get("OP_CONFIG_DIR")
if DEFAULT_CONFIG_DIR is None:
    DEFAULT_CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
DEFAULT_CONFIG_DIR = str(Path(DEFAULT_CONFIG_DIR).resolve())


@dataclass
class SweepResult:
    params: Dict[str, float]
    metrics: Dict[str, float]


def compose_base_config(overrides: List[str]) -> DictConfig:
    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base="1.2", config_dir=DEFAULT_CONFIG_DIR):
        cfg = compose(config_name="test", overrides=overrides)
    return cfg


def clone_config(config: DictConfig) -> DictConfig:
    return OmegaConf.create(OmegaConf.to_container(config, resolve=False))


def iter_param_grid(param_values: Dict[str, List[float]]):
    if not param_values:
        yield {}
        return

    keys = list(param_values.keys())
    value_lists = [param_values[key] for key in keys]
    for combo in itertools.product(*value_lists):
        yield {key: value for key, value in zip(keys, combo)}


def evaluate_config(config: DictConfig) -> Dict[str, float]:
    pl.seed_everything(config.get("seed", 42), workers=True)

    model_module, data_module = get_pl_modules_by_cfg(config)

    trainer = pl.Trainer(
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        deterministic=True,
    )

    results = trainer.test(model_module, data_module, ckpt_path=config.checkpoint_path)
    if not results:
        raise RuntimeError("Lightning trainer did not return any test metrics.")

    converted = {}
    for key, value in results[0].items():
        if hasattr(value, "item"):
            converted[key] = float(value.item())
        else:
            converted[key] = float(value)

    torch.cuda.empty_cache()

    return converted


def _format_param(params: Dict[str, float], key: str) -> str:
    value = params.get(key)
    if value is None:
        return "base"
    return f"{value:.3f}"


def format_result(result: SweepResult) -> str:
    params = result.params
    metrics = result.metrics
    return (
        f"thresh={_format_param(params, 'thresh')}, "
        f"box_thresh={_format_param(params, 'box_thresh')}, "
        f"box_unclip_ratio={_format_param(params, 'box_unclip_ratio')}, "
        f"polygon_unclip_ratio={_format_param(params, 'polygon_unclip_ratio')} -> "
        f"recall={metrics.get('test/recall', float('nan')):.4f}, "
        f"precision={metrics.get('test/precision', float('nan')):.4f}, "
        f"hmean={metrics.get('test/hmean', float('nan')):.4f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Grid search for DB post-process parameters.")
    parser.add_argument("--checkpoint", required=True, help="Path to the Lightning checkpoint.")
    parser.add_argument(
        "--exp-name",
        default="postprocess_tuning",
        help="Experiment name used for logging directories.",
    )
    parser.add_argument(
        "--overrides",
        nargs="*",
        default=[],
        help="Additional Hydra overrides applied to the base configuration.",
    )
    parser.add_argument("--thresh", type=float, nargs="*", default=None)
    parser.add_argument("--box-thresh", type=float, nargs="*", default=None)
    parser.add_argument("--box-unclip-ratio", type=float, nargs="*", default=None)
    parser.add_argument("--polygon-unclip-ratio", type=float, nargs="*", default=None)
    parser.add_argument("--min-size", type=float, nargs="*", default=None)

    args = parser.parse_args()

    base_overrides = list(args.overrides)
    base_overrides.append(f"exp_name={args.exp_name}")

    base_config = compose_base_config(base_overrides)
    base_config.checkpoint_path = str(Path(args.checkpoint).resolve())

    param_grid: Dict[str, List[float]] = {}
    if args.thresh:
        param_grid["thresh"] = args.thresh
    if args.box_thresh:
        param_grid["box_thresh"] = args.box_thresh
    if args.box_unclip_ratio:
        param_grid["box_unclip_ratio"] = args.box_unclip_ratio
    if args.polygon_unclip_ratio:
        param_grid["polygon_unclip_ratio"] = args.polygon_unclip_ratio
    if args.min_size:
        param_grid["min_size"] = args.min_size

    results: List[SweepResult] = []

    for params in iter_param_grid(param_grid):
        current_cfg = clone_config(base_config)
        postprocess_cfg = current_cfg.models.head.postprocess
        for key, value in params.items():
            postprocess_cfg[key] = value

        metrics = evaluate_config(current_cfg)
        results.append(SweepResult(params=params, metrics=metrics))

        print(format_result(results[-1]))

    if not results:
        print("No parameter combinations were provided; nothing was evaluated.")
        return

    sorted_results = sorted(
        results,
        key=lambda item: item.metrics.get("test/hmean", float("nan")),
        reverse=True,
    )

    best = sorted_results[0]
    print("\nBest configuration:")
    print(format_result(best))


if __name__ == "__main__":
    main()

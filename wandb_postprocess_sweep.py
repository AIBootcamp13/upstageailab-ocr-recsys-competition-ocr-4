"""
후처리 파라미터 전용 WandB 스윕 스크립트.

주요 사용 예시는 아래와 같다.
1) 스윕 생성: uv run python wandb_postprocess_sweep.py --create-sweep --config-path sweep_config_postprocess.yaml
2) 에이전트 실행: uv run python wandb_postprocess_sweep.py --sweep-id <SWEEP_ID> --checkpoint <CKPT_PATH> --count 30
3) 단일 평가: uv run python wandb_postprocess_sweep.py --checkpoint <CKPT_PATH>

학습은 수행하지 않고, 지정한 체크포인트로 후처리 파라미터만 변경하며 추론을 반복한다.
"""

import argparse
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List

import lightning.pytorch as pl
import torch
import wandb
from dotenv import load_dotenv
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent
BASELINE_CODE_ROOT = PROJECT_ROOT / "code" / "baseline_code"

sys.path.insert(0, str(BASELINE_CODE_ROOT))

from ocr.lightning_modules import get_pl_modules_by_cfg  # noqa: E402

DEFAULT_CONFIG_DIR = os.environ.get("OP_CONFIG_DIR")
if DEFAULT_CONFIG_DIR is None:
    DEFAULT_CONFIG_DIR = BASELINE_CODE_ROOT / "configs"
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


DEFAULT_EXP_NAME = "hrnet_w44_1024_reproduction"
BASELINE_OVERRIDES: List[str] = [
    "preset=example",
    "models.encoder.model_name=hrnet_w44",
    "models.encoder.select_features=[1,2,3,4]",
    "models.decoder.in_channels=[128,256,512,1024]",
    "transforms.train_transform.transforms.0.max_size=1024",
    "transforms.train_transform.transforms.1.min_width=1024",
    "transforms.train_transform.transforms.1.min_height=1024",
    "transforms.val_transform.transforms.0.max_size=1024",
    "transforms.val_transform.transforms.1.min_width=1024",
    "transforms.val_transform.transforms.1.min_height=1024",
    "transforms.test_transform.transforms.0.max_size=1024",
    "transforms.test_transform.transforms.1.min_width=1024",
    "transforms.test_transform.transforms.1.min_height=1024",
    "dataloaders.train_dataloader.batch_size=2",
    "dataloaders.val_dataloader.batch_size=2",
    "dataloaders.test_dataloader.batch_size=2",
    "models.head.postprocess.thresh=0.23105253214239585",
    "models.head.postprocess.box_thresh=0.4324259445084524",
    "models.head.postprocess.box_unclip_ratio=1.4745700672729625",
    "models.head.postprocess.polygon_unclip_ratio=1.9770744341268096",
    "models.loss.negative_ratio=2.824132345320219",
    "models.loss.prob_map_loss_weight=3.591196851512631",
    "models.loss.thresh_map_loss_weight=8.028627860143937",
    "models.loss.binary_map_loss_weight=0.6919312670387725",
    "models.head.k=45",
    "models.optimizer._target_=torch.optim.AdamW",
    "models.optimizer.lr=0.0013358832166152786",
    "models.optimizer.weight_decay=0.0003571900294890783",
    "models.scheduler._target_=torch.optim.lr_scheduler.CosineAnnealingLR",
    "~models.scheduler.step_size",
    "~models.scheduler.gamma",
    "+models.scheduler.T_max=15",
    "collate_fn.shrink_ratio=0.428584820771695",
    "collate_fn.thresh_max=0.7506908133484191",
    "collate_fn.thresh_min=0.33967147700431666",
    "wandb=true",
]

DEFAULT_POSTPROCESS_PARAMS: Dict[str, float] = {
    "thresh": 0.23105253214239585,
    "box_thresh": 0.4324259445084524,
    "box_unclip_ratio": 1.4745700672729625,
    "polygon_unclip_ratio": 1.9770744341268096,
}

POSTPROCESS_KEYS: Iterable[str] = (
    "thresh",
    "box_thresh",
    "box_unclip_ratio",
    "polygon_unclip_ratio",
)


def _ensure_wandb_login() -> None:
    api_key = os.environ.get("WANDB_API_KEY")
    if not api_key:
        return
    try:
        wandb.login(key=api_key)
    except Exception as exc:
        print(f"W&B 로그인 실패: {exc}")


def build_overrides(exp_name: str, extra_overrides: List[str]) -> List[str]:
    overrides = list(BASELINE_OVERRIDES)
    overrides.append(f"exp_name={exp_name}")
    overrides.extend(extra_overrides)
    return overrides


def apply_postprocess_params(config, params: Dict[str, float]) -> None:
    post_cfg = config.models.head.postprocess
    for key, value in params.items():
        post_cfg[key] = value


def extract_postprocess_params(wandb_config) -> Dict[str, float]:
    params: Dict[str, float] = {}
    for key in POSTPROCESS_KEYS:
        if key in wandb_config:
            params[key] = float(wandb_config[key])
    return params


def evaluate_postprocess_run(
    checkpoint_path: Path,
    exp_name: str,
    extra_overrides: List[str],
    params: Dict[str, float],
) -> Dict[str, float]:
    overrides = build_overrides(exp_name, extra_overrides)
    base_cfg = compose_base_config(overrides)
    base_cfg.checkpoint_path = str(checkpoint_path.resolve())

    current_cfg = clone_config(base_cfg)
    apply_postprocess_params(current_cfg, params)

    metrics = evaluate_config(current_cfg)
    return metrics


def run_sweep_trial(
    checkpoint_path: Path,
    exp_name: str,
    extra_overrides: List[str],
) -> None:
    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    run_name = f"postprocess_{timestamp}"
    wandb.init(
        name=run_name,
        config=dict(DEFAULT_POSTPROCESS_PARAMS),
    )
    sweep_params = extract_postprocess_params(wandb.config)
    metrics = evaluate_postprocess_run(checkpoint_path, exp_name, extra_overrides, sweep_params)

    wandb.log(metrics)
    if "test/hmean" in metrics:
        wandb.run.summary["best_hmean"] = metrics["test/hmean"]
    wandb.run.summary["postprocess_params"] = sweep_params

    result = SweepResult(params=sweep_params, metrics=metrics)
    print(format_result(result))


def load_sweep_config(config_path: Path) -> Dict:
    import yaml

    if not config_path.exists():
        raise FileNotFoundError(f"Sweep config 파일을 찾을 수 없습니다: {config_path}")

    with open(config_path, "r", encoding="utf-8") as fp:
        return yaml.safe_load(fp)


def create_sweep(config_path: Path, project: str, entity: str | None) -> str:
    sweep_config = load_sweep_config(config_path)
    sweep_id = wandb.sweep(sweep_config, project=project, entity=entity)
    print(f"Sweep 생성: {sweep_id}")
    print(f"URL: https://wandb.ai/{entity or 'your_username'}/{project}/sweeps/{sweep_id}")
    print(f"Agent 실행: wandb agent {sweep_id}")
    return sweep_id


def start_agent(
    sweep_id: str,
    checkpoint_path: Path,
    exp_name: str,
    extra_overrides: List[str],
    project: str,
    entity: str | None,
    count: int,
) -> None:
    runner = lambda: run_sweep_trial(checkpoint_path, exp_name, extra_overrides)
    wandb.agent(
        sweep_id,
        function=runner,
        project=project,
        entity=entity,
        count=count,
    )


def print_sweep_config(config_path: Path) -> None:
    import yaml

    cfg = load_sweep_config(config_path)
    print(yaml.dump(cfg, default_flow_style=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="사전 학습된 체크포인트로 후처리 파라미터만 W&B Sweep 합니다.",
    )
    parser.add_argument("--checkpoint", type=Path, help="평가에 사용할 Lightning 체크포인트 경로")
    parser.add_argument("--exp-name", default=DEFAULT_EXP_NAME, help="Hydra exp_name 오버라이드 값")
    parser.add_argument(
        "--overrides",
        nargs="*",
        default=[],
        help="추가 Hydra 오버라이드",
    )
    parser.add_argument("--create-sweep", action="store_true", help="새 Sweep 생성")
    parser.add_argument("--sweep-id", type=str, help="기존 Sweep ID")
    parser.add_argument("--count", type=int, default=50, help="에이전트가 실행할 실험 횟수")
    parser.add_argument(
        "--config-path",
        type=Path,
        default=PROJECT_ROOT / "sweep_config_postprocess.yaml",
        help="W&B Sweep 설정 파일 경로",
    )
    parser.add_argument("--project", type=str, default=os.environ.get("WANDB_PROJECT", "ocr-postprocess"))
    parser.add_argument("--entity", type=str, default=os.environ.get("WANDB_ENTITY"))
    return parser.parse_args()


def main() -> None:
    _ensure_wandb_login()
    args = parse_args()

    if args.create_sweep:
        create_sweep(args.config_path, args.project, args.entity)
        return

    if args.sweep_id:
        if not args.checkpoint:
            raise ValueError("--sweep-id 사용 시 --checkpoint 인자를 제공해야 합니다.")
        start_agent(
            sweep_id=args.sweep_id,
            checkpoint_path=args.checkpoint,
            exp_name=args.exp_name,
            extra_overrides=args.overrides,
            project=args.project,
            entity=args.entity,
            count=args.count,
        )
        return

    if args.checkpoint:
        wandb.init(config=dict(DEFAULT_POSTPROCESS_PARAMS), mode="disabled")
        params = dict(DEFAULT_POSTPROCESS_PARAMS)
        metrics = evaluate_postprocess_run(args.checkpoint, args.exp_name, args.overrides, params)
        result = SweepResult(params=params, metrics=metrics)
        print("단일 평가 실행 결과:")
        print(format_result(result))
        return

    print("Sweep 설정 미리보기:")
    print_sweep_config(args.config_path)


if __name__ == "__main__":
    main()

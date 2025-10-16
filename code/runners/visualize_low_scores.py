import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import cv2
import numpy as np
import torch
from hydra import compose, initialize
from omegaconf import OmegaConf
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ocr.lightning_modules import get_pl_modules_by_cfg  # noqa: E402
from ocr.metrics import CLEvalMetric  # noqa: E402


CONFIG_DIR = os.environ.get("OP_CONFIG_DIR") or "../configs"


@dataclass
class SampleResult:
    filename: str
    hmean: float
    recall: float
    precision: float
    pred_polygons: List[List[List[int]]]
    gt_polygons: Optional[Sequence[np.ndarray]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "지정한 체크포인트로 추론을 수행하고 CLEval 점수가 낮은 이미지를 "
            "GT/예측 바운딩박스와 함께 시각화합니다."
        )
    )
    parser.add_argument(
        "--checkpoint-path",
        required=True,
        help="사용할 Lightning 체크포인트(.ckpt) 경로",
    )
    parser.add_argument(
        "--config-path",
        default=CONFIG_DIR,
        help="Hydra 설정 파일이 위치한 경로 (기본값: ../configs)",
    )
    parser.add_argument(
        "--config-name",
        default="test",
        help="불러올 Hydra 설정 이름 (기본값: test)",
    )
    parser.add_argument(
        "--config-yaml",
        default=None,
        help="Hydra 실행 결과로 저장된 통합 config.yaml 경로 (설정 시 config-path/name 대신 사용)",
    )
    parser.add_argument(
        "--split",
        choices=("val", "test"),
        default="test",
        help="평가에 사용할 데이터 split",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/low_score_visuals",
        help="시각화 결과를 저장할 출력 디렉터리",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=None,
        help="CLEval h-mean 이하의 샘플만 저장 (설정하지 않으면 상위 N개 선택)",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=50,
        help="저장할 최대 이미지 수 (threshold 미지정 시 적용)",
    )
    parser.add_argument(
        "--overrides",
        nargs="*",
        default=[],
        help="Hydra override 문자열 목록 (예: datasets.test_dataset.batch_size=4)",
    )
    return parser.parse_args()


def load_config(args: argparse.Namespace):
    if args.config_yaml:
        cfg = OmegaConf.load(args.config_yaml)
        OmegaConf.set_struct(cfg, False)
        cfg.checkpoint_path = args.checkpoint_path
        return cfg

    config_path = Path(args.config_path)
    if config_path.is_absolute():
        hydra_config_path = os.path.relpath(
            config_path,
            start=Path(__file__).resolve().parent,
        )
    else:
        hydra_config_path = config_path.as_posix()

    with initialize(version_base="1.2", config_path=hydra_config_path, job_name="visualizer"):
        cfg = compose(config_name=args.config_name, overrides=args.overrides)

    OmegaConf.set_struct(cfg, False)
    cfg.checkpoint_path = args.checkpoint_path
    return cfg


def get_image_root(cfg, split: str) -> Path:
    dataset_cfg = getattr(cfg.datasets, f"{split}_dataset")
    image_root = Path(dataset_cfg.image_path)
    if image_root.is_absolute():
        return image_root

    relative_path = image_root
    search_roots = [
        PROJECT_ROOT,
        PROJECT_ROOT.parent,
        PROJECT_ROOT.parents[1],
    ]
    for base in search_roots:
        candidate = (base / relative_path).resolve()
        if candidate.exists():
            return candidate

    return (PROJECT_ROOT / relative_path).resolve()


def ensure_output_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def move_batch_to_device(batch: Dict, device: torch.device) -> Dict:
    batch_on_device = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            batch_on_device[key] = value.to(device)
        else:
            batch_on_device[key] = value
    return batch_on_device


def to_numpy_polygons(polygons: Optional[Sequence[np.ndarray]]) -> List[np.ndarray]:
    if polygons is None:
        return []
    return [np.array(poly).squeeze(0) for poly in polygons if poly is not None]


def flatten_polygon(polygon: Sequence[Sequence[int]]) -> List[int]:
    flat: List[int] = []
    for x, y in polygon:
        flat.extend((int(x), int(y)))
    return flat


def evaluate_sample(
    pred_polygons: List[List[List[int]]],
    gt_polygons: Optional[Sequence[np.ndarray]],
) -> Dict[str, float]:
    det_quads = [flatten_polygon(poly) for poly in pred_polygons]
    gt_quads = []
    if gt_polygons is not None:
        for poly in gt_polygons:
            flat = poly.reshape(-1).tolist()
            if flat:
                gt_quads.append(flat)

    if not gt_quads and not det_quads:
        return {"det_h": 1.0, "det_r": 1.0, "det_p": 1.0}

    metric = CLEvalMetric()
    metric(det_quads, gt_quads)
    cleval = metric.compute()

    return {
        "det_h": float(cleval["det_h"]),
        "det_r": float(cleval["det_r"]),
        "det_p": float(cleval["det_p"]),
    }


def draw_polygons(image: np.ndarray, polygons: Iterable[Iterable[Sequence[int]]], color, label: str):
    for idx, polygon in enumerate(polygons):
        pts = np.array(polygon, dtype=np.int32)
        if pts.ndim != 2 or pts.shape[0] < 3:
            continue
        cv2.polylines(image, [pts], isClosed=True, color=color, thickness=2)
        centroid = pts.mean(axis=0).astype(int)
        cv2.putText(
            image,
            f"{label}{idx+1}",
            (int(centroid[0]), int(centroid[1])),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )


def visualize_sample(
    image_path: Path,
    pred_polygons: List[List[List[int]]],
    gt_polygons: List[np.ndarray],
    metrics: Dict[str, float],
    output_path: Path,
):
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"[경고] 이미지를 열 수 없습니다: {image_path}")
        return

    draw_polygons(image, gt_polygons, color=(0, 255, 0), label="GT")
    draw_polygons(image, pred_polygons, color=(0, 0, 255), label="PT")

    header = (
        f"hmean: {metrics['det_h']:.3f} | "
        f"precision: {metrics['det_p']:.3f} | recall: {metrics['det_r']:.3f}"
    )
    cv2.rectangle(image, (5, 5), (5 + 420, 30), (0, 0, 0), thickness=-1)
    cv2.putText(
        image,
        header,
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)


def run_inference(args: argparse.Namespace):
    cfg = load_config(args)
    model_module, data_module = get_pl_modules_by_cfg(cfg)

    checkpoint = torch.load(cfg.checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    try:
        model_module.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        print(f"[경고] strict=True 로 state_dict 로드 실패: {exc}. strict=False로 재시도합니다.")
        model_module.load_state_dict(state_dict, strict=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_module = model_module.to(device)
    model_module.eval()

    if args.split == "test":
        dataloader = data_module.test_dataloader()
        gt_dataset = data_module.dataset["test"]
    else:
        dataloader = data_module.val_dataloader()
        gt_dataset = data_module.dataset["val"]

    image_root = get_image_root(cfg, args.split)
    output_dir = ensure_output_dir(Path(args.output_dir))

    results: List[SampleResult] = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="추론 중"):
            batch_on_device = move_batch_to_device(batch, device)
            preds = model_module.model(
                images=batch_on_device["images"],
                return_loss=False,
            )
            boxes_batch, _ = model_module.model.get_polygons_from_maps(batch, preds)

            for idx, filename in enumerate(batch["image_filename"]):
                pred_polygons = boxes_batch[idx]
                gt_polygons = gt_dataset.anns.get(filename)
                metrics = evaluate_sample(pred_polygons, gt_polygons)

                results.append(
                    SampleResult(
                        filename=filename,
                        hmean=metrics["det_h"],
                        recall=metrics["det_r"],
                        precision=metrics["det_p"],
                        pred_polygons=pred_polygons,
                        gt_polygons=gt_polygons,
                    )
                )

    if not results:
        print("평가할 샘플이 없습니다.")
        return

    results.sort(key=lambda item: item.hmean)

    if args.score_threshold is not None:
        filtered = [res for res in results if res.hmean <= args.score_threshold]
    else:
        filtered = results[: args.max_images]

    if not filtered:
        print("조건을 만족하는 저성능 샘플이 없습니다.")
        return

    print(f"총 {len(filtered)}개의 이미지를 {output_dir}에 저장합니다.")

    for sample in filtered:
        target_path = output_dir / sample.filename
        image_path = image_root / sample.filename
        if not image_path.exists():
            print(f"[경고] 이미지 파일이 존재하지 않습니다: {image_path}")
            continue
        gt_polygons = to_numpy_polygons(sample.gt_polygons)
        metrics = {
            "det_h": sample.hmean,
            "det_r": sample.recall,
            "det_p": sample.precision,
        }
        visualize_sample(image_path, sample.pred_polygons, gt_polygons, metrics, target_path)


def main():
    args = parse_args()
    run_inference(args)


if __name__ == "__main__":
    main()

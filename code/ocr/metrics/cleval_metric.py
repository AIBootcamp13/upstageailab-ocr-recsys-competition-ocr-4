'''
*****************************************************************************************
* 참고 논문:
* CLEval: Character-Level Evaluation for Text Detection and Recognition Tasks
* https://arxiv.org/pdf/2006.06244.pdf
*
* 출처 Repository:
* https://github.com/clovaai/CLEval/tree/master/cleval
*****************************************************************************************
'''

import torch
from torchmetrics import Metric

from .box_types import POLY
from .data import SampleResult
from .eval_functions import evaluation


class Options:
    def __init__(
        self,
        case_sensitive,
        recall_gran_penalty,
        precision_gran_penalty,
        vertical_aspect_ratio_thresh,
        ap_constraint,
    ):
        self.DUMP_SAMPLE_RESULT = False
        self.ORIENTATION = False
        self.CASE_SENSITIVE = case_sensitive
        self.RECALL_GRANULARITY_PENALTY_WEIGHT = recall_gran_penalty
        self.PRECISION_GRANULARITY_PENALTY_WEIGHT = precision_gran_penalty
        self.VERTICAL_ASPECT_RATIO_THRESH = vertical_aspect_ratio_thresh
        self.AREA_PRECISION_CONSTRAINT = ap_constraint


class CLEvalMetric(Metric):
    full_state_update: bool = False

    def __init__(
        self,
        dist_sync_on_step=False,
        case_sensitive=True,
        recall_gran_penalty=1.0,
        precision_gran_penalty=1.0,
        vertical_aspect_ratio_thresh=0.5,
        ap_constraint=0.3,
        scale_wise=False,
        scale_bins=(0.0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.1, 0.5, 1.0),
        scale_range=(0.0, 1.0),
        max_polygons=500,
    ):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.options = Options(
            case_sensitive,
            recall_gran_penalty,
            precision_gran_penalty,
            vertical_aspect_ratio_thresh,
            ap_constraint,
        )
        self.scale_range = scale_range
        self.max_polygons = max_polygons

        self.scalewise_metric = {}
        if scale_wise:
            bin_ranges = [scale_bins[i:i + 2] for i in range(len(scale_bins) - 1)]
            for bin_range in bin_ranges:
                self.scalewise_metric[bin_range] = CLEvalMetric(
                    dist_sync_on_step=dist_sync_on_step,
                    case_sensitive=case_sensitive,
                    recall_gran_penalty=recall_gran_penalty,
                    precision_gran_penalty=precision_gran_penalty,
                    vertical_aspect_ratio_thresh=vertical_aspect_ratio_thresh,
                    ap_constraint=ap_constraint,
                    scale_wise=False,
                    scale_range=bin_range,
                )

        # Detection
        self.add_state("det_num_char_gt", torch.tensor(0, dtype=torch.int32), dist_reduce_fx="sum")
        self.add_state("det_num_char_det", torch.tensor(0, dtype=torch.int32), dist_reduce_fx="sum")
        self.add_state(
            "det_gran_score_recall",
            torch.tensor(0, dtype=torch.float32),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "det_num_char_tp_recall",
            torch.tensor(0, dtype=torch.int32),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "det_gran_score_precision",
            torch.tensor(0, dtype=torch.float32),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "det_num_char_tp_precision",
            torch.tensor(0, dtype=torch.int32),
            dist_reduce_fx="sum",
        )

        self.add_state("det_num_char_fp", torch.tensor(0, dtype=torch.int32), dist_reduce_fx="sum")

        # split-merge cases
        self.add_state("num_splitted", torch.tensor(0, dtype=torch.int32), dist_reduce_fx="sum")
        self.add_state("num_merged", torch.tensor(0, dtype=torch.int32), dist_reduce_fx="sum")
        self.add_state(
            "num_char_overlapped",
            torch.tensor(0, dtype=torch.int32),
            dist_reduce_fx="sum",
        )

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        for key, metric in self.scalewise_metric.items():
            self.scalewise_metric[key] = metric.to(*args, **kwargs)
        return self

    def update(
        self,
        det_quads,
        gt_quads,
        det_letters=None,
        gt_letters=None,
        gt_is_dcs=None,
    ):
        """
        Args:
            det_quads (NDArray[float32]): (N, M) detected POLYGONS
            gt_quads (NDArray[float32]): (N, M) target POLYGONS
            det_letters (List[str]): detected letters
            gt_letters (List[str]): target letters
            gt_is_dcs (List[bool]): is dc gt quad?
        """
        gt_inps = self.__make_eval_input(gt_quads, gt_letters, gt_is_dcs)
        det_inps = self.__make_eval_input(det_quads[:self.max_polygons], det_letters)
        sample_res = evaluation(self.options, gt_inps, det_inps, scale_range=self.scale_range)
        self.__accumulate(sample_res)

        for metric in self.scalewise_metric.values():
            metric(
                det_quads,
                gt_quads,
                det_letters,
                gt_letters,
                gt_is_dcs,
            )

    def __make_eval_input(self, quads, letters, is_dcs=None):
        eval_inps = []

        for i in range(len(quads)):
            if len(quads[i]) < 4 * 2 or len(quads[i]) % 2 != 0:
                print(f" {__file__}: [WARN] Skip invalid polygon length: {len(quads[i])}")
                continue
            eval_inp = POLY(
                quads[i],
                confidence=0.0,
                transcription=None if letters is None else letters[i],
                is_dc=None if is_dcs is None else is_dcs[i],
            )
            eval_inps.append(eval_inp)
        return eval_inps

    def __accumulate(self, sample_res: SampleResult):
        self.num_splitted += sample_res.stats.num_splitted
        self.num_merged += sample_res.stats.num_merged
        self.num_char_overlapped += sample_res.stats.num_char_overlapped

        self.det_num_char_gt += sample_res.stats.det.num_char_gt
        self.det_num_char_det += sample_res.stats.det.num_char_det
        self.det_gran_score_recall += sample_res.stats.det.gran_score_recall
        self.det_num_char_tp_recall += sample_res.stats.det.num_char_tp_recall
        self.det_gran_score_precision += sample_res.stats.det.gran_score_precision
        self.det_num_char_tp_precision += sample_res.stats.det.num_char_tp_precision
        self.det_num_char_fp += sample_res.stats.det.num_char_fp

    def compute(self):
        det_r, det_p, det_h = self.__calculate_rph(
            self.det_num_char_gt,
            self.det_num_char_det,
            self.det_gran_score_recall,
            self.det_num_char_tp_recall,
            self.det_gran_score_precision,
            self.det_num_char_tp_precision,
        )
        return_dict = {
            "det_r": det_r,
            "det_p": det_p,
            "det_h": det_h,
            "num_splitted": self.num_splitted,
            "num_merged": self.num_merged,
            "num_char_overlapped": self.num_char_overlapped,
            "scale_wise": {},
        }

        for scale_bin, metric in self.scalewise_metric.items():
            return_dict["scale_wise"][scale_bin] = metric.compute()

        return return_dict

    def reset(self):
        super().reset()
        for metric in self.scalewise_metric.values():
            metric.reset()

    def __calculate_rph(
        self,
        num_char_gt,
        num_char_det,
        gran_score_recall,
        num_char_tp_recall,
        gran_score_precision,
        num_char_tp_precision,
    ):
        total_gt = num_char_gt
        total_det = num_char_det
        gran_gt = gran_score_recall
        tp_gt = num_char_tp_recall
        gran_det = gran_score_precision
        tp_det = num_char_tp_precision

        # Sample Score : Character correct length - Granularity Penalty
        recall = torch.tensor(0.0, dtype=torch.float32, device=self.device) if total_gt == 0 else (
                max(0.0, tp_gt - gran_gt) / total_gt)
        precision = torch.tensor(0.0, dtype=torch.float32, device=self.device) if total_det == 0 \
            else max(0.0, tp_det - gran_det) / total_det
        hmean = self.harmonic_mean(recall, precision)
        return recall, precision, hmean

    def harmonic_mean(self, score1, score2):
        """get harmonic mean value"""
        if score1 + score2 == 0:
            return torch.tensor(0, dtype=torch.float32, device=self.device)
        else:
            return (2 * score1 * score2) / (score1 + score2)

import numpy as np
import json
from datetime import datetime
from pathlib import Path
import lightning.pytorch as pl
from tqdm import tqdm
from collections import defaultdict
from collections import OrderedDict
from torch.utils.data import DataLoader
from hydra.utils import instantiate
from ocr.metrics import CLEvalMetric


class OCRPLModule(pl.LightningModule):
    def __init__(self, model, dataset, config):
        super(OCRPLModule, self).__init__()
        self.model = model
        self.dataset = dataset
        self.metric = CLEvalMetric()
        self.config = config

        self.validation_step_outputs = OrderedDict()
        self.test_step_outputs = OrderedDict()
        self.predict_step_outputs = OrderedDict()

    def forward(self, x):
        return self.model(return_loss=False, **x)

    def training_step(self, batch, batch_idx):
        pred = self.model(**batch)
        self.log('train/loss', pred['loss'], batch_size=len(batch))
        for key, value in pred['loss_dict'].items():
            self.log(f'train/{key}', value, batch_size=len(batch))
        return pred

    def validation_step(self, batch, batch_idx):
        pred = self.model(**batch)
        self.log('val/loss', pred['loss'], batch_size=len(batch))
        for key, value in pred['loss_dict'].items():
            self.log(f'val/{key}', value, batch_size=len(batch))

        boxes_batch, _ = self.model.get_polygons_from_maps(batch, pred)
        for idx, boxes in enumerate(boxes_batch):
            self.validation_step_outputs[batch['image_filename'][idx]] = boxes
        return pred

    def on_validation_epoch_end(self):
        cleval_metrics = defaultdict(list)

        for gt_filename, gt_words in tqdm(self.dataset['val'].anns.items(), desc="Evaluation"):
            if gt_filename not in self.validation_step_outputs:
                # TODO: Check if this is on_sanity?
                cleval_metrics['recall'].append(np.array(0., dtype=np.float32))
                cleval_metrics['precision'].append(np.array(0., dtype=np.float32))
                cleval_metrics['hmean'].append(np.array(0., dtype=np.float32))
                continue

            pred = self.validation_step_outputs[gt_filename]
            det_quads = [[point for coord in polygons for point in coord]
                         for polygons in pred]
            gt_quads = [item.squeeze().reshape(-1) for item in gt_words]

            self.metric(det_quads, gt_quads)
            cleval = self.metric.compute()
            cleval_metrics['recall'].append(cleval['det_r'].cpu().numpy())
            cleval_metrics['precision'].append(cleval['det_p'].cpu().numpy())
            cleval_metrics['hmean'].append(cleval['det_h'].cpu().numpy())
            self.metric.reset()

        recall = np.mean(cleval_metrics['recall'])
        precision = np.mean(cleval_metrics['precision'])
        hmean = np.mean(cleval_metrics['hmean'])

        self.log('val/recall', recall, on_epoch=True, prog_bar=True)
        self.log('val/precision', precision, on_epoch=True, prog_bar=True)
        self.log('val/hmean', hmean, on_epoch=True, prog_bar=True)

        self.validation_step_outputs.clear()

    def test_step(self, batch):
        pred = self.model(return_loss=False, **batch)

        boxes_batch, _ = self.model.get_polygons_from_maps(batch, pred)
        for idx, boxes in enumerate(boxes_batch):
            self.test_step_outputs[batch['image_filename'][idx]] = boxes
        return pred

    def on_test_epoch_end(self):
        cleval_metrics = defaultdict(list)

        for gt_filename, gt_words in tqdm(self.dataset['test'].anns.items(), desc="Evaluation"):
            pred = self.test_step_outputs[gt_filename]
            det_quads = [[point for coord in polygons for point in coord]
                         for polygons in pred]
            gt_quads = [item.squeeze().reshape(-1) for item in gt_words]

            self.metric(det_quads, gt_quads)
            cleval = self.metric.compute()
            cleval_metrics['recall'].append(cleval['det_r'].cpu().numpy())
            cleval_metrics['precision'].append(cleval['det_p'].cpu().numpy())
            cleval_metrics['hmean'].append(cleval['det_h'].cpu().numpy())
            self.metric.reset()

        recall = np.mean(cleval_metrics['recall'])
        precision = np.mean(cleval_metrics['precision'])
        hmean = np.mean(cleval_metrics['hmean'])

        self.log('test/recall', recall, on_epoch=True, prog_bar=True)
        self.log('test/precision', precision, on_epoch=True, prog_bar=True)
        self.log('test/hmean', hmean, on_epoch=True, prog_bar=True)

        self.test_step_outputs.clear()

    def predict_step(self, batch):
        pred = self.model(return_loss=False, **batch)
        boxes_batch, _ = self.model.get_polygons_from_maps(batch, pred)

        for idx, boxes in enumerate(boxes_batch):
            self.predict_step_outputs[batch['image_filename'][idx]] = boxes
        return pred

    def on_predict_epoch_end(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        submission_file = Path(f"{self.config.submission_dir}") / f"{timestamp}.json"
        submission_file.parent.mkdir(parents=True, exist_ok=True)

        submission = OrderedDict(images=OrderedDict())
        for filename, pred_boxes in self.predict_step_outputs.items():
            # Separate box
            boxes = OrderedDict()
            for idx, box in enumerate(pred_boxes):
                boxes[f'{idx + 1:04}'] = OrderedDict(points=box)

            # Append box
            submission['images'][filename] = OrderedDict(words=boxes)

        # Export submission
        with submission_file.open("w") as fp:
            if self.config.minified_json:
                json.dump(submission, fp, indent=None, separators=(',', ':'))
            else:
                json.dump(submission, fp, indent=4)

        self.predict_step_outputs.clear()

    def configure_optimizers(self):
        return self.model.get_optimizers()


class OCRDataPLModule(pl.LightningDataModule):
    def __init__(self, dataset, config):
        super(OCRDataPLModule, self).__init__()
        self.dataset = dataset
        self.config = config
        self.collate_fn = instantiate(self.config.collate_fn)

    def train_dataloader(self):
        train_loader_config = self.config.dataloaders.train_dataloader
        self.collate_fn.inference_mode = False
        return DataLoader(self.dataset['train'], collate_fn=self.collate_fn, **train_loader_config)

    def val_dataloader(self):
        val_loader_config = self.config.dataloaders.val_dataloader
        self.collate_fn.inference_mode = False
        return DataLoader(self.dataset['val'], collate_fn=self.collate_fn, **val_loader_config)

    def test_dataloader(self):
        test_loader_config = self.config.dataloaders.test_dataloader
        self.collate_fn.inference_mode = False
        return DataLoader(self.dataset['test'], collate_fn=self.collate_fn, **test_loader_config)

    def predict_dataloader(self):
        predict_loader_config = self.config.dataloaders.predict_dataloader
        self.collate_fn.inference_mode = True
        return DataLoader(self.dataset['predict'], collate_fn=self.collate_fn,
                          **predict_loader_config)

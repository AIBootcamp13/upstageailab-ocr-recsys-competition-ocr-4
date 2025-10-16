from .base import OCRDataset
from .db_collate_fn import DBCollateFN
from .transforms import DBTransforms
from hydra.utils import instantiate


def get_datasets_by_cfg(config):
    train_dataset = instantiate(config.train_dataset)
    val_dataset = instantiate(config.val_dataset)
    test_dataset = instantiate(config.test_dataset)
    predict_dataset = instantiate(config.predict_dataset)
    return {
        'train': train_dataset,
        'val': val_dataset,
        'test': test_dataset,
        'predict': predict_dataset
    }

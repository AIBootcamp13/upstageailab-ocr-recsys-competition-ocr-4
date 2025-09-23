from ocr.models import get_model_by_cfg
from ocr.datasets import get_datasets_by_cfg
from .ocr_pl import OCRPLModule, OCRDataPLModule


def get_pl_modules_by_cfg(config):
    model = get_model_by_cfg(config.models)
    dataset = get_datasets_by_cfg(config.datasets)
    modules = OCRPLModule(model=model, dataset=dataset, config=config)
    data_modules = OCRDataPLModule(dataset=dataset, config=config)
    return modules, data_modules

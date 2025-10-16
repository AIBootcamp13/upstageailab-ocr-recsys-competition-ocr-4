from .architecture import OCRModel


def get_model_by_cfg(config):
    model = OCRModel(config)
    return model

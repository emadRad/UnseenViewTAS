from yacs.config import CfgNode

import torch
from uvtas.model.ltcontext import LTC


def build_model(cfg: CfgNode) -> torch.nn.Module:
    """
    Builds the action segmentation model (LTContext with predictor heads).

    :param cfg: configs that contains the parameters to build the model.
                Details can be seen in uvtas/config/defaults.py.
    :return:
        model
    """
    assert (
        cfg.NUM_GPUS <= torch.cuda.device_count()
    ), "Cannot use more GPU devices than available"

    return LTC(cfg.MODEL)

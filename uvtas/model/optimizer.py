"""Optimizer and learning rate schedule."""

from yacs.config import CfgNode
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler


def construct_optimizer(model: torch.nn.Module, cfg: CfgNode) -> Optimizer:
    """
    Construct the Adam optimizer.

    :param model: model to optimize
    :param cfg: configs with the learning rate and weight decay
    :return: optimizer
    """
    if hasattr(model, 'predictor') and isinstance(model.predictor, nn.Module):
        predictor_params = model.predictor.parameters()
        other_params = [param for name, param in model.named_parameters() if 'predictor' not in name]
        param_groups = [
            {'params': predictor_params, 'lr': cfg.SOLVER.BASE_LR},
            {'params': other_params, 'lr':  cfg.SOLVER.BASE_LR}
        ]
    else:
        param_groups = filter(lambda p: p.requires_grad, model.parameters())

    return torch.optim.Adam(
        param_groups,
        lr=cfg.SOLVER.BASE_LR,
        betas=(0.9, 0.999),
        weight_decay=cfg.SOLVER.WEIGHT_DECAY,
    )


def construct_lr_scheduler(cfg: CfgNode, optimizer: Optimizer) -> LRScheduler:
    """
    Linear warm-up followed by cosine decay (per epoch).

    :param cfg: configs of hyper-parameters
    :param optimizer:
    :return: lr scheduler
    """
    main_lr_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=cfg.SOLVER.T_MAX - cfg.SOLVER.WARMUP_EPOCHS,
        eta_min=cfg.SOLVER.ETA_MIN,
    )
    warmup_lr_scheduler = LinearLR(
        optimizer, start_factor=cfg.SOLVER.WARMUP_FACTOR, total_iters=cfg.SOLVER.WARMUP_EPOCHS
    )
    return SequentialLR(
        optimizer,
        schedulers=[warmup_lr_scheduler, main_lr_scheduler],
        milestones=[cfg.SOLVER.WARMUP_EPOCHS]
    )


def get_epoch_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def get_grad_norm_(parameters, norm_type=2.0):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.0)
    device = parameters[0].grad.device
    total_norm = torch.norm(
        torch.stack(
            [
                torch.norm(p.grad.detach(), norm_type).to(device)
                for p in parameters
            ]
        ),
        norm_type,
    )
    return total_norm

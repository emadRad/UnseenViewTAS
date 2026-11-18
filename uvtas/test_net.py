import os
from os.path import join, splitext
from tqdm import tqdm
from yacs.config import CfgNode

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
import pandas as pd


import uvtas.utils.checkpoint as cu
import uvtas.utils.misc as misc
from uvtas.dataset import loader
from uvtas.model import model_builder
from uvtas.utils.metrics import calculate_metrics

import uvtas.utils.logging as logging

logger = logging.get_logger(__name__)


@torch.no_grad()
def eval_model(
        val_loader: DataLoader,
        model: nn.Module,
        device: torch.device.type,
        cfg: CfgNode):
    """
    Evaluate the model on the val set.
    :param val_loader: data loader to provide validation data.
    :param model: model to evaluate the performance.
    :param device: device to use (cuda or cpu)
    :param cfg:
    :return:
    """
    # Evaluation mode enabled. The running stats would not be updated.
    model.eval()
    logger.info(f"Testing the trained model.")

    cfg_filename = splitext(cfg.CONFIG_FILE.split("/")[-1])[0]
    save_path = join(cfg.TEST.SAVE_RESULT_PATH,
                     cfg.TRAIN.DATASET,
                     cfg_filename,
                     cfg.TRAIN.EVAL_SPLIT,
                     )
    logger.info(save_path)
    os.makedirs(save_path, exist_ok=True)
    unseen_views = set(cfg.DATA.UNSEEN_EGO_VIEWS + cfg.DATA.UNSEEN_EXO_VIEWS)
    test_metrics = {"video_name": [],
                    "view_name": [],
                    "is_seen": [],
                    "MoF": [],
                    "Edit": [],
                    "F1@10": [],
                    "F1@25": [],
                    "F1@50": [],
                    }
    ignored_class_idx = cfg.TEST.IGNORED_CLASSES + [cfg.MODEL.PAD_IGNORE_IDX]
    logger.info(f"Ignored class idxs: {ignored_class_idx}")

    for batch_dict in tqdm(val_loader, total=len(val_loader)):
        batch_dict = batch_dict['Input']
        mb_size = batch_dict["targets"].shape[0]
        assert mb_size == 1, "Validation batch size should be one."

        misc.move_to_device(batch_dict, device)
        framewise_logits, _, _ = model(batch_dict['features'], batch_dict['masks'])

        prediction = misc.prepare_prediction(framewise_logits)

        target = batch_dict['targets'].cpu()
        prediction = prediction.cpu()
        video_name = batch_dict['video_name'][0]
        view_name = batch_dict['view_name'][0]

        video_metrics = calculate_metrics(target,
                                          prediction,
                                          ignored_class_idx)

        if view_name in unseen_views:
            test_metrics['is_seen'].append(False)
        else:
            test_metrics['is_seen'].append(True)

        test_metrics['video_name'].append(video_name)
        test_metrics['view_name'].append(view_name)

        for name, score in video_metrics.items():
            if name != 'Edit':
                score = score * 100
            test_metrics[name].append(score)

        if cfg.TEST.SAVE_PREDICTIONS:
            base_path = join(save_path, video_name)
            os.makedirs(base_path, exist_ok=True)
            np.save(join(base_path, "pred.npy"), prediction[0].long().numpy())
            np.save(join(base_path, "gt.npy"), target[0].long().numpy())

    test_res_df = pd.DataFrame(test_metrics)
    test_res_df.round(5).to_csv(join(save_path, "testing_metrics.csv"))

    # mean metrics of the seen views, the unseen views and, if both exist, the unseen exo and ego views
    metric_names = ['F1@10', 'F1@25', 'F1@50', 'Edit', 'MoF']
    groups = {"seen": test_res_df['is_seen'],
              "unseen": ~test_res_df['is_seen']}
    if len(cfg.DATA.UNSEEN_EXO_VIEWS) > 0 and len(cfg.DATA.UNSEEN_EGO_VIEWS) > 0:
        groups["unseen_exo"] = test_res_df['view_name'].isin(cfg.DATA.UNSEEN_EXO_VIEWS)
        groups["unseen_ego"] = test_res_df['view_name'].isin(cfg.DATA.UNSEEN_EGO_VIEWS)
    summary = {}
    for group, cond in groups.items():
        if cond.sum() == 0:
            continue
        mean_metrics = test_res_df.loc[cond, metric_names].mean()
        mean_metrics['F1@Avg'] = mean_metrics[['F1@10', 'F1@25', 'F1@50']].mean()
        mean_metrics['num_videos'] = int(cond.sum())
        summary[group] = mean_metrics
        logger.info(f"Testing metrics ({group}):")
        logging.log_json_stats(mean_metrics.to_dict(), precision=1)
    pd.DataFrame(summary).T.round(2).to_csv(join(save_path, "summary.csv"))
    logger.info(f"Metrics saved to {save_path}")


def test(cfg: CfgNode):
    """
    Evaluate a trained model on TRAIN.EVAL_SPLIT and save the per-video metrics (seen and unseen views)
    :param cfg: config file. Details can be found in uvtas/config/defaults.py
    :return:
    """
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)
    logging.setup_logging(cfg.OUTPUT_DIR, cfg.EXPR_NUM)

    model = model_builder.build_model(cfg)
    logger.info(f"Number of params: {misc.params_to_string(misc.params_count(model))}")

    if cfg.NUM_GPUS > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Transfer the model to device(s)
    model = model.to(device)

    checkpoint_path = cfg.TEST.CHECKPOINT_PATH
    cu.load_model(
        checkpoint_path,
        model=model,
        num_gpus=cfg.NUM_GPUS,
    )
    test_loader = loader.construct_loader(cfg, cfg.TRAIN.EVAL_SPLIT)
    eval_model(test_loader, model, device, cfg)

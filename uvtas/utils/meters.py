"""Meters."""

from typing import List
from collections import deque

import torch
import numpy as np
import pandas as pd
from yacs.config import CfgNode
from torch.utils.tensorboard import SummaryWriter

from uvtas.utils.metrics import calculate_metrics
import uvtas.utils.misc as misc
import uvtas.utils.plot_utils as plot_utils
from uvtas.dataset.assembly101 import VIEW_NAMES

import uvtas.utils.logging as logging
logger = logging.get_logger(__name__)


class ScalarMeter(object):
    """
    A scalar meter uses a deque to track a series of scaler values with a given
    window size. It supports calculating the median and average values of the
    window, and also supports calculating the global average.
    """

    def __init__(self, window_size: int):
        """

        :param window_size:
        """
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0

    def reset(self):
        """
        Reset the deque.
        """
        self.deque.clear()
        self.total = 0.0
        self.count = 0

    def add_value(self, value: float):
        """
        Add a new scalar value to the deque.
        """
        self.deque.append(value)
        self.count += 1
        self.total += value

    def get_win_median(self):
        """
        Calculate the current median value of the deque.
        """
        return np.median(self.deque)

    def get_win_avg(self):
        """
        Calculate the current average value of the deque.
        """
        return np.mean(self.deque)

    def get_global_avg(self):
        """
        Calculate the global mean value.
        """
        return self.total / self.count


class TrainMeter(object):
    """
    Measure training stats.
    """

    def __init__(self,
                 epoch_iters: int,
                 cfg: CfgNode,
                 global_step: int,
                 writer: SummaryWriter = None):
        """

        :param epoch_iters: the overall number of iterations of one epoch
        :param cfg:
        :param global_step:
        :param writer:
        """
        self._cfg = cfg
        self.writer = writer
        self.epoch_iters = epoch_iters

        self.loss_dict = {
            "loss": ScalarMeter(cfg.LOG_PERIOD),
            "loss_ce": ScalarMeter(cfg.LOG_PERIOD),
            "loss_mse": ScalarMeter(cfg.LOG_PERIOD)
        }
        self._grad_norms = []

        self._iter_log_window = cfg.LOG_PERIOD
        self._ignore_idx = cfg.MODEL.PAD_IGNORE_IDX
        if isinstance(cfg.MODEL.PAD_IGNORE_IDX, int):
            self._ignore_idx = [cfg.MODEL.PAD_IGNORE_IDX]
        self._ignore_idx = self._ignore_idx + cfg.TRAIN.EVAL_IGNORE_LABELS
        logger.info(f"Ignored label idxs: {self._ignore_idx}")

        self.metrics_dict = {
            "MoF": [],
            "Edit": [],
            "F1@10": [],
            "F1@25": [],
            "F1@50": [],
        }
        self.num_samples = 0
        self.lr = None
        self.global_iter = global_step

    def reset(self):
        """
        Reset the Meter.
        """
        for loss in self.loss_dict.values():
            loss.reset()
        # for name in self.loss_sums.keys():
        #     self.loss_sums[name] = 0.0
        self.metrics_dict = {name: [] for name in self.metrics_dict.keys()}
        self.lr = None
        self.num_samples = 0

    def update_stats(self,
                     target: torch.Tensor,
                     prediction: torch.Tensor,
                     loss_dict: dict,
                     grad_norm: float,
                     lr: float,
                     ):
        """

        :param target:
        :param prediction:
        :param loss_dict:
        :param grad_norm:
        :param lr:
        :return:
        """
        mb_size = target.shape[0]
        target = target.detach().cpu()
        prediction = prediction.detach().cpu()
        video_metrics = calculate_metrics(target, prediction, self._ignore_idx)
        for name, score in video_metrics.items():
            self.metrics_dict[name].append(score)

        if self.writer:
            for name, metric_val in video_metrics.items():
                self.writer.add_scalar(f"4-Train_Metric/{name}", metric_val, global_step=self.global_iter)

            self.writer.add_scalar("4-Grad_Norm", grad_norm, global_step=self.global_iter)

        self._grad_norms.append(grad_norm)

        for name, loss in loss_dict.items():
            loss = loss.item()
            if name not in self.loss_dict:
                self.loss_dict[name] = ScalarMeter(self._cfg.LOG_PERIOD)
                # self.loss_sums[name] = 0.0

            self.loss_dict[name].add_value(loss)
            # self.loss_sums[name] += loss * mb_size
            if self.writer:
                self.writer.add_scalar(f"2-Loss/train/{name}",
                                       loss,
                                       global_step=self.global_iter)
        self.lr = lr
        self.num_samples += mb_size
        self.global_iter += 1

    def log_iter_stats(self, cur_epoch, cur_iter):
        """
        Log the stats of the current iteration.

        :param cur_epoch: the number of current epoch.
        :param cur_iter: the number of current iteration.
        :return:
        """
        if (cur_iter + 1) % self._cfg.LOG_PERIOD != 0:
            return
        mem_usage = misc.gpu_mem_usage()
        stats = {
            "_type": "train_iter",
            "epoch": "{}/{}".format(cur_epoch + 1, self._cfg.SOLVER.MAX_EPOCH),
            "iter": "{}/{}".format(cur_iter + 1, self.epoch_iters),
            # "max_grad_norm": max(self._grad_norms),
            # "avg_grad_norm": np.mean(np.array(self._grad_norms)),
            "lr": f"{self.lr:.6f}",
            "mem": int(np.ceil(mem_usage)),
        }

        for name, metric_list in self.metrics_dict.items():
            metric_val = np.median(metric_list[-self._iter_log_window:])
            stats[name] = f"{metric_val:.4f}"
        for name, loss in self.loss_dict.items():
            stats[name] = f"{loss.get_win_median():.5f}"

        logging.log_json_stats(stats)

    def log_epoch_stats(self, cur_epoch):
        """
        Log the end of epochs stats.

        :param cur_epoch:
        :return:
        """
        mem_usage = misc.gpu_mem_usage()
        avg_loss = {}

        # for name, loss in self.loss_sums.items():
        #     avg_loss[name] = np.round(loss / self.num_samples, decimals=4)

        for name, loss in self.loss_dict.items():
            avg_loss[name] = loss.get_global_avg()

        stats = {
            "_type": "train_epoch",
            "epoch": "{}/{}".format(cur_epoch + 1, self._cfg.SOLVER.MAX_EPOCH),
            "loss": avg_loss,
            # "max_grad_norm": max(self._grad_norms),
            # "avg_grad_norm": np.mean(np.array(self._grad_norms)),
            "lr": self.lr,
            "mem": int(np.ceil(mem_usage)),
        }
        stats.update(avg_loss)
        for name, metric_list in self.metrics_dict.items():
            metric_mean = np.mean(metric_list)
            stats[name] = metric_mean
            if self.writer:
                self.writer.add_scalar(f"3-Avg_Train_Metric/{name}",
                                       metric_mean,
                                       global_step=cur_epoch)
                
        if self.writer:
            for name, loss_val in avg_loss.items():
                self.writer.add_scalar(f"2-Loss/train/avg_{name}", loss_val, global_step=cur_epoch)
            self.writer.add_scalar("4-Grad_Norm/avg",
                                   np.mean(np.array(self._grad_norms)),
                                   global_step=cur_epoch)

        logging.log_json_stats(stats)
        if self.writer:
            self.writer.add_scalar("5-LR/train_epoch/lr", self.lr, cur_epoch)
        return avg_loss['loss']


class ValMeter(object):
    """
    Measures validation stats.
    """
    def __init__(self, max_iter: int, cfg: CfgNode, writer: SummaryWriter = None):
        """

        :param max_iter:  the max number of iteration of the current epoch.
        :param cfg:
        :param writer:
        """
        self._cfg = cfg
        self.max_iter = max_iter
        self.colors = plot_utils.generate_distinct_colors(n=cfg.MODEL.NUM_CLASSES,
                                                          random_seed=115)
        self._iter_log_window = cfg.LOG_PERIOD
        self.loss_dict = {
            "loss": ScalarMeter(cfg.LOG_PERIOD),
            "loss_ce": ScalarMeter(cfg.LOG_PERIOD),
            "loss_mse": ScalarMeter(cfg.LOG_PERIOD)
        }

        self._ignore_idx = cfg.MODEL.PAD_IGNORE_IDX
        if isinstance(cfg.MODEL.PAD_IGNORE_IDX, int):
            self._ignore_idx = [cfg.MODEL.PAD_IGNORE_IDX]
        self._ignore_idx = self._ignore_idx + cfg.TRAIN.EVAL_IGNORE_LABELS
        logger.info(f"Ignored label idxs: {self._ignore_idx}")
        self.writer = writer
        self.metrics_dict = {
            "video_name": [],
            "view_name": [],
            "is_seen": [],
            "MoF": [],
            "Edit": [],
            "F1@10": [],
            "F1@25": [],
            "F1@50": [],
        }
        self._unseen_views = set(cfg.DATA.UNSEEN_EGO_VIEWS + cfg.DATA.UNSEEN_EXO_VIEWS)
        self._exo_unseen_views = cfg.DATA.UNSEEN_EXO_VIEWS
        self._ego_unseen_views = cfg.DATA.UNSEEN_EGO_VIEWS
        self.num_samples = 0

    def reset(self):
        """
        Reset the Meter.
        """
        self.metrics_dict = {name: [] for name in self.metrics_dict.keys()}
        for loss in self.loss_dict.values():
            loss.reset()
        self.num_samples = 0

    def update_stats(self,
                     target: torch.Tensor,
                     prediction: torch.Tensor,
                     video_name: str,
                     view_name: str,
                     loss_dict: dict
                     ):
        video_metrics = calculate_metrics(target,
                                          prediction,
                                          self._ignore_idx)
        mb_size = target.shape[0]
        for name, score in video_metrics.items():
            self.metrics_dict[name].append(score)
        self.metrics_dict['video_name'].append(video_name)
        self.metrics_dict['view_name'].append(view_name)
        if view_name in self._unseen_views:
            self.metrics_dict['is_seen'].append(False)
        else:
            self.metrics_dict['is_seen'].append(True)

        # for name, loss in loss_dict.items():
        #     loss = loss.item()
        #     self.loss_sums[name] += loss * mb_size

        for name, loss in loss_dict.items():
            loss = loss.item()
            if name not in self.loss_dict:
                self.loss_dict[name] = ScalarMeter(self._cfg.LOG_PERIOD)

            self.loss_dict[name].add_value(loss)

        self.num_samples += mb_size
        return video_metrics

    def log_iter_stats(self, cur_epoch: int, cur_iter: int):
        """

        :param cur_epoch:
        :param cur_iter:
        :return:
        """

        if (cur_iter + 1) % self._cfg.LOG_PERIOD != 0:
            return
        mem_usage = misc.gpu_mem_usage()
        stats = {
            "_type": "val_iter",
            "epoch": "{}/{}".format(cur_epoch + 1, self._cfg.SOLVER.MAX_EPOCH),
            "iter": "{}/{}".format(cur_iter + 1, self.max_iter),
            "mem": int(np.ceil(mem_usage)),
        }
        for name, metric_list in self.metrics_dict.items():
            if name not in ['video_name', 'view_name']:
                metric_val = np.median(metric_list[-self._iter_log_window:])
                stats[name] = metric_val
        logging.log_json_stats(stats)

    def visualize_prediction_result(self, vis_data: List, cur_epoch: int):
        """
        Generate and add visualization of predictions to tensorboard.
        :param vis_data: a list of dictionaries. Each dictionary contains the target
                         and prediction array (numpy) with video name
        :param cur_epoch:
        :return:
        """
        res_list = []
        if len(vis_data) % 2 != 0:
            vis_data = vis_data[:-1]
        for data in vis_data:
            gt_lbl, gt_lens = plot_utils.summarize_list(data['target'].tolist())
            pred_lbl, pred_lens = plot_utils.summarize_list(data['pred'].tolist())

            gt_res = plot_utils.generate_image_for_segmentation(gt_lbl, gt_lens,
                                                                colors=self.colors,
                                                                height=10,
                                                                white_label=self._cfg.DATA.BACKGROUND_INDICES)
            pred_res = plot_utils.generate_image_for_segmentation(pred_lbl, pred_lens,
                                                                  colors=self.colors,
                                                                  height=10,
                                                                  white_label=self._cfg.DATA.BACKGROUND_INDICES)
            res_list.append({"gt": gt_res,
                             "pred": pred_res,
                             "video_name": data['video_name'],
                             "len": sum(gt_lens)})

        fig = plot_utils.create_result_fig(res_list)

        if self.writer:
            self.writer.add_figure("Val_Pred", fig, global_step=cur_epoch)

    def log_epoch_stats(self, cur_epoch: int):
        """
         Log the stats of the current epoch.
        :param cur_epoch: the number of current epoch.
        :return:
        """
        mem_usage = misc.gpu_mem_usage()
        avg_loss = {}
        for name, loss in self.loss_dict.items():
            avg_loss[name] = loss.get_global_avg()
        stats = {
            "_type": "val_epoch",
            "epoch": "{}/{}".format(cur_epoch + 1, self._cfg.SOLVER.MAX_EPOCH),
            "mem": int(np.ceil(mem_usage)),
        }
        for name, loss in avg_loss.items():
            stats[name] = loss

        if self.writer:
            for name, loss_val in avg_loss.items():
                self.writer.add_scalar(f"2-Val_Loss/avg_{name}", loss_val, global_step=cur_epoch)
        metrics_list = ['F1@10', 'F1@25', 'F1@50', 'Edit', 'MoF']
        per_video_metrics_df = pd.DataFrame(self.metrics_dict)
        seen_metrics = per_video_metrics_df[per_video_metrics_df['is_seen'] == True]
        unseen_metrics = per_video_metrics_df[per_video_metrics_df['is_seen'] == False]

        seen_metrics = seen_metrics[metrics_list].mean().to_dict()
        unseen_metrics = unseen_metrics[metrics_list].mean().to_dict()
        # unseen exocentric and egocentric views are reported separately in the paper
        view_groups = {}
        if len(self._exo_unseen_views) > 0 and len(self._ego_unseen_views) > 0:
            for split, views in {"unseen_exo": self._exo_unseen_views, "unseen_ego": self._ego_unseen_views}.items():
                view_groups[split] = per_video_metrics_df[
                    per_video_metrics_df['view_name'].isin(views)][metrics_list].mean().to_dict()

        for view in self._exo_unseen_views:
            view_metrics = per_video_metrics_df[per_video_metrics_df['view_name'] == view]
            mean_metrics = view_metrics[metrics_list].mean().to_dict()
            for metric_name, value in mean_metrics.items():
                if np.isnan(value):
                    value = 0.0
                if self._cfg.TRAIN.DATASET == 'assembly101':
                    view_no_rgb = view.split("_")[0]
                    tag = f"1-Val_{VIEW_NAMES[view]} ({view_no_rgb})/{metric_name}"
                else:
                    tag = f"1-Val_{view}/{metric_name}"
                if self.writer:
                    self.writer.add_scalar(tag,
                                           value,
                                           global_step=cur_epoch)

        # will be used to select best model
        metrics_sum = 0
        for split, metrics_dict in {"seen": seen_metrics,
                                    "unseen": unseen_metrics}.items():
            for name, metric_mean in metrics_dict.items():
                if np.isnan(metric_mean):
                    # no video of this group in the evaluation set
                    continue
                stats[f"{split}/{name}"] = metric_mean
                if self.writer:
                    self.writer.add_scalar(f"1-Val_Metric_{split}/{name}",
                                           metric_mean,
                                           global_step=cur_epoch)
                if name == 'Edit':
                    metric_mean = metric_mean / 100.0
                metrics_sum += metric_mean

        for split, metrics_dict in view_groups.items():
            for name, metric_mean in metrics_dict.items():
                if np.isnan(metric_mean):
                    continue
                stats[f"{split}/{name}"] = metric_mean
                if self.writer:
                    self.writer.add_scalar(f"1-Val_Metric_{split}/{name}",
                                           metric_mean,
                                           global_step=cur_epoch)

        out_metrics = {'metrics_sum': metrics_sum}
        logging.log_json_stats(stats)
        return out_metrics, per_video_metrics_df

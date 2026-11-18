
import os
import numpy as np

import torch
from yacs.config import CfgNode

import uvtas.utils.logging as logging
from uvtas.utils.plot_utils import summarize_list


logger = logging.get_logger(__name__)


def move_to_device(batch_dict, device, keys=('features', 'targets', 'masks')):
    for key in keys:
        if key in batch_dict:
            batch_dict[key] = batch_dict[key].to(device)
    return batch_dict


def params_count(model):
    """
    Compute the number of parameters.
    Args:
        model (model): model to count the number of parameters.
    """
    return np.sum([p.numel() for p in model.parameters()]).item()


def gpu_mem_usage():
    """
    Compute the GPU memory usage for the current device (MB).
    """
    mem_usage_bytes = torch.cuda.max_memory_allocated()
    return mem_usage_bytes / (1024 * 1024)


def is_eval_epoch(cfg: CfgNode, cur_epoch: int):
    """
     Determine if the model should be evaluated at the current epoch.
    :param cfg:
    :param cur_epoch:
    :return:
    """
    return (
        cur_epoch + 1
    ) % cfg.TRAIN.EVAL_PERIOD == 0 or cur_epoch + 1 == cfg.SOLVER.MAX_EPOCH


def find_latest_experiment(path):
    if not os.path.exists(path):
        return 0

    list_of_experiments = os.listdir(path)
    list_of_int_experiments = []
    for exp in list_of_experiments:
        try:
            int_exp = int(exp)
        except ValueError:
            continue
        list_of_int_experiments.append(int_exp)

    if len(list_of_int_experiments) == 0:
        return 0

    return max(list_of_int_experiments)


def check_path(path):
    os.makedirs(path, exist_ok=True)
    return path


def prepare_prediction(outputs, argmax_dim=1):
    """

    :param outputs: [batch_size, n_stages, seq_len, n_classes]
    :param argmax_dim:
    :return:
    """
    if outputs.ndim == 4:
        out_logits = outputs[:, -1]
    else:
        out_logits = outputs

    pred = torch.argmax(out_logits.data, dim=argmax_dim)
    return pred


def exists(val):
    return val is not None


def params_to_string(params_num, units=None, precision=2):
    if units is None:
        if params_num // 10 ** 6 > 0:
            return str(round(params_num / 10 ** 6, precision)) + ' M'
        elif params_num // 10 ** 3:
            return str(round(params_num / 10 ** 3, precision)) + ' k'
        else:
            return str(params_num)
    else:
        if units == 'M':
            return str(round(params_num / 10.**6, precision)) + ' ' + units
        elif units == 'K':
            return str(round(params_num / 10.**3, precision)) + ' ' + units
        else:
            return str(params_num)


def get_commit_hash():
    import subprocess
    """https://stackoverflow.com/questions/14989858/get-the-current-git-hash-in-a-python-script"""
    # try:
    #     subprocess.check_output(['git', 'diff-index', '--quiet',
    #                              'HEAD'])  # Verify client is clean
    # except:
    #     raise RuntimeError(
    #         " !! Commit before training to get the commit hash.")
    try:
        commit = subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD']).decode().strip()
    # Not copying .git folder into docker container
    except subprocess.CalledProcessError:
        commit = "0000000"
    return commit


def check_nan_losses(loss, epoch, iter):
    """
    Determine whether the loss is NaN (not a number).
    Args:
        loss (loss): loss to check whether is NaN.
    """
    import math
    import datetime
    if math.isnan(loss):
        logger.warn(f"Stopping program at epoch {epoch}/ iter: {iter}")
        raise RuntimeError("ERROR: Got NaN losses {}".format(datetime.now()))


def segm_to_transcript(segm):
    """

    :param segm:
    :return:
    a list containing the action label of segment and start and end time
     (action_lbl, (st, end))

    e.g. [4, 5, 5, 6] -> [4, 5, 6], [1, 2, 1]
    lens [0, 1, 2, 1]
    ends    [1, 3, 4]
    (0, 1), (1, 3), (2, 4)
    """
    transcript = []
    action_segm, lens = summarize_list(segm)
    ends = np.cumsum(lens)
    starts = [0, *ends[:-1]]
    assert ((np.array(ends) - np.array(starts)) == np.array(lens)).all(),\
        "Start and end of segments are wrong"
    for idx, (st, en) in enumerate(zip(starts, ends)):
        transcript.append((action_segm[idx], (st, en)))
    return transcript

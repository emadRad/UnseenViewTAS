from typing import List, Dict
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from einops import rearrange


import logging
logger = logging.getLogger(__name__)


def collate_single_sample(batch: List[Dict], pad_ignore_idx: int):
    """
    Pads the features and targets of a batch of input sequences.

    :param batch: list of samples with 'features' [D, T], 'targets' [T], 'video_name' and 'view_name'
    :param pad_ignore_idx: value used to pad the targets
    :return: dict with 'features' [B, D, T], 'targets' [B, T], 'masks' [B, 1, T], 'video_name', 'view_name'
    """
    batch_features = []
    batch_targets = []
    batch_vid_names = []
    batch_view_names = []

    for item in batch:
        batch_features.append(rearrange(item['features'], "dim seq_len -> seq_len dim"))
        batch_targets.append(item['targets'])
        batch_vid_names.append(item['video_name'])
        batch_view_names.append(item['view_name'])

    # for pad_sequence the sequence length should be in the first dimension
    batch_feat_tensor = pad_sequence(batch_features, batch_first=True)  # [B, T, D]
    batch_target_tensor = pad_sequence(batch_targets, batch_first=True, padding_value=pad_ignore_idx)  # [B, T]

    masks = torch.where(batch_target_tensor == pad_ignore_idx, 0, 1)  # [batch_size, max_seq_len]
    masks = masks[:, None, :].bool()

    return {"features": rearrange(batch_feat_tensor, "b seq_len dim -> b dim seq_len"),
            "targets": batch_target_tensor,
            "video_name": batch_vid_names,
            "view_name": batch_view_names,
            "masks": masks}


def sequence_collate(batch: List[Dict], pad_ignore_idx: int):
    """

    :param batch: list of batches to collate
    :param pad_ignore_idx: the integer value to use for padding targets
    :return:
        padded inputs
        'features': torch.Tensor [batch_size, input_dim_size, sequence_length]
        'targets': torch.Tensor [batch_size, sequence_length]
        'masks': torch.Tensor [batch_size, 1, sequence_length]
        'video_name': list of video names
    """
    batch_dict = {}
    if "Input1" in batch[0]:
        batch_dict = {"Input1": collate_single_sample([item['Input1'] for item in batch], pad_ignore_idx),
                      "Input2": collate_single_sample([item['Input2'] for item in batch], pad_ignore_idx)}
    elif "Input" in batch[0]:
        batch_dict['Input'] = collate_single_sample([item['Input'] for item in batch], pad_ignore_idx)

    if "Segm_Feat" in batch[0]:
        segm_feats = []
        per_segm_masks = []
        len_segm = []
        for item in batch:
            segm_feats.append(rearrange(item['Segm_Feat'], "p dim seq_len -> seq_len p dim"))
            per_segm_masks.append(item['Segm_Mask'])
            len_segm.append(item['Segm_Feat'].shape[-1])

        batch_dict["Segm_Feat"] = rearrange(pad_sequence(segm_feats, batch_first=True),
                                            "b seq_len p dim -> b p dim seq_len")

        masks_segm = torch.zeros(batch_dict["Segm_Feat"].shape[0],
                                 batch_dict["Segm_Feat"].shape[1],
                                 batch_dict["Segm_Feat"].shape[-1]).bool()  # [batch_size, 2, max_seq_len]
        for batch_idx, _ in enumerate(len_segm):
            mask1, mask2 = per_segm_masks[batch_idx]
            l1 = mask1.sum().item()
            l2 = mask2.sum().item()
            masks_segm[batch_idx, 0, :l1] = True
            masks_segm[batch_idx, 1, :l2] = True

        batch_dict['segm_mask'] = masks_segm[:, :, None, :]

    return batch_dict


def conform_temporal_sizes(tensor, seq_len):
    """

    :param tensor: torch.Tensor of shape [T] or [B, T]
    :param seq_len:
    :return:
    """
    if tensor.shape[0] == seq_len:
        return tensor
    # logger.info(f"Resizing label's size from {tensor.shape[0]} to {seq_len}")
    if isinstance(tensor, np.ndarray):
        tensor = torch.from_numpy(tensor)
    if len(tensor.shape) == 1:
        tensor = tensor[None, None, :]
    elif len(tensor.shape) == 2:
        tensor = tensor[None, :]
    resized_tensor = F.interpolate(tensor.float(), size=seq_len, mode='nearest')
    return resized_tensor.squeeze().long()


# this code is inspired from https://github.com/Deep-MI/FastSurfer


def linear_align_segments(segm1: torch.Tensor, segm2: torch.Tensor):
    """
    Linear temporal alignment of two action segments for the action loss (Eq. 3 of the paper):
    the shorter segment is kept (pi(t) = t) and the longer one is linearly subsampled to the same length.

    Segments are returned with at least 2 frames (a 1-frame segment is repeated), since instance normalization
    needs more than one temporal element during training.

    :param segm1: features of shape [D, T1]
    :param segm2: features of shape [D, T2]
    :return: both segments with shape [D, max(min(T1, T2), 2)]
    """
    seq_len = max(min(segm1.shape[-1], segm2.shape[-1]), 2)

    def subsample(x):
        if x.shape[-1] == seq_len:
            return x
        idx = torch.linspace(0, x.shape[-1] - 1, seq_len).round().long()
        return x[:, idx]

    return subsample(segm1), subsample(segm2)

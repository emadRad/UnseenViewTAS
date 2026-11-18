import random
from collections import defaultdict
import os
from glob import glob
from os.path import join

import numpy as np
import torch
from torch.utils.data import Dataset

from uvtas.dataset.utils import conform_temporal_sizes, linear_align_segments
import uvtas.utils.logging as logging
import uvtas.utils.misc as misc
logger = logging.get_logger(__name__)
VIEWS = [
    'dev1',
    'dev2',
    'dev3'
]



class IKEA_ASM(Dataset):
    def __init__(self, cfg, mode):
        assert mode in [
            "train",
            "test",
        ], "Split '{}' not supported".format(mode)

        self._mode = mode
        self._cfg = cfg

        # SET VIEWS
        if mode == "train":
            views = set(VIEWS) - set(cfg.DATA.UNSEEN_EXO_VIEWS)
            logger.info(f"Number of training Views: {len(views)}/{len(VIEWS)}")
        else:
            views = VIEWS

        self._views = list(views)

        self._load_seq_pairs = mode == 'train' and cfg.MODEL.SEQUENCE_LOSS_WEIGHT > 0
        self._load_segm_pairs = mode == 'train' and cfg.MODEL.ACTION_LOSS_WEIGHT > 0

        # PARAMS
        self._video_meta = {}
        self._path_to_data = cfg.DATA.PATH_TO_DATA_DIR   
        self._feat_backbone = cfg.DATA.FEATURE_BACKBONE
        self._video_sampling_rate = cfg.DATA.FRAME_SAMPLING_RATE  

        # LOAD
        self._data, self._vid2view, self._action_to_videos = self._construct_loader()
        self._dataset_size = len(self._data)

    # GET DATA
    def _construct_loader(self):
        """
        Construct the list of features and segmentations.
        """
        list_of_videos = []
        annot_path = join(os.getcwd(), "data/ikea_asm")
        furniture_list = glob(join(annot_path,  "segm_anns_pv", f"{self._mode}/*",))
        for furniture in furniture_list:
            list_of_videos.extend(glob(furniture + '/*'))

        # ACTION DICT
        with open(join(annot_path, "mapping.txt"), "r") as f:
            lines = f.readlines()
            lines = [line.strip('\n') for line in lines]
            lines = [line.split('-') for line in lines]
            idx_to_action = {int(str_idx): action for str_idx, action in lines}
            action_to_idx = {v: k for (k, v) in idx_to_action.items()}

        # ANNS
        action_to_videos = defaultdict(list)
        data_dicts = defaultdict(list)
        video_data = []
        num_videos = int(len(list_of_videos) * self._cfg.DATA.DATA_FRACTION)
        for gt_filename in list_of_videos[:num_videos]:
            # video id
            video_id = gt_filename.replace('.txt', '')
            view = video_id.split('_')[-1]
            if view not in self._views:
                continue
            video_id = '/'.join(video_id.split('/')[-2:])    # FURNITURE/VIDEO
            video_id = video_id[:-5]
            video_name = video_id + '/' + view   # /VIEW (DEV1, DEV2, DEV3)

            # features
            feat_filename = video_name + ".npy"
            feat_path = join(self._path_to_data, "features", self._feat_backbone, feat_filename)
            assert os.path.isfile(feat_path), f"Feature {feat_path} not found."

            segmentation = self._load_segmentations(gt_filename, action_to_idx)
            transcript = misc.segm_to_transcript(segmentation)

            for action, (st, en) in transcript:
                key = (action, view)
                action_to_videos[key].append({"segm_start": st,
                                              "segm_end": en,
                                              "view": view,
                                              "feat_path": feat_path})

            data_dicts[video_id].append({"feat_path": feat_path,
                                         "segm": segmentation,
                                         "view_name": view,
                                         "transcript": transcript,
                                         "video_name": video_id})

        for vid_id, view_list in data_dicts.items():
            video_data.extend(view_list)

        return video_data, data_dicts, action_to_videos

    def _load_segmentations(self, segm_path, action_to_idx):
        with open(segm_path, "r") as f:
            actions = map(lambda l: l.strip(), f.readlines())
        segmentation = list(map(lambda ac: action_to_idx[ac], actions))
        return np.array(segmentation, dtype=np.int16)

    def _load_features(self, feature_path: str):
        features = np.load(feature_path)
        features = features.astype(np.float32).T
        features = features[:, :: self._video_sampling_rate]  # [D, T]
        return features

    def _load_single_input(self, data_dict):
        sample = {}
        targets = data_dict['segm']

        sample['targets'] = torch.tensor(targets).long()[:: self._video_sampling_rate]  # [T]

        sample['features'] = torch.tensor(self._load_features(data_dict['feat_path']))  # [D, T]

        seq_length = sample['features'].shape[-1]
        sample['video_name'] = data_dict['video_name']
        sample['view_name'] = data_dict['view_name']

        if np.abs(sample['targets'].shape[0] - seq_length) > 200:
            logger.warn(f"Down/Up-sampling targets from {sample['targets'].shape[0]} to {seq_length}")
        sample['targets'] = conform_temporal_sizes(sample['targets'], seq_length)

        return sample

    def __getitem__(self, index: int):
        """
        :param index:
        :return: sample dict with 'Input' (or 'Input1' and 'Input2' for the sequence loss) and, for the action loss,
         'Segm_Feat' [2, D, T] and 'Segm_Mask' [2, T]. Each input contains
         'features': torch.Tensor [D, T]
         'targets': torch.Tensor [T]
        """
        sample = {}
        data_dict = self._data[index]
        if self._load_seq_pairs:
            views_list = self._vid2view[data_dict['video_name']]
            rand_idx = np.random.randint(0, len(views_list))
            if data_dict['view_name'] == views_list[rand_idx]['view_name']:
                rand_idx = (rand_idx + 1) % len(views_list)
            sample["Input1"] = self._load_single_input(data_dict)
            sample["Input2"] = self._load_single_input(views_list[rand_idx])
            seq1_len = sample["Input1"]['features'].shape[-1]
            seq2_len = sample["Input2"]['features'].shape[-1]
            seq_len = min(seq1_len, seq2_len)
            # the views of a recording can differ by a few frames in length
            for k in ["Input1", "Input2"]:
                sample[k]['features'] = sample[k]['features'][:, :seq_len]
                sample[k]['targets'] = sample[k]['targets'][:seq_len]
        else:
            sample["Input"] = self._load_single_input(data_dict)

        if self._load_segm_pairs:
            # a segment of the current video and a segment with the same action label from a different view
            transcripts = data_dict['transcript']
            rand_segm_idx = np.random.randint(0, len(transcripts))
            lbl, (segm_start, segm_end) = transcripts[rand_segm_idx]

            view = data_dict['view_name']
            candidate_views = list(set(self._views) - set([view]))
            view = random.choice(candidate_views)

            key = (lbl, view)
            candidate_segments = self._action_to_videos[key]
            if len(candidate_segments) == 0:
                candidate_segments = [{"segm_start": segm_start,
                                       "segm_end": segm_end,
                                       "feat_path": data_dict['feat_path']}]
            segm_dict = candidate_segments[np.random.randint(0, len(candidate_segments))]

            segm1 = torch.tensor(self._load_features(data_dict['feat_path']))[:, segm_start: segm_end]
            segm2 = torch.tensor(self._load_features(segm_dict['feat_path']))[:,
                                                                              segm_dict['segm_start']:segm_dict['segm_end']]
            segm1, segm2 = linear_align_segments(segm1, segm2)

            # temporal augmentation: random speed-up of long segment pairs
            if np.random.rand() < 0.3 and segm1.shape[-1] > 50:
                sample_rate = np.random.randint(2, 6)
                segm1, segm2 = segm1[:, ::sample_rate], segm2[:, ::sample_rate]

            sample['Segm_Feat'] = torch.stack([segm1, segm2], dim=0)  # -> [2, D, T]
            sample['Segm_Mask'] = torch.ones(2, segm1.shape[-1]).bool()  # -> [2, T], no padding
        return sample

    def __len__(self):
        """
        :return: the number of data point (videos) in the dataset.
        """
        return self._dataset_size

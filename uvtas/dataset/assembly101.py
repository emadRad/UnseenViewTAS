from collections import defaultdict
import os
from os.path import join, isfile

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from uvtas.dataset.utils import conform_temporal_sizes, linear_align_segments
import uvtas.utils.logging as logging

logger = logging.get_logger(__name__)

EGO_VIEWS = [
    'HMC_84355350_mono10bit',
    'HMC_84358933_mono10bit',
    'HMC_21110305_mono10bit',
    'HMC_21179183_mono10bit'
]

EXO_VIEWS = [
    'C10095_rgb',
    'C10115_rgb',
    'C10118_rgb',
    'C10119_rgb',
    'C10390_rgb',
    'C10404_rgb',
    'C10379_rgb',
    'C10395_rgb',
]

# names of the views used in the paper (Fig. 3)
VIEW_NAMES = {
     'C10095_rgb': 'Exo_1',
     'C10115_rgb': 'Exo_2',
     'C10118_rgb': 'Exo_3',
     'C10119_rgb': 'Exo_4',
     'C10404_rgb': 'Exo_5',
     'C10390_rgb': 'Exo_6',
     'C10395_rgb': 'Exo_7',
     'C10379_rgb': 'Exo_8',
     'HMC_84355350_mono10bit': 'Ego_1',
     'HMC_84358933_mono10bit': 'Ego_2',
     'HMC_21110305_mono10bit': 'Ego_3',
     'HMC_21179183_mono10bit': 'Ego_4'
}


class Assembly101(Dataset):
    """
    Assembly101 with the coarse action segmentation labels.

    For training, a sample contains
      - two synchronized views of the same sequence ('Input1', 'Input2') if the sequence loss is used,
        otherwise a single view ('Input'),
      - a pair of fine-grained action segments with the same action label from different views
        ('Segm_Feat', 'Segm_Mask') if the action loss is used.
    For evaluation, a sample contains a single view ('Input').
    """
    def __init__(self, cfg, mode):
        assert mode in ["train", "val"], "Split '{}' not supported".format(mode)

        self._mode = mode
        self._cfg = cfg

        if mode == 'train':
            ego_views = set(EGO_VIEWS) - set(cfg.DATA.UNSEEN_EGO_VIEWS)
            exo_views = set(EXO_VIEWS) - set(cfg.DATA.UNSEEN_EXO_VIEWS)
            logger.info(f"Number of training EGO Views: {len(ego_views)}/{len(EGO_VIEWS)}")
            logger.info(f"Number of training EXO Views: {len(exo_views)}/{len(EXO_VIEWS)}")
        else:
            ego_views = EGO_VIEWS
            exo_views = EXO_VIEWS

        self._views = list(exo_views) + list(ego_views)
        if mode == 'train':
            views_name = sorted([f'{VIEW_NAMES[v]}/{v}' for v in self._views])
            logger.info(f"All training views: {', '.join(views_name)}")

        self._path_to_data = self._cfg.DATA.PATH_TO_DATA_DIR
        self._video_sampling_rate = cfg.DATA.FRAME_SAMPLING_RATE
        self._feat_backbone = cfg.DATA.FEATURE_BACKBONE

        self._load_seq_pairs = mode == 'train' and cfg.MODEL.SEQUENCE_LOSS_WEIGHT > 0
        self._load_segm_pairs = mode == 'train' and cfg.MODEL.ACTION_LOSS_WEIGHT > 0

        self._data, self._vid2view = self._construct_loader()
        self._dataset_size = len(self._data)
        logger.info(f"Number {mode} input sequences is: {self._dataset_size}")

    def _construct_loader(self):
        """
        Construct the list of features and segmentations.
        """
        annotation_path = join(os.getcwd(), "data/assembly101", "coarse-annotations")
        path_to_csvfile = join(os.getcwd(), "data/assembly101", f"{self._mode}.csv")
        assert os.path.exists(path_to_csvfile), f"Split csv file not found at {path_to_csvfile}"

        coarse_df = pd.read_csv(path_to_csvfile)
        actions_df = pd.read_csv(join(annotation_path, "actions.csv"))

        # fine-grained action segments, used to sample segment pairs for the action loss
        fine_annotation_file = join(os.getcwd(), "data/assembly101", "fine-grained-annotations",
                                    "train_segm_modified_l.csv")
        assert os.path.isfile(fine_annotation_file), \
            f"{fine_annotation_file} not found. See Section 2.1 of the README for the download."
        self._fine_meta_df = pd.read_csv(fine_annotation_file)
        self._fine_meta_df['video_id'] = self._fine_meta_df['video'].transform(lambda x: x.split("/")[0])
        cols = ['video_id', 'view', 'action_type', 'video_start_frame', 'video_end_frame']
        data_df = coarse_df[cols].drop_duplicates().reset_index()

        self._fine_meta_df['view'] = self._fine_meta_df['video'].transform(lambda x: x.split("/")[-1].split(".")[0])
        self._fine_meta_df = self._fine_meta_df.drop(
            index=self._fine_meta_df[self._fine_meta_df['end_frame'] > self._fine_meta_df['video_end_frame']].index
        )

        # the features are stored separately for the assembly and disassembly part of a recording, therefore
        # the start frame of the fine-grained segments is adjusted to the start of the assembly part
        cond = self._fine_meta_df['action_type'] == 'assembly'
        self._fine_meta_df.loc[cond, 'start_frame'] = self._fine_meta_df.loc[cond, 'start_frame'] - self._fine_meta_df.loc[cond, 'video_start_frame']
        self._fine_meta_df.loc[cond, 'end_frame'] = self._fine_meta_df.loc[cond, 'end_frame'] - self._fine_meta_df.loc[cond, 'video_start_frame']
        self._fine_meta_df = self._fine_meta_df[self._fine_meta_df['start_frame'] >= 0]
        self._fine_meta_df = self._fine_meta_df[self._fine_meta_df['end_frame'] >= 0]

        self._fine_meta_df['segm_len'] = self._fine_meta_df['end_frame'] - self._fine_meta_df['start_frame']
        self._fine_meta_df = self._fine_meta_df[self._fine_meta_df['segm_len'] >= 16]

        feat_name = self._fine_meta_df['video'].transform(lambda x: x.split(".")[0] + ".npy")
        feat_dir = join(self._path_to_data, "features", self._feat_backbone)
        self._fine_meta_df['feat_path'] = feat_dir + "/" + self._fine_meta_df['action_type'] + "_" + feat_name
        self._fine_meta_df = self._fine_meta_df[self._fine_meta_df.apply(lambda row: isfile(row['feat_path']), axis=1)]
        self._fine_meta_df['video_id'] = self._fine_meta_df['action_type'] + "_" + self._fine_meta_df['video_id']

        num_videos = int(self._cfg.DATA.DATA_FRACTION * len(data_df))
        data_df = data_df[:num_videos]
        video_data = []
        data_dicts = defaultdict(list)
        views = []
        for idx, entry in tqdm(data_df.iterrows(), total=num_videos):
            sample = entry.to_dict()
            if sample['view'] not in self._views:
                continue
            views.append(sample['view'])
            vid_id = f"{sample['action_type']}_{entry['video_id']}"
            feature_path = join(self._path_to_data,
                                "features",
                                self._feat_backbone,
                                vid_id,
                                f"{entry['view']}.npy")
            if not isfile(feature_path):
                logger.warn(f"Feature {join(entry['video_id'], entry['view'])} not found")
                continue

            sample['feat_path'] = feature_path
            sample['vid_id'] = vid_id
            segm_path = join(annotation_path, "coarse_labels", f"{vid_id}.txt")
            segm = self._load_segmentations(segm_path, actions_df)
            if len(segm) == 0:
                continue

            sample['segm'] = segm
            # the features cover exactly the annotated part (assembly or disassembly) of the recording
            sample['start_frame'] = 0
            sample['end_frame'] = len(segm)
            data_dicts[vid_id].append(sample)

        for vid_id, view_list in data_dicts.items():
            video_data.extend(view_list)

        self._fine_meta_df = self._fine_meta_df[self._fine_meta_df['video_id'].isin(list(data_dicts.keys()))]

        logger.info(f"{self._mode} number of views: {len(set(views))}")
        return video_data, data_dicts

    def _load_features(self,
                       feature_path: str,
                       start_frame: int,
                       end_frame: int,
                       sample_rate: int):
        """
        :return:
            the features of shape [D, T] where T is the temporal dimension
        """
        features = np.load(feature_path)  # [T, D]
        features = features.transpose(1, 0)  # -> [D, T]
        features = features[:, start_frame:end_frame:sample_rate]  # [D, T]
        if features.dtype == np.float16:
            features = features.astype(np.float32)
        return torch.tensor(features)

    @staticmethod
    def _load_segmentations(segm_path, actions_df):
        segment_labels = []
        start_indices = []
        end_indices = []
        with open(segm_path, 'r') as f:
            lines = list(map(lambda s: s.split("\n"), f.readlines()))
            for line in lines:
                start, end, lbl = line[0].split("\t")[:-1]
                start_indices.append(int(start))
                end_indices.append(int(end))
                action_id = actions_df.loc[actions_df['action_cls'] == lbl, 'action_id']
                segm_len = int(end) - int(start)
                segment_labels.append(np.full(segm_len, fill_value=action_id.item()))

        segmentation = np.concatenate(segment_labels)
        assert segmentation.shape[0] == (max(end_indices) - min(start_indices)), \
            "Length of Segmentation doesn't match with clip length."
        return segmentation

    def _load_single_input(self, data_dict, sampling_rate):
        sample = {}
        sample['targets'] = torch.tensor(data_dict['segm']).long()[::sampling_rate]  # [T]
        sample['features'] = self._load_features(data_dict['feat_path'],
                                                 data_dict['start_frame'],
                                                 data_dict['end_frame'],
                                                 sampling_rate)  # [D, T]

        seq_length = sample['features'].shape[-1]
        sample['video_name'] = join(data_dict['action_type'], data_dict['video_id'], data_dict['view'])
        sample['view_name'] = data_dict['view']

        if np.abs(sample['targets'].shape[0] - seq_length) > 200:
            logger.warn(f"Down/Up-sampling targets from {sample['targets'].shape[0]} to {seq_length}")
        sample['targets'] = conform_temporal_sizes(sample['targets'], seq_length)
        return sample

    def _get_fine_segment_pair(self, vid_id, action_type, view):
        """
        Samples a fine-grained action segment of the given video and view, and a second segment with the same
        fine-grained action label from a different view. The two segments are linearly aligned.

        :return: features [2, D, T] and masks [2, T]
        """
        vid_cond = ((self._fine_meta_df['video_id'] == vid_id)
                    & (self._fine_meta_df['action_type'] == action_type))

        fine_actions = self._fine_meta_df[vid_cond]['action_cls'].unique()
        fine_action = np.random.choice(fine_actions)
        cond = ((self._fine_meta_df['action_cls'] == fine_action) &
                (self._fine_meta_df['view'] == view))
        action_df = self._fine_meta_df[cond]

        cols = ['start_frame', 'end_frame', "feat_path"]
        vid_id_cond = action_df['video_id'] == vid_id
        segm1 = action_df[vid_id_cond].sample(n=1, random_state=115)[cols]
        segm1_start, segm1_end = segm1['start_frame'].iloc[0], segm1['end_frame'].iloc[0]

        # the second segment comes from a different view
        cond = ((self._fine_meta_df['action_cls'] == fine_action) &
                (self._fine_meta_df['view'] != view))
        remain_action_df = self._fine_meta_df[cond]
        if remain_action_df.shape[0] == 0:
            remain_action_df = action_df

        segm2 = remain_action_df.sample(n=1)[cols]
        segm2_start, segm2_end = segm2['start_frame'].iloc[0], segm2['end_frame'].iloc[0]

        segm1_feat = self._load_features(segm1['feat_path'].iloc[0], segm1_start, segm1_end, sample_rate=1)
        segm2_feat = self._load_features(segm2['feat_path'].iloc[0], segm2_start, segm2_end, sample_rate=1)

        segm1_feat, segm2_feat = linear_align_segments(segm1_feat, segm2_feat)
        segm_feats = torch.stack([segm1_feat, segm2_feat], dim=0)  # -> [2, D, T]
        segm_masks = torch.ones(2, segm_feats.shape[-1]).bool()  # -> [2, T], no padding
        return segm_feats, segm_masks

    def __getitem__(self, index: int):
        """
        :param index:
        :return: sample dict (see class description); each input contains
         'features': torch.Tensor [D, T]
         'targets': torch.Tensor [T]
        """
        sample = {}
        data_dict = self._data[index]
        sampling_rate = self._video_sampling_rate

        if self._load_seq_pairs:
            # a second, synchronized view of the same sequence
            views_list = self._vid2view[data_dict['vid_id']]
            rand_idx = np.random.randint(0, len(views_list))
            if data_dict['view'] == views_list[rand_idx]['view']:
                rand_idx = (rand_idx + 1) % len(views_list)
            sample["Input1"] = self._load_single_input(data_dict, sampling_rate)
            sample["Input2"] = self._load_single_input(views_list[rand_idx], sampling_rate)
        else:
            sample["Input"] = self._load_single_input(data_dict, sampling_rate)

        if self._load_segm_pairs:
            curr_vid_id = data_dict['vid_id']
            action_type = data_dict['action_type']
            view = data_dict['view']
            if (self._fine_meta_df['video_id'] == curr_vid_id).sum() == 0:
                # no fine-grained annotation for this video: use a random one
                rand_vid = self._fine_meta_df.sample(n=1).iloc[0]
                curr_vid_id = rand_vid['video_id']
                action_type = rand_vid['action_type']
                view = rand_vid['view']

            sample['Segm_Feat'], sample['Segm_Mask'] = self._get_fine_segment_pair(vid_id=curr_vid_id,
                                                                                   action_type=action_type,
                                                                                   view=view)
        return sample

    def __len__(self):
        """
        Returns:
            (int): the number of input sequences.
        """
        return self._dataset_size

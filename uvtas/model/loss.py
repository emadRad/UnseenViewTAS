from typing import Dict, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from yacs.config import CfgNode


class ActionSegmentationLoss(nn.Module):
    """
    Temporal action segmentation loss L_TAS from MS-TCN: frame-wise cross entropy + smoothing (truncated MSE) loss.
    https://arxiv.org/abs/1903.01945
    """
    def __init__(self, cfg: CfgNode):
        super(ActionSegmentationLoss, self).__init__()
        self.ce = nn.CrossEntropyLoss(ignore_index=cfg.MODEL.PAD_IGNORE_IDX)
        self.mse = nn.MSELoss(reduction='none')
        self.mse_fraction = cfg.MODEL.MSE_LOSS_FRACTION
        self.mse_clip_val = cfg.MODEL.MSE_LOSS_CLIP_VAL

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> Dict:
        """
        :param logits: [batch_size, n_stages, n_classes, seq_len]
        :param targets: [batch_size, seq_len]
        :return: dict with the total loss ('loss') and its terms
        """
        loss_dict = {"loss": 0.0, "loss_ce": 0.0, "loss_mse": 0.0}
        n_stages = logits.shape[1]
        stage_logits = torch.tensor_split(logits, n_stages, dim=1)
        for p in stage_logits:
            p = p.squeeze(1)  # [b, 1, n_classes, seq_len] ->  [b, n_classes, seq_len]
            loss_dict['loss_ce'] += self.ce(rearrange(p, "b n_classes seq_len -> (b seq_len) n_classes"),
                                            rearrange(targets, "b seq_len -> (b seq_len)"))

            loss_dict['loss_mse'] += torch.mean(torch.clamp(self.mse(F.log_softmax(p[:, :, 1:], dim=1),
                                                                     F.log_softmax(p.detach()[:, :, :-1], dim=1)),
                                                            min=0,
                                                            max=self.mse_clip_val))

        loss_dict['loss'] = loss_dict['loss_ce'] + self.mse_fraction * loss_dict['loss_mse']

        return loss_dict


class CrossViewSimilarityLoss(nn.Module):
    """
    Symmetric negative frame-wise cosine similarity between the predictor output of one view and the
    (stop-gradient) representation of the other view.
    Used for the sequence loss (Eq. 1-2, two views of the same sequence) and for the action loss
    (Eq. 3-4, two segments of the same action from different views).
    """
    def __init__(self, loss_weight: float):
        super().__init__()
        self.loss_weight = loss_weight
        self.cosine_sim = nn.CosineSimilarity(dim=1)

    def forward(self,
                features: List[torch.Tensor],
                predictions: List[torch.Tensor],
                masks: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param features: representations z of the two inputs, each of shape [b, n_stages, d, seq_len]
        :param predictions: predictor outputs p of the two inputs, each of shape [b, n_stages, d, seq_len]
        :param masks: masks of the two inputs, each of shape [b, 1, seq_len]
        :return: the weighted loss and the unweighted negative cosine similarity
        """
        n_stages = features[0].shape[1]
        z1_per_stage, z2_per_stage = features
        p1_per_stage, p2_per_stage = predictions
        m1 = masks[0][:, 0]  # [b, seq_len]
        m2 = masks[1][:, 0]

        # stop gradient
        z1_per_stage = z1_per_stage.detach()
        z2_per_stage = z2_per_stage.detach()

        neg_cosine = 0.0
        for s in range(n_stages):
            sim1 = self.cosine_sim(p1_per_stage[:, s], z2_per_stage[:, s])[m1].mean()
            sim2 = self.cosine_sim(p2_per_stage[:, s], z1_per_stage[:, s])[m2].mean()
            neg_cosine += - (sim1 + sim2) * 0.5

        return self.loss_weight * neg_cosine, neg_cosine

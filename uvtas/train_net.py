from typing import Dict
import os
import pprint

from tqdm import tqdm
from yacs.config import CfgNode
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.dataloader import DataLoader

from uvtas.model.loss import ActionSegmentationLoss, CrossViewSimilarityLoss
import uvtas.model.optimizer as optim
import uvtas.utils.checkpoint as cu
import uvtas.utils.misc as misc
from uvtas.dataset import loader
from uvtas.model import model_builder
from uvtas.utils.meters import TrainMeter, ValMeter

import uvtas.utils.logging as logging

logger = logging.get_logger(__name__)


def train_epoch(
        train_loader: DataLoader,
        model: nn.Module,
        optimizer: Optimizer,
        train_meter: TrainMeter,
        cur_epoch: int,
        cfg: CfgNode,
        device: torch.device.type
):
    """
    Perform the training for one epoch with L = L_TAS + lambda * L_seq + beta * L_action (Eq. 5).

    A training batch contains
      - 'Input1' and 'Input2': two synchronized views of the same sequences (if the sequence loss is used),
        otherwise 'Input': a single view,
      - 'Segm_Feat' and 'segm_mask': pairs of segments of the same action from different views (if the action loss is used).
    L_TAS is applied to every input sequence.

    :param train_loader: video features training loader.
    :param model: the model to train
    :param optimizer: the optimizer to perform optimization on the model's parameters.
    :param train_meter: training meters to log the training performance.
    :param cur_epoch: current epoch of training.
    :param cfg: configs. Details can be found in uvtas/config/defaults.py
    :param device: device to use (cuda or cpu)
    """
    model.train()
    action_segm_loss = ActionSegmentationLoss(cfg)
    sequence_loss = CrossViewSimilarityLoss(cfg.MODEL.SEQUENCE_LOSS_WEIGHT)
    action_loss = CrossViewSimilarityLoss(cfg.MODEL.ACTION_LOSS_WEIGHT)
    use_sequence_loss = cfg.MODEL.SEQUENCE_LOSS_WEIGHT > 0
    use_action_loss = cfg.MODEL.ACTION_LOSS_WEIGHT > 0

    lr = optim.get_epoch_lr(optimizer)

    train_bar = tqdm(enumerate(train_loader), total=len(train_loader))
    for cur_iter, batch in train_bar:
        if "Input1" in batch:
            batch_list = [batch['Input1'], batch['Input2']]
        else:
            batch_list = [batch['Input']]

        loss = 0.0
        out_features = []
        out_predictor = []
        masks = []
        for batch_dict in batch_list:
            misc.move_to_device(batch_dict, device)
            framewise_logits, out_feat, predictor_out = model(batch_dict['features'], batch_dict['masks'])
            loss_dict = action_segm_loss(framewise_logits, batch_dict['targets'])

            out_features.append(out_feat)
            out_predictor.append(predictor_out)
            masks.append(batch_dict['masks'])
            loss += loss_dict['loss']

        # action loss: two segments of the same action from different views
        if use_action_loss and 'Segm_Feat' in batch:
            segm_pairs_feat = batch['Segm_Feat'].to(device)
            segm_pairs_mask = batch['segm_mask'].to(device)
            _, segm1_out_feat, segm1_predictor = model(segm_pairs_feat[:, 0], segm_pairs_mask[:, 0])
            _, segm2_out_feat, segm2_predictor = model(segm_pairs_feat[:, 1], segm_pairs_mask[:, 1])
            action_loss_val, neg_cos = action_loss([segm1_out_feat, segm2_out_feat],
                                                   [segm1_predictor, segm2_predictor],
                                                   [segm_pairs_mask[:, 0], segm_pairs_mask[:, 1]])
            loss_dict['segm_neg_cosine'] = neg_cos
            loss = loss + action_loss_val

        # sequence loss: two synchronized views of the same sequence
        if use_sequence_loss:
            sequence_loss_val, neg_cos = sequence_loss(out_features, out_predictor, masks)
            loss_dict['neg_cosine'] = neg_cos
            loss = loss + sequence_loss_val

        optimizer.zero_grad()
        loss.backward()

        if cfg.SOLVER.CLIP_GRAD_L2NORM:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg.SOLVER.CLIP_GRAD_L2NORM
            )
        else:
            grad_norm = optim.get_grad_norm_(model.parameters())

        optimizer.step()
        misc.check_nan_losses(loss_dict['loss'], cur_epoch, cur_iter)
        status_text = f"Train [Epoch {cur_epoch}/{cfg.SOLVER.MAX_EPOCH}]" \
                      f" [{cur_iter}/{len(train_loader)}] [Loss: {loss_dict['loss'].item():.4f} ]"
        train_bar.set_description(status_text)

        prediction = misc.prepare_prediction(framewise_logits)

        # Update and log stats.
        train_meter.update_stats(target=batch_dict['targets'],
                                 prediction=prediction,
                                 loss_dict=loss_dict,
                                 grad_norm=grad_norm.item(),
                                 lr=lr)
        train_meter.log_iter_stats(cur_epoch, cur_iter)

    train_meter.log_epoch_stats(cur_epoch)
    train_meter.reset()


@torch.no_grad()
def eval_epoch(
        val_loader: DataLoader,
        model: nn.Module,
        val_meter: ValMeter,
        cur_epoch: int,
        device: torch.device.type,
        cfg: CfgNode) -> Dict:
    """
    Evaluate the model on the val set.
    :param val_loader: data loader to provide validation data.
    :param model: model to evaluate the performance.
    :param val_meter: meter instance to record and calculate the metrics.
    :param cur_epoch: number of the current epoch of training.
    :param device: device to use (cuda or cpu)
    :param cfg:
    :return:
         a dictionary of validation metrics
    """
    logger.info(f"Validation started")
    model.eval()
    action_segm_loss = ActionSegmentationLoss(cfg)
    visualization_samples = []
    vis_period = round(len(val_loader) / 50)
    vis_period = 1 if vis_period == 0 else vis_period
    for cur_iter, (batch_dict) in tqdm(enumerate(val_loader), total=len(val_loader)):
        batch_dict = batch_dict['Input']
        misc.move_to_device(batch_dict, device)

        framewise_logits, _, _ = model(batch_dict['features'], batch_dict['masks'])

        loss_dict = action_segm_loss(framewise_logits, batch_dict['targets'])

        prediction = misc.prepare_prediction(framewise_logits)

        targets = batch_dict['targets'].cpu()
        prediction = prediction.cpu()

        video_name = batch_dict['video_name'][0]
        view_name = batch_dict['view_name'][0]

        val_meter.update_stats(targets,
                               prediction,
                               video_name,
                               view_name,
                               loss_dict)

        if visualization_samples == 0 or cur_iter % vis_period == 0:
            visualization_samples.append({"target": targets.squeeze(0).numpy(),
                                          "pred": prediction.squeeze(0).numpy(),
                                          "video_name": video_name})

        val_meter.log_iter_stats(cur_epoch, cur_iter)

    # Log epoch stats.
    val_meter.visualize_prediction_result(vis_data=visualization_samples, cur_epoch=cur_epoch)
    val_metrics, per_video_metrics_df = val_meter.log_epoch_stats(cur_epoch)
    per_video_metrics_df.to_csv(os.path.join(cfg.VAL_METRICS_PATH,
                                             f"epoch_{cur_epoch+1:03d}.csv"))
    val_meter.reset()
    return val_metrics


def train(cfg: CfgNode):
    """
    Train an action segmentation model for many epochs on train set and validate it on val set
    :param cfg: config file. Details can be found in uvtas/config/defaults.py
    :return:
    """
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)
    logging.setup_logging(cfg.OUTPUT_DIR, cfg.EXPR_NUM)
    # refer to https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(True)

    # Print config.
    logger.info("Train with config:")
    cfg.GIT_COMMIT_HASH = misc.get_commit_hash()
    logger.info("Commit Hash: {}".format(cfg.GIT_COMMIT_HASH))
    logger.info(pprint.pformat(cfg))

    model = model_builder.build_model(cfg)
    logger.info(f"Number of params: {misc.params_to_string(misc.params_count(model))}")
    # logger.info(f"Number of params: {misc.params_count(model):,}")

    if cfg.NUM_GPUS > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        # model = torch.nn.parallel.DistributedDataParallel(model)
        model = torch.nn.DataParallel(model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Transfer the model to device(s)
    model = model.to(device)

    optimizer = optim.construct_optimizer(model, cfg)
    scheduler = optim.construct_lr_scheduler(cfg, optimizer)

    # Load a checkpoint to resume training if applicable.
    checkpoint_path = cu.get_path_to_checkpoint(cfg)
    if os.path.isfile(checkpoint_path):
        checkpoint_epoch = cu.load_checkpoint(
            checkpoint_path,
            model=model,
            num_gpus=cfg.NUM_GPUS,
            optimizer=optimizer,
            lr_scheduler=scheduler
        )
        start_epoch = checkpoint_epoch + 1
    else:
        start_epoch = 0

    # Create tensorboard summary writer
    writer = SummaryWriter(cfg.SUMMARY_PATH)
    writer.add_text(f"Config_EXPR_NUM={cfg.EXPR_NUM}", pprint.pformat(cfg).replace("\n", "\n\n"))
    # Create the video train and val loaders.
    train_loader = loader.construct_loader(cfg, "train")
    train_meter = TrainMeter(
        len(train_loader), cfg, start_epoch * (len(train_loader)), writer
    )

    test_loader = loader.construct_loader(cfg, cfg.TRAIN.EVAL_SPLIT)
    # Create meters.
    val_meter = ValMeter(len(test_loader), cfg, writer)
    # Print summary path.
    logger.info("Summary path {}".format(cfg.SUMMARY_PATH))

    logger.info("Process PID {}".format(os.getpid()))
    # Perform the training loop.
    logger.info("Start epoch: {}".format(start_epoch + 1))

    best_metrics_sum = -1
    eval_metrics = {}

    for cur_epoch in range(start_epoch, cfg.SOLVER.MAX_EPOCH):
        # Train for one epoch.
        train_epoch(
            train_loader, model, optimizer, train_meter, cur_epoch, cfg, device
        )
        scheduler.step()
        # Save a checkpoint.
        if cu.is_checkpoint_epoch(cur_epoch, cfg.TRAIN.CHECKPOINT_PERIOD):
            logger.info("Saving checkpoint")
            best_metrics_sum = cu.save_checkpoint(
                cfg.OUTPUT_DIR,
                cfg.EXPR_NUM,
                model,
                optimizer,
                scheduler,
                eval_metrics,
                best_metrics_sum,
                cur_epoch,
                cfg
            )

        # Evaluate the model on validation set.
        if misc.is_eval_epoch(cfg, cur_epoch):
            eval_metrics = eval_epoch(test_loader, model, val_meter, cur_epoch, device, cfg)

    writer.flush()

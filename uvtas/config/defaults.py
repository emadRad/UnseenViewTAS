"""Default configs. Every option can be overridden from a YAML file (--cfg) or on the command line."""
from yacs.config import CfgNode

_C = CfgNode()

# ---------------------------------------------------------------------------- #
# Training options
# ---------------------------------------------------------------------------- #
_C.TRAIN = CfgNode()

_C.TRAIN.ENABLE = True

# Dataset: ['assembly101', 'ikea_asm']
_C.TRAIN.DATASET = "assembly101"

# Mini-batch size for training.
_C.TRAIN.BATCH_SIZE = 3

# Evaluate the model every EVAL_PERIOD epochs.
_C.TRAIN.EVAL_PERIOD = 1

# Batch size for evaluation (must be 1).
_C.TRAIN.EVAL_BATCH_SIZE = 1

# Split used for evaluation: 'val' (Assembly101) or 'test' (IKEA ASM).
_C.TRAIN.EVAL_SPLIT = "val"

# Save a checkpoint every CHECKPOINT_PERIOD epochs.
_C.TRAIN.CHECKPOINT_PERIOD = 1

# Resume training from the last checkpoint of experiment number RESUME_EXPR_NUM.
_C.TRAIN.AUTO_RESUME = False
_C.TRAIN.RESUME_EXPR_NUM = -1

# Labels ignored during evaluation.
_C.TRAIN.EVAL_IGNORE_LABELS = []

# Optional checkpoint used to initialize the weights.
_C.TRAIN.CHECKPOINT_FILE_PATH = ""

# ---------------------------------------------------------------------------- #
# Testing options
# ---------------------------------------------------------------------------- #
_C.TEST = CfgNode()

_C.TEST.ENABLE = False

# Checkpoint of the trained model to evaluate.
_C.TEST.CHECKPOINT_PATH = ""

# Labels ignored when computing the metrics.
_C.TEST.IGNORED_CLASSES = []

# Save the frame-wise predictions and ground truth as .npy files.
_C.TEST.SAVE_PREDICTIONS = False

# Folder where the metrics (and predictions) are saved.
_C.TEST.SAVE_RESULT_PATH = "./experiments/results"

# ---------------------------------------------------------------------------- #
# Model options
# ---------------------------------------------------------------------------- #
_C.MODEL = CfgNode()

# Number of action classes (Assembly101: 202, IKEA ASM: 33).
_C.MODEL.NUM_CLASSES = 202

# Dimension of the input features.
_C.MODEL.INPUT_DIM = 1024

# Ignore index of the cross entropy, also used to pad the targets.
_C.MODEL.PAD_IGNORE_IDX = 202

# Weight of the smoothing (MSE) term of the temporal action segmentation loss.
_C.MODEL.MSE_LOSS_FRACTION = 0.17

# Clamp value of the smoothing (MSE) term.
_C.MODEL.MSE_LOSS_CLIP_VAL = 16.0

# Weight lambda of the sequence loss (Eq. 2 of the paper). 0 disables the loss.
# The sequence loss requires at least two synchronized views of each training sequence.
_C.MODEL.SEQUENCE_LOSS_WEIGHT = 0.0

# Weight beta of the action loss (Eq. 4 of the paper). 0 disables the loss.
_C.MODEL.ACTION_LOSS_WEIGHT = 0.0

# ----------------------------------------
# LTContext (segmentation network E_theta)
# ----------------------------------------
_C.MODEL.LTC = CfgNode()

# Number of layers per stage.
_C.MODEL.LTC.NUM_LAYERS = 9

# Size of the window of the windowed attention (W).
_C.MODEL.LTC.WINDOWED_ATTN_W = 64

# Sparseness of the long-term context attention (G).
_C.MODEL.LTC.LONG_TERM_ATTN_G = 64

_C.MODEL.LTC.USE_INSTANCE_NORM = True

# Dilation factor of the convolutions: CONV_DILATION_FACTOR ** layer_idx.
_C.MODEL.LTC.CONV_DILATION_FACTOR = 2

_C.MODEL.LTC.NUM_STAGES = 4

_C.MODEL.LTC.MODEL_DIM = 64

# The hidden dimension of the stages after the first one is MODEL_DIM / DIM_REDUCTION.
_C.MODEL.LTC.DIM_REDUCTION = 2.0

# Probability of masking input channels.
_C.MODEL.LTC.CHANNEL_MASKING_PROB = 0.3

_C.MODEL.LTC.DROPOUT_PROB = 0.2

# Stages whose features are used by the sequence and action losses;
# a predictor head P_psi (MLP) is added on top of each selected stage.
_C.MODEL.LTC.PREDICTOR_HEAD_STAGES = [False, False, False, True]

_C.MODEL.ATTENTION = CfgNode()

_C.MODEL.ATTENTION.NUM_ATTN_HEADS = 1

_C.MODEL.ATTENTION.DROPOUT = 0.2

# ---------------------------------------------------------------------------- #
# Data options
# ---------------------------------------------------------------------------- #
_C.DATA = CfgNode()

# Folder that contains the features (see README).
_C.DATA.PATH_TO_DATA_DIR = ""

# Name of the feature folder, i.e. <PATH_TO_DATA_DIR>/features/<FEATURE_BACKBONE>.
_C.DATA.FEATURE_BACKBONE = "dinov2_vitl14_spatial"

# Temporal stride used to load the frame-wise features.
_C.DATA.FRAME_SAMPLING_RATE = 1

# Views that are not used for training (unseen views).
_C.DATA.UNSEEN_EXO_VIEWS = []
_C.DATA.UNSEEN_EGO_VIEWS = []

# Background class ids (only used for visualizing predictions).
_C.DATA.BACKGROUND_INDICES = [0]

# Fraction of the videos to use (for debugging).
_C.DATA.DATA_FRACTION = 1.0

# ---------------------------------------------------------------------------- #
# Optimizer options (Adam with linear warm-up and cosine decay)
# ---------------------------------------------------------------------------- #
_C.SOLVER = CfgNode()

_C.SOLVER.BASE_LR = 0.0008

_C.SOLVER.WEIGHT_DECAY = 5e-4

# Max L2 norm of the gradients (None: no clipping).
_C.SOLVER.CLIP_GRAD_L2NORM = 5.0

_C.SOLVER.MAX_EPOCH = 40

# Minimal learning rate of the cosine decay.
_C.SOLVER.ETA_MIN = 0.00005

# The warm-up starts from BASE_LR * WARMUP_FACTOR and lasts WARMUP_EPOCHS epochs.
_C.SOLVER.WARMUP_FACTOR = 0.5
_C.SOLVER.WARMUP_EPOCHS = 5

# T_max of the cosine decay.
_C.SOLVER.T_MAX = 60

# ---------------------------------------------------------------------------- #
# Data loader options
# ---------------------------------------------------------------------------- #
_C.DATA_LOADER = CfgNode()

_C.DATA_LOADER.NUM_WORKERS = 2

_C.DATA_LOADER.PIN_MEMORY = True

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #

# Number of GPUs (only single-GPU training was tested).
_C.NUM_GPUS = 1

# Output folder; the results of a config are stored in <OUTPUT_DIR>/<config name>[_<EXPR_NAME>].
_C.OUTPUT_DIR = "./experiments"

_C.EXPR_NAME = ''

_C.RNG_SEED = 1115

# Log period in iterations.
_C.LOG_PERIOD = 100

# Set at runtime.
_C.CONFIG_FILE = ""
_C.EXPR_NUM = -1
_C.SUMMARY_PATH = ""
_C.CONFIG_LOG_PATH = ""


def _assert_and_infer_cfg(cfg):
    import torch
    if cfg.NUM_GPUS > 0:
        gpu_count = torch.cuda.device_count()
        assert gpu_count >= cfg.NUM_GPUS, \
            f"Number of available gpus is {gpu_count} but {cfg.NUM_GPUS} is requested!"
    return cfg


def get_cfg():
    """
    Get a copy of the default config.
    """
    return _assert_and_infer_cfg(_C.clone())

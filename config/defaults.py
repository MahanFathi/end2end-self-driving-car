import os
from yacs.config import CfgNode as CN

# ---------------------------------------------------------------------------- #
# Define Config Node
# ---------------------------------------------------------------------------- #
_C = CN()

# ---------------------------------------------------------------------------- #
# Model Configs
# ---------------------------------------------------------------------------- #
_C.MODEL = CN()
_C.MODEL.META_ARCHITECTURE = 'PilotNet'
_C.MODEL.BACKWARD_META_ARCHITECTURE = 'BackwardPilotNet'
_C.MODEL.DEVICE = "cuda"
_C.MODEL.WEIGHTS = "./weights_final.pth"  # should be a path to pth or ckpt file

# ---------------------------------------------------------------------------- #
# __CNN Configs
# ---------------------------------------------------------------------------- #
_C.MODEL.CNN = CN()
_C.MODEL.CNN.INPUT_CHANNELS = 3
_C.MODEL.CNN.LAYERS = [     # Adhere to [out_channels, (kernel_h, kernel_w), stride]
    {'out_channels': 24, 'kernel': (5, 5), 'stride': 2, },
    {'out_channels': 36, 'kernel': (5, 5), 'stride': 2, },
    {'out_channels': 48, 'kernel': (5, 5), 'stride': 2, },
    {'out_channels': 64, 'kernel': (3, 3), 'stride': 1, },
    {'out_channels': 64, 'kernel': (3, 3), 'stride': 1, },
]
_C.MODEL.CNN.DROPOUT = 0.2
# ---------------------------------------------------------------------------- #
# __Fully Connected Output Layer
# ---------------------------------------------------------------------------- #
_C.MODEL.FC = CN()
_C.MODEL.FC.INPUT = 2304
_C.MODEL.FC.LAYERS = [             # Adhere to [out_channels, (kernel_h, kernel_w), stride]
    {'to_size': 100, 'dropout': .5, 'norm': False},
    {'to_size': 50, 'dropout': .5, 'norm': False},
    {'to_size': 10, 'dropout': .0, 'norm': False},
]

# ---------------------------------------------------------------------------- #
# __BACKWARD_CNN Configs
# ---------------------------------------------------------------------------- #
_C.MODEL.BACKWARD_CNN = CN()
_C.MODEL.BACKWARD_CNN.INPUT_CHANNELS = 1
_C.MODEL.BACKWARD_CNN.LAYERS = [     # Adhere to [out_channels, (kernel_h, kernel_w), stride]
    {'out_channels': 1, 'kernel': (3, 3), 'stride': 1, },
    {'out_channels': 1, 'kernel': (3, 3), 'stride': 1, },
    {'out_channels': 1, 'kernel': (5, 5), 'stride': 2, },
    {'out_channels': 1, 'kernel': (5, 6), 'stride': 2, },
    {'out_channels': 1, 'kernel': (6, 6), 'stride': 2, },
]

# ---------------------------------------------------------------------------- #
# Input Pipeline Configs
# ---------------------------------------------------------------------------- #
_C.INPUT = CN()
_C.INPUT.BATCH_SIZE = 256

# ---------------------------------------------------------------------------- #
# Datasets
# ---------------------------------------------------------------------------- #
_C.DATASETS = CN()
_C.DATASETS.FACTORY = 'DrivingDatasetDataset'               # datasets class
_C.DATASETS.DATA_DIR = './driving_dataset'                  # path to dataset
_C.DATASETS.ANN_TRAIN_PATH = './driving_dataset/train.csv'  # path to annotation csv
_C.DATASETS.ANN_VAL_PATH = './driving_dataset/val.csv'      # path to annotation csv
_C.DATASETS.ANN_VIS_PATH = './driving_dataset/vis.csv'      # path to annotation csv
_C.DATASETS.SHUFFLE = True                                  # load in shuffle fashion

# ---------------------------------------------------------------------------- #
# Image and Steer parameters
# ---------------------------------------------------------------------------- #
_C.IMAGE = CN()
_C.IMAGE.TARGET_HEIGHT = 70
_C.IMAGE.TARGET_WIDTH = 200
_C.IMAGE.CROP_HEIGHT = [70, 256]
_C.IMAGE.DO_AUGMENTATION = True
_C.IMAGE.AUGMENTATION_BRIGHTNESS_MIN = 0.2
_C.IMAGE.AUGMENTATION_BRIGHTNESS_MAX = 1.5
_C.IMAGE.AUGMENTATION_DELTA_CORRECTION = 15.
_C.IMAGE.MIRROR_BIAS = 0.4

_C.STEER = CN()
_C.STEER.AUGMENTATION_SIGMA = 1.    # yet to be determined

# ---------------------------------------------------------------------------- #
# Dataloader Configs
# ---------------------------------------------------------------------------- #
_C.DATALOADER = CN()
_C.DATALOADER.NUM_WORKERS = 0   # Number of data loading threads

# ---------------------------------------------------------------------------- #
# Solver Configs
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
_C.SOLVER.EPOCHS = 100

_C.SOLVER.BASE_LR = 0.0001
_C.SOLVER.BIAS_LR_FACTOR = 2

_C.SOLVER.MOMENTUM = 0.9

_C.SOLVER.WEIGHT_DECAY = 0.0005
_C.SOLVER.WEIGHT_DECAY_BIAS = 0

_C.SOLVER.GAMMA = 0.1
_C.SOLVER.STEPS = (30000,)

_C.SOLVER.CHECKPOINT_PERIOD = 2500
_C.SOLVER.CHECKPOINT_PATH = os.path.join('.', 'checkpoints')

# ---------------------------------------------------------------------------- #
# Output Configs
# ---------------------------------------------------------------------------- #
_C.OUTPUT = CN()
_C.OUTPUT.DIR = './output'
_C.OUTPUT.VIS_DIR = './output/visualization'


# ---------------------------------------------------------------------------- #
# Log Configs
# ---------------------------------------------------------------------------- #
_C.LOG = CN()
_C.LOG.PATH = './log'
_C.LOG.PERIOD = 100
_C.LOG.WEIGHTS_SAVE_PERIOD = 10000
_C.LOG.PLOT = CN()
_C.LOG.PLOT.DISPLAY_PORT = 8097
_C.LOG.PLOT.ITER_PERIOD = 1000  # effective plotting step is _C.LOG.PERIOD * LOG.PLOT.ITER_PERIOD


# ---------------------------------------------------------------------------- #
# Drive Configs
# ---------------------------------------------------------------------------- #
_C.DRIVE = CN()
_C.DRIVE.IMAGE = CN()
_C.DRIVE.IMAGE.CROP_HEIGHT = [240, 880]


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()

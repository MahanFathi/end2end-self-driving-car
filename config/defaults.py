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
_C.MODEL.DEVICE = "cuda"
_C.MODEL.WEIGHTS = ""  # should be a path to pth or ckpt file

# ---------------------------------------------------------------------------- #
# __CNN Configs
# ---------------------------------------------------------------------------- #
_C.MODEL.CNN = CN()
_C.MODEL.CNN.INPUT_CHANNELS = 3
_C.MODEL.CNN.LAYERS = [     # Adhere to [out_channels, (kernel_h, kernel_w), stride]
    {'out_channels': 24, 'kernel': (5, 5), 'stride': 5, 'pool_kernel': (2, 2)},
    {'out_channels': 36, 'kernel': (5, 5), 'stride': 5, 'pool_kernel': (2, 2)},
    {'out_channels': 48, 'kernel': (5, 5), 'stride': 5, 'pool_kernel': (2, 2)},
    {'out_channels': 64, 'kernel': (3, 3), 'stride': 3, 'pool_kernel': None},
    {'out_channels': 64, 'kernel': (3, 3), 'stride': 3, 'pool_kernel': None},
]
_C.MODEL.CNN.DROPOUT = 0.2
# ---------------------------------------------------------------------------- #
# __Fully Connected Output Layer
# ---------------------------------------------------------------------------- #
_C.MODEL.FC = CN()
_C.MODEL.FC.INPUT = 123412341234123412341234 # TODO
_C.MODEL.FC.LAYERS = [             # Adhere to [out_channels, (kernel_h, kernel_w), stride]
    {'to_size': 100, 'dropout': .5, 'norm': False},
    {'to_size': 50, 'dropout': .5, 'norm': False},
    {'to_size': 10, 'dropout': .0, 'norm': False},
]
_C.MODEL.FC.INPUT.LAYERS = []    # a list of hidden layer sizes for output fc. [] means no hidden
_C.MODEL.FC.INPUT.NORM_LAYERS = [0]     # should be a list of layer indices, example [0, 1, ...]
_C.MODEL.FC.INPUT.OUT_SIZE = 64     # output size of pre-sequential network, which is the input size of LSTM

# ---------------------------------------------------------------------------- #
# Input Pipeline Configs
# ---------------------------------------------------------------------------- #
_C.INPUT = CN()
_C.INPUT.BATCH_SIZE = 64

# ---------------------------------------------------------------------------- #
# Datasets
# ---------------------------------------------------------------------------- #
_C.DATASETS = CN()
_C.DATASETS.FACTORY = 'RouteLogsDataset'    # dataset class
_C.DATASETS.TRAIN_PATH = ''                 # path to dataset
_C.DATASETS.VAL_PATH = ''                   # path to dataset
_C.DATASETS.TEST_PATH = ''                  # path to dataset
_C.DATASETS.SHUFFLE = True                  # load in shuffle fashion

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

_C.SOLVER.BASE_LR = 0.001
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

# ---------------------------------------------------------------------------- #
# Log Configs
# ---------------------------------------------------------------------------- #
_C.LOG = CN()
_C.LOG.PERIOD = 100
_C.LOG.PLOT = CN()
_C.LOG.PLOT.DISPLAY_PORT = 8097
_C.LOG.PLOT.ITER_PERIOD = 1000  # effective plotting step is _C.LOG.PERIOD * LOG.PLOT.ITER_PERIOD


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()

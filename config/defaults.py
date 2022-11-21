from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Convention about Training / Test specific parameters
# -----------------------------------------------------------------------------
# Whenever an argument can be either used for training or for testing, the
# corresponding name will be post-fixed by a _TRAIN for a training parameter,
# or _TEST for a test-specific parameter.
# For example, the number of images during training will be
# IMAGES_PER_BATCH_TRAIN, while the number of images for testing will be
# IMAGES_PER_BATCH_TEST

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()

_C.MODEL = CN()
# Using cuda or cpu for training
_C.MODEL.DEVICE = "cuda:0"
# ID number of GPU
_C.MODEL.DEVICE_ID = '0'
# Name of backbone
_C.MODEL.NAME = 'resnet50'
# Last stride of backbone
_C.MODEL.LAST_STRIDE = 1
# Path to pretrained model of backbone
_C.MODEL.PRETRAIN_PATH = ''
_C.MODEL.PRETRAIN_DIR = '/data/sw_data/SecureReID/log/market1501/OI-PI-18'
# Use ImageNet pretrained model to initialize backbone or use self trained model to initialize the whole model
# Options: 'imagenet' or 'self', refer using pretrained resnet50 or pretrained re-id baseline respectively.
_C.MODEL.PRETRAIN_CHOICE = 'imagenet'
# If train with center loss, options: 'bnneck' or 'no'
_C.MODEL.CENTER_LOSS = 'on'
_C.MODEL.CENTER_FEAT_DIM = 2048
# If train with weighted regularized triplet loss, options: 'on', 'off'
_C.MODEL.WEIGHT_REGULARIZED_TRIPLET = 'off'
# If train with generalized mean pooling, options: 'on', 'off'
_C.MODEL.GENERALIZED_MEAN_POOL = 'off'

# gan mode 'the type of GAN objective. [vanilla] # [vanilla| lsgan | wgangp]. vanilla GAN loss is the cross-entropy objective used in the original GAN paper.'
_C.MODEL.GAN_MODE = 'vanilla'
# # of gen filters in the last conv layer
_C.MODEL.NGF = 64
# # of discrim filters in the first conv layer
_C.MODEL.NDF = 64
# specify discriminator architecture [basic | n_layers | pixel]. The basic model is a 70x70 PatchGAN. n_layers allows you to specify the layers in the discriminator
_C.MODEL.NETD = 'basic'
# specify generator architecture [resnet_9blocks | resnet_6blocks | unet_256 | unet_128]
_C.MODEL.NETG = 'unet_128'
# only used if netD==n_layers
_C.MODEL.N_LAYERS_D = 3
# instance normalization or batch normalization [batch] # [instance | batch | none]
_C.MODEL.NORM = 'batch'
# network initialization [normal | xavier | kaiming | orthogonal]
_C.MODEL.INIT_TYPE = 'normal'
# scaling factor for normal, xavier and orthogonal.
_C.MODEL.INIT_GAIN = 0.02
# no dropout for the generator
_C.MODEL.NO_DROPOUT = 'store_true'
# the size of image buffer that stores previously generated images
_C.MODEL.POOL_SIZE = 50
# which epoch to load? set to latest to use latest cached model
_C.MODEL.EPOCH = 'best'
# train mode options: 'C', 'GCR'
_C.MODEL.MODE = 'GCR'
# train input mode options: 'origin', 'protected', 'both'
_C.MODEL.OUTPUT_MODE = 'origin'
_C.MODEL.MI_PERIOD = 20
_C.MODEL.VAL_R1 = 0.7
_C.MODEL.PSNR = 0.5
_C.MODEL.SSIM = 0.05


# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()
# Size of the image
_C.INPUT.IMG_SIZE = [384, 128]
# Random probability for image horizontal flip
_C.INPUT.PROB = 0.0
# Random probability for random erasing
_C.INPUT.RE_PROB = 0.0
# Values to be used for image normalization
_C.INPUT.PIXEL_MEAN = [0.485, 0.456, 0.406]
# Values to be used for image normalization
_C.INPUT.PIXEL_STD = [0.229, 0.224, 0.225]
# Value of padding size
_C.INPUT.PADDING = 0
# direction
_C.INPUT.DIRECTION = 'AtoB'
_C.INPUT.TYPE = 'mosaic'
_C.INPUT.RADIUS = 24.0

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
# List of the dataset names for training, as present in paths_catalog.py
_C.DATASETS.NAMES = ('market1501')
# Root directory where datasets should be used (and downloaded if not found)
_C.DATASETS.ROOT_DIR = ('/data/market1501_val')

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 8
# If use PK sampler for data loading
_C.DATALOADER.PK_SAMPLER = 'on'
# Number of instance for one batch
_C.DATALOADER.NUM_INSTANCE = 16

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
# Name of optimizer
_C.SOLVER.OPTIMIZER_NAME = "Adam"
# Number of max epoches
_C.SOLVER.MAX_EPOCHS = 50
# Base learning rate
_C.SOLVER.BASE_LR = 3e-4
# Momentum
_C.SOLVER.MOMENTUM = 0.9
# Margin of triplet loss
_C.SOLVER.MARGIN = 0.3
# Learning rate of SGD to learn the centers of center loss
_C.SOLVER.CENTER_LR = 0.5
# Balanced weight of center loss
_C.SOLVER.CENTER_LOSS_WEIGHT = 0.0005

# Settings of weight decay
_C.SOLVER.WEIGHT_DECAY = 0.0005

# decay rate of learning rate
_C.SOLVER.GAMMA = 0.1
# decay step of learning rate
_C.SOLVER.STEPS = (30, 55)

# warm up factor
_C.SOLVER.WARMUP_FACTOR = 1.0 / 3
# iterations of warm up
_C.SOLVER.WARMUP_ITERS = 500
# method of warm up, option: 'constant','linear'
_C.SOLVER.WARMUP_METHOD = "linear"

# epoch number of saving checkpoints
_C.SOLVER.CHECKPOINT_PERIOD = 20
# iteration of display training log
_C.SOLVER.LOG_PERIOD = 100
# epoch number of validation
_C.SOLVER.EVAL_PERIOD = 10

_C.SOLVER.VAL_PERIOD = 5

# Number of images per batch
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_C.SOLVER.IMS_PER_BATCH = 64

# ---------------------------------------------------------------------------- #
# Test
# ---------------------------------------------------------------------------- #
_C.TEST = CN()
# Number of images per batch during test
_C.TEST.IMS_PER_BATCH = 128
# If test with re-ranking, options: 'on','off'
_C.TEST.RE_RANKING = 'off'
# Path to trained model
_C.TEST.WEIGHT = ""
# Whether feature is nomalized before test, if on, it is equivalent to cosine distance
_C.TEST.FEAT_NORM = 'on'
_C.TEST.EVALUATE_ONLY = 'off'
# Whether to evaluate on partial re-id dataset
_C.TEST.PARTIAL_REID = 'off'
# gallery type  options: 'origin', 'protected', 'both'
_C.TEST.GALLERY_MODE = 'both'
# recovery with key
_C.TEST.RECOVERY_WITH_KEY = True
# retrieve with key
_C.TEST.RETRIEVE_WITH_KEY = True
# query mode
_C.TEST.QUERY_MODE = 'both'

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
# Path to checkpoint and saved log of trained model
_C.OUTPUT_DIR = './log/market1501/OI-PI'

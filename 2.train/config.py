# Author: Bichen Wu (bichen@berkeley.edu) 08/25/2016

"""Base Model configurations"""

import os
import os.path as osp
import numpy as np
from easydict import EasyDict as edict

def base_model_config(dataset='MNIST'):
  assert dataset.upper()=='MNIST' or dataset.upper()=='KITTI', \
      'Currently only support PASCAL_VOC or KITTI dataset'

  cfg = edict()

  # Dataset used to train/val/test model. Now support PASCAL_VOC or KITTI
  cfg.DATASET = dataset.upper()

  if cfg.DATASET == 'MNIST':
    # object categories to classify
    cfg.CLASS_NAMES = ('one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten')
  elif cfg.DATASET == 'KITTI':
    cfg.CLASS_NAMES = ('car', 'pedestrian', 'cyclist')

  # number of categories to classify
  cfg.CLASSES = len(cfg.CLASS_NAMES)    

  # parameter used in leaky ReLU
  cfg.LEAKY_COEF = 0.1

  # Probability to keep a node in dropout
  cfg.KEEP_PROB = 0.5

  # image width
  cfg.IMAGE_WIDTH = 28

  # image height
  cfg.IMAGE_HEIGHT = 28

  #------------------------------------------------------------------
  	#tk (DECAY,LR) = (1000, 0.1), (1000, 0.01)
  #------------------------------------------------------------------
  # batch size
  cfg.BATCH_SIZE = 100			#original : 20

  # max # of steps for batch
  cfg.MAX_STEP = 10000 			# original : 100 0000 // good : 10000

  # check point step
  cfg.CHECKPOINT_STEP = 1000	# original : 1000

  # summary step
  cfg.SUMMARY_STEP = 10			# original : 10

  # reduce step size after this many steps
  cfg.DECAY_STEPS = 1000		#original : 10000

  # weight decay
  cfg.WEIGHT_DECAY = 0.001		# original : 0.0005

  #          ***********************************************

  # momentum
  cfg.MOMENTUM = 0.9			# original : 0.9

  # multiply the learning rate by this factor
  # decayed_lr  = lr * decay_factor^(global/decay_step)
  cfg.LR_DECAY_FACTOR = 0.009 	# original : 0.1 // good : 0.07

  # learning rate
  cfg.LEARNING_RATE = 0.016 		# original : 0.005 // good : 0.12
  #-------------------------------------------------------------------

  # wether to load pre-trained model
  cfg.LOAD_PRETRAINED_MODEL = True

  # path to load the pre-trained model
  cfg.PRETRAINED_MODEL_PATH = ''

  # a small value used to prevent numerical instability
  cfg.EPSILON = 1e-16

  # threshold for safe exponential operation
  cfg.EXP_THRESH=1.0

  # gradients with norm larger than this is going to be clipped.
  cfg.MAX_GRAD_NORM = 10.0 # original : 10.0

  # Whether to do data augmentation
  cfg.DATA_AUGMENTATION = False

  # small value used in batch normalization to prevent dividing by 0. The
  # default value here is the same with caffe's default value.
  cfg.BATCH_NORM_EPSILON = 1e-5

  # number of threads to fetch data
  # cfg.NUM_THREAD = 4 #error : DONT KNOWì WHY
  # tk: maybe because of (input stream)
  cfg.NUM_THREAD = 1

  # capacity for FIFOQueue
  cfg.QUEUE_CAPACITY = 100

  # indicate if the model is in training mode
  cfg.IS_TRAINING = False

  return cfg

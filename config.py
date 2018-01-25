import os
import os.path as osp
import numpy as np
from distutils import spawn
# `pip install easydict` if you don't have it
from easydict import EasyDict as edict

__C = edict()
cfg = __C

class Config():
  self.CNN_OBJ_MAXINPUT = 100.0
  self.EPS = 0.00000001
  self.REFTIMES = 8
  self.INLIERTHRESHOLD2D = 10
  self.INLIERCOUNT = 20
  self.SKIP = 10
  self..HYPNUM = 256

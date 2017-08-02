import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from Loss import NSNLLLoss
from Loss import DSSMLoss
import random
import pickle
import torch.optim as optim
from sklearn.neighbors import DistanceMetric
from sklearn.neighbors import BallTree


feature_cur = Variable(torch.ones(1, 4))
feature_cur2 = Variable(torch.ones(1, 4))
print feature_cur * 0.2 + feature_cur2 * 0.8

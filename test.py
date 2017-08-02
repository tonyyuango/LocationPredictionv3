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


feature_cur = Variable(torch.zeros(1, 4))
dist = 0.5
feature_cat = torch.cat((feature_cur, Variable(torch.FloatTensor([dist])).view(1, -1)), 1)
print feature_cat
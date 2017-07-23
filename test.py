import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from Loss import NSNLLLoss
import random
import pickle
import torch.optim as optim
from sklearn.neighbors import DistanceMetric
from sklearn.neighbors import BallTree
from Util import IndexLinear
import pandas as pd

# instances = pd.read_table( header=None)
f_data = open('/Users/quanyuan/Dropbox/Research/LocationPrediction/small/foursquare_test.txt', 'r')
lines = f_data.readlines()
f_data.close()
session_idx_tmp = [map(int, lines[i].split(',')) for i in range(0, len(lines), 5)]
session_idx = []
# print session_idx_tmp
print '\n'
for session_u in session_idx_tmp:
    session_idx.append([(session_u[i], session_u[i + 1]) for i in range(0, len(session_u), 2)])
for ll in session_idx:
    print ll
# vids = [map(int, lines[i].split(',')) for i in range(2, len(lines), 6)]
# tids = [map(int, lines[i].split(',')) for i in range(3, len(lines), 6)]
# vids_next = [map(int, lines[i].split(',')) for i in range(4, len(lines), 6)]
# tids_next = [map(int, lines[i].split(',')) for i in range(5, len(lines), 6)]
# print uids
# for ll in vids:
#     print ll
# str = '1 3 4 6 8'
# a = map(int, str.split())
# a = [1, 3, 4, 6, 8]
# for i in a:
#     print i, type(i)
#     print str(i)
# print [i for i in a]
# v = Variable(torch.LongTensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]))
# print v
# session_idx = [(0, 2), (2, 5)]
# print session_idx
# a = np.array([0.4, 0.6])
# print ','.join([str(coor) for coor in a])
# print v[0, session_idx[0][0]:session_idx[0][1]]
# print ','.join([str(int(i)) for i in a])

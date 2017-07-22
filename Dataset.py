import math
from collections import Counter

import numpy as np
import numpy.random as nprd
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

class CheckinData(Dataset):
    def __init__(self, data_file, ):
        pass
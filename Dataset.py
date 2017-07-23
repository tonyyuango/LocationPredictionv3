import math
from collections import Counter

import numpy as np
import numpy.random as nprd
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

class CheckinData(Dataset):
    def __init__(self, data_path):
        f_data = open(data_path, 'r')
        lines = f_data.readlines()
        f_data.close()
        uid_session_idx_tmp = [map(int, lines[i].split(',')) for i in range(0, len(lines), 5)]
        self.uid_session_idx = []
        for session_u in uid_session_idx_tmp:
            self.uid_session_idx.append([(session_u[i], session_u[i + 1]) for i in range(0, len(session_u), 2)])
        self.uid_vids = [map(int, lines[i].split(',')) for i in range(1, len(lines), 5)]
        self.uid_tids = [map(int, lines[i].split(',')) for i in range(2, len(lines), 5)]
        self.uid_vids_next = [map(int, lines[i].split(',')) for i in range(3, len(lines), 5)]
        self.uid_tids_next = [map(int, lines[i].split(',')) for i in range(4, len(lines), 5)]
        self.max_seq_len = self.get_max_seq_len()

    def to_string(self):
        for uid, l in enumerate(self.uid_vids_next):
            print uid, l

    def get_max_seq_len(self):
        max_len = 0
        for seq in self.uid_vids:
            max_len = max((max_len, len(seq)))
        return max_len

    def __len__(self):
        return len(self.uid_vids)

    def __getitem__(self, uid):
        vids = np.zeros(self.max_seq_len, dtype=np.int)
        tids = np.zeros(self.max_seq_len, dtype=np.int)
        vids_next = np.zeros(self.max_seq_len, dtype=np.int)
        tids_next = np.zeros(self.max_seq_len, dtype=np.int)
        mask = np.ones(self.max_seq_len, dtype=np.int)
        length = len(self.uid_vids[uid])
        for i in range(length):
            vids[i] = self.uid_vids[uid][i]
            tids[i] = self.uid_tids[uid][i]
            vids_next[i] = self.uid_vids_next[uid][i]
            tids_next[i] = self.uid_tids_next[uid][i]
            mask[i] = 0
        return torch.LongTensor(self.uid_session_idx[uid]), torch.from_numpy(vids), torch.from_numpy(tids), torch.from_numpy(vids_next), torch.from_numpy(tids_next), torch.LongTensor([length]), torch.from_numpy(mask).byte()

class Vocabulary:
    def __init__(self, data_file):
        self.id_name = {}
        self.name_id = {}
        with open(data_file, 'r') as fin:
            for line in fin:
                al = line.strip().split(',')
                id = int(al[1])
                name = al[0]
                self.id_name[id] = name
                self.name_id[name] = id

    def size(self):
        return len(self.id_name)


class DataSet:
    def __init__(self, opt):
        u_vocab_file = opt['u_vocab_file']
        v_vocab_file = opt['v_vocab_file']
        train_file = opt['train_data_file']
        test_file = opt['test_data_file']
        batch_size = opt['batch_size']
        n_worker = opt['data_worker']
        print batch_size
        print n_worker
        self.u_vocab = Vocabulary(u_vocab_file)
        self.v_vocab = Vocabulary(v_vocab_file)
        train_data = CheckinData(train_file)
        test_data = CheckinData(test_file)
        self.train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=n_worker)
        self.test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=n_worker)


root_path = '/Users/quanyuan/Dropbox/Research/LocationPrediction/small/'
dataset_name = 'foursquare'
opt = {'u_vocab_file': root_path + dataset_name + '_u.txt',
       'v_vocab_file': root_path + dataset_name + '_v.txt',
       'train_data_file': root_path + dataset_name + '_train.txt',
       'test_data_file': root_path + dataset_name + '_test.txt',
       'batch_size': 1,
       'data_worker': 1}
dataset = DataSet(opt)
train_data = dataset.train_loader
for i, data_batch in enumerate(train_data):
    print i
    print data_batch
    raw_input()
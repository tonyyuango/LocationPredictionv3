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


class SpatioTemporalModel(nn.Module):
    def __init__(self, u_size, v_size, t_size, emb_dim_u=32, emb_dim_v=32, emb_dim_t=16, hidden_dim=32, nb_cnt=100, sampling_list=None, vid_coor_rad=None, vid_pop=None, dropout=0.5):
        super(SpatioTemporalModel, self).__init__()
        self.emb_dim_u = emb_dim_u
        self.emb_dim_v = emb_dim_v
        self.emb_dim_t = emb_dim_t
        self.hidden_dim = hidden_dim
        self.u_size = u_size
        self.v_size = v_size
        self.t_size = t_size
        self.nb_cnt = nb_cnt
        self.dropout = dropout
        self.sampling_list = sampling_list
        self.vid_coor_rad = vid_coor_rad
        self.vid_pop = vid_pop
        self.tree = BallTree(vid_coor_rad.values(), leaf_size=40, metric='haversine')
        self.dist_metric = DistanceMetric.get_metric('haversine')
        self.uid_rid_sampling_info = {}
        for uid in range(0, u_size):
            self.uid_rid_sampling_info[uid] = {}

        self.rnn_short = nn.RNNCell(self.emb_dim_v, self.hidden_dim) #TODO check GRU
        # self.rnn_short = nn.GRUCell(self.emb_dim_v, self.hidden_dim) #TODO check GRU
        self.rnn_long = nn.GRUCell(self.emb_dim_v, self.hidden_dim)
        self.embedder_u = nn.Embedding(self.u_size, self.emb_dim_u)
        self.embedder_v = nn.Embedding(self.v_size, self.emb_dim_v)
        self.embedder_t = nn.Embedding(self.t_size, self.emb_dim_t)
        dim_merged = self.hidden_dim * 2 + self.emb_dim_u + self.emb_dim_t
        self.decoder = IndexLinear(dim_merged, v_size)

    def forward(self, records_u, is_train, mod=0):
        predicted_scores = Variable(torch.zeros(records_u.get_predicting_records_cnt(mod=0), self.nb_cnt + 1)) if is_train else []
        rid_vids_true = []
        rid_vids = []
        vids_visited = set()

        records_al = records_u.get_records(mod=0) if is_train else records_u.get_records(mod=2)
        emb_u = self.embedder_u(Variable(torch.LongTensor([records_u.uid])).view(1, -1)).view(1, -1)
        hidden_long = self.init_hidden()
        idx = 0
        for rid, record in enumerate(records_al):
            if record.is_first:
                hidden_short = self.init_hidden()
            vids_visited.add(record.vid)
            emb_v = self.embedder_v(Variable(torch.LongTensor([record.vid])).view(1, -1)).view(1, -1)
            emb_t_next = self.embedder_t(Variable(torch.LongTensor([record.tid_next])).view(1, -1)).view(1, -1)
            hidden_long = self.rnn_long(emb_v, hidden_long)
            hidden_short = self.rnn_short(emb_v, hidden_short)
            if record.is_last:
                continue

            hidden = torch.cat((hidden_long.view(1, -1), hidden_short.view(1, -1), emb_u.view(1, -1), emb_t_next.view(1, -1)), 1)
            if is_train:
                rid_vids_true.append(record.vid_next)
                vid_candidates = self.get_vids_candidate(records_u.uid, rid, record.vid_next, vids_visited, True, False if mod == 0 else True)
            else:
                if rid >= records_u.test_idx:
                    rid_vids_true.append(record.vid_next)
                    vid_candidates = self.get_vids_candidate(records_u.uid, rid, record.vid_next, vids_visited, False, False if mod == 0 else True)
                    predicted_scores.append([])
                else:
                    continue
            output = self.decoder(hidden, Variable(torch.LongTensor(vid_candidates)).view(1, -1))
            predicted_scores[idx] = F.sigmoid(output) if is_train else F.softmax(output)
            rid_vids.append(vid_candidates)
            idx += 1
        return predicted_scores, rid_vids, rid_vids_true

    def get_vids_candidate(self, uid, rid, vid_true=None, vids_visited=None, is_train=True, use_distance=True):
        if not use_distance:
            if is_train:
                vid_candidates = [vid_true]
                while len(vid_candidates) <= self.nb_cnt:
                    vid_candidate = self.sampling_list[random.randint(0, len(self.sampling_list) - 1)]
                    if vid_candidate != vid_true:
                        vid_candidates.append(vid_candidate)
                return vid_candidates
            else:
                return range(self.v_size)
        else:
            if rid in self.uid_rid_sampling_info[uid]:
                vids, probs = self.uid_rid_sampling_info[uid][rid]
            else:
                nbs = set()
                for vid_visited in vids_visited:
                    vids = self.tree.query_radius([self.vid_coor_rad[vid_visited]], r=0.000172657)
                    for vid in vids[0]:
                        if (not is_train) or (is_train and vid != vid_true):
                            nbs.add(vid)
                vids = list(nbs)
                probs = np.array([self.vid_pop[vid] for vid in vids], dtype=np.float64)
                probs /= probs.sum()
                self.uid_rid_sampling_info[uid][rid] = (vids, probs)
            if is_train:
                id_cnt = np.random.multinomial(self.nb_cnt, probs)
                vid_candidates = [vid_true]
                for id, cnt in enumerate(id_cnt):
                    for _ in range(cnt):
                        vid_candidates.append(vids[id])
                return vid_candidates
            else:
                return vids

    def init_hidden(self):
        return Variable(torch.zeros(1, self.hidden_dim))


def train(root_path, dataset, n_iter=500, iter_start=0, mod=0):
    torch.manual_seed(0)
    random.seed(0)
    dl = pickle.load(open(root_path + 'dl_' + dataset + '.pk', 'rb'))
    model = SpatioTemporalModel(dl.nu, dl.nv, dl.nt, sampling_list=dl.sampling_list, vid_coor_rad=dl.vid_coor_rad, vid_pop=dl.vid_pop)
    if iter_start != 0:
        model.load_state_dict(torch.load(root_path + 'model_decoder_' + str(mod) + '_' + str(iter_start) + '.md'))
    optimizer = optim.Adam(model.parameters())
    criterion = NSNLLLoss()
    uids = dl.uid_records.keys()
    for iter in range(iter_start + 1, n_iter + 1):
        print_loss_total = 0
        random.shuffle(uids)
        for idx, uid in enumerate(uids):
            records_u = dl.uid_records[uid]
            optimizer.zero_grad()
            predicted_probs, _, _ = model(records_u, is_train=True, mod=mod)
            loss = criterion(predicted_probs)
            loss.backward()
            print_loss_total += loss.data[0]
            optimizer.step()
            if idx % 100 == 0:
                print 'uid: \t%d\tloss: %f' % (idx, print_loss_total)
        print iter, print_loss_total
        if iter % 5 == 0:
            torch.save(model.state_dict(), root_path + 'model_decoder_' + str(mod) + '_' + str(iter) + '.md')

def test(root_path, dataset, iter_start=0, mod=0):
    torch.manual_seed(0)
    random.seed(0)
    dl = pickle.load(open(root_path + 'dl_' + dataset + '.pk', 'rb'))
    for iter in range(iter_start, 0, -5):
        hit = 0
        # print root_path + 'model_simple_' + str(mod) + '_' + str(iter) + '.md'
        model = SpatioTemporalModel(dl.nu, dl.nv, dl.nt, sampling_list=dl.sampling_list, vid_coor_rad=dl.vid_coor_rad, vid_pop=dl.vid_pop)
        if iter_start != 0:
            model.load_state_dict(
                torch.load(root_path + 'model_decoder_' + str(mod) + '_' + str(iter) + '.md'))
        hits = np.zeros(3)
        cnt = 0
        for uid, records_u in dl.uid_records.items():
            id_scores, id_vids, vids_true = model(records_u, is_train=False, mod=mod)
            # print 'id_scores: ', id_scores
            # print 'id_vids: ', id_vids
            # print 'vids_true: ', vids_true
            # print ''

            for idx in range(len(id_vids)):
                probs_sorted, vid_sorted = torch.sort(id_scores[idx].view(-1), 0, descending=True)
                # print 'probs_sorted: ', probs_sorted
                # print '\tvid_sorted: ', vid_sorted[0:20]
                vid_ranked = [id_vids[idx][id] for id in vid_sorted.data]
                # print '\tvid_true: ', vids_true[idx]
                # print '\tvid_ranked: ', vid_ranked[0:20]
                # raw_input()
                # print vids_true[idx]
                # for vid in id_vids[idx]:
                #     print vid, vids_true[idx]
                #     if vid == vids_true[idx]:
                #         hit += 1
                #         break
                # raw_input()
                cnt += 1
                # if uid >= 477:
                #     print vid_ranked
                #     print vids_true
                #     print min(len(vid_ranked), 10)
                for j in range(min(len(vid_ranked), 10)):
                    # if uid >= 477:
                    #     print j
                    #     print vids_true[idx]
                    #     print vid_ranked[j]
                    if vids_true[idx] == vid_ranked[j]:
                        if j == 0:
                            hits[0] += 1
                        if j < 5:
                            hits[1] += 1
                        if j < 10:
                            hits[2] += 1
            if (uid + 1) % 100 == 0:
                print (uid + 1), hits / cnt
            # raw_input()
        hits /= cnt
        print 'iter:', iter, 'hits: ', hits, 'cnt: ', cnt
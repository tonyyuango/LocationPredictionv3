import torch
import os
import pickle
import attention_model_enhance
import SimpleModelDecoder

def analyze_session_len(root_path):
    dl = pickle.load(open(root_path + 'dl_' + dataset + '.pk', 'rb'))
    len_cnt = {}
    for _, records_u in dl.uid_records.items():
        for record in records_u.records:
            if record.is_first:
                len = 0
            len += 1
            if record.is_last:
                if len not in len_cnt:
                    len_cnt[len] = 0
                len_cnt[len] += 1
    for len in len_cnt:
        print len, len_cnt[len]
    raw_input()

if __name__ == "__main__":
    torch.set_num_threads(1)
    torch.manual_seed(0)
    root_path = '/Users/quanyuan/Dropbox/Research/LocationData/' \
        if os.path.exists('/Users/quanyuan/Dropbox/Research/LocationData/') \
        else '/shared/data/qyuan/LocationData/'
    small_path = root_path + 'small/'
    dataset = 'foursquare'
    task = int(input('please input task (0: train, 1: test, 2: baselines): '))
    model = int(input('please input model (0: our, 1: decoder): '))
    mod = int(input("input mod: "))
    iter = int(input('please input last iter: '))
    if task == 0:
        if model == 0:
            attention_model_enhance.train(small_path, dataset, iter_start=iter, mod=mod)
        elif model == 1:
            SimpleModelDecoder.train(small_path, dataset, iter_start=iter, mod=mod)
    else:
        if model == 0:
            attention_model_enhance.test(small_path, dataset, iter_start=iter, mod=mod)
        elif model == 1:
            SimpleModelDecoder.test(small_path, dataset, iter_start=iter, mod=mod)
import torch
import os
import DataStructure
import FileProcessing as fp
from DataStructure import DataLoader
import DSSM
import SimpleModelDecoder
import NCF

def prepare_data(root_path, small_path=None, u_cnt_max = -1, task=0):
    if task == 0:
        for data_set in ['gowalla', 'brightkite']:
            fp.process_raw_file(root_path + 'loc-' + data_set + '_totalCheckins.txt', root_path + data_set + '_total.txt',
                                u_freq_min=5, u_vcnt_min=5, lat_min=40.4774, lat_max=40.9176, lng_min=-74.2589,
                                lng_max=-73.7004)
            fp.generate_session_file(root_path + data_set + '_total.txt', root_path + data_set + '_session.txt')
    if task == 1:
        for data_set in ['gowalla', 'brightkite']:
            dl = DataLoader(hour_gap=6)
            dl.add_records(root_path + data_set + '_session.txt', small_path + 'dl_' + data_set + '.pk', u_cnt_max)
            f = open(root_path + 'blacklist_' + data_set + '.txt', 'w', -1)
            for uid, records_u in dl.uid_records.items():
                cnt_train = records_u.get_predicting_records_cnt(mod=0)
                cnt_test = records_u.get_predicting_records_cnt(mod=1)
                if cnt_train == 0 or cnt_test == 0:
                    f.write(dl.uid_u[uid] + '\n')
            f.close()
    if task == 2:
        for data_set in ['gowalla', 'brightkite']:
            blacklist = set()
            f = open(root_path + 'blacklist_' + data_set + '.txt', 'r', -1)
            for l in f:
                blacklist.add(l.strip())
            f.close()
            dl = DataLoader(hour_gap=6)
            dl.add_records(root_path + data_set + '_session.txt', small_path + 'dl_' + data_set + '.pk', u_cnt_max=u_cnt_max,
                           blacklist=blacklist)
            dl.show_info()
        # for uid, records_u in dl.uid_records.items():
        #     for rid, record in enumerate(dl.uid_records[uid].records):
        #         record.peek()
        #         if rid < dl.uid_records[uid].test_idx:
        #             print "train"
        #         else:
        #             print "test"



if __name__ == "__main__":
    torch.set_num_threads(2)
    torch.manual_seed(0)
    root_path = '/Users/quanyuan/Dropbox/Research/LocationPrediction/' \
        if os.path.exists('/Users/quanyuan/Dropbox/Research/LocationPrediction/') \
        else '/shared/data/qyuan/LocationPrediction/'
    # prepare_data(root_path, root_path + 'small/', task=0)
    # prepare_data(root_path, root_path + 'small/', task=1)
    # prepare_data(root_path, root_path + 'small/', u_cnt_max=300, task=2)
    # prepare_data(root_path, root_path + 'full/', u_cnt_max=-1, task=2)
    # raw_input()
    dataset = 'gowalla'
    dir = 0
    task = 0
    mod = 0
    submod = 1
    iter = 0
    # dir = int(input('please dir (0: small, 1: full): '))
    if dir == 0:
        small_path = root_path + 'small/'
    elif dir == 1:
        small_path = root_path + 'full/'
    task = int(input('please input task (0: train, 1: test, 2: baselines): '))
    mod = int(input(
        'please input mod (0: NCF, 1: Decoder, 2: DSSM): '))
    submod = int(input('please input sub mod: '))
    iter = int(input('please input last iter: '))
    if task == 0:
        if mod == 0:
            NCF.train(small_path, dataset, iter_start=iter, mod=submod)
        elif mod == 1:
            SimpleModelDecoder.train(small_path, dataset, iter_start=iter, mod=submod)
        elif mod == 2:
            DSSM.train(small_path, dataset, iter_start=iter, mod=submod)
    else:
        if mod == 0:
            NCF.test(small_path, dataset, iter_start=iter, mod=submod)
        elif mod == 1:
            SimpleModelDecoder.test(small_path, dataset, iter_start=iter, mod=submod)
        elif mod == 2:
            DSSM.test(small_path, dataset, iter_start=iter, mod=submod)

import torch
import os
import DataStructure
import FileProcessing as fp
from DataStructure import DataLoader
import DSSM
import SimpleModelDecoder
import NCF

def prepare_data(root_path, small_path, u_cnt_max = -1):
    # fp.reverse_raw_file(root_path + 'Gowalla_totalCheckins.txt', root_path + 'checkins_total.txt', u_freq_min=10, v_freq_min=15)
    # fp.generate_session_file(root_path + 'checkins_total_gowalla.txt', root_path + 'checkins_session_gowalla.txt')
    # fp.generate_session_file(root_path + 'checkins_total_4sq.txt', root_path + 'checkins_session_4sq.txt')

    data_set = 'gowalla'
    dl = DataLoader(hour_gap=6)
    blacklist = set()
    f = open(root_path + 'blacklist_' + data_set + '.txt', 'r', -1)
    for l in f:
        blacklist.add(l.strip())
    f.close()
    # print blacklist
    dl.add_records(root_path + 'checkins_session_' + data_set + '.txt', small_path + 'dl_' + data_set + '.pk',
                   root_path + 'glove.twitter.27B.50d.txt',
                   small_path + 'glove.txt', u_cnt_max, blacklist=blacklist)
    # f = open(root_path + 'blacklist_' + data_set + '.txt', 'w', -1)
    # for uid, records_u in dl.uid_records.items():
    #     cnt_train = records_u.get_predicting_records_cnt(mod=0)
    #     cnt_test = records_u.get_predicting_records_cnt(mod=1)
    #     if cnt_train == 0 or cnt_test == 0:
    #         f.write(dl.uid_u[uid] + '\n')
    # f.close()

    # print u_invalid
    # for uid, records_u in dl.uid_records.items():
    #     for rid, record in enumerate(dl.uid_records[uid].records):
    #         record.peek()
    #         if rid < dl.uid_records[uid].test_idx:
    #             print "train"
    #         else:
    #             print "test"
    dl.show_info()

if __name__ == "__main__":
    torch.set_num_threads(2)
    torch.manual_seed(0)
    root_path = '/Users/quanyuan/Dropbox/Research/LocationPrediction/' \
        if os.path.exists('/Users/quanyuan/Dropbox/Research/LocationPrediction/') \
        else '/shared/data/qyuan/LocationPrediction/'
    # prepare_data(root_path, root_path + 'small/', 500)
    # prepare_data(root_path, root_path + 'medium/', 2000)
    # prepare_data(root_path, root_path + 'larege/', 10000)
    # prepare_data(root_path, root_path + 'full/', -1)
    dataset = '4sq'
    dir = int(input('please dir (0: small, 1: medium, 2: large): '))
    if dir == 0:
        small_path = root_path + 'small/'
    elif dir == 1:
        small_path = root_path + 'medium/'
    else:
        small_path = root_path + 'large/'

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

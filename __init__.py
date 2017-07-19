import torch
import os
import DataStructure
from DataStructure import DataLoader
import SimpleModel

def prepare_data(root_path, small_path, u_cnt_max = 500):
    # DataStructure.reverse_raw_file(root_path + 'Gowalla_totalCheckins.txt', root_path + 'checkins_total.txt', u_freq_min=10, v_freq_min=15)
    dl = DataLoader(hour_gap=6)
    dl.add_records(root_path + 'checkins_session_4sq.txt', small_path + 'dl.pk',
                   '/Users/quanyuan/Dropbox/Research/Spatial/glove.twitter.27B.50d.txt',
                   small_path + 'glove.txt', u_cnt_max)
    for uid in dl.uid_records:
        for rid, record in enumerate(dl.uid_records[uid].records):
            record.peek()
            # if rid < dl.uid_records[uid].test_idx:
            #     print "train"
            # else:
            #     print "test"
    dl.show_info()

if __name__ == "__main__":
    torch.set_num_threads(2)
    torch.manual_seed(0)
    root_path = '/Users/quanyuan/Dropbox/Research/LocationPrediction/' \
        if os.path.exists('/Users/quanyuan/Dropbox/Research/LocationPrediction/') \
        else '/shared/data/qyuan/'
    small_path = root_path + 'small/'
    # prepare_data(root_path, small_path, 500)
    SimpleModel.train(small_path, iter_start=0)

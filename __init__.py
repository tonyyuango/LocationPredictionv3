import torch
import os
import attention_model

if __name__ == "__main__":
    torch.set_num_threads(2)
    torch.manual_seed(0)
    root_path = '/Users/quanyuan/Dropbox/Research/LocationData/' \
        if os.path.exists('/Users/quanyuan/Dropbox/Research/LocationData/') \
        else '/shared/data/qyuan/LocationData/'
    small_path = root_path + 'small/'
    dataset = 'foursquare'
    task = int(input('please input task (0: train, 1: test, 2: baselines): '))
    mod = int(input("input mod: "))
    iter = int(input('please input last iter: '))
    if task == 0:
        attention_model.train(small_path, dataset, iter_start=iter, mod=mod)
    else:
        attention_model.test(small_path, dataset, iter_start=iter, mod=mod)
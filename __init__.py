import torch
import os
import attention_model
import SimpleModelDecoder

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
            attention_model.train(small_path, dataset, iter_start=iter, mod=mod)
        elif model == 1:
            SimpleModelDecoder.train(small_path, dataset, iter_start=iter, mod=mod)
    else:
        if model == 0:
            attention_model.test(small_path, dataset, iter_start=iter, mod=mod)
        elif model == 1:
            SimpleModelDecoder.test(small_path, dataset, iter_start=iter, mod=mod)
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
    mod = int(input("input mod: \n"))
    attention_model.train(small_path, dataset, mod=mod)
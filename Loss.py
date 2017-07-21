import torch
import torch.nn as nn

class DSSMLoss(nn.Module):
    def __init__(self):
        super(DSSMLoss, self).__init__()

    def forward(self, predicted_probs, size_average=True):
        loss = torch.log(predicted_probs).sum().neg()
        if size_average:
            loss /= predicted_probs.size(0)
        return loss

class NSNLLLoss(nn.Module):
    def __init__(self):
        super(NSNLLLoss, self).__init__()
    def forward(self, predicted_probs, size_average=True):
        loss = torch.sum(torch.log(predicted_probs[:, 0])) + torch.sum(torch.log(1 - predicted_probs[:, 1:]))
        if size_average:
            loss /= predicted_probs.size(0)
        return loss.neg()

class BPRLoss(nn.Module):
    def __init__(self):
        super(BPRLoss, self).__init__()
    def forward(self, predicted_probs, size_average=True):
        loss = torch.sum(torch.log(1 + torch.exp(predicted_probs.neg())))
        if size_average:
            loss /= predicted_probs.size(0)
        return loss
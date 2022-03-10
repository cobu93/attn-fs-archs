import torch
import torch.nn as nn

class BasePreprocessor(nn.Module):
    def __init__(self):
        super(BasePreprocessor, self).__init__()

    def forward(self, src):
        raise NotImplementedError("This feature hasn't been implemented yet!")

class IdentityPreprocessor(BasePreprocessor):

    def forward(self, src):
        return src

class CLSPreprocessor(BasePreprocessor):
    def forward(self, src):
        #src with shape [batch_size, seq_len]
        device = src.get_device()
        return torch.cat((torch.zeros(src.shape[0],1).long().to(device), src), dim=1)
        #src with shape [batch_size, seq_len+1]
import torch
import torch.nn as nn

class BaseAggregator(nn.Module):
    def __init__(self, output_size):
        super(BaseAggregator, self).__init__()
        self.output_size = output_size

    def forward(self, src):
        raise NotImplementedError("This feature hasn't been implemented yet!")

class ConcatenateAggregator(BaseAggregator):
    def forward(self, src):
        return torch.flatten(src, start_dim=1)

class SumAggregator(BaseAggregator):
    def forward(self, src):
        return torch.sum(src, dim=1, keepdim=False)

class CLSAggregator(BaseAggregator):
    def forward(self, src):
        #src with shape [batch_size, seq_len, num_features]
        return src[:,0,:]
        #src with shape [batch_size, num_features]
import torch.nn as nn

class BaseAttentionAggregator(nn.Module):
    def __init__(self):
        super(BaseAttentionAggregator, self).__init__()
        
    def forward(self, src):
        raise NotImplementedError("This feature hasn't been implemented yet!")

class NaiveAttentionAggregator(BaseAttentionAggregator):
    def forward(self, src):
        instances = src.shape[0]
        src = src.mean(dim=1).reshape((instances, -1))
        return src

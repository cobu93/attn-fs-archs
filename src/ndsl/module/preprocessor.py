import torch.nn as nn

class BasePreprocessor(nn.Module):
    def __init__(self):
        super(BasePreprocessor, self).__init__()

    def forward(self, src):
        raise NotImplementedError("This feature hasn't been implemented yet!")

class IdentityPreprocessor(BasePreprocessor):

    def forward(self, src):
        return src
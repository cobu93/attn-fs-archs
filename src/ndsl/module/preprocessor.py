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
    def __init__(self, embed_dim):
        super(CLSPreprocessor, self).__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

    def forward(self, src):
        #src with shape [batch_size, seq_len]
        tokens = self.cls_token.repeat(src.shape[0], 1, 1)
        return torch.cat((tokens, src), dim=1)
        #src with shape [batch_size, seq_len+1]
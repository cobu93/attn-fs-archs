import torch.nn as nn
import torch.nn.functional as F

class FeatureEncoder(nn.Module):
    def __init__(self, output_size):
        super(FeatureEncoder, self).__init__()
        self.output_size = output_size
    
    def forward(self, src):
        raise NotImplementedError("This feature hasn't been implemented yet!")


class CategoricalOneHotEncoder(FeatureEncoder):
    def __init__(self, output_size, n_labels, eps=1e-15):
        super(CategoricalOneHotEncoder, self).__init__(output_size)
        self.eps = eps
        self.output_size = output_size
        self.n_labels = n_labels + 1
        self.embedding = nn.Linear(self.n_labels, output_size)

    def forward(self, src):
        src = F.one_hot(src.squeeze().long() % self.n_labels, num_classes=self.n_labels).float()
        out = self.embedding(src)
        return out / out.sum()+ self.eps

class NumericalEncoder(FeatureEncoder):
    def __init__(self, output_size):
        super(NumericalEncoder, self).__init__(output_size)
        self.output_size = output_size
        self.embedding = nn.utils.weight_norm(nn.Linear(1, output_size))

    def forward(self, src):
        return self.embedding(src)
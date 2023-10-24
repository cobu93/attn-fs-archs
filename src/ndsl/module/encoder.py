import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.chi2 import Chi2

class FeatureEncoder(nn.Module):
    def __init__(self, output_size):
        super(FeatureEncoder, self).__init__()
        self.output_size = output_size
    
    def forward(self, src):
        raise NotImplementedError("This feature hasn't been implemented yet!")


class CategoricalOneHotEncoder(FeatureEncoder):
    def __init__(self, output_size, n_labels, eps=1e-15, include_nan=True):
        super(CategoricalOneHotEncoder, self).__init__(output_size)
        self.eps = eps
        self.output_size = output_size
        if include_nan:
            self.n_labels = n_labels + 1
        else:
            self.n_labels = n_labels
        self.embedding = nn.Linear(self.n_labels, output_size)

    def forward(self, src): 
        src = F.one_hot(src.squeeze().long(), num_classes=self.n_labels).float()
        out = self.embedding(src)
        return out / out.sum(dim=-1, keepdim=True) + self.eps

class NumericalEncoder(FeatureEncoder):
    def __init__(self, output_size, n_numerical):
        super(NumericalEncoder, self).__init__(output_size)
        self.output_size = output_size
        self.weights = nn.Parameter(torch.randn(n_numerical, output_size))
        self.biases = nn.Parameter(torch.randn(n_numerical, output_size))
        
    def forward(self, src):
        output = src.unsqueeze(-1) * self.weights + self.biases
        return output
    

class FundamentalEmbeddingsEncoder(FeatureEncoder):
    def __init__(self, output_size, n_fundamentals, n_splits, n_samples=32, aggregation="max"):
        super(FundamentalEmbeddingsEncoder, self).__init__(output_size)
        self.output_size = output_size

        self.fundamentals = nn.Parameter(torch.rand(n_fundamentals, output_size) * 2 - 1)
        self.n_fundamentals = n_fundamentals
        self.n_splits = n_splits - 1
        self.n_samples = n_samples
        self.aggregation = aggregation

        assert self.aggregation in ["max", "mean", "sum"], "Fundamentals aggregation method not valid"


        self.offset = int(n_fundamentals/self.n_splits)
        assert self.offset >= 1, f"The number of splits {n_splits} is too large for the number of fundamentals {n_fundamentals}"
    
    def forward(self, src):

        assert torch.all(src >= 1), "The features coding must start in 1"

        dof = torch.clip(src, 1, self.n_splits)
        fundamental_indices = Chi2(dof).sample((self.n_samples,)) * self.offset
        fundamental_indices = torch.clip(fundamental_indices, 0, self.n_fundamentals - 1).long()
        embeddings = self.fundamentals[fundamental_indices]
        
        if self.aggregation == "mean":
            embeddings = embeddings.mean(dim=0)
        elif self.aggregation == "sum":
            embeddings = embeddings.sum(dim=0)
        elif self.aggregation == "max":
            embeddings = embeddings.max(dim=0)[0]
        else:
            raise ValueError("Fundamentals aggregation method not valid")

        return embeddings

import torch
import torch.nn as nn
import torch.nn.functional as F
from ndsl.utils.builder import build_mlp

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
    
class MLPEncoder(FeatureEncoder):
    def __init__(self, 
                 output_size, 
                 input_size, 
                 hidden_sizes=[128], 
                 activations=[nn.ReLU(), nn.Identity()]
        ):

        super(MLPEncoder, self).__init__(output_size)

        self.shared_mlp = build_mlp(
            input_size, 
            output_size,
            hidden_sizes, 
            activations
        )
        
    def forward(self, src):
        batch_size, n_features = src.shape
        output = self.shared_mlp(src.view(batch_size * n_features, 1))
        output = output.view(batch_size, n_features, self.output_size)
        return output
    
class CategoricalEncoder(FeatureEncoder):
    def __init__(self, output_size, n_categories, variational=True):
        super(CategoricalEncoder, self).__init__(output_size)
        self.output_size = output_size
        self.variational = variational

        self.categories_means = nn.Parameter(torch.randn(n_categories, output_size))
        self.categories_logvars = nn.Parameter(torch.randn(n_categories, output_size))
    
    def forward(self, src):

        means = self.categories_means[src]
        norm_means = (means / torch.norm(means, dim=-1, keepdim=True))

        if self.training and self.variational:
            logvars = self.categories_logvars[src]
            z = torch.randn(*logvars.shape).to(src.device)
            embeddings = (z * torch.exp(0.5 * logvars)) + norm_means
            embeddings = embeddings / torch.norm(embeddings, dim=-1, keepdim=True)
        else:
            embeddings = norm_means
                
        return embeddings

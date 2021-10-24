import torch
import torch.nn as nn

class CELWithWeightsEntropyMinimization(nn.CrossEntropyLoss):
    def __init__(self, *args, beta_1=1, beta_2=1, **kwargs):
        super(CELWithWeightsEntropyMinimization, self).__init__(*args, **kwargs)
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    def forward(self, input, target, weights):
        # Cross-entropy loss
        loss = super().forward(input, target)

        # Entropy for each row
        weights_row_entropy = -torch.sum(weights * torch.log(weights + 1e-13), dim=-2)
        weights_row_entropy = torch.mean(weights_row_entropy)

        # Entropy for each column
        column_norm_weights = weights / weights.sum(dim=-1, keepdims=True)
        weights_columns_entropy = -torch.sum(column_norm_weights * torch.log(column_norm_weights + 1e-13), dim=-1)
        weights_columns_entropy = torch.mean(weights_columns_entropy)

        loss = loss + self.beta_1 * weights_row_entropy + self.beta_2 * weights_columns_entropy
        return loss


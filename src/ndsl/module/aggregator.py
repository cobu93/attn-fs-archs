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

class MeanAggregator(BaseAggregator):
    def forward(self, src):
        return torch.mean(src, dim=1, keepdim=False)

class MaxAggregator(BaseAggregator):
    def forward(self, src):
        return torch.max(src, dim=1, keepdim=False)[0]

class CLSAggregator(BaseAggregator):
    def forward(self, src):
        #src with shape [batch_size, seq_len, num_features]
        return src[:, 0]
        #src with shape [batch_size, num_features]

class RNNAggregator(BaseAggregator):
    def __init__(self, output_size, cell, input_size, num_layers, dropout):
        super(RNNAggregator, self).__init__(output_size)
        self.output_size = output_size

        if cell == 'GRU':
            self.rnn = nn.GRU(input_size, output_size, num_layers,
                            batch_first=True, dropout=dropout)
        elif cell == 'LSTM':
            self.rnn = nn.LSTM(input_size, output_size, num_layers,
                            batch_first=True, dropout=dropout)
        else:
            raise TypeError("{} is not a valid cell, try with 'LSTM' or 'GRU'.".format(cell))

    def forward(self, src):
        #src with shape [batch_size, seq_len, num_features]
        output, _ = self.rnn(src)
        #output: [batch_size, seq_len, hidden_size]
        return output[:, -1, :]
        #src with shape [batch_size, num_features]
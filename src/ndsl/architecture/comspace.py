import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtext.nn import MultiheadAttentionContainer, InProjContainer, ScaledDotProduct
from torch import Tensor
from typing import Optional


from ndsl.module.encoder import NumericalEncoder
from ndsl.module.preprocessor import CLSPreprocessor
from ndsl.module.aggregator import ConcatenateAggregator, CLSAggregator, MaxAggregator, MeanAggregator, SumAggregator, RNNAggregator
from ndsl.architecture.attention import TTransformerEncoderLayer, TTransformerEncoder

class CommonSpaceTransformer(nn.Module):
    
    def __init__(
        self, 
        n_categories, # Number of maximum categories
        n_bins, # Number of bins for numerical features
        n_head, # Number of heads per layer
        n_hid, # Size of the MLP inside each transformer encoder layer
        n_layers, # Number of transformer encoder layers    
        n_output, # The number of output neurons
        embed_dim,
        numerical_mean=0,
        numerical_std=1,
        n_times_std=3,
        attn_dropout=0., # Used dropout,
        ff_dropout=0., # Used dropout
        aggregator=None, # The aggregator for output vectors before decoder
        rnn_aggregator_parameters=None,
        decoder_hidden_units=None,
        decoder_activation_fn=None,
        need_weights=False
        ):


        super(CommonSpaceTransformer, self).__init__()

        assert n_bins % 2 == 0, "n_bins must be even"

        self.__need_weights = need_weights

        # Building transformer encoder
        encoder_layers = TTransformerEncoderLayer(embed_dim, n_head, n_hid, attn_dropout=attn_dropout, ff_dropout=ff_dropout)
        self.transformer_encoder = TTransformerEncoder(encoder_layers, n_layers, need_weights=self.__need_weights)

        self.n_head = n_head
        self.n_hid = n_hid

        self.n_bins = n_bins
        self.n_categories = n_categories

        self.numerical_mean = numerical_mean
        self.numerical_std = numerical_std
        self.n_times_std = n_times_std
        self.bin_width = 2 * self.n_times_std / self.n_bins

        self.embeddings_preprocessor = None

        # The default aggregator will be CLSAggregator
        if aggregator is None or aggregator == "cls":
            self.aggregator = CLSAggregator(embed_dim)
            self.embeddings_preprocessor = CLSPreprocessor(embed_dim)
        elif aggregator == "max":
            self.aggregator = MaxAggregator(embed_dim)
        elif aggregator == "mean":
            self.aggregator = MeanAggregator(embed_dim)
        elif aggregator == "sum":
            self.aggregator = SumAggregator(embed_dim)
        elif aggregator == "rnn":
            if rnn_aggregator_parameters is None:
                raise ValueError("The aggregator 'rnn' requires 'rnn_aggregator_parameters' not null.")
            self.aggregator = RNNAggregator(input_size=embed_dim, **rnn_aggregator_parameters)
        else:
            raise ValueError(f"The aggregator '{aggregator}' is not valid.")

        
        # The extra element will be for missed categories
        self.embeeding_table = nn.Embedding(self.n_bins + self.n_categories + 1, embed_dim)
        
        #self.decoder = nn.Linear(self.aggregator.output_size + self.n_numerical_features, n_output)
        input_size = self.aggregator.output_size
        if decoder_hidden_units is not None:
            
            decoder_layers  = []
            for decoder_units in decoder_hidden_units:
                decoder_layers.append(nn.Linear(input_size, decoder_units)) 
                input_size = decoder_units

                if decoder_activation_fn is not None:
                    decoder_layers.append(decoder_activation_fn)

            decoder_layers.append(nn.Linear(input_size, n_output))  
            self.decoder = nn.Sequential(*decoder_layers)
        else:
            self.decoder = nn.Linear(self.aggregator.output_size, n_output)
        
    @property
    def need_weights(self):
        return self.__need_weights

    @need_weights.setter
    def need_weights(self, new_need_weights):
        self.__need_weights = new_need_weights
        self.transformer_encoder.need_weights = self.__need_weights

    def forward(self, x_categorical, x_numerical):
        ###################
        # Binning numerical
        ###################

        # Normalizing
        x_numerical = (x_numerical - self.numerical_mean) / self.numerical_std
        # Clipping outliers ans assign its bin
        x_numerical = torch.clip(
                        x_numerical, 
                        min=-self.n_times_std, 
                        max=self.n_times_std
                    ) // self.bin_width
        # Moving bins tostart in zero
        x_numerical += (self.n_bins // 2)
        x_categorical += self.n_bins + 1

        x = torch.cat([x_categorical, x_numerical], dim=1).to(torch.long)

        # src came with two dims: (batch_size, num_features)
        embeddings = self.embeeding_table(x)
                
        output = None

        if self.embeddings_preprocessor is not None:
            embeddings = self.embeddings_preprocessor(embeddings)

        if self.__need_weights:
            output, layer_outs, weights = self.transformer_encoder(embeddings)
        else:
            output = self.transformer_encoder(embeddings)

        # Aggregation of encoded vectors
        output = self.aggregator(output)

        # Decoding
        output = self.decoder(output)

        if self.__need_weights:
            return output.squeeze(dim=-1), layer_outs, weights

        return output.squeeze(dim=-1)
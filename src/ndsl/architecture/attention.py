import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtext.nn import MultiheadAttentionContainer, InProjContainer, ScaledDotProduct
from torch import Tensor
from typing import Optional


from ndsl.module.encoder import NumericalEncoder
from ndsl.module.preprocessor import CLSPreprocessor
from ndsl.module.aggregator import ConcatenateAggregator, CLSAggregator, MaxAggregator, MeanAggregator, SumAggregator, RNNAggregator

"""
TTransformerEncoderLayer

Custom transformer layer which return attention cubes(weights)

"""
class TTransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, n_head, n_hid, attn_dropout=0., ff_dropout=0.):
        super(TTransformerEncoderLayer, self).__init__()
        
        in_proj_container = InProjContainer(
                                torch.nn.Linear(embed_dim, embed_dim),
                                torch.nn.Linear(embed_dim, embed_dim),
                                torch.nn.Linear(embed_dim, embed_dim)
                            )

        self.pre_norm_1 = nn.LayerNorm(embed_dim)
        self.pre_norm_2 = nn.LayerNorm(embed_dim)
        
        self.self_attn = MultiheadAttentionContainer(
                            n_head,
                            in_proj_container,
                            ScaledDotProduct(dropout=attn_dropout),
                            torch.nn.Linear(embed_dim, embed_dim)
                        )

        self.ff_network = nn.Sequential(
            nn.Linear(embed_dim, n_hid),
            nn.ReLU(),
            nn.Dropout(ff_dropout),
            nn.Linear(n_hid, embed_dim)
        )

    def forward(self, 
                src: Tensor, 
                src_mask: Optional[Tensor] = None
            ) -> Tensor:
            
            src2 = self.pre_norm_1(src)
            src2, weights = self.self_attn(src2, src2, src2, attn_mask=src_mask)
            src = src + src2

            src2 = self.pre_norm_2(src)
            src2 = self.ff_network(src2)
            src = src + src2

            if self.self_attn.batch_first:
                batch_size = src.shape[-3]
                num_features = src.shape[-2]
            else:
                batch_size = src.shape[-2]
                num_features = src.shape[-3]

            weights = weights.reshape((batch_size, -1, num_features, num_features))

            return src, weights


"""
TTransformerEncoder

Custom transformer encoder which return attention cubes (weights)

"""

class TTransformerEncoder(nn.TransformerEncoder):
    
    def __init__(self, *args, need_weights=False, **kwargs):
        super(TTransformerEncoder, self).__init__(*args, **kwargs)
        self.need_weights = need_weights
        
    def forward(
                self, 
                src: Tensor, 
                mask: Optional[Tensor] = None, 
                src_key_padding_mask: Optional[Tensor] = None
            ) -> Tensor:
        
        output = src
        # At the end of the loop it will have a size of:
        # [num_layers, batch, number of heads, number of features, number of features]
        stacked_weights = []

        for mod in self.layers:
            output, weights = mod(output, src_mask=mask)

            if self.need_weights:
                stacked_weights.append(weights)

        if self.norm is not None:
            output = self.norm(output)

        if self.need_weights:
            return output, torch.stack(stacked_weights)

        return output

class TabularTransformer(nn.Module):
    
    def __init__(
        self, 
        n_categories, # List of number of categories
        n_numerical, # Number of numerical features
        n_head, # Number of heads per layer
        n_hid, # Size of the MLP inside each transformer encoder layer
        n_layers, # Number of transformer encoder layers    
        n_output, # The number of output neurons
        embed_dim,
        attn_dropout=0., # Used dropout,
        ff_dropout=0., # Used dropout
        aggregator=None, # The aggregator for output vectors before decoder
        rnn_aggregator_parameters=None,
        decoder_hidden_units=None,
        decoder_activation_fn=None,
        need_weights=False,
        numerical_passthrough=False
        ):


        super(TabularTransformer, self).__init__()

        self.numerical_passthrough = numerical_passthrough

        self.n_numerical_features = 0

        self.n_numerical = n_numerical
        self.numerical_encoder = None

        if self.numerical_passthrough:
            self.n_numerical_features = self.n_numerical
        else:
            self.numerical_encoder = NumericalEncoder(embed_dim, self.n_numerical)
            
        self.__need_weights = need_weights

        # Building transformer encoder
        encoder_layers = TTransformerEncoderLayer(embed_dim, n_head, n_hid, attn_dropout=attn_dropout, ff_dropout=ff_dropout)
        self.transformer_encoder = TTransformerEncoder(encoder_layers, n_layers, need_weights=self.__need_weights)

        self.n_head = n_head
        self.n_hid = n_hid

        self.n_categories = list(n_categories)

        self.categorical_preprocessor = None
        #self.numerical_preprocessor = None

        # The default aggregator will be ConcatenateAggregator
        if aggregator is None or aggregator == "concatenate":
            self.aggregator = ConcatenateAggregator(
                embed_dim * (len(n_categories) + (0 if not self.numerical_encoder else self.n_numerical))
            )
        elif aggregator == "cls":
            self.aggregator = CLSAggregator(embed_dim)
            self.n_categories = [1] + self.n_categories  
            self.categorical_preprocessor = CLSPreprocessor()
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

        if self.numerical_passthrough:
            self.numerical_layer_norm = nn.LayerNorm(self.n_numerical_features)

        # Adding 1 for a nan on each column
        self.n_categories = torch.IntTensor([-1] + self.n_categories) + 1
        self.categorical_encoder = nn.Embedding(self.n_categories.sum().item(), embed_dim)
        
        #self.decoder = nn.Linear(self.aggregator.output_size + self.n_numerical_features, n_output)
        input_size = self.aggregator.output_size + self.n_numerical_features
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
            self.decoder = nn.Linear(self.aggregator.output_size + self.n_numerical_features, n_output)
        
    @property
    def need_weights(self):
        return self.__need_weights

    @need_weights.setter
    def need_weights(self, new_need_weights):
        self.__need_weights = new_need_weights
        self.transformer_encoder.need_weights = self.__need_weights

    def forward(self, x_categorical, x_numerical):

        # Preprocess source if needed
        #if self.numerical_preprocessor is not None:
        #    x_numerical = self.numerical_preprocessor(x_numerical)

        if self.categorical_preprocessor is not None:
            x_categorical = self.categorical_preprocessor(x_categorical)

        # src came with two dims: (batch_size, num_features)
        self.n_categories = self.n_categories.to(x_categorical.device)
        embeddings = self.categorical_encoder(x_categorical + self.n_categories.cumsum(dim=-1)[:-1])

        numerical_embedding = []
        if self.numerical_passthrough:
            numerical_embedding = self.numerical_layer_norm(x_numerical)
        else:
            encodings = self.numerical_encoder(x_numerical)
            embeddings = torch.cat([embeddings, encodings], dim=1)
                
        # Encodes through transformer encoder
        # Due transpose, the output will be in format (batch, num_features, embedding_size)
        output = None

        if len(embeddings) > 0:
            if self.__need_weights:
                output, weights = self.transformer_encoder(embeddings)
            else:
                output = self.transformer_encoder(embeddings)

            # Aggregation of encoded vectors
            output = self.aggregator(output)

        if len(numerical_embedding) > 0:
            if output is not None:
                output = torch.cat([output, numerical_embedding], dim=-1)
            else:
                output = numerical_embedding

        # Decoding
        output = self.decoder(output)

        if self.__need_weights:
            return output.squeeze(dim=-1), weights

        return output.squeeze(dim=-1)
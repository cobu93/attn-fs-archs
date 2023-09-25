import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtext.nn import MultiheadAttentionContainer, InProjContainer, ScaledDotProduct
from torch import Tensor
from typing import Optional


from ndsl.module.encoder import NumericalEncoder
from ndsl.module.preprocessor import CLSPreprocessor
from ndsl.module.aggregator import ConcatenateAggregator, CLSAggregator, MaxAggregator, MeanAggregator, SumAggregator, RNNAggregator, LearnableAggregator

"""
TTransformerEncoderLayer

Custom transformer layer which return attention cubes(weights)

"""
class FeatureSelectorTransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, n_head, n_hid, n_queries, aggregator=None, attn_dropout=0., ff_dropout=0.):
        super(FeatureSelectorTransformerEncoderLayer, self).__init__()
        
        in_proj_container = InProjContainer(
                                torch.nn.Linear(embed_dim, embed_dim),
                                torch.nn.Linear(embed_dim, embed_dim),
                                torch.nn.Linear(embed_dim, embed_dim)
                            )

        self.pre_norm_1 = nn.LayerNorm(embed_dim)
        self.pre_norm_2 = nn.LayerNorm(embed_dim)
        
        self.attn = MultiheadAttentionContainer(
                            n_head,
                            in_proj_container,
                            ScaledDotProduct(dropout=attn_dropout),
                            torch.nn.Linear(embed_dim, embed_dim),
                            batch_first=True
                        )

        self.ff_network = nn.Sequential(
            nn.Linear(embed_dim, n_hid),
            nn.ReLU(),
            nn.Dropout(ff_dropout),
            nn.Linear(n_hid, embed_dim)
        )

        #if aggregator is None or aggregator == "concatenate":
        #    # This isn't the correct dimension
        #    self.aggregator = ConcatenateAggregator(embed_dim)
        if aggregator is None or aggregator == "max":
            self.aggregator = MaxAggregator(embed_dim)
        elif aggregator == "mean":
            self.aggregator = MeanAggregator(embed_dim)
        elif aggregator == "sum":
            self.aggregator = SumAggregator(embed_dim)
        elif aggregator == "learnable":
            self.aggregator = LearnableAggregator(embed_dim)
        else:
            raise ValueError(f"The aggregator '{aggregator}' is not valid.")

        
        self.q_weights = nn.Parameter(torch.randn(n_queries, embed_dim))
        self.q_biases = nn.Parameter(torch.randn(n_queries, embed_dim))

        self.n_queries = n_queries

        # Delete
        self.n_head = n_head


    def forward(self, 
                src: Tensor, 
                src_mask: Optional[Tensor] = None
            ) -> Tensor:
            
            if self.attn.batch_first:
                batch_size = src.shape[-3]
                num_features = src.shape[-2]
            else:
                batch_size = src.shape[-2]
                num_features = src.shape[-3]

            src2 = self.pre_norm_1(src)
            
            queries = self.aggregator(src2)

            if self.attn.batch_first:
                queries = queries.unsqueeze(1)
            else:
                queries = queries.unsqueeze(0)

            queries = self.q_weights * queries + self.q_biases
            
            src2, weights = self.attn(queries, src2, src2, attn_mask=src_mask)
            #src = src + src2
            src = src2

            src2 = self.pre_norm_2(src)
            src2 = self.ff_network(src2)
            src = src + src2
            
            weights = weights.reshape((batch_size, self.n_head, self.n_queries, num_features))

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
        stacked_outputs = []

        if self.need_weights:
            stacked_outputs.append(src)

        for mod in self.layers:
            output, weights = mod(output, src_mask=mask)

            if self.need_weights:
                stacked_weights.append(weights)
                stacked_outputs.append(output)

        if self.norm is not None:
            output = self.norm(output)

        if self.need_weights:
            return output, torch.stack(stacked_outputs), torch.stack(stacked_weights)

        return output


class FeatureSelectorTransformerEncoder(nn.Module):
    
    def __init__(self, encoder_layers, *args, need_weights=False, **kwargs):
        super(FeatureSelectorTransformerEncoder, self).__init__(*args, **kwargs)
        self.layers = nn.ModuleList(encoder_layers)
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
        stacked_outputs = []

        if self.need_weights:
            stacked_outputs.append(src)

        for mod in self.layers:
            output, weights = mod(output, src_mask=mask)

            if self.need_weights:
                stacked_weights.append(weights)
                stacked_outputs.append(output)

        if self.need_weights:
            return output, torch.stack(stacked_outputs), torch.stack(stacked_weights)

        return output


class FeatureSelectorTransformer(nn.Module):
    
    def __init__(
        self, 
        n_categories, # List of number of categories
        n_numerical, # Number of numerical features
        n_head, # Number of heads per layer
        n_hid, # Size of the MLP inside each transformer encoder layer
        n_layers, # Number of transformer encoder layers    
        n_output, # The number of output neurons
        embed_dim,
        n_queries,
        queries_agg=None,
        attn_dropout=0., # Used dropout,
        ff_dropout=0., # Used dropout
        aggregator=None, # The aggregator for output vectors before decoder
        rnn_aggregator_parameters=None,
        decoder_hidden_units=None,
        decoder_activation_fn=None,
        need_weights=False,
        ):


        super(FeatureSelectorTransformer, self).__init__()

        # Building the numerical encoder
        self.n_numerical_features = 0
        self.n_numerical = n_numerical
        self.numerical_encoder = NumericalEncoder(embed_dim, self.n_numerical)

        # Building categorical encoder
        self.n_categories = list(n_categories)
        self.n_categories = torch.IntTensor([-1] + self.n_categories) + 1
        categories_offset = self.n_categories.cumsum(dim=-1)[:-1]
        self.register_buffer("categories_offset", categories_offset)
        self.categorical_encoder = nn.Embedding(self.n_categories.sum().item(), embed_dim, norm_type=1.)
            
        self.__need_weights = need_weights

        # Building transformer encoder
        encoder_layers = []
        for n_query in n_queries:
            encoder_layers.append(FeatureSelectorTransformerEncoderLayer(
                                    embed_dim, 
                                    n_head, 
                                    n_hid, 
                                    n_query, 
                                    queries_agg, 
                                    attn_dropout=attn_dropout, 
                                    ff_dropout=ff_dropout
                                ))
        self.transformer_encoder = FeatureSelectorTransformerEncoder(encoder_layers, need_weights=self.__need_weights)

        self.n_head = n_head
        self.n_hid = n_hid

        self.embeddings_preprocessor = None

        # The default aggregator will be ConcatenateAggregator
        if aggregator is None or aggregator == "concatenate":
            self.aggregator = ConcatenateAggregator(
                embed_dim * n_queries[-1]
            )
        elif aggregator == "cls":
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
            self.decoder = nn.Linear(input_size, n_output)
        
    @property
    def need_weights(self):
        return self.__need_weights

    @need_weights.setter
    def need_weights(self, new_need_weights):
        self.__need_weights = new_need_weights
        self.transformer_encoder.need_weights = self.__need_weights

    def forward(self, x_categorical, x_numerical):

        # src came with two dims: (batch_size, num_features)
        embeddings = self.categorical_encoder(x_categorical + self.categories_offset)

        numerical_embedding = []
        encodings = self.numerical_encoder(x_numerical)
        embeddings = torch.cat([embeddings, encodings], dim=1)
                
        # Encodes through transformer encoder
        # Due transpose, the output will be in format (batch, num_features, embedding_size)
        output = None

        if len(embeddings) > 0:

            if self.embeddings_preprocessor is not None:
                embeddings = self.embeddings_preprocessor(embeddings)

            if self.__need_weights:
                output, layer_outs, weights = self.transformer_encoder(embeddings)
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
            return output.squeeze(dim=-1), layer_outs, weights

        return output.squeeze(dim=-1)
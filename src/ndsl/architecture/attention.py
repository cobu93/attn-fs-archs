import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtext.nn import MultiheadAttentionContainer, InProjContainer, ScaledDotProduct
from torch import Tensor
from typing import Optional


from ndsl.module.encoder import NumericalEncoder, CategoricalOneHotEncoder
from ndsl.module.preprocessor import BasePreprocessor
from ndsl.module.aggregator import BaseAggregator, ConcatenateAggregator

"""
TTransformerEncoderLayer

Custom transformer layer which return attention cubes(weights)

"""
class TTransformerEncoderLayer(nn.TransformerEncoderLayer):
    def __init__(self, *args, **kwargs):
        super(TTransformerEncoderLayer, self).__init__(*args, **kwargs)
        embed_dim = args[0] # d_model
        self.num_heads = args[1] # nhead
        dropout =  args[3] if len(args) > 3 else kwargs["dropout"]

        in_proj_container = InProjContainer(
                                torch.nn.Linear(embed_dim, embed_dim),
                                torch.nn.Linear(embed_dim, embed_dim),
                                torch.nn.Linear(embed_dim, embed_dim)
                            )

        self.self_attn = MultiheadAttentionContainer(
                            self.num_heads,
                            in_proj_container,
                            ScaledDotProduct(dropout=dropout),
                            nn.Linear(embed_dim, embed_dim)
                        )

    def forward(self, 
                src: Tensor, 
                src_mask: Optional[Tensor] = None
            ) -> Tensor:

            src2, weights = self.self_attn(src, src, src, attn_mask=src_mask)
            src = src + self.dropout1(src2)
            src = self.norm1(src)
                                    
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
            src = src + self.dropout2(src2)
            src = self.norm2(src)

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
        dropout=0.1, # Used dropout
        aggregator=None, # The aggregator for output vectors before decoder
        decoder_hidden_units=None,
        decoder_activation_fn=None,
        categorical_preprocessor=None,
        numerical_preprocessor=None,
        need_weights=False,
        numerical_passthrough=False
        ):


        super(TabularTransformer, self).__init__()

        self.numerical_passthrough = numerical_passthrough

        self.n_numerical_features = 0

        self.categorical_encoders = nn.ModuleList()
        
        for n_cats in n_categories:
            self.categorical_encoders.append(CategoricalOneHotEncoder(embed_dim, n_cats))

        self.n_numerical = n_numerical
        self.numerical_encoders = nn.ModuleList()

        if self.numerical_passthrough:
            self.n_numerical_features = self.n_numerical
        else:
            for _ in range(n_numerical):
                self.numerical_encoders.append(NumericalEncoder(embed_dim))
            
        self.__need_weights = need_weights

        # Building transformer encoder
        encoder_layers = TTransformerEncoderLayer(embed_dim, n_head, n_hid, dropout=dropout)
        self.transformer_encoder = TTransformerEncoder(encoder_layers, n_layers, need_weights=self.__need_weights)

        self.n_head = n_head
        self.n_hid = n_hid
        self.dropout = dropout

        # The default aggregator will be ConcatenateAggregator
        if aggregator is None:
            self.aggregator = ConcatenateAggregator(
                embed_dim * (len(self.categorical_encoders) + len(self.numerical_encoders))
            )
        else:
            self.aggregator = aggregator

        if categorical_preprocessor is not None:
            if not issubclass(type(categorical_preprocessor), BasePreprocessor):
                raise TypeError("Categorical preprocessor must inherit from BasePreprocessor")
        
        self.categorical_preprocessor = categorical_preprocessor

        if numerical_preprocessor is not None:
            if not issubclass(type(numerical_preprocessor), BasePreprocessor):
                raise TypeError("Numerical preprocessor must inherit from BasePreprocessor")
        
        self.numerical_preprocessor = numerical_preprocessor

        # Validates that aggregator inherit from BaseAggregator
        if not issubclass(type(self.aggregator), BaseAggregator):
            raise TypeError("Parameter aggregator must inherit from BaseAggregator")

        if self.numerical_passthrough:
            self.numerical_layer_norm = nn.LayerNorm(self.n_numerical_features)

        #self.decoder = nn.Linear(self.aggregator.output_size + self.n_numerical_features, n_output)
        input_size = self.aggregator.output_size + self.n_numerical_features
        if decoder_hidden_units is not None:
            
            decoder_layers  = []
            for decoder_layer_idx, decoder_units in enumerate(decoder_hidden_units):
                decoder_layers.append(nn.Linear(input_size, decoder_units)) 
                input_size = decoder_units

                # Check if last layer
                if decoder_layer_idx == len(decoder_hidden_units) - 1:
                    continue

                if decoder_activation_fn is not None:
                    try:
                        decoder_layers.append(decoder_activation_fn[decoder_layer_idx])
                    except:
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
        if self.numerical_preprocessor is not None:
            x_numerical = self.numerical_preprocessor(x_numerical)

        
        if self.categorical_preprocessor is not None:
            x_numerical = self.categorical_preprocessor(x_categorical)
        
        # src came with two dims: (batch_size, num_features)
        embeddings = []
        numerical_embedding = []

        # Computes embeddings for each feature
        for ft_idx, encoder in enumerate(self.categorical_encoders):
            # Each encoder must return a two dims tensor (batch, embedding_size)
            encoding = encoder(x_categorical[:, ft_idx].unsqueeze(1))
            embeddings.append(encoding)

        if self.numerical_passthrough:
            numerical_embedding = self.numerical_layer_norm(x_numerical)
        else:
            for ft_idx, encoder in enumerate(self.numerical_encoders):
                encoding = encoder(x_numerical[:, ft_idx].unsqueeze(1))
                embeddings.append(encoding)
                
        # embeddings has 3 dimensions (num_features, batch, embedding_size)
        if len(embeddings) > 0:
            embeddings = torch.stack(embeddings)

        # Encodes through transformer encoder
        # Due transpose, the output will be in format (batch, num_features, embedding_size)
        output = None

        if len(embeddings) > 0:
            if self.__need_weights:
                output, weights = self.transformer_encoder(embeddings)
                output = output.transpose(0, 1)
            else:
                output = self.transformer_encoder(embeddings).transpose(0, 1)
        
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
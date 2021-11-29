import torch
import torch.nn as nn
from torchtext.nn import MultiheadAttentionContainer, InProjContainer, ScaledDotProduct
from torch import Tensor
from typing import Optional


from ndsl.module.encoder import FeatureEncoder
from ndsl.module.preprocessor import BasePreprocessor
from ndsl.module.aggregator import BaseAggregator, ConcatenateAggregator
from ndsl.module.attention_aggregator import BaseAttentionAggregator, NaiveAttentionAggregator

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
                            torch.nn.Linear(embed_dim, embed_dim)
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
        n_head, # Number of heads per layer
        n_hid, # Size of the MLP inside each transformer encoder layer
        n_layers, # Number of transformer encoder layers    
        n_output, # The number of output neurons
        encoders, # List of features encoders
        dropout=0.1, # Used dropout
        aggregator=None, # The aggregator for output vectors before decoder
        preprocessor=None,
        need_weights=False
        ):


        super(TabularTransformer, self).__init__()

        # Verify that encoders are correct
        if not isinstance(encoders, nn.ModuleList):
            raise TypeError("Parameter encoders must be an instance of torch.nn.ModuleList")

        # Embedding size
        self.n_input = None

        for idx, encoder in enumerate(encoders):
            
            if not issubclass(type(encoder), FeatureEncoder):
                raise TypeError("All encoders must inherit from FeatureEncoder. Invalid index {}".format(idx))

            if self.n_input is None:
                self.n_input = encoder.output_size
            elif self.n_input != encoder.output_size:
                raise ValueError("All encoders must have the same output")

        self.encoders = encoders
        self.__need_weights = need_weights

        # Building transformer encoder
        encoder_layers = TTransformerEncoderLayer(self.n_input, n_head, n_hid, dropout=dropout)
        self.transformer_encoder = TTransformerEncoder(encoder_layers, n_layers, need_weights=self.__need_weights)

        self.n_head = n_head
        self.n_hid = n_hid
        self.dropout = dropout

        # The default aggregator will be ConcatenateAggregator
        if aggregator is None:
            self.aggregator = ConcatenateAggregator(self.n_input * len(self.encoders))
        else:
            self.aggregator = aggregator

        # Validates that aggregator inherit from BaseAggregator
        if not issubclass(type(self.aggregator), BaseAggregator):
            raise TypeError("Parameter aggregator must inherit from BaseAggregator")

        self.preprocessor = preprocessor

        if self.preprocessor is not None:
            if not issubclass(type(self.preprocessor), BasePreprocessor):
                    raise TypeError("Preprocessor must inherit from BasePreprocessor.")


        self.decoder = nn.Linear(self.aggregator.output_size, n_output)

    @property
    def need_weights(self):
        return self.__need_weights

    @need_weights.setter
    def need_weights(self, new_need_weights):
        self.__need_weights = new_need_weights
        self.transformer_encoder.need_weights = self.__need_weights

    def forward(self, src):

        # Preprocess source if needed
        if self.preprocessor is not None:
            src = self.preprocessor(src)

        
        # Validate than src features and num of encoders is the same
        if src.size()[1] != len(self.encoders):
            raise ValueError("The number of features must be the same as the number of encoders.\
                 Got {} features and {} encoders".format(src.size()[1], len(self.encoders)))

        # src came with two dims: (batch_size, num_features)
        embeddings = []

        # Computes embeddings for each feature
        for ft_idx, encoder in enumerate(self.encoders):
            # Each encoder must return a two dims tensor (batch, embedding_size)
            encoding = encoder(src[:, ft_idx].unsqueeze(1))
            embeddings.append(encoding)

        # embeddings has 3 dimensions (num_features, batch, embedding_size)
        embeddings = torch.stack(embeddings)
        # Encodes through transformer encoder
        # Due transpose, the output will be in format (batch, num_features, embedding_size)
        if self.__need_weights:
            output, weights = self.transformer_encoder(embeddings)
            output = output.transpose(0, 1)
        else:
            output = self.transformer_encoder(embeddings).transpose(0, 1)

        # Aggregation of encoded vectors
        output = self.aggregator(output)

        # Decoding
        output = self.decoder(output)
        
        if self.__need_weights:
            return output.squeeze(), weights

        return output.squeeze()

        
class MixtureModels(nn.Module):

    def __init__(
        self, 
        n_head, # Number of heads per layer
        n_hid, # Size of the MLP inside each transformer encoder layer
        n_output, # The number of output neurons
        encoders, # List of features encoders
        n_models, # The number of models
        dropout=0.1, # Used dropout
        aggregator=None, # The aggregator for output vectors before decoder
        attn_aggregator=None, # The aggregator for output vectors before decoder
        preprocessor=None,
        need_weights=False
    ):
        #self, ninp, nhead, nhid, nmodels, nfeatures, nclasses, dropout=0.5):
        super(MixtureModels, self).__init__()


        # Verify that encoders are correct
        if not isinstance(encoders, nn.ModuleList):
            raise TypeError("Parameter encoders must be an instance of torch.nn.ModuleList")

        # Embedding size
        self.n_input = None

        for idx, encoder in enumerate(encoders):
            
            if not issubclass(type(encoder), FeatureEncoder):
                raise TypeError("All encoders must inherit from FeatureEncoder. Invalid index {}".format(idx))

            if self.n_input is None:
                self.n_input = encoder.output_size
            elif self.n_input != encoder.output_size:
                raise ValueError("All encoders must have the same output")

        self.encoders = encoders
        self.__need_weights = need_weights

        n_features = len(self.encoders)
       
        in_proj_container = InProjContainer(
                                torch.nn.Linear(self.n_input, self.n_input),
                                torch.nn.Linear(self.n_input, self.n_input),
                                torch.nn.Linear(self.n_input, self.n_input)
                            )

        self.self_attn = MultiheadAttentionContainer(
                            n_head,
                            in_proj_container,
                            ScaledDotProduct(dropout=dropout),
                            torch.nn.Linear(self.n_input, self.n_input)
                        )
                    
        self.n_head = n_head
        self.n_hid = n_hid
        self.dropout = dropout
        self.n_features = len(self.encoders)

        # The default aggregator will be ConcatenateAggregator
        if aggregator is None:
            self.aggregator = ConcatenateAggregator(self.n_input * self.n_features)
        else:
            self.aggregator = aggregator

        # Validates that aggregator inherit from BaseAggregator
        if not issubclass(type(self.aggregator), BaseAggregator):
            raise TypeError("Parameter aggregator must inherit from BaseAggregator")


        # The default aggregator will be NaiveAttentionAggregator
        if attn_aggregator is None:
            self.attn_aggregator = NaiveAttentionAggregator()
        else:
            self.attn_aggregator = attn_aggregator

        # Validates that aggregator inherit from BaseAggregator
        if not issubclass(type(self.attn_aggregator), BaseAttentionAggregator):
            raise TypeError("Parameter attn_aggregator must inherit from BaseAttentionAggregator")



        self.preprocessor = preprocessor

        if self.preprocessor is not None:
            if not issubclass(type(self.preprocessor), BasePreprocessor):
                    raise TypeError("Preprocessor must inherit from BasePreprocessor.")


        self.representation = nn.Sequential(
                                nn.Linear(self.aggregator.output_size, n_hid),
                                nn.BatchNorm1d(n_hid),                          
                                nn.Dropout(dropout)
                            )

        self.model_weighting = nn.Sequential(
                                    nn.Linear(self.n_features ** 2, n_models),
                                    nn.Softmax(dim=-1)
                                )
        
        self.models = nn.ModuleList()
        self.n_models = n_models
        for model in range(n_models):
            self.models.append(nn.Linear(n_hid, n_output))

    @property
    def need_weights(self):
        return self.__need_weights
        

    def forward(self, src):

        # Preprocess source if needed
        if self.preprocessor is not None:
            src = self.preprocessor(src)

        
        # Validate than src features and num of encoders is the same
        if src.size()[1] != len(self.encoders):
            raise ValueError("The number of features must be the same as the number of encoders.\
                 Got {} features and {} encoders".format(src.size()[1], len(self.encoders)))

        # src came with two dims: (batch_size, num_features)
        embeddings = []

        # Computes embeddings for each feature
        for ft_idx, encoder in enumerate(self.encoders):
            # Each encoder must return a two dims tensor (batch, embedding_size)
            encoding = encoder(src[:, ft_idx].unsqueeze(1))
            embeddings.append(encoding)

        # embeddings has 3 dimensions (num_features, batch, embedding_size)
        embeddings = torch.stack(embeddings)
        # Encodes through transformer encoder
        # Due transpose, the output will be in format (batch, num_features, embedding_size)
        output, weights = self.self_attn(embeddings, embeddings, embeddings)

        if self.self_attn.batch_first:
            batch_size = embeddings.shape[-3]
            num_features = embeddings.shape[-2]
        else:
            batch_size = embeddings.shape[-2]
            num_features = embeddings.shape[-3]

        weights = weights.reshape((batch_size, -1, num_features, num_features))
        output = output.transpose(0, 1)
        
        # Aggregation of encoded vectors
        output = self.aggregator(output)
        weights = self.attn_aggregator(weights)

        # Get attention output representation
        representation = self.representation(output)
        # Get attention matrix weighting
        model_weights = self.model_weighting(weights).unsqueeze(1)

        outputs = []
        
        for model in self.models:
            outputs.append(
                model(representation)
            )

        output = torch.stack(outputs, dim=0).transpose(0, 1)
        output = torch.bmm(model_weights, output).sum(dim=1)


        if self.__need_weights:
            return output.squeeze(), weights

        return output.squeeze()
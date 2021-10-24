import torch
import torch.nn as nn
from torchtext.nn import MultiheadAttentionContainer, InProjContainer, ScaledDotProduct
from torch import Tensor
from typing import Optional


from ndsl.module.encoder import FeatureEncoder
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


class MixtureModelv0(nn.Module):

    def __init__(self, ninp, nhead, nhid, nmodels, nfeatures, nclasses, dropout=0.5):
        super(MixtureModelv0, self).__init__()

        self.attention_mechanism = nn.MultiheadAttention(
                                        ninp, 
                                        nhead, 
                                        dropout=dropout
                                    )

                    
        self.nfeatures = nfeatures
        self.nmodels = nmodels
        #self.num_embedding = nn.Linear(1, ninp)

        self.embedding = nn.ModuleList()
        
        for feature in range(nfeatures):
            self.embedding.append(nn.utils.weight_norm(nn.Linear(1, ninp)))
        

        self.representation = nn.Sequential(
                                nn.Linear(nfeatures * ninp, nhid),
                                nn.BatchNorm1d(nhid),                          
                                nn.Dropout(dropout)
                            )


        self.model_weighting = nn.Sequential(
                                    nn.Linear(nfeatures, nmodels),
                                    nn.Softmax(dim=-1)
                                )
        
        self.models = nn.ModuleList()
        
        for model in range(nmodels):
            self.models.append(nn.Linear(nhid, nclasses))
        
    def aggregate(self, attn_mat):
        return attn_mat.sum(dim=1)


    def forward(self, src):
        
        #src = self.num_embedding(src)
        src_nums = []
        
        for feature in range(self.nfeatures):
            src_nums.append(
                self.embedding[feature](src[:, feature]).unsqueeze(1)
            )
        
        #src_num = self.num_embedding(src[:, len(categorical_cols):])
        src = torch.cat(src_nums, dim=1)
        src = src.transpose(0, 1)

        attn_out, attn_mat = self.attention_mechanism(src, src, src)

        attn_out = attn_out.transpose(0, 1).flatten(start_dim=1)
        attn_mat = self.aggregate(attn_mat)

        representation = self.representation(attn_out)

        model_weights = self.model_weighting(attn_mat).unsqueeze(1)

        outputs = []
        
        for model in range(self.nmodels):
            outputs.append(
                self.models[model](representation)
            )

        output = torch.stack(outputs, dim=0).transpose(0, 1)
        output = torch.bmm(model_weights, output).sum(dim=1)

        return output

        
class MixtureModelv1(nn.Module):

    def __init__(self, ninp, nhead, nhid, nmodels, nfeatures, nclasses, dropout=0.5):
        super(MixtureModelv1, self).__init__()

        self.attention_mechanism = nn.MultiheadAttention(
                                        ninp, 
                                        nhead, 
                                        dropout=dropout
                                    )

                    
        self.nfeatures = nfeatures
        self.nmodels = nmodels
        #self.num_embedding = nn.Linear(1, ninp)

        self.embedding = nn.ModuleList()
        
        for feature in range(nfeatures):
            self.embedding.append(nn.utils.weight_norm(nn.Linear(1, ninp)))
        

        self.representation = nn.Sequential(
                                nn.Linear(nfeatures * ninp, nhid),
                                nn.BatchNorm1d(nhid),                          
                                nn.Dropout(dropout)
                            )


        self.model_weighting = nn.Sequential(
                                    nn.Linear(nfeatures, nmodels),
                                    nn.Softmax(dim=-1)
                                )
        
        self.models = nn.ModuleList()

        self.aggregator = nn.Linear(nfeatures, nfeatures)
        
        for model in range(nmodels):
            self.models.append(nn.Linear(nhid, nclasses))
        
    def aggregate(self, attn_mat):
        return self.aggregator(attn_mat).sum(dim=1)


    def forward(self, src):
        
        #src = self.num_embedding(src)
        src_nums = []
        
        for feature in range(self.nfeatures):
            src_nums.append(
                self.embedding[feature](src[:, feature]).unsqueeze(1)
            )
        
        #src_num = self.num_embedding(src[:, len(categorical_cols):])
        src = torch.cat(src_nums, dim=1)
        src = src.transpose(0, 1)

        attn_out, attn_mat = self.attention_mechanism(src, src, src)

        attn_out = attn_out.transpose(0, 1).flatten(start_dim=1)
        attn_mat = self.aggregate(attn_mat)

        representation = self.representation(attn_out)

        model_weights = self.model_weighting(attn_mat).unsqueeze(1)

        outputs = []
        
        for model in range(self.nmodels):
            outputs.append(
                self.models[model](representation)
            )

        output = torch.stack(outputs, dim=0).transpose(0, 1)
        output = torch.bmm(model_weights, output).sum(dim=1)

        return output


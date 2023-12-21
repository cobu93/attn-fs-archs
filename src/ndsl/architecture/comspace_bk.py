import torch
import torch.nn as nn
import torch.nn.functional as F


from ndsl.module.encoder import FundamentalEmbeddingsEncoder
from ndsl.module.preprocessor import CLSPreprocessor
from ndsl.module.aggregator import CLSAggregator, MaxAggregator, MeanAggregator, SumAggregator, RNNAggregator
from ndsl.architecture.attention import TTransformerEncoderLayer, TTransformerEncoder

def build_mlp(
        n_input,
        n_output,
        hidden_units=None,
        activation_fn=None
):

    if hidden_units is None: 
        hidden_units = []
    
    sizes = [n_input] + hidden_units + [n_output]

    if activation_fn is None: 
        activation_fn = nn.Identity()

    if not isinstance(activation_fn, list):
        activation_fn = [activation_fn for _ in range(len(sizes) - 1)]

    assert len(activation_fn)==len(sizes)-1, "The number of activation functions is not valid"
        
    layers = [] 

    for sizes_idx in range(len(sizes)-1):
        layers.append(nn.Linear(sizes[sizes_idx], sizes[sizes_idx + 1]))
        layers.append(activation_fn[sizes_idx])   
            
    return nn.Sequential(*layers)


class CommonSpaceTransformer(nn.Module):
    
    def __init__(
        self, 
        n_head, # Number of heads per layer
        n_hid, # Size of the MLP inside each transformer encoder layer
        n_layers, # Number of transformer encoder layers    
        n_output, # The number of output neurons
        embed_dim,
        numerical_encoder_hidden_sizes=[128],
        numerical_encoder_activations=[nn.ReLU(), nn.Identity()],
        n_categories=10,
        variational_categories=True,
        attn_dropout=0., # Used dropout,
        ff_dropout=0., # Used dropout
        aggregator=None, # The aggregator for output vectors before decoder
        rnn_aggregator_parameters=None,
        decoder_hidden_units=None,
        decoder_activation_fn=None,
        need_weights=False
        ):

        super(CommonSpaceTransformer, self).__init__()

        self.__need_weights = need_weights

        # Building transformer encoder
        encoder_layers = TTransformerEncoderLayer(embed_dim, n_head, n_hid, attn_dropout=attn_dropout, ff_dropout=ff_dropout)
        self.transformer_encoder = TTransformerEncoder(encoder_layers, n_layers, need_weights=self.__need_weights, enable_nested_tensor=False)

        self.n_head = n_head
        self.n_hid = n_hid

        self.embed_dim = embed_dim
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
        #self.embeeding_table = nn.Embedding(self.n_bins + self.n_categories + 1, embed_dim)
        self.numerical_encoder = build_mlp(
            1, #n_input 
            self.embed_dim, #n_output 
            numerical_encoder_hidden_sizes, 
            numerical_encoder_activations
        )

        self.categorical_encoder = FundamentalEmbeddingsEncoder(
                        self.embed_dim, 
                        n_categories,
                        variational=variational_categories
                        )
    
        #self.decoder = nn.Linear(self.aggregator.output_size + self.n_numerical_features, n_output)
        self.decoder = build_mlp(
            self.aggregator.output_size,
            n_output,
            hidden_units=decoder_hidden_units,
            activation_fn=decoder_activation_fn
        )        
        
    @property
    def need_weights(self):
        return self.__need_weights

    @need_weights.setter
    def need_weights(self, new_need_weights):
        self.__need_weights = new_need_weights
        self.transformer_encoder.need_weights = self.__need_weights

    def forward(self, x_categorical, x_numerical):
        
        batch_size = x_numerical.shape[0]
        n_numerical_features = x_numerical.shape[1]
        n_categorical_features = x_categorical.shape[1]

        numerical_embedddings = torch.zeros(batch_size, n_numerical_features, self.embed_dim).to(x_numerical.device)
        categorical_embedddings = torch.zeros(batch_size, n_categorical_features, self.embed_dim).to(x_categorical.device)

        for ft_idx in range(x_numerical.shape[1]):
            numerical_embedddings[:, ft_idx] = self.numerical_encoder(x_numerical[:, ft_idx].unsqueeze(1))

        for ft_idx in range(x_categorical.shape[1]):
            categorical_embedddings[:, ft_idx] = self.categorical_encoder(x_categorical[:, ft_idx])
        
        embeddings = torch.cat([categorical_embedddings, numerical_embedddings], dim=1)
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
            return output, layer_outs, weights

        return output
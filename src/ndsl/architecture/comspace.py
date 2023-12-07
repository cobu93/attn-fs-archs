import torch
import torch.nn as nn
import torch.nn.functional as F


from ndsl.module.encoder import CategoricalEncoder
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


def build_numerical_embedding(
    n_input=1,   
    embed_dim=128,
    hidden_sizes=[128],
    activations=[nn.ReLU(), nn.Identity()]
    ):
    
    return build_mlp(
            n_input, 
            embed_dim,
            hidden_sizes, 
            activations
        )


def build_categorical_embedding(
    n_categories,
    embed_dim=128,    
    variational=True
    ):
    
    return  CategoricalEncoder(
                        embed_dim, 
                        n_categories,
                        variational=variational
                        )

def build_embedding_processor(
    aggregator="cls",
    embed_dim=128
    ):

    if aggregator == "cls":
        return CLSPreprocessor(embed_dim)
    
    return None

def build_encoder(
    embed_dim=128,
    n_head=4,
    n_hid=128,
    attn_dropout=0.1,
    ff_dropout=0.1,
    n_layers=1,
    need_weights=False
    ):

    encoder_layers = TTransformerEncoderLayer(embed_dim, n_head, n_hid, attn_dropout=attn_dropout, ff_dropout=ff_dropout)
    return TTransformerEncoder(encoder_layers, n_layers, need_weights=need_weights, enable_nested_tensor=False)


def build_encoder_decoder_mid(
    aggregator="cls",
    embed_dim=128,
    **kwargs
    ):

    encoder_decoder_mid_fn = None

    if aggregator == "cls":
        encoder_decoder_mid_fn = CLSAggregator(embed_dim)
    elif aggregator == "max":
        encoder_decoder_mid_fn = MaxAggregator(embed_dim)
    elif aggregator == "mean":
        encoder_decoder_mid_fn = MeanAggregator(embed_dim)
    elif aggregator == "sum":
        encoder_decoder_mid_fn = SumAggregator(embed_dim)
    elif aggregator == "rnn":
        encoder_decoder_mid_fn = RNNAggregator(input_size=embed_dim, **kwargs)

    return encoder_decoder_mid_fn



def build_mlp_decoder(
        n_input,
        n_output,
        hidden_sizes=[128],
        activations=[nn.ReLU(), nn.Identity()]
    ):
        return build_mlp(
            n_input,
            n_output,
            hidden_sizes,
            activations
        ) 
    
def build_transformer_decoder(        
    embed_dim=128,
    n_head=4,
    n_hid=128,
    dropout=0.1,
    n_layers=1,
    ):

    decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=n_head, dim_feedforward=n_hid, dropout=dropout, batch_first=True, norm_first=True)
    return nn.TransformerDecoder(decoder_layer, num_layers=n_layers)


class CommonSpaceTransformer(nn.Module):
    def __init__(
            self,
            numerical_embedding,
            categorical_embedding,
            embedding_processor,
            encoder,
            encoder_decoder_mid,
            decoder,
            need_embeddings=False,
        ):

        super(CommonSpaceTransformer, self).__init__()

        self.need_embeddings = need_embeddings

        self.numerical_embedding = numerical_embedding
        self.categorical_embedding = categorical_embedding
        self.embedding_processor = embedding_processor
        self.encoder = encoder
        self.encoder_decoder_mid = encoder_decoder_mid
        self.decoder = decoder

        self.embed_dim = self.categorical_embedding.output_size
    
    def forward(self, x_categorical, x_numerical, mask=None):

        batch_size = x_numerical.shape[0]
        n_numerical = x_numerical.shape[1]
        n_categorical = x_categorical.shape[1]

        num_embeddings = torch.zeros(batch_size, n_numerical, self.embed_dim).to(x_numerical.device)
        cat_embeddings = torch.zeros(batch_size, n_categorical, self.embed_dim).to(x_categorical.device)

        for ft_idx in range(x_numerical.shape[1]):
            num_embeddings[:, ft_idx] = self.numerical_embedding(x_numerical[:, ft_idx].unsqueeze(1))

        for ft_idx in range(x_categorical.shape[1]):
            cat_embeddings[:, ft_idx] = self.categorical_embedding(x_categorical[:, ft_idx])
        
        embeddings = torch.cat([cat_embeddings, num_embeddings], dim=1)

        if self.need_embeddings:
            o_embeddings = embeddings.clone().detach()

        output = None

        if mask is not None:
            tgt_embeddings = mask * embeddings
            embeddings = (1 - mask) * embeddings
        else:
            tgt_embeddings = embeddings

        if self.embedding_processor is not None:
            embeddings = self.embedding_processor(embeddings)

        if self.encoder.need_weights:
            output, layer_outs, weights = self.encoder(embeddings)
        else:
            output = self.encoder(embeddings)

        if self.encoder_decoder_mid:
            output = self.encoder_decoder_mid(output)

        if isinstance(self.decoder, nn.TransformerDecoder):
            output = self.decoder(tgt_embeddings, output)
        else:
            output = self.decoder(output)

        o_dict = {
            "output": output
        }

        if self.encoder.need_weights:
            o_dict["layer_outs"] = layer_outs
            o_dict["weights"] = weights

        if self.need_embeddings:
            o_dict["embeddings"] = o_embeddings
        
        return o_dict
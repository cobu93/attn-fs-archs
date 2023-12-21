import torch
import torch.nn as nn
from torchtext.nn import MultiheadAttentionContainer, InProjContainer, ScaledDotProduct
from torch import Tensor
from typing import Optional

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
                            torch.nn.Linear(embed_dim, embed_dim),
                            batch_first=True
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

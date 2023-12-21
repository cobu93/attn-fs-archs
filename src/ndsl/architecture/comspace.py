import torch
import torch.nn as nn
import torch.nn.functional as F

    
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

        # num_embeddings = torch.zeros(batch_size, n_numerical, self.embed_dim).to(x_numerical.device)
        num_embeddings = self.numerical_embedding(x_numerical)
        cat_embeddings = self.categorical_embedding(x_categorical)
        

        #for ft_idx in range(x_numerical.shape[1]):
        #    num_embeddings[:, ft_idx] = self.numerical_embedding(x_numerical[:, ft_idx].unsqueeze(1))

        
        
        embeddings = torch.cat([cat_embeddings, num_embeddings], dim=1)

        if self.need_embeddings:
            o_embeddings = embeddings.clone().detach()

        output = None

        if mask is not None:
            embeddings = (1 - mask) * embeddings

        if self.embedding_processor is not None:
            embeddings = self.embedding_processor(embeddings)

        if self.encoder.need_weights:
            output, layer_outs, weights = self.encoder(embeddings)
        else:
            output = self.encoder(embeddings)

        if self.encoder_decoder_mid:
            output = self.encoder_decoder_mid(output)

        if self.decoder:
            output = self.decoder(output)

        o_dict = {
            "output": output
        }

        if self.encoder.need_weights:
            o_dict["layer_outs"] = layer_outs
            o_dict["weights"] = weights

        if self.need_embeddings:
            o_dict["embeddings"] = o_embeddings

        if len(o_dict) == 1:
            return output
        
        return o_dict
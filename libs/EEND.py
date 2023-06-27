import torch
import torch.nn as nn
from typing import List, Tuple
from torch.nn import TransformerEncoder, TransformerEncoderLayer

EPSILON = 1e-10


class EEND(nn.Module):
    def __init__(self,
                 n_speakers=2,
                 in_size=345,
                 n_heads=4,
                 n_units=256,
                 n_encoder_layers=4,
                 dim_feedforward=2048,
                 dropout=0.5):
        """ Self-attention-based diarization model.

        Args:
          n_speakers (int): Number of speakers in recording
          in_size (int): Dimension of input feature vector
          n_heads (int): Number of attention heads
          n_units (int): Number of units in a self-attention block
          n_encoder_layers (int): Number of transformer-encoder layers
          dropout (float): dropout ratio
        """
        super(EEND, self).__init__()
        self.n_speakers = n_speakers
        self.in_size = in_size
        self.n_heads = n_heads
        self.n_units = n_units
        self.n_encoder_layers = n_encoder_layers
        self.key_padding_masks = None

        self.encoder = nn.Linear(in_size, n_units)
        self.encoder_norm = nn.LayerNorm(n_units)

        encoder_layers = TransformerEncoderLayer(n_units, n_heads, dim_feedforward, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, n_encoder_layers)

        self.decoder = nn.Linear(n_units, n_speakers)

    def forward(self,
                src: List[torch.Tensor],
                activation=None) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        # Padding
        ilens = [x.shape[0] for x in src]
        src = nn.utils.rnn.pad_sequence(src, padding_value=-1, batch_first=True)

        # Masking
        self.key_padding_masks = torch.zeros((src.shape[0], src.shape[1]), dtype=torch.bool, device=src.device)
        for i in range(self.key_padding_masks.shape[0]):
            self.key_padding_masks[i, ilens[i]:] = True

        # src: (B, T, E)
        src = self.encoder(src)
        # src: (B, T, E)
        src = self.encoder_norm(src)
        # src: (T, B, E)
        src = src.transpose(0, 1)
        # embedding: (T, B, E)
        embedding = self.transformer_encoder(src, src_key_padding_mask=self.key_padding_masks)
        # embedding: (B, T, E)
        embedding = embedding.transpose(0, 1)
        # global_result: (B, T, C)
        global_result = self.decoder(embedding)

        if activation:
            global_result = activation(global_result)

        global_result = [out[:ilen] for out, ilen in zip(global_result, ilens)]

        return global_result


if __name__ == "__main__":
    from ptflops import get_model_complexity_info
    from hparam import hparam as hp

    model = EEND(n_speakers=hp.model.n_speakers,
                 in_size=hp.data.dimension,
                 n_heads=hp.model.n_heads,
                 n_units=hp.model.hidden_size,
                 n_encoder_layers=hp.model.n_encoder_layers,
                 dim_feedforward=hp.model.dim_feedforward).eval()

    # ptflops

    macs, params = get_model_complexity_info(model, (500, 345), as_strings=True,
                                             print_per_layer_stat=True, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))

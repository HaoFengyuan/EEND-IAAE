import torch
import torch.nn as nn
from typing import List, Tuple
from torch.nn import TransformerEncoder, TransformerEncoderLayer

EPSILON = 1e-10


def _get_activation_fn(activation):
    if activation == "relu":
        return torch.nn.functional.relu
    elif activation == "gelu":
        return torch.nn.functional.gelu
    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))


def WeigthtStatisticsPooling(preweight, inputs, ilens):
    p = torch.sigmoid(preweight)
    index = torch.where(p > 0.5, 1, 0)
    for i in range(len(ilens)):
        index[i, ilens[i]:, :] = 0
    index = index * p
    total = torch.sum(index, 1)

    # Local Attractor1
    attention_weight1 = index[:, :, 0].unsqueeze(2).expand(-1, -1, inputs.shape[-1])
    weight_input1 = torch.mul(attention_weight1, inputs)
    mean1 = torch.sum(weight_input1, 1) / (total[:, 0].unsqueeze(1) + EPSILON)

    # Local Attractor2
    attention_weight2 = index[:, :, 1].unsqueeze(2).expand(-1, -1, inputs.shape[-1])
    weight_input2 = torch.mul(attention_weight2, inputs)
    mean2 = torch.sum(weight_input2, 1) / (total[:, 1].unsqueeze(1) + EPSILON)

    return torch.stack((mean1, mean2), 1)


class TransformerAttractorLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0., activation="relu"):
        super(TransformerAttractorLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # Implementation of Feedforward Layer
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = torch.nn.functional.relu
        super(TransformerAttractorLayer, self).__setstate__(state)

    def forward(self, src, attractor, src_key_padding_mask=None):
        # Multi-head Self-attention
        attractor1 = self.self_attn(attractor, attractor, attractor)[0]
        attractor = attractor + self.dropout3(attractor1)
        attractor = self.norm3(attractor)

        # Multi-head Source-target-attention
        attractor2 = self.attn(attractor, src, src, key_padding_mask=src_key_padding_mask)[0]
        attractor = attractor + self.dropout1(attractor2)
        attractor = self.norm1(attractor)

        # Feedforward
        attractor2 = self.linear2(self.dropout(self.activation(self.linear1(attractor))))
        attractor = attractor + self.dropout2(attractor2)
        attractor = self.norm2(attractor)
        return attractor


class AttractorDecode(nn.Module):
    def __init__(self, depth, d_model, n_heads, dim_feedforward, dropout):
        super(AttractorDecode, self).__init__()
        self.depth = depth
        self.attractor_layer = nn.ModuleList([])
        for i in range(depth):
            self.attractor_layer.append(TransformerAttractorLayer(d_model, n_heads, dim_feedforward, dropout))

    def forward(self, src, attractor, src_key_padding_mask=None):
        attractor = attractor.transpose(0, 1)
        src = src.transpose(0, 1)

        for i in range(self.depth):
            attractor = self.attractor_layer[i](src, attractor, src_key_padding_mask)
        return attractor.permute(1, 0, 2)


class EEND_IAAE(nn.Module):
    def __init__(self,
                 n_speakers=2,
                 in_size=345,
                 n_heads=4,
                 n_units=256,
                 n_encoder_layers=4,
                 n_decoder_layers=2,
                 dim_feedforward=2048,
                 dropout=0.5):
        """ Self-attention-based diarization model.

        Args:
          n_speakers (int): Number of speakers in recording
          in_size (int): Dimension of input feature vector
          n_heads (int): Number of attention heads
          n_units (int): Number of units in a self-attention block
          n_encoder_layers (int): Number of transformer-encoder layers
          n_decoder_layers (int): Number of transformer-decoder layers
          dropout (float): dropout ratio
        """
        super(EEND_IAAE, self).__init__()
        self.n_speakers = n_speakers
        self.in_size = in_size
        self.n_heads = n_heads
        self.n_units = n_units
        self.n_encoder_layers = n_encoder_layers
        self.n_decoder_layers = n_decoder_layers
        self.key_padding_masks = None

        self.encoder = nn.Linear(in_size, n_units)
        self.encoder_norm = nn.LayerNorm(n_units)

        encoder_layers = TransformerEncoderLayer(n_units, n_heads, dim_feedforward, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, n_encoder_layers)

        self.decoder = nn.Linear(n_units, n_speakers)
        self.attractor_layer = AttractorDecode(n_decoder_layers, n_units, n_heads, dim_feedforward, dropout)

    def forward(self,
                src: List[torch.Tensor],
                depth: int = 4,
                activation=None) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        # Padding
        ilens = [x.shape[0] for x in src]
        src = nn.utils.rnn.pad_sequence(src, padding_value=-1, batch_first=True)

        # Masking
        self.key_padding_masks = torch.zeros((src.shape[0], src.shape[1]), dtype=torch.bool, device=src.device)
        for i in range(self.key_padding_masks.shape[0]):
            self.key_padding_masks[i, ilens[i]:] = True

        # Stage 1
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

        # Stage 2
        # attractor: (B, C, E)
        attractor = WeigthtStatisticsPooling(global_result, embedding, ilens)
        for _ in range(depth):
            # attractor: (B, C, E)
            attractor = self.attractor_layer(embedding, attractor, src_key_padding_mask=self.key_padding_masks)
            # embedding: (B, T, C)
            local_result = torch.bmm(embedding, attractor.transpose(1, 2))
            # attractor: (B, C, E)
            attractor = WeigthtStatisticsPooling(local_result, embedding, ilens)

        if activation:
            global_result = activation(global_result)
            local_result = activation(local_result)

        global_result = [out[:ilen] for out, ilen in zip(global_result, ilens)]
        local_result = [out[:ilen] for out, ilen in zip(local_result, ilens)]

        return global_result, local_result


if __name__ == "__main__":
    from ptflops import get_model_complexity_info
    from hparam import hparam as hp

    model = EEND_IAAE(n_speakers=hp.model.n_speakers,
                      in_size=hp.data.dimension,
                      n_heads=hp.model.n_heads,
                      n_units=hp.model.hidden_size,
                      n_encoder_layers=hp.model.n_encoder_layers,
                      n_decoder_layers=hp.model.n_decoder_layers,
                      dim_feedforward=hp.model.dim_feedforward).eval()

    # ptflops
    macs, params = get_model_complexity_info(model, (500, 345), as_strings=True,
                                             print_per_layer_stat=True, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))

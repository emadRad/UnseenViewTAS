import math
from math import pi
from dataclasses import dataclass

import torch
import torch.nn as nn
from einops import rearrange, repeat


@dataclass
class PosEmbeddingParams:
    type: str
    hidden_size: int
    max_seq_len: int = 10000
    relative_attention_num_buckets: int = 32
    relative_attention_max_distance: int = 128
    num_freq_band: int = 15,
    num_heads: int = 4


class SinePositionalEmbedding(nn.Module):
    """
    Adds positional embedding to an input embedding.

    Based on https://arxiv.org/pdf/1706.03762.pdf.
    """
    def __init__(self, hidden_size):
        super(SinePositionalEmbedding, self).__init__()

        assert hidden_size % 2 == 0 and 'Model vector size must be even for sinusoidal encoding'
        power = torch.arange(0, hidden_size, step=2, dtype=torch.float32)[:] / hidden_size
        divisor = 10000 ** power
        self.divisor = divisor
        self.hidden_size = hidden_size

    def forward(self, inputs):
        """
            Args:
                inputs: a float32 Tensor with shape [batch_size, sequence_length, hidden_size]

            Returns:
                embedding: a float32 Tensor with shape [batch_size, sequence_length, hidden_size]
        """
        assert inputs.shape[-1] == self.hidden_size and 'Input final dim must match model hidden size'

        batch_size = inputs.shape[0]
        sequence_length = inputs.shape[1]

        # obtain a sequence that starts at `start` and increments for `sequence_length `
        seq_pos = torch.arange(0, sequence_length, dtype=torch.float32)
        seq_pos_expanded = seq_pos[None, :, None]
        index = seq_pos_expanded.repeat(*[1, 1, self.hidden_size//2])

        sin_embedding = torch.sin(index / self.divisor)
        cos_embedding = torch.cos(index / self.divisor)

        # interleave the sin and cos. For more info see:
        # https://discuss.pytorch.org/t/how-to-interleave-two-tensors-along-certain-dimension/11332/3
        position_shape = (1, sequence_length, self.hidden_size)  # fill in the other two dimensions
        position_embedding = torch.stack((sin_embedding, cos_embedding), dim=3).view(position_shape)

        pos_embed_deviced = position_embedding.to(inputs.device)
        return inputs + pos_embed_deviced


class FourierEncoding(nn.Module):
    def __init__(self, num_bands: int):
        super(FourierEncoding, self).__init__()
        self.num_bands = num_bands

    def fourier_encode(self, x, max_res, concat_pos=True):
        max_res = max(max_res)
        x = x.unsqueeze(-1)
        device, dtype = x.device, x.dtype
        scales = torch.linspace(1.,
                                max_res / 2,
                                self.num_bands,
                                device=device,
                                dtype=dtype)
        new_axis = " ".join(["1" for _ in range(x.ndim-1)])
        scales = rearrange(scales, f'n -> {new_axis} n')
        l = x * scales * pi
        l = torch.cat([l.sin(), l.cos()], dim=-1)  # [seq_len, d, 2*num_bands]
        if concat_pos:
            x = torch.cat((l, x), dim=-1)  # [seq_len, d, 2*num_bands+1]
        else:
            x = l
        return x

    def forward(self, inputs, concat_pos=True, concat_inputs=True):
        """
        :param inputs: [batch_size, *(axis_shape), hidden_size]
        :param concat_pos: concat raw position to the Fourier features
        :param concat_inputs:
        :return:
        """
        b, *axis, _ = inputs.shape
        axis_pos = list(map(lambda size: torch.linspace(-1., 1.,
                                                        steps=size,
                                                        device=inputs.device,
                                                        dtype=inputs.dtype),
                            axis))
        pos = torch.stack(torch.meshgrid(*axis_pos, indexing='ij'), dim=-1)
        pos_embeddings = self.fourier_encode(pos, axis, concat_pos)  # [*(axis_shape), d, 2*num_bands+1]
        pos_embeddings = rearrange(pos_embeddings, '... n d -> ... (n d)')  # [*(axis_shape), d*2*num_bands+1]
        pos_embeddings = repeat(pos_embeddings, '... -> b ...', b=b)  # [batch_size, *(axis_shape), d*2*num_bands+1]
        if concat_inputs:
            outputs = torch.cat((inputs, pos_embeddings), dim=-1)  # [batch_size, *(axis_shape), dim+2*num_bands+1]
        else:
            outputs = pos_embeddings
        outputs = rearrange(outputs, 'b ... d -> b (...) d')
        return outputs


class LearnedPositionalEmbedding(nn.Module):
    def __init__(self, max_seq_len, hidden_size):
        super(LearnedPositionalEmbedding, self).__init__()
        self.position_embeddings = nn.Parameter(torch.randn(max_seq_len, hidden_size))

    def forward(self, inputs):
        """

        :param inputs: [batch_size, seq_len, hidden_size]
        :return:
        """
        batch_size, seq_len, _ = inputs.shape
        pos_embedding = self.position_embeddings[:seq_len]
        pos_embedding = repeat(pos_embedding, 'n d -> b n d', b=batch_size)
        return inputs + pos_embedding


class RelPositionalEmbedding(nn.Module):
    def __init__(self, relative_attention_num_buckets, relative_attention_max_distance, num_heads):
        super(RelPositionalEmbedding, self).__init__()
        self.num_heads = num_heads
        self.relative_attention_num_buckets = relative_attention_num_buckets
        self.relative_attention_max_distance = relative_attention_max_distance
        self.relative_attention_bias = nn.Embedding(self.relative_attention_num_buckets, num_heads)

    @staticmethod
    def _relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        """
        Adapted from Mesh Tensorflow:
        https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593
        Translate relative position to a bucket number for relative attention. The relative position is defined as
        memory_position - query_position, i.e. the distance in tokens from the attending position to the attended-to
        position. If bidirectional=False, then positive relative positions are invalid. We use smaller buckets for
        small absolute relative_position and larger buckets for larger absolute relative_positions. All relative
        positions >=max_distance map to the same bucket. All relative positions <=-max_distance map to the same bucket.
        This should allow for more graceful generalization to longer sequences than the model has been trained on
        Args:
            relative_position: an int32 Tensor
            bidirectional: a boolean - whether the attention is bidirectional
            num_buckets: an integer
            max_distance: an integer
        Returns:
            a Tensor with the same shape as relative_position, containing int32 values in the range [0, num_buckets)
        """
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
        # now relative_position is in the range [0, inf)

        # half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
        relative_postion_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)
        relative_postion_if_large = torch.min(
            relative_postion_if_large, torch.full_like(relative_postion_if_large, num_buckets - 1)
        )

        relative_buckets += torch.where(is_small, relative_position, relative_postion_if_large)
        return relative_buckets

    def compute_bias(self, query_length, key_length):
        """Compute binned relative position bias"""
        context_position = torch.arange(
            query_length, dtype=torch.long, device=self.relative_attention_bias.weight.device
        )[:, None]
        memory_position = torch.arange(
            key_length, dtype=torch.long, device=self.relative_attention_bias.weight.device
        )[None, :]
        relative_position = memory_position - context_position  # shape (query_length, key_length)
        relative_position_bucket = self._relative_position_bucket(
            relative_position,  # shape (query_length, key_length)
            bidirectional=True,
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
        )
        values = self.relative_attention_bias(relative_position_bucket)  # shape (query_length, key_length, num_heads)
        values = rearrange(values, 'q k n_heads -> n_heads q k')  # shape (num_heads, query_length, key_length)
        return values

    def forward(self, q: torch.Tensor, k: torch.Tensor):
        """
        :param q: tensor w/ shape [batch_size*n_heads, n_queries, channels]
        :param k: tensor w/ shape [batch_size*n_heads, n_keyval, channels]
        :return:
        """
        bs_nh, q_length, _ = q.shape
        _, k_length, _ = k.shape
        batch_size = bs_nh // self.num_heads
        bias = self.compute_bias(q_length, k_length)  # [n_heads, n_queries, n_keyval]
        bias = repeat(bias, 'nh q k -> batch_size nh q k', batch_size=batch_size)
        bias = rearrange(bias, 'b n_heads q_len k_len -> (b n_heads) q_len k_len')
        return bias


def build_position_embedding(pos_embedding_params):
    if pos_embedding_params.type == "sine":
        return SinePositionalEmbedding(
            pos_embedding_params.hidden_size,
        )
    if pos_embedding_params.type == "fourier":
        return FourierEncoding(
            pos_embedding_params.num_freq_band,
        )
    elif pos_embedding_params.type == "learned":
        return LearnedPositionalEmbedding(max_seq_len=pos_embedding_params.max_seq_len,
                                          hidden_size=pos_embedding_params.hidden_size)
    elif pos_embedding_params.type == "relative":
        return RelPositionalEmbedding(
            relative_attention_num_buckets=pos_embedding_params.relative_attention_num_buckets,
            relative_attention_max_distance=pos_embedding_params.relative_attention_max_distance,
            num_heads=pos_embedding_params.num_heads,
        )
    else:
        raise ValueError(f"Unknown position embedding type {pos_embedding_params.type}")


if __name__ == "__main__":
    params = PosEmbeddingParams(
        type="relative",
        num_heads=4,
        relative_attention_num_buckets=32,
        relative_attention_max_distance=128,
    )
    pos_embed = build_position_embedding(params)
    print(pos_embed)

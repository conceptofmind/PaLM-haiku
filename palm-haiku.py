import haiku as hk
from haiku import PRNGSequence

import jax
from jax import random, nn, lax
import jax.numpy as jnp
from jax.numpy import einsum

from einops import rearrange

# Global variables

ATTN_MASK_VALUE = -1e10

# bias-less layernorm

class LayerNorm(hk.Module):
    def __init__(self, dim, eps = 1e-5):
        super(LayerNorm, self).__init__()
        self.gamma = jnp.ones((dim,))
        self.eps = eps

    def __call__(self, x):
        mean = jnp.mean(x, axis = -1, keepdims = True)
        mean_of_squares = jnp.mean(jnp.square(x), axis = -1, keepdims = True)
        variance = mean_of_squares - jnp.square(mean)
        inv = lax.rsqrt(variance + self.eps)
        return inv * (x - mean) * self.gamma

# prenorm

class PreNorm(hk.Module):
    def __init__(self, fn):
        super(PreNorm, self).__init__()
        self.norm = LayerNorm()
        self.fn = fn
    def __call__(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


# rotary positional embedding
# https://arxiv.org/abs/2104.09864

class RotaryEmbedding(hk.Module):
    def __init__(self, dim):
        super(RotaryEmbedding, self).__init__()
        self.inv_freq = 1.0 / (10000 ** (jnp.arange(0, dim, 2) / dim)) 

    def __call__(self, max_seq_len):
        seq = jnp.arange(max_seq_len)
        freqs = einsum("i , j -> i j", seq, self.inv_freq)
        return jnp.concatenate((freqs, freqs), axis = -1)

def jax_unstack(x, axis = 0):
    return jnp.moveaxis(x, axis, 0)

def rotate_half(x):
    x = rearrange(x, "... (j d) -> ... j d", j = 2)
    x1, x2 = jax_unstack(x, axis = -2)
    return jnp.concatenate((-x2, x1), axis = -1)

def apply_rotary_pos_emb(pos, t):
    return (t * jnp.cos(pos)) + (rotate_half(t) * jnp.sin(pos))


# classic Noam Shazeer paper, except here they use SwiGLU instead of the more popular GEGLU for gating the feedforward
# https://arxiv.org/abs/2002.05202

class SwiGLU(hk.Module):
    def __call__(self, x):
        x, gate = x.split(2, axis = -1)
        return jnp.multiply(nn.swish(gate), x)


# parallel attention and feedforward with residual
# discovered by Wang et al + EleutherAI from GPT-J fame

class ParallelTransformerBlock(hk.Module):
    def __init__(self, dim, dim_head=64, heads=8, ff_mult=4):
        super(ParallelTransformer, self).__init__()
        self.norm = hk.LayerNorm(dim)

        attn_inner_dim = dim_head * heads
        ff_inner_dim = dim * ff_mult
        self.fused_dims = (attn_inner_dim, dim_head, dim_head, (ff_inner_dim * 2))

        self.heads = heads
        self.scale = dim_head**-0.5
        self.rotary_emb = RotaryEmbedding(dim_head)

        self.fused_attn_ff_proj = hk.Linear(sum(self.fused_dims), bias=False)
        self.attn_out = hk.Linear(dim, bias=False)

        self.ff_out = hk.Sequential(
            SwiGLU(),
            hk.Linear(dim, bias=False)
        )

    def __call__(self, x):
        """
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension
        """

        n, h = x.shape[1], self.heads

        # pre layernorm

        x = self.norm(x)

        # attention queries, keys, values, and feedforward inner

        q, k, v, ff = self.fused_attn_ff_proj(x).split(self.fused_dims, dim=-1)

        # split heads
        # they use multi-query single-key-value attention, yet another Noam Shazeer paper
        # they found no performance loss past a certain scale, and more efficient decoding obviously
        # https://arxiv.org/abs/1911.02150

        q = rearrange(q, "b n (h d) -> b h n d", h=h)

        # rotary embeddings

        positions = self.rotary_emb(n)

        if positions is not None and positions.shape[-2] >= n:
            positions = positions[:n]

        q, k = map(lambda t: apply_rotary_pos_emb(positions, t), (q, k))

        # scale

        q = q * self.scale

        # similarity

        sim = einsum("b h i d, b j d -> b h i j", q, k)

        # causal mask

        mask = jnp.tril(jnp.ones((n, n)))

        if mask is not None and mask.shape[-1] >= n:
            mask = mask[:n, :n]

        sim = jnp.where(mask, sim, ATTN_MASK_VALUE)

        # attention

        attn = sim.softmax(dim=-1)

        # aggregate values

        out = einsum("b h i j, b j d -> b h i d", attn, v)

        # merge heads

        out = rearrange(out, "b h n d -> b n (h d)")
        return self.attn_out(out) + self.ff_out(ff)

# transformer

class ParallelTransformer(hk.Module):
    def __init__(self, dim, depth, heads, dim_head, ff_mult):
        super(ParallelTransformer, self).__init__()

        self.layers = []
        for _ in range(depth):
            self.layers.append(
                PreNorm(ParallelTransformerBlock(dim, dim_head, heads, ff_mult))
            )

    def __call__(self, x):
        for block in self.layers:
            x = block(x) + x
        return x

# model

class PaLM_base(hk.Module):
    def __init__(self, *, dim, num_tokens, depth, dim_head = 64, heads = 8, ff_mult = 4):
        super(PaLM_base, self).__init__()
        self.embed = nn.Embed(num_tokens, dim)
        self.net = ParallelTransformerBlock(dim=dim, dim_head=dim_head, heads=heads, ff_mult=ff_mult)
        self.layer_norm = LayerNorm(dim)
        self.linear = hk.Linear(dim, num_tokens, bias=False)


    return net

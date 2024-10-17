
from typing import Callable, Optional

import flax.linen as nn
import jax.numpy as jnp
from flax.linen import initializers
from collections.abc import Iterable
from typing import Callable, List, Optional

import flax.linen as nn
import jax
import jax.numpy as jnp


class DNN(nn.Module):
    width: int
    layers: List[str]
    out_dim: int
    activation: Callable = jax.nn.swish
    period: Optional[jnp.ndarray] = None
    rank: int = 1
    full: bool = False

    @nn.compact
    def __call__(self, x):
        depth = len(self.layers)
        width = self.width

        A = self.activation
        for i, layer in enumerate(self.layers):
            is_last = i == depth - 1

            if isinstance(self.activation, Iterable):
                A = self.activation[i]

            if is_last:
                width = self.out_dim
            L = get_layer(layer=layer, width=width, rank=self.rank, full=self.full)
            x = L(x)
            if not is_last:
                x = A(x)

        return x


def get_layer(layer, width, rank=1, full=False):
    if layer == 'D':
        L = nn.Dense(width)
    elif layer == 'C':
        L = CoLoRA(width, rank, full)
    else:
        raise Exception(f"unknown layer: {layer}")
    return L



class CoLoRA(nn.Module):

    width: int
    rank: int
    full: bool
    w_init: Callable = initializers.lecun_normal()
    b_init: Callable = initializers.zeros_init()
    with_bias: bool = True
    param_dtype = jnp.float32

    @nn.compact
    def __call__(self, X):
        D, K, r = X.shape[-1], self.width, self.rank

        w_init = self.w_init
        b_init = self.b_init
        z_init = initializers.zeros_init()

        W = self.param('W', w_init, (D, K), self.param_dtype)
        A = self.param('A', w_init, (D, r), self.param_dtype)
        B = self.param('B', z_init, (r, K), self.param_dtype)

        if self.full:
            n_alpha = self.rank
        else:
            n_alpha = 1

        alpha = self.param('alpha', z_init, (n_alpha,), self.param_dtype)

        AB = (A*alpha)@B
        AB = AB  # / r
        W = (W + AB)

        out = X@W

        if self.with_bias:
            b = self.param("b", b_init, (K,))
            b = jnp.broadcast_to(b, out.shape)
            out += b

        return out

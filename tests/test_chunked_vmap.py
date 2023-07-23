# Copyright Karl Otness
# SPDX-License-Identifier: MIT


import jax
import jax.numpy as jnp
import powerpax as ppx


def test_matches_vmap():
    def fun(arg):
        return arg * 2

    arr = jnp.arange(10)
    jax_res = jax.vmap(fun)(arr)
    chunked_res = ppx.chunked_vmap(fun, chunk_size=3)(arr)
    assert jax_res.dtype == chunked_res.dtype
    assert jax_res.shape == chunked_res.shape
    assert jnp.all(jax_res == chunked_res)

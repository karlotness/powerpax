# Copyright 2023 Karl Otness
# SPDX-License-Identifier: MIT


import jax
import jax.numpy as jnp
import powerpax as ppx
import pytest


@pytest.mark.parametrize("chunk_size", [1, 3, 5, 9, 10, 11])
@pytest.mark.parametrize(
    "use_args,use_kwargs", [(True, True), (True, False), (False, True)]
)
def test_matches_vmap(chunk_size, use_args, use_kwargs):
    def fun(arg=1, arg2=1):
        return arg * arg2

    arr = jnp.arange(10)
    args = (arr,) if use_args else ()
    kwargs = {"arg2": arr} if use_kwargs else {}
    jax_res = jax.vmap(fun)(*args, **kwargs)
    chunked_res = jax.jit(ppx.chunked_vmap(fun, chunk_size=chunk_size))(*args, **kwargs)
    assert jax_res.dtype == chunked_res.dtype
    assert jax_res.shape == chunked_res.shape
    assert jnp.all(jax_res == chunked_res)


@pytest.mark.parametrize("chunk_size", [-1, 0])
def test_error_invalid_chunk_size(chunk_size):
    def fun(arg):
        return arg * 2

    with pytest.raises(ValueError, match="chunk_size must be positive"):
        _ = ppx.chunked_vmap(fun, chunk_size=chunk_size)

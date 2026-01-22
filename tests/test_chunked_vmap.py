# Copyright 2023 Karl Otness
# SPDX-License-Identifier: MIT


import jax
import jax.numpy as jnp
import powerpax as ppx
import pytest
from hypothesis import given, settings, strategies as st


@st.composite
def length_chunk_size(draw):
    length = draw(st.integers(min_value=0, max_value=15))
    chunk_size = draw(st.integers(min_value=1, max_value=length + 1))
    return length, chunk_size


@given(
    len_cs=length_chunk_size(),
    use_args_kwargs=st.tuples(st.booleans(), st.booleans()).filter(any),
)
@settings(deadline=None)
def test_matches_vmap(len_cs, use_args_kwargs):
    length, chunk_size = len_cs
    use_args, use_kwargs = use_args_kwargs

    def fun(arg=1, arg2=1):
        return {"res": arg * arg2, "arg": arg, "arg2": arg2}

    args = (jnp.arange(length),) if use_args else ()
    kwargs = (
        {"arg2": jnp.arange(length * 6).reshape(length, 2, 3)} if use_kwargs else {}
    )
    jax_res = jax.vmap(fun)(*args, **kwargs)
    chunked_res = jax.jit(ppx.chunked_vmap(fun, chunk_size=chunk_size))(*args, **kwargs)
    assert jax.tree_util.tree_structure(jax_res) == jax.tree_util.tree_structure(
        chunked_res
    )
    assert jax.tree_util.tree_all(
        jax.tree_util.tree_map(
            lambda a, b: jnp.array_equal(a, b) and a.dtype == b.dtype,
            jax_res,
            chunked_res,
        )
    )


@pytest.mark.parametrize("chunk_size", [-1, 0])
def test_error_invalid_chunk_size(chunk_size):
    def fun(arg):
        return arg * 2

    with pytest.raises(ValueError, match="chunk_size must be positive"):
        _ = ppx.chunked_vmap(fun, chunk_size=chunk_size)

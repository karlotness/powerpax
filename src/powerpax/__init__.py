# Copyright Karl Otness
# SPDX-License-Identifier: MIT

r"""Useful utilities for JAX.

This package provides a small set of useful utilities for working with
JAX. For example, functions to limit the memory consumption of
:func:`jax.vmap` (:func:`ppx.chunked_vmap <powerpax.chunked_vmap>`) or
to keep the outputs of only a subset of :func:`jax.scan` iterations
(:func:`ppx.sliced_scan <powerpax.sliced_scan>`).
"""

__version__ = "0.1.0"
__all__ = ["chunked_vmap", "sliced_scan", "checkpoint_chunked_scan", "Static"]

from ._vmap import chunked_vmap
from ._loop import sliced_scan, checkpoint_chunked_scan
from ._tree import Static

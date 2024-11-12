# Copyright 2023 Karl Otness
# SPDX-License-Identifier: MIT


import dataclasses
import typing
import jax


T = typing.TypeVar("T")


@dataclasses.dataclass(frozen=True)
class Static(typing.Generic[T]):
    r"""Treat a value as :term:`jax:static` when processing a pytree.

    This class wraps `value` and will instruct JAX to treat it as a
    static value during pytree processing.

    Parameters
    ----------
    value: object
        The object to wrap and treat as static (must be
        :term:`python:hashable`).

    Attributes
    ----------
    value: object
        The wrapped, static value.
    """

    value: T


jax.tree_util.register_pytree_node(
    Static, lambda node: ((), node.value), lambda aux, _: Static(aux)
)

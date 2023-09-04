import dataclasses
import typing
import jax


T = typing.TypeVar("T")


@dataclasses.dataclass(frozen=True)
class Static(typing.Generic[T]):
    value: T


jax.tree_util.register_pytree_node(
    Static, lambda node: ((), node.value), lambda aux, _: Static(aux)
)

# Copyright Karl Otness
# SPDX-License-Identifier: MIT


__version__ = "0.1.0.dev"
__all__ = ["chunked_vmap", "sliced_scan"]

from ._vmap import chunked_vmap
from ._loop import sliced_scan

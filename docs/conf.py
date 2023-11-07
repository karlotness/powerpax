# Copyright Karl Otness
# SPDX-License-Identifier: MIT

import inspect
import pkgutil
import pathlib
import packaging.version
import powerpax as ppx

# Project information
project = "powerpax"
copyright = "Karl Otness"
author = "Karl Otness"
version = ppx.__version__
release = version

# Other configuration
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.linkcode",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
html_static_path = ["_static"]
html_css_files = []
suppress_warnings = ["epub.unknown_project_files"]

# Insert code into each rst file
rst_prolog = r"""

.. role:: pycode(code)
   :language: python

.. role:: cppcode(code)
   :language: cpp

"""

# Theme
html_theme = "furo"

# Autodoc configuration
autodoc_mock_imports = []
autodoc_typehints = "none"
autodoc_member_order = "bysource"

# Napoleon configuration
napoleon_google_docstring = False

# Intersphinx configuration
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "jax": ("https://jax.readthedocs.io/en/latest/", None),
}


# Linkcode configuration
def linkcode_resolve(domain, info):
    if domain != "py":
        return None
    mod_name = info["module"]
    if mod_name != "powerpax" and not mod_name.startswith("powerpax."):
        return None
    fullname = info["fullname"]
    pkg_root = pathlib.Path(ppx.__file__).parent
    try:
        obj = pkgutil.resolve_name(f"{mod_name}:{fullname}")
    except AttributeError:
        return None
    if isinstance(obj, property):
        obj = obj.fget
    if obj is None:
        return None
    try:
        source_file = inspect.getsourcefile(obj)
        if source_file is None:
            return None
        source_file = pathlib.Path(source_file).relative_to(pkg_root)
        lines, line_start = inspect.getsourcelines(obj)
        line_end = line_start + len(lines) - 1
    except (ValueError, TypeError):
        return None
    # Form the URL from the pieces
    repo_url = "https://github.com/karlotness/powerpax"
    if packaging.version.Version(version).is_devrelease:
        ref = "master"
    else:
        ref = f"v{version}"
    if line_start and line_end:
        line_suffix = f"#L{line_start}-L{line_end}"
    elif line_start:
        line_suffix = f"#L{line_start}"
    else:
        line_suffix = ""
    return f"{repo_url}/blob/{ref}/src/powerpax/{source_file!s}{line_suffix}"

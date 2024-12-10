# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Flow Matching"
copyright = "2024 Meta Platforms, Inc"
author = "FAIR"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "nbsphinx",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinxcontrib.katex",
    "sphinx.ext.autosectionlabel",
    "sphinxcontrib.bibtex",
]

bibtex_bibfiles = ["refs.bib"]
bibtex_default_style = "unsrt"

templates_path = ["_templates"]
exclude_patterns = ["_build", "**.ipynb_checkpoints"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
html_theme = "pydata_sphinx_theme"
html_static_path = ["_static", "_images"]

# katex config
katex_css_path = "https://cdn.jsdelivr.net/npm/katex@0.16.10/dist/katex.min.css"
katex_js_path = "katex.min.js"
katex_autorender_path = "auto-render.min.js"
katex_inline = [r"\(", r"\)"]
katex_display = [r"\[", r"\]"]
katex_prerender = False
katex_options = ""

# autodoc config
autodoc_member_order = "bysource"
autosummary_generate = True  # Turn on sphinx.ext.autosummary

from custom_directives import (
    CustomCardEnd,
    CustomCardItem,
    CustomCardStart,
    SupportedDevices,
    SupportedProperties,
)

# Register custom directives

from docutils.parsers import rst

rst.directives.register_directive("devices", SupportedDevices)
rst.directives.register_directive("properties", SupportedProperties)
rst.directives.register_directive("customcardstart", CustomCardStart)
rst.directives.register_directive("customcarditem", CustomCardItem)
rst.directives.register_directive("customcardend", CustomCardEnd)


def setup(app):
    app.add_css_file("css/custom.css")  # may also be an URL

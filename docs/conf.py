# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# Add project root to path so we can import bamengine
sys.path.insert(0, os.path.abspath("../src"))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "BAM Engine"
copyright = "2024, Kostas Ganitis"
author = "Kostas Ganitis"

# The version info for the project you're documenting
import bamengine

version = bamengine.__version__
release = bamengine.__version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",  # Auto-generate API docs from docstrings
    "sphinx.ext.autosummary",  # Generate summary tables
    "sphinx.ext.napoleon",  # Support for NumPy-style docstrings
    "sphinx.ext.viewcode",  # Add links to highlighted source code
    "sphinx.ext.intersphinx",  # Link to other project's documentation
    "sphinx.ext.mathjax",  # Render math via MathJax
    "sphinx_gallery.gen_gallery",  # Generate example gallery
    "numpydoc",  # Enhanced NumPy docstring support
]

# Napoleon settings for NumPy-style docstrings
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# Numpydoc settings
numpydoc_show_class_members = False  # Don't show inherited members
numpydoc_class_members_toctree = False

# Autodoc settings
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
}

# Autosummary settings
autosummary_generate = True

# Intersphinx mapping to link to other projects
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
}

# Sphinx-Gallery configuration
sphinx_gallery_conf = {
    "examples_dirs": "../examples",  # Path to example scripts
    "gallery_dirs": "auto_examples",  # Where to save gallery generated output
    "filename_pattern": r"example_.*\.py$",  # Only run files starting with example_
    "download_all_examples": False,  # Don't create download all button
    "matplotlib_animations": False,  # Don't try to save animations
    "image_scrapers": ("matplotlib",),  # Scrape matplotlib figures
    "default_thumb_file": None,  # No default thumbnail
    "line_numbers": False,  # Don't show line numbers
    "remove_config_comments": True,  # Remove config comments from code
    "min_reported_time": 0.1,  # Minimum time to report for example execution
    "show_memory": False,  # Don't show memory usage
    "junit": "",  # Don't generate JUnit XML
    # Use ExplicitOrder if you want specific order, otherwise use default (alphabetical)
    # "subsection_order": ExplicitOrder([...]),
    # "within_subsection_order": FileNameSortKey,
    "capture_repr": ("_repr_html_", "__repr__"),  # Capture these repr methods
    "nested_sections": True,  # Allow nested sections in galleries
    "expected_failing_examples": [],  # List of examples expected to fail
}

# Templates path
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# The master toctree document
master_doc = "index"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

# Theme options
html_theme_options = {
    "collapse_navigation": False,
    "sticky_navigation": True,
    "navigation_depth": 4,
    "includehidden": True,
    "titles_only": False,
    "logo_only": False,
    "display_version": True,
    "prev_next_buttons_location": "bottom",
    "style_external_links": False,
}

# Output file base name for HTML help builder
htmlhelp_basename = "BAMEnginedoc"

# If true, "Created using Sphinx" is shown in the HTML footer
html_show_sphinx = True

# If true, "(C) Copyright ..." is shown in the HTML footer
html_show_copyright = True

# -- Options for LaTeX output ------------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    "papersize": "letterpaper",
    # The font size ('10pt', '11pt' or '12pt').
    "pointsize": "10pt",
    # Additional stuff for the LaTeX preamble.
    "preamble": "",
    # Latex figure (float) alignment
    "figure_align": "htbp",
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title, author, documentclass [howto, manual, or own class]).
latex_documents = [
    (
        master_doc,
        "BAMEngine.tex",
        "BAM Engine Documentation",
        "Kostas Ganitis",
        "manual",
    ),
]

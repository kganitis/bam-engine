# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

from sphinx_gallery.sorting import ExplicitOrder

# Add project root to path so we can import bamengine
sys.path.insert(0, os.path.abspath("../src"))
# TODO: Remove this once examples are refactored to not depend on validation package
sys.path.insert(0, os.path.abspath(".."))  # For validation and calibration packages

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
napoleon_preprocess_types = True
napoleon_type_aliases = {
    "Float1D": "Float1D",
    "Int1D": "Int1D",
    "Bool1D": "Bool1D",
    "Idx1D": "Idx1D",
    "Float": "Float",
    "Int": "Int",
    "Bool": "Bool",
    "Agent": "Agent",
    "AgentId": "AgentId",
    "Rng": "Rng",
}
napoleon_attr_annotations = True

# Custom sections for domain-specific docstrings
napoleon_custom_sections = [
    ("Algorithm", "notes_style"),
    ("Mathematical Notation", "notes_style"),
    ("Class Attributes", "params_style"),
    ("Design Guidelines", "notes_style"),
]

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

# Hide types from signatures entirely to avoid verbose numpy.ndarray[...] types
# Types are documented in the class docstrings instead
autodoc_typehints = "none"

# Type aliases for cleaner type display in docs
# Maps verbose NumPy array types to user-friendly names
autodoc_type_aliases = {
    "Float1D": "Float1D",
    "Int1D": "Int1D",
    "Bool1D": "Bool1D",
    "Idx1D": "Idx1D",
    "Float": "Float",
    "Int": "Int",
    "Bool": "Bool",
    "Agent": "Agent",
    "AgentId": "AgentId",
    "Rng": "Rng",
}

# Autosummary settings
autosummary_generate = True

# Intersphinx mapping to link to other projects
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
}


# Custom sorting for examples within each subsection
class BasicExampleOrder:
    """Custom ordering for basic examples."""

    # Explicit order for basic examples
    order = [
        "example_hello_world.py",
        "example_configuration.py",
        "example_yaml_configuration.py",
        "example_logging.py",
        "example_ops_module.py",
        "example_typing_module.py",
        "example_results_module.py",
        "example_baseline_scenario.py",
    ]

    def __init__(self, src_dir):
        self.src_dir = src_dir

    def __call__(self, filename):
        # Get just the filename from the path
        basename = os.path.basename(filename)
        try:
            return self.order.index(basename)
        except ValueError:
            # Fall back to end of list for unlisted files
            return len(self.order) + 1000


# Sphinx-Gallery configuration
sphinx_gallery_conf = {
    "examples_dirs": "../examples",  # Path to example scripts
    "gallery_dirs": "auto_examples",  # Where to save gallery generated output
    "filename_pattern": r"/example_",  # Match all example_*.py files
    "download_all_examples": False,  # Don't create download all button
    "matplotlib_animations": False,  # Don't try to save animations
    "image_scrapers": ("matplotlib",),  # Scrape matplotlib figures
    "default_thumb_file": None,  # No default thumbnail
    "line_numbers": False,  # Don't show line numbers
    "remove_config_comments": True,  # Remove config comments from code
    "min_reported_time": 0.1,  # Minimum time to report for example execution
    "show_memory": False,  # Don't show memory usage
    "junit": "",  # Don't generate JUnit XML
    # Order subsections: basic first, then advanced, then extensions
    "subsection_order": ExplicitOrder(
        [
            "../examples/basic",
            "../examples/advanced",
            "../examples/extensions",
        ]
    ),
    # Order examples within each subsection
    "within_subsection_order": BasicExampleOrder,
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

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]

# Theme options for PyData Sphinx Theme
html_theme_options = {
    "github_url": "https://github.com/kganitis/bam-engine",
    "show_prev_next": True,
    "navigation_with_keys": True,
    "show_toc_level": 2,
    "navbar_align": "left",
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "icon_links": [
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/bamengine/",
            "icon": "fa-brands fa-python",
        },
    ],
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

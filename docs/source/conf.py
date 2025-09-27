# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('../..'))

# -- Project information -----------------------------------------------------

project = 'Consciousness Mathematics Compression Engine'
copyright = '2024, Consciousness Mathematics Research Team'
author = 'Consciousness Mathematics Research Team'

# The full version, including alpha/beta/rc tags
release = '1.0.0'
version = '1.0.0'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',          # Core autodoc functionality
    'sphinx.ext.autosummary',      # Generate autosummary tables
    'sphinx.ext.doctest',          # Test code in documentation
    'sphinx.ext.intersphinx',      # Link to other projects' documentation
    'sphinx.ext.todo',             # Support for todo items
    'sphinx.ext.coverage',         # Coverage report
    'sphinx.ext.mathjax',          # Math rendering
    'sphinx.ext.viewcode',         # Add source code links
    'sphinx.ext.githubpages',      # GitHub Pages support
    'sphinx.ext.napoleon',         # Google/NumPy style docstrings
    'myst_parser',                 # Markdown support
    'nbsphinx',                    # Jupyter notebook support
    'sphinx_rtd_theme',            # Read the Docs theme
    'sphinx_autodoc_typehints',    # Type hints in signatures
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
source_suffix = {
    '.rst': None,
    '.md': None,
}

# The master toctree document.
master_doc = 'index'

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# Theme options
html_theme_options = {
    'canonical_url': '',
    'analytics_id': '',
    'display_version': True,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': False,
    'vcs_pageview_mode': '',
    'style_nav_header_background': '#2980B9',
    # Toc options
    'collapse_navigation': True,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False
}

# Custom CSS
html_css_files = [
    'css/custom.css',
]

# -- Options for LaTeX output --------------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    'papersize': 'a4paper',

    # The font size ('10pt', '11pt' or '12pt').
    'pointsize': '11pt',

    # Additional stuff for the LaTeX preamble.
    'preamble': r'''
        \usepackage{amsmath}
        \usepackage{amssymb}
        \usepackage{physics}
        \usepackage{siunitx}
        \DeclareSIUnit\phi{\text{\textphi}}
    ''',

    # Latex figure (float) alignment
    'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (master_doc, 'ConsciousnessCompressionEngine.tex',
     'Consciousness Mathematics Compression Engine Documentation',
     'Consciousness Mathematics Research Team', 'manual'),
]

# -- Options for manual page output ------------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    (master_doc, 'consciousnesscompressionengine',
     'Consciousness Mathematics Compression Engine Documentation',
     [author], 1)
]

# -- Options for Texinfo output ----------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (master_doc, 'ConsciousnessCompressionEngine',
     'Consciousness Mathematics Compression Engine Documentation',
     author, 'ConsciousnessCompressionEngine',
     'Revolutionary compression technology using consciousness mathematics.',
     'Miscellaneous'),
]

# -- Extension configuration --------------------------------------------------

# -- Options for intersphinx extension ---------------------------------------

# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/reference/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
}

# -- Options for todo extension ----------------------------------------------

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True

# -- Options for autodoc extension -------------------------------------------

# This value selects what content will be inserted into the main body of an
# autoclass directive.
autoclass_content = 'both'

# This value selects if automatically documented members are sorted
# alphabetical (value 'alphabetical'), by member type (value 'groupwise')
# or by source order (value 'bysource'). The default is alphabetical.
autodoc_member_order = 'bysource'

# The default options for autodoc directives
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
    'member-order': 'bysource',
}

# -- Options for autosummary extension ---------------------------------------

# Generate stub files
autosummary_generate = True

# -- Options for napoleon extension ------------------------------------------

# Parse Google style docstrings
napoleon_google_docstring = True

# Parse NumPy style docstrings
napoleon_numpy_docstring = True

# Include private members
napoleon_include_private_with_doc = False

# Include special members
napoleon_include_special_with_doc = True

# Use parameter descriptions from docstrings in the parameter list
napoleon_use_param = True

# -- Options for MyST parser ------------------------------------------------

# Enable extensions
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "html_admonition",
    "html_image",
    "linkify",
    "replacements",
    "smartquotes",
    "strikethrough",
    "substitution",
    "tasklist",
]

# -- Options for nbsphinx ---------------------------------------------------

# Don't execute notebooks
nbsphinx_execute = 'never'

# Timeout for notebook execution (if enabled)
nbsphinx_timeout = 60

# Allow errors in notebook execution
nbsphinx_allow_errors = True

# -- Custom configuration ---------------------------------------------------

# Suppress warnings for missing references in examples
nitpicky = False

# Suppress specific warnings
suppress_warnings = [
    'ref.python',  # Suppress warnings about missing Python references
]

# MathJax configuration for mathematical formulas
mathjax3_config = {
    'tex': {
        'inlineMath': [['$', '$'], ['\\(', '\\)']],
        'processEscapes': True,
        'processEnvironments': True,
    },
}

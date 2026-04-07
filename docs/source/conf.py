# Copyright 2025 Entalpic
# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "atompack"  # Project name
copyright = "2026, Entalpic"
author = "Ali Ramlaoui and contributors"  # Contributors to the package
release = (
    "0.2.0"  # Current version of the package, can be retrieved programmatically too.
)

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx_design",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinx_math_dollar",
    "sphinx.ext.mathjax",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "autoapi.extension",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
    "code_include.extension",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = []
rst_prolog = """
.. role:: python(code)
    :language: python
    :class: highlight
"""

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_title = (
    "atompack"  # Title for the HTML output for the docs, typically the project's name
)
html_theme = "shibuya"
html_static_path = ["_static"]
html_css_files = ["css/custom.css"]
html_js_files = ["js/custom.js"]
html_logo = "_static/img/entalpic-logo-rounded.png"
html_favicon = "_static/img/entalpic-logo-rounded-96.png"
html_theme_options = {
    "nav_links": [
        {
            "title": "atompack",
            "url": "index",
        },
        {
            "title": "Rust API (internal)",
            "url": "rust-api",
        },
    ],
    # Shibuya supports a standard social icon in the header when these are set.
    "github_url": "https://github.com/LeMaterial/atompack",
}

# -----------------------------
# -----  Plugins configs  -----
# -----------------------------

# Napoleon
# https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html#configuration
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = False
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# https://www.sphinx-doc.org/en/master/usage/extensions/todo.html#directive-todo
todo_include_todos = True


# sphinx.ext.intersphinx
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
}

# sphinx.ext.autodoc & autoapi.extension
# https://autoapi.readthedocs.io/
autodoc_typehints = "signature"
autoapi_type = "python"
autoapi_dirs = [
    "../../atompack-py/python/atompack"
]  # list of paths to the packages to document
autoapi_ignore = ["*.venv/*", "*/tests/*", "*/boilerplate/*", "docs/*"]
autoapi_member_order = "bysource"
autoapi_template_dir = "_templates/autoapi"
autoapi_python_class_content = "class"
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
    "imported-members",
]
autoapi_keep_files = False

# sphinx_math_dollar
# Note: CHTML is the only output format that works with \mathcal{}
mathjax_path = "https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS_CHTML"
mathjax3_config = {
    "tex": {
        "inlineMath": [
            ["$", "$"],
            ["\\(", "\\)"],
        ],
        "processEscapes": True,
    },
    "jax": ["input/TeX", "output/CommonHTML", "output/HTML-CSS"],
}

# sphinx_autodoc_typehints
# https://github.com/tox-dev/sphinx-autodoc-typehints
typehints_fully_qualified = False
always_document_param_types = True
typehints_document_rtype = True
typehints_defaults = "comma"

# MyST
# https://myst-parser.readthedocs.io/en/latest/intro.html
myst_enable_extensions = ["colon_fence"]

# Hover X Ref
# https://sphinx-hoverxref.readthedocs.io/en/latest/index.html
hoverxref_auto_ref = True
hoverxref_mathjax = True

# Open Graph

# ogp_site_url = "https://entalpic-dev-guide.readthedocs.io/en/latest/"
# ogp_social_cards = {
#     "enable": True,
#     # "image": "./_static/images/logo.png",
# }

# :siesta: <update>
# DO NOT change what is between <update> and </update>
# it may be overwritten in subsequent `siesta update` calls
# ----------------------------X---------------------------------


def skip_submodules(app, what, name, obj, skip, options):
    """Skips submodules that are not documented (used by ``autoapi-skip-member`` event).

    Parameters
    ----------
    app : Sphinx
        The Sphinx application object.
    what : str
        The type of the member.
    name : str
        The name of the member (like ``module.Class.attribute`` for instance).
    obj : object
        The Sphinx object representing the member.
    skip : bool
        Whether the member should be skipped (at this point, from the configuration).
    options : list[str]
        The options passed to the directive from ``conf.py``.

    Returns
    -------
    bool
        Whether the member should be skipped.
    """
    if what == "attribute":
        if obj.is_undoc_member:
            print(f"  • Skipping {what} {name} because it is not documented.")
            return True
    return skip


def setup(sphinx):
    """Set up the Sphinx application and adds a hook to skip un-documented attributes.

    Parameters
    ----------
    sphinx : Sphinx
        The Sphinx application object.
    """
    sphinx.connect("autoapi-skip-member", skip_submodules)


# ----------------------------X---------------------------------
# :siesta: </update>

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
from os import environ
from os.path import relpath
from sysconfig import get_paths

_SITE_PACKAGES: str = relpath(get_paths()["purelib"])

project = "MIPCandy"
copyright = "Project Neura"
author = "Project Neura"

root_doc = "index"
language = "en"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_parser",
    "autodoc2",
    "sphinx_design",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx_copybutton"
]
myst_enable_extensions = [
    "amsmath",
    "attrs_inline",
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
    "tasklist"
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

autodoc2_packages = [
    _SITE_PACKAGES + "/mipcandy",
]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_baseurl = environ.get("READTHEDOCS_CANONICAL_URL", "")
html_title = "MIP Candy Docs"
html_logo = "https://mipcandy.projectneura.org/assets/logo.png"
html_favicon = "https://mipcandy.projectneura.org/assets/logo.png"
html_theme = "sphinx_book_theme"
html_theme_options = {
    "home_page_in_toc": True,
    "github_url": "https://github.com/ProjectNeura/MIPCandy",
    "repository_url": "https://github.com/ProjectNeura/mipcandy-docs",
    "repository_branch": "master",
    "path_to_docs": "docs",
    "use_repository_button": True,
    "use_edit_page_button": False,
    "use_issues_button": True,
    "announcement": "Docs to be completed",
}
html_static_path = ["_static"]

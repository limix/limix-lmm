from __future__ import unicode_literals

import sphinx_rtd_theme


def get_version():
    import limix_lmm

    return limix_lmm.__version__


extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
]
templates_path = ["_templates"]
source_suffix = ".rst"
master_doc = "index"
project = "Limix-LMM"
copyright = "2017, Rachel Moore, Francesco Paolo Casale, Oliver Stegle"
author = "Rachel Moore, Francesco Paolo Casale, Oliver Stegle"
version = get_version()
release = version
language = None
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "conf.py"]
pygments_style = "sphinx"
todo_include_todos = False
html_theme = "sphinx_rtd_theme"
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
htmlhelp_basename = "struclmmdoc"
latex_elements = {}
latex_documents = [
    (
        master_doc,
        "limix_lmm.tex",
        "Limix-LMM Documentation",
        "Rachel Moore, Francesco Paolo Casale, Oliver Stegle",
        "manual",
    )
]
man_pages = [(master_doc, "limix_lmm", "Limix-LMM Documentation", [author], 1)]
texinfo_documents = [
    (
        master_doc,
        "limix_lmm",
        "Limix-LMM Documentation",
        author,
        "limix_lmm",
        "A mixed-model approach to model complex GxE signals.",
        "Miscellaneous",
    )
]
intersphinx_mapping = {"https://docs.python.org/": None, "http://matplotlib.org": None}

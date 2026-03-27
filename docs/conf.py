"""msaf documentation build configuration file."""

import importlib.util
import os
import sys

sys.path.insert(0, os.path.abspath("../"))

needs_sphinx = "4.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.doctest",
    "numpydoc",
    "sphinx.ext.autosummary",
]

autosummary_generate = True

try:
    from matplotlib.sphinxext import plot_directive  # noqa: F401

    extensions.append("matplotlib.sphinxext.plot_directive")
    use_matplotlib_plot_directive = True
except ImportError:
    use_matplotlib_plot_directive = False

numpydoc_use_plots = True

doctest_global_setup = """
import numpy as np
import scipy
import msaf
np.random.seed(123)
np.set_printoptions(precision=3, linewidth=64, edgeitems=2, threshold=200)
"""

plot_pre_code = """
import numpy as np
import msaf
np.random.seed(123)
np.set_printoptions(precision=3, linewidth=64, edgeitems=2, threshold=200)
"""
plot_include_source = True
plot_formats = [("png", 96)]
plot_html_show_formats = False

font_size = 13 * 72 / 96.0

plot_rcparams = {
    "font.size": font_size,
    "axes.titlesize": font_size,
    "axes.labelsize": font_size,
    "xtick.labelsize": font_size,
    "ytick.labelsize": font_size,
    "legend.fontsize": font_size,
    "figure.subplot.bottom": 0.2,
    "figure.subplot.left": 0.2,
    "figure.subplot.right": 0.9,
    "figure.subplot.top": 0.85,
    "figure.subplot.wspace": 0.4,
    "text.usetex": False,
}

if not use_matplotlib_plot_directive:
    import matplotlib

    matplotlib.rcParams.update(plot_rcparams)

numpydoc_show_class_members = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
    "librosa": ("https://librosa.org/doc/latest/", None),
}

templates_path = ["_templates"]
source_suffix = ".rst"
master_doc = "index"

project = "msaf"
copyright = "2015-2026, Oriol Nieto"

spec = importlib.util.spec_from_file_location(
    "msaf.version", os.path.abspath("../msaf/version.py")
)
msaf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(msaf)

version = msaf.short_version
release = msaf.version

exclude_patterns = ["_build"]
default_role = "autolink"
add_function_parentheses = False
add_module_names = True
show_authors = False
pygments_style = "sphinx"

# -- HTML output --

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_domain_indices = True
html_use_index = True
html_use_modindex = True
htmlhelp_basename = "msafdoc"

# -- LaTeX output --

latex_documents = [
    ("index", "msaf.tex", "msaf Documentation", "Oriol Nieto", "manual"),
]

# -- Manual page output --

man_pages = [("index", "msaf", "msaf Documentation", ["Oriol Nieto"], 1)]

autodoc_member_order = "bysource"

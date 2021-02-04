(TeX-add-style-hook
 "paper.tex"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-class-options
                     '(("tufte-handout" "nobib")))
   (TeX-add-to-alist 'LaTeX-provided-package-options
                     '(("xcolor" "svgnames") ("subfig" "caption=false") ("biblatex" "style=authoryear" "backend=bibtex" "natbib" "maxcitenames=2" "doi=false")))
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "href")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperref")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperimage")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperbaseurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "nolinkurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "url")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "path")
   (add-to-list 'LaTeX-verbatim-macros-with-delims-local "path")
   (TeX-run-style-hooks
    "latex2e"
    "article"
    "art10"
    "tufte-handout"
    "tufte-handout10"
    "xcolor"
    "times"
    "hyperref"
    "url"
    "amsmath"
    "amssymb"
    "mathtools"
    "cleveref"
    "svg"
    "bm"
    "booktabs"
    "multirow"
    "grffile"
    "pgfplots"
    "subfig"
    "wrapfig"
    "microtype"
    "xspace"
    "tikz"
    "biblatex")
   (TeX-add-symbols
    '("todo" 1)
    '("Set" 1)
    "AR"
    "bAR"
    "risk"
    "foi"
    "E")
   (LaTeX-add-labels
    "sec:orgd22a04a"
    "sec:org06d6d51"
    "sec:org2b52466"
    "eq:rnn"
    "sec:org9b3d94e"
    "eq:beta-ar"
    "eq:objective"
    "sec:org8f398bc"
    "fig:dispersion"
    "sec:orgaee16ad")
   (LaTeX-add-bibliographies
    "./references"))
 :latex)


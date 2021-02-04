(TeX-add-style-hook
 "paper"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-package-options
                     '(("xcolor" "svgnames") ("subfig" "caption=false")))
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
    "xcolor"
    "times"
    "hyperref"
    "url"
    "icml2021"
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
    "tikz")
   (TeX-add-symbols
    '("todo" 1)
    '("Set" 1)
    "AR"
    "bAR"
    "risk"
    "foi"
    "E")
   (LaTeX-add-labels
    "sec:orgcdd5422"
    "fig:ranking-covidhub-mae"
    "fig:county-variability"
    "sec:org6298d16"
    "eq:foi-ar"
    "eq:rnn"
    "sec:orgac99b1b"
    "eq:beta-ar"
    "sec:org2abffde"
    "fig:dispersion"
    "sec:org751d6db"
    "eq:objective"
    "sec:org99c3fc8"
    "tab:forecasts"
    "tab:data-sources"
    "fig:mae-covidhub"
    "fig:mae-google"
    "fig:mae-covidhub-granger"
    "fig:quality-ratio"
    "fig:mae-covidhub-loss"
    "sec:org3676b8c"
    "sec:orgf720d80")
   (LaTeX-add-bibliographies
    "references"))
 :latex)


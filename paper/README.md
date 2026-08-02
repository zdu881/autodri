# AutoUI Method Paper

This directory contains the maintained manuscript, its bibliography, generated figures, and the compiled PDF.

## Contents

- `autoui.tex`: manuscript source
- `method_citation_entries.bib`: bibliography
- `figures/`: generated publication figures
- `autoui.pdf`: compiled manuscript

## Rebuild

From the repository root:

```bash
python scripts/draw_autoui_figures.py
latexmk -cd -pdf paper/autoui.tex
```

The figure script is deterministic. LaTeX intermediate files are ignored; the maintained PDF is committed alongside the source.

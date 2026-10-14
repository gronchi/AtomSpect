# Run from doc/, or use make -C doc from the repository root.
DOC = AtomSpect_theory_manual
PYTHON ?= python
PDFLATEX ?= pdflatex

.DEFAULT_GOAL := all
all: $(DOC).pdf

$(DOC).pdf: $(DOC).tex $(wildcard figures/*.pdf) $(wildcard figures/data/*.tex)
	$(PDFLATEX) -interaction=nonstopmode -halt-on-error $(DOC).tex
	$(PDFLATEX) -interaction=nonstopmode -halt-on-error $(DOC).tex
	@echo "built $(DOC).pdf"

# Re-run the analytic checks and redraw every figure and data table.
data:
	cd .. && $(PYTHON) doc/scripts/make_figures.py --data-only

plots:
	cd .. && $(PYTHON) doc/scripts/plot_figures.py

figures:
	cd .. && $(PYTHON) doc/scripts/make_figures.py

# Run the minimal example.
check:
	cd .. && $(PYTHON) doc/scripts/example_minimal.py

rebuild: figures
	$(MAKE) all

clean:
	$(PYTHON) -c "from pathlib import Path; [Path('$(DOC).' + ext).unlink(missing_ok=True) for ext in ('aux', 'log', 'out', 'toc')]"

.PHONY: all data plots figures check rebuild clean

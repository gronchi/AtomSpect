"""Build the manual with Python on Windows or Unix; optionally regenerate figures."""
import argparse
import subprocess
import sys
from pathlib import Path


def main():
    """Regenerate requested inputs and compile twice, propagating build failures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rebuild', action='store_true', help='regenerate checked data and figures first')
    parser.add_argument('--pdflatex', default='pdflatex', help='path to pdflatex executable')
    args = parser.parse_args()
    doc = Path(__file__).resolve().parents[1]
    if args.rebuild:
        subprocess.run([sys.executable, str(doc/'scripts'/'make_figures.py')], cwd=doc.parent, check=True)
    for _ in range(2):
        subprocess.run([args.pdflatex, '-interaction=nonstopmode', '-halt-on-error',
                        'AtomSpect_theory_manual.tex'], cwd=doc, check=True)
    print(f'Built {doc / "AtomSpect_theory_manual.pdf"}')


if __name__ == '__main__':
    main()

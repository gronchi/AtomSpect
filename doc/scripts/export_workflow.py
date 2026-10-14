"""Export the manual's TikZ calculation flowchart for the repository README.

Requires pdflatex (with standalone, TikZ and Latin Modern) and PyMuPDF.
The diagram itself is read from AtomSpect_theory_manual.tex.
"""

import argparse
from pathlib import Path
import re
import subprocess
import tempfile

import fitz


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdflatex", default="pdflatex")
    args = parser.parse_args()
    doc = Path(__file__).resolve().parents[1]
    source = (doc / "AtomSpect_theory_manual.tex").read_text(encoding="utf-8")
    label = source.index(r"\label{fig:flow}")
    start = source.rindex(r"\begin{figure}", 0, label)
    figure = source[start:label]
    diagram = re.search(r"\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}",
                        figure, re.DOTALL).group()
    libraries = re.search(r"\\usetikzlibrary\{[^}]+\}", source).group()
    routine = next(line for line in source.splitlines()
                   if line.startswith(r"\newcommand{\routine}"))
    document = "\n".join([
        r"\documentclass[11pt,tikz,border=8pt]{standalone}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage{lmodern}",
        libraries,
        routine,
        r"\begin{document}",
        diagram,
        r"\end{document}",
    ])
    with tempfile.TemporaryDirectory(prefix="atomspect-workflow-") as directory:
        work = Path(directory)
        (work / "workflow.tex").write_text(document, encoding="utf-8")
        result = subprocess.run(
            [args.pdflatex, "-interaction=nonstopmode", "-halt-on-error", "workflow.tex"],
            cwd=work, capture_output=True, text=True,
        )
        if result.returncode:
            raise RuntimeError(f"Flowchart compilation failed:\n{result.stdout}\n{result.stderr}")
        with fitz.open(work / "workflow.pdf") as pdf:
            svg = pdf[0].get_svg_image(matrix=fitz.Matrix(2, 2), text_as_path=True)
    # Keep the manual's black text readable in both light and dark Markdown themes.
    boundary = svg.index(">") + 1
    svg = (svg[:boundary] + '\n<rect width="100%" height="100%" fill="white"/>\n'
           + svg[boundary:])
    output = doc / "figures" / "calculation_flow.svg"
    output.write_text(svg, encoding="utf-8")
    print(f"Exported {output}")


if __name__ == "__main__":
    main()

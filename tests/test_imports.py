"""`import AtomSpect` must work headless: no matplotlib.pyplot and no tkinter until plotting is used."""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run(code):
    env = dict(os.environ, MPMATH_NOGMPY="1", SYMPY_GROUND_TYPES="python", PYTHONPATH=str(ROOT))
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, cwd=ROOT)
    assert out.returncode == 0, out.stderr
    return out.stdout.strip()


def test_import_does_not_load_plotting_backends():
    loaded = _run("import sys, AtomSpect; AtomSpect.AtomSpect_Main; "
                  "print(any(m in sys.modules for m in ('tkinter', 'matplotlib.pyplot', 'AtomSpect.plotting')))")
    assert loaded == "False"


def test_plotting_names_load_on_first_use():
    result = _run("import sys, AtomSpect; f = AtomSpect.PlotFunction; from AtomSpect import MakeSlider, plotZfan; "
                  "print(callable(f) and callable(MakeSlider) and callable(plotZfan), "
                  "'matplotlib.pyplot' in sys.modules, 'tkinter' in sys.modules, 'MakeSlider' in dir(AtomSpect))")
    assert result == "True True False True"

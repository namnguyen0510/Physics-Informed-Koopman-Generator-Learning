"""make_figures.py -- render every diagnostic figure and the numeric summary.

usage: python -m r2r_nn.make_figures --model M1 [--data datasets --models models --out figures --reports reports]
"""
import argparse

from . import style
from .fnn import load_model_data
from . import plots_data, plots_models, report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default="out/datasets")
    ap.add_argument("--models", default="out/models")
    ap.add_argument("--out", default="out/figures")
    ap.add_argument("--reports", default="out/reports")
    ap.add_argument("--only", default="data,models,report")
    a = ap.parse_args()
    style.apply()
    fams = load_model_data(a.data, a.model)
    parts = a.only.split(",")
    if "data" in parts:
        plots_data.make_all(a.model, fams, f"{a.out}/{a.model}/dataset")
        print("dataset figures done", flush=True)
    if "models" in parts:
        plots_models.make_all(a.model, fams, f"{a.models}/{a.model}", f"{a.out}/{a.model}/models")
        print("model figures done", flush=True)
    if "report" in parts:
        report.make(a.model, fams, f"{a.models}/{a.model}", a.reports)
        print("report done", flush=True)


if __name__ == "__main__":
    main()

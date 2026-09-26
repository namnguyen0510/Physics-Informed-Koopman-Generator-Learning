"""Shared figure style (validated categorical palette, hairline chrome, one axis per panel)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# categorical slots in fixed order (validated: adjacent CVD dE >= 9.1, normal >= 19.6)
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")
SLOTS = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED]
INK, INK2, MUTED, GRID, AXIS, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#ffffff"
REFC = INK2            # reference trajectories: secondary ink, dashed
CONTEXT = "#c9c8c2"    # de-emphasised context data

SEQ = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
       "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
CMAP_SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#f4f8fe"] + SEQ)
CMAP_DIV = LinearSegmentedColormap.from_list(
    "div_blue_red", ["#104281", "#3987e5", "#9ec5f4", "#f0efec", "#f3a7a6", "#e34948", "#9c2626"])

# controller identity (fixed across every figure)
# order expert, raw_dagger, phys_dagger, raw_bc, phys_bc = validated slot order 1-5 (adjacent bars stay CVD-safe);
# raw = orange and phys = aqua everywhere (feature-set identity), BC variants take slots 4-5
CTRL_COLORS = {"expert": BLUE, "raw_dagger": ORANGE, "phys_dagger": AQUA, "raw_bc": YELLOW, "phys_bc": MAGENTA}
CTRL_LABELS = {"expert": "Expert (paper SMC)", "raw_bc": "FNN-raw BC", "raw_dagger": "FNN-raw DAgger",
               "phys_bc": "FNN-phys BC", "phys_dagger": "FNN-phys DAgger"}
FEAT_COLORS = {"raw": ORANGE, "phys": AQUA}
FEAT_LABELS = {"raw": "FNN-raw", "phys": "FNN-phys"}


def apply():
    plt.rcParams.update({
        "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
        "font.family": "DejaVu Sans", "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
        "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.titlecolor": INK,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.6, "axes.labelcolor": INK2,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "grid.linestyle": "-",
        "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
        "lines.linewidth": 1.4, "legend.frameon": False, "legend.fontsize": 7.5,
        "legend.labelcolor": INK2, "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight",
        "text.color": INK, "mathtext.default": "regular", "axes.formatter.useoffset": False,
    })


def save(fig, path_noext, pdf=True):
    fig.savefig(path_noext + ".png")
    if pdf:
        fig.savefig(path_noext + ".pdf")
    plt.close(fig)


def suptitle(fig, text, sub=None):
    """Left-aligned title (+ optional subtitle) in a fixed-height header band.
    Sets fig._top = the axes-area top to pass to tight_layout(rect=...)."""
    h = fig.get_figheight()
    fig.text(0.01, 1 - 0.10 / h, text, ha="left", va="top", fontsize=10.5, fontweight="bold", color=INK)
    band = 0.36
    if sub:
        fig.text(0.01, 1 - 0.36 / h, sub, ha="left", va="top", fontsize=8, color=INK2)
        band = 0.58
    fig._top = 1 - band / h
    return fig._top

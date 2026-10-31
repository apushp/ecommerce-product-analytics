"""
Shared chart look for every analysis script: one blue accent, everything else
grey, titles that say the finding with a smaller subtitle under them.
"""
import matplotlib.pyplot as plt

# one accent, everything else grey, so the eye goes to what the title is about
ACCENT = "#2a78d6"
GREY = "#b4b2ab"
INK = "#0b0b0b"
INK_2 = "#52514e"
SURFACE = "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": GREY,
    "axes.labelcolor": INK_2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.color": "#e6e5e0",
    "grid.linewidth": 0.8,
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "font.size": 10.5,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.titlecolor": INK,
})


def title(ax, headline, sub):
    ax.set_title(headline, pad=26)
    ax.text(0, 1.02, sub, transform=ax.transAxes, color=INK_2, fontsize=10)


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)

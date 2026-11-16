"""The talk's figures, drawn from data/ alone (data/paper/ and data/network.gpkg, written by
sync_from_paper.py). The logic follows the paper's figures/paper_figures.py; the drawing is resized for
1280 x 720 slides, split into single panels, and labelled for a broad audience.

Run from the deck root:  python3 scripts/talk_figures.py [figure names, to draw only those]
Writes figures/<name>.svg (maps: .png).
"""

import json
import re
import sys
from pathlib import Path

import logging

import matplotlib

matplotlib.use("Agg")
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)  # Source Sans Pro is for the browser
import matplotlib.patheffects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parent.parent
PAPER = ROOT / "data" / "paper"
GPKG = ROOT / "data" / "network.gpkg"
OUT = ROOT / "figures"
G = PAPER / "scenarios" / "greenway"
E = PAPER / "scenarios" / "equity"

CONST = yaml.safe_load((PAPER / "constants.yaml").read_text())
PARAMS = yaml.safe_load((PAPER / "config" / "params.yaml").read_text())
COMP = CONST["composite"]
PLANT = CONST["outlet"]
ARRIVING = {"WT2203", "WT1420", "OS2272"}

# The paper's palette (checked for colour-vision deficiency): model or upstream blue, plant orange,
# a third series aqua (always with a marker or label), grey for context. Western purple for emphasis.
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
GREY, INK, MUTED, GRID = "#8a8984", "#1a1a1a", "#52514e", "#e4e3df"
PURPLE = "#4B2682"

# Every stream that reaches the plant: the three arriving sewers and the force main of the Berkshire
# Village pumping station, which the paper's zoning cuts on its own. Marker, colour, and legend label for each.
STREAMS = {
    "WT2203": ("^", BLUE, "Sewer WT2203"),
    "WT1420": ("s", AQUA, "Sewer WT1420"),
    "OS2272": ("v", YELLOW, "Sewer OS2272"),
    "WT6005": ("D", "#8a5cc9", "Force main WT6005"),  # a lighter purple, for the lightness band
}
assert ARRIVING < set(STREAMS)
STOP_PROB = 0.95  # a zone is named when its posterior probability reaches this (paper: AdaptStopPct)

W_FULL, H_FULL = 11.0, 3.7  # inches; 100 px per inch on a 1280 x 720 slide (full-width plots)
W_COL, H_COL = 7.2, 3.7  # plots beside an equation or text column
plt.rcParams.update({
    "font.family": ["Source Sans Pro", "Helvetica Neue", "Helvetica", "Arial", "sans-serif"],
    "font.size": 15, "axes.labelsize": 15, "axes.titlesize": 15, "legend.fontsize": 13,
    "xtick.labelsize": 13, "ytick.labelsize": 13, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "lines.linewidth": 2.2, "legend.frameon": False, "svg.fonttype": "none",
    "savefig.bbox": "tight", "savefig.pad_inches": 0.05, "savefig.transparent": False,
})


def grid(ax, axis="y"):
    ax.grid(axis=axis, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def save(fig, name, raster=False):
    """SVG for plots; PNG for the maps, whose 13,000 pipes would make a multi-megabyte SVG."""
    OUT.mkdir(exist_ok=True)
    path = OUT / f"{name}.{'png' if raster else 'svg'}"
    fig.savefig(path, dpi=220 if raster else None)
    plt.close(fig)
    print(path.relative_to(ROOT))


def sens():
    d = np.load(G / "sensitivities.npz")
    sites, zones = [str(s) for s in d["sites"]], [str(z) for z in d["zones"]]
    return pd.DataFrame(d[f"W_{COMP}"], index=sites, columns=zones), d


def zones_summary():
    return pd.read_csv(PAPER / "scenarios" / "zones" / "zones_summary.csv")


def n_sites() -> int:
    W, _ = sens()
    return int(np.isfinite(W.to_numpy()).all(axis=1).sum())


def pass_through(W, summary) -> pd.Series:
    """True for a zone whose outlet site carries other zones too (the paper's OWN_SHARE rule)."""
    cut = summary.set_index("zone")["cut_node"]
    return pd.Series({z: not (W.loc[cut[z], z] > CONST["own_share"]) for z in cut.index if z != "rest"})


# ------------------------------------------------------------------------------------------- maps
class Net:
    """The catchment's map layers from network.gpkg."""

    def __init__(self):
        import geopandas as gpd
        self.pipes = gpd.read_file(GPKG, layer="pipes")
        self.nodes = gpd.read_file(GPKG, layer="nodes").set_index("node")
        self.base = {k: gpd.read_file(GPKG, layer=f"basemap_{k}") for k in
                     ("water", "watercourses", "roads_major", "roads_minor")}
        self.bounds = self.pipes.total_bounds

    def basemap(self, ax, minor=False, scale=0.7):
        if minor:
            self.base["roads_minor"].plot(ax=ax, color="#ececec", linewidth=0.3 * scale, zorder=1)
        self.base["roads_major"].plot(ax=ax, color="#cfcfcf", linewidth=0.8 * scale, zorder=2)
        self.base["watercourses"].plot(ax=ax, color="#a9cfe8", linewidth=0.6 * scale, zorder=3)
        self.base["water"].plot(ax=ax, color="#a9cfe8", linewidth=0, zorder=4)
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel(""); ax.set_ylabel("")
        for sp in ax.spines.values():
            sp.set_visible(False)

    def frame(self, ax, pad=400, scalebar=True):
        xmin, ymin, xmax, ymax = self.bounds
        ax.set_xlim(xmin - pad, xmax + pad)
        ax.set_ylim(ymin - pad, ymax + pad)
        if scalebar:
            ax.plot([xmax - 2400, xmax - 400], [ymin - 150, ymin - 150], color=INK, lw=2, zorder=12)
            ax.text(xmax - 1400, ymin, "2 km", ha="center", va="bottom", fontsize=12, zorder=12)

    def pipes_lc(self, colours, wmin=0.3, wmax=4.0, zorder=5):
        segs, cols, widths = [], [], []
        for geom, c, f in zip(self.pipes.geometry, colours, self.pipes.up_frac):
            w = wmin + wmax * np.sqrt(f)
            for part in getattr(geom, "geoms", [geom]):
                segs.append(np.asarray(part.coords)); cols.append(c); widths.append(w)
        o = np.argsort(widths, kind="stable")
        return LineCollection([segs[i] for i in o], colors=[cols[i] for i in o],
                              linewidths=[widths[i] for i in o], capstyle="round", zorder=zorder)

    def xy(self, node):
        g = self.nodes.geometry[node]
        return g.x, g.y


def width_legend(wmin=0.3, wmax=4.0, colour=MUTED):
    return [Line2D([], [], color=colour, lw=wmin + wmax * np.sqrt(f), label=f"{f:.0%} of people upstream")
            for f in (0.01, 0.1, 0.5)]


def fig_catchment():
    """Slide 2: the Greenway sewers, pipe width by the people upstream, and the plant."""
    n = Net()
    fig, ax = plt.subplots(figsize=(4.3, 6.1))
    n.basemap(ax, minor=True)
    ax.add_collection(n.pipes_lc([PURPLE] * len(n.pipes)))
    x, y = n.xy(PLANT)
    ax.scatter([x], [y], s=260, marker="*", color=ORANGE, edgecolor="white", linewidths=1.2, zorder=10)
    ax.annotate("treatment plant", (x, y), xytext=(-14, -26), textcoords="offset points", ha="right",
                fontsize=14, color=INK, zorder=11,
                path_effects=[matplotlib.patheffects.withStroke(linewidth=3.5, foreground="white")])
    n.frame(ax)
    xmin, xmax = ax.get_xlim()
    ax.set_xlim(xmin, xmax + 500)  # a little more room in the north-east corner for the legend
    ax.legend(handles=width_legend(colour=PURPLE), loc="upper right", bbox_to_anchor=(1.2, 1.0),
              fontsize=11, frameon=True, facecolor="white",
              edgecolor="none", framealpha=0.9, borderpad=0.3, handletextpad=0.4, labelspacing=0.5,
              borderaxespad=0.2)  # as on the zones map: the north-east corner, partly past the map's edge
    save(fig, "catchment", raster=True)


def fig_dilution(zone="Z09"):
    """Slide 2: one neighbourhood's share of the composite at each sampling point on its way to the plant
    (the representation fractions W along the path of largest flow), beside a map of that path. Kept
    generic (no zone numbers or site IDs): zones are introduced later in the talk."""
    from matplotlib.collections import LineCollection
    n = Net()
    W, _ = sens()
    s = zones_summary().set_index("zone")
    _, hulls = zone_colours(n)
    down = n.pipes.sort_values("up_pop", ascending=False).drop_duplicates("from_node").set_index("from_node")
    node, path, segs = s.at[zone, "cut_node"], [], []
    path.append(node)
    while node != PLANT:  # follow the pipe carrying the most people, as for the zone tree
        r = down.loc[node]
        segs += [np.asarray(p.coords) for p in getattr(r.geometry, "geoms", [r.geometry])]
        node = r.to_node
        path.append(node)
    stops = [p for p in path if p in W.index]
    total = int(round(s.population.sum()))
    pop = {p: int(down.loc[p].up_pop) if p in down.index else total for p in stops}
    zone_of = {c: z for z, c in s["cut_node"].items() if z != "rest"}

    def name(p):
        if p == PLANT:
            return "The treatment plant"
        if p in STREAMS:
            return "A trunk sewer arriving at the plant"
        return "Where its sewers leave it" if zone_of[p] == zone else "Further downstream"
    fig = plt.figure(figsize=(11, 4.0))
    ax = fig.add_axes([0.0, 0.0, 0.30, 1.0])
    n.basemap(ax, scale=0.5)
    ax.add_collection(n.pipes_lc(["#c9c6c0"] * len(n.pipes), wmin=0.15, wmax=2.2))
    h = hulls[zone]
    for part in getattr(h, "geoms", [h]):
        x, y = part.exterior.xy
        ax.fill(x, y, color=ORANGE, alpha=0.22, lw=0, zorder=3)
        ax.plot(x, y, color=ORANGE, lw=1.2, zorder=4)
    ax.add_collection(LineCollection(segs, colors=ORANGE, linewidths=3.2, zorder=6))
    for i, p in enumerate(stops, 1):
        x, y = n.xy(p)
        dark = p == PLANT
        ax.scatter([x], [y], s=300, color=INK if dark else "white", edgecolor=INK, linewidths=1.6, zorder=8)
        ax.text(x, y, str(i), ha="center", va="center", fontsize=11, fontweight="bold",
                color="white" if dark else INK, zorder=9)
    # the half of the catchment the path crosses: path and zone with a margin
    xs = np.concatenate([g[:, 0] for g in segs] + [np.asarray(h.bounds)[[0, 2]]])
    ys = np.concatenate([g[:, 1] for g in segs] + [np.asarray(h.bounds)[[1, 3]]])
    ax.set_xlim(xs.min() - 1000, xs.max() + 400); ax.set_ylim(ys.min() - 400, ys.max() + 1700)
    b = fig.add_axes([0.54, 0.13, 0.37, 0.76])
    yy = np.arange(len(stops))[::-1]
    for y, (i, p) in zip(yy, enumerate(stops, 1)):
        sh = float(W.loc[p, zone])
        b.barh(y, 100 * sh, color=ORANGE, height=0.56, zorder=3)
        b.barh(y, 100 - 100 * sh, left=100 * sh, color="#e4e3df", height=0.56, zorder=3)
        lab = f"{100 * sh:.0f} %" if sh > 0.1 else f"{100 * sh:.1f} %"
        inside = sh > 0.9
        b.text(100 * sh + (-2 if inside else 2), y, lab, va="center", ha="right" if inside else "left", fontsize=15,
               fontweight="bold", color="white" if inside else INK, zorder=4)
        b.text(103, y, f"{pop[p]:,} people\nupstream", va="center", fontsize=11, color=MUTED)
        b.text(-4, y, f"{name(p)}  {i}", ha="right", va="center", fontsize=13, color=INK)
    b.set_yticks([]); b.set_xlim(0, 100); b.set_xticks([0, 50, 100], ["0", "50", "100 %"])
    for sp in ("top", "right", "left"):
        b.spines[sp].set_visible(False)
    fig.text(0.54, 0.95, "One neighbourhood's share of the sample, on its way to the plant", fontsize=13, color=INK)
    save(fig, "dilution", raster=True)


def fig_network():
    """Slide 6: the model network, with the force mains and pumping stations."""
    n = Net()
    fig, ax = plt.subplots(figsize=(4.3, 6.1))
    n.basemap(ax)
    fm = n.pipes.force_main.to_numpy()
    ax.add_collection(n.pipes_lc(np.where(fm, ORANGE, BLUE), wmin=0.25, wmax=3.0))
    ps = n.nodes[n.nodes.pump_station]
    ax.scatter(ps.geometry.x, ps.geometry.y, s=40, marker="s", color=INK, edgecolor="white", linewidths=0.8, zorder=10)
    x, y = n.xy(PLANT)
    ax.scatter([x], [y], s=220, marker="*", color=ORANGE, edgecolor="white", linewidths=1.2, zorder=11)
    n.frame(ax)
    xmin, xmax = ax.get_xlim()
    ax.set_xlim(xmin, xmax + 500)  # a little more room in the north-east corner for the legend
    ax.legend(handles=[Line2D([], [], color=BLUE, lw=2, label="gravity sewer"),
                       Line2D([], [], color=ORANGE, lw=2, label="force main"),
                       Line2D([], [], ls="", marker="s", color=INK, ms=7, label="pumping station"),
                       Line2D([], [], ls="", marker="*", color=ORANGE, ms=14, label="treatment plant")],
              loc="upper right", bbox_to_anchor=(1.2, 1.0), fontsize=11, frameon=True, facecolor="white",
              edgecolor="none", framealpha=0.9, borderpad=0.3, handletextpad=0.4, labelspacing=0.5,
              borderaxespad=0.2)  # as on the zones map: the north-east corner, partly past the map's edge
    save(fig, "network", raster=True)


BLOCK_BINS = [0, 10, 25, 40, 60, np.inf]  # people per hectare
BLOCK_RAMP = ["#e9e1f4", "#c9b5e6", "#a283d1", "#7651b4", "#4b2682"]  # one hue, light to dark


def block_colours(density):
    """The fill of each block by its people per hectare."""
    idx = np.clip(np.digitize(density, BLOCK_BINS) - 1, 0, len(BLOCK_RAMP) - 1)
    return [BLOCK_RAMP[i] for i in idx]


def block_legend():
    labs = [f"under {BLOCK_BINS[1]}"] + [f"{a}–{b}" for a, b in zip(BLOCK_BINS[1:-2], BLOCK_BINS[2:-1])] + \
           [f"{BLOCK_BINS[-2]} or more"]
    return [Patch(facecolor=c, edgecolor="white", label=l) for c, l in zip(BLOCK_RAMP, labs)]


def zoom_window(blocks, size):
    """The size x size window, centred on a block, that holds the most blocks with people in it."""
    pts = blocks[blocks.population > 0].geometry.representative_point()
    xy = np.c_[pts.x, pts.y]
    counts = [(np.abs(xy - p) < size / 2).all(axis=1).sum() for p in xy]
    cx, cy = xy[int(np.argmax(counts))]
    return cx - size / 2, cy - size / 2, cx + size / 2, cy + size / 2


def fig_flow_inset():
    """Slide 7, beside the block maps: the daily flow pattern, small. The household input, the model at
    the plant, and the plant's dry-day maximum and minimum (median and IQR)."""
    q = pd.read_csv(PAPER / "hydraulics" / "greenway" / "outlet_flow_day3.csv")
    s = PARAMS["loading"]["diurnal_pattern"]
    p = pd.read_csv(PAPER / "data" / "plant" / "greenway_daily_mps.csv")
    dd = p[p["dry"].astype(bool)]
    fig, ax = plt.subplots(figsize=(3.7, 2.4))
    grid(ax)
    ax.step(np.arange(25), np.r_[s, s[0]], where="post", color=GREY, lw=1.3)
    his = []
    for col in ("max_over_total", "min_over_total"):
        lo, med, hi = dd[col].quantile([0.25, 0.5, 0.75])
        ax.axhspan(lo, hi, color=ORANGE, alpha=0.15, lw=0)
        ax.axhline(med, color=ORANGE, lw=1.8, ls=(0, (4, 2.5)))
        his.append(hi)
    ax.plot(q["hour"], q["flow_L_s"] / q["flow_L_s"].mean(), color=BLUE, lw=2.4)
    ax.set_xlim(0, 24); ax.set_xticks(range(0, 25, 6)); ax.set_ylim(0, 2.0); ax.set_yticks([0, 1, 2])
    ax.tick_params(labelsize=12)
    ax.set_xlabel("Hour of day", fontsize=12, labelpad=2); ax.set_ylabel("Flow / daily mean", fontsize=12)
    # direct labels in place of a legend
    ax.text(0.3, his[0] + 0.04, "plant, dry days\n(max, min)", color="#c2501c", fontsize=11.5, ha="left",
            va="bottom", linespacing=1.0)
    ax.text(3.8, 0.84, "model,\nat the plant", color=BLUE, fontsize=11.5, ha="center", va="bottom", linespacing=1.0)
    ax.text(14.0, 0.08, "household use (input)", color=MUTED, fontsize=11.5, ha="center", va="bottom")
    save(fig, "flow_inset")


def fig_blocks_legend():
    """Slide 7: the legend of the block maps, on its own so the slide can stack it over the flow inset."""
    fig, ax = plt.subplots(figsize=(3.9, 1.72))
    ax.axis("off")
    h = block_legend() + [Patch(facecolor="none", edgecolor="#c9c8c3", label="no one"),
                          Line2D([], [], color=BLUE, lw=2, label="sewer"),
                          Line2D([], [], ls="", marker="o", color=ORANGE, ms=8, label="people placed at a node")]
    ax.legend(handles=h[:5], title="People per hectare", loc="upper left", bbox_to_anchor=(0, 1), fontsize=12,
              title_fontsize=12.5, alignment="left", labelspacing=0.12, borderaxespad=0, handleheight=0.9)
    ax.add_artist(ax.get_legend())
    ax.legend(handles=h[5:], loc="upper left", bbox_to_anchor=(0.45, 0.82), fontsize=12, labelspacing=0.45,
              borderaxespad=0, handlelength=1.4)
    save(fig, "blocks_legend")


def fig_blocks(zoom_m=1200):
    """Slide 7: the census dissemination blocks that load the catchment, shaded by people per hectare.
    Left, the whole catchment with the zoomed window marked; right, the window, with the pipes and the
    people each node receives. The legend is fig_blocks_legend."""
    import geopandas as gpd
    from matplotlib.patches import Rectangle
    n = Net()
    blocks = gpd.read_file(GPKG, layer="blocks")
    has = blocks.population.to_numpy() > 0
    fill = block_colours(blocks.population.to_numpy() / (blocks.area.to_numpy() / 1e4))
    x0, y0, x1, y1 = zoom_window(blocks, zoom_m)
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.6, 4.6), gridspec_kw=dict(width_ratios=[0.74, 1], wspace=0.04))
    for ax, lw in ((a, 0.15), (b, 0.6)):
        n.basemap(ax, minor=ax is b)
        blocks[has].plot(ax=ax, color=[c for c, h in zip(fill, has) if h], edgecolor="white", linewidth=lw,
                         alpha=0.85, zorder=6)
        blocks[~has].plot(ax=ax, facecolor="none", edgecolor="#c9c8c3", linewidth=lw, zorder=6)
        ax.set_xlabel(""); ax.set_ylabel("")
    a.add_collection(n.pipes_lc([GREY] * len(n.pipes), wmin=0.15, wmax=1.6, zorder=7))
    a.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor=INK, lw=1.6, zorder=12))
    n.frame(a)
    b.add_collection(n.pipes_lc([BLUE] * len(n.pipes), wmin=0.8, wmax=3.0, zorder=7))
    nd = n.nodes[(n.nodes.population > 0)]
    nd = nd[nd.geometry.x.between(x0, x1) & nd.geometry.y.between(y0, y1)]
    b.scatter(nd.geometry.x, nd.geometry.y, s=4 + 1.2 * nd.population, color=ORANGE, edgecolor="white",
              linewidths=0.5, zorder=9)
    b.set_xlim(x0, x1); b.set_ylim(y0, y1)
    b.plot([x1 - 500, x1 - 100], [y0 + 70] * 2, color=INK, lw=2, zorder=12)
    b.text(x1 - 300, y0 + 95, "400 m", ha="center", va="bottom", fontsize=12, zorder=12,
           bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1))
    for sp in b.spines.values():
        sp.set_visible(True); sp.set_color(INK); sp.set_linewidth(1.6)
    save(fig, "blocks", raster=True)


def zone_colours(n: Net, buffer=310):
    """Touching zones get different hues (exact colouring of the adjacency graph, as in the paper)."""
    import itertools
    import networkx as nx
    from shapely.ops import unary_union
    hulls = {z: unary_union(list(g.geometry)).buffer(buffer).buffer(-250)
             for z, g in n.pipes.groupby("zone") if z != "rest"}
    Z = nx.Graph(); Z.add_nodes_from(hulls)
    for a, b in itertools.combinations(hulls, 2):
        if hulls[a].intersects(hulls[b]):
            Z.add_edge(a, b)
    hues = [BLUE, ORANGE, AQUA, YELLOW]
    order = sorted(Z, key=lambda z: (-Z.degree(z), z))
    for k in (3, 4):
        col = {}

        def fill(i):
            if i == len(order):
                return True
            for c in range(k):
                if all(col.get(m) != c for m in Z[order[i]]):
                    col[order[i]] = c
                    if fill(i + 1):
                        return True
                    del col[order[i]]
            return False
        if fill(0):
            return {z: hues[c] for z, c in col.items()} | {"rest": GREY}, hulls
    raise RuntimeError("zone map needs more than four colours")


def fig_zones():
    """Slide 8: the 14 zones and their outlet sites; pass-through zones marked."""
    n = Net()
    W, _ = sens()
    summary = zones_summary()
    pt = pass_through(W, summary)
    colour, hulls = zone_colours(n)
    fig, ax = plt.subplots(figsize=(4.6, 6.4))
    n.basemap(ax)
    for z, h in hulls.items():
        for part in getattr(h, "geoms", [h]):
            x, y = part.exterior.xy
            ax.fill(x, y, color=colour[z], alpha=0.08, lw=0, zorder=3)
            ax.plot(x, y, color=colour[z], lw=0.9, alpha=0.8, zorder=4)
    ax.add_collection(n.pipes_lc([colour[z] for z in n.pipes.zone]))
    sites = [s for s in W.index if np.isfinite(W.loc[s]).all()]
    for s in sites:
        x, y = n.xy(s)
        ax.scatter([x], [y], s=55, facecolor=INK, edgecolor="white", linewidths=1.2, zorder=10)
    for z in summary.zone:
        if z == "rest":
            continue
        pts = n.nodes[n.nodes.zone == z].geometry
        ax.text(pts.x.median(), pts.y.median(), z[1:], fontsize=12, fontweight="bold", color=INK, ha="center",
                va="center", zorder=9, bbox=dict(boxstyle="round,pad=0.18", facecolor="white" if not pt[z] else "#f1ecf7",
                                                 edgecolor=colour[z], lw=1.6 if not pt[z] else 1.0,
                                                 ls="-" if not pt[z] else "--"))
    x, y = n.xy(PLANT)
    ax.scatter([x], [y], s=240, marker="*", color=INK, edgecolor="white", linewidths=1.2, zorder=11)
    n.frame(ax)
    xmin, xmax = ax.get_xlim()
    ax.set_xlim(xmin, xmax + 500)  # a little more room in the empty north-east corner for the legend
    ax.legend(handles=[Line2D([], [], ls="", marker="o", color=INK, ms=7, label="candidate site\n(zone outlet)"),
                       Line2D([], [], ls="", marker="*", color=INK, ms=12, label="treatment plant"),
                       Patch(facecolor="white", edgecolor=MUTED, label="zone with\nits own site"),
                       Patch(facecolor="#f1ecf7", edgecolor=MUTED, ls="--", label="pass-through\nzone")],
              loc="upper right", bbox_to_anchor=(1.2, 1.0), fontsize=11, frameon=True, facecolor="white",
              edgecolor="none", framealpha=0.9, borderpad=0.3, handletextpad=0.4, labelspacing=0.5,
              borderaxespad=0.2)  # in the empty north-east corner, partly past the map's edge
    save(fig, "zones", raster=True)


def zone_tree(n: Net, summary):
    """Each zone's downstream zone ('plant' for an arriving stream): from its outlet, follow the pipe that
    carries the most people until the path enters another zone."""
    out = n.pipes.sort_values("up_pop", ascending=False).drop_duplicates("from_node").set_index("from_node")
    parent = {}
    for z, c in summary.set_index("zone")["cut_node"].items():
        if z == "rest":
            continue
        node = c
        while node in out.index and n.nodes.zone.get(node, "rest") == z:
            node = out.at[node, "to_node"]
        zz = n.nodes.zone.get(node, "rest")
        parent[z] = "plant" if zz in ("rest", z) else zz
    return parent


def fig_zone_tree(min_share=0.01):
    """Slide 8, beside the zone map: which zone drains into which, with the plant at the bottom. Solid
    arrows follow the main path; dashed arrows are flow that a split (or a pipe flowing against its drawn
    direction) sends into another zone's outlet, at least min_share of that outlet's composite."""
    from matplotlib.patches import FancyArrowPatch
    n = Net()
    W, _ = sens()
    summary = zones_summary()
    pt = pass_through(W, summary)
    colour, _ = zone_colours(n)
    parent = zone_tree(n, summary)
    kids = {}
    for z, p in parent.items():
        kids.setdefault(p, []).append(z)
    order = ["Z11", "Z07", "Z12", "Z14"]  # the four arriving streams, in slide 9's order
    pos, leaf = {}, [0]

    # Zone 01 goes on the right of zone 07's branches, next to zone 12, so the dashed split arrows
    # between 07, 01, and 12 are short and pass behind no other box.
    child_order = {"Z07": ["Z06", "Z03", "Z01"]}

    def place(z, depth):
        ch = child_order.get(z, sorted(kids.get(z, [])))
        for c in ch:
            place(c, depth + 1)
        if ch:
            x = np.mean([pos[c][0] for c in ch])
        else:
            x = leaf[0]; leaf[0] += 1
        pos[z] = (x, depth)

    for z in order:
        place(z, 1)
    pos["plant"] = (np.mean([pos[z][0] for z in order]), 0)
    fig, ax = plt.subplots(figsize=(5.8, 3.4))
    ax.axis("off")
    for z, p in parent.items():
        ax.add_patch(FancyArrowPatch(pos[z], pos[p], arrowstyle="-|>", mutation_scale=11, color=MUTED, lw=1.3,
                                     shrinkA=13, shrinkB=13 if p != "plant" else 11, zorder=2))
    cut = summary.set_index("zone")["cut_node"]
    for z in parent:  # partial flows into another zone's outlet
        for y in parent:
            if y != z and parent.get(y) != z and W.loc[cut[z], y] >= min_share and z not in _ancestors(y, parent):
                ax.add_patch(FancyArrowPatch(pos[y], pos[z], arrowstyle="-|>", mutation_scale=9, color=GREY,
                                             lw=1.0, ls=(0, (3, 2)), shrinkA=13, shrinkB=13,
                                             connectionstyle="arc3,rad=-0.3", zorder=1))
    for z, (x, y) in pos.items():
        if z == "plant":
            ax.scatter([x], [y], s=260, marker="*", color=INK, edgecolor="white", zorder=4)
            ax.text(x + 0.35, y, "plant", va="center", fontsize=12, color=INK)
            continue
        ax.text(x, y, z[1:], fontsize=12, fontweight="bold", color=INK, ha="center", va="center", zorder=4,
                bbox=dict(boxstyle="round,pad=0.25", facecolor="#f1ecf7" if pt[z] else "white",
                          edgecolor=colour[z], lw=1.0 if pt[z] else 1.6, ls="--" if pt[z] else "-"))
    xs = [p[0] for p in pos.values()]; ys = [p[1] for p in pos.values()]
    ax.set_xlim(min(xs) - 0.6, max(xs) + 1.6); ax.set_ylim(-0.5, max(ys) + 0.5)  # room on the right for the legend
    ax.legend(handles=[Line2D([], [], color=MUTED, lw=1.3, label="drains into\n(largest flow at each split)"),
                       Line2D([], [], color=GREY, lw=1.0, ls=(0, (3, 2)),
                              label="smaller branch of a split,\nor a reversed pipe")],
              loc="upper right", fontsize=11, handlelength=1.6, borderaxespad=0)
    save(fig, "zone_tree")


def _ancestors(z, parent):
    out = []
    while z in parent:
        z = parent[z]; out.append(z)
    return out


# ------------------------------------------------------------------------------------ hydraulics
def fig_hydraulics():
    """Slide 7: modelled inflow at the plant against the plant's dry-day range."""
    q = pd.read_csv(PAPER / "hydraulics" / "greenway" / "outlet_flow_day3.csv")
    s = PARAMS["loading"]["diurnal_pattern"]
    p = pd.read_csv(PAPER / "data" / "plant" / "greenway_daily_mps.csv")
    dd = p[p["dry"].astype(bool)]
    fig, ax = plt.subplots(figsize=(W_COL, H_COL))
    grid(ax)
    ax.step(np.arange(25), np.r_[s, s[0]], where="post", color=GREY, lw=1.5)
    for col, lab, above in (("max_over_total", "plant daily maximum", True), ("min_over_total", "plant daily minimum", False)):
        lo, med, hi = dd[col].quantile([0.25, 0.5, 0.75])
        ax.axhspan(lo, hi, color=ORANGE, alpha=0.15, lw=0)
        ax.axhline(med, color=ORANGE, lw=2, ls=(0, (5, 3)))
        ax.text(0.3 if above else 23.7, hi + 0.03 if above else lo - 0.03, lab, color=MUTED, fontsize=13,
                ha="left" if above else "right", va="bottom" if above else "top")
    ax.plot(q["hour"], q["flow_L_s"] / q["flow_L_s"].mean(), color=BLUE, lw=2.8)
    ax.axhline(1, color=MUTED, lw=0.6)
    ax.set_xlim(0, 24); ax.set_xticks(range(0, 25, 3)); ax.set_ylim(0, 2.45)
    ax.set_xlabel("Hour of day"); ax.set_ylabel("Flow / daily mean")
    ax.legend(handles=[Line2D([], [], color=BLUE, lw=2.8, label="Model, inflow at the plant"),
                       Line2D([], [], color=ORANGE, lw=2, ls=(0, (5, 3)), label="Plant, dry days 2022–24 (median, IQR)"),
                       Line2D([], [], color=GREY, lw=1.5, label="Household water-use pattern (input)")],
              loc="upper left", fontsize=12, ncol=1)
    save(fig, "hydraulics")


def fig_representation():
    """Slide 9: each zone's share of the composite at the plant and at every stream that reaches it,
    against its share of the people. Zones are grouped by the stream that carries most of them."""
    W, _ = sens()
    share = zones_summary().set_index("zone")["share"]
    W = W.drop(columns="rest")
    streams = list(STREAMS)
    main = W.loc[streams].idxmax()  # the stream that carries most of each zone
    z = [zz for s_ in streams for zz in W.columns if main[zz] == s_]
    x = np.arange(len(z))
    fig, ax = plt.subplots(figsize=(W_COL, H_COL + 0.3))
    grid(ax)
    ax.bar(x, share[z] * 100, width=0.72, color="#d9d8d3", zorder=2, label="Share of the people")
    ax.scatter(x, W.loc[PLANT, z] * 100, s=110, color=ORANGE, zorder=5, label="Plant composite",
               edgecolor="white", linewidths=1)
    offsets = {"WT2203": 0.0, "WT1420": -0.14, "OS2272": 0.14, "WT6005": 0.0}  # WT1420 and OS2272 share Z01, Z07
    for s_, (m, c, lab) in STREAMS.items():
        v = W.loc[s_, z] * 100
        ax.scatter(x + offsets[s_], np.where(v > 0.05, v, np.nan), s=70, marker=m, color=c, zorder=4,
                   label=lab, edgecolor="white", linewidths=0.8)
    # bracket each group of zones under the stream that carries it
    for s_ in streams:
        idx = [i for i, zz in enumerate(z) if main[zz] == s_]
        if idx[0]:
            ax.axvline(idx[0] - 0.5, color=GRID, lw=1.2, zorder=1)
    ax.set_xticks(x, [zz[1:] for zz in z])
    ax.set_xlim(-0.6, len(z) - 0.4)
    ax.set_xlabel("Zone, grouped by the stream that carries it")
    ax.set_ylabel("Share (%)")
    h, l = ax.get_legend_handles_labels()  # column by column: plant and people, then the streams
    order = [l.index("Plant composite"), l.index("Share of the people")] + [l.index(STREAMS[s_][2]) for s_ in streams]
    ax.legend([h[i] for i in order], [l[i] for i in order], loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3,
              fontsize=12, columnspacing=1.0, handletextpad=0.3)
    save(fig, "representation")


# ---------------------------------------------------------------------------------------- noise
def fig_noise():
    """Slide 10: noise components of a composite at the plant and the hospital sewershed (N1/PMMoV,
    after the April 2023 extraction switch)."""
    def load(f):
        n = pd.read_csv(PAPER / "scenarios" / "odm" / f)
        return n[(n.method == "robust") & (n.subset == "all")]
    post = load("noise_components_postswitch.csv")
    comps = [("sd_qpcr", "qPCR replicate"), ("sd_biorep", "Biological replicate"), ("sd_day", "Day to day"),
             ("sigma_mean_two_reps", "One composite, in total")]
    fig, axes = plt.subplots(1, 2, figsize=(W_COL + 0.6, H_COL), sharey=True)
    fig.subplots_adjust(left=0.36, right=0.97, top=0.9, bottom=0.29, wspace=0.12)
    for ax, (target, title) in zip(axes, (("covN1", "SARS-CoV-2 N1"), ("covN1/PMMoV", "N1 / PMMoV"))):
        grid(ax, "x")
        y = np.arange(len(comps))[::-1]
        for off, (site, lab, col) in zip((0.19, -0.19), (("GRWY", "Plant", ORANGE), ("VICT", "Hospital sewershed (upstream)", BLUE))):
            r = post[(post.site == site) & (post.target == target)].iloc[0]
            v = [r[c] for c, _ in comps]
            ax.barh(y + off, v, height=0.36, color=col, label=lab, zorder=3)
            for yy, vv in zip(y + off, v):
                ax.text(vv + 0.02, yy, f"{vv:.2f}", va="center", fontsize=12, color=MUTED)
        ax.set_yticks(y, [lab for _, lab in comps])
        ax.set_title(title, loc="left", color=INK)
        if target == "covN1":
            ax.text(0.97, 0.97, "used in\nthe design", transform=ax.transAxes, ha="right", va="top",
                    fontsize=11, color=PURPLE, fontweight="bold")
            ax.patch.set_facecolor("#f1ecf8")
        ax.set_xlim(0, 1.55)
        ax.set_xticks([0, 0.5, 1])
    fig.supxlabel("SD of ln concentration", x=0.665, y=0.11, fontsize=plt.rcParams["axes.labelsize"])
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.665, 0.07), bbox_transform=fig.transFigure, ncol=2,
                   fontsize=12, frameon=False)
    save(fig, "noise")


def fig_noise_tails():
    """Appendix: measured residuals against the fitted Student-t and a normal (share beyond |r|)."""
    R = PAPER / "scenarios" / "odm"
    res, ref, fits = (pd.read_csv(R / f) for f in ("noise_tails_residuals.csv", "noise_tails_reference.csv", "noise_tails.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(W_FULL, H_FULL - 0.2), sharey=True)
    for ax, (site, title) in zip(axes, (("GRWY", "Plant"), ("VICT", "Hospital sewershed"))):
        grid(ax)
        r = res[res.site == site].residual.to_numpy()
        h = fits[(fits.site == site) & (fits.statistic == "loo")].iloc[0]
        e = ref[(ref.site == site) & (ref.kind == "exceedance")]
        a = np.sort(np.abs(r))
        ax.step(np.r_[0, a], 1 - np.arange(len(a) + 1) / len(a), where="post", color=INK, lw=1.6, label="Measured")
        ax.plot(e.x, e.t, color=BLUE, label=f"Student-t, ν = {h.nu:.1f}")
        ax.plot(e.x, e.normal, color=ORANGE, ls=(0, (5, 3)), label="Normal, same typical spread")
        ax.set_yscale("log"); ax.set_ylim(5e-3, 1.1); ax.set_xlim(0, 4)
        ax.set_xlabel("Size of residual |r| (ln units)")
        ax.set_title(f"{title}, {len(r)} days", loc="left", color=INK)
        ax.legend(loc="lower left", fontsize=12)
    axes[0].set_ylabel("Share of days beyond |r|")
    fig.tight_layout(w_pad=2)
    save(fig, "noise_tails")


# --------------------------------------------------------------------------------------- design
def fig_identical_shift():
    """Slide 11: what each candidate site would read (rise of the log composite) if everyone's shedding
    doubled, against a 5-fold hot spot in one zone. Computed from the representation fractions."""
    W, _ = sens()
    sites = [s for s in W.index if np.isfinite(W.loc[s]).all()]
    Wz = W.loc[sites].drop(columns="rest")
    kappa = float(PARAMS["design"]["hot_spot_factor"])
    hot = "Z09"  # the zone of the paper's example campaign
    rise_all = np.full(len(sites), np.log(2.0))
    rise_hot = np.log(1 + (kappa - 1) * Wz[hot].to_numpy())
    x = np.arange(len(sites))
    fig, axes = plt.subplots(1, 2, figsize=(W_COL, H_COL), sharey=True)
    for ax, v, title, col in ((axes[0], rise_all, "Everyone sheds 2× as much", GREY),
                              (axes[1], rise_hot, f"Zone {hot[1:]} sheds {kappa:g}× as much", ORANGE)):
        grid(ax)
        ax.bar(x, v, width=0.72, color=col, zorder=3)
        ax.set_xticks(x, ["plant" if s == PLANT else s for s in sites], rotation=90, fontsize=11)
        ax.set_title(title, loc="left", color=INK)
    # mark the hot zone's own outlet, where the hot spot shows most
    outlet = zones_summary().set_index("zone").at[hot, "cut_node"]
    i = sites.index(outlet)
    b = axes[1]
    b.bar([i], [rise_hot[i]], width=0.72, color="#c2501c", zorder=4)
    b.annotate(f"zone {hot[1:]}'s outlet", (i - 0.4, rise_hot[i] * 0.8), xytext=(i - 4.5, rise_hot[i] * 0.8),
               ha="center", va="center", fontsize=12, color=INK,
               arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.0, mutation_scale=10, shrinkA=4, shrinkB=1))
    lab = b.get_xticklabels()[i]
    lab.set_fontweight("bold"); lab.set_color("#c2501c")
    axes[0].set_ylabel("Rise of ln composite")
    fig.tight_layout(w_pad=2)
    save(fig, "identical_shift")


def fig_level_contrasts(lam=3.0):
    """Slide 11, under the equation: each zone's log shedding, ln(lambda) + ln(f_z), drawn as a shared
    level m plus zero-sum differences. Illustrative: zone 09 sheds kappa times as much, and lambda is an
    arbitrary rate (its value only moves the level)."""
    kappa = float(PARAMS["design"]["hot_spot_factor"])
    zones = [z for z in zones_summary().zone if z != "rest"]
    hot = "Z09"
    y = np.log(lam) + np.log(np.where(np.array(zones) == hot, kappa, 1.0))
    m = y.mean()
    x = np.arange(len(zones))
    fig, ax = plt.subplots(figsize=(4.6, 2.75))
    grid(ax)
    ax.bar(x, np.minimum(y, m), width=0.72, color="#d9d8d3", zorder=2)  # the shared level
    dev = y - m
    ax.bar(x, dev, bottom=m, width=0.72, color=ORANGE, zorder=3)  # all of Hβ, up and down, in its colour
    ax.axhline(m, color=INK, lw=1.4, ls=(0, (5, 3)), zorder=4)
    ax.text(len(zones) - 0.4, m + 0.06, "level $m$", ha="right", va="bottom", fontsize=15, color=INK)
    i = zones.index(hot)
    ax.annotate("differences $\\mathbf{H}\\boldsymbol{\\beta}$\n(sum to zero)", (i + 0.4, m + dev[i] * 0.6),
                xytext=(i + 1.4, m + dev[i] * 0.6), ha="left", va="center", fontsize=15, color=INK, linespacing=1.1,
                arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.0, mutation_scale=10))
    ax.set_xticks(x, [z[1:] for z in zones], fontsize=12, rotation=90)
    ax.set_xlabel("Zone", fontsize=14, labelpad=2)
    ax.set_ylabel("ln shedding\nper person", fontsize=13, linespacing=1.0)
    ax.set_yticks([])
    ax.set_ylim(0, (m + dev.max()) * 1.12)
    save(fig, "level_contrasts")


def _greedy_path(ax, d, y, scale=1.0):
    d = pd.concat([pd.DataFrame({"site_days": [1], "distinct_sites": [1], y: [0.0]}),
                   d[["site_days", "distinct_sites", y]]], ignore_index=True)
    new = d["distinct_sites"].diff().fillna(1) > 0
    ax.plot(d.site_days, d[y] * scale, color=BLUE, lw=1.8, zorder=3)
    ax.scatter(d.site_days[new], d[y][new] * scale, s=40, color=BLUE, zorder=4)
    ax.scatter(d.site_days[~new], d[y][~new] * scale, s=40, facecolor="white", edgecolor=BLUE, linewidths=1.4, zorder=4)


def _design_data():
    plans = pd.read_csv(G / "plans.csv")
    plant = plans[plans.plan.str.match(r"^(WWTP|inlets)") & ~plans.plan.str.contains("-prop.", regex=False)]
    return plans, plant


def fig_design_rank():
    """Slide 12: contrasts identified against site-days, with the plant and arriving-sewer plans labelled."""
    plans, plant = _design_data()
    greedy = pd.read_csv(G / "greedy.csv")
    nc = int(plans["rank"].max())
    fig, ax = plt.subplots(figsize=(W_COL + 0.4, H_COL + 0.3))
    grid(ax)
    _greedy_path(ax, greedy, "rank")
    ax.scatter(plant.site_days, plant["rank"], s=110, marker="D", color=ORANGE, zorder=5, edgecolor="white")
    names = {"WWTP, 1 day": None, "WWTP, 7 days": "plant, 1 or 7 days",
             "inlets, 1 day each": "3 sewers, 1 day each", "inlets, 7 days each": "3 sewers, 7 days each",
             "WWTP + inlets, 7 days": "plant + 3 sewers,\n7 days each"}
    offs = {"WWTP, 1 day": (8, 8), "WWTP, 7 days": (10, 0), "inlets, 1 day each": (10, -3),
            "inlets, 7 days each": (0, -16), "WWTP + inlets, 7 days": (0, 10)}
    for _, r in plant.iterrows():
        if names[r.plan] is None:
            continue
        dx, dy = offs[r.plan]
        ax.annotate(names[r.plan], (r.site_days, r["rank"]), xytext=(dx, dy), textcoords="offset points",
                    ha="left" if dx > 0 else ("right" if dx < 0 else "center"),
                    va="bottom" if dy > 0 else ("top" if dy < 0 else "center"),
                    fontsize=11.5, color="#c2501c", linespacing=1.0)
    ax.axhline(nc, color=MUTED, lw=0.8, ls=":")
    ax.text(29.5, nc + 0.3, f"all {nc}", fontsize=12, color=MUTED, ha="right")
    ax.set_xlabel("Site-days"); ax.set_ylabel("Contrasts identified (rank of S)")
    ax.set_yticks(range(0, nc + 1, 2))
    ax.set_ylim(-0.7, nc + 1.5); ax.set_xlim(0, plant.site_days.max() + 2)
    ax.legend(handles=[Line2D([], [], color=BLUE, marker="o", ms=7, label="Precision plan: new site"),
                       Line2D([], [], color=BLUE, marker="o", ms=7, mfc="white", label="Precision plan: repeat day"),
                       Line2D([], [], ls="", marker="D", ms=9, color=ORANGE, label="Plant and its arriving sewers")],
              loc="center right", bbox_to_anchor=(1.0, 0.62), fontsize=12)
    save(fig, "design_rank")


PLANT_NOISE_SITES = {PLANT} | ARRIVING  # sites with the plant's noise (the paper's design noise by site)


def pair_separation(plan: str) -> pd.DataFrame:
    """delta_ij for every pair of zones under a plan 'SITExN; ...' (the paper's Eq. separation): the
    difference between a hot spot in zone i and one in zone j, site by site, with its weighted mean (the
    unknown level) removed, in units of the noise."""
    W, _ = sens()
    W = W.drop(columns="rest")
    kappa = float(PARAMS["design"]["hot_spot_factor"])
    sp, su = PARAMS["noise"]["sigma_ln_plant"], PARAMS["noise"]["sigma_ln_upstream"]
    n = {s: int(c) for s, c in re.findall(r"(\w+)x(\d+)", plan)}
    w = np.array([n[s] / (sp if s in PLANT_NOISE_SITES else su) ** 2 for s in n])
    Y = np.log(1 + (kappa - 1) * W.loc[list(n)].to_numpy())
    dd = Y[:, :, None] - Y[:, None, :]
    d2 = (w[:, None, None] * dd ** 2).sum(0) - (w[:, None, None] * dd).sum(0) ** 2 / w.sum()
    return pd.DataFrame(np.sqrt(np.clip(d2, 0, None)), index=W.columns, columns=W.columns)


def fig_greedy_steps(stages=(4, 10, 38)):
    """New methods slide before slide 13: the greedy plan for hot-spot discrimination at a few stages,
    each pair of zones filled once it is told apart (delta >= 2). Recomputed from the representation
    fractions; matches the paper's pairs_separable at every step."""
    h = pd.read_csv(G / "greedy_hotspot.csv").set_index("site_days")
    sep = float(CONST["separation"])
    fig, axes = plt.subplots(1, len(stages), figsize=(W_FULL, 3.9))
    for ax, sd in zip(axes, stages):
        r = h.loc[sd]
        D = pair_separation(r.plan)
        assert abs((D.to_numpy()[np.triu_indices(len(D), 1)] >= sep).mean() - r.pairs_separable) < 1e-3
        nz = len(D)
        for i in range(nz):
            for j in range(i):
                ok = D.iat[i, j] >= sep
                ax.add_patch(matplotlib.patches.Rectangle((j, nz - 1 - i), 0.9, 0.9, color=PURPLE if ok else "#e4e3df",
                                                          lw=0))
        ax.set_xlim(-0.2, nz - 1); ax.set_ylim(-0.2, nz - 1)
        ax.set_aspect("equal"); ax.axis("off")
        for k, z in enumerate(D.index):
            if k:
                ax.text(-0.35, nz - 1 - k + 0.45, z[1:], ha="right", va="center", fontsize=9.5, color=MUTED)
            if k < nz - 1:
                ax.text(k + 0.45, nz - 1 - k + 0.2, z[1:], ha="center", va="bottom", fontsize=9.5, color=MUTED,
                        rotation=90)
        n_sites_ = int(r.distinct_sites)
        ax.text(nz - 1, nz - 1, f"{sd} site-days\nat {n_sites_} sites", ha="right", va="top", fontsize=14,
                color=INK, fontweight="bold", linespacing=1.1)
        ax.text(nz - 1, nz - 3.3, f"{r.pairs_separable:.0%} of pairs\ntold apart", ha="right", va="top", fontsize=13,
                color=PURPLE, linespacing=1.1)
    fig.legend(handles=[Patch(color=PURPLE, label=f"pair told apart (δ ≥ {sep:g})"),
                        Patch(color="#e4e3df", label="not yet")],
               loc="lower center", ncol=2, fontsize=12.5, bbox_to_anchor=(0.5, -0.02), handlelength=1.0)
    fig.tight_layout(w_pad=1.0, rect=(0, 0.06, 1, 1))
    save(fig, "greedy_steps")


def fig_greedy_frames():
    """Slide 14: every step of the greedy plan for telling pairs apart, as data for the animated pair grid
    and plan (figures/greedy_frames.js). Pairs told apart are recomputed from the representation fractions
    and checked against the paper's pairs_separable at every step."""
    h = pd.read_csv(G / "greedy_hotspot.csv").sort_values("site_days")
    sep = float(CONST["separation"])
    outlet = zones_summary().set_index("cut_node").zone
    col = lambda site: "P" if site == PLANT else outlet[site][1:]
    zones = None
    steps = [{"site_days": 1, "added": "P", "sites": 1, "pct": 0, "days": {"P": 1}, "told": ""}]
    for _, r in h.iterrows():
        D = pair_separation(r.plan)
        zones = zones or [z[1:] for z in D.index]
        low = np.tril_indices(len(D), -1)
        ok = D.to_numpy()[low] >= sep
        assert abs(ok.mean() - r.pairs_separable) < 1e-3
        days = {col(s): int(c) for s, c in re.findall(r"(\w+)x(\d+)", r.plan)}
        steps.append({"site_days": int(r.site_days), "added": col(r.added), "sites": int(r.distinct_sites),
                      "pct": round(100 * float(r.pairs_separable)), "days": days,
                      "told": "".join("1" if x else "0" for x in ok)})
    steps[0]["told"] = "0" * len(steps[1]["told"])
    path = OUT / "greedy_frames.js"
    path.write_text("// Written by scripts/talk_figures.py (fig_greedy_frames); do not edit.\n"
                    "// told: one character per pair (i > j), row by row, 1 = told apart (delta >= 2).\n"
                    f"window.GREEDY_FRAMES = {json.dumps({'zones': zones, 'steps': steps}, separators=(',', ':'))};\n")
    print(path.relative_to(ROOT))


def fig_design_pairs():
    """Slide 13: share of zone pairs separated against site-days."""
    plans, plant = _design_data()
    hot = pd.read_csv(G / "greedy_hotspot.csv")
    ns = n_sites()
    one = plans[plans.plan.eq(f"all {ns} sites, 1 day")]
    fig, ax = plt.subplots(figsize=(W_COL, H_COL))
    grid(ax)
    _greedy_path(ax, hot, "pairs_separable", 100)
    ax.scatter(plant.site_days, plant.pairs_separable * 100, s=110, marker="D", color=ORANGE, zorder=5, edgecolor="white")
    ax.scatter(one.site_days, one.pairs_separable * 100, s=110, marker="s", color=AQUA, zorder=5, edgecolor="white")
    ax.set_xlabel("Site-days"); ax.set_ylabel("Zone pairs told apart (%)")
    ax.set_ylim(-3, 105); ax.set_xlim(0, max(hot.site_days.max(), 25) + 1)
    ax.legend(handles=[Line2D([], [], color=BLUE, marker="o", ms=7, label="Separation plan: new site"),
                       Line2D([], [], color=BLUE, marker="o", ms=7, mfc="white", label="Separation plan: repeat day"),
                       Line2D([], [], ls="", marker="D", ms=9, color=ORANGE, label="Plant and its arriving sewers only"),
                       Line2D([], [], ls="", marker="s", ms=9, color=AQUA, label=f"All {ns} sites, 1 day each")],
              loc="lower right", fontsize=12)
    save(fig, "design_pairs")


# ---------------------------------------------------------------------------------- localisation
def _loc_axes():
    fig, ax = plt.subplots(figsize=(W_FULL, H_FULL))
    grid(ax)
    nz = len(pd.read_csv(G / "zone_sd.csv", index_col="zone"))
    ax.axhline(100 / nz, color=MUTED, lw=0.8, ls=":")
    ax.text(0.9, 100 / nz + 1.5, "chance", fontsize=13, color=MUTED, ha="left")
    ax.set_xscale("log")
    ax.set_xticks([1, 3, 10, 30, 100, 300], ["1", "3", "10", "30", "100", "300"])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlim(0.8, 300); ax.set_ylim(0, 102)
    ax.set_xlabel("Site-days"); ax.set_ylabel("Hot zone named correctly (%)")
    return fig, ax


def _fixed(ax, faded=False):
    """Fixed plans, in two analyses: the simplest case (normal errors, no background, kappa known; the naming
    model of slide 15 with u_z and the heavy tails removed), scored by best guess; and the realistic case
    (heavy tails + background, slide 15's model), scored by best guess and by naming at 95 %. Returns the
    handles of the two legend groups."""
    fx = pd.read_csv(G / "localization_fixed.csv")
    fr = pd.read_csv(G / "localization_fixed_robust.csv")
    fr = fr[fr.analysis == "robust"]
    a = 0.35 if faded else 1.0
    g = fx[fx.plan.str.startswith("greedy hot-spot")]
    al = fx[fx.plan.str.startswith("all ")]
    pl = fx[fx.plan.str.startswith(("plant", "inlets"))]
    simple, real = [], []
    ax.plot(g.site_days, 100 * g.correct_mean, color=BLUE, marker="o", ms=5, alpha=a)
    simple.append(Line2D([], [], color=BLUE, marker="o", ms=6, label="Separation plan, as it grows"))
    ax.plot(al.site_days, 100 * al.correct_mean, color=AQUA, marker="s", ms=6, alpha=a)
    simple.append(Line2D([], [], color=AQUA, marker="s", ms=7, label="All sites, 1–14 days"))
    ax.scatter(pl.site_days, 100 * pl.correct_mean, s=100, marker="D", color=ORANGE, zorder=5, alpha=a)
    simple.append(Line2D([], [], ls="", marker="D", ms=9, color=ORANGE, label="Plant or arriving sewers, 7 days"))
    ax.scatter(fr.site_days, 100 * fr.correct, s=120, marker="o", color=PURPLE, zorder=7, alpha=a, linewidths=0)
    real.append(Line2D([], [], ls="", marker="o", ms=10, color=PURPLE, label="Best guess right"))
    ax.scatter(fr.site_days, 100 * fr.correct_confident, s=120, marker="o", facecolor="white", edgecolor=PURPLE,
               linewidths=2.2, zorder=7, alpha=a)
    real.append(Line2D([], [], ls="", marker="o", ms=10, mfc="white", mec=PURPLE, mew=2.2,
                       label="Named at 95 %"))
    return simple, real


def _fixed_legends(ax, simple, real, fontsize=12):
    """One legend, two columns: the simplest case (scored by best guess) and the realistic case."""
    head = lambda t: Line2D([], [], ls="", marker="", label=t)
    h = [head("Simplest case, best guess")] + simple + [head("Realistic: heavy tails + background")] + real
    leg = ax.legend(handles=h, loc="upper left", fontsize=fontsize, handletextpad=0.6, labelspacing=0.35)
    for t, hd in zip(leg.get_texts(), h):
        if not hd.get_marker():
            t.set_fontweight("bold"); t.set_position((-34, 0))
    return leg


def fig_localisation_fixed():
    """Slide 16: fixed plans only, with the two drops at the best plan's 38 site-days marked."""
    fig, ax = _loc_axes()
    simple, real = _fixed(ax)
    _fixed_legends(ax, simple, real)
    fx = pd.read_csv(G / "localization_fixed.csv")
    fr = pd.read_csv(G / "localization_fixed_robust.csv")
    g = fx[fx.plan.str.startswith("greedy hot-spot")].sort_values("site_days").iloc[-1]
    r = fr[(fr.analysis == "robust") & fr.plan.str.startswith("greedy hot-spot")].iloc[0]
    assert int(r.site_days) == int(g.site_days)
    x, ys = float(g.site_days) * 0.86, (100 * g.correct_mean, 100 * r.correct, 100 * r.correct_confident)
    for (y0, y1), lab, ly in zip(((ys[0], ys[1]), (ys[1], ys[2])),
                                 ("1. realistic noise:\n" f"{ys[0]:.0f} → {ys[1]:.0f} %",
                                  "2. named at 95 %,\nnot best guess:\n" f"{ys[1]:.0f} → {ys[2]:.0f} %"),
                                 (80, 27)):
        ax.annotate("", (x, y1 + 1.2), (x, y0 - 1.2), arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.4,
                                                                        mutation_scale=12))
        right = ly < ys[1]  # the second label sits right of the arrow, clear of the plant markers
        ax.text(x * (1.15 if right else 0.94), ly, lab, ha="left" if right else "right", va="center", fontsize=12,
                color=INK)
    save(fig, "localisation_fixed")


def fig_localisation_adaptive():
    """Slide 17: fixed plans faded, adaptive campaigns on top."""
    fig, ax = _loc_axes()
    simple, real = _fixed(ax, faded=True)
    for l, t in zip(simple + real, ("Separation plan, simplest case", "All sites, 1–14 days, simplest case",
                                    "Plant or arriving sewers, 7 days", "Fixed plans, realistic: best guess right",
                                    "Fixed plans, realistic: named at 95 %")):
        l.set_label(t)
    h = simple + real
    ad = pd.read_csv(G / "localization_adaptive.csv").sort_values("K")
    lg = pd.read_csv(G / "localization_adaptive_lag.csv")
    held = lg[(lg["mode"] == "held") & (lg.case == "realistic")].sort_values("K")
    ax.plot(ad.site_days_median, 100 * ad.correct, color=INK, marker="*", ms=12, lw=2)
    ax.plot(held.site_days_median, 100 * held.correct, color=PURPLE, marker="*", ms=13, mfc="white", lw=2.4)
    r = held[held.K == 2].iloc[0]
    ax.annotate(f"plant + 2 a day:\n{100 * r.correct:.0f} % in {r.site_days_median:.0f} site-days",
                (r.site_days_median, 100 * r.correct), xytext=(-120, 42), textcoords="offset points", fontsize=13, ha="center",
                color=PURPLE, arrowprops=dict(arrowstyle="-", color=PURPLE, lw=1))
    h += [Line2D([], [], color=INK, marker="*", ms=12, label="Adaptive, results the same day"),
          Line2D([], [], color=PURPLE, marker="*", ms=13, mfc="white", lw=2.4, label="Adaptive, held samples, 2-day lab")]
    ax.legend(handles=h, loc="upper left", fontsize=12)
    save(fig, "localisation_adaptive")


def fig_kappa():
    """Slide 18 (left half): hot zone named at 95 % against the hot-spot factor, held campaign and two fixed
    plans."""
    k = pd.read_csv(G / "localization_kappa.csv")
    ns = n_sites()
    greedy = sorted({p for p in k.plan if re.fullmatch(r"greedy hot-spot, \d+ site-days", p)})[0]
    plans = (("adaptive, held, K=2", PURPLE, "*", 13, "Adaptive, held samples, 2 a day"),
             (f"all {ns} sites, 7 days", AQUA, "s", 8, f"All {ns} sites, 7 days each"),
             (greedy, BLUE, "o", 7, "Separation plan, " + greedy.split(", ")[1]))
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    grid(ax)
    for plan, col, mk, ms, lab in plans:
        for tau, ls in ((CONST["tau_analysis"], "-"), (CONST["tau_cautious"], "--")):
            x = k[(k.plan == plan) & np.isclose(k.tau, tau)].sort_values("kappa")
            ax.plot(x.kappa, 100 * x.correct, color=col, ls=ls, marker=mk if ls == "-" else None, ms=ms,
                    lw=2.4 if ls == "-" else 1.4, label=lab if ls == "-" else None,
                    mfc="white" if mk == "*" else col)
    ax.plot([], [], color=MUTED, ls="--", lw=1.4, label=f"Zones differ more (τ = {CONST['tau_cautious']:g})")
    ax.set_xscale("log")
    ax.set_xticks([2, 3, 5, 7, 10], ["2×", "3×", "5×", "7×", "10×"])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("Hot-spot strength (fold)")
    ax.set_ylabel("Named correctly at 95 % (%)")
    ax.set_ylim(0, 102)
    ax.legend(loc="upper left", fontsize=11, handlelength=1.8)
    save(fig, "kappa")


def fig_mismatch():
    """Slide 18 (right half): share of names that are right against the true background, by hot-spot strength."""
    d = pd.read_csv(G / "localization_mismatch.csv")
    kappas = sorted(d.kappa.unique())
    cols = dict(zip(kappas, ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]))
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    grid(ax)
    for kk in kappas:
        q = d[(d.kappa == kk) & (d.bias == 0)].sort_values("tau")
        ax.plot(q.tau, 100 * q.correct_given_localized, color=cols[kk], marker="o", ms=7, lw=2.4)
        ax.annotate(f"{kk:g}-fold", (q.tau.iloc[-1], 100 * q.correct_given_localized.iloc[-1]), xytext=(8, 0),
                    textcoords="offset points", va="center", fontsize=13, color=cols[kk])
    ta = CONST["tau_analysis"]
    ax.axvline(ta, color=MUTED, lw=1, ls=":")
    ax.text(ta, 4, f" analysis\n assumes τ = {ta:g}", fontsize=12, color=MUTED, ha="left")
    ax.set_xlabel("True spread between zones, τ (ln units)")
    ax.set_ylabel("Names that are right (%)")
    ax.set_ylim(0, 102); ax.set_xticks(sorted(d.tau.unique()))
    ax.set_xlim(None, d.tau.max() + 0.14)
    save(fig, "mismatch")


def fig_example_campaign():
    """Slide 16: one held campaign, each zone's probability by day, and the composites analyzed."""
    probs = pd.read_csv(G / "example_campaign_probs.csv")
    an = pd.read_csv(G / "example_campaign_analyses.csv")
    share_col = next(c for c in an.columns if c.startswith("share_from_"))
    hot = share_col.removeprefix("share_from_")
    zones = [c for c in probs.columns if c != "morning"]
    days = probs.morning.to_numpy()
    P = probs[zones].to_numpy()
    zi = zones.index(hot)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(W_FULL, H_FULL + 0.2), gridspec_kw=dict(width_ratios=[1.15, 1]))
    for z in range(len(zones)):
        if z != zi:
            a1.plot(days, P[:, z], color=GREY, lw=1, alpha=0.6, drawstyle="steps-post")
    a1.plot(days, P[:, zi], color=ORANGE, lw=3, drawstyle="steps-post")
    # label the hot zone's line where it first leads for good, above the line and clear of the others
    lead = int(np.argmax([(P[k:, zi] >= P[k:].max(axis=1)).all() for k in range(len(days))]))
    a1.annotate(f"zone {hot[1:]} (hot)", (days[lead], P[lead, zi]), xytext=(-6, 34), textcoords="offset points",
                ha="right", fontsize=13, color=ORANGE, fontweight="bold",
                arrowprops=dict(arrowstyle="-", color=ORANGE, lw=1))
    a1.axhline(0.95, color=INK, lw=1, ls=(0, (5, 3)))
    a1.text(days[0], 0.97, "95 %: name the zone", fontsize=13, color=MUTED)
    a1.axhline(1 / len(zones), color=GREY, lw=0.8, ls=":")
    a1.set_ylim(0, 1.05); a1.set_xlabel("Day of campaign")
    a1.set_ylabel("Probability the zone is hot")
    grid(a1)
    first = an.groupby("site").lab_day.min()
    used = [s for s in first.sort_values(kind="stable").index if s != PLANT]
    used = [PLANT] + sorted(used, key=lambda s: (an.loc[an.site == s, share_col].iloc[0] > 0.01, first[s]))
    row = {s: i for i, s in enumerate(used)}
    for t, b in an.groupby("lab_day", sort=True):
        sites = list(b.site)
        for j, s in enumerate(sites):
            dx = 0.2 * (sites.count(s) > 1) * (1 if sites.index(s) != j else -1)
            sh = b[share_col].iloc[j]
            a2.scatter(t + dx, row[s], s=40, color=GREY if s == PLANT else (ORANGE if sh > 0.01 else BLUE), zorder=3,
                       linewidths=0)
    a2.set_yticks(range(len(used)), ["plant" if s == PLANT else f"{s} (arriving)" if s in ARRIVING else s for s in used],
                  fontsize=11)
    a2.invert_yaxis()
    a2.set_xlabel("Day the lab starts the composite")
    grid(a2, "x")
    a2.legend(handles=[Line2D([], [], ls="", marker="o", color=ORANGE, ms=8, label=f"receives zone {hot[1:]}"),
                       Line2D([], [], ls="", marker="o", color=BLUE, ms=8, label=f"none from zone {hot[1:]}")],
              loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2, fontsize=12)
    fig.tight_layout(w_pad=2.5)
    save(fig, "example_campaign")


def fig_posterior_bars():
    """Zone probabilities on two mornings of the example campaign (real output). Static version; slide 15
    now uses the animated bars (fig_posterior_frames)."""
    probs = pd.read_csv(G / "example_campaign_probs.csv")
    an = pd.read_csv(G / "example_campaign_analyses.csv")
    hot = next(c for c in an.columns if c.startswith("share_from_")).removeprefix("share_from_")
    zones = [c for c in probs.columns if c != "morning"]
    mid = int(probs.morning.iloc[len(probs) // 2])
    last = int(probs.morning.iloc[-1])
    fig, axes = plt.subplots(2, 1, figsize=(5.0, 4.4), sharex=True, sharey=True)
    for ax, m in zip(axes, (mid, last)):
        p = probs[probs.morning == m][zones].iloc[0]
        grid(ax)
        ax.bar(range(len(zones)), p, color=[ORANGE if z == hot else BLUE for z in zones], width=0.72, zorder=3)
        ax.axhline(0.95, color=INK, lw=1, ls=(0, (5, 3)))
        ax.set_xticks(range(len(zones)), [z[1:] for z in zones], fontsize=12)
        ax.set_title(f"Day {m}", loc="left", color=INK, fontsize=13)
        ax.set_ylabel("P(hot)")
    axes[1].set_xlabel("Zone")
    axes[0].set_ylim(0, 1.02)
    fig.tight_layout(h_pad=0.8)
    save(fig, "posterior_bars")


def fig_posterior_frames():
    """Slide 15: every morning's zone probabilities of the example campaign, as data for the animated bars
    (figures/posterior_frames.js, read by the slide's own script)."""
    probs = pd.read_csv(G / "example_campaign_probs.csv")
    an = pd.read_csv(G / "example_campaign_analyses.csv")
    hot = next(c for c in an.columns if c.startswith("share_from_")).removeprefix("share_from_")
    zones = [c for c in probs.columns if c != "morning"]
    # Every candidate site is a zone's outlet (or the plant), so a composite is drawn under its zone.
    outlet = zones_summary().set_index("cut_node").zone
    col = lambda site: "P" if site == PLANT else outlet[site][1:]
    lag = int(yaml.safe_load((ROOT / "_variables.yml").read_text())["LagTurnaround"])  # the paper's turnaround, days
    by_day = {int(d): [col(s) for s in g.site] for d, g in an.groupby("lab_day")}
    data = {"zones": [z[1:] for z in zones], "hot": hot[1:], "stop": STOP_PROB,
            "mornings": [int(m) for m in probs.morning],
            "p": [[round(float(v), 4) for v in row] for row in probs[zones].to_numpy()],
            # composites whose results arrive that morning (analysis started lag days earlier), and those
            # sent to the lab that morning
            "results": [by_day.get(int(m) - lag, []) for m in probs.morning],
            "sent": [by_day.get(int(m), []) for m in probs.morning], "lag": lag}
    OUT.mkdir(exist_ok=True)
    path = OUT / "posterior_frames.js"
    path.write_text("// Written by scripts/talk_figures.py (fig_posterior_frames); do not edit.\n"
                    f"window.POSTERIOR_FRAMES = {json.dumps(data, separators=(',', ':'))};\n")
    print(path.relative_to(ROOT))


def fig_campaign_map():
    """Slide 16: the base map (PNG, axes filling the figure so pixels map linearly to metres) and the zone
    shapes, label points, and site positions (figures/campaign_map.js) for the animated campaign map; the
    day-by-day probabilities and composites come from figures/posterior_frames.js."""
    n = Net()
    summary = zones_summary()
    _, hulls = zone_colours(n)
    x0, y0, x1, y1 = n.bounds
    pad = 250
    x0, y0, x1, y1 = x0 - pad, y0 - pad, x1 + pad, y1 + pad
    w_in = 3.6
    fig = plt.figure(figsize=(w_in, w_in * (y1 - y0) / (x1 - x0)))
    ax = fig.add_axes([0, 0, 1, 1])
    n.basemap(ax, minor=True, scale=0.6)
    ax.add_collection(n.pipes_lc(["#b3afa7"] * len(n.pipes), wmin=0.15, wmax=2.4))
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "campaign_map_base.png", dpi=220)  # no bbox trimming: the extent must be exact
    plt.close(fig)
    print("figures/campaign_map_base.png")

    def path(geom):
        parts = []
        for poly in getattr(geom, "geoms", [geom]):
            c = np.asarray(poly.simplify(40).exterior.coords)
            parts.append("M" + "L".join(f"{x:.0f},{-y:.0f}" for x, y in c) + "Z")
        return "".join(parts)
    zones = {}
    for z, h in hulls.items():
        pts = n.nodes[n.nodes.zone == z].geometry
        zones[z[1:]] = {"d": path(h), "lx": round(float(pts.x.median())), "ly": round(float(-pts.y.median()))}
    sites = {"P": [round(v) for v in n.xy(PLANT)]}
    for z, c in summary.set_index("zone")["cut_node"].items():
        if z != "rest":
            x, y = n.xy(c)
            sites[z[1:]] = [round(x), round(y)]
    sites = {k: [v[0], -v[1]] for k, v in sites.items()}
    data = {"viewBox": [round(x0), round(-y1), round(x1 - x0), round(y1 - y0)], "zones": zones, "sites": sites}
    (OUT / "campaign_map.js").write_text("// Written by scripts/talk_figures.py (fig_campaign_map); do not edit.\n"
                                        "// SVG coordinates: metres east, metres north negated.\n"
                                        f"window.CAMPAIGN_MAP = {json.dumps(data, separators=(',', ':'))};\n")
    print("figures/campaign_map.js")


def fig_plan_grid():
    """Slide 4: a plan as a grid of sites x days; each filled cell is one site-day. Left, the fixed plan
    that separates every pair (greedy_hotspot.csv, days placed one after another); right, the example
    adaptive campaign's analyses (example_campaign_analyses.csv). Zones aren't introduced until slide 8,
    so the figure is generic: one row per site, grouped as plant, arriving sewers, and upstream sites,
    with no site IDs or zone numbers."""
    W, _ = sens()
    summary = zones_summary()
    # Each site's role, in the order the rows are grouped: (key, marker colour, text colour, group label).
    roles = [("plant", ORANGE, "#c2501c", "The plant"),
             ("stream", "#8a5cc9", PURPLE, "Sewers\narriving\nat the plant"),
             ("up", BLUE, "#1f62b5", "Sites further\nupstream")]
    role_of = lambda s: "plant" if s == PLANT else "stream" if s in STREAMS else "up"
    cand = [s for s in W.index if np.isfinite(W.loc[s]).all()]
    sites = [s for r in roles for s in cand if role_of(s) == r[0]]
    marker_c = {r[0]: r[1] for r in roles}
    text_c = {r[0]: r[2] for r in roles}
    hot = pd.read_csv(G / "greedy_hotspot.csv")
    full = hot[hot.pairs_separable >= 1.0].iloc[0]
    fixed = {s: int(n) for s, n in re.findall(r"(\w+)x(\d+)", full.plan)}
    an = pd.read_csv(G / "example_campaign_analyses.csv")
    row = {s: i for i, s in enumerate(sites)}
    fig, axes = plt.subplots(1, 2, figsize=(W_FULL, H_FULL + 0.4), sharey=True,
                             gridspec_kw=dict(width_ratios=[1, 1.6]))
    a, b = axes
    for s, n in fixed.items():
        a.scatter(np.arange(1, n + 1), [row[s]] * n, s=70, marker="s", color=marker_c[role_of(s)], zorder=3)
    a.set_title(f"Separation plan: fixed in advance\n{int(full.site_days)} site-days",
                loc="left", color=INK, fontsize=13)
    a.set_xlabel("Days at each site (any order)")
    a.set_xlim(0.3, max(fixed.values()) + 0.7)
    for t, g in an.groupby("lab_day"):
        ss = list(g.site)
        for j, s in enumerate(ss):
            dx = 0.22 * (ss.count(s) > 1) * (1 if ss.index(s) != j else -1)
            b.scatter(t + dx, row[s], s=55, marker="s", color=marker_c[role_of(s)], zorder=3)
    b.set_title(f"Adaptive campaign: chosen day by day from the results\n{len(an)} site-days",
                loc="left", color=INK, fontsize=13)
    b.set_xlabel("Day of campaign")
    # The hot zone's outlet, and the morning its probability first reaches the stopping level.
    hot_zone = next(c.removeprefix("share_from_") for c in an.columns if c.startswith("share_from_"))
    hot_row = row[summary.set_index("zone").at[hot_zone, "cut_node"]]
    probs = pd.read_csv(G / "example_campaign_probs.csv")
    named = int(probs.loc[probs[hot_zone] >= STOP_PROB, "morning"].iloc[0])
    b.axhspan(hot_row - 0.5, hot_row + 0.5, color="#fbecc8", zorder=0)
    # label the band in the widest gap between its own composites, so no square covers the text
    days = sorted(set(an.loc[an.site == summary.set_index("zone").at[hot_zone, "cut_node"], "lab_day"]) | {0, named})
    gap = max(zip(days, days[1:]), key=lambda d: d[1] - d[0])
    b.text(sum(gap) / 2, hot_row, "the rise is here", ha="center", va="center", fontsize=12, color=INK,
           style="italic")
    b.scatter(named, hot_row, s=190, marker="*", color=INK, zorder=4)
    # one line in the row above the band, ending over the star (that row has nothing late in the campaign)
    b.text(named + 0.4, hot_row - 1, f"located (day {named})", ha="right",
           va="center", fontsize=11, color=INK)
    b.set_xlim(None, named + 1.6)
    breaks = [i - 0.5 for i in range(1, len(sites)) if role_of(sites[i]) != role_of(sites[i - 1])]
    for ax in axes:
        grid(ax, "x")
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        for y in breaks:  # a rule between the groups of sites
            ax.axhline(y, color=GRID, lw=1.2, zorder=1)
    # one label per group of rows, at its middle, instead of a site ID per row
    centres = [np.mean([i for i, s in enumerate(sites) if role_of(s) == r[0]]) for r in roles]
    a.set_yticks(centres, [r[3] for r in roles], fontsize=12)
    for lab, r in zip(a.get_yticklabels(), roles):
        lab.set_color(r[2])
        lab.set_fontweight("bold")
    for ax in axes:
        ax.tick_params(axis="y", length=0)
    a.set_ylim(len(sites) - 0.5, -0.5)  # inverted: plant on top
    fig.tight_layout(w_pad=2)
    save(fig, "plan_grid")


# --------------------------------------------------------------------------------------- equity
def _first_full_rank(columns, plan="Ds"):
    n = min(int(m.group(1)) for c in columns if (m := re.fullmatch(rf"{plan}, (\d+) site-days", c)))
    return f"{plan}, {n} site-days", n


def fig_equity_onmarg():
    """Slide 21: zone SD in the first full-rank D_s plan against each of the four ON-Marg dimensions, with
    Spearman's rho (matches the paper's EqRho* values)."""
    from scipy.stats import spearmanr
    zs = pd.read_csv(E / "minimax_zone_sd.csv", index_col="zone")
    col, start = _first_full_rank(zs.columns)
    zs = zs[col]
    zi = pd.read_csv(E / "zone_indices.csv", index_col=0)
    W, _ = sens()
    pt = pass_through(W, zones_summary()).reindex(zs.index).to_numpy()
    dims = (("onm_material", "Material resources"), ("onm_households", "Households and dwellings"),
            ("onm_age_labour", "Age and labour force"), ("onm_racialized", "Racialized and newcomer"))
    fig, axes = plt.subplots(1, 4, figsize=(W_FULL, H_FULL + 0.5), sharey=True)
    for ax, (c, title) in zip(axes, dims):
        grid(ax)
        x = zi.loc[zs.index, c]
        for m, colr, lab in ((~pt, AQUA, "Zone with its own site"), (pt, BLUE, "Pass-through zone")):
            ax.scatter(x[m], zs[m], s=70, color=colr, label=lab, zorder=3, edgecolor="white", linewidths=0.8)
        r = spearmanr(x, zs)
        ax.set_title(f"{title}\nρ = {r.statistic:.2f}, p = {r.pvalue:.2f}", loc="left", fontsize=13, color=INK)
        ax.tick_params(axis="x", labelsize=11)
    ax = axes[0]
    ax.set_yscale("log")
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_yticks([1, 2, 4, 8], ["1", "2", "4", "8"])
    ax.set_ylabel("Uncertainty of zone estimate\n(SD, ln units; higher = worse)", fontsize=13)
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, fontsize=13, bbox_to_anchor=(0.5, -0.02))
    fig.supxlabel("ON-Marg score of the zone (higher = more marginalized)", fontsize=14, y=0.1)
    fig.subplots_adjust(left=0.1, right=0.99, top=0.84, bottom=0.25, wspace=0.14)
    save(fig, "equity_onmarg")


def fig_equity_minimax():
    """Slide 20 (left): worst and best zone SD against site-days, D_s and minimax designs."""
    st = pd.read_csv(E / "minimax_steps.csv")
    zs = pd.read_csv(E / "minimax_zone_sd.csv", index_col="zone")
    _, start = _first_full_rank(zs.columns)
    fig, ax = plt.subplots(figsize=(6.4, 4.3))
    grid(ax)
    for plan, col, lab in (("Ds", BLUE, "Precision plan"), ("minimax", PURPLE, "Fair plan")):
        q = st[st.plan == plan]
        ax.plot(q.site_days, q.max_sd, color=col, lw=2.6, label=f"{lab}: worst zone")
        ax.plot(q.site_days, q.min_sd, color=col, lw=1.4, ls="--", label=f"{lab}: best zone")
    for fac in PARAMS["design"]["equity_factors"]:
        ax.axhline(np.log(fac), color=MUTED, lw=0.8, ls=":")
        ax.text(st.site_days.max() * 1.05, np.log(fac), f"within ×{fac}", fontsize=12, color=MUTED, va="center")
    budget = int(yaml.safe_load((ROOT / "_variables.yml").read_text())["EqFactorThreeMm"])
    ax.axvline(budget, color=PURPLE, lw=1, ls=(0, (2, 2)))  # the budget shown in the maps beside it
    ax.text(budget * 1.04, 0.16, f"{budget}: maps", fontsize=11, color=PURPLE, va="bottom")
    ax.set_yscale("log"); ax.set_xscale("log")
    for a in (ax.xaxis, ax.yaxis):
        a.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_yticks([0.25, 0.5, 1, 2, 4, 8], ["0.25", "0.5", "1", "2", "4", "8"])
    ticks = [start] + [t for t in (30, 100, 300, 1000) if t > start]
    ax.set_xticks(ticks, [f"{t:,}" for t in ticks])
    ax.set_xlabel("Site-days"); ax.set_ylabel("Zone SD (ln units)")
    ax.legend(loc="upper right", fontsize=11)
    save(fig, "equity_minimax")


def fig_minimax_map():
    """Slide 20 (right): where the D_s and minimax designs put their days at the minimax design's
    factor-of-3 budget (EqFactorThreeMm site-days); markers sized by days, coloured by role as on slide 4."""
    n = Net()
    W, _ = sens()
    summary = zones_summary()
    pt = pass_through(W, summary)
    zone_of = {c: z for z, c in summary.set_index("zone")["cut_node"].items() if z != "rest"}
    role_c = {"plant": ORANGE, "stream": "#8a5cc9", "pass": BLUE, "own": AQUA}
    role_of = lambda s: ("plant" if s == PLANT else "stream" if s in STREAMS
                         else "pass" if pt[zone_of[s]] else "own")
    st = pd.read_csv(E / "minimax_steps.csv")
    budget = int(yaml.safe_load((ROOT / "_variables.yml").read_text())["EqFactorThreeMm"])
    fig, axes = plt.subplots(1, 2, figsize=(5.8, 4.5))
    for ax, (plan, title) in zip(axes, (("Ds", "Precision plan"), ("minimax", "Fair plan"))):
        r = st[(st.plan == plan) & (st.site_days == budget)].iloc[0]
        days = {s: int(c) for s, c in re.findall(r"(\w+)x(\d+)", r.plan_sites)}
        n.basemap(ax, scale=0.5)
        ax.add_collection(n.pipes_lc(["#a8a49c"] * len(n.pipes), wmin=0.15, wmax=2.4))
        px, py = n.xy(PLANT)
        top = sorted(days, key=lambda k: -days[k])[:3]
        for site, d in sorted(days.items(), key=lambda kv: -kv[1]):
            x, y = n.xy(site)
            ax.scatter([x], [y], s=8 * d, color=role_c[role_of(site)], edgecolor="white", linewidths=0.8,
                       zorder=10, alpha=0.85)
            if plan == "minimax" and site in top:  # the busiest sites, labelled outward from the plant
                v = np.array([x - px, y - py]); v = v / (np.hypot(*v) or 1)
                ax.annotate(f"{d} days", (x, y), xytext=tuple(30 * v + np.array([0, 4])), textcoords="offset points",
                            ha="left" if v[0] >= 0 else "right", va="center", fontsize=13, color=INK, zorder=11,
                            arrowprops=dict(arrowstyle="-", color=INK, lw=0.7),
                            path_effects=[matplotlib.patheffects.withStroke(linewidth=2.5, foreground="white")])
        n.frame(ax, pad=-600, scalebar=False)  # crop the empty rural edge
        spread = "same days everywhere" if plan == "Ds" else "days go to the trunk"
        ax.set_title(f"{title}:\n{spread}\nworst zone SD {r.max_sd:.2f}", fontsize=13, color=INK)
    h = [Line2D([], [], ls="", marker="o", ms=8, color=c, label=l) for c, l in
         ((ORANGE, "plant"), ("#8a5cc9", "arriving stream"), (BLUE, "pass-through outlet"), (AQUA, "own outlet"))]
    fig.legend(handles=h, loc="lower center", ncol=2, fontsize=13, bbox_to_anchor=(0.5, -0.02), columnspacing=1.0,
               handletextpad=0.3)
    fig.suptitle(f"Days at each site, {budget} site-days (area = days)", fontsize=13, color=MUTED, y=1.0)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.78, bottom=0.15, wspace=0.04)
    save(fig, "minimax_map", raster=True)


def fig_confusion():
    """Appendix 6: which zone the best guess names when each zone is hot, simplest case (normal noise, no
    background, kappa known), for three fixed plans (localization_confusion_*.csv; rows sum to one)."""
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("named", ["#ffffff", "#c9b5e6", "#7651b4", PURPLE])
    plans = (("inlets_7_days_each", "Plant and arriving sewers,\n7 days each"),
             ("greedy_hot-spot_38_site-days", "Separation plan,\n38 site-days"),
             (f"all_{n_sites()}_sites_7_days", f"All {n_sites()} sites,\n7 days each"))
    fig, axes = plt.subplots(1, 3, figsize=(W_FULL, 4.3))
    for ax, (f, title) in zip(axes, plans):
        c = pd.read_csv(G / f"localization_confusion_{f}.csv", index_col=0)
        im = ax.imshow(c.to_numpy(), cmap=cmap, vmin=0, vmax=1)
        labs = [z[1:] for z in c.index]
        ax.set_xticks(range(len(labs)), labs, fontsize=9, rotation=90)
        ax.set_yticks(range(len(labs)), labs, fontsize=9)
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
        right = 100 * np.diag(c.to_numpy()).mean()
        ax.set_title(f"{title}: best guess right {right:.0f} %", fontsize=12, color=INK)
        ax.set_xlabel("Zone named (best guess)", fontsize=12)
    axes[0].set_ylabel("Zone that is hot", fontsize=12)
    cb = fig.colorbar(im, ax=axes, fraction=0.015, pad=0.015)
    cb.set_label("Share of runs", fontsize=12); cb.ax.tick_params(labelsize=10)
    save(fig, "confusion")


# ------------------------------------------------------------------------------------- appendix
def fig_sigma_sweep():
    """Appendix: site-days needed against the upstream noise."""
    sw = pd.read_csv(G / "localization_sigma_sweep.csv")
    up = sw[sw.noise.str.startswith("upstream")]
    fig, ax = plt.subplots(figsize=(W_FULL, H_FULL))
    grid(ax)
    ax.plot(up.sigma_upstream, up.greedy_all_pairs_site_days, color=BLUE, marker="o", ms=7, label="Separation plan, all pairs ≥ 2σ")
    ax.plot(up.sigma_upstream, up.adaptive_K2_site_days_median, color=INK, marker="*", ms=12, label="Adaptive, 2 upstream a day, results the same day")
    ax.axvline(float(PARAMS["noise"]["sigma_ln_upstream"]), color=MUTED, lw=1, ls=":")
    ax.set_xlabel("Upstream σ (the plant's σ scaled with it, in the measured ratio)"); ax.set_ylabel("Site-days")
    ax.set_ylim(0, None)
    ax.legend(loc="upper left")
    save(fig, "sigma_sweep")


def fig_campaign_length():
    """Appendix: median campaign length against the number of upstream composites a day."""
    ad = pd.read_csv(G / "localization_adaptive.csv").sort_values("K")
    lg = pd.read_csv(G / "localization_adaptive_lag.csv")
    held = lg[(lg["mode"] == "held") & (lg.case == "realistic")].sort_values("K")
    fig, ax = plt.subplots(figsize=(W_FULL, H_FULL))
    grid(ax)
    ax.plot(ad.K, ad.days_median, color=INK, marker="*", ms=12, label="Results the same day")
    ax.plot(held.K, held.days_median, color=PURPLE, marker="*", ms=13, mfc="white", label="Held samples, 2-day lab")
    ax.set_xscale("log")
    ks = sorted(ad.K.unique())
    ax.set_xticks(ks, [f"{k:g}" for k in ks])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlabel("Upstream composites analyzed a day, K"); ax.set_ylabel("Campaign length, days (median)")
    ax.set_ylim(0, None)
    ax.legend(loc="upper right")
    save(fig, "campaign_length")


def fig_multi_zone():
    """Appendix 10: the analysis that also allows two hot zones, as bars of the shares in _variables.yml
    (Multi*), grouped by the simulated truth: one hot zone, two, or none."""
    V = yaml.safe_load((ROOT / "_variables.yml").read_text())
    rows = [("One hot zone", BLUE, "First zone named is the hot one", "MultiOneFirstCorrect"),
            ("One hot zone", BLUE, "First names that are right", "MultiOneFirstGivenNamed"),
            ("One hot zone", BLUE, "Confirmed as the only hot zone", "MultiOneConfirmed"),
            ("Two hot zones", PURPLE, "A first zone named", "MultiTwoFirstNamed"),
            ("Two hot zones", PURPLE, "First names that are a hot zone", "MultiTwoFirstGivenNamed"),
            ("Two hot zones", PURPLE, "The exact pair named", "MultiTwoPair"),
            ("No hot spot", MUTED, "A zone or pair named", "MultiNullFalse")]
    fig, ax = plt.subplots(figsize=(W_FULL, 4.0))
    y, ys, groups = 0, [], {}
    for i, (g, c, lab, key) in enumerate(rows):
        if i and g != rows[i - 1][0]:
            y += 0.6  # a gap between the groups
        v = float(V[key])
        ax.barh(y, v, color=c, height=0.72)
        ax.text(v + 1, y, f"{v:g}\u2009%", va="center", fontsize=12, color=INK)
        groups.setdefault(g, []).append(y)
        ys.append((y, lab)); y += 1
    # the one-zone analysis of the talk, for the first bar
    one = float(V["HeldKTwoCorrect"])
    ax.plot([one, one], [-0.42, 0.42], color=INK, lw=2)
    ax.annotate(f"one-zone analysis: {one:g}\u2009%", (one, -0.42), xytext=(one + 4, -0.9), fontsize=11,
                color=INK, va="center", arrowprops=dict(arrowstyle="-", color=INK, lw=0.7))
    ax.set_yticks([a for a, _ in ys], [b for _, b in ys], fontsize=12)
    ax.invert_yaxis()
    for g, yy in groups.items():  # group label to the right of its bars
        ax.text(112, np.mean(yy), g, va="center", ha="left", fontsize=13, color=INK, weight="bold")
    ax.set_xlim(0, 135); ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Share of campaigns (%)")
    grid(ax, "x")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    save(fig, "multi_zone")


def _graph(n: Net):
    import networkx as nx
    G = nx.MultiDiGraph()
    G.add_edges_from(zip(n.pipes.from_node, n.pipes.to_node, n.pipes.pipe_id))
    return G


# Register examples (data/london_data_issues.csv): (id, pipes, kind, title). "cut": the nodes that reach the
# plant only through these pipes; "upstream": everything that drains through the pipe.
DATA_FIXES = (("D08", ("328065", "328066"), "cut", "Two Ridout St pipes listed\nas proposed are in service"),
              ("D28", ("900001",), "cut", "Berkshire Village force main\nends 25 m short of the plant"),
              ("D05", ("142170",), "upstream", "An invert of 25,089 m\n(for 250.89 m) blocks a trunk"))


def fig_data_fixes():
    """Appendix 2: three entries of the register of data issues on the model network. For the two missing
    links, the part of the final network that reaches the plant only through them; for the invert typo,
    the part that drains through the pipe, which the typo backed up."""
    import networkx as nx
    n = Net()
    G = _graph(n)
    reach = nx.ancestors(G, PLANT) | {PLANT}
    issues = pd.read_csv(PAPER / "data" / "london_data_issues.csv").set_index("id")
    fig, axes = plt.subplots(1, 3, figsize=(W_FULL, 5.0))
    for ax, (iid, ids, kind, title) in zip(axes, DATA_FIXES):
        assert iid in issues.index, iid
        if kind == "cut":
            H = G.copy()
            H.remove_edges_from([(u, v, k) for u, v, k in G.edges(keys=True) if k in ids])
            hit = reach - (nx.ancestors(H, PLANT) | {PLANT})
            what = "would not reach the plant"
        else:
            u = n.pipes.set_index("pipe_id").at[ids[0], "from_node"]
            hit = nx.ancestors(G, u) | {u}
            what = "drain through this pipe"
        n.basemap(ax, scale=0.5)
        on = n.pipes.from_node.isin(hit).to_numpy()
        ax.add_collection(n.pipes_lc(np.where(on, ORANGE, "#cfcdc8"), wmin=0.25, wmax=3.0))
        fx = n.pipes[n.pipes.pipe_id.isin(ids)]
        c = fx.geometry.union_all().centroid
        ax.scatter([c.x], [c.y], s=420, facecolor="none", edgecolor=PURPLE, linewidths=2.4, zorder=12)
        x, y = n.xy(PLANT)
        ax.scatter([x], [y], s=200, marker="*", color=INK, edgecolor="white", linewidths=1.0, zorder=11)
        n.frame(ax, scalebar=ax is axes[-1])
        people = n.nodes.loc[[h for h in hit if h in n.nodes.index], "population"].sum()
        ax.set_title(f"{iid}: {title}", loc="left", fontsize=13, color=INK)
        ax.text(0.02, -0.02, f"{len(hit):,} nodes, about {round(people, -2):,.0f} people,\n{what}",
                transform=ax.transAxes, fontsize=12,
                color="#c2501c", va="top")
    fig.legend(handles=[Line2D([], [], color=ORANGE, lw=3, label="affected part of the network"),
                        Line2D([], [], ls="", marker="o", mfc="none", mec=PURPLE, mew=2, ms=13, label="the pipe in the register"),
                        Line2D([], [], ls="", marker="*", color=INK, ms=13, label="treatment plant")],
               loc="lower center", ncol=3, fontsize=12, bbox_to_anchor=(0.5, -0.02))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.2, wspace=0.05)
    save(fig, "data_fixes", raster=True)


def fig_data_register():
    """Appendix 2: the register of data issues by kind and status."""
    d = pd.read_csv(PAPER / "data" / "london_data_issues.csv")
    tab = d.groupby(["area", "status"]).size().unstack(fill_value=0)
    status = [("resolved", AQUA, "resolved"), ("provisional", YELLOW, "provisional (an assumption)"),
              ("open", ORANGE, "open")]
    tab = tab.reindex(columns=[s for s, *_ in status], fill_value=0)
    tab = tab.loc[tab.sum(axis=1).sort_values().index]
    fig, ax = plt.subplots(figsize=(W_COL, 4.2))
    grid(ax, "x")
    left = np.zeros(len(tab))
    for s, col, lab in status:
        ax.barh(range(len(tab)), tab[s], left=left, color=col, label=f"{lab} ({tab[s].sum()})", height=0.7, zorder=3)
        left += tab[s].to_numpy()
    ax.set_yticks(range(len(tab)), tab.index)
    ax.set_xlabel("Issues in the register")
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.legend(loc="lower right", fontsize=12)
    save(fig, "data_register")


def fig_limits():
    """Appendix 11: hot zone named at 95 % against the plant N1 concentration under uniform shedding, with
    composites censored at the laboratory's limits (the paper's fig_si_limits)."""
    d = pd.read_csv(G / "detection.csv")
    d["plant_conc"] = d.plant_conc.astype(float)
    q = pd.read_csv(G / "detection_loq_only.csv")
    q["plant_conc"] = q.plant_conc.astype(float)
    lv = pd.read_csv(PAPER / "scenarios" / "odm" / "n1_levels.csv").set_index("site")
    det = PARAMS["detection"]
    greedy = next(p for p in d.plan.unique() if re.fullmatch(r"greedy hot-spot, \d+ site-days", p))
    plans = (("adaptive, held, K=2", INK, "*", 13, "Adaptive, held, 2 a day"),
             (f"all {n_sites()} sites, 7 days", AQUA, "s", 7, f"All {n_sites()} sites, 7 days"),
             (greedy, BLUE, "o", 7, "Separation plan, " + greedy.split(", ")[1]))
    fig, ax = plt.subplots(figsize=(W_FULL - 1.5, H_COL))
    grid(ax)
    xmax, xnl = 16000, 32000
    ax.axvspan(lv.loc["GRWY", "q05"], xmax * 1.25, color=GREY, alpha=0.13, lw=0)
    ax.text(lv.loc["GRWY", "q05"] * 1.08, 96, "Greenway plant: 95 % of\ndates above this", fontsize=12, color=MUTED,
            va="top")
    ax.axvline(xmax * 1.25, color=MUTED, lw=0.6)
    for v, t in ((det["slod_gc_L"], "sLOD"), (det["sloq_gc_L"], "sLOQ")):
        ax.axvline(v, color=MUTED, lw=0.9, ls=":")
        ax.text(v, 103, t, fontsize=12, color=MUTED, ha="center", va="bottom")
    for plan, col, mk, ms, lab in plans:
        x = d[d.plan == plan].set_index("plant_conc").sort_index()
        fin = x[np.isfinite(x.index)]
        ax.plot(fin.index, 100 * fin.confident, color=col, marker=mk, ms=ms, lw=2.2, label=lab)
        ax.plot([xnl], [100 * x.loc[np.inf, "confident"]], color=col, marker=mk, ms=ms, mfc="white", ls="none")
        y = q[q.plan == plan].sort_values("plant_conc")
        ax.plot(y.plant_conc, 100 * y.confident, color=col, lw=1.2, ls="--")
    ax.plot([], [], color=MUTED, lw=1.2, ls="--", label="censored at the sLOQ only")
    ax.set_xscale("log")
    ax.set_xlim(240, xnl * 1.3)
    ax.set_xticks([300, 1000, 3000, 10000, xnl], ["300", "1,000", "3,000", "10,000", "no\nlimits"])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("Plant N1 concentration under uniform shedding (gc/L)")
    ax.set_ylabel("Hot zone named at 95 % (%)")
    ax.set_ylim(0, 102)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=2, fontsize=12)
    save(fig, "limits")


def _zone_hulls(pipes, zone_col, buffer=310):
    from shapely.ops import unary_union
    return {z: unary_union(list(g.geometry)).buffer(buffer).buffer(-250)
            for z, g in pipes.groupby(zone_col) if z != "rest"}


def fig_zones_onmarg():
    """Appendix 12: the alternative zoning cut by ON-Marg material resources, each zone shaded and labelled
    by its mean score, with its candidate sites (the paper's fig_zones_onmarg)."""
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    Z = E / "zoning_onm_material"
    zmap = pd.read_csv(Z / "zones_proposed.csv", index_col="node")["zone"]
    summary = pd.read_csv(Z / "zones_summary.csv").set_index("zone")
    score = summary["onm_material"].drop("rest", errors="ignore")
    sites = [str(s) for s in np.load(Z / "greenway" / "sensitivities.npz")["sites"]]
    n = Net()
    n.pipes["zone_onm"] = n.pipes.from_node.map(zmap).fillna("rest")
    hulls = _zone_hulls(n.pipes, "zone_onm")
    cmap = LinearSegmentedColormap.from_list("onm", [AQUA, "#f3f2ee", PURPLE])
    lim = float(np.abs(score).max())
    norm = TwoSlopeNorm(0, -lim, lim)
    colour = {z: cmap(norm(s)) for z, s in score.items()} | {"rest": GREY}
    fig, ax = plt.subplots(figsize=(4.6, 6.4))
    n.basemap(ax)
    for z, h in hulls.items():
        for part in getattr(h, "geoms", [h]):
            x, y = part.exterior.xy
            ax.fill(x, y, color=colour[z], alpha=0.35, lw=0, zorder=3)
            ax.plot(x, y, color=MUTED, lw=0.7, alpha=0.8, zorder=4)
    ax.add_collection(n.pipes_lc([colour[z] if z != "rest" else GREY for z in n.pipes.zone_onm], wmin=0.25, wmax=3.0))
    for s in sites:
        if s in n.nodes.index:
            x, y = n.xy(s)
            ax.scatter([x], [y], s=45, facecolor=INK, edgecolor="white", linewidths=1.0, zorder=10)
    nz = n.nodes.assign(zone_onm=n.nodes.index.map(zmap))
    for z, s in score.items():
        pts = nz[nz.zone_onm == z].geometry
        ax.text(pts.x.median(), pts.y.median(), f"{s:+.2f}", fontsize=11, fontweight="bold", color=INK,
                ha="center", va="center", zorder=9,
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.85))
    x, y = n.xy(PLANT)
    ax.scatter([x], [y], s=240, marker="*", color=INK, edgecolor="white", linewidths=1.2, zorder=11)
    n.frame(ax)
    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    cax = ax.inset_axes([0.62, 0.93, 0.36, 0.025])
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_ticks([-0.5, 0, 0.5]); cb.ax.tick_params(labelsize=10)
    cb.set_label("material resources score\n(higher = more marginalized)", fontsize=10)
    cb.ax.xaxis.set_label_position("top")
    ax.legend(handles=[Line2D([], [], ls="", marker="o", color=INK, ms=7, label="candidate site"),
                       Line2D([], [], ls="", marker="*", color=INK, ms=12, label="treatment plant")],
              loc="lower left", fontsize=11, frameon=True, facecolor="white", edgecolor="none", framealpha=0.9)
    save(fig, "zones_onmarg", raster=True)


POOL_FIG_SITE_DAYS = 100  # the paper's PoolFigSiteDays


def fig_candidates():
    """Appendix 12: the wider pool of candidate sites on the main zoning, by kind; filled where the best
    fractional minimax plan of 100 site-days spends at least half a day (the paper's fig_candidates)."""
    k = pd.read_csv(E / "pool_min5000_flow" / "sites.csv")
    c = k[k["kept"].astype(bool)].set_index("node")["kind"]
    rel = pd.read_csv(E / "alternatives_relaxed.csv")
    plan = rel[(rel.case == "pool") & (rel.site_days == POOL_FIG_SITE_DAYS)].iloc[0].plan
    used = {s for s, v in re.findall(r"(\w+) ([\d.]+)", plan) if float(v) >= 0.5}
    n = Net()
    colour, hulls = zone_colours(n)
    fig, ax = plt.subplots(figsize=(4.6, 6.4))
    n.basemap(ax)
    for z, h in hulls.items():
        for part in getattr(h, "geoms", [h]):
            x, y = part.exterior.xy
            ax.fill(x, y, color=colour[z], alpha=0.06, lw=0, zorder=3)
            ax.plot(x, y, color=colour[z], lw=0.7, alpha=0.6, zorder=4)
    ax.add_collection(n.pipes_lc([matplotlib.colors.to_rgba(colour[z], 0.45) for z in n.pipes.zone], wmin=0.25, wmax=3.0))
    marks = {"current": ("o", 50, "current site (zone outlet)"), "pump station": ("s", 50, "pumping station"),
             "flow monitor": ("D", 38, "flow-monitoring\nmaintenance hole"),
             "off-road maintenance hole": ("^", 38, "off-road maintenance hole"),
             "trunk maintenance hole": ("P", 60, "trunk maintenance hole")}
    handles = []
    for kind, (m, size, lab) in marks.items():
        nodes = [s for s in c.index[c == kind] if s in n.nodes.index]
        for filled in (False, True):
            sel = [s for s in nodes if (s in used) == filled]
            if sel:
                g = n.nodes.geometry[sel]
                ax.scatter(g.x, g.y, s=size, marker=m, facecolor=PURPLE if filled else "white",
                           edgecolor=PURPLE if filled else INK, linewidths=0.9, zorder=10 + filled)
        if nodes:
            handles.append(Line2D([], [], ls="", marker=m, mfc="white", mec=INK, ms=7, label=lab))
    handles.append(Line2D([], [], ls="", marker="o", color=PURPLE, ms=7,
                          label=f"used by the best fair\nplan, {POOL_FIG_SITE_DAYS} site-days"))
    x, y = n.xy(PLANT)
    ax.scatter([x], [y], s=240, marker="*", color=INK, edgecolor="white", linewidths=1.2, zorder=12)
    n.frame(ax)
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=2, fontsize=10,
              handletextpad=0.3, labelspacing=0.4, columnspacing=1.0)
    save(fig, "candidates", raster=True)


FIGURES = [fig_catchment, fig_dilution, fig_plan_grid, fig_network, fig_blocks, fig_blocks_legend, fig_flow_inset, fig_hydraulics, fig_zones, fig_zone_tree, fig_representation, fig_noise, fig_noise_tails,
           fig_identical_shift, fig_level_contrasts, fig_design_rank, fig_greedy_steps, fig_greedy_frames, fig_design_pairs, fig_posterior_bars, fig_localisation_fixed,
           fig_example_campaign, fig_campaign_map, fig_posterior_frames, fig_localisation_adaptive, fig_kappa, fig_mismatch, fig_equity_onmarg, fig_minimax_map, fig_confusion,
           fig_equity_minimax, fig_sigma_sweep, fig_campaign_length, fig_multi_zone, fig_data_fixes, fig_data_register, fig_limits,
           fig_zones_onmarg, fig_candidates]

if __name__ == "__main__":
    want = set(sys.argv[1:])
    for f in FIGURES:
        if not want or f.__name__ in want or f.__name__.removeprefix("fig_") in want:
            f()

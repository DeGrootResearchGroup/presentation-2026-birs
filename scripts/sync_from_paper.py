"""Copy the paper's result files into data/paper/ and export the map layers, so that the talk's figures
rebuild from this repository alone.

Run it with the paper repository's own interpreter, since the map export uses the paper's network code
(swmmbuilder):

    ~/JRN-059/.venv/bin/python scripts/sync_from_paper.py [--paper ~/JRN-059] [--allow-dirty]
    ~/JRN-059/.venv/bin/python scripts/sync_from_paper.py --blocks-only   # replace only the blocks layer

What it writes:

- ``data/paper/<same relative path as in the paper>``: verbatim copies of the files in MANIFEST.
- ``data/paper/constants.yaml``: the few constants the figures read from the paper's code rather than
  from a result file (the outlet, the own-site share, the separation threshold, the background tau).
- ``data/network.gpkg``: the Greenway catchment's pipes (zone, population upstream, force main), its
  nodes (zone), its pumping stations, its census dissemination blocks (population), and the basemap
  clipped to the catchment, all in the paper's metric CRS.
- ``_variables.yml``: every macro of ``paper/generated_values.tex``, for ``{{< var Name >}}`` on the
  slides.
- ``data/paper/SOURCE.txt``: the paper commit, whether its tree was clean, and the date.

It refuses to run on a paper repository with uncommitted changes unless ``--allow-dirty`` is given, in
which case SOURCE.txt says so. Copy from a tagged, clean commit before the talk.
"""

import argparse
import datetime as dt
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = DATA / "paper"

# Result files the talk's figures read, as globs relative to the paper repository.
MANIFEST = [
    "config/params.yaml",
    "paper/generated_values.tex",
    "hydraulics/greenway/outlet_flow_day3.csv",
    "hydraulics/greenway/node_loading.csv",
    "data/plant/greenway_daily_mps.csv",
    "data/london_data_issues.csv",
    "scenarios/zones/zones_proposed.csv",
    "scenarios/zones/zones_summary.csv",
    "scenarios/greenway/sensitivities.npz",
    "scenarios/greenway/plans.csv",
    "scenarios/greenway/greedy.csv",
    "scenarios/greenway/greedy_hotspot.csv",
    "scenarios/greenway/zone_sd.csv",
    "scenarios/greenway/discrimination_*.csv",
    "scenarios/greenway/localization_*.csv",
    "scenarios/greenway/example_campaign_*.csv",
    "scenarios/greenway/detection*.csv",
    "scenarios/odm/noise_components*.csv",
    "scenarios/odm/n1_levels.csv",
    "scenarios/odm/noise_tails*.csv",
    "scenarios/equity/minimax_*.csv",
    "scenarios/equity/zone_indices.csv",
    "scenarios/equity/constrained_steps_r2.csv",
    "scenarios/equity/alternatives_relaxed.csv",
    "scenarios/equity/trunk_sites.csv",
    "scenarios/equity/pool_min5000_flow/sites.csv",
    "scenarios/equity/zoning_onm_material/zones_proposed.csv",
    "scenarios/equity/zoning_onm_material/zones_summary.csv",
    "scenarios/equity/zoning_onm_material/greenway/sensitivities.npz",
]

BASEMAP_PAD_M = 1500  # basemap kept this far beyond the catchment's pipes
SIMPLIFY_M = 3.0  # basemap geometry simplification tolerance, well below a pixel on a slide map


def yaml_outlet() -> str:
    import yaml
    return yaml.safe_load((OUT / "constants.yaml").read_text())["outlet"]


def git(paper: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(paper), *args], capture_output=True, text=True, check=True).stdout.strip()


def copy_manifest(paper: Path) -> list[Path]:
    copied = []
    for pattern in MANIFEST:
        hits = sorted(paper.glob(pattern))
        if not hits:
            sys.exit(f"no file in the paper matches {pattern}")
        for src in hits:
            rel = src.relative_to(paper)
            dst = OUT / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            copied.append(rel)
    return copied


def constants(paper: Path) -> dict:
    """Constants that the paper's figures take from its code, read from the source text (not imported),
    so that a renamed constant fails loudly here."""
    def grab(rel, name):
        m = re.search(rf"^{name} = ([\d.]+)", (paper / rel).read_text(), re.M)
        if not m:
            sys.exit(f"{name} not found in {rel}")
        return float(m.group(1))
    import yaml
    params = yaml.safe_load((paper / "config" / "params.yaml").read_text())
    return {
        "outlet": params["hydraulics"]["outlet"],
        "composite": params["design"]["composite"],
        "own_share": grab("scenarios/common.py", "OWN_SHARE"),
        "separation": grab("scenarios/greenway_design.py", "SEPARATION"),
        "tau_analysis": grab("scenarios/greenway_adaptive.py", "TAU_A"),
        "tau_cautious": grab("scenarios/greenway_adaptive.py", "TAU_CAUTIOUS"),
    }


def inp_sections(path: Path) -> dict[str, list[list[str]]]:
    """The rows of each [SECTION] of a SWMM input file, split on whitespace, comments dropped."""
    out: dict[str, list[list[str]]] = {}
    sec = None
    for line in path.read_text().splitlines():
        line = line.split(";", 1)[0].strip()
        if not line:
            continue
        if line.startswith("["):
            sec = line.strip("[]").upper()
            out.setdefault(sec, [])
        elif sec:
            out[sec].append(line.split())
    return out


def export_network(paper: Path, outlet: str, blocks_only: bool = False) -> None:
    """The map layers, from the paper's network code. Pipe width on the maps is the population upstream,
    shared equally at flow splits (swmmbuilder's accumulate), as in the paper's zone maps."""
    import geopandas as gpd
    import pandas as pd
    from shapely.geometry import box

    sys.path.insert(0, str(paper))
    from figures.basemap import load_basemap
    from hydraulics.greenway_data import load_network
    from scenarios.common import METRIC_CRS

    net = load_network(lambda msg: None)
    sub = net.catchment(outlet)
    export_blocks(paper, net, sub)
    if blocks_only:
        return
    G = sub.graph
    zones = pd.read_csv(paper / "scenarios" / "zones" / "zones_proposed.csv", index_col="node")["zone"]
    pop = pd.read_csv(paper / "hydraulics" / "greenway" / "node_loading.csv", index_col="node")["population"]
    pop = pop.reindex(list(G)).fillna(0.0)
    up = sub.accumulate(pop).pipes
    total = float(pop.sum())

    P = net.full_data.set_index("pipe_id").to_crs(METRIC_CRS)
    rows = []
    for u, v, k in G.edges(keys=True):
        g = P.geometry[k]
        if g is None or g.is_empty:
            continue
        rows.append(dict(pipe_id=str(k), from_node=str(u), to_node=str(v), zone=zones.get(u, "rest"),
                         up_pop=float(up[k]), up_frac=min(float(up[k]) / total, 1.0),
                         flow_type=str(P.at[k, "flow_type"]), diameter=P.at[k, "diameter"], geometry=g))
    pipes = gpd.GeoDataFrame(rows, crs=METRIC_CRS)

    N = net.nodes.to_crs(METRIC_CRS)
    keep = [n for n in G if n in N.index]
    nodes = gpd.GeoDataFrame({"node": keep, "zone": [zones.get(n, "rest") for n in keep],
                              "node_type": N.loc[keep, "node_type"].astype(str).to_numpy(),
                              "population": pop.reindex(keep).to_numpy()},
                             geometry=N.geometry[keep].to_numpy(), crs=METRIC_CRS)
    # Pumping stations and force mains as the built SWMM model has them (the paper's NetPumps count).
    inp = inp_sections(paper / "hydraulics" / "greenway" / "greenway.inp")
    pump_ids = {row[1] for row in inp["PUMPS"]}
    force_mains = {row[0] for row in inp["XSECTIONS"] if row[1] == "FORCE_MAIN"}
    nodes["pump_station"] = nodes["node"].isin(pump_ids)
    pipes["force_main"] = pipes["pipe_id"].isin(force_mains)
    missing = pump_ids - set(nodes["node"])
    if missing:
        sys.exit(f"pumping stations not among the catchment's nodes: {sorted(missing)}")

    xmin, ymin, xmax, ymax = pipes.total_bounds
    clip = box(xmin - BASEMAP_PAD_M, ymin - BASEMAP_PAD_M, xmax + BASEMAP_PAD_M, ymax + BASEMAP_PAD_M)
    gpkg = DATA / "network.gpkg"
    blocks = gpd.read_file(gpkg, layer="blocks")  # written by export_blocks, kept across the rewrite
    gpkg.unlink(missing_ok=True)
    blocks.to_file(gpkg, layer="blocks", driver="GPKG")
    pipes.to_file(gpkg, layer="pipes", driver="GPKG")
    nodes.to_file(gpkg, layer="nodes", driver="GPKG")
    for name, layer in load_basemap().items():
        c = gpd.clip(layer[["geometry"]], clip)
        c = c[~c.geometry.is_empty]
        c["geometry"] = c.geometry.simplify(SIMPLIFY_M)
        c.to_file(gpkg, layer=f"basemap_{name}", driver="GPKG")
    print(f"network.gpkg: {len(pipes):,} pipes, {len(nodes):,} nodes, {int(nodes.pump_station.sum())} pumping stations")


def export_blocks(paper: Path, net, sub) -> None:
    """The census dissemination blocks that load the catchment, with their population, for the map of
    how people are placed on the pipes. The paper shares each block's people among the pipes within
    reach (swmmbuilder's census allocation), which it does not write out block by block; for the map a
    block is drawn when the nearest pipe of the whole network to the block lies in the catchment."""
    import geopandas as gpd
    from scenarios.common import METRIC_CRS

    b = gpd.read_file(paper / "data" / "census" / "blocks_2021_london.gpkg", layer="polygons").to_crs(METRIC_CRS)
    P = net.full_data[["pipe_id", "geometry"]].to_crs(METRIC_CRS)
    P = P[P.geometry.notna() & ~P.geometry.is_empty]
    pts = gpd.GeoDataFrame(b[["id"]], geometry=b.geometry.representative_point(), crs=METRIC_CRS)
    near = gpd.sjoin_nearest(pts, P, how="left").drop_duplicates("id").set_index("id")["pipe_id"]
    in_catchment = {str(k) for _, _, k in sub.graph.edges(keys=True)}
    keep = b[b["id"].map(near).astype(str).isin(in_catchment)].copy()
    keep["geometry"] = keep.geometry.simplify(1.0)
    gpkg = DATA / "network.gpkg"
    keep[["id", "population", "parent", "geometry"]].to_file(gpkg, layer="blocks", driver="GPKG")
    print(f"network.gpkg blocks: {len(keep):,} blocks, {keep.population.sum():,.0f} people")


def tex_to_text(v: str) -> str:
    """A macro's body as plain slide text: thousands separators, thin spaces, en dashes, zone numbers."""
    v = v.replace("{,}", ",").replace("\\,", "\u2009").replace("--", "\u2013").replace("~", "\u00a0")
    # Zone IDs as the talk writes them: the paper's "Z09" becomes "09", and the slide says "zone 09".
    v = re.sub(r"\bZ(\d\d)\b", r"\1", v)
    return v.strip()


def write_variables() -> int:
    tex = (OUT / "paper" / "generated_values.tex").read_text()
    pairs = re.findall(r"\\newcommand\{\\(\w+)\}\{(.*)\}\s*$", tex, re.M)
    lines = ["# Generated by scripts/sync_from_paper.py from paper/generated_values.tex; do not edit."]
    for name, body in pairs:
        text = tex_to_text(body).replace("\\", "\\\\").replace('"', '\\"')
        lines.append(f'{name}: "{text}"')
    (ROOT / "_variables.yml").write_text("\n".join(lines) + "\n")
    return len(pairs)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--paper", type=Path, default=Path.home() / "JRN-059")
    ap.add_argument("--allow-dirty", action="store_true", help="copy from a tree with uncommitted changes")
    ap.add_argument("--no-network", action="store_true", help="skip the map export (data/network.gpkg)")
    ap.add_argument("--blocks-only", action="store_true",
                    help="replace only the census blocks layer of data/network.gpkg; copy nothing else")
    a = ap.parse_args()
    paper = a.paper.expanduser().resolve()
    if a.blocks_only:
        export_network(paper, yaml_outlet(), blocks_only=True)
        return

    commit = git(paper, "rev-parse", "HEAD")
    describe = git(paper, "describe", "--tags", "--always", "--dirty")
    dirty = bool(git(paper, "status", "--porcelain", "--untracked-files=no"))
    if dirty and not a.allow_dirty:
        sys.exit(f"{paper} has uncommitted changes; commit (or tag) first, or pass --allow-dirty for a draft copy")

    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    copied = copy_manifest(paper)
    import yaml
    const = constants(paper)
    (OUT / "constants.yaml").write_text(yaml.safe_dump(const, sort_keys=False))
    n_vars = write_variables()
    if not a.no_network:
        export_network(paper, const["outlet"])

    (OUT / "SOURCE.txt").write_text(
        f"paper repository: {paper.name}\n"
        f"commit: {commit}\n"
        f"describe: {describe}\n"
        f"clean: {'no (draft copy, uncommitted changes)' if dirty else 'yes'}\n"
        f"copied: {dt.datetime.now().isoformat(timespec='seconds')}\n"
        f"files: {len(copied)}\n")
    print(f"copied {len(copied)} files from {describe}{' (DIRTY)' if dirty else ''}; {n_vars} variables")


if __name__ == "__main__":
    main()

# BIRS 2026: Where Does the Signal Come From?

*High-Resolution Sewer Network Modelling Toward Wastewater-Based Epidemiology*, by Christopher T.
DeGroot (with Anthony P. R. Sherratt), Department of Mechanical and Materials Engineering, Western
University, at the BIRS workshop *Advances in Statistical Methodology for the Analysis of Wastewater
Data*. The slides are a [Quarto](https://quarto.org) reveal.js deck on the group's
`presentation-template`, in the assertion-evidence style. Press `S` in the deck for the speaker
notes. The talk walks through the backing paper (repository `JRN-059`), step by step, for a mixed
audience.

## Build the slides

The figures and `_variables.yml` are committed, so the deck needs only Quarto (built with 1.10):

```bash
quarto preview slides.qmd      # or: quarto render slides.qmd
```

`_quarto.yml` exists only so that `{{< var Name >}}` reads `_variables.yml`.

## Where the numbers and figures come from

Every number on a slide is a `{{< var Name >}}` shortcode, a macro of the paper's
`paper/generated_values.tex`. Every figure is drawn by `scripts/talk_figures.py` from the files in
`data/`, which are copies of the paper's result files.

| step | command | needs |
|---|---|---|
| 1. copy the paper's results | `~/JRN-059/.venv/bin/python scripts/sync_from_paper.py` | the paper repository and its environment (swmmbuilder, for the map layers) |
| 2. draw the figures | `python scripts/talk_figures.py [names]` | `data/` only; `requirements.txt` |
| 3. render | `quarto render slides.qmd` | Quarto |

Step 1 writes:

- `data/paper/`: verbatim copies of the paper's result files, at their paths in the paper repository;
- `data/paper/constants.yaml`: the few constants the figures take from the paper's code;
- `data/paper/SOURCE.txt`: the paper commit copied, and whether its tree was clean;
- `data/network.gpkg`: the catchment's pipes, nodes, and pumping stations, and the basemap clipped to
  it (City of London open data);
- `_variables.yml`: the paper's generated values.

It refuses a paper tree with uncommitted changes unless `--allow-dirty` is given, and `SOURCE.txt`
then says so. Copy from a tagged, clean commit before the talk.

## Data licences

The network and basemap layers are City of London open data (Open Government Licence – London). The
census-derived populations are Statistics Canada data. The noise summaries in
`data/paper/scenarios/odm/` and `scenarios/review/` come from the group's monitoring composites.
Decide whether they can be public before this repository is.

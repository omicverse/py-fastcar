# pyfastcar

A **pure-Python re-implementation of [FastCAR](https://github.com/LungCellAtlas/FastCAR)** (Gocht / Berg et al., *BMC Genomics* 2023) for fast, deterministic per-gene correction of ambient RNA in droplet-based single-cell RNA-seq data.

- AnnData-native — drop-in for the scanpy ecosystem
- **No `rpy2`**, no R install — the empty-droplet detection, per-gene `gMax`/`frC` profiling, and count subtraction are all implemented directly in NumPy/SciPy
- Same function surface as the R workflow (`determine.background.to.remove` → `remove.background`, plus `describe.ambient.RNA.sequence` / `recommend.empty.cutoff`)
- Bit-for-bit reproducibility against the R reference — FastCAR is fully deterministic (no RNG anywhere), so `pyfastcar` matches R FastCAR exactly (see `tests/test_exact_match.py`)

> This is a **standalone mirror** of the canonical implementation that lives in [`omicverse`](https://github.com/Starlitnightly/omicverse). All algorithmic work is developed upstream in omicverse and synced here for users who want FastCAR without the full omicverse stack.

## Install

```bash
pip install pyfastcar
```

Dependencies: `numpy`, `scipy`, `pandas`, `anndata`.

## What it does

Droplet-based scRNA-seq libraries are contaminated by **ambient RNA** — transcripts released by lysed cells that end up in every droplet. FastCAR estimates this contamination from the **empty droplets** of the raw, unfiltered count matrix and subtracts it, gene by gene, from the real cells.

The algorithm is fully deterministic — a handful of vectorised operations on the raw count matrix:

1. **Empty droplets** are the libraries whose total UMI count is below an `empty_droplet_cutoff` (`thE`, default 100). All-zero "unused barcodes" are excluded from the empty-droplet population.
2. For each gene, compute `gMax` — the highest count of that gene in any single empty droplet — and `frC` — the fraction of (non-zero) empty droplets that contain the gene.
3. A gene is corrected only when its `frC` clears the allowable contamination fraction `contamination_chance_cutoff` (`frAA`, default 0.005). For those genes, `gMax` is subtracted from every cell's count, with negative results floored at zero.

## Quick-start (AnnData API)

```python
import anndata as ad
from pyfastcar import correct_anndata

# raw = unfiltered AnnData (cells × genes): real cells + empty droplets
raw   = ad.read_h5ad("raw_feature_bc_matrix.h5ad")
cells = ad.read_h5ad("filtered_feature_bc_matrix.h5ad")   # the real cells

corrected = correct_anndata(
    raw, cells,
    empty_droplet_cutoff=100, contamination_chance_cutoff=0.005,
)

corrected.uns["fastcar"]["ambient_rna_profile"]   # per-gene subtract amounts
corrected.uns["fastcar"]["diagnostics"]           # gMax / frC table
corrected.uns["fastcar"]["n_genes_corrected"]
```

The wrapper transposes internally, so you always work in the natural cells × genes AnnData orientation.

## Low-level functional API (mirrors R one-to-one)

```python
from pyfastcar import (
    determine_background_to_remove, remove_background,
    describe_ambient_rna_sequence, recommend_empty_cutoff,
)

# full = unfiltered genes × droplets matrix (real cells + empty droplets)
# cells = filtered genes × cells matrix of real cells
profile = determine_background_to_remove(
    full, cells, empty_droplet_cutoff=100, contamination_chance_cutoff=0.005)

corrected = remove_background(cells, profile)        # ambient-corrected matrix

# pass return_table=True for a per-gene diagnostic DataFrame:
profile, table = determine_background_to_remove(full, cells, return_table=True)
```

### Choosing the empty-droplet cutoff

```python
desc = describe_ambient_rna_sequence(
    full, start=50, stop=500, by=25, contamination_chance_cutoff=0.005)
cutoff = recommend_empty_cutoff(desc)
```

## What's included

| Python | R counterpart | Purpose |
|---|---|---|
| `determine_background_to_remove` | `determine.background.to.remove` | per-gene ambient-RNA profile (`gMax`/`frC`) |
| `remove_background` | `remove.background` | subtract the profile, floor at zero |
| `correct_anndata` | *(new)* | AnnData-native one-shot wrapper |
| `describe_ambient_rna_sequence` | `describe.ambient.RNA.sequence` | empty-droplet cutoff profiling |
| `recommend_empty_cutoff` | `recommend.empty.cutoff` | suggest a cutoff from the profile |
| `read_cell_matrix` / `read_full_matrix` | `read.cell.matrix` / `read.full.matrix` | 10x CellRanger loaders |

## Reproducing R results exactly

FastCAR has no stochastic component — given the same raw matrix and the same
`empty_droplet_cutoff` / `contamination_chance_cutoff`, the output is fully
determined. `pyfastcar` therefore reproduces R FastCAR **bit-for-bit**: the
per-gene subtraction amounts, the corrected integer matrix and the
threshold-profiling table are all identical.

`tests/test_exact_match.py` runs the R reference (`FastCAR::determine.background.to.remove`
+ `remove.background` + `describe.ambient.RNA.sequence`) inside the `CMAP`
environment on a deterministic synthetic raw matrix, saves every intermediate,
and checks that the Python port reproduces them element-for-element.

```bash
pip install -e ".[dev]"
pytest tests/ -q              # smoke tests + R-parity tests
pytest tests/test_smoke.py -q # smoke tests only (no R required)
```

`examples/compare_R_vs_Python.ipynb` runs both implementations on a real raw
10x dataset (`data/pbmc1k_raw.h5ad`, a subset of 10x Genomics' `pbmc_1k_v3`
raw feature-barcode matrix that still contains empty droplets) and visualizes
the bit-exact agreement with omicverse.

## Relationship to omicverse

Developed **upstream** in [`omicverse`](https://github.com/Starlitnightly/omicverse):

- Canonical implementation: lives in the omicverse single-cell preprocessing stack
- Standalone mirror (this repo): same code, same API, minus the omicverse packaging

## Citation

If you use this package, please cite the original FastCAR paper:

> Gocht A.M., Berg M., *et al.* **FastCAR: fast correction for ambient RNA to facilitate differential gene expression analysis in single-cell RNA-sequencing datasets.** *BMC Genomics* 24, 2023.

and acknowledge omicverse / this repo for the Python port.

## License

Apache-2.0. The upstream R package FastCAR is GPL-3; this is an independent
re-implementation of its published algorithm.

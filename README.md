# py-fastcar

Pure-Python port of the R package
**[FastCAR](https://github.com/LungCellAtlas/FastCAR)** — **Fast** **C**orrection
of **A**mbient **R**NA (Gocht / Berg *et al.*, *BMC Genomics* 2023): a fast,
deterministic per-gene correction for ambient RNA in droplet-based single-cell
RNA-sequencing data.

`pyfastcar` is a standalone, dependency-light re-implementation that does
**not** require R or `rpy2`. It is **bit-exact** with R FastCAR 0.1.0.

| | |
|---|---|
| PyPI / import name | `pyfastcar` |
| Repository | `omicverse/py-fastcar` |
| License | Apache-2.0 |
| Upstream | FastCAR 0.1.0 (GPL-3, R) |
| Numerical parity | bit-exact vs R FastCAR (fully deterministic) |

## Install

```bash
pip install pyfastcar              # once published
# or, from a checkout:
pip install -e .
```

Dependencies: `numpy`, `scipy`, `pandas`, `anndata`.

## What it does

Droplet-based scRNA-seq libraries are contaminated by **ambient RNA** —
transcripts released by lysed cells that end up in every droplet. FastCAR
estimates this contamination from the **empty droplets** and subtracts it,
gene by gene, from the real cells.

The algorithm is fully deterministic — a handful of vectorised operations
on the raw count matrix:

1. **Empty droplets** are the libraries whose total UMI count is below an
   `empty_droplet_cutoff` (`thE`, default 100). All-zero "unused barcodes"
   are excluded from the empty-droplet population.
2. For each gene, compute
   * `gMax` — the highest count of that gene in any single empty droplet, and
   * `frC` — the fraction of (non-zero) empty droplets that contain the gene.
3. A gene is corrected only when its `frC` clears the allowable
   contamination fraction `contamination_chance_cutoff` (`frAA`, default
   0.005). For those genes, **`gMax` is subtracted from every cell's count**,
   with negative results floored at zero.

## Quick start

### scipy / pandas matrices (FastCAR's native genes × droplets orientation)

```python
import pyfastcar as fc

# full = unfiltered genes x droplets matrix (real cells + empty droplets)
# cells = filtered genes x cells matrix of real cells
profile = fc.determine_background_to_remove(
    full, cells, empty_droplet_cutoff=100, contamination_chance_cutoff=0.005)

corrected = fc.remove_background(cells, profile)   # ambient-corrected matrix
```

`determine_background_to_remove(..., return_table=True)` also returns a
per-gene diagnostic `DataFrame` with the `gMax`, `frC`, `occurrences` and
`selected` columns.

### AnnData (cells × genes — the scanpy / omicverse convention)

```python
import pyfastcar as fc

# full_adata: unfiltered AnnData (cells x genes), cell_adata: filtered cells
corrected = fc.correct_anndata(
    full_adata, cell_adata,
    empty_droplet_cutoff=100, contamination_chance_cutoff=0.005)

corrected.uns["fastcar"]["ambient_rna_profile"]   # per-gene subtract amounts
corrected.uns["fastcar"]["diagnostics"]           # gMax / frC table
corrected.uns["fastcar"]["n_genes_corrected"]
```

The wrapper transposes internally, so you always work in the natural
cells × genes AnnData orientation.

### Choosing the empty-droplet cutoff

```python
desc = fc.describe_ambient_rna_sequence(
    full, start=50, stop=500, by=25, contamination_chance_cutoff=0.005)
# DataFrame indexed by cutoff with nEmptyDroplets / genesInBackground /
# genesContaminating columns.

cutoff = fc.recommend_empty_cutoff(desc)
```

### Loading 10x CellRanger output

```python
full  = fc.read_full_matrix("raw_feature_bc_matrix/")       # genes x droplets
cells = fc.read_cell_matrix("filtered_feature_bc_matrix/")
# or, as AnnData:
adata = fc.read_full_matrix("raw_feature_bc_matrix/", as_anndata=True)
```

## API

| Function | R FastCAR equivalent |
|---|---|
| `determine_background_to_remove` | `determine.background.to.remove` |
| `remove_background` | `remove.background` |
| `describe_ambient_rna_sequence` | `describe.ambient.RNA.sequence` |
| `recommend_empty_cutoff` | `recommend.empty.cutoff` |
| `read_cell_matrix` / `read_full_matrix` | `read.cell.matrix` / `read.full.matrix` |
| `correct_anndata` | *(new — AnnData convenience wrapper)* |

## R-parity testing

FastCAR is fully deterministic, so the port is verified to be **bit-exact**
against R FastCAR. `tests/r_reference_driver.R` generates a deterministic
synthetic raw matrix (real cells + empty droplets + unused barcodes), runs
it through R `FastCAR`, and the Python suite asserts the per-gene
`gMax`/`frC`, the selected genes, the corrected integer matrix and the
threshold-profiling table are identical.

```bash
pip install -e ".[dev]"
pytest tests/ -q              # smoke tests + R-parity tests
pytest tests/test_smoke.py -q # smoke tests only (no R required)
```

## Citation

If you use `pyfastcar`, please cite the original FastCAR paper:

> Gocht A.M., Berg M., *et al.* FastCAR: fast correction for ambient RNA to
> facilitate differential gene expression analysis in single-cell
> RNA-sequencing datasets. *BMC Genomics* 24, 2023.

## License

Apache-2.0. The upstream R package FastCAR is GPL-3; this is an independent
re-implementation of its published algorithm.

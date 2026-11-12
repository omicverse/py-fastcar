# Benchmark: pyfastcar vs R FastCAR 0.1.0

Five 10x Genomics human–mouse mixtures (hgmm 1k/5k/10k v3, 6k/12k v2). The tasks are those of the
py-soupx benchmark (`py-soupx/report/scripts/01_prepare.py`, copied here): the raw droplet matrix,
Cell Ranger's cells and species labels. Counts of the other species in a cell can only be ambient RNA.

```
python  scripts/01_prepare.py <dir with extracted 10x archives> <tasks_dir>
python  scripts/run_all.py <tasks_dir> <runs_dir> <Rscript> <python> 10 1,2    # idle cores, 1 thread each
python  scripts/05_figures.py <runs_dir> <tasks_dir> figures
```

Both run two configurations: a fixed empty-droplet cutoff of 100 UMI, and the cutoff recommended by
`describe.ambient.RNA.sequence(10, 500, 10)` + `recommend.empty.cutoff`; contamination chance cutoff
0.05 in both, as in the FastCAR README. R: FastCAR 0.1.0 (GitHub) with qlcMatrix on R 4.5.3. Times are
the faster of two runs, each on an idle core with one BLAS thread. `figures/comparison_per_dataset.tsv`
holds the per-dataset numbers.

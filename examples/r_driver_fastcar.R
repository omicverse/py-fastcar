#!/usr/bin/env Rscript
# Run the original R FastCAR end-to-end on a raw 10x gene x droplet count
# matrix (genes x droplets TSV, the same format the Python side dumps via
# pandas.to_csv). Outputs: the per-gene ambient-RNA profile, the corrected
# cell matrix, the threshold-profiling table and a small JSON of parameters
# -- everything the notebook needs to overlay on the Python-computed results.
#
# FastCAR needs the *raw, unfiltered* matrix (real cells + empty droplets)
# to estimate ambient RNA, plus the filtered cell matrix it then corrects.
#
# Usage:
#   Rscript r_driver_fastcar.R <full_tsv> <cell_tsv> <outdir> \
#                              <emptyDropletCutoff> <contaminationChanceCutoff>

suppressMessages({
  library(Matrix)
  library(qlcMatrix)   # FastCAR uses qlcMatrix::rowMax internally
  library(FastCAR)
  library(jsonlite)
})

args <- commandArgs(trailingOnly = TRUE)
full_path <- args[[1]]
cell_path <- args[[2]]
outdir    <- args[[3]]
emptyCutoff  <- if (length(args) >= 4) as.integer(args[[4]]) else 100L
contamCutoff <- if (length(args) >= 5) as.numeric(args[[5]]) else 0.005

dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

cat(sprintf("[R] reading full matrix %s\n", full_path))
fullDense <- as.matrix(read.table(full_path, sep = "\t", header = TRUE,
                                  row.names = 1, check.names = FALSE))
cat(sprintf("[R] reading cell matrix %s\n", cell_path))
cellDense <- as.matrix(read.table(cell_path, sep = "\t", header = TRUE,
                                  row.names = 1, check.names = FALSE))

fullSparse <- as(as(fullDense, "CsparseMatrix"), "dgCMatrix")
cellSparse <- as(as(cellDense, "CsparseMatrix"), "dgCMatrix")
cat(sprintf("[R] full = %d x %d, cells = %d x %d\n",
            nrow(fullSparse), ncol(fullSparse),
            nrow(cellSparse), ncol(cellSparse)))

# ---- run FastCAR --------------------------------------------------------
cat("[R] determine.background.to.remove()\n")
ambProfile <- determine.background.to.remove(
  fullSparse, cellSparse, emptyCutoff, contamCutoff)

cat("[R] remove.background()\n")
corrected <- remove.background(cellSparse, ambProfile)

cat("[R] describe.ambient.RNA.sequence()\n")
ambDesc <- describe.ambient.RNA.sequence(
  fullSparse, start = 50, stop = 500, by = 25,
  contaminationChanceCutoff = contamCutoff)
recCutoff <- recommend.empty.cutoff(ambDesc)

# ---- write outputs ------------------------------------------------------
# per-gene ambient profile -- named numeric vector over every gene.
write.table(
  data.frame(gene = names(ambProfile), profile = as.numeric(ambProfile)),
  file.path(outdir, "ambient_profile.tsv"),
  sep = "\t", quote = FALSE, row.names = FALSE)

# corrected matrix (dense integer, genes x cells).
write.table(as.matrix(corrected),
            file.path(outdir, "corrected_matrix.tsv"),
            sep = "\t", quote = FALSE, col.names = NA)

# threshold-profiling table.
write.table(
  data.frame(emptyDropletCutoff = as.integer(rownames(ambDesc)), ambDesc),
  file.path(outdir, "ambient_description.tsv"),
  sep = "\t", quote = FALSE, row.names = FALSE)

# scalar parameters / results.
write(toJSON(list(emptyDropletCutoff = emptyCutoff,
                  contaminationChanceCutoff = contamCutoff,
                  recommendedCutoff = recCutoff,
                  nGenesCorrected = sum(ambProfile > 0),
                  nGenes = length(ambProfile),
                  nCells = ncol(cellSparse)),
             auto_unbox = TRUE),
      file = file.path(outdir, "meta.json"))

cat(sprintf("[R] wrote outputs to %s (genes corrected = %d, rec. cutoff = %d)\n",
            outdir, sum(ambProfile > 0), recCutoff))

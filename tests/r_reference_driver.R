#!/usr/bin/env Rscript
# ---------------------------------------------------------------------------
# R-parity reference driver for pyfastcar.
#
# Generates a deterministic synthetic raw gene x droplet matrix (real cells
# plus empty droplets), runs it through R FastCAR, and writes the inputs and
# outputs as plain TSV so the Python test suite can assert bit-exact parity.
#
# Usage:  Rscript r_reference_driver.R <out_dir>
# ---------------------------------------------------------------------------
suppressMessages({
  library(Matrix)
  library(qlcMatrix)   # FastCAR uses qlcMatrix::rowMax internally
  library(FastCAR)
})

args <- commandArgs(trailingOnly = TRUE)
out_dir <- if (length(args) >= 1) args[1] else "R_out"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# --------------------------------------------------------------------------
# 1. synthetic raw matrix -- fully deterministic
# --------------------------------------------------------------------------
set.seed(20240521)

nGenes        <- 60
nCells        <- 80     # real cells: high UMI counts
nEmpty        <- 400    # empty droplets: low UMI counts
nUnused       <- 40     # unused barcodes: all-zero columns
emptyCutoff   <- 100
contamCutoff  <- 0.005

geneNames <- sprintf("GENE%03d", seq_len(nGenes))

# real cells -- moderate Poisson counts so column sums far exceed the cutoff.
cellCounts <- matrix(rpois(nGenes * nCells, lambda = 6),
                     nrow = nGenes, ncol = nCells)

# empty droplets -- sparse, low counts; a block of "ambient" genes appear in
# many empty droplets (crossing the contamination threshold) while the rest
# only appear in a small handful (staying below it) -- giving a clear split.
emptyCounts <- matrix(0L, nrow = nGenes, ncol = nEmpty)
ambientGenes  <- 1:12       # heavily contaminating
rareGenes     <- 13:60      # rarely contaminating
for (j in seq_len(nEmpty)) {
  # background genes contaminate most empty droplets at low level.
  hit <- ambientGenes[runif(length(ambientGenes)) < 0.6]
  if (length(hit) > 0) {
    emptyCounts[hit, j] <- rpois(length(hit), lambda = 1.2) + 1L
  }
  # rare genes get an occasional read -- low enough to stay sub-threshold.
  nExtra <- rpois(1, lambda = 0.05)
  if (nExtra > 0) {
    extra <- sample(rareGenes, size = min(nExtra, length(rareGenes)))
    emptyCounts[extra, j] <- emptyCounts[extra, j] + 1L
  }
}
# make sure every empty droplet stays below the cutoff.
emptyCounts[emptyCounts > 8L] <- 8L

unusedCounts <- matrix(0L, nrow = nGenes, ncol = nUnused)

fullDense <- cbind(cellCounts, emptyCounts, unusedCounts)
rownames(fullDense) <- geneNames
colnames(fullDense) <- c(
  sprintf("cell%03d",   seq_len(nCells)),
  sprintf("empty%04d",  seq_len(nEmpty)),
  sprintf("unused%03d", seq_len(nUnused)))

# the "cell matrix" is just the real-cell columns.
cellDense <- fullDense[, seq_len(nCells), drop = FALSE]

fullSparse <- as(as(fullDense, "CsparseMatrix"), "dgCMatrix")
cellSparse <- as(as(cellDense, "CsparseMatrix"), "dgCMatrix")

# --------------------------------------------------------------------------
# 2. run FastCAR
# --------------------------------------------------------------------------
ambProfile <- determine.background.to.remove(
  fullSparse, cellSparse, emptyCutoff, contamCutoff)

corrected <- remove.background(cellSparse, ambProfile)

# threshold profiling helper
ambDesc <- describe.ambient.RNA.sequence(
  fullSparse, start = 50, stop = 200, by = 25,
  contaminationChanceCutoff = contamCutoff)

recCutoff <- recommend.empty.cutoff(ambDesc)

# --------------------------------------------------------------------------
# 3. write everything out as plain TSV
# --------------------------------------------------------------------------
write.table(as.matrix(fullDense), file.path(out_dir, "full_matrix.tsv"),
            sep = "\t", quote = FALSE, col.names = NA)
write.table(as.matrix(cellDense), file.path(out_dir, "cell_matrix.tsv"),
            sep = "\t", quote = FALSE, col.names = NA)

# ambient profile -- named numeric vector over every gene.
write.table(
  data.frame(gene = names(ambProfile), profile = as.numeric(ambProfile)),
  file.path(out_dir, "ambient_profile.tsv"),
  sep = "\t", quote = FALSE, row.names = FALSE)

# corrected matrix (dense integer).
write.table(as.matrix(corrected), file.path(out_dir, "corrected_matrix.tsv"),
            sep = "\t", quote = FALSE, col.names = NA)

# threshold-profiling table.
write.table(
  data.frame(emptyDropletCutoff = as.integer(rownames(ambDesc)), ambDesc),
  file.path(out_dir, "ambient_description.tsv"),
  sep = "\t", quote = FALSE, row.names = FALSE)

# scalar parameters / results.
params <- data.frame(
  key = c("emptyDropletCutoff", "contaminationChanceCutoff",
          "recommendedCutoff", "nGenesCorrected"),
  value = c(emptyCutoff, contamCutoff, recCutoff,
            sum(ambProfile > 0)))
write.table(params, file.path(out_dir, "params.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE)

cat("R reference written to", out_dir, "\n")

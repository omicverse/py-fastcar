# Reference run: FastCAR 0.1.0 on one task, the README workflow with fixed cut-offs (100 UMIs, 0.05) and with the
# recommended empty-droplet cut-off (describe.ambient.RNA.sequence 10-500 by 10, recommend.empty.cutoff).
# Usage: Rscript 02_run_fastcar.R <task_dir> <out_dir>
suppressMessages({library(Matrix); library(qlcMatrix); library(FastCAR); library(jsonlite)})   # rowMax comes from qlcMatrix
args <- commandArgs(trailingOnly = TRUE)
td <- args[1]; od <- args[2]
dir.create(od, recursive = TRUE, showWarnings = FALSE)
tod <- as(readMM(gzfile(file.path(td, "tod.mtx.gz"))), "CsparseMatrix")
genes <- read.delim(file.path(td, "genes.tsv"))
rownames(tod) <- genes$id; colnames(tod) <- readLines(file.path(td, "droplets.tsv"))
cells <- read.delim(file.path(td, "clusters.tsv"), colClasses = "character")$cell
toc <- tod[, cells]
dump <- function(m, prefix) {
  m <- as(m, "CsparseMatrix")
  writeBin(m@p, paste0(prefix, "_p.bin"), size = 4); writeBin(m@i, paste0(prefix, "_i.bin"), size = 4)
  writeBin(m@x, paste0(prefix, "_x.bin"), size = 8)
}
res <- list(engine = "FastCAR", version = as.character(packageVersion("FastCAR")), R = R.version.string)
t0 <- proc.time()[["elapsed"]]
amb <- determine.background.to.remove(tod, toc, 100, 0.05)
fixed <- remove.background(toc, amb)
res$seconds_fixed <- proc.time()[["elapsed"]] - t0
writeBin(as.numeric(amb), file.path(od, "profile_fixed.bin"), size = 8); dump(fixed, file.path(od, "fixed"))
t0 <- proc.time()[["elapsed"]]
desc <- describe.ambient.RNA.sequence(tod, 10, 500, 10, 0.05)
cutoff <- recommend.empty.cutoff(desc)
amb2 <- determine.background.to.remove(tod, toc, cutoff, 0.05)
rec <- remove.background(toc, amb2)
res$seconds_recommended <- proc.time()[["elapsed"]] - t0
writeBin(as.numeric(amb2), file.path(od, "profile_recommended.bin"), size = 8); dump(rec, file.path(od, "recommended"))
write.table(desc, file.path(od, "describe.tsv"), sep = "\t", quote = FALSE)
res$cutoff <- cutoff; res$genes_fixed <- sum(amb > 0); res$genes_recommended <- sum(amb2 > 0); res$dim <- dim(toc)
res$seconds <- res$seconds_fixed
write_json(res, file.path(od, "result.json"), auto_unbox = TRUE, digits = NA)
cat(basename(td), sprintf("cutoff=%d genes=%d/%d fixed %.1fs recommended %.1fs", cutoff, res$genes_fixed, res$genes_recommended,
                          res$seconds_fixed, res$seconds_recommended), "\n")

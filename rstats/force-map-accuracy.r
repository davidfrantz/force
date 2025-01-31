#!/usr/bin/env Rscript

# This file is part of FORCE - Framework for Operational Radiometric 
# Correction for Environmental monitoring.
# 
# Copyright (C) 2013-2024 David Frantz
# 
# FORCE is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# FORCE is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with FORCE.  If not, see <http://www.gnu.org/licenses/>.

info <- "Compute map accuracy and area statistics"


# program name ######################################################
library(getopt)
progpath <<- get_Rscript_filename()
progname <<- basename(progpath)
progdir  <<- dirname(progpath)


# shared R functions ################################################
source(sprintf("%s/force-misc/force-rstats-library.r", progdir))


# more R libraries ##################################################
silent_library("dplyr")
silent_library("sf")


# input #############################################################

# usage function in same style as other FORCE tools,
# do not use getop's built-in usage function for consistency
usage <- function(exit){

  message <- c(
    sprintf(
      "Usage: %s [-h] [-v] [-i] [-o output-file] \n",
      progname
    ),
    "       -c count-file -s sample-file -a pixel-area\n",
    "\n",
    "  -h  = show this help\n",
    "  -v  = show version\n",
    "  -i  = show program's purpose\n",
    "\n",
    "  -o output-file  = output file path with extension,\n",
    "     defaults to './accuracy-assessment.txt'\n",
    "  -c count-file  = csv table with pixel counts per class\n",
    "     2 columns named class and count\n",
    "  -s sample-file = vector file (or csv table) with predicted and reference class labels\n",
    "     2 columns named label_map and label_reference\n",
    "  -a pixel-area  = area of one pixel in desired reporting unit, e.g.\n",
    "      100 for a Sentinel-2 based map to be reported in m², or\n",
    "     0.01 for a Sentinel-2 based map to be reported in ha\n",
    "\n"
  )

  cat(
    message,
    file = if (exit == 0) stdout else stderr()
  )

  quit(
    save = "no",
    status = exit
  )

}


spec <- matrix(
  c(
    "help",       "h", 0, "logical",
    "version",    "v", 0, "logical",
    "info",       "i", 0, "logical",
    "output",     "o", 2, "character",
    "counts",     "c", 1, "character",
    "sample",     "s", 1, "character",
    "pixel_area", "a", 1, "numeric"
  ),
  byrow = TRUE,
  ncol = 4
)

opt <- getopt(spec)

if (!is.null(opt$help)) usage()
if (!is.null(opt$info)) exit_normal(info)
if (!is.null(opt$version)) print_version(progdir)

if (is.null(opt$counts)) exit_with_error("count-file is missing!")
if (is.null(opt$sample)) exit_with_error("sample-file is missing!")

file_existing(opt$counts)
file_existing(opt$sample)

if (is.null(opt$output)) opt$output <- "accuracy-assessment.txt"


# read data
count <- read.csv(opt$counts) %>%
  mutate_all(as.integer)
sample <- read_sf(opt$sample) %>%
  mutate_all(as.integer)


# columns OK?
if (!"label_map" %in% colnames(sample)) exit_with_error("label_map column in sample-file missing")
if (!"label_reference" %in% colnames(sample)) exit_with_error("label_reference column in sample-file missing")
if (!"class" %in% colnames(count)) exit_with_error("class column in count-file missing")
if (!"count" %in% colnames(count)) exit_with_error("count column in count-file missing")


# main thing ########################################################

# get all classes
classes <-
  c(
    sample$label_map,
    sample$label_reference
  ) %>%
  unique() %>%
  sort()
n_classes <- length(classes)


# initialize confusion matrix
confusion_counts <-
  matrix(
    NA,
    n_classes,
    n_classes,
    dimnames = list(
      map = classes,
      reference = classes
    )
  )

# populate confusion matrix
for (m in 1:n_classes){
  for (r in 1:n_classes){

    confusion_counts[m, r] <-
      sample %>%
      filter(
        label_map == classes[m] &
        label_reference == classes[r]
      ) %>%
      nrow()

  }
}

if (any(rowSums(confusion_counts) < 2)) {
  exit_with_error("at least one map class has less than two samples")
}


acc_metrics <- function(confusion_matrix) {

  sum_all       <- sum(confusion_matrix)
  sum_map_class <- rowSums(confusion_matrix)
  sum_ref_class <- colSums(confusion_matrix)

  # overall accuracy
  oa <- 
    confusion_matrix %>%
    diag() %>%
    sum() %>%
    `/`(sum_all)

  # producer's accuracy
  pa <- 
    confusion_matrix %>%
    diag() %>%
    `/`(sum_ref_class)

  # user's accuracy
  ua <- 
    confusion_matrix %>%
    diag() %>%
    `/`(sum_map_class)

  # error of omission and commission
  oe <- 1 - pa
  ce <- 1 - ua


  list(
    oa = oa,
    pa = pa,
    ua = ua,
    oe = oe,
    ce = ce
  ) %>%
  return()

}


# compute propoertional area per class, area in reporting unit, and sort the classes
count <- 
  count %>% 
  arrange(class) %>%
  mutate(weight = count / sum(count)) %>%
  mutate(area = count * opt$pixel_area)

# classes should now align with map/reference dataset
if (!any(count$class == classes))
  exit_with_error("classes in file-count do not match with map or reference classes")

# Olofsson et al. 2013, eq. 1
confusion_adjusted <-
  confusion_counts /
  matrix(
    rowSums(confusion_counts),
    n_classes,
    n_classes,
    byrow = FALSE
  ) *
  matrix(
    count$weight,
    n_classes,
    n_classes,
    byrow = FALSE
  )



# Olofsson et al. 2013, eq. 2
area_adjusted <-
  colSums(confusion_adjusted) * sum(count$area)



proportions <- 
  confusion_counts / 
  matrix(
    rowSums(confusion_counts),
    n_classes,
    n_classes,
    byrow = FALSE
  )


# Olofsson et al. 2013, eq. 3-5
confidence_area_adjusted <-
{
proportions * 
(1 - proportions) / 
matrix(
  rowSums(confusion_counts) - 1,
  n_classes,
  n_classes,
  byrow = FALSE
) *
matrix(
  count$weight**2,
  n_classes,
  n_classes,
  byrow = FALSE
)
 } %>%
  colSums() %>%
  sqrt() %>%
  `*`(sum(count$area)) %>%
  `*`(1.96)


# Olofsson et al. 2013, eq. 6-8
acc_traditional <- acc_metrics(confusion_counts)
acc_adjusted    <- acc_metrics(confusion_adjusted)

# Olofsson et al. 2014, eq. 5
oa_se <- sum(count$weight**2 * acc_adjusted$ua * (1 - acc_adjusted$ua) / (rowSums(confusion_counts) - 1)) %>%
sqrt() %>%
`*`(1.96)

# Olofsson et al. 2014, eq. 6
ua_se <- (acc_adjusted$ua * (1 - acc_adjusted$ua) / (rowSums(confusion_counts) - 1)) %>%
sqrt() %>%
`*`(1.96)

# Olofsson et al. 2014, eq. 7


Nj <-
  {
  matrix(
    count$count,
    n_classes,
    n_classes,
    byrow = FALSE
  ) / 
  matrix(
    rowSums(confusion_counts),
    n_classes,
    n_classes,
    byrow = FALSE
  ) *
  confusion_counts
  } %>%
  colSums()


term1 <- count$count**2 *
(1 - acc_adjusted$pa)**2 * 
acc_adjusted$ua * 
(1 - acc_adjusted$ua) / 
(colSums(confusion_counts) - 1)

leave_class_out <- 
matrix(
  1,  n_classes,
    n_classes
)
diag(leave_class_out) <- 0

term2 <- {
matrix(
  count$count**2,
  n_classes,
  n_classes,
  byrow = FALSE
) *
confusion_counts /
matrix(
  rowSums(confusion_counts),
   n_classes,
    n_classes,
    byrow = FALSE
) *
(
  1 -
  confusion_counts /
  matrix(
    rowSums(confusion_counts),
    n_classes,
      n_classes,
      byrow = FALSE
  )
) /
matrix(
(rowSums(confusion_counts) - 1),
   n_classes,
    n_classes,
    byrow = FALSE
) *
leave_class_out
} %>%
colSums() %>%
`*`(acc_adjusted$pa**2)



pa_se <- 
{
  (1/Nj**2) *
  (term1 + term2)
} %>%
sqrt() %>%
`*`(1.96)




fo <- file(opt$output, "w")
sink(file = fo, append = TRUE, type = "output")

cat("Traditional Accuracy assessment\n")
cat("-----------------------------------------------------------------\n")
cat("\n")
cat("Traditional confusion matrix, expressed in terms of pixel counts:\n")
cat("\n")
print(confusion_counts)
cat("\n")
cat(sprintf("Overall Accuracy (OA): %.2f\n", acc_traditional$oa))
cat("\n")
print_stats <- cbind(
  sprintf("%.2f", acc_traditional$pa),
  sprintf("%.2f", acc_traditional$ua),
  sprintf("%.2f", acc_traditional$oe),
  sprintf("%.2f", acc_traditional$ce)
)
colnames(print_stats) <- c("Producer's Accuracy", "User's Accuracy", "Error of Omission", "Error of Commission")
rownames(print_stats) <- classes
print(print_stats, quote = FALSE)
cat("\n")
cat("\n")
cat("Area-Adjusted Accuracy\n")
cat("-----------------------------------------------------------------\n")
cat("\n")
cat("Confusion matrix, expressed in terms of proportion of area:\n")
cat("\n")
print(confusion_adjusted)
cat("\n")
cat(sprintf("Overall Accuracy (OA): %.2f \u00b1 %.2f\n", acc_adjusted$oa, oa_se))
cat("\n")
print_stats <- cbind(
  sprintf("%.2f \u00b1 %.2f", acc_adjusted$pa, pa_se),
  sprintf("%.2f \u00b1 %.2f", acc_adjusted$ua, ua_se),
  sprintf("%.2f \u00b1 %.2f", acc_adjusted$oe, pa_se),
  sprintf("%.2f \u00b1 %.2f", acc_adjusted$ce, ua_se)
)
colnames(print_stats) <- c("Producer's Accuracy", "User's Accuracy", "Error of Omission", "Error of Commission")
rownames(print_stats) <- classes
print(print_stats, quote = FALSE)
cat("\n")
print_area <- cbind(
  sprintf("%.2f \u00b1 %.2f", area_adjusted, confidence_area_adjusted),
  sprintf("%.2f", count$area)
)
colnames(print_area) <- c("Estimated Area", "Mapped Area")
rownames(print_area) <- classes
print(print_area, quote = FALSE)

sink(file = NULL)
close(fo)


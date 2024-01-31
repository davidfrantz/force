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

info <- "Compute map accuracy and area statistics.\n"

# load libraries ####################################################
library(dplyr)
library(getopt)


# input #############################################################

# usage function in same style as other FORCE tools,
# do not use getop's built-in usage function for consistency
usage <- function(exit){

  message <- c(
    sprintf(
      "Usage: %s [-h] [-v] [-i] [-o output-file] -c count-file \n", 
      get_Rscript_filename()
    ),
    "       -m map-file -r reference-file -a pixel-area\n",
    "\n",
    "  -h  = show this help\n",
    "  -v  = show version\n",
    "  -i  = show program's purpose\n",
    "\n",
    "  -o output-file  = output file path with extension,\n",
    "     defaults to './accuracy-assessment.txt'\n",
    "\n",
    "  -c count-file  = csv table with pixel counts per class\n",
    "     2 columns named class and count",
    "\n",
    "  -m map-file  = csv table with predicted class labels\n",
    "     2 columns named ID and map",
    "\n",
    "  -r reference-file  = csv table with reference class labels\n",
    "     2 columns named ID and reference",
    "\n",
    "  -a pixel-area  = area of one pixel in desired reporting unit, e.g.\n",
    "      100 for a Sentinel-2 based map to be reported in m², or\n",
    "     0.01 for a Sentinel-2 based map to be reported in ha\n",
    "\n",

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

exit_normal <- function(argument) {
  cat(
    sprintf("%s\n", argument)
  )
  quit(
    save = "no",
    status = 0
  )
}

exit_with_error <- function(argument) {
  cat(
    sprintf("%s\n", argument), 
    file = stderr()
  )
  usage(1)
}

file_existing <- function(path) {
  if (!file.exist(path)){
    cat(
      sprintf("file %s does not exist\n", path),
      file = stderr()
    )
    usage(1)
  }
}

spec <- matrix(
  c(
    "help",       "h", 0, "logical",
    "version",    "v", 0, "logical",
    "info",       "i", 0, "logical",
    "output",     "o", 2, "character",
    "counts",     "c", 1, "character",
    "map",        "m", 1, "character",
    "reference",  "r", 1, "character",
    "pixel_area", "a", 1, "numeric"
  ), 
  byrow = TRUE, 
  ncol = 4
)

opt <- getopt(spec)

if (!is.null(opt$help)) usage()
if (!is.null(opt$info)) exit_normal(info)
if (!is.null(opt$version)) exit_normal("Printing function not implemented yet. Sorry.\n")

if (is.null(opt$counts)) exit_with_error("count-file is missing!")
if (is.null(opt$map)) exit_with_error("map-file is missing!")
if (is.null(opt$reference)) exit_with_error("reference-file is missing!")
if (is.null(opt$area)) exit_with_error("pixel-area is missing!")

file_existing(opt$counts)
file_existing(opt$map)
file_existing(opt$reference)

if (is.null(opt$output)) opt$output <- "accuracy-assessment.txt"


cnt <- read.csv(opt$counts)
map <- read.csv(opt$map)
ref <- read.csv(opt$reference)

# columns OK?
if (!"ID" %in% colnames(map)) exit_with_error("ID column in map-file missing")
if (!"ID" %in% colnames(ref)) exit_with_error("ID column in reference-file missing")
if (!"map" %in% colnames(map)) exit_with_error("map column in map-file missing")
if (!"reference" %in% colnames(ref)) exit_with_error("reference column in reference-file missing")
if (!"class" %in% colnames(cnt)) exit_with_error("class column in count-file missing")
if (!"count" %in% colnames(cnt)) exit_with_error("count column in count-file missing")


# main thing ########################################################

# join input tables
table <- 
  reference %>% 
  inner_join(
    map,
    by = "ID"
  )

# join worked?
if (nrow(table) != nrow(reference)){
  exit_with_error("map and reference files could not be joined")
}

table <- data.frame(ID = 1:1000, map = round(runif(1000, 8, 12)), reference = round(runif(1000, 8, 12)))

# get all classes
classes <- 
  c(
    table$map, 
    table$reference
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
      table %>%
      filter(
        map == classes[m] & 
        reference == classes[r]
      ) %>%
      nrow()

  }
}

classes <- 1:4
n_classes <- 4
confusion_counts <- matrix(
  c(
    150,	12,	1,	2,
    32,	100,	21,	3,
    0,	32,	120,	0,
    0,	0,	5,	130
  ), byrow = TRUE,
  n_classes, 
    n_classes, 
    dimnames = list(
      map = classes, 
      reference = classes
    )
)

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



cnt <- data.frame(class = 1:4, count = c(20000,200000,300000,350000))
opt <- list(pixel_area = 30^2/10000, output = "accuracy-assessment.txt") # ha

# compute propoertional area per class, area in reporting unit, and sort the classes
cnt <- 
  cnt %>% 
  arrange(class) %>%
  mutate(weight = count / sum(count)) %>%
  mutate(area = count * opt$pixel_area)

# classes should now align with map/reference dataset
if (!any(cnt$class == classes))
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
    cnt$weight,
    n_classes,
    n_classes,
    byrow = FALSE
  )

confusion_counts
confusion_adjusted


# Olofsson et al. 2013, eq. 2
area_adjusted <-
  colSums(confusion_adjusted) * sum(cnt$area)



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
  cnt$weight**2,
  n_classes,
  n_classes,
  byrow = FALSE
)
 } %>%
  colSums() %>%
  sqrt() %>%
  `*`(sum(cnt$area)) %>%
  `*`(1.96)
confidence_area_adjusted


# Olofsson et al. 2013, eq. 6-8
acc_traditional <- acc_metrics(confusion_counts)
acc_adjusted    <- acc_metrics(confusion_adjusted)

# Olofsson et al. 2014, eq. 5
oa_se <- sum(cnt$weight**2 * acc_adjusted$ua * (1 - acc_adjusted$ua) / (rowSums(confusion_counts) - 1)) %>%
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
    cnt$count,
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


term1 <- cnt$count**2 *
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
  cnt$count**2,
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

cat("# Accuracy assessment", file = fo)

close(fo)


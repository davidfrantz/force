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
      "Usage: %s [-h] [-v] [-i] [-o output-file] -c count-file -m map-file -r reference-file\n", 
      get_Rscript_filename()
    ),
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
    "help",      "h", 0, "logical",
    "version",   "v", 0, "logical",
    "info",      "i", 0, "logical",
    "output",    "o", 2, "character",
    "counts",    "c", 1, "character",
    "map",       "m", 1, "character",
    "reference", "r", 1, "character"
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
table <- reference %>% 
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
classes <- c(
    table$map, 
    table$reference
  ) %>%
  unique() %>%
  sort()
n_classes <- length(classes)


# initialize confusion matrix
confusion_counts <- matrix(
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


acc_metrics <- function(confusion_matrix) {

  sum_all <- sum(confusion_matrix)
  sum_map_class <- rowSums(confusion_matrix)
  sum_ref_class <- colSums(confusion_matrix)

  oa <- confusion_matrix %>%
    diag() %>%
    sum() %>%
    `/`(sum_all)
  oa

  pa <- confusion_matrix %>%
    diag() %>%
    `/`(sum_ref_class)
  pa

  ua <- confusion_matrix %>%
    diag() %>%
    `/`(sum_map_class)
  ua

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



cnt <- data.frame(class = 12:8, count = runif(5, 1e4, 1e6))

cnt <- cnt %>% 
  arrange(class) %>%
  mutate(weight = count / sum(count))

if (!any(cnt$class == classes)) exit_with_error("classes in file-count do not match with map or reference classes")

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

acc_metrics(confusion_counts)
acc_metrics(confusion_adjusted)

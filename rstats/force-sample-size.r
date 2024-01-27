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


# load libraries ####################################################
library(dplyr)
library(getopt)


# input #############################################################

# usage function in same style as other FORCE tools,
# do not use getop's built-in usage function for consistency
usage <- function(exit){

  message <- c(
    sprintf(
      "Usage: %s [-h] [-v] [-i] [-e error] [-s min_size] [-o output-file] -c count-file -u user_acc-file\n", 
      get_Rscript_filename()
    ),
    "\n",
    "  -h  = show this help\n",
    "  -v  = show version\n",
    "  -i  = show program's purpose\n",
    "\n",
    "  -e error = standard error, defaults to 0.01\n",
    "\n",
    "  -s min_size = minimum sample size per class\n",
    "\n",
    "  -o output-file  = output file path with extension,\n",
    "     defaults to './sample-size.csv'\n",
    "\n",
    "  -c count-file  = csv table with pixel counts per class\n",
    "     2 columns named value and count",
    "\n",
    "  -u user_acc-file  = csv table with expected user accuracy per class\n",
    "     2 columns named value and UA",
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
    "help",     "h", 0, "logical",
    "version",  "v", 0, "logical",
    "info",     "i", 0, "logical",
    "error",    "e", 2, "numeric",
    "min_size", "s", 2, "integer",
    "counts",   "c", 1, "character",
    "user_acc", "u", 1, "character",
    "output",   "o", 1, "character"
  ), 
  byrow = TRUE, 
  ncol = 4
)

opt <- getopt(spec)

if (!is.null(opt$help)) usage()
if (!is.null(opt$info)) exit_normal("Suggest sample size for validation of a classification.\n")
if (!is.null(opt$version)) exit_normal("Printing function not implemented yet. Sorry.\n")

if (is.null(opt$counts)) exit_with_error("count-file is missing!")
if (is.null(opt$user_acc)) exit_with_error("user_acc-file is missing!")

file_existing(opt$counts)
file_existing(opt$user_acc)

if (is.null(opt$error)) opt$error <- 0.01
if (is.null(opt$min_size)) opt$min_size <- 50
if (is.null(opt$output)) opt$output <- "sample-size.csv"


pixel_counts   <- read.csv(opt$counts)
users_accuracy <- read.csv(opt$user_acc)


# main thing ########################################################

# join input tables
table <- users_accuracy %>% 
  inner_join(
    pixel_counts,
    by = "value"
  )

# join worked?
if (nrow(table) != nrow(pixel_counts)){
  exit_with_error("count and user_acc could not be joined")
}

# compute proportional area, and standard deviation of UA
table <- table %>% 
  mutate(area = count / sum(count)) %>%
  mutate(stdev = sqrt(UA*(1-UA))) %>%
  mutate(areaXstdev = area * stdev)

# number of recommended samples
number <- (sum(table$areaXstdev)/standard_error)**2

sprintf("%d samples are suggested.\n", number) %>%
  cat()

# compute class-wise sample size for equal and proportional allocation
table <- table %>% 
  mutate(equal = round(number / nrow(table))) %>%
  mutate(proportional = round(number * area)) %>%
  mutate(compromise = NA)


# do we have enough samples in proportional allocation?
if (min(table$proportional) < minsize){

  cat("Proportional allocation yields too few samples.\n")
  cat("A compromise between equal and proportional allocation is recommended.\n")

  # first, assign minimum sample size to small classes
  rare <- table %>% 
    filter(proportional < minsize) %>% 
    mutate(compromise = minsize)
  
  n_rare <- sum(rare$compromise)
  
  # allocate remaining samples to big classes proportionally
  big <- table %>% 
    filter(proportional >= minsize) %>% 
    mutate(compromise = (number-n_rare) * area)

  # check if proportionally allocated classes are big enough
  if (any(big$compromise < minsize)){
    exit_with_error("Compromising failed. Adjust input parameters.")
  }

  table <- rbind(rare, big)

} else {
  cat("Proportional allocation recommended.\n")
}

# compute deviation of compromised allocation from proportional in percent
table <- table %>%
    mutate(deviation = (compromise - proportional) / proportional * 100)

# write output
write.csv(
  table, 
  opt$output, 
  row.names = FALSE, 
  quote = FALSE
)

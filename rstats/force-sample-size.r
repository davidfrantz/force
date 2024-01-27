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
# do not use getop's built-in for consistency
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
    "\n",
    "  -u user_acc-file  = csv table with expected user accuracy per class\n",
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

version <- function() {
  cat("Printing function not implemented yet. Sorry.\n")
  quit(
    save = "no",
    status = exit
  )
}

info <- function() {
  cat("Suggest sample size for validation of a classification.\n")
  quit(
    save = "no",
    status = exit
  )
}

mandatory_missing <- function(argument) {
  cat(
    sprintf("%s is missing!\n", argument), 
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
if (!is.null(opt$info)) info()
if (!is.null(opt$version)) version()

if (is.null(opt$counts)) mandatory_missing("count-file")
if (is.null(opt$user_acc)) mandatory_missing("user_acc-file")

file_existing(opt$counts)
file_existing(opt$user_acc)

if (is.null(opt$error)) opt$error <- 0.01
if (is.null(opt$min_size)) opt$min_size <- 50
if (is.null(opt$output)) opt$output <- "sample-size.csv"


standard_error <- 0.01
minsize <- 175

pixel_counts <- data.frame(
  value = 0:2, 
  count = c(
    1868242749,
    496472240,
    502851998
    )
)

users_accuracy <- data.frame(
  value = 0:2, 
  UA = c(
    0.9,
    0.9,
    0.9
  )
) 

df <- users_accuracy %>% 
  inner_join(
    pixel_counts,
    by = "value"
  )

if (nrow(df) != nrow(pixel_counts)){
  "input tables do not match\n"
  stop()
}

df <- df %>% 
  mutate(proportion = count / sum(count)) %>%
  mutate(stdev = sqrt(UA*(1-UA))) %>%
  mutate(WS = proportion * stdev)

number <- (sum(df$WS)/standard_error)**2
number

df <- df %>% 
  mutate(equal = round(number / nrow(df))) %>%
  mutate(proportional = round(number * proportion)) %>%
  mutate(compromise = NA)

if (min(df$proportional) < minsize){
  
  rare <- df %>% 
    filter(proportional < minsize) %>% 
    mutate(compromise = minsize)
  
  n_rare <- sum(rare$compromise)
  
  big <- df %>% 
    filter(proportional >= minsize) %>% 
    mutate(compromise = (number-n_rare) * proportion)

  if (any(big$compromise < minsize)){
    c(
      "compromising the sample allocation failed...\n",
      "either decrease minsize or adjust input parameters to \n",
      "have more samples overall."
      )
  }

  df <- rbind(rare, big)

}

df %>%
    mutate(deviation = (compromise - proportional) / proportional * 100)

write.csv(df, file_output, row.names = FALSE, quote = FALSE)




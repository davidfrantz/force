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


silent_library <- function(package) {
  suppressMessages(library(package, character.only = TRUE))
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
  if (!file.exists(path)) {
    cat(
      sprintf("file %s does not exist\n", path),
      file = stderr()
    )
    usage(1)
  }
}

print_version <- function(progdir) {
  path <- sprintf("%s/force-misc/force-version.txt", progdir)
  file_existing(path)
  path %>%
    readLines() %>%
    exit_normal()
}

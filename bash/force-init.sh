#!/bin/bash

##########################################################################
# 
# This file is part of FORCE - Framework for Operational Radiometric 
# Correction for Environmental monitoring.
# 
# Copyright (C) 2013-2022 David Frantz
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
# 
##########################################################################

# this script initializes a new project with common directories to help you getting started

# functions/definitions ------------------------------------------------------------------
export PROG=`basename $0`;
export BIN="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

MANDATORY_ARGS=1

echoerr(){ echo "$PROG: $@" 1>&2; }    # warnings and/or errormessages go to STDERR
export -f echoerr

export DEBUG=true # display debug messages?
debug(){ if [ "$DEBUG" == "true" ]; then echo "DEBUG: $@"; fi } # debug message
export -f debug

help(){
cat <<HELP

Usage: $PROG [-hvi] project

  optional:
  -h = show this help
  -v = show version
  -i = show program's purpose

  mandatory:
  project = a directory name for your project, this should not exist yet

  -----
    see https://force-eo.readthedocs.io/en/latest/components/auxilliary/init.html

HELP
exit 1
}
export -f help



# now get the options --------------------------------------------------------------------
ARGS=`getopt -o hvi --long help,version,info -n "$0" -- "$@"`
if [ $? != 0 ] ; then help; fi
eval set -- "$ARGS"

while :; do
  case "$1" in
    -h|--help) help ;;
    -v|--version) echo "version-print to be implemented"; exit 0;;
    -i|--info) echo "Initialization of a new project"; exit 0;;
    -- ) shift; break ;;
    * ) break ;;
  esac
  shift
done

if [ $# -gt $MANDATORY_ARGS ] ; then 
  echoerr "Too many mandatory arguments."; help
elif [ $# -lt $MANDATORY_ARGS ] ; then 
  echoerr "Mandatory argument is missing."; help
else
  DOUT=$(readlink -f $1) # absolute directory path
fi
debug "new project will be created here: $DOUT"

# options received, check now ------------------------------------------------------------
if [[ -d "$DOUT" ]]; then
  echoerr "$DOUT is an existing directory, exiting."; exit 1;
fi

# main thing -----------------------------------------------------------------------------

# create main project directory
mkdir "$DOUT"
if ! [[ -d "$DOUT" ]]; then
  echoerr "creating project directory failed: $DOUT"; exit 1;
fi

# as we were successful in creating the parent directoy, we assume that
# all children will work

mkdir "$DOUT"/vector
mkdir "$DOUT"/vector/aoi
mkdir "$DOUT"/vector/grid

mkdir "$DOUT"/param
mkdir "$DOUT"/param/level2
mkdir "$DOUT"/param/level3
mkdir "$DOUT"/param/level4

mkdir "$DOUT"/misc
mkdir "$DOUT"/misc/dem
mkdir "$DOUT"/misc/wvp

mkdir "$DOUT"/assets
mkdir "$DOUT"/assets/mask
mkdir "$DOUT"/assets/endmember
mkdir "$DOUT"/assets/labels
mkdir "$DOUT"/assets/samples
mkdir "$DOUT"/assets/models

mkdir "$DOUT"/udf
mkdir "$DOUT"/udf/ard
mkdir "$DOUT"/udf/tsa

mkdir "$DOUT"/schedule
mkdir "$DOUT"/schedule/cron

mkdir "$DOUT"/level1
mkdir "$DOUT"/level1/metadata
mkdir "$DOUT"/level1/landsat
mkdir "$DOUT"/level1/sentinel2

mkdir "$DOUT"/level2
mkdir "$DOUT"/level2/ard
mkdir "$DOUT"/level2/base
mkdir "$DOUT"/level2/cso
mkdir "$DOUT"/level2/temp
mkdir "$DOUT"/level2/log
mkdir "$DOUT"/level2/provenance
mkdir "$DOUT"/level2/report

mkdir "$DOUT"/level3
mkdir "$DOUT"/level4

exit 0

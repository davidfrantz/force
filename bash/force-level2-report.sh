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

# this script generates a report for FORCE Level-2 processing system executions

# functions/definitions ------------------------------------------------------------------
export PROG=`basename $0`;
export BIN="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
export MISC="$BIN/force-misc"

# source bash "library" file
LIB="$MISC/force-bash-library.sh"
eval ". ${LIB}" >/dev/null 2>&1 ;[[ "$?" -ne "0" ]] && echo "loading bash library failed" && exit 1;
export LIB


MANDATORY_ARGS=1

export REPORT_EXE="R"
export REPORT_TEMPLATE="$MISC/force-level2-report.Rmd"

help () {
cat <<HELP

Usage: $PROG [-hvi] [-o] dir-log

  -h = show this help
  -v = show version
  -i = show program's purpose

  -o = output file
        defaults to FORCE_L2PS_YYYYMMDD-HHMMSS.html

$PROG:  generate Level 2 processing report
        see https://force-eo.readthedocs.io/en/latest/components/...tbd

HELP
exit 1
}
export -f help

cmd_not_found "$REPORT_EXE";
file_not_found "$REPORT_TEMPLATE";

# now get the options --------------------------------------------------------------------
ARGS=`getopt -o hvio: --long help,version,info,output: -n "$0" -- "$@"`
if [ $? != 0 ] ; then help; fi
eval set -- "$ARGS"


while :; do
  case "$1" in
    -h|--help) help ;;
    -v|--version) force_version; exit 0;;
    -i|--info) echo "Generate Level 2 processing report"; exit 0;;
    -o|--output) OUTPUT="$2"; shift ;;
    -- ) shift; break ;;
    * ) break ;;
  esac
  shift
done


if [ $# -lt $MANDATORY_ARGS ] ; then 
  echoerr "Mandatory argument is missing."; help
fi

LOGDIR=$(readlink -f $1) # absolute directory path
export LOGDIR

if [ ! -d $LOGDIR ]; then
  echoerr "$LOGDIR does not exist."; help
fi

if [ ! -d $LOGDIR ]; then
  echoerr "$LOGDIR does not exist."; help
fi

if [ -z "$OUTPUT" ]; then
  if [ ! -w $LOGDIR ]; then
    echoerr "$LOGDIR is not writeable."; help
  fi
  TIME=$(date +"%Y%m%d-%H%M%S")
  OUTPUT="$LOGDIR/FORCE_L2PS_$TIME.html"
fi
export OUTPUT=$(readlink -f $OUTPUT) # absolute directory path
export OUTDIR=`dirname $OUTPUT`;

debug "binary directory: $BIN"
debug "log directory: $LOGDIR"
debug "output: $OUTPUT"

$REPORT_EXE -e "rmarkdown::render('$REPORT_TEMPLATE', output_file = '$OUTPUT', intermediates_dir = '$OUTDIR', params = list(dlog = '$LOGDIR'))"

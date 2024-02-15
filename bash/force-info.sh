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

# Entrypoint, print short disclaimer, available modules etc.

# functions/definitions ------------------------------------------------------------------
export PROG=`basename $0`;
export BIN="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
export MISC="$BIN/force-misc"

# source bash "library" file
LIB="$MISC/force-bash-library.sh"
eval ". ${LIB}" >/dev/null 2>&1 ;[[ "$?" -ne "0" ]] && echo "loading bash library failed" && exit 1;
export LIB


print_info(){
  for cmd in "$@"; do
    if [ -f $cmd ] && [ -x $cmd ]; then
      message=`$cmd -i`
      if [ $? -eq 0 ]; then 
        echo $(basename $cmd): "$message"
      fi
    fi
  done
}
export -f print_info

help(){
cat <<HELP

Usage: $PROG [-hvi]

  optional:
  -h = show this help
  -v = show version
  -i = show program's purpose

  -----
    see https://force-eo.readthedocs.io

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
    -v|--version) force_version; exit 0;;
    -i|--info) echo "Entrypoint, print short disclaimer, available modules etc."; exit 0;;
    -- ) shift; break ;;
    * ) break ;;
  esac
  shift
done


# main thing -----------------------------------------------------------------------------

info(){
cat <<INFO

###############################################################################

Hello $(whoami)! You are currently running FORCE v. $(force_version)

Framework for Operational Radiometric Correction for Environmental monitoring
Copyright (C) 2013-2024 David Frantz, david.frantz@uni-trier.de
  + many community contributions (https://github.com/davidfrantz/force)

FORCE is free software under the terms of the GNU General Public License as 
  published by the Free Software Foundation, see http://www.gnu.org/licenses/

Thank you for using FORCE! This software is being developed in the hope that it
  will be helpful for you and your work.

However, it is requested that you to use the software in accordance with 
  academic standards and fair usage. Without this, software like FORCE will not
  survive. This includes citation of the software and the scientific publica-
  tions, proper acknowledgement in any public presentation, or an offer of co-
  authorship of scientific articles in case substantial help in setting up, 
  modifying or running the software is provided by the author(s).

At minimum, the citation of following paper is requested:
  Frantz, D. (2019): FORCE — Landsat + Sentinel-2 Analysis Ready Data and 
  Beyond. Remote Sensing, 11, 1124

Each FORCE module will generate a "CITEME" file with suggestions for references
 to be cited. This list is based on the specific parameterization you are using

Usefule resources:
- Code, issues, discussions etc.: https://github.com/davidfrantz/force
- Documentation and tutorials: https://force-eo.readthedocs.io
- Docker images: https://hub.docker.com/r/davidfrantz/force/

##########################################################################

INFO
}
export -f info

info

echo "available programs:"
echo ""

executables=$BIN/force*
debug $executables
print_info $executables

echo ""
echo "##########################################################################"

exit 0

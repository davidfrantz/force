#!/bin/bash

##########################################################################
# 
# This file is part of FORCE - Framework for Operational Radiometric 
# Correction for Environmental monitoring.
# 
# Copyright (C) 2013-2020 David Frantz
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

# this script increases the version


EXPECTED_ARGS=1

# if wrong number of input args, stop
if [ $# -ne $EXPECTED_ARGS ]; then
  echo "Usage: `basename $0` version"
  echo ""
  exit 1
fi


VERSION=$1

V_README=README.md
V_DOCS=docs/source/index.rst
V_CODE=src/cross-level/_version-cl.h

sed -i -e "s/\(\*\*Version\) .*\(\*\*\)/\1 $VERSION\2/" $V_README
sed -i -e "s/\(\*\*Version\) .*\(\*\*\)/\1 $VERSION\2/" $V_DOCS
sed -i -e "s/\(#define _VERSION_ \"\).*\(\"\)/\1$VERSION\2/" $V_CODE

exit 0


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

EXPECTED_ARGS=1

if [ $# -ne $EXPECTED_ARGS ]
then
  echo "Usage: `basename $0` parameter-file"
  echo ""
  exit
fi


INP=$1

if [ ! -r $INP ]; then
  echo "$INP is not existing/readable"
  exit
fi

DIR=$(dirname $INP)
BASE=$(basename $INP)
BASE=${BASE%%.*}


# detect replacement vectors
KEYS=$(sed -e '/^+/q' $INP | grep '%:' | sed 's/^%//g' | sed 's/%:.*$//g' )
NKEYS=$(echo $KEYS | wc -w)
#echo $KEYS
#echo $NKEYS

if [ $NKEYS -lt 1 ]; then
  echo "No replacement vector detected"
  exit
else 
  echo "$NKEYS replacement vectors detected"
fi


# combine values
for k in $KEYS; do

  VALUES={$(grep "%$k%:" $INP | sed 's/^.*%: //' | tr -s ' ' | sed 's/^ //' | sed 's/ /,/g' )}
  #echo $VALUES

  combis="$combis%$VALUES"

done

COMBS=$(eval "echo "$combis"")
#echo $COMBS


# for each combination: replace
NPAR=0
for comb in $COMBS; do 

  v=($(echo $comb | sed 's/%/ /g'))
  #echo ${v[*]}

  # bandname
  ((NPAR++))
  C_NPAR=$(printf "%05d" $NPAR)
  OUT=$DIR/$BASE"_"$C_NPAR".prm"

  # init new par
  cp $INP $OUT

  # remove magic parameter definition
  sed -i '/^\+/,$!d' $OUT

  # replace
  n=0
  for k in $KEYS; do
    #echo "replace {%$k%} with ${v[n]}"
    sed -i "s+{%$k%}+${v[n]}+g" $OUT
    ((n++))
  done

done

echo "$NPAR parameter files were generated"

exit 0


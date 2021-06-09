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

# functions/definitions ------------------------------------------------------------------
PROG=`basename $0`;
BIN="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

MANDATORY_ARGS=1

echoerr(){ echo "$PROG: $@" 1>&2; }    # warnings and/or errormessages go to STDERR

cmd_not_found(){      # check required external commands
  for cmd in "$@"; do
    stat=`which $cmd`
    if [ $? != 0 ] ; then echoerr "\"$cmd\": external command not found, terminating..."; exit 1; fi
  done
}

help(){
cat <<HELP

Usage: $PROG [-h] [-c {all,paired}] [-o] parameter-file

  optional:
  -h = show this help
  -c = combination type
       all:    all combinations (default)
       paired: pairwise combinations
  -o = output directory, defaults to directory of parameter-file

  mandatory:
  parameter-file: base parameter-file that includes replacement vectors

$PROG: replace variables in parameterfile
  see https://force-eo.readthedocs.io/en/latest/components/auxilliary/magic-parameters.html

HELP
exit 1
}

#cmd_not_found "...";    # important, check required commands !!! dies on missing

# now get the options --------------------------------------------------------------------
ARGS=`getopt -o hc:o: --long help,combine:,output: -n "$0" -- "$@"`
if [ $? != 0 ] ; then help; fi
eval set -- "$ARGS"

COMB='all'
DOUT='NA'
while :; do
  case "$1" in
    -h|--help) help ;;
    -c|--combine) COMB="$2"; shift ;;
    -o|--output) DOUT="$2"; shift ;;
    -- ) shift; break ;;
    * ) break ;;
  esac
  shift
done

if [ $# -ne $MANDATORY_ARGS ] ; then 
  echoerr "Mandatory argument is missing."; help
else
  FINP=$(readlink -f $1) # absolute file path
  BINP=$(basename $FINP) # basename
  CINP=${BINP%%.*}       # corename (without extension)
  DINP=$(dirname  $FINP) # directory name
fi

# options received, check now ------------------------------------------------------------
if [ ! "$COMB" = "all" ] && [ ! "$COMB" = "paired" ]; then 
  echoerr "Invalid combination type"; help
fi

if [ "$DOUT" == "NA" ]; then DOUT=$DINP; fi

# further checks and preparations --------------------------------------------------------
if ! [[ -f "$FINP" && -r "$FINP" ]]; then
  echoerr "$FINP is not a readable file, exiting."; exit 1;
fi

if ! [[ -d "$DOUT" && -w "$DOUT" ]]; then
  echoerr "$DOUT is not a writeable directory, exiting."; exit 1;
fi

# main thing -----------------------------------------------------------------------------

# detect replacement vectors
KEYS=$(sed -e '/^+/q' $FINP | grep '%:' | sed 's/^%//g' | sed 's/%:.*$//g' )
NKEYS=$(echo $KEYS | wc -w)
#echo $KEYS
#echo $NKEYS

if [ $NKEYS -lt 1 ]; then
  echoerr "No replacement vector detected"; help
else 
  echo "$NKEYS replacement vectors detected"
fi


if [ "$COMB" = "all" ]; then 

  # combine values
  for k in $KEYS; do
    VALUES={$(grep "%$k%:" $FINP | sed 's/^.*%: //' | tr -s ' ' | sed 's/^ //' | sed 's/ /,/g' )}
    #echo $VALUES
    combis="$combis%$VALUES"
    #echo $combis
  done

  COMBS=$(eval "echo "$combis"")
  #echo $COMBS

elif [ "$COMB" = "paired" ]; then 

  # get array lengths
  NVALUES=0
  for k in $KEYS; do
    VALUES=$(grep "%$k%:" $FINP | sed 's/^.*%: //' | tr -s ' ' | sed 's/^ //')
    if [ "$NVALUES" -gt 0 ] && [ "$NVALUES" -ne "$(echo $VALUES | wc -w)" ]; then
      echoerr "Replacement vector misformed. If -c paired, all vectors must have the same length"; help
    fi
    NVALUES=$(echo $VALUES | wc -w)
  done

  # combine values
  combis=()
  for k in $KEYS; do
    VALUES=$(grep "%$k%:" $FINP | sed 's/^.*%: //' | tr -s ' ' | sed 's/^ //')
    #echo ${VALUES[*]}
    i=0
    for v in $VALUES; do
      combis[$i]=${combis[$i]}"%""$v"
      ((i++))
    done
    #echo ${combis[*]}
  done
  
  COMBS=${combis[*]}
  #echo $COMBS

else
  echoerr "Invalid combination type"; help
fi


# for each combination: replace
NPAR=0
for comb in $COMBS; do 

  v=($(echo $comb | sed 's/%/ /g'))
  #echo ${v[*]}

  # bandname
  ((NPAR++))
  C_NPAR=$(printf "%05d" $NPAR)
  FOUT=$DOUT/$CINP"_"$C_NPAR".prm"

  # init new par
  cp $FINP $FOUT

  # remove magic parameter definition
  sed -i '/^\+/,$!d' $FOUT

  # replace
  n=0
  for k in $KEYS; do
    #echo "replace {%$k%} with ${v[n]}"
    sed -i "s+{%$k%}+${v[n]}+g" $FOUT
    ((n++))
  done

done

echo "$NPAR parameter files were generated"


exit 0


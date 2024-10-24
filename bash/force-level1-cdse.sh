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

# Copyright (C) 2024 David Frantz


# functions/definitions ------------------------------------------------------------------
PROG=`basename $0`;
BIN="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
MISC="$BIN/force-misc"
export PROG BIN MISC

echo "change library!!!"
MISC="misc"

# source bash "library" file
LIB="$MISC/force-bash-library.sh"
if ! eval ". ${LIB}" >/dev/null 2>&1; then 
  echo "loading bash library failed" 1>&2
  exit 1
fi

MANDATORY_ARGS=0

export CURL_EXE="curl"
export PARALLEL_EXE="parallel"


help(){
cat <<HELP

Usage: $PROG [-hvi]

  optional:
  -h = show this help
  -v = show version
  -i = show program's purpose

  -----
    see ...

HELP
exit 1
}
export -f help

# important, check required commands !!! dies on missing
cmd_not_found "$CURL_EXE"
cmd_not_found "$PARALLEL_EXE"


# now get the options --------------------------------------------------------------------
ARGS=$(getopt -o hvic:d: --long help,version,info -n "$0" -- "$@")
if [ $? != 0 ] ; then help; fi
eval set -- "$ARGS"

MAX_CC=100
DATE_MIN="1970-01-01"
DATE_MAX=$(date +%Y-%m-%d)

while :; do
  case "$1" in
    -h|--help) help ;;
    -v|--version) force_version; exit 0;;
    -i|--info) echo "Download from CDSE"; exit 0;;
    -c) MAX_CC="$2"; shift ;;
    -d)
      check_params "$2" "date"
      DATE_MIN=$(echo "$2" | cut -d"," -f1)
      DATE_MAX=$(echo "$2" | cut -d"," -f2)
      shift ;;
    -- ) shift; break ;;
    * ) break ;;
  esac
  shift
done

if [ $# -lt $MANDATORY_ARGS ] ; then 
  echoerr "Mandatory argument is missing."; 
  help
else
  FILE_PRM=$(readlink -f "$1") # absolute file path
fi


# options received, check now ------------------------------------------------------------
#file_not_found "$FILE_PRM"
#$UNIX_EOL_EXE -q "$FILE_PRM"

# check if dates are correct
if ! date -d $DATE_MIN &> /dev/null || ! [ ${#DATE_MIN} -eq 10 ]; then
  echoerr "Starttime ($DATE_MIN) is not a valid date. Make sure dates are formatted as YYYY-MM-DD"
elif ! date -d $DATE_MAX &> /dev/null || ! [ ${#DATE_MAX} -eq 10 ]; then
  echoerr "Endtime ($DATE_MAX) is not a valid date. Make sure dates are formatted as YYYY-MM-DD"
elif [ $(date -d $DATE_MIN +%s) -gt $(date -d $DATE_MAX +%s) ]; then
  echoerr "Start of date is larger than end date. Start: $DATE_MIN, End: $DATE_MAX"
fi

# check if cloud cover is valid
if is_lt $MAX_CC 0 || is_gt $MAX_CC 100; then
  echoerr "Maximum cloud cover must be specified between 0 and 100 ($MAX_CC)"
fi



# main thing -----------------------------------------------------------------------------

TIME=$(date +"%Y%m%d%H%M%S")
debug "$TIME"



# Operators
#AND="%20and%20"
#OR="%20or%20"
#NOT="%20not%20"
#GT="%20gt%20"
#LT="%20lt%20"
#EQ="%20eq%20"

Q_BASE_URL="https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
Q_COLLECTION="Collection/Name eq 'SENTINEL-2'"
Q_CLOUD_COVER="Attributes/OData.CSC.DoubleAttribute/any(att:att/Name eq 'cloudCover' and att/OData.CSC.DoubleAttribute/Value le $MAX_CC)"
Q_PRODUCT_TYPE="Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'productType' and att/OData.CSC.StringAttribute/Value eq 'S2MSI1C')"
Q_AOI="OData.CSC.Intersects(area=geography'SRID=4326;POINT(7.86036658114807 54.2838645073117)')"
Q_START="ContentDate/Start ge ""$DATE_MIN""T00:00:00.000Z"
Q_END="ContentDate/End le ""$DATE_MAX""T00:00:00.000Z"

QUERY="$Q_BASE_URL?\$filter=$Q_COLLECTION and $Q_START and $Q_END and $Q_CLOUD_COVER and $Q_PRODUCT_TYPE and $Q_AOI&\$expand=Locations&\$top=1000"
#jj
echo $QUERY


QUERY=${QUERY// /%20}

echo $QUERY

#"https://catalogue.dataspace.copernicus.eu/odata/v1/Products?\$filter=Collection/Name%20eq%20'SENTINEL-2'%20and%20ContentDate/Start%20gt%202024-06-01T00:00:00.000Z%20and%20$CLOUDCOVER%20and%20Attributes/OData.CSC.StringAttribute/any(att:att/Name%20eq%20'productType'%20and%20att/OData.CSC.StringAttribute/Value%20eq%20'S2MSI1C')%20and%20OData.CSC.Intersects(area=geography'SRID=4326;POINT(7.86036658114807%2054.2838645073117)')&\$expand=Locations"

$CURL_EXE $QUERY

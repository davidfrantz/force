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

# Copyright (C) 2020-2021 Stefan Ernst
# Contact: stefan.ernst@hu-berlin.de

# This script downloads Landsat and Sentinel-2 Level 1 data from GCS
trap "echo Exited!; exit;" SIGINT SIGTERM # make sure that CTRL-C breaks out of download loop
set -e # make sure script exits if any process exits unsuccessfully
echoerr() { echo "$PROG: $@" 1>&2; }

# ============================================================
# check dependencies
DEPENDENCIES=('gsutil' 'gunzip' 'ogrinfo')
for DEPENDENCY in "${DEPENDENCIES[@]}"; do
  if ! [ -x "$(command -v $DEPENDENCY)" ]; then
    printf "%s\n" "" "Error: could not find $DEPENDENCY" "Please make sure all required external programs are installed" "Required programs: ${DEPENDENCIES[@]}"
    exit 1
  fi
done

# ============================================================
# set up functions
check_params() {
  # make sure we have two comma-separated arguments to optional params
  if ! echo "$1" | grep -q ","; then show_help "$(printf "%s\n       " "Only one ""$2"" specified or ""$2""s not separated by comma")"; fi
  local secondarg=$(echo "$1" | cut -d"," -f2)
  if ! [ -n $secondarg ]; then show_help "$(printf "%s\n       " "Only one ""$2"" specified." "Provide exactly two arguments separated by a comma ,")"; fi
}

is_in_range() {
  awk -v value="$1" -v lower="$2" -v upper="$3" 'BEGIN {print (lower <= value && value <= upper)}'
}

is_smaller() {
  awk -v val1="$1" -v val2="$2" 'BEGIN {print (val1 < val2)}'
}

round() {
  local valmult=$(awk -v val="$1" -v digits="$2" 'BEGIN {print (val * 10^digits)}')
  awk -v val="$valmult" -v digits="$2" 'BEGIN {print (int(val+0.5)) / 10^digits}'
}

show_help() {
cat << HELP

force-level1-csd - FORCE cloud storage downloader for Landsat and Sentinel-2
https://force-eo.readthedocs.io/en/latest/components/lower-level/level1/level1-csd.html

Usage: force-level1-csd [-c min,max] [-d starttime,endtime] [-n] [-k] [-s sensor]
                        [-t tier] [-u]
                        metadata-dir level-1-datapool queue aoi

Error: `printf "$1"`
HELP

exit 1
}

show_progress() {
  SIZEDONE=$(awk -v done=$SIZEDONE -v fsize=$FILESIZE 'BEGIN { print (done + fsize) }' )
  PERCDONE=$(awk -v total=$TOTALSIZE -v done=$SIZEDONE 'BEGIN { printf( "%.2f\n", (100 / total * done) )}')
  local WIDTH=$(($(tput cols) - 9)) PERCINT=$(( $(echo $PERCDONE | cut -d"." -f1) + 1 ))
  if [ $ITER -eq 1 ]; then local PERCINT=0; fi
  printf -v INCREMENT "%*s" "$(( $PERCINT*$WIDTH/100 ))" ""; INCREMENT=${INCREMENT// /=}
  printf "\r\e[K|%-*s| %3d %% %s" "$WIDTH" "$INCREMENT" "$PERCINT" "$*"
}

update_meta() {
  printf "%s\n" "" "Downloading ${1^} metadata catalogue..."
  gsutil -m -q cp gs://gcp-public-data-$1/index.csv.gz "$METADIR"
  printf "%s\n" "Extracting compressed ${1^} metadata catalogue..."
  gunzip "$METADIR"/index.csv.gz
  mv "$METADIR"/index.csv "$METADIR"/metadata_$2.csv
}

which_satellite() {
  # set SENTINEL and LANDSAT flags and check if sensor names are valid
  SENSIN=$(echo $SENSIN | tr '[:lower:]' '[:upper:]')  # convert sensor strings to upper case to prevent unnecessary headaches
  for SENSOR in $(echo $SENSIN | sed 's/,/ /g'); do
    case $SENSOR in
      S2A|S2B)
        SENTINEL=1 ;;
      LT04|LT05|LE07|LC08)
        LANDSAT=1 ;;
      *)
        show_help "$(printf "%s\n       " "Invalid sensor(s) specified" "Sensors provided: $SENSIN" "Valid sensors: S2A,S2B,LT04,LT05,LE07,LC08")"
        exit 1
    esac
  done
}


# ============================================================
# Initialize arguments and parse command line input
SENSIN="LT04,LT05,LE07,LC08,S2A,S2B"
DATEMIN="19700101"
DATEMAX=$(date +%Y%m%d)
CCMIN=0
CCMAX=100
TIER="T1"
DRYRUN=0
LANDSAT=0
SENTINEL=0
UPDATE=0
KEEPMETA=0
CHECKLOGS=0

# Negative coordinates: change ( -) to %dummy% if followed by integer: prevent interpretation as option by getopt
ARGS=$(echo "$*" | sed -E "s/ -([0-9])/ %dummy%\1/g")
set -- $ARGS

ARGS=`getopt -o c:d:l:nks:t:u -l cloudcover:,daterange:,logs:,no-act,keep-meta,sensors:,tier:,update -n $0 -- "$@"`
if [ $? != 0 ] ; then show_help "$(printf "%s\n       " "Error in command line options. Please check your options.")"; fi
eval set -- "$ARGS"

while :; do
  case "$1" in
    -c|--cloudcover)
      check_params "$2" "cloud cover threshold"
      CCMIN=$(echo "$2" | cut -d"," -f1)
      CCMAX=$(echo "$2" | cut -d"," -f2)
      shift ;;
    -d|--daterange)
      check_params "$2" "date"
      DATEMIN=$(echo "$2" | cut -d"," -f1)
      DATEMAX=$(echo "$2" | cut -d"," -f2)
      shift ;;
    -n|--no-act)
      DRYRUN=1 ;;
    -k|--keep-meta)
      KEEPMETA=1 ;;
    -l|--logs)
      CHECKLOGS=1
      LPATH="$2"
      shift ;;
    -s|--sensors)
      SENSIN="$2"
      shift ;;
    -t|--tier)
      TIER="$2"
      shift ;;
    -u|--update)
      UPDATE=1 ;;
    --)
      shift; break ;;
    *)
      break
  esac
  shift
done

# change %dummy% back to -
ARGS=$(echo "$@" | sed -E "s/%dummy%([0-9])/-\1/g")
eval set -- "$ARGS"

# Check for update flag and update metadata catalogue if set
if [ $UPDATE -eq 1 ]; then
  METADIR="$1"
  if [ $# -lt  1 ]; then
    show_help "$(printf "%s\n       " "Metadata directory not specified")"
  elif [ $# -gt 1 ]; then
    show_help "$(printf "%s\n       " "Too many arguments." "Only specify the metadata directory when using the update option (-u)." "The only allowed optional argument is -s. Use it if you would like to only" "update either the Landsat or Sentinel-2 metadata catalogue.")"
  elif ! [ -w "$METADIR" ]; then
    show_help "$(printf "%s\n       " "Metadata directory does not exist, exiting")"
  elif ! [ -w "$METADIR" ]; then
    show_help "$(printf "%s\n       " "Can not write to metadata directory, exiting")"
  else
    which_satellite
    if [ $LANDSAT -eq 1 ]; then
      update_meta landsat landsat
    fi
    if [ $SENTINEL -eq 1 ]; then
      update_meta sentinel-2 sentinel2
    fi
  fi
  printf "%s\n" "" "Done. You can run this script without option -u to download data now." ""
  exit 0
fi

# check if number of mandatory args is correct
if [ $# -ne 4 ]; then
  show_help "$(printf "%s\n       " "Incorrect number of mandatory input arguments provided" "Expected: 4 Received: $#: $(echo "$@" | sed 's/ /,/g')")"
fi

which_satellite

# ============================================================
# Check user input and set up variables
METADIR="$1"
POOL="$2"
QUEUE="$3"
AOI="$4"

# check for empty arguments
if [[ -z $CHECKLOGS || -z $METADIR || -z $POOL || -z $QUEUE || -z $AOI || -z $CCMIN || -z $CCMAX || -z $DATEMIN || -z $DATEMAX || -z $SENSIN || -z $TIER ]]; then
  show_help "$(printf "%s\n       " "One or more arguments are undefined, please check the following" "" "Metadata directory: $METADIR" "Level-1 pool: $POOL" "Queue: $QUEUE" "AOI: $AOI" "Sensors: $SENSIN" "Start date: $DATEMIN, End date: $DATEMAX" "Cloud cover minimum: $CCMIN, cloud cover maximum: $CCMAX" "Tier (Landsat only): $TIER")"
fi

# check for correct tier
for T in $(echo $TIER | sed 's/,/ /g'); do
  case $T in
    T1|T2|RT)
      true ;;
    *)
      show_help "$(printf "%s\n       " "Invalid tier specified. Valid tiers: T1,T2,RT")" ;;
   esac
done

# check if dates are correct
if ! [[ $DATEMIN =~ ^[[:digit:]]+$ ]] || ! [[ $DATEMAX  =~ ^[[:digit:]]+$ ]]; then
  show_help "$(printf "%s\n       " "Entered dates seem to contain non-numeric characters." "Start: $DATEMIN, End: $DATEMAX")"
elif ! date -d $DATEMIN &> /dev/null || ! [ ${#DATEMIN} -eq 8 ]; then
  show_help "$(printf "%s\n       " "Starttime ($DATEMIN) is not a valid date." "Make sure dates are formatted as YYYYMMDD")"
elif ! date -d $DATEMAX &> /dev/null || ! [ ${#DATEMAX} -eq 8 ]; then
  show_help "$(printf "%s\n       " "Endtime ($DATEMAX) is not a valid date." "Make sure dates are formatted as YYYYMMDD")"
elif [ $(date -d $DATEMIN +%s) -gt $(date -d $DATEMAX +%s) ]; then
  show_help "$(printf "%s\n       " "Start of date is larger than end date." "Start: $DATEMIN, End: $DATEMAX")"
fi

# check if cloud cover is valid
if [ $(is_smaller $CCMIN 0) -eq 1 ] || [ $(is_smaller 100 $CCMIN) -eq 1 ] || [ $(is_smaller $CCMAX 0) -eq 1 ] || [ $(is_smaller 100 $CCMAX ) -eq 1 ]; then
  show_help "$(printf "%s\n       " "Cloud cover minimum and maximum must be specified between 0 and 100" "Cloud cover minimum: $CCMIN" "Cloud cover maximum: $CCMAX")"
  elif [ $(is_smaller $CCMAX $CCMIN) -eq 1 ]; then
    show_help "$(printf "%s\n       " "Cloud cover minimum is larger than cloud cover maximum" "Cloud cover minimum: $CCMIN" "Cloud cover maximum: $CCMAX")"
fi

# check if POOL folder exists and is writeable
if [ ! -w "$POOL" ]; then
  show_help "$(printf "%s\n       " "Level 1 datapool folder does not exist or is not writeable.")"
fi

# check if LOGS folder exists and create list of processed scenes
if [ $CHECKLOGS -eq 1 ]; then
  if [ ! -d "$LPATH" ]; then
    show_help "$(printf "%s\n       " "Log folder does not seem to exist.")"
  fi

# set gsutil config var (necessary for docker installations)
if [ -z "$FORCE_CREDENTIALS" ]; then
  BOTO_CONFIG=$HOME/.boto
else
  BOTO_CONFIG=$FORCE_CREDENTIALS/.boto
fi
export BOTO_CONFIG
if [ ! -r $BOTO_CONFIG ]; then
  show_help "$(printf "%s\n       " "gsutil config file was not found in $CREDDIR.")"
fi


# ======================================
# check type of AOI
# 1 - shapefile
# 2 - coordinates as text
# 3 - PathRow as text

# Is AOI a file?
if [ -f $AOI ]; then
  # is AOI a GDAL readable file?
  if ogrinfo $AOI >& /dev/null; then
    AOITYPE=1
    OGR=1
  else
    # Must be tile list or bounding box
    # check if tile list / bounding box file contains whitespaces or non-unix eol
    if grep -q " " $AOI; then
      show_help "$(printf "%s\n       " "Whitespace in AOI definition detected." "Please make sure this file does not contain whitespace.")"
    elif
      grep -U -q $'\015' $AOI; then
        show_help "$(printf "%s\n       " "AOI file seems to contain CR characters." "Did you create this file under Windows/MacOS?" "Please make sure this file uses UNIX end of line (LF) and does not contain whitespace.")"
    fi

    AOI=$(cat $AOI | sed 's/,/./g')
    OGR=0
  fi
# if aoi is not a file, it's a polygon or tile list as cmd line input
else
  AOI=$(echo $AOI | sed 's/,/ /g')
  OGR=0
fi

if [ $OGR -eq 0 ]; then
  # check if AOI input contains bounding box coordinates
  if $(echo $AOI | grep -q "/"); then
    AOITYPE=2
    # are coords valid lat/lon?
    for COORD in $AOI; do
      if ! $(echo $COORD | grep -q "/"); then
        show_help "$(printf "%s\n       " "At least one of the AOI coordinates does not seem to be separated by a forward slash /" "Coordinate: $COORD")"
      fi
      LAT=$(echo $COORD | cut -d"/" -f1)
      LON=$(echo $COORD | cut -d"/" -f2)

      if ! [ $(is_in_range $LAT -90 90) -eq 1 ]; then
        show_help "$(printf "%s\n       " "Latitude out of range" "Coordinate: $COORD - $LAT is not in range -90 to 90" "This error may also mean that you tried to use a vector file as AOI but provided an incorrect path")"
      elif ! [ $(is_in_range $LON -180 180) -eq 1 ]; then
        show_help "$(printf "%s\n       " "Longitute out of range" "Coordinate: $COORD - $LON is not in range -180 to 180")"
      fi
    done
  # else, AOI input must be tile list - check if tiles are valid Path/Row or S2 tiles
  else
    AOITYPE=3
    for ENTRY in $AOI
    do
      if $(echo $ENTRY | grep -q -E "[0-2][0-9]{2}[0-2][0-9]{2}"); then
        LSPATH="${ENTRY:0:3}"
        LSROW="${ENTRY:3:6}"
        if [ $(is_in_range $LSPATH 1 233) -eq 0 ] || [ $(is_in_range $LSPATH 1 248) -eq 0 ]; then
          show_help "$(printf "%s\n       " "Landsat PATH / ROW out of range. PATH not in range 1 to 233 or ROW not in range 1 to 248." "PATH / ROW received: $ENTRY")"
        fi
        continue
      elif $(echo $ENTRY | grep -q -E "T[0-9]{2}[A-Z]{3}"); then
        if [ $(is_in_range "${ENTRY:1:2}" 1 60) -eq 0 ]; then
          show_help "$(printf "%s\n       " "MGRS tile number out of range." "Valid range: 0 to 60, received: $ENTRY")"
        elif [[ -z "$(echo ${ENTRY:3:1} | grep -E "[C,D,E,F,G,H,J,K,L,M,N,P,Q,R,S,T,U,V,W,X]")" || -z "$(echo ${ENTRY:4:1} | grep -E "[A,B,C,D,E,F,G,H,K,L,M,N,P,Q,R,T,U,V,W,X,Y,Z]")" || -z "$(echo ${ENTRY:5:1} | grep -E "[A,B,C,D,E,F,G,H,J,K,L,M,N,P,Q,R,S,T,U,V]")" ]]; then
          show_help "$(printf "%s\n       " "Tile does not seem to be a valid Sentinel-2 tile: $ENTRY" "Please make sure all tiles exist.")"
        fi
        continue
      else
        show_help "$(printf "%s\n       " "Tile list as AOI detected." "One or more tiles seem to be formatted incorrectly." "Please check $ENTRY" "If you are trying to define an AOI using coordinates, makes sure X and Y are separated using a forward slash /")"
      fi
    done
  fi
fi



# ============================================================
# Function get_data:
# 1. Prepare request
# 2. Query metadata catalogue
# 3. Download data
get_data() {
  SATELLITE=$1
  PRINTNAME=${SATELLITE^}
  case $SATELLITE in
    landsat) SENSORS=$(echo $SENSIN | grep -o "L[C,E,T]0[4,5,7,8]") ;;
    sentinel2) SENSORS=$(echo $SENSIN | grep -o "S2[A-B]") ;;
  esac


  # ============================================================
  # Check if metadata catalogue exists and is up to date
  METACAT="$METADIR/metadata_$SATELLITE.csv"
  if ! [ -f $METACAT ]; then
    printf "%s\n" "" "Error: $METACAT: Metadata catalogue does not exist." "Use the -u option to download / update the metadata catalogue" ""
    exit 1
  fi

  METADATE=$(date -r "$METACAT" +%s)
  if [ $(date -d $DATEMAX +%s) -gt $METADATE ]; then
    printf "%s\n" "" "WARNING: The selected time window exceeds the last update of the $PRINTNAME metadata catalogue." "Results may be incomplete, please consider updating the metadata catalogue using the -u option."
  fi

  if [ "$AOITYPE" -eq 1 ]; then
    printf "%s\n" "" "Searching for footprints / tiles intersecting with geometries of AOI shapefile..."
    OGRTEMP="$POOL"/l1csd-temp_$(date +%FT%H-%M-%S-%N)
    mkdir "$OGRTEMP"
    # get first layer of vector file and reproject to epsg4326
    AOILAYER=$(ogrinfo "$AOI" | grep "1: " | sed "s/1: //; s/ ([[:alnum:]]*.*)//")
    AOIREPRO="$OGRTEMP"/aoi_reprojected.shp
    ogr2ogr -t_srs EPSG:4326 -f "ESRI Shapefile" "$AOIREPRO" "$AOI"
    # get ls/s2 tiles intersecting with bounding box of AOI
    BBOX=$(ogrinfo -so "$AOIREPRO" "$AOILAYER" | grep "Extent: " | sed 's/Extent: //; s/(//g; s/)//g; s/, /,/g; s/ - /,/')
    WFSURL="http://ows.geo.hu-berlin.de/cgi-bin/qgis_mapserv.fcgi?MAP=/owsprojects/grids.qgs&SERVICE=WFS&REQUEST=GetCapabilities&typename="$SATELLITE"&bbox="$BBOX
    ogr2ogr -f "ESRI Shapefile" "$OGRTEMP"/$SATELLITE.shp WFS:"$WFSURL" -append -update
    # intersect AOI and tiles
    # remove duplicate entries resulting from multiple features in same tiles: | xargs -n 1 | sort -u | xargs |
    TILERAW=$(ogr2ogr -f CSV /vsistdout/ -dialect sqlite -sql "SELECT $SATELLITE.PRFID FROM $SATELLITE, aoi_reprojected WHERE ST_Intersects($SATELLITE.geometry, aoi_reprojected.geometry)" "$OGRTEMP" | xargs -n 1 | sort -u | xargs | sed 's/PRFID,//')
    TILES="_"$(echo $TILERAW | sed 's/ /_|_/g')"_"
    rm -rf "$OGRTEMP"

  elif [ "$AOITYPE" -eq 2 ]; then
    printf "%s\n" "" "Searching for footprints / tiles intersecting with input geometry..."
    WKT=$(echo $AOI | sed 's/ /%20/g; s/\//,/g')
    WFSURL="http://ows.geo.hu-berlin.de/cgi-bin/qgis_mapserv.fcgi?MAP=/owsprojects/grids.qgs&SERVICE=WFS&REQUEST=GetFeature&typename="$SATELLITE"&Filter=%3Cogc:Filter%3E%3Cogc:Intersects%3E%3Cogc:PropertyName%3Eshape%3C/ogc:PropertyName%3E%3Cgml:Polygon%20srsName=%22EPSG:4326%22%3E%3Cgml:outerBoundaryIs%3E%3Cgml:LinearRing%3E%3Cgml:coordinates%3E"$WKT"%3C/gml:coordinates%3E%3C/gml:LinearRing%3E%3C/gml:outerBoundaryIs%3E%3C/gml:Polygon%3E%3C/ogc:Intersects%3E%3C/ogc:Filter%3E"
    TILERAW=$(ogr2ogr -f CSV /vsistdout/ -select "PRFID" WFS:"$WFSURL")
    TILES="_"$(echo $TILERAW | sed 's/PRFID, //; s/ /_|_/g')"_"

  elif [ "$AOITYPE" -eq 3 ]; then
    sensor_tile_mismatch() {
      printf "%s\n" "" "Error: $PRINTNAME sensor(s) specified, but no $PRINTNAME tiles identified." "Check if sensors and footprints match or use the -s option to specify sensors to query." ""
      exit 1
    }
    case $SATELLITE in
      landsat)
        TILERAW=$(echo "$AOI" | grep -E -o "[0-9]{6}") || sensor_tile_mismatch
        TILES="_"$(echo $TILERAW | sed 's/ /_|_/g')"_" ;;
     sentinel2)
        TILERAW=$(echo "$AOI" | grep -E -o "T[0-6][0-9][A-Z]{3}") || sensor_tile_mismatch
        TILES="_"$(echo $TILERAW | sed 's/ /_|_/g')"_" ;;

    esac
  fi


  printf "%s\n" "" "Querying the metadata catalogue for $PRINTNAME data" "Sensor(s): "$(echo $SENSORS | sed 's/ /,/g')
  if [ $SATELLITE == "landsat" ]; then printf "%s\n" "Tier(s): $TIER"; fi
  printf "%s\n" "Tile(s): "$(echo $TILERAW | sed 's/PRFID, //; s/ /,/g') "Daterange: "$DATEMIN" to "$DATEMAX "Cloud cover minimum: "$CCMIN"%, maximum: "$CCMAX"%" ""


  # ============================================================
  # Filter metadata and extract download links
  # sort by processing date (+ baseline for s2), drop duplicates entries of tile + sensing time
  if [ $SATELLITE = "sentinel2" ]; then
    # 4: tile 5: sensing time 9: generation time 14: url
    LINKS=$(grep -E $TILES $METACAT | grep -E $(echo ""$SENSORS"" | sed 's/ /_|/g')"_" | awk -F "," '{OFS=","} {gsub("T[0-9]{2}:[0-9]{2}:[0-9]{2}.[0-9]{6}Z|-","",$5)}1' | awk -v start="$DATEMIN" -v stop="$DATEMAX" -v clow="$CCMIN" -v chigh="$CCMAX" -F "," '{OFS=","} $5 >= start && $5 <= stop && $7 >= clow && $7 <= chigh'| sort -t"," -k 14.76,14.78r -k9r | awk -F"," '{OFS=","} !a[$4,$5]++' | sort -t"," -k 5)
  elif [ $SATELLITE = "landsat" ]; then
    # 5: acqu time 10: path 11: row
    LINKS=$(grep -E $TILES $METACAT | grep -E $(echo ""$SENSORS"" | sed 's/ /_|/g')"_" | grep -E $(echo "_"$TIER | sed 's/,/,|_/g')"," | awk -F "," '{OFS=","} {gsub("-","",$5)}1' | awk -v start="$DATEMIN" -v stop="$DATEMAX" -v clow="$CCMIN" -v chigh="$CCMAX" -F "," '$5 >= start && $5 <= stop && $6 == 01 && $12 >= clow && $12 <= chigh' | sort -t"," -k 2.27,2.34r | awk -F"," '{OFS=","} !a[$10$11,$5]++' | sort -t"," -k 5)
  fi

  METAFNAME="$POOL"/csd_metadata_"$SATELLITE"_$(date +%FT%H-%M-%S-%N).txt
  printf "%s" "$LINKS" > "$METAFNAME"
  NSCENES=$(sed -n '$=' "$METAFNAME")
  
  # ============================================================
  # Check log folder for processed products
  if [[ $CHECKLOGS -eq 1 ]]; then

    LOGS=$(find "$LPATH" -maxdepth 1 -type f -name '*.log' | rev | cut -d/ -f1 | rev | cut -d. -f1)
    LOGSFNAME="$POOL"/logs_$(date +%FT%H-%M-%S-%N).txt
    printf "%s\n" "$LOGS" > "$LOGSFNAME"
    # make sure there are log files present in the provided path
    if [ -z "$LOGS" ]; then
      printf "%s\n" "Error: No FORCE Level 2 log files found in ""$LPATH"" " "Please make sure you provided the correct file path" ""
      exit 1
    fi

    LINKS=$(grep -v -f $LOGSFNAME $METAFNAME) || true  # bypass set -e to avoid exiting when all scenes were already processed
    rm "$LOGSFNAME"

    if [[ -z $LINKS ]]; then
      printf "%s\n" "$NSCENES $PRINTNAME Level 1 scenes found." "According to the log files, all of these have already been processed. Exiting."
      rm "$METAFNAME"
      exit 0
    fi
    
    printf "%s\n" "$LINKS" >| "$METAFNAME"
    NSCENESUPDATED=$(wc -l "$METAFNAME" | cut -d" " -f1)
    NPROCESSED=$(( $NSCENES - $NSCENESUPDATED ))
    NSCENES="$NSCENESUPDATED"
  fi
  
  if [ $KEEPMETA -eq 0 ]; then
    rm "$METAFNAME"
  else
    sed -i "1 s/^/$(head -n 1 $METACAT)\n/" "$METAFNAME"
  fi


  # ============================================================
  # Get total number and size of scenes matching criteria
  
  case $SATELLITE in
    sentinel2) 
      TOTALSIZE=$(printf "%s" "$LINKS" | awk -F "," '{s+=$6/1048576} END {printf "%f", s}') ;;
    landsat) 
      TOTALSIZE=$(printf "%s" "$LINKS" | awk -F "," '{s+=$17/1048576} END {printf "%f", s}') ;;
  esac  
  PRSIZE=$TOTALSIZE
  UNIT="MB"
  if [ ${PRSIZE%%.*} -gt 1024 ]; then
    PRSIZE=$(echo $PRSIZE | awk '{print $1 / 1024}')
    UNIT="GB"
  fi
  if [ ${PRSIZE%%.*} -gt 1024 ]; then
    PRSIZE=$(echo $PRSIZE | awk '{print $1 / 1024}')
    UNIT="TB"
  fi
  if [ ${PRSIZE%%.*} -gt 1024 ]; then
    PRSIZE=$(echo $PRSIZE | awk '{print $1 / 1024}')
    UNIT="PB"
  fi
  PRSIZE=$(round $PRSIZE 2)

  if [ -z $NSCENES ]; then
    printf "%s\n" "There were no $PRINTNAME Level 1 scenes found matching the search criteria." ""
  else
    printf "%s\n" "$NSCENES $PRINTNAME Level 1 scenes matching criteria found" "$PRSIZE$UNIT data volume found."
    if [ $CHECKLOGS -eq 1 ]; then
      printf "%s\n" "" "$(( $NSCENES + $NPROCESSED )) scenes found in total." "$NPROCESSED scenes from this search were already processed and are not included in the results."
    fi
  fi

  # ============================================================
  # Download scenes
  PERCDONE=0
  SIZEDONE=0
  if [[ $DRYRUN -eq 0 && ! -z $LINKS ]]; then

    POOL=$(cd "$POOL"; pwd)
    printf "%s\n" "" "Starting to download "$NSCENES" "$PRINTNAME" Level 1 scenes" "" "" "" "" ""

    ITER=1
    
    for LINK in $LINKS
    do
      
    if [ $SATELLITE = "sentinel2" ]; then
        SCENEID=$(echo $LINK | cut -d"," -f14 | sed 's+^.*/++')
        TILE=$(echo $LINK | cut -d"," -f 1 | grep -o -E "T[0-9]{2}[A-Z]{3}")
        URL=$(echo $LINK | cut -d"," -f 14)
        FILESIZEBYTE=$(echo $LINK | cut -d"," -f 6)
      elif [ $SATELLITE = "landsat" ]; then
        SCENEID=$(echo $LINK | cut -d"," -f 2)
        TILE=$(echo $SCENEID | cut -d"_" -f 3)
        URL=$(echo $LINK | cut -d"," -f 18)
        FILESIZEBYTE=$(echo $LINK | cut -d, -f 17)
      fi
      FILESIZE=$(echo $(echo $FILESIZEBYTE | awk '{print $1 / 1048576}') | cut -d"." -f1)

      show_progress

      TILEPATH="$POOL"/$TILE
      SCENEPATH="$TILEPATH"/$SCENEID
      if [ -d "$SCENEPATH" ]; then
        if ! ls -R "$SCENEPATH" | grep -q ".gstmp" && ! [ -z "$(ls -A $SCENEPATH)" ]; then
          printf "\e[500D\e[4A\e[2KScene "$SCENEID"("$ITER" of "$NSCENES") exists, skipping...\e[4B"

          ((ITER++))
          continue
        fi
      fi

      # create target directory if it doesn't exist
      if [ ! -w "$TILEPATH" ]; then
        mkdir "$TILEPATH"
        if [ ! -w "$TILEPATH" ]; then
          printf "%s\n" "" "$TILEPATH: Creating directory failed." ""
          exit 1
        fi
      fi

      printf "\e[500D\e[2A\e[2KDownloading "$SCENEID"("$ITER" of "$NSCENES")...\e[2B"
      gsutil -m -q cp -R $URL "$TILEPATH"

      lockfile-create $QUEUE
      echo "$SCENEPATH QUEUED" >> $QUEUE
      lockfile-remove $QUEUE

      ((ITER++))
    done
  fi
}

if [[ $LANDSAT -eq 1 && $SENTINEL -eq 1 ]]; then
  printf "%s\n" "" "Landsat and Sentinel-2 data requested." "Landsat data will be queried and downloaded first."
fi
if [ $LANDSAT -eq 1 ]; then
  get_data landsat
fi
if [ $SENTINEL -eq 1 ]; then
  get_data sentinel2
fi

printf "%s\n" "" "Done." ""
exit 0

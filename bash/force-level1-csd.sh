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

# Copyright (C) 2020 Stefan Ernst
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
show_help() {
cat << HELP

Usage: `basename $0` [optional arguments] metadata-dir level-1-datapool queue aoi

Mandatory arguments:

  metadata-dir
  Directory where the metadata catalogues (csv file) are stored

  level-1-datapool
  An existing directory, your files will be stored here

  queue
  Downloaded files are appended to a file queue, which is needed for the 
  Level 2 processing. The file doesn't need to exist. If it exists, new 
  lines will be appended on successful ingestion

  area of interest
  (1) user-supplied coordinates of your study area: 
      The polygon must be closed (first X/Y = last X/Y). X/Y must be given as
      decimal degrees with negative values for West and South coordinates.
      You can either give the path to a file, or give the coordinates on the command line.
      If in the file, put one coordinate per line.
      If on the command line, give a comma separated list.
  (2) a shapefile (point/polygon/line). On-the-fly reprojection is provided,
      but using EPSG4326 is recommended.
  (3) Scene identifier.
      Landsat: Path/Row as "PPPRRR". Make sure to keep leading zeros:
        correct: 181034, incorrect: 18134
      Sentinel-2: MGRS tile as "TXXXXX". Make sure to keep the leading T before the MGRS tile number.
      You can either give the path to a file, or give the IDs on the command line.
      If in the file, put one ID per line.
      If on the command line, give a comma separated list.
  
Optional arguments:

  -h | --help
  Show this help
  
  -c | --cloudcover
  minimum,maximum
  The cloud cover range must be specified in %
  Default: 0,100
  
  -d | --daterange
  starttime,endtime
  Dates must be given in the following format: YYYYMMDD,YYYYMMDD
  Default: 19700101,today
  
  -n | --no-act
  Will trigger a dry run that will only return the number of images
  and their total data volume
  
  -k | --keep-meta
  Will write the results of the query to the metadata directory.
  Two files will be created if Landsat and Sentinel-2 data is queried
  at the same time. Filename: csd_metadata_YYYY-MM-DDTHH-MM-SS
  
  -s | --sensor
  Sensors to include in the query, comma-separated.
  Valid sensors:
  Landsat                             Sentinel
  LT04 - Landsat 4 TM                 S2A - Sentinel-2A MSI
  LT05 - Landsat 5 TM                 S2B - Sentinel-2B MSI
  LE07 - Landsat 7 ETM+               
  LC08 - Landsat 8 OLI
  Default: LT04,LT05,LE07,LC08,S2A,S2B
  
  -t | --tier
  Landsat collection tier level. Valid tiers: T1,T2,RT
  Default: T1
  
  -u | --update
  Will update the metadata catalogue (download and extract from GCS)
  If this option is used, only one mandatory argument is expected (metadata-dir).
  Use the -s option to only update Landsat or Sentinel-2 metadata.
    
HELP
exit 1
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
  SENSIN=$(echo $SENSIN | tr '[:lower:]' '[:upper:]')  # convert sensor strings to upper case to prevent unnecessary headaches
  for SENSOR in $(echo $SENSIN | sed 's/,/ /g'); do
    case $SENSOR in
      S2A|S2B)
        SENTINEL=1 ;;
      LT04|LT05|LE07|LC08)
        LANDSAT=1 ;;
      *)
        printf "%s\n" "" "Error: invalid sensor(s) specified" "Sensors provided: $SENSIN" "Valid sensors: S2A,S2B,LT04,LT05,LE07,LC08" ""
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

ARGS=`getopt -o c:d:nhks:t:u --long cloudcover:,daterange:,no-act,help,keep-meta,sensors:,tier:,update -n $0 -- "$@"`
if [ $? != 0 ] ; then  printf "%s\n" "" "Error in command line options. Please check your options." >&2 ; show_help ; fi
eval set -- "$ARGS"

while :; do
  case "$1" in
    -c|--cloudcover)
      CCMIN=$(echo "$2" | cut -d"," -f1)
      CCMAX=$(echo "$2" | cut -d"," -f2)
      shift ;;
    -d|--daterange)
      DATEMIN=$(echo "$2" | cut -d"," -f1)
      DATEMAX=$(echo "$2" | cut -d"," -f2)
      shift ;;
    -n|--no-act)
      DRYRUN=1 ;;
    -h|--help)
      show_help ;;
    -k|--keepmeta)
      KEEPMETA=1 ;;
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


# Check for update flag and update metadata catalogue if set
if [ $UPDATE -eq 1 ]; then
  METADIR="$1"
  if [ $# -lt  1 ]; then
    printf "%s\n" "" "Metadata directory not specified, exiting" ""
    exit 1
  elif [ $# -gt 1 ]; then
    printf "%s\n" "" "Error: Invalid argument." "Only specify the metadata directory when using the update option (-u)." "The only allowed optional argument is -s. Use it if you don't want to" "update the Landsat and Sentinel-2 metadata catalogues at the same time." ""
    #"Please only specify the metadata directory when using the update option (-u)" "To only update either of the LS / S2 catalogues, you may also use the -s option" ""
    exit 1
  elif ! [ -w $METADIR ]; then
    printf "%s\n" "" "Metadata directory does not exist, exiting" ""
    exit 1
  elif ! [ -w $METADIR ]; then
    printf "%s\n" "" "Can not write to metadata directory, exiting" ""
    exit 1
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
  printf "%s\n" "" "Incorrect number of mandatory input arguments provided" "Expected: 4 Received: $#: $(echo "$@" | sed 's/ /,/g')"
  show_help
fi

which_satellite

# ============================================================
# Check user input and set up variables
METADIR="$1"
POOL="$2"
QUEUE="$3"
AOI="$4"

# check for empty arguments
if [[ -z $METADIR || -z $POOL || -z $QUEUE || -z $AOI || -z $CCMIN || -z $CCMAX || -z $DATEMIN || -z $DATEMAX || -z $SENSIN || -z $TIER ]]; then
  printf "%s\n" "" "Error: One or more arguments are undefined, please check the following" "" "Metadata directory: $METADIR" "Level-1 pool: $POOL" "Queue: $QUEUE" "AOI: $AOI" "Sensors: $SENSIN" "Start date: $DATEMIN, End date: $DATEMAX" "Cloud cover minimum: $CCMIN, cloud cover maximum: $CCMAX" "Tier (Landsat only): $TIER" ""
  exit 1
fi

# check for correct tier
for T in $(echo $TIER | sed 's/,/ /g'); do
  case $T in
    T1|T2|RT)
      true ;;
    *)
      printf "%s\n" "Error: Invalid tier specified. Valid tiers: T1,T2,RT" ""
      exit 1 ;;
   esac
done

# check if dates are correct
if ! [[ $DATEMIN =~ ^[[:digit:]]+$ ]] || ! [[ $DATEMAX  =~ ^[[:digit:]]+$ ]]; then
  printf "%s\n" "" "Error: One of the entered dates seems to contain non-numeric characters." "Start: $DATEMIN, End: $DATEMAX" ""
  exit 1
elif ! date -d $DATEMIN &> /dev/null || ! [ ${#DATEMIN} -eq 8 ]; then
  printf "%s\n" "" "starttime ($DATEMIN) is not a valid date." "Make sure date is formatted as YYYYMMDD" ""
  exit 1
elif ! date -d $DATEMAX &> /dev/null || ! [ ${#DATEMAX} -eq 8 ]; then
    printf "%s\n" "" "endtime ($DATEMAX) is not a valid date." "Make sure date is formatted as YYYYMMDD" ""
  exit 1
elif [ $(date -d $DATEMIN +%s) -gt $(date -d $DATEMAX +%s) ]; then
  printf "%s\n" "Error: Start of date range is larger than end of date range" "Start: $DATEMIN, End: $DATEMAX" ""
  exit 1
fi

# check if cloud cover is valid
if [ $(is_smaller $CCMIN 0) -eq 1 ] || [ $(is_smaller 100 $CCMIN) -eq 1 ] || [ $(is_smaller $CCMAX 0) -eq 1 ] || [ $(is_smaller 100 $CCMAX ) -eq 1 ]; then
  printf "%s\n" "" "Error: Cloud cover minimum and maximum must be specified between 0 and 100" "Cloud cover minimum: $CCMIN" "Cloud cover maximum: $CCMAX" ""
  exit 1
  elif [ $(is_smaller $CCMAX $CCMIN) -eq 1 ]; then
    printf "%s\n" "" "Error: Cloud cover minimum is larger than cloud cover maximum" "Cloud cover minimum: $CCMIN" "Cloud cover maximum: $CCMAX" ""
    exit 1
fi

# check type of AOI
# 1 - shapefile
# 2 - coordinates as text
# 3 - PathRow as text
if [ -f $AOI ]; then
  # check if AOI is GDAL readable file
  if ogrinfo $AOI >& /dev/null; then
    AOITYPE=1
    OGR=1
  else
    # check if tile list / bounding box file contains whitespaces
    if grep -q " " $AOI; then
      printf "%s\n" "Error: whitespace in AOI definition detected." "Please make sure this file uses Linux style end of lines and does not contain whitespaces." ""
      exit 1
    fi
    AOI=$(cat $AOI | sed 's/,/./g')
    OGR=0
  fi
# if soi is not a file, it's a polygon or tile list as cmd line input
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
      LAT=$(echo COORD | cut -d"/" -f1)
      LON=$(echo COORD | cut -d"/" -f2)
      if ! grep -q "/" $COORD; then
        printf  "%s\n" "Error: At least one of the AOI coordinates does not seem to be in the format LAT/LON" "Coordinate: $COORD" ""
        exit 1
      elif ! [ $(is_in_range $LAT -90 90) -eq 1 ]; then
        printf "%s\n" "Error: Latitude out of range" "Coordinate: $COORD - $LAT is not in range -90 to 90" ""
        exit 1
      elif ! [ $(is_in_range $LON -180 180) -eq 1 ]; then
        printf "%s\n" "Error: Longitute out of range" "Coordinate: $COORD - $LON is not in range -180 to 180" ""
        exit 1
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
          printf "%s\n" "" "Landsat PATH / ROW out of range. PATH not in range 1 to 233 or ROW not in range 1 to 248." "PATH / ROW received: $ENTRY" ""
          exit 1
        fi
        continue
      elif $(echo $ENTRY | grep -q -E "T[0-6][0-9][A-Z]{3}"); then
        if ! [ $(is_in_range ${ENTRY:2:3} 1 60) ]; then
          printf "%s\n" "" "MGRS tile number out of range. Valid range: 0 to 60, received: $ENTRY" ""
          exit 1
        elif [[ -z "$(echo ${ENTRY:3:1} | grep -E "[C,D,E,F,G,H,J,K,L,M,N,P,Q,R,S,T,U,V,W,X]")" || -z "$(echo ${ENTRY:4:1} | grep -E "[A,B,C,D,E,F,G,H,K,L,M,N,P,Q,R,T,U,V,W,X,Y,Z]")" || -z "$(echo ${ENTRY:5:1} | grep -E "[A,B,C,D,E,F,G,H,J,K,L,M,N,P,Q,R,S,T,U,V]")" ]]; then
          printf "%s\n" "" "Tile does not seem to be a valid Sentinel-2 tile: $ENTRY" "Please make sure all tiles exist." ""
          exit 1
        fi
        continue
      else
        printf "%s\n" "" "Tile list as AOI detected." "" "Error: One or more tiles seem to be formatted incorrectly." "Please check $ENTRY" ""
        exit 1
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
  METACAT=$METADIR"/metadata_$SATELLITE.csv"
  if ! [ -f $METACAT ]; then
    printf "%s\n" "" "$METACAT: Metadata catalogue does not exist. Use the -u option to download / update the metadata catalogue" ""
    exit 1
  fi

  METADATE=$(date -d $(stat $METACAT | grep "Change: " | cut -d" " -f2) +%s)
  if [ $(date -d $DATEMAX +%s) -gt $METADATE ]; then
    printf "%s\n" "" "WARNING: The selected time window exceeds the last update of the $PRINTNAME metadata catalogue." "Results may be incomplete, please consider updating the metadata catalogue using the -d option."
  fi

  if [ "$AOITYPE" -eq 1 ]; then
    printf "%s\n" "" "Searching for footprints / tiles intersecting with geometries of AOI shapefile..."
    AOINE=$(echo $(basename "$AOI") | rev | cut -d"." -f 2- | rev)
    BBOX=$(ogrinfo -so $AOI $AOINE | grep "Extent: " | sed 's/Extent: //; s/(//g; s/)//g; s/, /,/g; s/ - /,/')
    WFSURL="http://ows.geo.hu-berlin.de/cgi-bin/qgis_mapserv.fcgi?MAP=/owsprojects/grids.qgs&SERVICE=WFS&REQUEST=GetCapabilities&typename="$SATELLITE"&bbox="$BBOX

    ogr2ogr -f "GPKG" merged.gpkg WFS:"$WFSURL" -append -update
    ogr2ogr -f "GPKG" merged.gpkg $AOI -append -update

    TILERAW=$(ogr2ogr -f CSV /vsistdout/ -dialect sqlite -sql "SELECT $SATELLITE.PRFID FROM $SATELLITE, $AOINE WHERE ST_Intersects($SATELLITE.geom, ST_Transform($AOINE.geom, 4326))" merged.gpkg)
    TILES="_"$(echo $TILERAW | sed 's/PRFID, //; s/ /_|_/g')"_"
    rm merged.gpkg

  elif [ "$AOITYPE" -eq 2 ]; then
    printf "%s\n" "" "Searching for footprints / tiles intersecting with input geometry..."
    WKT=$(echo $AOI | sed 's/,/%20/g; s/\//,/g')
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
  if [ $SATELLITE = "sentinel2" ]; then
    # 5: sensing time 9: generation time 
    LINKS=$(grep -E $TILES $METACAT | grep -E $(echo ""$SENSORS"" | sed 's/ /_|/g')"_" | awk -F "," '{OFS=","} {gsub("T[0-9]{2}:[0-9]{2}:[0-9]{2}.[0-9]{6}Z|-","",$5)}1' | awk -v start="$DATEMIN" -v stop="$DATEMAX" -v clow="$CCMIN" -v chigh="$CCMAX" -F "," '{OFS=","} $5 >= start && $5 <= stop && $7 >= clow && $7 <= chigh'| sort -t"," -k 14.76,14.78r -k9.1,9.4r -k9.6,9.7r -k9.9,9.10r  | awk -F"," '{OFS=","} !a[$4,$5]++' | sort -t"," -k 5)
  elif [ $SATELLITE = "landsat" ]; then
    LINKS=$(grep -E $TILES $METACAT | grep -E $(echo ""$SENSORS"" | sed 's/ /_|/g')"_" | grep -E $(echo "_"$TIER | sed 's/,/,|_/g')"," | awk -F "," '{OFS=","} {gsub("-","",$5)}1' | awk -v start="$DATEMIN" -v stop="$DATEMAX" -v clow="$CCMIN" -v chigh="$CCMAX" -F "," '$5 >= start && $5 <= stop && $6 == 01 && $12 >= clow && $12 <= chigh' | sort -t"," -k 5)
  fi

  METAFNAME=$METADIR/csd_metadata_$(date +%FT%H-%M-%S).txt
  printf "%s" "$LINKS" > $METAFNAME
  case $SATELLITE in
    sentinel2) TOTALSIZE=$(printf "%s" "$LINKS" | awk -F "," '{s+=$6/1048576} END {printf "%f", s}') ;;
    landsat) TOTALSIZE=$(printf "%s" "$LINKS" | awk -F "," '{s+=$17/1048576} END {printf "%f", s}') ;;
  esac
  NSCENES=$(sed -n '$=' $METAFNAME)
  if [ $KEEPMETA -eq 0 ]; then
    rm $METAFNAME
  else
    sed -i "1 s/^/$(head -n 1 $METACAT)\n/" $METAFNAME
  fi


  # ============================================================
  # Get total number and size of scenes matching criteria
  UNIT="MB"
  PRSIZE=$TOTALSIZE
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

  if [ -z $NSCENES ];then
    printf "%s\n" "There were no $PRINTNAME Level 1 scenes found matching the search criteria." ""
  else
    #LC_NUMERIC="en_US.UTF-8" printf "%s\n%.2f%s\n" "$NSCENES $PRINTNAME Level 1 scenes matching criteria found" "$PRSIZE" "$UNIT data volume found."
    printf "%s\n" "$NSCENES $PRINTNAME Level 1 scenes matching criteria found" "$PRSIZE$UNIT data volume found."
  fi


  # ============================================================
  # Download scenes
  PERCDONE=0
  SIZEDONE=0
  if [[ $DRYRUN -eq 0 && ! -z $LINKS ]]; then

    POOL=$(cd $POOL; pwd)
    printf "%s\n" "" "Starting to download "$NSCENES" "$PRINTNAME" Level 1 scenes" "" "" "" "" ""
    
    ITER=1
    for LINK in $LINKS
    do
      SCENEID=$(echo $LINK | cut -d"," -f 2)

      if [ $SATELLITE = "sentinel2" ]; then
        TILE=$(echo $LINK | cut -d"," -f 1 | grep -o -E "T[0-9]{2}[A-Z]{3}")
        URL=$(echo $LINK | cut -d"," -f 14)
        FILESIZEBYTE=$(echo $LINK | cut -d"," -f 6)
      elif [ $SATELLITE = "landsat" ]; then
        TILE=$(echo $SCENEID | cut -d"_" -f 3)
        URL=$(echo $LINK | cut -d"," -f 18)
        FILESIZEBYTE=$(echo $LINK | cut -d, -f 17)
      fi
      FILESIZE=$(echo $(echo $FILESIZEBYTE | awk '{print $1 / 1048576}') | cut -d"." -f1)
      
      show_progress
      
      TILEPATH=$POOL/$TILE    
      SCENEPATH=$TILEPATH/$SCENEID
      if [ $SATELLITE = "sentinel2" ]; then
        if [[ $SCENEID == *"_OPER_"* ]]; then
          SCENEID=$(echo $URL | rev | cut -d"/" -f1 | rev | cut -d"." -f1)
        fi
        SCENEPATH=$TILEPATH/$SCENEID".SAFE"
      fi
      # Check if scene already exists, download anyway if gsutil temp files are present
      if [ -d $SCENEPATH ]; then
        if ! ls -r $SCENEPATH | grep -q ".gstmp" && ! [ -z "$(ls -A $SCENEPATH)" ]; then
          printf "\e[500D\e[4A\e[2KScene "$SCENEID"("$ITER" of "$NSCENES") exists, skipping...\e[4B"
          
          ((ITER++))
          continue
        fi
      fi
      
      # create target directory if it doesn't exist
      if [ ! -w $TILEPATH ]; then
        mkdir $TILEPATH
        if [ ! -w $TILEPATH ]; then
          printf "%s\n" "" "$TILEPATH: Creating directory failed." ""
          exit 1
        fi
      fi

      printf "\e[500D\e[2A\e[2KDownloading "$SCENEID"("$ITER" of "$NSCENES")...\e[2B"
      gsutil -m -q cp -R $URL $TILEPATH

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

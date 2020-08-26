#!/bin/bash

# TO DO 
# 1. Sanity check for CC fails if cc is specified as float
# 2. Check filesize if a scene has been downloaded already to catch broken downloads (delete and do again or check if gsutil can handle partial downloads)

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

echoerr() { echo "$PROG: $@" 1>&2; }    # warnings and/or errormessages go to STDERR

show_help() {
cat << HELP

Usage: `basename $0` [-d] [-u] metadata-dir level-1-datapool queue aoi
                   aoitype sensor starttime endtime min-cc max-cc

Mandatory arguments:
  metadata-dir
  directory where the Landsat metadata (csv file) is stored

  level-1-datapool
  An existing directory, your files will be stored here

  queue
  Downloaded files are appended to a file queue, which is needed for
  the Level 2 processing. The file doesn't need to exist. If it exists,
  new lines will be appended on successful ingestion

  area of interest
  (1) The coordinates of your study area: "X1/Y1,X2/Y2,X3/Y3,...,X1/Y1"
  The polygon must be closed (first X/Y = last X/Y). X/Y must be given as
  decimal degrees with negative values for West and South coordinates.
  (2) a shapefile (point/polygon/line). On-the-fly reprojection is provided,
  but using EPSG4326 is recommended
  (3) Path/Row (Landsat): "PPPRRR,PPPRRR,PPPRRR"
      Make sure to keep leading zeros - correct: 181034, incorrect: 18134
      Tile name (Sentinel-2): "34UEU,33UUU"

  type of area of interest
  1 - coordinates as text
  2 - shapefile
  3 - PathRow as text

  sensor
  Specify the sensor(s) to include. Separate with commas while retaining the
  order below. Landsat and Sentinel-2 sensors can not be combined.
  Landsat                             Sentinel-2
  LT05 - Landsat 5 TM                 S2A
  LE07 - Landsat 7 ETM+               S2B
  LC08 - Landsat 8 OLI
  Correct: "LT05,LC08", incorrect: "LC08,LT05" or "LE07,S2B"

  starttime endtime
  Dates must be given as YYYY-MM-DD

  min-cc max-cc
  The cloud cover range must be specified in %

Optional arguments (always placed AFTER platform/mirr and BEFORE mandatory arguments):
  -d dry
  will trigger a dry run that will only return the number of images
  and their total data volume

  -u update
  will update the metadata catalogue (download and extract from GCS)
  only the metadata dir is required as argument when using this option

  -h|--help
  show this help

  -t|--tier
  Landsat collection tier level. Valid tiers: T1,T2,RT
  Default: T1

HELP
exit 1
}

# TODO#

# TIER for CSV check - how to handle if T1, T2, NRT


which_satellite() {
  SENSIN=$(echo $SENSIN | tr '[:lower:]' '[:upper:]')  # convert sensor strings to upper case to prevent unnecessary headaches
  for SENSOR in $(echo $SENSIN | sed 's/,/ /g'); do
    case $SENSOR in
      S2A|S2B)
        SENTINEL=1 ;;
      LT04|LT05|LE07|LC08)
        LANDSAT=1 ;;
      *)
        printf "%s\n" "Error: invalid sensor(s) specified" "Valid sensors: S2A,S2B,LT04,LT05,LE07,LC08" ""
        exit 1
    esac
  done
}


update_meta() {
  echo "Updating metadata catalogue..."
  gsutil -m cp gs://gcp-public-data-$1/index.csv.gz $METADIR
  gunzip $METADIR/index.csv.gz
  mv $METADIR/index.csv $METADIR/metadata_$2.csv
}


SENSIN="LT04,LT05,LE07,LC08,S2A,S2B"
DATEMIN="1970-01-01"
DATEMAX=$(date +%Y-%m-%d)
CCMIN=0
CCMAX=100
TIER="T1"
DRYRUN=0
LANDSAT=0
SENTINEL=0
# set variables for urls, file names, layer names, print, ...

TEMP=`getopt --o c:d:nhs:t:u --long cloudcover:,daterange:,no-act,help,sensors:,tier:,update -n 'force-level1' -- "$@"`
eval set -- $TEMP


echo $@
while :; do
  case $1 in
    -c | --cloudcover)
      CCMIN=$(echo $2 | cut -d"," -f1)
      CCMAX=$(echo $2 | cut -d"," -f2)
      shift ;;
    -d | --daterange)
      DATEMIN=$(echo $2 | cut -d"," -f1)
      DATEMAX=$(echo $2 | cut -d"," -f2)
      shift ;;
    -n | --no-act)
      DRYRUN=1 ;;
    -h | --help)
      show_help ;;
    -s | --sensors)
      SENSIN=$2
      shift ;;
    -t | --tier)
      TIER=$2
      shift ;;
    -u | --update)
      METADIR=$2
      if [ $# -lt 2 ]; then
        echo "Metadata directory not specified, exiting"
        exit 1
      elif [ $# -gt 2 ]; then
       echo "Error: Please only specify the metadata directory when using the update option (-u)"
       exit 1
      elif ! [ -w $METADIR ]; then
        echo "Can not write to metadata directory, exiting"
        exit 1
      else
        which_satellite
        if [ $SENTINEL -eq 1 ]; then
          update_meta sentinel-2 sentinel2
        fi
        if [ $LANDSAT -eq 1 ]; then
          update_meta landsat landsat
        fi
      echo "Done. You can run this script without option -u to download data now."
      exit
      fi ;;
    -- ) shift; break ;;
    #-?*)
    #  printf "%s\n" "" "Incorrect option specified" ""
    #  show_help >&2 ;;
    *)
      break #no more options
  esac
  shift
done
echo $@
if [ $# -ne 4 ]; then
  printf "%s\n" "" "Incorrect number of mandatory input arguments provided" "Expected: 4 Received: $#: $(echo "$@" | sed 's/ /,/g')"
  show_help
fi

which_satellite

# ============================================================
# Check user input and set up variables
METADIR=$1
POOL=$2
QUEUE=$3
AOI=$4

if [[ -z $METADIR || -z $POOL || -z $QUEUE || -z $AOI || -z $CCMIN || -z $CCMAX || -z $DATEMIN || -z $DATEMAX || -z $SENSIN || -z $TIER ]]; then
  printf "%s\n" "Error: One or more variables are undefined, please check:" "Metadata directory: $METADIR" "Level-1 pool: $POOL" "Queue: $QUEUE" "AOI: $AOI" "Sensors: $SENSIN" "Start date: $DATEMIN, End date: $DATEMAX" "Cloud cover minimum: $CCMIN, cloud cover maximum: $CCMAX" "Tier (Landsat only): $TIER"
  exit 1
fi


for T in $(echo $TIER | sed 's/,/ /g'); do
  case $T in
    T1|T2|RT)
      true ;;
    *)
      printf "%s\n" "Error: Invalid tier specified. Valid tiers: T1,T2,RT" ""
      exit 1 ;;
   esac
done

if [ $(date -d $DATEMIN +%s) -ge $(date -d $DATEMAX +%s) ]; then
  printf "%s\n" "Error: Start of date range is larger or equal to end of date range" "Start: $DATEMIN, End: $DATEMAX" ""
  exit 1
  elif ! date -d $DATEMIN &> /dev/null; then
    printf "%s\n" "" "starttime ($DATEMIN) is not a valid date." "Make sure date is formatted as YYYY-MM-DD" ""
    exit 1
    elif ! date -d $DATEMAX &> /dev/null; then
      printf "%s\n" "" "endtime ($DATEMAX) is not a valid date." "Make sure date is formatted as YYYY-MM-DD" ""
    exit 1
fi




# FAILS FOR FLOATING POINTS, BASH DOESN'T DO FLOAT COMPARISON
if [ $CCMIN -lt 0 ] || [ $CCMIN -gt 100 ] || [ $CCMAX -lt 0 ] || [ $CCMAX -gt 100 ]; then
  printf "%s\n" "Error: Cloud cover minimum and maximum must be specified between 0 and 100" "Cloud cover minimum: $CCMIN, cloud cover maximum: $CCMAX" ""
  exit 1
  elif [ $CCMIN -gt $CCMAX ]; then
    printf "%s\n" "Error: Cloud cover minimum is larger than cloud cover maximum" "Cloud cover minimum: $CCMIN, cloud cover maximum: $CCMAX" ""
    exit 1
fi

# type of area of interest
# 1 - coordinates as text
# 2 - shapefile
# 3 - PathRow as text
if [ -f $AOI ]; then
  # check if AOI is GDAL readable file
  if ogrinfo $AOI >& /dev/null; then
    AOITYPE=2
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
else
  # tile list / bounding box is command line input
  AOI=$(echo $AOI | sed 's/,/ /g')
  OGR=0
fi

isinrange() {
  awk -v value="$1" -v lower="$2" -v upper="$3" 'BEGIN {print (lower <= value && value <= upper)}'
}
if [ $OGR -eq 0 ]; then
  # check if AOI file contains bounding box coordinates and check if coords are valid lat/lon
  if $(echo $AOI | grep -q "/"); then
    AOITYPE=1
    for COORD in $AOI; do
      LAT=$(echo COORD | cut -d"/" -f1)
      LON=$(echo COORD | cut -d"/" -f2)
      if ! grep -q "/" $COORD; then
        printf  "%s\n" "Error: At least one of the AOI coordinates does not seem to be in the format LAT/LON" "Coordinate: $COORD" ""
        exit 1
      elif ! [ $(isinrange $LAT -90 90) -eq 1 ]; then
        printf "%s\n" "Error: Latitude out of range" "Coordinate: $COORD - $LAT is not in range -90 to 90" ""
        exit 1
      elif ! [ $(isinrange $LON -180 180) -eq 1 ]; then
        printf "%s\n" "Error: Longitute out of range" "Coordinate: $COORD - $LON is not in range -180 to 180" ""
        exit 1
      fi
    done
  # else, AOI file must be tile list - check if tiles are formatted correctly
  else
    AOITYPE=3
    for ENTRY in $AOI
    do
      if $(echo $ENTRY | grep -q -E "[0-2][0-9]{2}[0-2][0-9]{2}"); then
        LSPATH="${ENTRY:0:3}"
        LSROW="${ENTRY:3:6}"
        if [ $(isinrange $LSPATH 1 233) -eq 0 ] || [ $(isinrange $LSPATH 1 248) -eq 0 ]; then
          printf "%s\n" "Landsat PATH / ROW out of range. PATH not in range 1 to 233 or ROW not in range 1 to 248." "PATH / ROW received: $ENTRY" ""
          exit 1
        fi
        continue
      elif $(echo $ENTRY | grep -q -E "T[0-6][0-9][A-Z]{3}"); then
        if ! [ $(isinrange ${ENTRY:2:3} 1 60) ]; then
          printf "%s\n" "MGRS tile number out of range. Valid range: 0 to 60, received: $ENTRY" ""
          exit 1
        elif [[ -z "$(echo ${ENTRY:3:1} | grep -E "[C,D,E,F,G,H,J,K,L,M,N,P,Q,R,S,T,U,V,W,X]")" || -z "$(echo ${ENTRY:4:1} | grep -E "[A,B,C,D,E,F,G,H,K,L,M,N,P,Q,R,T,U,V,W,X,Y,Z]")" || -z "$(echo ${ENTRY:5:1} | grep -E "[A,B,C,D,E,F,G,H,J,K,L,M,N,P,Q,R,S,T,U,V]")" ]]; then
          echo "$(echo ${ENTRY:5:1} | grep -E "[A,B,C,D,E,F,G,H,K,L,M,N,P,Q,R,T,U,V,W,X,Y,Z]")"
          printf "%s\n" "Tile does not seem to be a valid Sentinel-2 tile: $ENTRY" "Please make sure all tiles exist."
          exit 1
        fi
        continue
      else
        printf "%s\n" "Tile list as AOI detected." "Error: One or more tiles seem to be formatted incorrectly." "Please check $ENTRY" ""
        exit 1
      fi
    done
  fi
fi


# ============================================================
# Get tiles / footprints of interest
if [ "$AOITYPE" -eq 1 ] || [ "$AOITYPE" -eq 2 ]; then
  if ! [ -x "$(command -v ogr2ogr)" ]; then
    printf "%s\n" "Could not find ogr2ogr, is gdal installed?" "Define the AOI polygon using coordinates (option 3) if gdal is not available." >&2
    exit 1
  fi
fi


# ============================================================
# Function get_data:
# 1. Prepare request
# 2. Query metadata catalogue
# 3. Download data
get_data() {
  SATELLITE=$1
  PRINTNAME=$2
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
    printf "%s\n" "" "Searching for footprints / tiles intersecting with input geometry..."
    WKT=$(echo $AOI | sed 's/,/%20/g; s/\//,/g')
    WFSURL="http://ows.geo.hu-berlin.de/cgi-bin/qgis_mapserv.fcgi?MAP=/owsprojects/grids.qgs&SERVICE=WFS&REQUEST=GetFeature&typename="$SATELLITE"&Filter=%3Cogc:Filter%3E%3Cogc:Intersects%3E%3Cogc:PropertyName%3Eshape%3C/ogc:PropertyName%3E%3Cgml:Polygon%20srsName=%22EPSG:4326%22%3E%3Cgml:outerBoundaryIs%3E%3Cgml:LinearRing%3E%3Cgml:coordinates%3E"$WKT"%3C/gml:coordinates%3E%3C/gml:LinearRing%3E%3C/gml:outerBoundaryIs%3E%3C/gml:Polygon%3E%3C/ogc:Intersects%3E%3C/ogc:Filter%3E"
    TILERAW=$(ogr2ogr -f CSV /vsistdout/ -select "Name" WFS:"$WFSURL")
    TILES="_"$(echo $TILERAW | sed 's/Name, //; s/ /_|_/g')"_"
    # case $SATELLITE in
    #   sentinel2) TILES="_T"$(echo $TILERAW | sed 's/Name, //; s/ /_|_T/g')"_" ;;
    #   landsat) TILES="_"$(echo $TILERAW | sed 's/Name, //; s/ /_|_/g')"_" ;;
    # esac

  elif [ "$AOITYPE" -eq 2 ]; then
    printf "%s\n" "" "Searching for footprints / tiles intersecting with geometries of AOI shapefile..."
    AOINE=$(echo $(basename "$AOI") | rev | cut -d"." -f 2- | rev)
    BBOX=$(ogrinfo -so $AOI $AOINE | grep "Extent: " | sed 's/Extent: //; s/(//g; s/)//g; s/, /,/g; s/ - /,/')
    WFSURL="http://ows.geo.hu-berlin.de/cgi-bin/qgis_mapserv.fcgi?MAP=/owsprojects/grids.qgs&SERVICE=WFS&REQUEST=GetCapabilities&typename="$SATELLITE"&bbox="$BBOX

    ogr2ogr -f "GPKG" merged.gpkg WFS:"$WFSURL" -append -update
    ogr2ogr -f "GPKG" merged.gpkg $AOI -append -update

    TILERAW=$(ogr2ogr -f CSV /vsistdout/ -dialect sqlite -sql "SELECT $SATELLITE.Name FROM $SATELLITE, $AOINE WHERE ST_Intersects($SATELLITE.geom, ST_Transform($AOINE.geom, 4326))" merged.gpkg)
    TILES="_"$(echo $TILERAW | sed 's/Name, //; s/ /_|_/g')"_"
    rm merged.gpkg

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
  if [ $SATELLITE == "landsat" ]; then
  printf "%s\n" "Tier(s): $TIER"
  fi
  printf "%s\n" "Tile(s): "$(echo $TILERAW | sed 's/Name, //; s/ /,/g') "Daterange: "$DATEMIN" to "$DATEMAX "Cloud cover minimum: "$CCMIN"%, maximum: "$CCMAX"%" ""

  # ============================================================
  # Filter metadata and extract download links
  if [ $SATELLITE = "sentinel2" ]; then
    LINKS=$(grep -E $TILES $METACAT | grep -E $(echo ""$SENSORS"" | sed 's/ /_|/g')"_" | awk -F "," '{OFS=","} {gsub("T[0-9]{2}:[0-9]{2}:[0-9]{2}.[0-9]{6}Z|-","",$5)}1' | awk -v start=$DATEMIN -v stop=$DATEMAX -v clow=$CCMIN -v chigh=$CCMAX -F "," '{OFS=","} $5 >= start && $5 <= stop && $7 >= clow && $7 <= chigh')
  elif [ $SATELLITE = "landsat" ]; then
    LINKS=$(grep -E $TILES $METACAT | grep -E $(echo ""$SENSORS"" | sed 's/ /_|/g')"_" | grep -E $(echo "_"$TIER | sed 's/,/,|_/g')"," | awk -F "," '{OFS=","} {gsub("-","",$5)}1' | awk -v start=$DATEMIN -v stop=$DATEMAX -v clow=$CCMIN -v chigh=$CCMAX -F "," '$5 >= start && $5 <= stop && $6 == 01 && $12 >= clow && $12 <= chigh')
  fi

  printf "%s" "$LINKS" > filtered_metadata.txt
  case $SATELLITE in
    sentinel2) TOTALSIZE=$(printf "%s" "$LINKS" | awk -F "," '{s+=$6/1048576} END {printf "%f", s}') ;;
    landsat) TOTALSIZE=$(printf "%s" "$LINKS" | awk -F "," '{s+=$17/1048576} END {printf "%f", s}') ;;
  esac
  NSCENES=$(sed -n '$=' filtered_metadata.txt)
  #rm filtered_metadata.txt


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

  if [ -z $NSCENES ];then
    printf "%s\n" "There were no $PRINTNAME Level 1 scenes found matching the search criteria." ""
  else
    LC_NUMERIC="en_US.UTF-8" printf "%s\n%.2f%s\n" "$NSCENES $PRINTNAME Level 1 scenes matching criteria found" "$PRSIZE" "$UNIT data volume found."
  fi


  # ============================================================
  # Download scenes
  progress() {
    SIZEDONE=$(awk -v done=$SIZEDONE -v fsize=$FILESIZE 'BEGIN { print (done + fsize) }' )
    PERCDONE=$(awk -v total=$TOTALSIZE -v done=$SIZEDONE 'BEGIN { printf( "%.2f\n", (100 / total * done) )}')
    local WIDTH=$(($(tput cols) - 9)) PERCINT=$(( $(echo $PERCDONE | cut -d"." -f1) + 1 ))
    printf -v INCREMENT "%*s" "$(( $PERCINT*$WIDTH/100 ))" ""; INCREMENT=${INCREMENT// /=}
    printf "\r\e[K|%-*s| %3d %% %s" "$WIDTH" "$INCREMENT" "$PERCINT" "$*"
    }

  PERCDONE=0
  SIZEDONE=0
  if [[ $DRYRUN -eq 0 && ! -z $LINKS ]]; then

    POOL=$(cd $POOL; pwd)
    printf "%s\n" "" "Starting to download "$NSCENES" "$PRINTNAME" Level 1 scenes" "" "" "" "" ""
    
    ITER=1
    for LINK in $LINKS
    do
      SCENEID=$(echo $LINK | cut -d, -f 2)

      if [ $SATELLITE = "sentinel2" ]; then
        TILE=$(echo $LINK | cut -d, -f 1 | grep -o -E "T[0-9]{2}[A-Z]{3}")
        URL=$(echo $LINK | cut -d, -f 14)
        FILESIZE=$(( $(echo $LINK | cut -d, -f 6) / 1048576 ))
      elif [ $SATELLITE = "landsat" ]; then
        TILE=$(echo $SCENEID | cut -d_ -f 3)
        URL=$(echo $LINK | cut -d, -f 18)
        FILESIZE=$(( $(echo $LINK | cut -d, -f 17) / 1048576 ))
      fi

      SCENEPATH=$TILEPATH/$SCENEID
      if [ $SATELLITE = "sentinel2" ]; then
        SCENEPATH=$SCENEPATH".SAFE"
      fi
      # Check if scene already exists
      # Implement size check to catch broken downloads!
      if [ -d $SCENEPATH ]; then
        printf "\e[4A\e[100D\e[2KScene "$SCENEID"("$ITER" of "$NSCENES") exists, skipping...\e[4B"
        #dl_done 
        progress
        ((ITER++))
        continue
      fi
      
      # create target directory if it doesn't exist
      TILEPATH=$POOL/$TILE
      if [ ! -w $TILEPATH ]; then
        mkdir $TILEPATH
        if [ ! -w $TILEPATH ]; then
          echo "$TILEPATH: Creating directory failed."
          exit 1
        fi
      fi


      printf "\e[100D\e[2A\e[2KDownloading "$SCENEID"("$ITER" of "$NSCENES")...\e[2B"
      gsutil -m -q cp -c -L $POOL"/download_log.txt" -R $URL $TILEPATH

      lockfile-create $QUEUE
      echo "$SCENEPATH QUEUED" >> $QUEUE
      lockfile-remove $QUEUE

      #dl_done
      progress 
      ((ITER++))
    done
  fi
}

if [[ $LANDSAT -eq 1 && $SENTINEL -eq 1 ]]; then
  printf "%s\n" "" "Landsat and Sentinel-2 data requested." "Landsat data will be queried and downloaded first."
fi
if [ $LANDSAT -eq 1 ]; then
  get_data landsat Landsat
fi
if [ $SENTINEL -eq 1 ]; then
  get_data sentinel2 Sentinel-2
fi

printf "%s\n" "" "Done." ""
exit 0
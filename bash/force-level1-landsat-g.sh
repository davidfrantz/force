#!/bin/bash

# ======================================================================================
# Name: LS_queryAndDownload_gsutil.sh
# Author: Stefan Ernst
# Date: 2020-06-20
# Last change: 2020-08-10
# Desc: Query and download the public Google Cloud Storage Sentinel-2 archive.
#       Requirements:
#		1. Google Sentinel-2 metadata catalogue:
#		   https://console.cloud.google.com/storage/browser/gcp-public-data-landsat
# 		2. shapefile containing the Landsat WRS-2 descending orbits:
#		   https://www.usgs.gov/media/files/landsat-wrs-2-descending-path-row-shapefile
#		3. gsutil - available through pip and conda
#          Run the command 'gsutil config' after installation to set up authorization
#		   with your Google account.
#		4. gdal - specify the AOI as path/row if gdal is not available
# ======================================================================================


trap "echo Exited!; exit;" SIGINT SIGTERM #make sure that CTRL-C stops the whole process

show_help() {
cat << EOF

Usage: `basename $0` [-d] [-u] metadata-dir level-1-datapool queue aoi
                   aoitype sensor starttime endtime min-cc max-cc

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
  (3) Path/Row of the Landsat footprints of interest: "PPPRRR,PPPRRR,PPPRRR"
  Make sure to keep leading zeros - correct: 181034, incorrect: 18134

  type of area of interest
  1 - coordinates as text
  2 - shapefile
  3 - PathRow as text
  
  sensor
  Landsat sensor identifier:
  LT05 - Landsat 5 TM
  LE07 - Landsat 7 ETM+
  LC08 - Landsat OLI

  starttime endtime
  Dates must be given as YYYY-MM-DD

  min-cc max-cc
  The cloud cover range must be specified in %

  -d dry will trigger a dry run that will only return the number of images
  and their total data volume
  
  -u will update the metadata catalogue (download and extract from GCS)
  only the metadata dir is required as argument when using this option
  
  -h|--help show this help
  
EOF
}


update_meta() {
  echo "Updating metadata catalogue..."
  gsutil -m cp gs://gcp-public-data-landsat/index.csv.gz $METADIR
  gunzip $METADIR/index.csv.gz
  mv $METADIR/index.csv $METADIR/metadata_LS.csv
}


# ============================================================
# check for options 
DRYRUN=0
while :; do
  case $1 in
    -d) DRYRUN=1 ;;
	-h|-\?|--help) show_help
	   exit 0 ;;
    -u) METADIR=$2
	    if [ $# -lt 2 ]; then
	      echo "Metadata directory not specified, exiting"
		  exit 1
		 elif [ $# -gt 2 ]; then
		  echo "Error: Please only specify the metadata directory when using the update option (-u)"
		  exit 1
		elif ! [ -w $METADIR ]; then
		  echo "Can not write to metadata directory, exiting"
		  exit 1
		fi
	  update_meta
	  echo "Done. You can run this script without option -d to download data now." 
	  exit ;;
    -?*) printf "%s\n" "" "Incorrect option specified" ""
	   show_help >&2
       exit 1 ;;
	*) break #no more options
  esac
  shift
done


# ============================================================
# if wrong number of input args and -u opt not set, stop
EXPECTED_ARGS=10
if [ $# -ne $EXPECTED_ARGS ]; then
  printf "%s\n" "" "Incorrect number of input arguments provided"
  show_help
  exit
fi

METADIR=$1
POOL=$2
QUEUE=$3
AOI=$4
AOITYPE=$5
SENSIN=$6
DATEMIN=$7
DATEMAX=$8
CCMIN=$9
CCMAX=${10}

METACAT=$METADIR"/metadata_LS.csv"


# ============================================================
# Check user input
for s in $(echo $SENSIN | sed 's/,/ /g')
do 
  case "$s" in
    LT05|LE07|LC08) continue ;;
    *) printf "%s\n" "" "$s is not a valid sensor type." "Valid Sensors: LT05, LE07, LC08" ""
	   exit ;;
  esac
done

if ! date -d $DATEMIN &> /dev/null; then
  printf "%s\n" "" "starttime ($DATEMIN) is not a valid date." "Make sure date is formatted as YYYY-MM-DD" ""
  exit 1
  elif ! date -d $DATEMAX &> /dev/null; then
    printf "%s\n" "" "endtime ($DATEMAX) is not a valid date." "Make sure date is formatted as YYYY-MM-DD" ""
	exit 1
fi


# ============================================================
# Check if metadata catalogue exists and is up to date
if ! [ -f $METACAT ]; then
  echo "Metadata catalogue does not exist."
  update_meta
fi

METADATE=$(date -d $(stat $METACAT | grep "Change: " | cut -d" " -f2) +%s)
if [ $(date -d $DATEMAX +%s) -gt $METADATE ]; then
  printf "%s\n" "" "WARNING: The selected time window exceeds the last update of the metadata catalogue" "Results may be incomplete, please consider updating the metadata catalogue using the -d option."
fi


# ============================================================
# Get path / rows of interest
if [ "$AOITYPE" -eq 2 ]; then
  if ! [  $(basename "$AOI" | cut -d"." -f 2-) == "shp" ]; then
    printf "%s\n" "" "WARNING: AOI does not seem to be a shapefile. Other filetypes supported by GDAL should work, but are untested."
  fi
fi  
if [ "$AOITYPE" -eq 1 ] || [ "$AOITYPE" -eq 2 ]; then
  if ! [ -x "$(command -v ogr2ogr)" ]; then
    printf "%s\n" "Could not find ogr2ogr, is gdal installed?" "Define the AOI polygon using coordinates (option 3) if gdal is not available." >&2
    exit 1
  fi
fi


if [ "$AOITYPE" -eq 1 ]; then
  
  WKT=$(echo $AOI | sed 's/,/%20/g; s/\//,/g')
  WFSURL="http://ows.geo.hu-berlin.de/cgi-bin/qgis_mapserv.fcgi?MAP=/owsprojects/grids.qgs&SERVICE=WFS&REQUEST=GetFeature&typename=landsat&Filter=%3Cogc:Filter%3E%3Cogc:Intersects%3E%3Cogc:PropertyName%3Eshape%3C/ogc:PropertyName%3E%3Cgml:Polygon%20srsName=%22EPSG:4326%22%3E%3Cgml:outerBoundaryIs%3E%3Cgml:LinearRing%3E%3Cgml:coordinates%3E"$WKT"%3C/gml:coordinates%3E%3C/gml:LinearRing%3E%3C/gml:outerBoundaryIs%3E%3C/gml:Polygon%3E%3C/ogc:Intersects%3E%3C/ogc:Filter%3E"
  PRRAW=$(ogr2ogr -f CSV /vsistdout/ -select "PR" WFS:"$WFSURL") 
  PR="_"$(echo $PRRAW | sed 's/PR, //; s/ /_|_/g')"_"
  
elif [ "$AOITYPE" -eq 2 ]; then
  
  printf "%s\n" "" "Searching for Landsat footprints intersecting with geometries of AOI shapefile..."
  AOINE=$(echo $(basename "$AOI") | rev | cut -d"." -f 2- | rev)
  BBOX=$(ogrinfo -so $AOI $AOINE | grep "Extent: " | sed 's/Extent: //; s/(//g; s/)//g; s/, /,/g; s/ - /,/')
  WFSURL="http://ows.geo.hu-berlin.de/cgi-bin/qgis_mapserv.fcgi?MAP=/owsprojects/grids.qgs&SERVICE=WFS&REQUEST=GetCapabilities&typename=landsat&bbox="$BBOX
  
  ogr2ogr -f "GPKG" merged.gpkg WFS:"$WFSURL" -append -update
  ogr2ogr -f "GPKG" merged.gpkg $AOI -append -update
  
  PRRAW=$(ogr2ogr -f CSV /vsistdout/ -dialect sqlite -sql "SELECT landsat.PR FROM landsat, $AOINE WHERE ST_Intersects(landsat.geom, ST_Transform($AOINE.geom, 4326))" merged.gpkg)  
  PR="_"$(echo $PRRAW | sed 's/PR, //; s/ /_|_/g')"_"
  rm merged.gpkg
  
elif [ "$AOITYPE" -eq 3 ]; then
  
  PRRAW=$AOI
  PR="_"$(echo $AOI | sed 's/,/_|_/g')"_"

else
  echo "  Error: Please specify aoitype as 1 for coordinates of a polygon, "
  echo "         2 for shapefile (point/polygon/line) or "
  echo "         3 for comma-separated PATHROW "
  exit 1
fi

SENSOR=$(echo "$SENSIN" | sed 's/,/_|/g')"_"


# ============================================================
# Filter metadata and extract download links
printf "%s\n" "" "Querying the metadata catalogue for" "Sensor(s): "$SENSIN "Path/Row: "$(echo $PR | sed 's/_//g; s/|/,/g') "Daterange: "$DATEMIN" to "$DATEMAX "Cloud cover minimum: "$CCMIN"%, maximum: "$CCMAX"%" ""

LINKS=$(grep -E $PR $METACAT | grep -E $SENSOR | awk -F "," '{OFS=","} {gsub("-","",$5)}1' | awk -v start=$DATEMIN -v stop=$DATEMAX -v clow=$CCMIN -v chigh=$CCMAX -F "," '$5 >= start && $5 <= stop && $6 == 01 && $7 == "T1" && $12 >= clow && $12 <= chigh')

printf "%s" "$LINKS" > LS_filtered_meta.txt
SIZE=$(printf "%s" "$LINKS" | awk -F "," '{s+=$17/1048576} END {printf "%f", s}')
#NSCENES=$(( $(printf "%s" "$LINKS" | wc -l | cut -d" " -f 1) + 1 ))
NSCENES=$(sed -n '$=' LS_filtered_meta.txt)
#rm LS_filtered_meta.txt

# ============================================================
# Get total number and size of scenes matching criteria
UNIT="MB"
if [ ${SIZE%%.*} -gt 1024 ]; then
  SIZE=$(echo $SIZE | awk '{print $1 / 1024}')
  UNIT="GB"
fi
if [ ${SIZE%%.*} -gt 1024 ]; then
  SIZE=$(echo $SIZE | awk '{print $1 / 1024}')
  UNIT="TB"
fi
if [ ${SIZE%%.*} -gt 1024 ]; then
  SIZE=$(echo $SIZE | awk '{print $1 / 1024}')
  UNIT="PB"
fi

if [ -z $NSCENES ];then
  printf "%s\n" "There were no Landsat Level 1 scenes found matching the search criteria" ""
  exit 0
else
  printf "%s\n" "$NSCENES Landsat Level 1 scenes matching criteria found" "$SIZE $UNIT data volume found" ""
fi

if [ $DRYRUN -eq 1 ]; then
  exit 0
fi  


# ============================================================
# Download scenes
echo "Starting to download "$NSCENES" Landsat Level 1 scenes"
ITER=1
for LINK in $LINKS
do
  SCENEID=$(echo $LINK | cut -d, -f 2)
  PR=$(echo $SCENEID | cut -d_ -f3)
  PRPATH=$POOL/$PR
  URL=$(echo $LINK | cut -d, -f 18)
  
  # create target directory if it doesn't exist
  if [ ! -w $PRPATH ]; then
    mkdir $PRPATH
    if [ ! -w $PRPATH ]; then
      echo "$PRPATH: Creating directory failed."
      exit 1
    fi
  fi
  ABSPRPATH=$(cd $POOL/$PR; pwd)
  
  # Check if scene already exists
  SCENEPATH=$ABSPRPATH/$SCENEID
  if [ -d $SCENEPATH ]; then
    echo "Scene "$SCENEID"("$ITER" of "$NSCENES") exists, skipping..."
	((ITER++))
    continue
  fi
  
  echo "Downloading "$SCENEID"("$ITER" of "$NSCENES")..."
  gsutil -m -q cp -c -L $POOL"/download_log.txt" -R $URL $ABSPRPATH
  
  echo "$SCENEPATH QUEUED" >> $QUEUE
  
  
  ((ITER++))
done

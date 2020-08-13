#!/bin/bash

# =====================================================================================
# Name: S2_queryAndDownload_gsutil.sh
# Author: Stefan Ernst
# Date: 2020-06-20
# Last change: 2020-08-11
# Desc: Query and download the public Google Cloud Storage Landsat archive.
#		Only Collection 1 Tier one products are considered.
#       Requirements:
#		1. Google Landsat metadata catalogue:
#		https://console.cloud.google.com/storage/browser/gcp-public-data-landsat
# 		2. shapefile containing the Landsat WRS-2 descending orbits:
#		https://www.usgs.gov/media/files/landsat-wrs-2-descending-path-row-shapefile
#		3. gsutil - available through pip and conda
#          Run the command 'gsutil config' after installation to set up authorization
#		   with your Google account.
#		4. gdal - specify the AOI as path/row if gdal is not available
# =====================================================================================


trap "echo Exited!; exit;" SIGINT SIGTERM #make sure that CTRL-C stops the whole process

show_help() {
cat << EOF

Usage: `basename $0` [-d] [-u] metadata-dir level-1-datapool queue aoi
                   aoitype sensor starttime endtime min-cc max-cc

  metadata-dir
  directory where the Sentinel-2 metadata (csv file) is stored
  
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
  gsutil -m cp gs://gcp-public-data-sentinel-2/index.csv.gz $METADIR
  gunzip $METADIR/index.csv.gz
  mv $METADIR/index.csv $METADIR/metadata_S2.csv
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
EXPECTED_ARGS=9
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
DATEMIN=$6
DATEMAX=$7
CCMIN=$8
CCMAX=$9

METACAT=$METADIR"/metadata_S2.csv"


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
# Get S2 MGRS tiles of interest
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
  WFSURL="http://ows.geo.hu-berlin.de/cgi-bin/qgis_mapserv.fcgi?MAP=/owsprojects/grids.qgs&SERVICE=WFS&REQUEST=GetFeature&typename=sentinel2&Filter=%3Cogc:Filter%3E%3Cogc:Intersects%3E%3Cogc:PropertyName%3Eshape%3C/ogc:PropertyName%3E%3Cgml:Polygon%20srsName=%22EPSG:4326%22%3E%3Cgml:outerBoundaryIs%3E%3Cgml:LinearRing%3E%3Cgml:coordinates%3E"$WKT"%3C/gml:coordinates%3E%3C/gml:LinearRing%3E%3C/gml:outerBoundaryIs%3E%3C/gml:Polygon%3E%3C/ogc:Intersects%3E%3C/ogc:Filter%3E"
  TILERAW=$(ogr2ogr -f CSV /vsistdout/ -select "Name" WFS:"$WFSURL") 
  TILES="_"$(echo $TILERAW | sed 's/Name, /T/; s/ /_|_T/g')"_"
  
elif [ "$AOITYPE" -eq 2 ]; then

  printf "%s\n" "" "Searching for S2 tiles intersecting with geometries of AOI shapefile..."
  AOINE=$(echo $(basename "$AOI") | rev | cut -d"." -f 2- | rev)
  BBOX=$(ogrinfo -so $AOI $AOINE | grep "Extent: " | sed 's/Extent: //; s/(//g; s/)//g; s/, /,/g; s/ - /,/')
  WFSURL="http://ows.geo.hu-berlin.de/cgi-bin/qgis_mapserv.fcgi?MAP=/owsprojects/grids.qgs&SERVICE=WFS&REQUEST=GetCapabilities&typename=sentinel2&bbox="$BBOX

  ogr2ogr -f "GPKG" merged.gpkg WFS:"$WFSURL" -append -update
  ogr2ogr -f "GPKG" merged.gpkg $AOI -append -update
  
  TILERAW=$(ogr2ogr -f CSV /vsistdout/ -dialect sqlite -sql "SELECT sentinel2.Name FROM sentinel2, $AOINE WHERE ST_Intersects(sentinel2.geom, ST_Transform($AOINE.geom, 4326))" merged.gpkg)  
  TILES="_"$(echo $TILERAW | sed 's/Name, /T/; s/ /_|_T/g')"_"
  rm merged.gpkg
    
elif [ "$AOITYPE" -eq 3 ]; then
  
  TILERAW=$AOI
  TILES="_T"$(echo $AOI | sed 's/,/_|_T/g')"_"

else
  echo "  Error: Please specify aoitype as 1 for coordinates of a polygon, "
  echo "         2 for shapefile (point/polygon/line) or "
  echo "         3 for comma-separated tile names "
  exit
fi


# ============================================================
# Filter metadata and extract download links
printf "%s\n" "" "Querying the metadata catalogue for" "Tile(s): "$(echo $TILERAW | sed 's/Name, //; s/ /,/g') "Daterange: "$DATEMIN" to "$DATEMAX "Cloud cover minimum: "$CCMIN"%, maximum: "$CCMAX"%" ""

LINKS=$(grep -E $TILES $METACAT | awk -F "," '{OFS=","} {gsub("T[0-9]{2}:[0-9]{2}:[0-9]{2}.[0-9]{6}Z|-","",$5)}1' | awk -v start=$DATEMIN -v stop=$DATEMAX -v clow=$CCMIN -v chigh=$CCMAX -F "," '{OFS=","} $5 >= start && $5 <= stop && $7 >= clow && $7 <= chigh')

printf "%s" "$LINKS" > S2_filtered_meta.txt
SIZE=$(printf "%s" "$LINKS" | awk -F "," '{s+=$6/1048576} END {printf "%f", s}')
NSCENES=$(sed -n '$=' S2_filtered_meta.txt)
rm S2_filtered_meta.txt


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
  printf "%s\n" "There were no Sentinel-2 Level 1 scenes found matching the search criteria" ""
  exit 0
else
  printf "%s\n" "$NSCENES Sentinel-2 Level 1 scenes matching criteria found" "$SIZE $UNIT data volume found" ""
fi

if [ $DRYRUN -eq 1 ]; then
  exit 0
fi  


# ============================================================
# Download scenes
echo "Starting to download "$NSCENES" Sentinel-2 Level 1 scenes"
ITER=1
for LINK in $LINKS
do
  SCENEID=$(echo $LINK | cut -d, -f 2)
  TILE=$(echo $LINK | cut -d, -f1 | grep -o -E "T[0-9]{2}[A-Z]{3}")
  TILEPATH=$POOL/$TILE
  URL=$(echo $LINK | cut -d, -f 14)
  
  # create target directory if it doesn't exist
  if [ ! -w $TILEPATH ]; then
    mkdir $TILEPATH
    if [ ! -w $TILEPATH ]; then
      echo "$TILEPATH: Creating directory failed."
      exit 1
    fi
  fi
  ABSTILEPATH=$(cd $POOL/$TILE; pwd)
  
  # Check if scene already exists
  SCENEPATH=$ABSTILEPATH/$SCENEID".SAFE"
  if [ -d $SCENEPATH ]; then
    echo "Scene "$SCENEID"("$ITER" of "$NSCENES") exists, skipping..."
	((ITER++))
    continue
  fi
  
  echo "Downloading "$SCENEID"("$ITER" of "$NSCENES")..."
  gsutil -m -q cp -c -L $POOL"/download_log.txt" -R $URL $ABSTILEPATH
  
  echo "$SCENEPATH QUEUED" >> $QUEUE
  
  
  ((ITER++))
done

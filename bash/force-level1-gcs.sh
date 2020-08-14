trap "echo Exited!; exit;" SIGINT SIGTERM # make sure that CTRL-C breaks out of download loop
set -e # make sure script exits if any process exits unsuccessfully

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
  
HELP
exit 1
}


update_meta() {
  echo "Updating metadata catalogue..."
  gsutil -m cp gs://gcp-public-data-$GCSNAME/index.csv.gz $METADIR
  gunzip $METADIR/index.csv.gz
  mv $METADIR/index.csv $METADIR/metadata_$SATELLITE.csv
}

# set variables for urls, file names, layer names, print, ...
case $PLATFORM in
  s2) 
    GCSNAME="sentinel-2" 
    SATELLITE="sentinel2"
    PRINTNAME="Sentinel-2" ;;
  ls) 
    GCSNAME="landsat"
    SATELLITE="landsat"
    PRINTNAME="Landsat" ;;
esac

while :; do
  case $1 in
    -d) 
      DRYRUN=1 ;;
    -h|-\?|--help) 
      show_help ;;
    -u) 
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
        update_meta
        echo "Done. You can run this script without option -u to download data now." 
        exit 
      fi ;;
    -?*) printf "%s\n" "" "Incorrect option specified" ""
     show_help >&2 ;;
  *) 
    break #no more options
  esac
  shift
done

if [ $# -ne 10 ]; then
  printf "%s\n" "" "Incorrect number of mandatory input arguments provided"
  show_help
fi

# ============================================================
# Check user input and set up variables
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

echo $PLATFORM

SENSIN=$(echo $SENSIN | tr '[:lower:]' '[:upper:]')  # convert sensor strings to upper case to prevent unnecessary headaches
case $SENSIN in
  S2A|S2A,S2B|S2B) 
    if [ $PLATFORM = "ls" ]; then
    print "%s\n" "Error: Sentinel-2 sensor names for Landsat query received" 
    show_help
    fi ;;
  LT05|LT05,LE07|LT05,LE07,LC08|LE07|LE07,LC08|LC08) 
    if [ $PLATFORM = "s2" ]; then
    printf "%s\n" "" "Error: Landsat sensor names for Sentinel-2 query received" 
    show_help
    fi ;;
  *) 
    printf "%s\n" "" "Error: invalid sensor or invalid combination of sensors speficied"
    show_help ;;
esac

if ! date -d $DATEMIN &> /dev/null; then
  printf "%s\n" "" "starttime ($DATEMIN) is not a valid date." "Make sure date is formatted as YYYY-MM-DD" ""
  exit 1
  elif ! date -d $DATEMAX &> /dev/null; then
    printf "%s\n" "" "endtime ($DATEMAX) is not a valid date." "Make sure date is formatted as YYYY-MM-DD" ""
  exit 1
fi


# ============================================================
# Check if metadata catalogue exists and is up to date
METACAT=$METADIR"/metadata_$SATELLITE.csv"
if ! [ -f $METACAT ]; then
   echo $METACAT
  printf "%s\n" "" "Metadata catalogue does not exist. Use the -u option to download / update the metadata catalogue" ""
  exit 1
fi

METADATE=$(date -d $(stat $METACAT | grep "Change: " | cut -d" " -f2) +%s)
if [ $(date -d $DATEMAX +%s) -gt $METADATE ]; then
  printf "%s\n" "" "WARNING: The selected time window exceeds the last update of the metadata catalogue" "Results may be incomplete, please consider updating the metadata catalogue using the -d option."
fi


# ============================================================
# Get tiles / footprints of interest
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
  WFSURL="http://ows.geo.hu-berlin.de/cgi-bin/qgis_mapserv.fcgi?MAP=/owsprojects/grids.qgs&SERVICE=WFS&REQUEST=GetFeature&typename="$SATELLITE"&Filter=%3Cogc:Filter%3E%3Cogc:Intersects%3E%3Cogc:PropertyName%3Eshape%3C/ogc:PropertyName%3E%3Cgml:Polygon%20srsName=%22EPSG:4326%22%3E%3Cgml:outerBoundaryIs%3E%3Cgml:LinearRing%3E%3Cgml:coordinates%3E"$WKT"%3C/gml:coordinates%3E%3C/gml:LinearRing%3E%3C/gml:outerBoundaryIs%3E%3C/gml:Polygon%3E%3C/ogc:Intersects%3E%3C/ogc:Filter%3E"
  TILERAW=$(ogr2ogr -f CSV /vsistdout/ -select "Name" WFS:"$WFSURL") 
  TILES="_"$(echo $TILERAW | sed 's/Name, /T/; s/ /_|_T/g')"_"
  
elif [ "$AOITYPE" -eq 2 ]; then

  printf "%s\n" "" "Searching for footprints / tiles intersecting with geometries of AOI shapefile..."
  AOINE=$(echo $(basename "$AOI") | rev | cut -d"." -f 2- | rev)
  BBOX=$(ogrinfo -so $AOI $AOINE | grep "Extent: " | sed 's/Extent: //; s/(//g; s/)//g; s/, /,/g; s/ - /,/')
  WFSURL="http://ows.geo.hu-berlin.de/cgi-bin/qgis_mapserv.fcgi?MAP=/owsprojects/grids.qgs&SERVICE=WFS&REQUEST=GetCapabilities&typename="$SATELLITE"&bbox="$BBOX

  ogr2ogr -f "GPKG" merged.gpkg WFS:"$WFSURL" -append -update
  ogr2ogr -f "GPKG" merged.gpkg $AOI -append -update
  
  TILERAW=$(ogr2ogr -f CSV /vsistdout/ -dialect sqlite -sql "SELECT $SATELLITE.Name FROM $SATELLITE, $AOINE WHERE ST_Intersects($SATELLITE.geom, ST_Transform($AOINE.geom, 4326))" merged.gpkg)  
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

if [ $PLATFORM = "s2" ]; then
  LINKS=$(grep -E $TILES $METACAT | grep -E $(echo $SENSIN | sed s'/,/|/g') | awk -F "," '{OFS=","} {gsub("T[0-9]{2}:[0-9]{2}:[0-9]{2}.[0-9]{6}Z|-","",$5)}1' | awk -v start=$DATEMIN -v stop=$DATEMAX -v clow=$CCMIN -v chigh=$CCMAX -F "," '{OFS=","} $5 >= start && $5 <= stop && $7 >= clow && $7 <= chigh')
elif [ $PLATFORM = "landsat" ]; then
  LINKS=$(grep -E $TILES $METACAT | grep -E $(echo "$SENSIN" | sed 's/,/_|/g')"_" | awk -F "," '{OFS=","} {gsub("-","",$5)}1' | awk -v start=$DATEMIN -v stop=$DATEMAX -v clow=$CCMIN -v chigh=$CCMAX -F "," '$5 >= start && $5 <= stop && $6 == 01 && $7 == "T1" && $12 >= clow && $12 <= chigh')
fi

printf "%s" "$LINKS" > filtered_metadata.txt
SIZE=$(printf "%s" "$LINKS" | awk -F "," '{s+=$6/1048576} END {printf "%f", s}')
NSCENES=$(sed -n '$=' filtered_metadata.txt)
rm filtered_metadata.txt


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
  printf "%s\n" "There were no $PRINTNAME Level 1 scenes found matching the search criteria" ""
  exit 0
else
  printf "%s\n" "$NSCENES $PRINTNAME Level 1 scenes matching criteria found" "$SIZE $UNIT data volume found" ""
fi

if [ $DRYRUN -eq 1 ]; then
  exit 0
fi  


# ============================================================
# Download scenes
POOL=$(cd $POOL; pwd)
echo "Starting to download "$NSCENES" "$PRINTNAME" Level 1 scenes"
ITER=1
for LINK in $LINKS
do
  SCENEID=$(echo $LINK | cut -d, -f 2)

  if [ $SATELLITE = "sentinel2" ]; then
    TILE=$(echo $LINK | cut -d, -f 1 | grep -o -E "T[0-9]{2}[A-Z]{3}")
    URL=$(echo $LINK | cut -d, -f 14)
  elif [ $SATELLITE = "landsat" ]; then
    TILE=$(echo $SCENEID | cut -d_ -f 3)
    URL=$(echo $LINK | cut -d, -f 18)
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
  
  # Check if scene already exists#
  SCENEPATH=$TILEPATH/$SCENEID
  if [ $SATELLITE = "sentinel2" ]; then
    SCENEPATH=$SCENEPATH".SAFE"
  fi
  if [ -d $SCENEPATH ]; then
    echo "Scene "$SCENEID"("$ITER" of "$NSCENES") exists, skipping..."
    ((ITER++))
    continue
  fi
  
  echo "Downloading "$SCENEID"("$ITER" of "$NSCENES")..."
  gsutil -m -q cp -c -L $POOL"/download_log.txt" -R $URL $TILEPATH
  
  echo "$SCENEPATH QUEUED" >> $QUEUE
  
  ((ITER++))
done

printf "%s\n" "" "Finished." ""
exit 0
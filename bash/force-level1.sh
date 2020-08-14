# ============================================================
# This script will act as 'master' for Level-1 download
# It calls the ESA or Google download script
#
# Check for satellite first:
#  - if ls, go for the google script
#  - if s2, check for -m option (esa, gcs, later also aws)
# 
#  - check if arguments are in right order, format for scripts
# 
# call landsat-level1-sentinel2 or landsat-level1-gcs

#set -e # makes sure that this script stops as soon as the sub scripts exit


# ============================================================
# check for options 
DRYRUN=0
PLATFORM=$1
# check for platform and mirror, discard platform ($1) and mirror ($2:-m and $3) afterwards
case $PLATFORM in
  s2)
    if ! [ $2 = "-m" ]; then
      printf "%s\n" "" "Mirror option (-m) must be set as first optional argument for Sentinel-2" "Valid mirrors: 'esa' for ESA and 'gcs' for Google Cloud Storage" ""
      exit 1
    else
      MIRROR=$3
      case $MIRROR in 
        "esa"|"gcs") 
          shift 2 ;;
        *) 
          printf "%s\n" "" "Mirror must be either esa (ESA archive) or gcs (Google Cloud Storage)" "" ;;
      esac
    fi ;;
  ls)
    MIRROR="gcs" ;;
  *)
    printf "%s\n" "" "Platform must be either ls (Landsat) or s2 (Sentinel-2)" ""
    exit 1 ;;
esac
shift

echo $(dirname $0)
# ============================================================
# run ESA or GCS scripts
BINDIR=$(dirname $0)
case $MIRROR in
  "esa")
    echo $@
    source $BINDIR"/"force-level1-esa $@ ;;

  "gcs")
    echo "$@" 
    source $BINDIR"/"force-level1-gcs $@ ;;
esac


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

# this script is an higher-level batch interface for the FORCE Level-2 processing system

# functions/definitions ------------------------------------------------------------------
PROG=$(basename "$0")
BIN="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
MISC="$BIN/force-misc"
export PROG BIN MISC

# source bash "library" file
LIB="$MISC/force-bash-library.sh"
if ! eval ". ${LIB}" >/dev/null 2>&1; then 
  echo "loading bash library failed" 1>&2
  exit 1
fi


MANDATORY_ARGS=1

export LOCK_CREATE_EXE="lockfile-create"
export LOCK_REMOVE_EXE="lockfile-remove"
export UNIX_EOL_EXE="dos2unix"
export FORCE_L2PS_CORE_EXE="$BIN/force-l2ps"
export PARALLEL_EXE="parallel"

help(){
cat <<HELP

Usage: $PROG [-hvi] parameter-file

  optional:
  -h = show this help
  -v = show version
  -i = show program's purpose

  mandatory:
  parameter-file = LEVEL 2 parameter file

  -----
    see https://force-eo.readthedocs.io/en/latest/components/lower-level/level2/level2.html

HELP
exit 1
}
export -f help

# important, check required commands !!! dies on missing
cmd_not_found "$LOCK_CREATE_EXE"
cmd_not_found "$LOCK_REMOVE_EXE"
cmd_not_found "$UNIX_EOL_EXE"
cmd_not_found "$FORCE_L2PS_CORE_EXE"
cmd_not_found "$PARALLEL_EXE"

function get_parameter(){

  PARAMETER=$1
  FILE=$2

  LINE=$(grep "^$PARAMETER" "$FILE")

  if [ -z "$LINE" ]; then
    echo "$PARAMETER is not given in parameter file"; 
    exit 1
  else
    VALUE=$(echo "$LINE" | sed 's/.* *= *//g')
    VALUE=$(echo "$VALUE" | sed 's/ *$//g')
  fi

  echo "$VALUE"

}
export -f get_parameter


function process_this_image(){

  # dummy test to enable linting through pipe
  if [ $# -eq 1 ] && [ "$1" == "TEST" ]; then
    return 0;
  fi

  FILE_IMAGE="$1"
  FILE_PRM="$2"
  DIR_LOG="$3"
  DIR_TEMP="$4"
  TIMEOUT_ZIP="$5"
  FILE_QUEUE="$6"

  debug "args: $FILE_IMAGE $FILE_PRM $DIR_LOG $DIR_TEMP $TIMEOUT_ZIP $FILE_QUEUE"

  # remove dangling white spaces to be sure
  FILE_IMAGE=${FILE_IMAGE// /}

  BASE_IMAGE=$(basename "$FILE_IMAGE")
  FILE_LOG="$DIR_LOG/$BASE_IMAGE.log"

  debug "FILE_IMAGE: $FILE_IMAGE"
  debug "BASE_IMAGE: $BASE_IMAGE"
  debug "FILE_LOG: $FILE_LOG"

  {
    echo "FORCE Level 2 Processing System"
    echo "-----------------------------------------------------------"
    echo ""
    echo "Start of processing: $(date +"%Y-%m-%d %H:%M:%S")"
    echo "Image: $FILE_IMAGE"
  } > "$FILE_LOG"


  # extract Landsat image (C1 tar.gz)
  if [[ $FILE_IMAGE == *".tar.gz" ]]; then

    unpacked=true
    file_not_found "$FILE_IMAGE" 2>> "$FILE_LOG"
    EXTRACT_IMAGE=$DIR_TEMP/${BASE_IMAGE//.tar.gz/}
    mkdir -p "$EXTRACT_IMAGE"
    dir_not_found "$EXTRACT_IMAGE" 2>> "$FILE_LOG"

    timeout -k "$TIMEOUT_ZIP" 10m \
      tar --ignore-command-error -xzf "$FILE_IMAGE" \
        --exclude gap_mask --exclude='*GCP.txt' --exclude='*VER.jpg' --exclude='*VER.txt' \
        --exclude='*BQA.TIF' --exclude='*.GTF' --exclude='LE07*B8.TIF' \
        --exclude='LE07*B6_VCID_2.TIF' --exclude='LC08*B11.TIF' --exclude='LC08*B8.TIF' \
        -C "$EXTRACT_IMAGE" &> /dev/null
    if [ ! $? -eq 0 ]; then
      UNPACK_STATUS="FAIL"
      echo "tar.gz container is corrupt, connection stalled or similar." >> "$FILE_LOG"
    else 
      UNPACK_STATUS="SUCCESS"
      echo "Unpacking tar.gz container successful" >> "$FILE_LOG"
    fi

  # extract Landsat image (C2 tar)
  elif [[ $FILE_IMAGE == *".tar" ]]; then

    unpacked=true
    file_not_found "$FILE_IMAGE" 2>> "$FILE_LOG"
    EXTRACT_IMAGE=$DIR_TEMP/${BASE_IMAGE//.tar/}
    mkdir -p "$EXTRACT_IMAGE"
    dir_not_found "$EXTRACT_IMAGE" 2>> "$FILE_LOG"

    timeout -k "$TIMEOUT_ZIP" 10m \
      tar --ignore-command-error -xf "$FILE_IMAGE" \
        --exclude gap_mask --exclude='*GCP.txt' --exclude='*VER.jpg' --exclude='*VER.txt' \
        --exclude='*BQA.TIF' --exclude='*.GTF' --exclude='LE07*B8.TIF' \
        --exclude='LE07*B6_VCID_2.TIF' --exclude='LC08*B11.TIF' --exclude='LC08*B8.TIF' \
        -C "$EXTRACT_IMAGE" &> /dev/null
    if [ ! $? -eq 0 ]; then
      UNPACK_STATUS="FAIL"
      echo "tar container is corrupt, connection stalled or similar." >> "$FILE_LOG"
    else 
      UNPACK_STATUS="SUCCESS"
      echo "Unpacking tar container successful" >> "$FILE_LOG"
    fi


  # extract Sentinel-2 image 
  elif [[ $FILE_IMAGE == *".zip" ]]; then

    unpacked=true
    file_not_found "$FILE_IMAGE" 2>> "$FILE_LOG"
    EXTRACT_IMAGE=$DIR_TEMP/${BASE_IMAGE//.zip/.SAFE}

    timeout -k "$TIMEOUT_ZIP" 10m unzip -qq -d "$DIR_TEMP" "$FILE_IMAGE" &>/dev/null
    if [ ! $? -eq 0 ]; then
      UNPACK_STATUS="FAIL"
      echo "zip container is corrupt, connection stalled or similar." >> "$FILE_LOG"
    else 
      UNPACK_STATUS="SUCCESS"
      echo "Unpacking zip container successful" >> "$FILE_LOG"
      dir_not_found "$EXTRACT_IMAGE" &>>"$FILE_LOG"
    fi

  # already unpacked
  else

    unpacked=false
    UNPACK_STATUS="SUCCESS"
    EXTRACT_IMAGE="$FILE_IMAGE"
    dir_not_found "$FILE_IMAGE" &>>"$FILE_LOG"
    echo "Image is already unpacked" >> "$FILE_LOG"

  fi

  debug "UNPACK_STATUS" "$UNPACK_STATUS"
  debug "unpacked: $unpacked"
  debug "EXTRACT_IMAGE: $EXTRACT_IMAGE"

  if [ "$UNPACK_STATUS" == "SUCCESS" ]; then

    {
      echo ""
      echo "Start core processing"
      echo "-----------------------------------------------------------"
      echo ""
    } >> "$FILE_LOG"

    # process
    if $FORCE_L2PS_CORE_EXE "$EXTRACT_IMAGE" "$FILE_PRM" &>> "$FILE_LOG"; then
      STATUS="DONE"
    else 
      STATUS="FAIL"
    fi

    debug "STATUS: $STATUS"

    {
      echo ""
      echo "-----------------------------------------------------------"
      echo "Core processing signaled $STATUS"
    }  >> "$FILE_LOG"

  else

    STATUS="FAIL"

  fi

  {
    echo ""
    echo "End of processing: $(date +"%Y-%m-%d %H:%M:%S")"
    echo "May the FORCE be with you!"
  }  >> "$FILE_LOG"


  # clean up temp file if previosuly extracted
  if is_true "$unpacked"; then
    rm -r "$EXTRACT_IMAGE"
  fi

  # update queue status; queue must be locked and unlocked
  $LOCK_CREATE_EXE "$FILE_QUEUE"
  sed -i.tmp -E "s+($FILE_IMAGE) .*+\1 $STATUS+" "$FILE_QUEUE"
  chmod --reference "$FILE_QUEUE.tmp" "$FILE_QUEUE"
  rm "$FILE_QUEUE.tmp"
  $LOCK_REMOVE_EXE "$FILE_QUEUE"

}
export -f process_this_image


# now get the options --------------------------------------------------------------------
ARGS=$(getopt -o hvi --long help,version,info -n "$0" -- "$@")
if [ $? != 0 ] ; then help; fi
eval set -- "$ARGS"

while :; do
  case "$1" in
    -h|--help) help ;;
    -v|--version) force_version; exit 0;;
    -i|--info) echo "Level 2 processing of image archive (bulk processing)"; exit 0;;
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
file_not_found "$FILE_PRM"
$UNIX_EOL_EXE -q "$FILE_PRM"


# main thing -----------------------------------------------------------------------------

TIME=$(date +"%Y%m%d%H%M%S")
debug "$TIME"

FILE_QUEUE=$(get_parameter "FILE_QUEUE" "$FILE_PRM")
debug "FILE_QUEUE: $FILE_QUEUE"
file_not_writeable "$FILE_QUEUE"

DIR_LEVEL2=$(get_parameter "DIR_LEVEL2" "$FILE_PRM")
debug "DIR_LEVEL2: $DIR_LEVEL2"
dir_not_writeable "$DIR_LEVEL2"

DIR_TEMP=$(get_parameter "DIR_TEMP" "$FILE_PRM")
debug "DIR_TEMP: $DIR_TEMP"
dir_not_writeable "$DIR_TEMP"

DIR_LOG=$(get_parameter "DIR_LOG" "$FILE_PRM")
debug "DIR_LOG: $DIR_LOG"
dir_not_writeable "$DIR_LOG"

DIR_PROVENANCE=$(get_parameter "DIR_PROVENANCE" "$FILE_PRM")
debug "DIR_PROVENANCE: $DIR_PROVENANCE"
dir_not_writeable "$DIR_PROVENANCE"

NPROC=$(get_parameter "NPROC" "$FILE_PRM")
debug "NPROC: $NPROC"
if is_lt "$NPROC" 1; then
  echoerr "NPROC needs to be >= 1"
  exit 1
fi

DELAY=$(get_parameter "DELAY" "$FILE_PRM")
debug "DELAY: $DELAY"
if is_lt "$DELAY" 0; then
  echoerr "DELAY needs to be >= 0"
  exit 1
fi

TIMEOUT_ZIP=$(get_parameter "TIMEOUT_ZIP" "$FILE_PRM")
debug "TIMEOUT_ZIP: $TIMEOUT_ZIP"
if is_lt "$TIMEOUT_ZIP" 0; then
  echoerr "TIMEOUT_ZIP needs to be >= 0"
  exit 1
fi


# do some cleaning in the queue
$UNIX_EOL_EXE -q "$FILE_QUEUE"
sed -i -E 's/\t/ /g' "$FILE_QUEUE"
sed -i -E 's/ +/ /g' "$FILE_QUEUE"
sed -i -E 's/^ //' "$FILE_QUEUE"
sed -i -E 's/ $//' "$FILE_QUEUE"

# get queued files
QUEUE=$(grep 'QUEUED' "$FILE_QUEUE" | sed 's/QUEUED//g')

# count input
NUM=$(echo "$QUEUE" | wc -w)
if is_eq "$NUM" 0; then
  echoerr "No images queued in " "$FILE_QUEUE"
  exit 1
else
  echo "$NUM images enqueued. Start processing with $NPROC NPROC"
fi

# test if full paths are given in "FILE_QUEUE"
for f in $QUEUE; do
  if [[ ! $f == /* ]]; then
    echoerr "relative pathnames are not allowed in FILE_QUEUE"
    exit 1
  fi
done


# test if force-l2ps was compiled in DEBUG mode
if ! $FORCE_L2PS_CORE_EXE -d; then
  echo "force-l2ps was compiled in DEBUG mode"
  if is_gt "$NUM" 1 || is_gt "$NPROC" 1; then
    echo "  cannot use DEBUG mode in force-level2"
    echo "  solutions:"
    echo "    (1) re-compile with disabled DEBUG mode or"
    echo "    (2) reduce NPROC to 1 AND reduce queue length in FILE_QUEUE to 1"
    exit 1
  fi
fi


FILE_NPROC="$DIR_TEMP/cpu-$TIME"
echo "$NPROC" > "$FILE_NPROC"

process_this_image "TEST" # dummy call to enable linting through pipe

echo "${QUEUE[*]}" | \
  $PARALLEL_EXE -j "$FILE_NPROC" --delay "$DELAY" --eta \
  process_this_image {} "$FILE_PRM" "$DIR_LOG" "$DIR_TEMP" "$TIMEOUT_ZIP" "$FILE_QUEUE"

rm "$FILE_NPROC"

exit 0

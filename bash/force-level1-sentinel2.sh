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

# this script downloads Sentinel-2 from ESA and maintains a clean Level-1 datapool

EXPECTED_ARGS=7
MAXIMUM_ARGS=8

# if wrong number of input args, stop
if [ $# -ne $EXPECTED_ARGS ] && [ $# -ne $MAXIMUM_ARGS ]; then
  echo ""
  echo "Usage: `basename $0` Level-1-Datapool queue Boundingbox"
  echo "                   starttime endtime min-cc max-cc [dry]"
  echo ""
  echo "  Level-1-Datapool"
  echo "  An existing directory, your files will be stored here"
  echo ""
  echo "  queue"
  echo "  Downloaded files are appended to a file queue, which is needed for"
  echo "  the Level 2 processing. The file doesn't need to exist. If it exists,"
  echo "  new lines will be appended on successful ingestion"
  echo ""
  echo "  Boundingbox"
  echo "  The coordinates of your study area: \"X1/Y1,X2/Y2,X3/Y3,...,X1/Y1\""
  echo "  The box must be closed (first X/Y = last X/Y). X/Y must be given as"
  echo "  decimal degrees with negative values for West and South coordinates."
  echo "  Note that the box doesn't have to be square, you can specify a polygon"
  echo ""
  echo "  starttime endtime"
  echo "  Dates must be given as YYYY-MM-DD"
  echo ""
  echo "  min-cc max-cc"
  echo "  The cloud cover range must be given in %"
  echo ""
  echo "  dry will trigger a dry run that will only return the number of images"
  echo "  and their total data volume"
  echo ""
  echo "  Your ESA credentials must be placed in \$HOME/.scihub"
  echo "  (OR in \$FORCE_CREDENTIALS/.scihub if the FORCE_CREDENTIALS environment"
  echo "   variable is defined)."
  echo "    First line: User name" 
  echo "    Second line: Password, special characters might be problematic"
  echo ""
  exit
fi

if [ $# -eq $MAXIMUM_ARGS ]; then
  if [ $8 == dry ]; then
    dryrun=1
  else
    echo "unknown option, optional argument 7 must be dry"
    exit
  fi
else
  dryrun=0
fi

POOL=$1
POOLLIST=$2
BOUND=$(echo $3 | sed 's_/_%20_g' | sed 's/ //g')
S0=$4
S1=$5
C0=$6
C1=$7

if [ ! -w $POOL ]; then
  echo "Level-1-Datapool must exist"
  exit
fi


if [ -z "$FORCE_CREDENTIALS" ]; then
  CREDDIR=$HOME/.scihub
else
  CREDDIR=$FORCE_CREDENTIALS/.scihub
fi


if [ ! -r $CREDDIR ]; then
  echo "Your ESA credentials were not found in $CREDDIR"
  echo "    First line: User name" 
  echo "    Second line: Password, special characters might be problematic"
  exit
fi

H=$(head -n 2 $CREDDIR)
USER=$(echo $H | cut -d ' ' -f 1)
PW=$(echo $H | cut -d ' ' -f 2)
CRED="--user=$USER --password=$PW"
HUB="https://scihub.copernicus.eu/dhus/search"


FNAME="S2?_MSIL1C*"

NMAX=100
FNAMEQ="filename:"$FNAME
BOUNDQ="footprint:\"Intersects(POLYGON(("$BOUND")))\""
DIRECQ="orbitdirection:Descending"
SENS="beginposition:["$S0"T00:00:00.000Z%20TO%20"$S1"T00:00:00.000Z]"
CC="cloudcoverpercentage:["$C0"%20TO%20"$C1"]"

QUERY="?q=$FNAMEQ%20AND%20$BOUNDQ%20AND%20$DIRECQ%20AND%20$SENS%20AND%20$CC&rows=$NMAX"

START=0
NUM=1

SIZE=0

while [ $START -lt $NUM ]; do

  # current time
  CTIME=$(date +"%Y-%m-%d_%H:%M:%S")
  LIST=$POOL"/query_"$C0"-"$C1"-"$START"_"$CTIME".html"

  QUERY="?q=$FNAMEQ%20AND%20$BOUNDQ%20AND%20$DIRECQ%20AND%20$SENS%20AND%20$CC&rows=$NMAX&start=$START"

  wget --no-check-certificate -q -O $LIST $CRED $HUB$QUERY#
  EXIT=$?

  # check exit code
  if [ $EXIT -ne 0 ]; then
    if [ $EXIT -eq 1 ]; then
      echo "Error. Unable to query Scihub. Generic error code."
    elif [ $EXIT -eq 2 ]; then
      echo "Error. Unable to query Scihub. Parse error."
    elif [ $EXIT -eq 3 ]; then
      echo "Error. Unable to query Scihub. File I/O error."
    elif [ $EXIT -eq 4 ]; then
      echo "Error. Unable to query Scihub. Network failure."
    elif [ $EXIT -eq 5 ]; then
      echo "Error. Unable to query Scihub. SSL verification failure."
    elif [ $EXIT -eq 6 ]; then
      echo "Error. Unable to query Scihub. Username/password authentication failure."
    elif [ $EXIT -eq 7 ]; then
      echo "Error. Unable to query Scihub. Protocol errors."
    elif [ $EXIT -eq 8 ]; then
      echo "Error. Unable to query Scihub. Server issued an error response."
    fi
    rm $LIST
    exit
  fi

  # test if query exists
  if [ ! -f $LIST ]; then
    echo "Error. Unable to query Scihub."
    exit
  fi

  NUM=$(grep 'totalResults'  $LIST | sed -r 's/.*>([0-9]*)<.*/\1/')
  TODO=$(($NUM-$START))
  if [ $TODO -gt 100 ]; then
    PAGE=100
  else
    PAGE=$TODO
  fi
  echo "$CTIME - Found $TODO S2A/B files. Downloading $PAGE files on this page."
  START=$(($START + $NMAX))

  SIZES=(`grep 'size' $LIST | sed 's/<[^<>]*>//g' | sed 's/[A-Z ]//g'`)
  UNITS=(`grep 'size' $LIST | sed 's/<[^<>]*>//g' | sed 's/[0-9. ]//g'`)
  
  for s in $(seq ${#SIZES[@]}); do
    if [ ! ${UNITS[$s]} == "MB" ]; then
      echo "warning: size not in MB. This script needs tuning"
    fi
    SIZE=$(echo $SIZE ${SIZES[$s]} | awk '{print $1 + $2}')
  done




  URL=($(grep '<link href="https:' $LIST | cut -d ' ' -f 2 | sed 's/href="//' | sed 's/"\/>//'))
  FNAMES=($(grep '\.SAFE' $LIST | cut -d '>' -f 2 | sed 's/SAFE.*/SAFE/'))
  FNAMEZ=($(grep '\.SAFE' $LIST | cut -d '>' -f 2 | sed 's/SAFE.*/zip/'))

  rm $LIST
  if [ -f $LIST ]; then
    echo "Warning. Unable to delete Scihub query."
  fi
  
  if [ $dryrun -eq 1 ]; then
    continue;
  fi
  
  #echo "$CTIME - Found ${#URL[*]} S2A/B files on this page."
  if [ ${#URL[*]} -eq 0 ]; then
    exit
  fi


  for i in $(seq 1 ${#URL[*]}); do

  #echo ${URL[$j]}

    j=$(($i-1))

    # get tile id and target directory
    TILE=$(echo ${FNAMES[$j]} | sed 's/.*_\(T[0-9]\{2\}[A-Z]\{3\}\)_.*/\1/')
    PPATH=$POOL/$TILE

    # create target directory if it doesn't exist
    if [ ! -w $PPATH ]; then
      mkdir $PPATH
      if [ ! -w $PPATH ]; then
        echo "$PPATH: Creating directory failed."
        exit
      fi
      #chmod 0755 $PPATH
    fi

    PNAMES=$PPATH/${FNAMES[$j]}
    PNAMEZ=$PPATH/${FNAMEZ[$j]}

    if [ -d $PNAMES ]; then
      # file already exists, do nothing
      #echo "${FNAMES[$j]}: File exists."
      continue
    else

      BASE=$(echo ${FNAMES[$j]} | sed 's/\(.*\)_N[0-9]\{4\}.*/\1/')
      #echo $BASE

      if [ -d $PPATH/$BASE* ]; then

        PNAME_POOL=$(ls -d $PPATH/$BASE*)
#      NPOOL=$(echo $PNAME_POOL | wc -w)

#      if [ $NPOOL -gt 1 ]; then
#        echo "should not happen."
#        continue
#      elif [ $NPOOL -eq 1 ]; then

        VERSION_HUB=$(echo ${FNAMES[$j]} | sed 's/.*_\(N[0-9]\{4\}\)_.*/\1/')
        VMAJOR_HUB=${VERSION_HUB:1:2}
        VMINOR_HUB=${VERSION_HUB:3:4}

        FNAME_POOL=$(basename $PNAME_POOL)

        VERSION_POOL=$(echo $FNAME_POOL | sed 's/.*_\(N[0-9]\{4\}\)_.*/\1/')
        VMAJOR_POOL=${VERSION_POOL:1:2}
        VMINOR_POOL=${VERSION_POOL:3:4}

        #echo $VERSION_HUB $VMAJOR_HUB $VMINOR_HUB
        #echo $VERSION_POOL $VMAJOR_POOL $VMINOR_POOL
        #echo $FNAME_POOL

        if [ $VMAJOR_HUB -lt $VMAJOR_POOL ]; then
          continue
        elif [ $VMAJOR_HUB -eq $VMAJOR_POOL ] && [ $VMINOR_HUB -le $VMINOR_POOL ]; then
          continue
        fi

        #echo "delete" $PNAME_POOL
        rm -r $PNAME_POOL
        if [ -d $PNAME_POOL ]; then
          echo "$FNAME_POOL: Could not update dataset."
          continue
        else
          echo "$FNAME_POOL: Removed dataset."
          sed -i.tmp "/$FNAME_POOL/d" $POOLLIST
          chmod --reference $POOLLIST".tmp" $POOLLIST
          rm $POOLLIST".tmp"
        fi

      fi

    fi

    # get HTTP response, and determine whether file was pulled from LTA, or is ready to download
    CTIME=$(date +"%Y%m%d%H%M%S")
    CHECK=$POOL"/LTA_CHECK_"$CTIME
    HTTP=$(wget --server-response --no-check-certificate -O $CHECK $CRED ${URL[$j]} 2>&1 | grep "HTTP/" | tail -n 1 | awk '{print $2}')
    rm $CHECK
    #HTTP=$(wget --spider --server-response --no-check-certificate $CRED ${URL[$j]} 2>&1 | grep "HTTP/" | tail -n 1 | awk '{print $2}')

    if [ $HTTP -eq 202 ]; then
      echo "${FNAMES[$j]}: Pulling from Long Term Archive. Success. Rerun this program after a while" 
      sleep 5
      continue
    elif [ $HTTP -eq 503 ]; then
      echo "${FNAMES[$j]}: Pulling from Long Term Archive. Failed. The LTA archive is busy. Rerun this program after a while" 
      sleep 5
      continue
    elif [ $HTTP -eq 403 ]; then
      echo "${FNAMES[$j]}: Pulling from Long Term Archive. Failed. You have exhausted your user quota. Rerun this program after a while" 
      sleep 5
      continue
    elif [ $HTTP -eq 500 ]; then
      echo "${FNAMES[$j]}: Pulling from Long Term Archive. Failed. Something is not right" 
      sleep 5
      continue
    elif [ $HTTP -eq 429 ]; then
      echo "${FNAMES[$j]}: Pulling from Long Term Archive. Failed. Too Many Requests" 
      sleep 5
      continue
    elif [ $HTTP -eq 200 ]; then
      wget -q --show-progress --no-check-certificate -O $PNAMEZ $CRED ${URL[$j]}
      EXIT=$?
    else 
      echo "${FNAMES[$j]}: Pulling from Long Term Archive. Failed. HTTP code" $HTTP
      sleep 5
      continue
    fi

    # check exit code
    if [ $EXIT -ne 0 ]; then
      if [ $EXIT -eq 1 ]; then
        echo "${FNAMES[$j]}: Generic error code."
      elif [ $EXIT -eq 2 ]; then
        echo "${FNAMES[$j]}: Parse error."
      elif [ $EXIT -eq 3 ]; then
        echo "${FNAMES[$j]}: File I/O error."
      elif [ $EXIT -eq 4 ]; then
        echo "${FNAMES[$j]}: Network failure."
      elif [ $EXIT -eq 5 ]; then
        echo "${FNAMES[$j]}: SSL verification failure."
      elif [ $EXIT -eq 6 ]; then
        echo "${FNAMES[$j]}: Username/password authentication failure."
      elif [ $EXIT -eq 7 ]; then
        echo "${FNAMES[$j]}: Protocol errors."
      elif [ $EXIT -eq 8 ]; then
        echo "${FNAMES[$j]}: Server issued an error response."
      fi
      rm $PNAMEZ
      continue
    fi

    # to be sure that file exists
    if [ ! -f $PNAMEZ ]; then
      echo "${FNAMES[$j]}: Error. File not downloaded."
      continue
    fi


    # extract zip
    #SAFE=$PPATH/$(unzip -l -q $PNAMEZ | head -n 3 | tail -n 1 | sed 's/.* //')
    unzip -qq -d $PPATH $PNAMEZ 2>/dev/null

    # delete zip
    rm $PNAMEZ
    if [ -f $PNAMEZ ]; then
      echo "Warning. Unable to delete zip file."
    fi

    PNAMES=$(ls -d $PPATH/$BASE*.SAFE)

    #to be sure that extracted directory exists
    if [ ! -d $PNAMES ]; then
      echo "$FNAMES: Extracting zip failed."
      exit
    fi


    # protect files
    #find $PNAMES -type f -exec chmod 0644 {} \;
    #find $PNAMES -type d -exec chmod 0755 {} \;

    #TILE=$(ls -d $PNAMES/GRANULE/*)

    echo "$PNAMES QUEUED" >> $POOLLIST

  done

done


if [ $dryrun -eq 1 ]; then

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
    
  echo $NUM "Sentinel-2 A/B L1C files available"
  echo $SIZE $UNIT "data volume available"

fi
   

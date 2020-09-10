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

# This script is intended for importing new files into the Landsat Level-1 data pool.

# (1) The input directory is scanned for *.tar.gz files.
# (2) The Path/Row ID is extracted from the file.
# (3) A subdirectory for the PR ID in the target directory is created if necessary
# (4) The *.tar.gz files are moved to the data pool.

# (.) Files are not imported, if they are duplicates of existing files.
# (.) Existing files are replaced by files with a newer production version.

# IMPORTANT NOTE: The input directory should not be a parent of the target directory.
#  The input directory is scanned recursively, thus files already in the target directory
#  will also be moved if target is a child of input.

EXP_ARGS=4
MAX_ARGS=5

printf "%s\n" "" "This tool is deprecated and will only receive minimal support in the future." "Please consider using force-level1-csd instead" ""

if [ $# -ne $EXP_ARGS ] && [ $# -ne $MAX_ARGS ]; then
  echo "Usage: `basename $0` from to queue cp/mv [dry]"
  echo ""
  exit
fi


# Copy or move files?
if [ $4 == cp ]; then
  MV=0
elif [ $4 == mv ]; then
  MV=1
else
  echo "unknown option, argument 3 must be cp or mv"
  exit
fi

if [ $# -eq $MAX_ARGS ]; then
  if [ $5 == dry ]; then
    dryrun=1
  else
    echo "unknown option, optional argument 4 must be dry"
    exit
  fi
else
  dryrun=0
fi

FROM=$1
TO=$2
POOLLIST=$3

# test input/output directories

if [[ $FROM == *" "* ]]; then
  echo "white spaces are not allowed in" $FROM
  exit
fi
if [[ $TO == *" "* ]]; then
  echo "white spaces are not allowed in" $TO
  exit
fi

if [ ! -d $FROM ]; then
  echo $FROM "is not an existing  directory."
  exit
fi
if [ ! -r $FROM ]; then
  echo $FROM "must be readable."
  exit
fi
if [ ! -w $FROM ] && [ $MV -eq 1 ]; then
  echo $FROM "must be writeable or cp mode must be used."
  exit
fi

if [ ! -d $TO ]; then
  echo $TO "is not an existing  directory."
  exit
fi
if [ ! -r $TO ]; then
  echo $TO "must be readable."
  exit
fi
if [ ! -w $TO ]; then
  echo $TO "must be writeable"
  exit
fi


# test if there are any *.tar.gz files
NUM=$(find $FROM -type f -name '*.tar.gz' | wc -l)
if [ $NUM -eq 0 ]; then
  echo "No *.tar.gz files found in " $FROM
  echo "Check read permissions if this is not correct."
  exit
else
  echo $NUM "*.tar.gz file(s) found. Starting import."
fi


# test for read/write permission
while IFS= read -r -d '' i; do
  if [ ! -r "$i" ]; then
    echo "All *.tar.gz files must be readable."
    echo "First occurence:"
    echo $(ls -l "$i")
    exit
  fi
  if [ $MV -eq 1 ]; then
    d=$(dirname "$i")
    if [ ! -w "$d" ]; then
      echo "The parent directories must be writeable or cp mode must be used."
      echo "First occurence:"
      echo $(ls -dl "$d")
      exit
    fi
  fi
done < <(find $FROM -type f -name '*.tar.gz' -print0)


# delete *.tar.gz with ([1-9]) suffix
NUM=$(find $FROM -type f -name '*([1-9])*' | wc -l)
if [ $NUM -ne 0 ]; then

  echo $NUM "*.tar.gz have 'duplicate status' tag in " $FROM
  echo "These files must be removed."

  # test if they can be deleted
  find $FROM -type f -name '*([1-9])*' | while read i; do
    if [ ! -w "$i" ]; then
      echo "The files must be writeable to remove them automatically."
      echo "Grant write permissions or delete manually."
      echo "First occurence:"
      echo $(ls -l "$i")
      exit
    fi
  done

  # ask if they should be deleted automatically
  read -p "Delete these files and go ahead? [y/n]: " -r -t 30
  if [ $? -gt 128 ];  then
    echo
    echo "Timeout. Try Again."
    exit
  fi

  # if yes, delete them; if not, abort
  if [[ $REPLY =~ ^[Yy]$ ]];  then
    find $FROM -type f -name '*([1-9])*' -exec rm {} \;
  else
    echo "Aborted by user."
    exit
  fi

fi



n_update=0
n_added=0
n_dupe=0


# for every *.tar.gz file
while IFS= read -r -d '' i; do

  base_new=$(basename "$i") # basename of new file
  core_new=${base_new%%.*}  # basename of new file without extension

  # pre-collection or collection >= 1
  if [[ $core_new == *"_"* ]]; then
    collection=1
  else
    collection=0
    tier="?"
  fi

  if [ $collection -eq 0 ]; then
    sensor=${base_new:1:1}
    satellite_1d=${base_new:2:1}
    satellite_2d=$(printf "%02d" $satellite_1d)
    pr=${base_new:3:6}       # path/row
    ddoy=${base_new:9:7}
    dyy=${ddoy:0:4}
    ddy=${ddoy:4:3}
    ddat=$(date -d "$dyy-01-01 +12 hours +$ddy days -1 day" "+%Y%m%d")
    body_p_new=${base_new:0:16} # basename of new file without extension and production version
    body_c_new="L"$sensor$satellite_2d"_*_"$pr"_"$ddat
  else
    sensor=${base_new:1:1}
    satellite_2d=${base_new:2:2}
    satellite_1d=$(echo $satellite_2d | sed 's/^0//') # remove trailing 0
    pr=${base_new:10:6}       # path/row
    ddat=${base_new:17:8}
    dyy=${ddat:0:4}
    dmm=${ddat:4:2}
    ddd=${ddat:6:2}
    ddoy=$(date -d "$dyy-$dmm-$ddd +12 hours" "+%Y%j")
    collection=${base_new:35:2}                   # update exact collection
    collection=$(echo $collection | sed 's/^0//') # remove trailing 0
    tier=${base_new:39:1}                   # tier level
    if [[ $tier == "T" ]]; then
      tier=3 # NRT tier
    fi
    body_p_new="L"$sensor$satellite_1d$pr$ddoy
    body_c_new=${base_new:0:25}
    body_c_new=$(echo $body_c_new | sed 's/_L1[A-Z][A-Z]_/_\*_/')
  fi

  echo "Landsat-"$satellite_2d"-"$sensor $ddat"/"$ddoy "P/R" $pr "collection " $collection "tier" $tier


  od=$TO/$pr         # target directory

  # create target directory if it doesn't exist
  if [ ! -d $od ]; then
    mkdir $od
    if [ ! -d $od ]; then
      echo "creating directory failed"
      exit
    fi
  fi


  DUPE=0
  IMPORT=0

  # if a pre-collection version is there
  if ls $od/$body_p_new* &> /dev/null; then

    DUPE=1

    old=$(ls $od/$body_p_new*) # filename of the existing file
    base_old=$(basename $old) # basename of the existing file

    echo "  pre-collection image is in Level-1 datapool"

    # if the next test is true, the data pool is somehow corrupt.
    # there shouldn't be more than one image of the same Landsat ID
    # several production versions are not allowed!!!
    if [ $(echo $old | wc -l) -gt 1 ]; then
      echo "Your data-pool is corrupt..."
      echo "there shouldn't be several processsing stages"
      echo "of the same file in the data-pool!"
      echo "Clean the data-pool and try again."
      exit
    fi

    if [ $collection -eq 0 ]; then

      num_old=${base_old:19:2}  # version number of the existing file
      num_new=${core_new:19:2}  # version number of the new file
      num_old=$(echo $num_old | sed 's/^0//') # remove trailing 0
      num_new=$(echo $num_new | sed 's/^0//') # remove trailing 0
      #echo $num_old $num_new

      if [ $num_new -gt $num_old ]; then
        IMPORT=1
        echo "  newer processing version" $num_old"->"$num_new "(collection" $collection") is imported"
      else 
        IMPORT=0
        echo "  older/same processing version" $num_old"->"$num_new "(collection" $collection") is skipped"
      fi

    else
      IMPORT=1
      echo "  newer collection 0->"$collection "is imported"
    fi

  fi

  # if a collection version is there
  if ls $od/$body_c_new* &> /dev/null; then

    DUPE=1

    old=$(ls $od/$body_c_new*) # filename of the existing file
    base_old=$(basename $old) # basename of the existing file

    echo "  collection image is in Level-1 datapool"

    # if the next test is true, the data pool is somehow corrupt.
    # there shouldn't be more than one image of the same Landsat ID
    # several production versions are not allowed!!!
    if [ $(echo $old | wc -l) -gt 1 ]; then
      echo "Your data-pool is corrupt..."
      echo "there shouldn't be several processsing stages/tiers/collections"
      echo "of the same file in the data-pool!"
      echo "Clean the data-pool and try again."
      exit
    fi

    if [ $collection -eq 0 ]; then
      IMPORT=0
      echo "  older collection ?->"$collection "is skipped"
    else
      collection_old=${base_old:35:2}
      collection_old=$(echo $collection_old | sed 's/^0//') # remove trailing 0
      tier_old=${base_old:39:1}                   # tier level
      if [[ $tier_old == "T" ]]; then
        tier_old=3 # NRT tier
      fi
      if [ $collection -gt $collection_old ]; then
        IMPORT=1
        echo "  newer collection" $collection_old"->"$collection "is imported"
      elif [ $tier -lt $tier_old ]; then
        IMPORT=1
        echo "  better tier" $tier_old"->"$tier "is imported"
      else
        IMPORT=0
        echo "  older/same collection" $collection_old"->"$collection "or tier" $tier_old"->"$tier "is skipped"
      fi
    fi

  fi


  if [ $DUPE -eq 1 ]; then

    if [ $IMPORT -eq 1 ]; then
      ((n_update++))
      if [ $dryrun -eq 0 ]; then
        rm $old
        sed -i.tmp "/$base_old/d" $POOLLIST
        chmod --reference $POOLLIST".tmp" $POOLLIST
        rm $POOLLIST".tmp"
        if  [ $MV -eq 1 ]; then
          mv "$i" $od/$base_new
        else
          cp "$i" $od/$base_new
        fi
        echo "$od/$base_new QUEUED" >> $POOLLIST
      else
        echo "remove" $old and move/copy "$i" to $od/$base_new
      fi
    else
      ((n_dupe++))
      if [ $dryrun -eq 1 ]; then
        echo "$i" is a duplicate and stays where it is
      fi 
    fi

  # if no match, simply move it to the target directory
  elif [ $DUPE -eq 0 ]; then
    ((n_added++))
    echo "  imported"
    if [ $dryrun -eq 0 ]; then
      if [ $MV -eq 1 ]; then
        mv "$i" $od/$base_new
      else
        cp "$i" $od/$base_new
      fi
      echo "$od/$base_new QUEUED" >> $POOLLIST
      #chmod 0644 $od/$base_new
    else
      echo move/copy "$i" to $od/$base_new
    fi
  fi

done < <(find $FROM -type f -name '*.tar.gz' -print0)


# print import report
n_total=$(($n_added+$n_update))
if [ $n_total -gt 0 ]; then
  echo $n_total "files were imported into the data pool"
  if [ $n_added -gt 0 ]; then
    echo $n_added "files were new imports"
  fi
  if [ $n_update -gt 0 ]; then
    echo $n_update "files were replaced by a newer version"
  fi
else
  echo "no file was imported"
fi

if [ $n_dupe -gt 0 ]; then
  echo $n_dupe "files were not imported because they are duplicates"
  echo "you should consider to delete them"
fi


#!/bin/bash

make -j force-parameter

MOD=$(temp-bin/force-parameter -m | grep -v 'available modules' | sed 's/^ *//' | cut -d ':' -f 1)

for m in $MOD; do
  temp-bin/force-parameter docs/source/_static/parameter-files/parameter_$m.prm $m
done

exit

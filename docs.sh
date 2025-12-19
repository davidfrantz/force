#!/bin/bash

make -j force-parameter force-runtime-data misc

MOD=$(bin/force-parameter -m | grep -v 'available modules' | sed 's/^ *//' | cut -d ':' -f 1)

for m in $MOD; do
  bin/force-parameter "docs/source/_static/parameter-files/parameter_$m.prm" "$m"
done

bin/force-runtime-data -s > docs/source/_static/runtime-data/sensors.txt
bin/force-runtime-data -x > docs/source/_static/runtime-data/indices.txt

make clean

exit

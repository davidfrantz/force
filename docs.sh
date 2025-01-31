#!/bin/bash

make -j aux

MOD=$(bin/force-parameter -m | grep -v 'available modules' | sed 's/^ *//' | cut -d ':' -f 1)

for m in $MOD; do
  bin/force-parameter docs/source/_static/parameter-files/parameter_$m.prm $m
done

make clean

exit

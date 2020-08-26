.. _aux-magic-parameters:


force-magic-parameters
======================

You may find yourself in a situation, where you want to iterate over one or more parameters (in any FORCE module).
As an example: for training machine learning models, you probably have several feature sets, which you want to train and validate.
For this, you would need to generate one parameter file for each feature set.

``force-magic-parameters`` introduces a convenient, and more automatic way to accomplish this.
Based on a replacement variable defined in a main paramaterfile, multiple new parameterfiles are generated according to a vector that holds the replacement values.
If multiple replacement variables are defined, all possible combinations of the vector elements are generated.

Usage
^^^^^

.. code-block:: bash

  force-magic-parameters

  Usage: force-magic-parameters.sh [-h] [-c {all,paired}] parameter-file

    -h  = show his help
    -c  = combination type
          all:    all combinations (default)
          paired: pairwise combinations


* parameter-file

  | Any FORCE parameterfile can be used.
  
* combination type

  | If this argument is not given, we will use all combinations of all replacement vectors.
  | This is the same as ``-c all``.
  
  | If ``-c paired``, pairwise combinations are used.
  | In this case, the repplacement vectors must be of the same length.


Syntax
^^^^^^

* The replacement variables need to be defined at the top of a main parameterfile - before the starting line, e.g. ++PARAM_TRAIN_START++
* Multiple replacement variables can be defined
* Each replacement variable is defined in a separate line
* A variable is defined like this: ``%VAR%: 001 002 003 005 010 100``
  VAR can be any variable name.
  The values can be integers, text, filenames, etc.
* The replacement variable is used in the main body like this: ``FILE_FEATURES = /data/FEATURESET_{%VAR%}.txt``

Example
^^^^^^^

In force-train, let's assume we have 6 feature sets, i.e. 6 files.

- data/FEATURESET_001.txt
- data/FEATURESET_002.txt
- data/FEATURESET_003.txt
- data/FEATURESET_005.txt
- data/FEATURESET_010.txt
- data/FEATURESET_100.txt

We want to perform a Random Forest Classification, and want to test different tree sizes, e.g. 100, 250, 500 and 1000.

We additionally want to test different tree depths, e.g. 5, 10 and maximal depth (0).

Thus we define following in main.prm:

.. code-block:: bash

  %SET%:   001 002 003 005 010 100
  %NTREE%: 100 250 500 1000
  %DEPTH%: 5 10 0

  ++PARAM_TRAIN_START++
  
  FILE_FEATURES = /data/FEATURESET_{%SET%}.txt
  
  # other parameters omitted
  
  FILE_MODEL = /data/FEATURESET_{%SET%}_NTREE_{%NTREE%}_DEPTH_{%DEPTH%}.xml
  FILE_LOG   = /data/FEATURESET_{%SET%}_NTREE_{%NTREE%}_DEPTH_{%DEPTH%}.log

  # other parameters omitted

  RF_NTREE = {%NTREE%}
  RF_DT_MAXDEPTH = {%DEPTH%}
  
  # other parameters omitted
  
  ++PARAM_TRAIN_END++

Then, we use ``force-magic-parameters`` to replace the variables and generate all possible parameterfiles.

.. code-block:: bash

  force-magic-parameters main.prm

  3 replacement vectors detected
  72 parameter files were generated

72 new parameterfiles were generated (6*4*3 combinations).
You can now run these parameterfiles, either sequentially or parallely (if this makes sense).

.. code-block:: bash

  # example for sequential execution
  for p in *.prm; do force-train $p; done
  
  # example for parallel execution
  ls *.prm | parallel force-train {}


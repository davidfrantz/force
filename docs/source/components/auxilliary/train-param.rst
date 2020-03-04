.. _train-param:

Parameterization
================

A parameter file is mandatory for the Time Series Analysis submodule of FORCE HLPS.

The file extension is ‘.prm’.
All parameters must be given, even if they are not used.
All parameters follow common tag-value notation.
Rudimentary checks are performed by the software components using this file.

The ``++PARAM_TRAIN_START++`` and ``++PARAM_TRAIN_END++`` keywords enclose the parameter file.

The following parameter descriptions are a print-out of ``force-parameter``, which can generate an empty parameter file skeleton.


* Input

  * File that is holding the features for training (and probably validation).
    The file needs to be a table with features in columns, and samples in rows.
    Column delimiter is whitespace.
    The same number of features must be given for each sample.
    Do not include a header.
    The samples need to match the response file.

    | *Type:* full file path
    | ``FILE_FEATURES = NULL``
    
  * File that is holding the response for training (class labels or numeric values).
    The file needs to be a table with one column, and samples in rows.
    Do not include a header.
    The samples need to match the feature file.

    | *Type:* full file path
    | ``FILE_RESPONSE = NULL``

* Output

  * File for storing the Machine Learning model in xml format.
    This file will be overwritten if it exists.

    | *Type:* full file path
    | ``FILE_MODEL = NULL``
    
  * File for storing the logfile.
    This file will be overwritten if it exists.

    | *Type:* full file path
    | ``FILE_LOG = NULL``

* Training

  * This parameter specifies how many samples (in %) should be used for training the model.
    The other samples are left out, and used to validate the model.

    | *Type:* Float. Valid range: ]0,100]
    | ``PERCENT_TRAIN = 70``
    
  * This parameter specifies whether the samples should be randomly drawn (TRUE) or if the first n samples (FALSE) should be used for training.

    | *Type:* Logical. Valid values: {TRUE,FALSE}
    | ``RANDOM_SPLIT = TRUE``

  * Machine learning method.
    Currently implemented are Random Forest and Support Vector Machines, both in regression and classification flavors.

    | *Type:* Character. Valid values: {SVR,SVC,RFR,RFC}
    | ``ML_METHOD = RFC``

* Random Forest parameters

  * This block only applies if method is Random Forest

  * Maximum number of trees in the forest.
    If RF_OOB_ACCURACY is 0, all trees will be grown.
    If RF_OOB_ACCURACY is set, the algorithm won't grow additional trees if the accuracy is already met.
    If set to 0, additional trees are grown until RF_OOB_ACCURACY is met; note that this might never happen.
    RF_NTREE and RF_OOB_ACCURACY cannot both be 0.

    | *Type:* Integer. Valid range: [0,...
    | ``RF_NTREE = 500``
    
  * Required accuracy of the ensemble, measured as OOB error.
    See also RF_NTREE.

    | *Type:* Float. Valid range: [0,...
    | ``RF_OOB_ACCURACY = 0``
    
  * The number of randomly selected features at each tree node, which are used to find the best split.
    If set to 0, the square root of the number of all feature is used for the classification flavor; a third of the number of all feature is used for the regression flavor.

    | *Type:* Integer. Valid range: [0,...
    | ``RF_NFEATURE = 0``
    
  * This parameter indicates whether the variable importance should be computed.

    | *Type:* Logical. Valid values: {TRUE,FALSE}
    | ``RF_FEATURE_IMPORTANCE = TRUE``
    
  * If the number of samples in a node is less than this parameter then the node will not be split.
    If set to 0, it defaults to 1 for the classification flavor; 5 for the regression flavor.

    | *Type:* Integer. Valid range: [0,...
    | ``RF_DT_MINSAMPLE = 0``
    
  * The maximum possible depth of the tree.
    That is the training algorithms attempts to split a node while its depth is less than RF_DT_MAXDEPTH.
    The root node has zero depth.
    If set to 0, the maximum possible depth is used.

    | *Type:* Integer. Valid range: [0,...
    | ``RF_DT_MAXDEPTH = 0``
    
  * Termination criteria for regression trees.
    If all absolute differences between an estimated value in a node and values of train samples in this node are less than this parameter then the node will not be split further.

    | *Type:* Float. Valid range: [0.01,...
    | ``RF_DT_REG_ACCURACY = 0.01``

* Support Vector Machine parameters

  * This block only applies if method is Support Vector Machine

  * Maximum number of iterations for the iterative SVM training procedure which solves a partial case of constrained quadratic optimization problem.
    If SVM_ACCURACY is 0, all iterations will be used.
    If SVM_ACCURACY is set, the algorithm will stop if the accuracy is already met.
    If set to 0, additional iterations are computed until SVM_ACCURACY is met; note that this might never happen.
    SVM_MAXITER and SVM_ACCURACY cannot both be 0.

    | *Type:* Integer. Valid range: [0,...
    | ``SVM_MAXITER = 1000000``
    
  * Required accuracy of the optimization.
    See also SVM_MAXITER.

    | *Type:* Float. Valid range: [0,...
    | ``SVM_ACCURACY = 0.001``
     
  * Cross-validation parameter.
    The training set is divided into kFold subsets.
    One subset is used to test the model, the others form the train set.
    So, the SVM algorithm is executed kFold times.

    | *Type:* Float. Valid range: [1,...
    | ``SVM_KFOLD = 10``
    
  * Parameter ϵ of a SVM optimization problem.

    | *Type:* Float. Valid range: [0,...
    | ``SVM_P = 0``

  * Parameter C of a SVM optimization problem.
    This parameter expects three values which are used to perform a grid search, i.e. minimum value, maximum value, logarithmic step.

    | *Type:* Float list. Valid range: [0,...
    | ``SVM_C_GRID = 0.001 10000 1``
    
  * Parameter γ of a kernel function.
  * This parameter expects three values which are used to perform a grid search, i.e. minimum value, maximum value, logarithmic step.

    | *Type:* Float list. Valid range: [0,...
    | ``SVM_GAMMA_GRID = 0.000010 10000 10``


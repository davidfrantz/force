.. _aux-train:


force-train
===========

...

  * Class weights.
    This parameter only applies for the classification flavor.
    This parameter lets you define à priori class weights, which can be useful if the training data are inbalanced.
    This parameter can be set to a number of different values. 
    EQUALIZED gives the same weight to all classes (default).
    PROPORTIONAL gives a weight proportional to the class frequency.
    ANTIPROPORTIONAL gives a weight, which is inversely proportional to the class frequency.
    Alternatively, you can use custom weights, i.e. a vector of weights for each class in your response file.
    The weights must sum to one, and must be given in ascending order.

    | *Type:* Character / Float list. Valid values: {EQUALIZED,PROPORTIONAL,ANTIPROPORTIONAL} or ]0,1[
    | ``FEATURE_WEIGHTS = EQUALIZED``

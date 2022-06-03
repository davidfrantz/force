# How to add a new spectral index to the TSA submodule

This page documents the steps that need to be taken to implement a new spectral index in force-tsa.
Give it a try!

General remarks: please stick to the syntax and indendation etc. of the existing code.

1) create a new branch from develop

2) add a new enum

    In src/cross-level/enum-cl.h: add a new enum before(!) _IDX_LENGTH_.
    Use a 3-digit ID that has not been used before, e.g. _IDX_XXX_ (replace XXX).

    In src/cross-level/enum-cl.c: provide a dictionary item that can translate between the Index in the parameter file and the enum, e.g. { _IDX_XXX_, "MyNewIndex" }

    Make sure that the added entries in both files are in the same order.

3) define band dependencies

    In src/higher-level/param-hl.c, function check_bandlist():

    - Add a new case clause by referencing the enum
    - enable the verification vector v[] for required wavelengths - this is needed to pre-check whether the index can be computed with a given set of sensors
    - choose a 3-digit ID for the filename - this should be the same ID as used for the enum

4) provide citation

    In src/cross-level/cite-cl.h: add a new citation enum before(!) _CITE_LENGTH_.
    Use the same 3-digit ID as for the enum (if not already taken)

    In src/cross-level/cite-cl.c: provide a dictionary item that holds the citation and the enum, as well as a bool variable that is initialized to false, e.g. { _IDX_XXX_, "Author, A. (2042). Fancy New Index. Remote Sensing of Environment", false }. Stick to the syntax of the other entries. Make sure that the added entries in both files are in the same order.

4) implement index

    In src/higher-level/index-hl.c, function tsa_spectral_index():

    - Add a new case clause by referencing the enum
    - cite the paper by using the cite enum
    - use one of the provided function templates if your new index matches one of those (e.g. use index_differenced() for an NDVI-like index). If this is the case, you are done with coding and can skip the next step.
    - provide a new function. Try to make it as generic as possible to enable other indices that use the same formulation (but just different bands or factors). Try to understand how the other functions work before doing this.

5) test whether the code compiles

6) make a test run (check that scaling and quantization is OK!)

7) submit a pull request on Github

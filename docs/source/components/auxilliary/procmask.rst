.. _procmask:

force-procmask
==============

...

Usage
^^^^^

.. code-block:: bash
    
    force-procmask [-s] [-l] [-d] [-o] [-b] [-j] input-basename calc-expr

  optional:
  -s = pixel resolution of cubed data, defaults to 10
  -l = input-layer: band number in case of multi-band input rasters,
       defaults to 1
  -d = input directory: the datacube directory                                                            
       defaults to current directory                                                                      
      'datacube-definition.prj' needs to exist in there                                                   
  -o = output directory: the directory where you want to store the cubes                                  
       defaults to current directory                                                                      
  -b = basename of output file (without extension)                                                        
       defaults to the basename of the input-file,                                                        
       appended by '_procmask'
  -j = number of jobs, defaults to 'as many as possible'

  Positional arguments:
  - input-basename: basename of input data
  - calc-expr: Calculation in gdalnumeric syntax, e.g. 'A>2500'"
               The input variable is 'A'
               For details about GDAL expressions, see
               https://gdal.org/programs/gdal_calc.html
               
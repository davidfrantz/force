/**+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

This file is part of FORCE - Framework for Operational Radiometric 
Correction for Environmental monitoring.

Copyright (C) 2013-2020 David Frantz

FORCE is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

FORCE is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with FORCE.  If not, see <http://www.gnu.org/licenses/>.

+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/

/**+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
This file contains functions for spectral temporal metrics
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


#include "py-plugin-hl.h"

#include <Python.h>
#include <numpy/ndarrayobject.h>
#include <numpy/ndarraytypes.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

/** public functions
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/

void register_python(par_hl_t *phl){


  Py_Initialize();
  import_array();
  PyRun_SimpleString("import numpy as np");

  register_python_tsa(&phl->tsa.pyp);

  return;
}


void deregister_python(){


  Py_Finalize();

  return;
}



void register_python_tsa(par_pyp_t *pyp){
PyObject *main_module = NULL;
PyObject *main_dict = NULL;
PyObject *py_fun = NULL;
PyObject *py_register = NULL;
FILE *fpy = NULL;


  if (!pyp->opyp){
    pyp->nb = 1;
    return;
  }

  //Py_Initialize();
  //import_array();

  main_module = PyImport_AddModule("__main__");
  main_dict   = PyModule_GetDict(main_module);

  fpy = fopen(pyp->f_code, "r");
  PyRun_SimpleFile(fpy, pyp->f_code);

  py_fun = PyDict_GetItemString(main_dict, "force_register_hl_tsa_tsi");

  py_register = PyObject_CallFunctionObjArgs(py_fun, NULL);

  pyp->nb = (int)Py_SIZE(py_register);
  Py_DECREF(py_register);

  printf("%d\n", pyp->nb);

  fclose(fpy);
  //Py_Finalize();

  return;
}


/** This function connects the TSA module to plug'n'play python code
--- ts:     pointer to instantly useable TSA image arrays
--- mask:   mask image
--- nc:     number of cells
--- ni:     number of interpolation steps
--- nodata: nodata value
--- pyp:    python plugin parameters
+++ Return: SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int tsa_python_plugin(tsa_t *ts, small *mask_, int nc, int ni, short nodata, par_pyp_t *pyp){
int b, t, p;

FILE *fpy = NULL;

npy_intp n = ni;

PyObject *main_module = NULL;
PyObject *main_dict = NULL;
PyObject *py_fun = NULL;
PyObject *py_return = NULL;

PyObject *py_val = NULL;
PyArrayObject* py_tsi = NULL;
PyArrayObject* py_ce = NULL;
PyArrayObject* py_year = NULL;
PyArrayObject* py_month = NULL;
PyArrayObject* py_day = NULL;
int* tsi_ = NULL;
int* ce_ = NULL;
int* year_ = NULL;
int* month_ = NULL;
int* day_ = NULL;


  if (ts->pyp_ == NULL) return CANCEL;

  //Py_Initialize();
  //import_array();

  main_module = PyImport_AddModule("__main__");
  main_dict   = PyModule_GetDict(main_module);


  //This is automatically checked in the function import_array. Thus remove again
  //printf("Version: %d\n", NPY_VERSION);
  //printf("Version: %u\n", PyArray_GetNDArrayCVersion());
  //printf("Version: %d\n", NPY_FEATURE_VERSION);
  //printf("Version: %u\n", PyArray_GetNDArrayCFeatureVersion());


  fpy = fopen(pyp->f_code, "r");
  PyRun_SimpleFile(fpy, pyp->f_code);

  py_fun = PyDict_GetItemString(main_dict, "force_hl_tsa_tsi");

//  #pragma omp parallel private(py_tsi,py_ce,py_year,py_month,py_day,tsi_,ce_,year_,month_,day_,t,b,py_return,py_val) shared(mask_,ts,nc,ni,nodata,pyp,py_fun,n,main_module) default(shared)
  {
    
 //   PyGILState_STATE gstate;
 //   gstate = PyGILState_Ensure();

    py_tsi   = (PyArrayObject *) PyArray_SimpleNew(1, &n, NPY_INT);
    py_ce    = (PyArrayObject *) PyArray_SimpleNew(1, &n, NPY_INT);
    py_year  = (PyArrayObject *) PyArray_SimpleNew(1, &n, NPY_INT);
    py_month = (PyArrayObject *) PyArray_SimpleNew(1, &n, NPY_INT);
    py_day   = (PyArrayObject *) PyArray_SimpleNew(1, &n, NPY_INT);
    
    tsi_   = (int*)py_tsi->data;
    ce_    = (int*)py_ce->data;
    year_  = (int*)py_year->data;
    month_ = (int*)py_month->data;
    day_   = (int*)py_day->data;


   // #pragma omp for
    for (p=0; p<nc; p++){

      if (mask_ != NULL && !mask_[p]){
        for (b=0; b<pyp->nb; b++) ts->pyp_[b][p] = nodata;
        continue;
      }


      for (t=0; t<ni; t++){
        tsi_[t]   = ts->tsi_[t][p];
        ce_[t]    = ts->d_tsi[t].ce;
        year_[t]  = ts->d_tsi[t].year;
        month_[t] = ts->d_tsi[t].month;
        day_[t]   = ts->d_tsi[t].day;
      }

      //PyObject *py_return = PyObject_CallFunctionObjArgs(py_fun, py_tsi, NULL);
      py_return = PyObject_CallFunctionObjArgs(py_fun, py_tsi, py_ce, py_year, py_month, py_day, NULL);
      
      if (py_return == NULL){
        printf("NULL returned from python. Clean up the python plugin code!\n");
        printf("Your code failed with this data:\n");
        printf("tsi:   [%d", tsi_[0]);   for (t=0; t<ni; t++) printf(" %d", tsi_[t]);   printf("]\n");
        printf("ce:    [%d", ce_[0]);    for (t=0; t<ni; t++) printf(" %d", ce_[t]);    printf("]\n");
        printf("year:  [%d", year_[0]);  for (t=0; t<ni; t++) printf(" %d", year_[t]);  printf("]\n");
        printf("month: [%d", month_[0]); for (t=0; t<ni; t++) printf(" %d", month_[t]); printf("]\n");
        printf("day:   [%d", day_[0]);   for (t=0; t<ni; t++) printf(" %d", day_[t]);   printf("]\n");
        exit(FAILURE);}
      
      #ifdef FORCE_DEBUG
      if ((int)Py_SIZE(py_return) != pyp->nb){
        printf("wrong number of values received from python. Got %d, expected %d\n", 
          (int)Py_SIZE(py_return), pyp->nb); 
        exit(FAILURE);}
      #endif
      //printf("%d values received from python\n", (int)Py_SIZE(py_return));

      for (b=0; b<pyp->nb; b++){
        py_val = PyList_GetItem(py_return, b);
        ts->pyp_[b][p] = (short)PyLong_AsLong(py_val);
      }

      Py_DECREF(py_return);

    }

    Py_DECREF(py_tsi);
    Py_DECREF(py_ce);
    Py_DECREF(py_year);
    Py_DECREF(py_month);
    Py_DECREF(py_day);

  // PyGILState_Release(gstate);

  }

  fclose(fpy);
  //Py_Finalize();

  //for (p=0; p<15; p++) printf("\n");

  return SUCCESS;
}


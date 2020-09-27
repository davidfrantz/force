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
PyObject *pyfun = NULL;
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

  pyfun = PyDict_GetItemString(main_dict, "force_register_hl_tsa_tsi");

  py_register = PyObject_CallFunctionObjArgs(pyfun, NULL);

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
PyObject *pyfun = NULL;
PyObject *py_return = NULL;
void *ptr = NULL;
PyObject *v = NULL;


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

  pyfun = PyDict_GetItemString(main_dict, "force_hl_tsa_tsi");

  //#pragma omp parallel private(b) shared(mask_,ts,nc,ni,nodata,pyp) default(none)
//  {

    PyObject* py_tsi        = PyArray_SimpleNew(1, &n, NPY_INT);
    PyObject* py_date_ce    = PyArray_SimpleNew(1, &n, NPY_INT);
    PyObject* py_date_year  = PyArray_SimpleNew(1, &n, NPY_INT);
    PyObject* py_date_month = PyArray_SimpleNew(1, &n, NPY_INT);
    PyObject* py_date_day   = PyArray_SimpleNew(1, &n, NPY_INT);

//    //#pragma omp for
    for (p=0; p<nc; p++){

      if (mask_ != NULL && !mask_[p]){
        for (b=0; b<pyp->nb; b++) ts->pyp_[b][p] = nodata;
        continue;
      }

      //PyRun_SimpleString("print ('hello world, Python inline speaking!')");
      //PyObject_CallObject(pyfun, NULL);
      //PyObject *py_return = PyObject_CallFunction(pyfun, "ii", p, nc);
      //PyObject* py_p   = PyLong_FromLong(p);
      //PyObject* py_nc  = PyLong_FromLong(nc);
      //PyObject* py_len = PyLong_FromLong(n);



    for (t=0; t<ni; t++){

      ptr = PyArray_GETPTR1(py_tsi, t);
      v = PyLong_FromLong(ts->tsi_[t][p]);
      PyArray_SETITEM(py_tsi, ptr, v); 
      Py_DECREF(v);

      ptr = PyArray_GETPTR1(py_date_ce, t);
      v = PyLong_FromLong(ts->d_tsi[t].ce);
      PyArray_SETITEM(py_date_ce, ptr, v);
      Py_DECREF(v);

      ptr = PyArray_GETPTR1(py_date_year, t);
      v = PyLong_FromLong(ts->d_tsi[t].year);
      PyArray_SETITEM(py_date_year, ptr, v);
      Py_DECREF(v);

      ptr = PyArray_GETPTR1(py_date_month, t);
      v = PyLong_FromLong(ts->d_tsi[t].month);
      PyArray_SETITEM(py_date_month, ptr, v); 
      Py_DECREF(v);

      ptr = PyArray_GETPTR1(py_date_day, t);
      v = PyLong_FromLong(ts->d_tsi[t].day);
      PyArray_SETITEM(py_date_day, ptr, v); 
      Py_DECREF(v);


      //PyArray_GETPTR1(PyArrayObject* obj, npy_intp i);

      //int PyArray_SETITEM(py_tsi, void* itemptr, PyObject* obj)

//Convert obj and place it in the ndarray, arr, at the place pointed to by itemptr. Return -1 if an error occurs or 0 on success.

  //    PyObject* py_tsi     = PyArray_SimpleNewFromData(1, &n, NPY_SHORT, (void*)tsi);
    //  PyObject* py_date_ce = PyArray_SimpleNewFromData(1, &n, NPY_SHORT, (void*)d_tsi[t].year);

    }

      //PyObject *py_return = PyObject_CallFunctionObjArgs(pyfun, py_tsi, NULL);
      py_return = PyObject_CallFunctionObjArgs(pyfun, py_tsi, py_date_ce, py_date_year, py_date_month, py_date_day, NULL);
      
      if (py_return == NULL){
        printf("NULL returned from python. Clean up the python plugin code!\n");
        exit(FAILURE);}
      
      #ifdef FORCE_DEBUG
      if ((int)Py_SIZE(py_return) != pyp->nb){
        printf("wrong number of values received from python. Got %d, expected %d\n", 
          (int)Py_SIZE(py_return), pyp->nb); 
        exit(FAILURE);}
      #endif
      //printf("%d values received from python\n", (int)Py_SIZE(py_return));

      for (b=0; b<pyp->nb; b++){
        //ptr = PyArray_GETPTR1(py_return, b);
        //v = PyArray_GETITEM(py_return, ptr); 
        //ts->pyp_[b][p] = (short)v;
        //Py_DECREF(v);
        
        v = PyList_GetItem(py_return, b);
        ts->pyp_[b][p] = (short)PyLong_AsLong(v);
        //printf("band %d: received %d\n", b, ts->pyp_[b][p]);
      /* Add 1 to each item in the list (trivial, I know) */
        
      }

      //assert(PyLong_Check(py_return) == 1);
      //printf("C has received %d\n", (short) PyLong_AsLong(py_return));
      Py_DECREF(py_return);
      //ts->pyp_[b][p] = ?

    }
    

    Py_DECREF(py_tsi);
    Py_DECREF(py_date_ce);
    Py_DECREF(py_date_year);
    Py_DECREF(py_date_month);
    Py_DECREF(py_date_day);
    
//
//  }

  fclose(fpy);
  //Py_Finalize();

  //for (p=0; p<15; p++) printf("\n");

  return SUCCESS;
}


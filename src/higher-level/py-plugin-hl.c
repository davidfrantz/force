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
  PyRun_SimpleString("from multiprocessing.pool import Pool");
  PyRun_SimpleString("import numpy as np");
  PyRun_SimpleString("from datetime import date as Date");

  PyRun_SimpleString("def forcepy_tsi_(iblock, ce, year, month, day, nodata, nproc):         \n"
                     "   pool = Pool(nproc)                                                  \n"
                     
                     "   date = np.array([Date(y,m,d) for y, m, d in zip(year, month, day)]) \n"
                     
                     "   argss = list()                                                      \n"
                     "   for ts in iblock.T:                                                 \n"
                     "       args = (ts, date, nodata)                                       \n"
                     "       argss.append(args)                                              \n"
                     "   res = pool.map(func=forcepy_tsi, iterable=argss)                    \n"
                     "   pool.close()                                                        \n"
                     "   del pool                                                            \n"
                     "   oblock = np.array(res, dtype=np.int16).T                            \n"
                     "   return oblock.copy()                                                \n");

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
int ni = 5, nc = 10, ndim;
npy_intp dim_2d[2] = { ni, nc };
npy_intp dim_1d[1] = { ni };
int t, p;
PyObject *py_nodata = NULL;
PyObject *py_nproc = NULL;
PyArrayObject* py_tsi = NULL;
PyArrayObject* py_ce = NULL;
PyArrayObject* py_year = NULL;
PyArrayObject* py_month = NULL;
PyArrayObject* py_day = NULL;
PyArrayObject *py_return = NULL;
short* tsi_ = NULL;
int* ce_ = NULL;
int* year_ = NULL;
int* month_ = NULL;
int* day_ = NULL;
npy_intp *dim = NULL;


  if (!pyp->opyp){
    pyp->nb = 1;
    return;
  }

  main_module = PyImport_AddModule("__main__");
  main_dict   = PyModule_GetDict(main_module);

  fpy = fopen(pyp->f_code, "r");
  PyRun_SimpleFile(fpy, pyp->f_code);

  py_fun = PyDict_GetItemString(main_dict, "forcepy_tsi_init");
  if (py_fun == NULL){
    printf("Python function \"%s\" was not found. Check your python plugin code!\n", "forcepy_tsi_init");
    exit(FAILURE);}

  py_register = PyObject_CallFunctionObjArgs(py_fun, NULL);

  pyp->nb = (int)Py_SIZE(py_register);
  Py_DECREF(py_register);

  //printf("%d\n", pyp->nb);


  py_fun = PyDict_GetItemString(main_dict, "forcepy_tsi_");
  if (py_fun == NULL){
    printf("Python function \"%s\" was not found. Check your python plugin code!\n", "forcepy_tsi_");
    exit(FAILURE);}


  py_nodata = PyLong_FromLong(-9999);
  py_nproc = PyLong_FromLong(2);

  py_tsi   = (PyArrayObject *) PyArray_SimpleNew(2, dim_2d, NPY_INT16);
  py_ce    = (PyArrayObject *) PyArray_SimpleNew(1, dim_1d, NPY_INT);
  py_year  = (PyArrayObject *) PyArray_SimpleNew(1, dim_1d, NPY_INT);
  py_month = (PyArrayObject *) PyArray_SimpleNew(1, dim_1d, NPY_INT);
  py_day   = (PyArrayObject *) PyArray_SimpleNew(1, dim_1d, NPY_INT);

  tsi_   = (short*)PyArray_DATA(py_tsi);
  ce_    = (int*)PyArray_DATA(py_ce);
  year_  = (int*)PyArray_DATA(py_year);
  month_ = (int*)PyArray_DATA(py_month);
  day_   = (int*)PyArray_DATA(py_day);


  for (t=0; t<ni; t++){
    for (p=0; p<nc; p++) tsi_[t*nc+p] = t*1000 + p*100;
    year_[t]  = 2020;
    month_[t] = t+1;
    day_[t]   = 15;
    ce_[t]    = date2ce(year_[t], month_[t], day_[t]);
  }

  py_return = (PyArrayObject *) PyObject_CallFunctionObjArgs(py_fun, py_tsi, py_ce, py_year, py_month, py_day, py_nodata, py_nproc, NULL);
  if (py_return == NULL){
    printf("Oops. Testing %s failed. "
           "NULL returned from python. "
           "Clean up the python plugin code!\n", "forcepy_tsi");
    exit(FAILURE);}

  ndim = PyArray_NDIM(py_return);
  dim  = PyArray_DIMS(py_return);

  if (ndim != 2){
    printf("Oops. Testing %s failed. "
           "Returned dimensions are incorrect: %d. "
           "Clean up the python plugin code!\n", "forcepy_tsi", ndim);
    exit(FAILURE);}

  if (dim[0] != pyp->nb){
    printf("Oops. Testing %s failed. "
           "Returned array size is incorrect. "
           "Expected %d elements in 1st dimension, received %d. "
           "Clean up the python plugin code!\n", "forcepy_tsi", pyp->nb, (int)dim[0]);
    exit(FAILURE);}

  if (dim[1] != nc){
    printf("Oops. Testing %s failed. "
           "Returned array size is incorrect. "
           "Expected %d elements in 2nd dimension (not all pixels returned), received %d. "
           "Clean up the python plugin code!\n", "forcepy_tsi", nc, (int)dim[1]);
    exit(FAILURE);}


  Py_DECREF(py_return);
  Py_DECREF(py_tsi);
  Py_DECREF(py_ce);
  Py_DECREF(py_year);
  Py_DECREF(py_month);
  Py_DECREF(py_day);
  Py_DECREF(py_nodata);
  Py_DECREF(py_nproc);

  fclose(fpy);

  return;
}


/** This function connects the TSA module to plug'n'play python code
--- ts:     pointer to instantly useable TSA image arrays
--- mask:   mask image
--- nc:     number of cells
--- ni:     number of interpolation steps
--- nodata: nodata value
--- phl:    HL parameters
+++ Return: SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int tsa_python_plugin(tsa_t *ts, small *mask_, int nc, int ni, short nodata, par_hl_t *phl){
int b, t;

FILE *fpy = NULL;

npy_intp dim_2d[2] = { ni, nc };
npy_intp dim_1d[1] = { ni };

PyObject *main_module = NULL;
PyObject *main_dict = NULL;
PyObject *py_fun = NULL;

//PyObject *py_val = NULL;
PyObject *py_nodata = NULL;
PyObject *py_nproc = NULL;
PyArrayObject* py_tsi = NULL;
PyArrayObject* py_ce = NULL;
PyArrayObject* py_year = NULL;
PyArrayObject* py_month = NULL;
PyArrayObject* py_day = NULL;
PyArrayObject *py_return = NULL;
short* tsi_ = NULL;
short* return_ = NULL;
int* ce_ = NULL;
int* year_ = NULL;
int* month_ = NULL;
int* day_ = NULL;


  if (ts->pyp_ == NULL) return CANCEL;

  main_module = PyImport_AddModule("__main__");
  main_dict   = PyModule_GetDict(main_module);


  fpy = fopen(phl->tsa.pyp.f_code, "r");
  PyRun_SimpleFile(fpy, phl->tsa.pyp.f_code);

  py_fun = PyDict_GetItemString(main_dict, "forcepy_tsi_");
  if (py_fun == NULL){
    printf("Python function \"%s\" was not found. Check your python plugin code!\n", "forcepy_tsi_");
    exit(FAILURE);}

  py_nodata = PyLong_FromLong(nodata);
  py_nproc = PyLong_FromLong(phl->cthread);

  py_tsi   = (PyArrayObject *) PyArray_SimpleNew(2, dim_2d, NPY_INT16);
  py_ce    = (PyArrayObject *) PyArray_SimpleNew(1, dim_1d, NPY_INT);
  py_year  = (PyArrayObject *) PyArray_SimpleNew(1, dim_1d, NPY_INT);
  py_month = (PyArrayObject *) PyArray_SimpleNew(1, dim_1d, NPY_INT);
  py_day   = (PyArrayObject *) PyArray_SimpleNew(1, dim_1d, NPY_INT);

  tsi_   = (short*)PyArray_DATA(py_tsi);
  ce_    = (int*)PyArray_DATA(py_ce);
  year_  = (int*)PyArray_DATA(py_year);
  month_ = (int*)PyArray_DATA(py_month);
  day_   = (int*)PyArray_DATA(py_day);


  for (t=0; t<ni; t++){
    memcpy(tsi_, ts->tsi_[t], sizeof(short)*nc);
    tsi_ += nc;
    ce_[t]    = ts->d_tsi[t].ce;
    year_[t]  = ts->d_tsi[t].year;
    month_[t] = ts->d_tsi[t].month;
    day_[t]   = ts->d_tsi[t].day;
  }


  //printf("tsi:   [%d", ts->tsi_[0][0]);   for (t=0; t<ni; t++) printf(" %d", ts->tsi_[t][0]);   printf("]\n");
  py_return = (PyArrayObject *) PyObject_CallFunctionObjArgs(py_fun, py_tsi, py_ce, py_year, py_month, py_day, py_nodata, py_nproc, NULL);


  if (py_return == NULL){
    printf("Oops. NULL returned from python. Clean up the python plugin code!\n");
    exit(FAILURE);}


  return_ = (short*)PyArray_DATA(py_return);
  for (b=0; b<phl->tsa.pyp.nb; b++){
    memcpy(ts->pyp_[b], return_, sizeof(short)*nc);
    return_ += nc;
  }


  Py_DECREF(py_return);
  Py_DECREF(py_tsi);
  Py_DECREF(py_ce);
  Py_DECREF(py_year);
  Py_DECREF(py_month);
  Py_DECREF(py_day);
  Py_DECREF(py_nodata);
  Py_DECREF(py_nproc);


  fclose(fpy);


  return SUCCESS;
}

/**int tsa_python_plugin(tsa_t *ts, small *mask_, int nc, int ni, short nodata, par_pyp_t *pyp){
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
}**/


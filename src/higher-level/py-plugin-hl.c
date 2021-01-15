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
This file contains functions for plugging-in python into the TSA submodule
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Copyright (C) 2020 David Frantz, Andreas Rabe
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


#include "py-plugin-hl.h"

#include <Python.h>
#include <numpy/ndarrayobject.h>
#include <numpy/ndarraytypes.h>

//#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

/** public functions
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


/** This function initializes the python interpreter, and defines a 
+++ function for python multi-processing on the block level
--- phl:    HL parameters
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void register_python(par_hl_t *phl){
par_udf_t *udf;


  if (phl->tsa.pyp.out){
    udf = &phl->tsa.pyp;
  } else if (phl->plg.pyp.out){
    udf = &phl->plg.pyp;
  } else {
    return;
  }


  Py_Initialize();

  import_array();
  PyRun_SimpleString("from multiprocessing.pool import Pool");
  PyRun_SimpleString("import numpy as np");
  PyRun_SimpleString("from datetime import date as Date");

  if (udf->type == _UDF_PROCESS_){
    PyRun_SimpleString("def forcepy_(iblock, year, month, day, nodata, nband, nproc):           \n"
                       "    print(iblock.shape)                                                 \n"
                       "    if iblock.shape[1] == 1:                                            \n"
                       "        iblock = iblock[:,0,:]                                          \n"
                       "    pool = Pool(nproc)                                                  \n"
                       "    date = np.array([Date(y,m,d) for y, m, d in zip(year, month, day)]) \n"
                       "    argss = list()                                                      \n"
                       "    for ts in iblock.T:                                                 \n"
                       "        args = (ts, date, nodata)                                       \n"
                       "        argss.append(args)                                              \n"
                       "    res = pool.map(func=forcepy, iterable=argss)                        \n"
                       "    pool.close()                                                        \n"
                       "    del pool                                                            \n"
                       "    try:                                                                \n"
                       "        oblock = np.array(res, dtype=np.int16).T                        \n"
                       "    except Exception as error:                                          \n"
                       "        print(f'could not cast to numpy array: {str(error)}')           \n"
                       "        raise ValueError()                                              \n"
                       "    if oblock.shape != (nband,iblock.shape[-1]):                        \n"
                       "        print(f'shape mismatch, delivered {oblock.shape[0]} bands, '    \n"
                       "              f'expected {nband}.')                                     \n"
                       "        raise ValueError()                                              \n"
                       "    return oblock.copy()                                                \n");
  } else if (udf->type == _UDF_THREAD_){
    PyRun_SimpleString("print('to be implemented')");
  } else if (udf->type == _UDF_BLOCK_){
    PyRun_SimpleString("def forcepy_(iblock, year, month, day, nodata, nband, nproc):           \n"
                       "    print(iblock.shape)                                                 \n"
                       "    reduced = iblock.shape[1] == 1                                      \n"
                       "    if reduced:                                                         \n"
                       "        iblock = iblock[:,0,:]                                          \n"
                       "    date = np.array([Date(y,m,d) for y, m, d in zip(year, month, day)]) \n"
                       "    oblock = forcepy((iblock, date, nodata))                            \n"
                       "    if isinstance(oblock, (list,tuple)):                                \n"
                       "        try:                                                            \n"
                       "            oblock = np.array(oblock, dtype=np.int16)                   \n"
                       "        except Exception as error:                                      \n"
                       "            print(f'could not cast to numpy array: {str(error)}')       \n"
                       "            raise ValueError()                                          \n"
                       "    if not isinstance(oblock, np.ndarray):                              \n"
                       "        print('no array, list or tuple returned. ')                     \n"
                       "        raise ValueError()                                              \n"
                       "    if oblock.ndim != 2:                                                \n"
                       "        print(f'shape mismatch, delivered shape: {oblock.shape} , '     \n"
                       "              f'expected {nband, *iblock.shape[-1:]}.')                 \n"
                       "        raise ValueError()                                              \n"
                       "    if oblock.shape[0] != nband:                                        \n"
                       "        print(f'shape mismatch, delivered {oblock.shape[0]} bands, '    \n"
                       "              f'expected {nband} bands.')                               \n"
                       "        raise ValueError()                                              \n"
                       "    if oblock.shape[-1:] != iblock.shape[-1:]:                          \n"
                       "        print(f'shape mismatch, delivered {oblock.shape[-1:]} pixels, ' \n"
                       "              f'expected {iblock.shape[-1:]} pixels.')                  \n"
                       "        raise ValueError()                                              \n"
                       "    if oblock.dtype != np.int16:                                        \n"
                       "        oblock = oblock.astype(np.int16)                                \n"
                       "    return oblock                                                       \n");
  } else {
    printf("unknown UDF type.\n"); 
    exit(FAILURE);
  }
//T B Y X

  init_pyp(udf);
  //test_pyp(udf);

  return;
}


/** This function cleans up the python interpreter
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void deregister_python(par_hl_t *phl){

  if (phl->tsa.pyp.out || phl->plg.pyp.out) Py_Finalize();

  return;
}


/** This function initializes the output provided python function
--- udf:    user-defined code parameters
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void init_pyp(par_udf_t *udf){
FILE *fpy                = NULL;
PyObject *main_module    = NULL;
PyObject *main_dict      = NULL;
PyObject *py_fun         = NULL;
PyObject *py_register    = NULL;


  if (!udf->out){
    udf->nb = 1;
    return;
  }

  main_module = PyImport_AddModule("__main__");
  main_dict   = PyModule_GetDict(main_module);

  // parse the provided python function
  fpy = fopen(udf->f_code, "r");
  PyRun_SimpleFile(fpy, udf->f_code);

  py_fun = PyDict_GetItemString(main_dict, "forcepy_init");
  if (py_fun == NULL){
    printf("Python function \"%s\" was not found. Check your python plugin code!\n", "forcepy_init");
    exit(FAILURE);}

  py_register = PyObject_CallFunctionObjArgs(py_fun, NULL);

  udf->nb = (int)Py_SIZE(py_register);
  Py_DECREF(py_register);

  fclose(fpy);

  return;
}


/** This function loads the provided python function and makes some 
+++ tests with dummy data
--- udf:    user-defined code parameters
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void test_pyp(par_udf_t *udf){
FILE *fpy                = NULL;
PyObject *main_module    = NULL;
PyObject *main_dict      = NULL;
PyObject *py_fun         = NULL;
PyObject *py_nodata      = NULL;
PyObject *py_nproc       = NULL;
PyArrayObject* py_data   = NULL;
PyArrayObject* py_year   = NULL;
PyArrayObject* py_month  = NULL;
PyArrayObject* py_day    = NULL;
PyArrayObject *py_return = NULL;
npy_intp *dim            = NULL;
int nt = 5, nb = 1, nc = 10, ndim;
npy_intp dim_3d[3] = { nt, nb, nc };
npy_intp dim_1d[1] = { nt };
int b, t, p, k;
short *data_  = NULL;
int   *year_  = NULL;
int   *month_ = NULL;
int   *day_   = NULL;


  if (!udf->out){
    udf->nb = 1;
    return;
  }

  main_module = PyImport_AddModule("__main__");
  main_dict   = PyModule_GetDict(main_module);

  // parse the provided python function
  fpy = fopen(udf->f_code, "r");
  PyRun_SimpleFile(fpy, udf->f_code);

  py_fun = PyDict_GetItemString(main_dict, "forcepy_");
  if (py_fun == NULL){
    printf("Python function \"%s\" was not found. Check your python plugin code!\n", "forcepy_");
    exit(FAILURE);}


  // test the provided python function with dummy data
  py_nodata = PyLong_FromLong(-9999);
  py_nproc  = PyLong_FromLong(2);

  py_data  = (PyArrayObject *) PyArray_SimpleNew(3, dim_3d, NPY_INT16);
  py_year  = (PyArrayObject *) PyArray_SimpleNew(1, dim_1d, NPY_INT);
  py_month = (PyArrayObject *) PyArray_SimpleNew(1, dim_1d, NPY_INT);
  py_day   = (PyArrayObject *) PyArray_SimpleNew(1, dim_1d, NPY_INT);

  data_  = (short*)PyArray_DATA(py_data);
  year_  = (int*)PyArray_DATA(py_year);
  month_ = (int*)PyArray_DATA(py_month);
  day_   = (int*)PyArray_DATA(py_day);


  for (t=0, k=0; t<nt; t++){
    for (b=0; b<nb; b++){
      for (p=0; p<nc; p++) data_[k++] = rand() % 10000;
    }
    year_[t]  = 2020;
    month_[t] = t+1;
    day_[t]   = 15;
  }

  py_return = (PyArrayObject *) PyObject_CallFunctionObjArgs(py_fun, py_data, py_year, py_month, py_day, py_nodata, py_nproc, NULL);
  if (py_return == NULL){
    printf("Oops. Testing %s failed with dummy data. "
           "NULL returned from python. "
           "Clean up the python plugin code!\n", "forcepy_tsi");
    exit(FAILURE);}

  ndim = PyArray_NDIM(py_return);
  dim  = PyArray_DIMS(py_return);

  if (ndim != 2){
    printf("Oops. Testing %s failed with dummy data. "
           "Returned dimensions are incorrect: %d. "
           "Clean up the python plugin code!\n", "forcepy_tsi", ndim);
    exit(FAILURE);}

  if (dim[0] != udf->nb){
    printf("Oops. Testing %s failed with dummy data. "
           "Returned array size is incorrect. "
           "Expected %d elements in 1st dimension, received %d. "
           "Clean up the python plugin code!\n", "forcepy_tsi", udf->nb, (int)dim[0]);
    exit(FAILURE);}

  if (dim[1] != nc){
    printf("Oops. Testing %s failed with dummy data. "
           "Returned array size is incorrect. "
           "Expected %d elements in 2nd dimension (not all pixels returned), received %d. "
           "Clean up the python plugin code!\n", "forcepy_tsi", nc, (int)dim[1]);
    exit(FAILURE);}


  Py_DECREF(py_return);
  Py_DECREF(py_data);
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
--- nx:     number of cols
--- ny:     number of rows
--- nt:     number of time steps
--- nodata: nodata value
--- phl:    HL parameters
+++ Return: SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int tsa_python_plugin(tsa_t *ts, small *mask_, int nc, int nt, short nodata, par_hl_t *phl){
int b, t, nb = 1;

FILE *fpy = NULL;

npy_intp dim_3d[3] = { nt, nb, nc };
npy_intp dim_1d[1] = { nt };

PyObject *main_module = NULL;
PyObject *main_dict = NULL;
PyObject *py_fun = NULL;

PyObject *py_nodata = NULL;
PyObject *py_nproc = NULL;
PyObject *py_nband = NULL;
PyArrayObject* py_data = NULL;
PyArrayObject* py_year = NULL;
PyArrayObject* py_month = NULL;
PyArrayObject* py_day = NULL;
PyArrayObject *py_return = NULL;
short* data_ = NULL;
short* return_ = NULL;
int* year_ = NULL;
int* month_ = NULL;
int* day_ = NULL;


  if (ts->pyp_ == NULL) return CANCEL;

  main_module = PyImport_AddModule("__main__");
  main_dict   = PyModule_GetDict(main_module);


  fpy = fopen(phl->tsa.pyp.f_code, "r");
  PyRun_SimpleFile(fpy, phl->tsa.pyp.f_code);

  py_fun = PyDict_GetItemString(main_dict, "forcepy_");
  if (py_fun == NULL){
    printf("Python function \"%s\" was not found. Check your python plugin code!\n", "forcepy_");
    exit(FAILURE);}

  py_nodata = PyLong_FromLong(nodata);
  py_nproc = PyLong_FromLong(phl->cthread);
  py_nband = PyLong_FromLong(phl->tsa.pyp.nb);

  py_data  = (PyArrayObject *) PyArray_SimpleNew(3, dim_3d, NPY_INT16);
  py_year  = (PyArrayObject *) PyArray_SimpleNew(1, dim_1d, NPY_INT);
  py_month = (PyArrayObject *) PyArray_SimpleNew(1, dim_1d, NPY_INT);
  py_day   = (PyArrayObject *) PyArray_SimpleNew(1, dim_1d, NPY_INT);

  data_  = (short*)PyArray_DATA(py_data);
  year_  = (int*)PyArray_DATA(py_year);
  month_ = (int*)PyArray_DATA(py_month);
  day_   = (int*)PyArray_DATA(py_day);


  // copy C data to python objects
  for (t=0; t<nt; t++){
    memcpy(data_, ts->tsi_[t], sizeof(short)*nc);
    data_ += nc;
    year_[t]  = ts->d_tsi[t].year;
    month_[t] = ts->d_tsi[t].month;
    day_[t]   = ts->d_tsi[t].day;
  }

  py_return = (PyArrayObject *) PyObject_CallFunctionObjArgs(
    py_fun, py_data, py_year, py_month, py_day, py_nodata, py_nband, py_nproc, NULL);

  if (py_return == NULL){
    printf("Oops. NULL returned from python. Clean up the python plugin code!\n");
    exit(FAILURE);}


  // copy to output brick
  return_ = (short*)PyArray_DATA(py_return);
  for (b=0; b<phl->tsa.pyp.nb; b++){
    memcpy(ts->pyp_[b], return_, sizeof(short)*nc);
    return_ += nc;
  }


  // clean
  Py_DECREF(py_return);
  Py_DECREF(py_data);
  Py_DECREF(py_year);
  Py_DECREF(py_month);
  Py_DECREF(py_day);
  Py_DECREF(py_nodata);
  Py_DECREF(py_nband);
  Py_DECREF(py_nproc);


  fclose(fpy);


  return SUCCESS;
}


int ard_python_plugin(ard_t *ard, plg_t *plg, small *mask_, int nc, int nb, int nt, short nodata, par_hl_t *phl){
int b, t;

FILE *fpy = NULL;

npy_intp dim_3d[3] = { nt, nb, nc };
npy_intp dim_1d[1] = { nt };

PyObject *main_module = NULL;
PyObject *main_dict = NULL;
PyObject *py_fun = NULL;

PyObject *py_nodata = NULL;
PyObject *py_nproc = NULL;
PyObject *py_nband = NULL;
PyArrayObject* py_data = NULL;
PyArrayObject* py_year = NULL;
PyArrayObject* py_month = NULL;
PyArrayObject* py_day = NULL;
PyArrayObject *py_return = NULL;
short* data_ = NULL;
short* return_ = NULL;
int* year_ = NULL;
int* month_ = NULL;
int* day_ = NULL;
date_t date;


  if (plg->pyp_ == NULL) return CANCEL;

  main_module = PyImport_AddModule("__main__");
  main_dict   = PyModule_GetDict(main_module);


  fpy = fopen(phl->plg.pyp.f_code, "r");
  PyRun_SimpleFile(fpy, phl->plg.pyp.f_code);

  py_fun = PyDict_GetItemString(main_dict, "forcepy_");
  if (py_fun == NULL){
    printf("Python function \"%s\" was not found. Check your python plugin code!\n", "forcepy_");
    exit(FAILURE);}

  py_nodata = PyLong_FromLong(nodata);
  py_nproc = PyLong_FromLong(phl->cthread);
  py_nband = PyLong_FromLong(phl->plg.pyp.nb);

  py_data  = (PyArrayObject *) PyArray_SimpleNew(3, dim_3d, NPY_INT16);
  py_year  = (PyArrayObject *) PyArray_SimpleNew(1, dim_1d, NPY_INT);
  py_month = (PyArrayObject *) PyArray_SimpleNew(1, dim_1d, NPY_INT);
  py_day   = (PyArrayObject *) PyArray_SimpleNew(1, dim_1d, NPY_INT);

  data_  = (short*)PyArray_DATA(py_data);
  year_  = (int*)PyArray_DATA(py_year);
  month_ = (int*)PyArray_DATA(py_month);
  day_   = (int*)PyArray_DATA(py_day);


  // copy C data to python objects
  for (t=0; t<nt; t++){
    for (b=0; b<nb; b++){
      memcpy(data_, ard[t].dat[b], sizeof(short)*nc);
      data_ += nc;
    }
    date = get_brick_date(ard[t].DAT, 0);
    year_[t]  = date.year;
    month_[t] = date.month;
    day_[t]   = date.day;
  }

  py_return = (PyArrayObject *) PyObject_CallFunctionObjArgs(
    py_fun, py_data, py_year, py_month, py_day, py_nodata, py_nband, py_nproc, NULL);

  if (py_return == NULL){
    printf("Oops. NULL returned from python. Clean up the python plugin code!\n");
    exit(FAILURE);}


  // copy to output brick
  return_ = (short*)PyArray_DATA(py_return);
  for (b=0; b<phl->plg.pyp.nb; b++){
    memcpy(plg->pyp_[b], return_, sizeof(short)*nc);
    return_ += nc;
  }


  // clean
  Py_DECREF(py_return);
  Py_DECREF(py_data);
  Py_DECREF(py_year);
  Py_DECREF(py_month);
  Py_DECREF(py_day);
  Py_DECREF(py_nodata);
  Py_DECREF(py_nband);
  Py_DECREF(py_nproc);


  fclose(fpy);


  return SUCCESS;
}


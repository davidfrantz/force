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
This file contains functions for plugging-in python into FORCE
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
  PyRun_SimpleString("from numba import jit, prange, set_num_threads");
  PyRun_SimpleString("from datetime import date as Date");
  PyRun_SimpleString("import traceback");

  PyRun_SimpleString("def init(): np.seterr(all='ignore')");
  PyRun_SimpleString("init()");

  PyRun_SimpleString(
    "def forcepy_wrapper(args):                                                    \n"
    "    forcepy_udf, inarray, nband, date, sensor, bandname, nodata, nproc = args \n"
    "    outarray = np.full(shape=(nband,), fill_value=nodata, dtype=np.int16)     \n"
    "    forcepy_udf(inarray, outarray, date, sensor, bandname, nodata, nproc)     \n"
    "    return outarray                                                           \n");

  if (udf->type == _UDF_PIXEL_ && !udf->justintime){
    PyRun_SimpleString(
      "def forcepy_(iblock, year, month, day, sensor, bandname, nodata, nband, nproc):           \n"
      "    try:                                                                                  \n"
      "        print('iblock', iblock.shape)                                                     \n"
      "        nDates, nBands, nY, nX = iblock.shape                                             \n"
      "        pool = Pool(nproc, initializer=init)                                              \n"
      "        date = np.array(                                                                  \n"
      "            [np.datetime64(f'{str(y).zfill(4)}-{str(m).zfill(2)}-{str(d).zfill(2)}')      \n"
      "             for y, m, d in zip(year, month, day)])                                       \n"
      "        argss = list()                                                                    \n"
      "        for yi in range(nY):                                                              \n"
      "            for xi in range(nX):                                                          \n"
      "                inarray = iblock[:, :, yi, xi]                                            \n"
      "                args = (forcepy_pixel, inarray, nband, date, sensor, bandname, nodata, 1) \n"
      "                argss.append(args)                                                        \n"
      "        res = pool.map(func=forcepy_wrapper, iterable=argss)                              \n"
      "        pool.close()                                                                      \n"
      "        del pool                                                                          \n"
      "        # reshape space dimensions                                                        \n"
      "        oblock = np.full(shape=(nband, nY, nX), fill_value=nodata, dtype=np.int16)        \n"
      "        i = 0                                                                             \n"
      "        for yi in range(nY):                                                              \n"
      "            for xi in range(nX):                                                          \n"
      "                oblock[:, yi, xi] = res[i]                                                \n"
      "                i += 1                                                                    \n"
      "        return oblock                                                                     \n"
      "    except:                                                                               \n"
      "        print(traceback.format_exc())                                                     \n"
      "        return None                                                                       \n");
  } else if (udf->type == _UDF_PIXEL_ && udf->justintime){
    PyRun_SimpleString(
      "@jit(nopython=True, nogil=True, parallel=True)                                         \n"
      "def forcepy_(iblock, year, month, day, sensor, bandname, nodata, nband, nproc):        \n"
      "    set_num_threads(nproc)                                                             \n"
      "        date = np.array(                                                               \n"
      "            [np.datetime64(f'{str(y).zfill(4)}-{str(m).zfill(2)}-{str(d).zfill(2)}')   \n"
      "             for y, m, d in zip(year, month, day)])                                    \n"
      "    buffer = [0, 0, 0, 0]  # todo pass buffer as argument                              \n"
      "    xBufMin, xBufMax, yBufMin, yBufMax = buffer                                        \n"
      "    nDates, nBands, nY, nX = iblock.shape                                              \n"
      "    outblock = np.full(shape=(nband, nY, nX), fill_value=nodata)                       \n"
      "    for iYX in prange(nY * nX):                                                        \n"
      "        iX = iYX % nX                                                                  \n"
      "        iY = iYX // nX                                                                 \n"
      "        inarray = iblock[:, :, iY-yBufMin: iY+yBufMax+1, iX-xBufMin: iX+xBufMax+1]     \n"
      "        outarray = outblock[:, iY, iX]                                                 \n"
      "        forcepy_pixel(inarray, outarray, date, sensor, bandname, nodata, 1)            \n"
      "    return outblock                                                                    \n");
  } else if (udf->type == _UDF_BLOCK_){
    PyRun_SimpleString(
      "def forcepy_(iblock, year, month, day, sensor, bandname, nodata, nband, nproc):        \n"
      "    try:                                                                               \n"
      "        print('iblock', iblock.shape)                                                  \n"
      "        nDates, nBands, nY, nX = iblock.shape                                          \n"
      "        date = np.array(                                                               \n"
      "            [np.datetime64(f'{str(y).zfill(4)}-{str(m).zfill(2)}-{str(d).zfill(2)}')   \n"
      "             for y, m, d in zip(year, month, day)])                                    \n"
      "        oblock = np.full(shape=(nband, nY, nX), fill_value=nodata, dtype=np.int16)     \n"
      "        forcepy_block(iblock, oblock, date, sensor, bandname, nodata, nproc)           \n"
      "        return oblock                                                                  \n"
      "    except:                                                                            \n"
      "        print(traceback.format_exc())                                                  \n"
      "        return None                                                                    \n");
  } else {
    printf("unknown UDF type.\n"); 
    exit(FAILURE);
  }


  init_pyp(udf);


  return;
}


/** This function cleans up the python interpreter
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void deregister_python(par_hl_t *phl){
par_udf_t *udf;


  if (phl->tsa.pyp.out){
    udf = &phl->tsa.pyp;
  } else if (phl->plg.pyp.out){
    udf = &phl->plg.pyp;
  } else {
    return;
  }

  if (udf->out) Py_Finalize();

  if (udf->bandname != NULL){
    free_2D((void**)udf->bandname, udf->nb); 
    udf->bandname = NULL;
  }

  return;
}


/** This function initializes the output provided python function
--- udf:    user-defined code parameters
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void init_pyp(par_udf_t *udf){
FILE *fpy             = NULL;
PyObject *main_module = NULL;
PyObject *main_dict   = NULL;
PyObject *py_fun      = NULL;
PyObject *py_return   = NULL;
PyObject *py_bandname = NULL;
PyObject *py_encoded  = NULL;
char *bandname = NULL;
int b;


  //make sure bandnames are NULL-initialized
  udf->bandname = NULL;

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

  py_return = PyObject_CallFunctionObjArgs(py_fun, NULL);

  if (py_return == NULL){
    printf("NULL returned from forcepy_init. Clean up the python plugin code!\n");
    exit(FAILURE);}

  if (!PyList_Check(py_return)){
    printf("forcepy_init did not return a list. Clean up the python plugin code!\n");
    exit(FAILURE);}


  udf->nb = (int)PyList_Size(py_return);

  alloc_2D((void***)&udf->bandname, udf->nb, NPOW_10, sizeof(char));

  for (b=0; b<udf->nb; b++){
    py_bandname = PyList_GetItem(py_return, b);
    py_encoded  = PyUnicode_AsEncodedString(py_bandname, "UTF-8", "strict");
    if ((bandname = PyBytes_AsString(py_encoded)) == NULL){
      printf("forcepy_init did not return a list of strings. Clean up the python plugin code!\n");
      exit(FAILURE);}
    Py_DECREF(py_encoded);
    copy_string(udf->bandname[b], NPOW_10, bandname);
    printf("bandname # %d: %s\n", b, udf->bandname[b]);
  }


  Py_DECREF(py_return);

  fclose(fpy);

  return;
}



/** This function connects FORCE to plug'n'play python code
--- ard:    pointer to instantly useable ARD image arrays
--- ts:     pointer to instantly useable TSA image arrays
--- plg:    pointer to instantly useable PLG image arrays
--- mask:   mask image
--- nt:     number of time steps
--- nodata: nodata value
--- phl:    HL parameters






+++ Return: SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int python_plugin(ard_t *ard, plg_t *plg, tsa_t *ts, small *mask_, int nx, int ny, int nc, int nb, int idx, int nt, short nodata, par_hl_t *phl){
int b, t,  submodule;
npy_intp dim_data[4] = { nt, nb, ny, nx };
npy_intp dim_nt[1] = { nt };
npy_intp dim_nb[1] = { nb };
FILE     *fpy         = NULL;
PyObject *main_module = NULL;
PyObject *main_dict   = NULL;
PyObject *py_fun      = NULL;
PyObject *py_nodata   = NULL;
PyObject *py_nproc    = NULL;
PyObject *py_nband    = NULL;
PyArrayObject* py_data     = NULL;
PyArrayObject* py_year     = NULL;
PyArrayObject* py_month    = NULL;
PyArrayObject* py_day      = NULL;
PyArrayObject* py_sensor   = NULL;
PyArrayObject* py_bandname = NULL;
PyArrayObject *py_return   = NULL;
PyArray_Descr *py_desc_sensor   = NULL;
PyArray_Descr *py_desc_bandname = NULL;
short* data_    = NULL;
short* return_  = NULL;
int* year_      = NULL;
int* month_     = NULL;
int* day_       = NULL;
char *sensor_   = NULL;
char *bandname_ = NULL;
date_t date;
char sensor[NPOW_04];
char bandname[NPOW_10];
par_udf_t *udf;


  if (phl->plg.pyp.out){
    udf = &phl->plg.pyp;
    submodule = _HL_PLG_;
    if (plg->pyp_ == NULL) return CANCEL;
  } else if (phl->tsa.pyp.out){
    udf = &phl->tsa.pyp;
    submodule = _HL_TSA_;
    if (ts->pyp_ == NULL) return CANCEL;
  } else {
    exit(FAILURE);
  }



  main_module = PyImport_AddModule("__main__");
  main_dict   = PyModule_GetDict(main_module);


  fpy = fopen(udf->f_code, "r");
  PyRun_SimpleFile(fpy, udf->f_code);

  py_fun = PyDict_GetItemString(main_dict, "forcepy_");
  if (py_fun == NULL){
    printf("Python function \"%s\" was not found.\n", "forcepy_");
    exit(FAILURE);}

  py_nodata = PyLong_FromLong(nodata);
  py_nproc = PyLong_FromLong(phl->cthread);
  py_nband = PyLong_FromLong(udf->nb);

  py_data     = (PyArrayObject *) PyArray_SimpleNew(4, dim_data, NPY_INT16);
  py_year     = (PyArrayObject *) PyArray_SimpleNew(1, dim_nt, NPY_INT);
  py_month    = (PyArrayObject *) PyArray_SimpleNew(1, dim_nt, NPY_INT);
  py_day      = (PyArrayObject *) PyArray_SimpleNew(1, dim_nt, NPY_INT);

  py_desc_sensor = PyArray_DescrNewFromType(NPY_STRING);
  py_desc_sensor->elsize = NPOW_04;
  py_sensor = (PyArrayObject *) PyArray_SimpleNewFromDescr(1, dim_nt, py_desc_sensor);

  py_desc_bandname = PyArray_DescrNewFromType(NPY_STRING);
  py_desc_bandname->elsize = NPOW_10;
  py_bandname = (PyArrayObject *) PyArray_SimpleNewFromDescr(1, dim_nb, py_desc_bandname);

  data_     = (short*)PyArray_DATA(py_data);
  year_     = (int*)PyArray_DATA(py_year);
  month_    = (int*)PyArray_DATA(py_month);
  day_      = (int*)PyArray_DATA(py_day);
  sensor_   = (char*)PyArray_DATA(py_sensor);
  bandname_ = (char*)PyArray_DATA(py_bandname);


  // copy C data to python objects
  
  if (submodule == _HL_PLG_){

    for (t=0; t<nt; t++){
      for (b=0; b<nb; b++){
        memcpy(data_, ard[t].dat[b], sizeof(short)*nc);
        data_ += nc;
      }
      date = get_brick_date(ard[t].DAT, 0);
      year_[t]  = date.year;
      month_[t] = date.month;
      day_[t]   = date.day;
      get_brick_sensor(ard[t].DAT, 0, sensor, NPOW_04);
      copy_string(sensor_, NPOW_04, sensor);
      sensor_ += NPOW_04;
    }

    for (b=0; b<nb; b++){
      get_brick_bandname(ard[0].DAT, b, bandname, NPOW_10);
      copy_string(bandname_, NPOW_10, bandname);
      bandname_ += NPOW_10;
    }

  } else if (submodule == _HL_TSA_){

    for (t=0; t<nt; t++){
      memcpy(data_, ts->tsi_[t], sizeof(short)*nc);
      data_ += nc;
      year_[t]  = ts->d_tsi[t].year;
      month_[t] = ts->d_tsi[t].month;
      day_[t]   = ts->d_tsi[t].day;
      copy_string(sensor_, NPOW_04, "BLEND");
      sensor_ += NPOW_04;
    }

    copy_string(bandname_, NPOW_10, phl->tsa.index_name[idx]);
    bandname_ += NPOW_10;

  } else {
    printf("unknown submodule. ");
    exit(FAILURE);
  }


  // fire up python
  py_return = (PyArrayObject *) PyObject_CallFunctionObjArgs(
    py_fun, 
    py_data, 
    py_year, py_month, py_day, 
    py_sensor, 
    py_bandname, 
    py_nodata, 
    py_nband, 
    py_nproc, 
    NULL);

  if (py_return == NULL){
    printf("NULL returned from python. Clean up the python plugin code!\n");
    exit(FAILURE);}


  // copy to output brick
  return_ = (short*)PyArray_DATA(py_return);

  if (submodule == _HL_PLG_){

    for (b=0; b<udf->nb; b++){
      memcpy(plg->pyp_[b], return_, sizeof(short)*nc);
      return_ += nc;
    }

  } else if (submodule == _HL_TSA_){

    for (b=0; b<udf->nb; b++){
      memcpy(ts->pyp_[b], return_, sizeof(short)*nc);
      return_ += nc;
    }

  } else {
    printf("unknown submodule.\n");
    exit(FAILURE);
  }



  // clean
  Py_DECREF(py_return);
  Py_DECREF(py_data);
  Py_DECREF(py_year);
  Py_DECREF(py_month);
  Py_DECREF(py_day);
  Py_DECREF(py_bandname);
  Py_DECREF(py_sensor);
  Py_DECREF(py_nodata);
  Py_DECREF(py_nband);
  Py_DECREF(py_nproc);


  fclose(fpy);


  return SUCCESS;
}


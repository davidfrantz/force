/**+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

This file is part of FORCE - Framework for Operational Radiometric 
Correction for Environmental monitoring.

Copyright (C) 2013-2022 David Frantz

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
Tables for radiometric processing
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


#include "table-ll.h"


float weighted_average(table_t *values, int value_col, table_t *weights, int weight_col);

// we need these tables in global scope for GSL optimizers
table_t _TABLE_RSR_;
table_t _TABLE_E0_;
table_t _TABLE_AW_;
table_t _TABLE_AO_;


/** This function computes weighted averages.
+++ Both values and weights are expected to come in a table struct
--- values:     table holding the values
--- value_col:  column of values
--- weights:    table holding the weights
--- weight_col: column of weights
+++ Return:     weighted average
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
float weighted_average(table_t *values, int value_col, table_t *weights, int weight_col){
int i;
double value_sum = 0, weight_sum = 0;
float average;


  if (values->nrow != weights->nrow){
    printf("Cannot compute weighted average. Number of rows do not match.\n");
    exit(FAILURE);
  }

  for (i=0; i<values->nrow; i++){
    value_sum  += values->data[i][value_col]*weights->data[i][weight_col];
    weight_sum += weights->data[i][weight_col];
  }

  if (weight_sum > 0){
    average = (float)(value_sum / weight_sum);
  } else {
    average = 0;
  }

  return average;
}


/** This function reads tables and compiles that information needed for
+++ the atmospheric correction
--- atc:    atmospheric correction factors
--- DN:     Digital Numbers
+++ Return: SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int compile_tables(atc_t *atc, brick_t *DN){
char dname_exe[NPOW_10];
char bname_rsr[NPOW_10];
char fname_rsr[NPOW_10];
char fname_E0[NPOW_10];
char fname_AW[NPOW_10];
char fname_AO[NPOW_10];
char sensor[NPOW_04];
char domain[NPOW_10];
int b, nbands_rsr, nbands_dn;
float wvl;


  get_brick_sensor(DN, 0, sensor, NPOW_04);

  get_install_directory(dname_exe, NPOW_10);
  concat_string_3(bname_rsr, NPOW_10, "spectral-response_", sensor, ".csv", "");
  concat_string_3(fname_rsr, NPOW_10, dname_exe, "force-misc", bname_rsr, "/");
  concat_string_3(fname_E0,  NPOW_10, dname_exe, "force-misc", "E0.csv", "/");
  concat_string_3(fname_AW,  NPOW_10, dname_exe, "force-misc", "absorption_water_vapor.csv", "/");
  concat_string_3(fname_AO,  NPOW_10, dname_exe, "force-misc", "absorption_ozone.csv", "/");

  // small trick here, use wavelengths as row names (RSR) to allow 
  // a simple match of columns with spectral bands of images
  _TABLE_RSR_ = read_table(fname_rsr, true,  true);
  _TABLE_E0_  = read_table(fname_E0,  false, true);
  _TABLE_AW_  = read_table(fname_AW,  false, true);
  _TABLE_AO_  = read_table(fname_AO,  false, true);

  print_table(&_TABLE_RSR_, true);
  print_table(&_TABLE_E0_,  true);
  print_table(&_TABLE_AW_,  true);
  print_table(&_TABLE_AO_,  true);

  if (strcmp(_TABLE_E0_.col_names[0], "wavelength") != 0){
    printf("1st column in E0 table needs to be 'wavelength'\n");
    return FAILURE;
  }

  if (strcmp(_TABLE_E0_.col_names[1], "E0") != 0){
    printf("2nd column in E0 table needs to be 'E0'\n");
    return FAILURE;
  }

  if (strcmp(_TABLE_AW_.col_names[0], "wavelength") != 0){
    printf("1st column in water vapor absorption table needs to be 'wavelength'\n");
    return FAILURE;
  }

  if (strcmp(_TABLE_AW_.col_names[1], "absorption") != 0){
    printf("2nd column in water vapor absorption table needs to be 'absorption'\n");
    return FAILURE;
  }
  if (strcmp(_TABLE_AO_.col_names[0], "wavelength") != 0){
    printf("1st column in ozone absorption table needs to be 'wavelength'\n");
    return FAILURE;
  }

  if (strcmp(_TABLE_AO_.col_names[1], "absorption") != 0){
    printf("2nd column in ozone absorption table needs to be 'absorption'\n");
    return FAILURE;
  }

  if (_TABLE_RSR_.nrow != _TABLE_E0_.nrow){
    printf("Number of rows in RSR table (%d) and E0 table (%d) do not match.\n", _TABLE_RSR_.nrow, _TABLE_E0_.nrow);
    exit(FAILURE);
  }

  if (_TABLE_RSR_.nrow != _TABLE_AW_.nrow){
    printf("Number of rows in RSR table (%d) and water vapor absorption table (%d) do not match.\n", _TABLE_RSR_.nrow, _TABLE_AW_.nrow);
    exit(FAILURE);
  }

  if (_TABLE_RSR_.nrow != _TABLE_AO_.nrow){
    printf("Number of rows in RSR table (%d) and ozone absorption table (%d) do not match.\n", _TABLE_RSR_.nrow, _TABLE_AO_.nrow);
    exit(FAILURE);
  }


  nbands_rsr = _TABLE_RSR_.ncol;
  nbands_dn = get_brick_nbands(DN);

  if (find_domain(DN, "TEMP") > 0){
    if (nbands_dn-1 != nbands_rsr){
      printf("number of bands in RSR table (%d) and expected bands (%d) do not match\n", nbands_rsr, nbands_dn-1);
      return FAILURE;
    }
  } else {
    if (nbands_dn != nbands_rsr){
      printf("number of bands in RSR table (%d) and expected bands (%d) do not match\n", nbands_rsr, nbands_dn);
      return FAILURE;
    }
  }

  for (b=0; b<nbands_dn; b++){

    get_brick_domain(DN, b, domain, NPOW_10);

    if (strcmp(domain, _TABLE_RSR_.col_names[b]) == 0){

      atc->E0[b] = weighted_average(&_TABLE_E0_, 1, &_TABLE_RSR_, b);

      wvl = weighted_average(&_TABLE_RSR_, 0, &_TABLE_RSR_, b);
      set_brick_wavelength(DN, b, wvl);
      atc->wvl[b]   = wvl;
      atc->lwvl[b]  = log(wvl);
      atc->lwvl2[b] = log(wvl)*log(wvl);

    } else if (strcmp(domain, "TEMP") == 0){

      // set some values to properly continue
      atc->E0[b] = 1.0;

      wvl = 11.0; // approximate wavelength for thermal
      set_brick_wavelength(DN, b, wvl);
      atc->wvl[b]   = wvl;
      atc->lwvl[b]  = log(wvl);
      atc->lwvl2[b] = log(wvl)*log(wvl);

    } else {
      printf("columns (spectral domains) in RSR do not match expectation in band %d (%s != %s)\n", b, _TABLE_RSR_.col_names[b], domain);
      return FAILURE;
    }

  }

  return SUCCESS;
}





/** Exoatmospheric irradiance
+++ Thuillier spectrum @1nm [410-2400] in W/m^2/µm
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/

/** Water vapor absorption coefficients
+++ Water vapor absorption from HITRAN 2016 @1nm [240-2400] in 1/cm
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/

/** Ozone absorption coefficients
+++ Bird & Riordan 1986 @1nm [410-2400] (interpolated to match other tables)
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/

/** Water spectral library
+++ Spectra @1nm [410-900] obtained from WASI (Gege 2004)
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/

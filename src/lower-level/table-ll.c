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
char sensor[NPOW_04];
char domain[NPOW_10];
int b_rsr, nbands_rsr;
int b_dn,  nbands_dn;
table_t E0;


  get_brick_sensor(DN, 0, sensor, NPOW_04);

  get_install_directory(dname_exe, NPOW_10);
  concat_string_3(bname_rsr, NPOW_10, "spectral-response_", sensor, ".csv", "");
  concat_string_3(fname_rsr, NPOW_10, dname_exe, "force-misc", bname_rsr, "/");
  concat_string_3(fname_E0,  NPOW_10, dname_exe, "force-misc", "E0.csv", "/");

  atc->rsr = read_table(fname_rsr, false, true);
  E0       = read_table(fname_E0,  false, true);

  if (strcmp(atc->rsr.col_names[0], "wavelength") != 0){
    printf("1st column in RSR needs to be 'wavelength'\n");
    return FAILURE;
  }

  if (strcmp(E0.col_names[0], "wavelength") != 0){
    printf("1st column in RSR needs to be 'wavelength'\n");
    return FAILURE;
  }

  if (strcmp(E0.col_names[1], "E0") != 0){
    printf("2nd column in RSR needs to be 'E0'\n");
    return FAILURE;
  }


  nbands_rsr = atc->rsr.ncol - 1; // wavelength is 1st column
  nbands_dn = get_brick_nbands(DN);


  if (nbands_dn != nbands_rsr){
    printf("number of bands in RSR and expected bands do not match\n");
    return FAILURE;
  }

  for (b_dn=0, b_rsr=1; b_dn<nbands_dn; b_dn++){
    get_brick_domain(DN, b_dn, domain, NPOW_10);
    if (strcmp(domain, atc->rsr.col_names[b_rsr]) != 0){
      printf("columns (spectral domains) in RSR do not match expectation\n");
      return FAILURE;
    }

    atc->E0[b_dn] = weighted_average(&E0, 1, &atc->rsr, b_rsr);

    set_brick_wavelength(DN, b_dn, weighted_average(&atc->rsr, 0, &atc->rsr, b_rsr));

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

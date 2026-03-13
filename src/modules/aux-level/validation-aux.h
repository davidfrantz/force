/**+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

This file is part of FORCE - Framework for Operational Radiometric 
Correction for Environmental monitoring.

Copyright (C) 2013-2026 David Frantz

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
Map validation header
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


#ifndef VALIDATION_H
#define VALIDATION_H

#include <stdio.h>   // core input and output functions
#include <stdlib.h>  // standard general utilities library

#include "../cross-level/const-cl.h"
#include "../cross-level/table-cl.h"

/** Geospatial Data Abstraction Library (GDAL) **/
#include "gdal.h"           // public (C callable) GDAL entry points
//#include "ogr_spatialref.h" // coordinate systems services
#include "ogr_api.h"        // OGR geometry and feature definition

#ifdef __cplusplus
extern "C" {
#endif



typedef struct{
  double **matrix; // [map][ref]
  double *row_sum; // rows
  double *col_sum; // cols
  double total_sum;
  double diag_sum;
  int n;
} confusion_t;

typedef struct{
  int *id;
  double *count; // pixel count
  double *area;
  double *weight;
  double *adjusted_area;
  double *confidence_adjusted_area;
  int n;
} class_t;

typedef struct{
  int *map;
  int *reference;
  int n;
} label_t;

typedef struct{
  double overall;
  double **class;
} accuracy_t;

enum { _ACC_PA_, _ACC_UA_, _ACC_OE_, _ACC_CE_, _ACC_LENGTH_ };


void compile_validation_classes(char *file_input_count, double pixel_area, class_t *classes);
void compile_validation_labels(char *file_input_sample, label_t *labels);
void compute_confusion_sums(confusion_t *confusion);
void compile_confusion_matrix(label_t *labels, class_t *classes, confusion_t *confusion);
void compute_estimated_area_proportions(confusion_t *classical_confusion, class_t *classes, confusion_t *adjusted_confusion);
void compute_unbiased_area(class_t *classes, confusion_t *adjusted_confusion);
void compute_confidence_of_adjusted_area(confusion_t *classical_confusion, class_t *classes, confusion_t *adjusted_confusion);
void accuracy_metrics(confusion_t *confusion, accuracy_t *accuracy);
void compute_confidence_of_overall_accuracy(confusion_t *classical_confusion, class_t *classes, accuracy_t *adjusted_accuracy, accuracy_t *standard_error);
void compute_confidence_of_users_accuracy(confusion_t *classical_confusion, class_t *classes, accuracy_t *adjusted_accuracy, accuracy_t *standard_error);
void compute_confidence_of_producers_accuracy(confusion_t *classical_confusion, class_t *classes, accuracy_t *adjusted_accuracy, accuracy_t *standard_error);
void generate_validation_report(char *file_output, class_t *classes, confusion_t *classical_confusion, confusion_t *adjusted_confusion, accuracy_t *classical_accuracy, accuracy_t *adjusted_accuracy, accuracy_t *standard_error);
void free_validation_memory(confusion_t *classical_confusion, confusion_t *adjusted_confusion, class_t *classes, label_t *labels, accuracy_t *classical_accuracy, accuracy_t *adjusted_accuracy, accuracy_t *standard_error);

#ifdef __cplusplus
}
#endif

#endif


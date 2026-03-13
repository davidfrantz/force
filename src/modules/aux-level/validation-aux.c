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
This file contains functions for map validation
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/

#include "validation-aux.h"


/** This function compiles the validation classes from the count file and 
+++ computes area and weight for each class.
--- file_input_count: input path of count file
--- pixel_area: area of pixel, e.g. 900 for typical Landsat
--- classes: validation classes (modified, must be freed)
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/

/** This function compiles the validation classes from the count file and 
+++ computes area and weight for each class.
--- file_input_count: input path of count file
--- pixel_area: area of pixel, e.g. 900 for typical Landsat
--- classes: validation classes (modified, must be freed)
+++ Return: void
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void compile_validation_classes(char *file_input_count, double pixel_area, class_t *classes){

  
  table_t map_histogram = read_table(file_input_count, false, true);

  int col_class = find_table_col(&map_histogram, "class");
  if (col_class < 0){
    printf("could not find column name %s in count file\n", "class"); exit(FAILURE);}
  int col_count = find_table_col(&map_histogram, "count");
  if (col_count < 0){
    printf("could not find column name %s in count file\n", "count"); exit(FAILURE);}
    
    
  #ifdef FORCE_DEBUG
  print_table(&map_histogram, false, false);
  printf("column %s in column %d\n", "class", col_class);
  printf("column %s in column %d\n", "count", col_count);
  printf("table has %d classes\n", map_histogram.nrow);
  #endif
  
  classes->n = map_histogram.nrow;

  
  alloc((void**)&classes->id, classes->n, sizeof(int));
  alloc((void**)&classes->count, classes->n, sizeof(double));
  alloc((void**)&classes->area, classes->n, sizeof(double));
  alloc((void**)&classes->weight, classes->n, sizeof(double));

  int error = 0;

  for (int class=0; class<classes->n; class++){
    if (map_histogram.data[class][col_count] < 2){
      fprintf(stderr, "class %d has a pixel count <= 1\n", class);
      error++;
    }
    classes->id[class] = (int)map_histogram.data[class][col_class];
    classes->count[class] = map_histogram.data[class][col_count];
    classes->area[class] = map_histogram.data[class][col_count] * pixel_area;
    classes->weight[class] = map_histogram.data[class][col_count] / map_histogram.sum[col_count];
  }

  if (error > 0) exit(FAILURE);

  free_table(&map_histogram);

  return;
}


/** This function compiles the validation labels from the sample file.
--- file_input_sample: input path of sample file
--- labels: validation labels (modified, must be freed)
+++ Return: void
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void compile_validation_labels(char *file_input_sample, label_t *labels){

  
  GDALAllRegister();

  GDALDatasetH dataset;

  dataset = GDALOpenEx(file_input_sample, GDAL_OF_VECTOR, NULL, NULL, NULL);
  if (dataset == NULL){
    fprintf(stderr, "Open failed.\n");
    exit(FAILURE);
  }

  if (GDALDatasetGetLayerCount(dataset) != 1){
    fprintf(stderr, "Dataset has more than one layer.\n");
    GDALClose(dataset);
    exit(FAILURE);
  }

  OGRLayerH layer = GDALDatasetGetLayer(dataset, 0);
  if (layer == NULL){
    fprintf(stderr, "Could not get layer from dataset.\n");
    GDALClose(dataset);
    exit(FAILURE);
  }

  // Get layer definition
  OGRFeatureDefnH layer_def = OGR_L_GetLayerDefn(layer);
  if (layer_def == NULL){
    fprintf(stderr, "Could not get layer definition.\n");
    GDALClose(dataset);
    exit(FAILURE);
  }

  // Get field indices
  int idx_label_map = OGR_FD_GetFieldIndex(layer_def, "label_map");
  if (idx_label_map < 0){
    fprintf(stderr, "Could not find field 'label_map' in layer.\n");
    GDALClose(dataset);
    exit(FAILURE);
  }

  int idx_label_reference = OGR_FD_GetFieldIndex(layer_def, "label_reference");
  if (idx_label_reference < 0){
    fprintf(stderr, "Could not find field 'label_reference' in layer.\n");
    GDALClose(dataset);
    exit(FAILURE);
  }

  // Count features
  int feature_count = (int)OGR_L_GetFeatureCount(layer, TRUE);
  if (feature_count < 1){
    fprintf(stderr, "No features found in layer.\n");
    GDALClose(dataset);
    exit(FAILURE);
  }


  alloc((void**)&labels->map, feature_count, sizeof(int));
  alloc((void**)&labels->reference, feature_count, sizeof(int));
  labels->n = feature_count;

  // Read features and extract attributes
  OGR_L_ResetReading(layer);
  OGRFeatureH feature;
  int i = 0;
  while ((feature = OGR_L_GetNextFeature(layer)) != NULL){
    labels->map[i] = (int)OGR_F_GetFieldAsDouble(feature, idx_label_map);
    labels->reference[i] = (int)OGR_F_GetFieldAsDouble(feature, idx_label_reference);
    OGR_F_Destroy(feature);
    i++;
  }

  GDALClose(dataset);

  return;
}


/** This function computes row and column sums, total sum, and diagonal 
+++ sum for a confusion matrix.
--- confusion: confusion matrix structure (modified, must be freed)
+++ Return: void
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void compute_confusion_sums(confusion_t *confusion){

  alloc((void*)&confusion->row_sum, confusion->n, sizeof(double));
  alloc((void*)&confusion->col_sum, confusion->n, sizeof(double));

  for (int i=0; i<confusion->n; i++){
  for (int j=0; j<confusion->n; j++){
    confusion->row_sum[i] += confusion->matrix[i][j];
    confusion->col_sum[j] += confusion->matrix[i][j];
    confusion->total_sum  += confusion->matrix[i][j];
    if (i == j) confusion->diag_sum += confusion->matrix[i][j];
  }
  }
}


/** This function compiles the confusion matrix from labels and classes.
--- labels: validation labels
--- classes: validation classes
--- confusion: confusion matrix (modified, must be freed)
+++ Return: void
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void compile_confusion_matrix(label_t *labels, class_t *classes, confusion_t *confusion){

  confusion->n = classes->n;
  alloc_2D((void***)&confusion->matrix, confusion->n, confusion->n, sizeof(double));

  for (int i=0; i<labels->n; i++){

    int map_label = (int)labels->map[i];
    int ref_label = (int)labels->reference[i];

    // find map label in histogram
    int map_row = -1;
    for (int j=0; j<classes->n; j++){
      int class = classes->id[j];
      if (class == map_label){
        map_row = j;
        break;
      }
    }
    if (map_row < 0){
      fprintf(stderr, "Warning: encountered map label %d not present in count file\n", map_label);
    }

    // find reference label in histogram
    int ref_row = -1;
    for (int j=0; j<classes->n; j++){
      int class = classes->id[j];
      if (class == ref_label){
        ref_row = j;
        break;
      }
    }
    if (ref_row < 0){
      fprintf(stderr, "Warning: encountered reference label %d not present in count file\n", ref_label);
    }

    if ((map_row >= 0) && (ref_row >= 0)){
      confusion->matrix[map_row][ref_row] += 1.0;
    } else{
      fprintf(stderr, "Warning: encountered label pair (%d, %d) not present in count file\n", map_label, ref_label);
      exit(FAILURE);
    }
  }

  compute_confusion_sums(confusion);

  return;
}


/** This function computes the area-adjusted confusion matrix (proportions).
--- classical_confusion: classical confusion matrix
--- classes: validation classes
--- adjusted_confusion: area-adjusted confusion matrix (modified, must be freed)
+++ Return: void
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void compute_estimated_area_proportions(confusion_t *classical_confusion, class_t *classes, confusion_t *adjusted_confusion){

  adjusted_confusion->n = classical_confusion->n;
  alloc_2D((void***)&adjusted_confusion->matrix, adjusted_confusion->n, adjusted_confusion->n, sizeof(double));

  // Olofsson et al. 2013, eq. 1
  for (int i=0; i<classes->n; i++){
  for (int j=0; j<classes->n; j++){

    adjusted_confusion->matrix[i][j] =
      classical_confusion->matrix[i][j] /
      classical_confusion->row_sum[i] *
      classes->weight[i];
  
  }
  }

  compute_confusion_sums(adjusted_confusion);

  return;
}


/** This function computes the unbiased area estimates for each class.
--- classes: validation classes (modified)
--- adjusted_confusion: area-adjusted confusion matrix
+++ Return: void
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void compute_unbiased_area(class_t *classes, confusion_t *adjusted_confusion){
    
  double total_area = 0.0;
  for (int j=0; j<classes->n; j++) total_area += classes->area[j];
    
  alloc((void**)&classes->adjusted_area, classes->n, sizeof(double));
    
  // Olofsson et al. 2013, eq. 2
  for (int j=0; j<classes->n; j++){
    classes->adjusted_area[j] = total_area * adjusted_confusion->col_sum[j];
  }

  return;
}


/** This function computes the confidence intervals for the adjusted area estimates.
--- classical_confusion: classical confusion matrix
--- classes: validation classes (modified)
--- adjusted_confusion: area-adjusted confusion matrix
+++ Return: void
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void compute_confidence_of_adjusted_area(confusion_t *classical_confusion, class_t *classes, confusion_t *adjusted_confusion){

  alloc((void**)&classes->confidence_adjusted_area, classes->n, sizeof(double));
  
  double total_area = 0.0;
  for (int j=0; j<classes->n; j++) total_area += classes->area[j];
    
  // Olofsson et al. 2013, eq. 3-5
  for (int j=0; j<classes->n; j++){
  
    double sum = 0.0;

    for (int i=0; i<classes->n; i++){

      sum += classes->weight[i] * classes->weight[i] * 
        classical_confusion->matrix[i][j] / classical_confusion->row_sum[i] * 
        (1.0 - classical_confusion->matrix[i][j] / classical_confusion->row_sum[i]) / 
        (classical_confusion->row_sum[i] - 1.0);

    }

    classes->confidence_adjusted_area[j] = 
      sqrt(sum) * // eq 3
      total_area * // eq 4
      1.96; // eq 5
  
  }

  return;
}

// confusion.matrix[map][ref]

/** This function computes accuracy metrics (overall, producer's, user's, 
+++ omission, commission) for a confusion matrix.
--- confusion: confusion matrix
--- accuracy: accuracy metrics (modified, must be freed)
+++ Return: void
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void accuracy_metrics(confusion_t *confusion, accuracy_t *accuracy){ 

  // overall accuracy
  accuracy->overall = confusion->diag_sum / confusion->total_sum;

  // class-wise accuracies
  alloc_2D((void***)&accuracy->class, confusion->n, _ACC_LENGTH_, sizeof(double));
  
  for (int class=0; class<confusion->n; class++){
    accuracy->class[class][_ACC_PA_] = confusion->matrix[class][class] / confusion->col_sum[class];
    accuracy->class[class][_ACC_UA_] = confusion->matrix[class][class] / confusion->row_sum[class];
    accuracy->class[class][_ACC_OE_] = 1.0 - accuracy->class[class][_ACC_PA_];
    accuracy->class[class][_ACC_CE_] = 1.0 - accuracy->class[class][_ACC_UA_];
  }

  return;
}


/** This function computes the confidence interval for overall accuracy.
--- classical_confusion: classical confusion matrix
--- classes: validation classes
--- adjusted_accuracy: adjusted accuracy metrics
--- standard_error: standard error metrics (modified)
+++ Return: void
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void compute_confidence_of_overall_accuracy(confusion_t *classical_confusion, class_t *classes, accuracy_t *adjusted_accuracy, accuracy_t *standard_error){

  // Olofsson et al. 2014, eq. 5
  double sum = 0.0;
  for (int i=0; i<classes->n; i++){
    sum += classes->weight[i] * classes->weight[i] * 
      adjusted_accuracy->class[i][_ACC_UA_] * 
      (1.0 - adjusted_accuracy->class[i][_ACC_UA_]) / 
      (classical_confusion->row_sum[i] - 1.0);
  }  
  
  standard_error->overall = sqrt(sum) * 1.96;
  
  return;
}


/** This function computes the confidence intervals for user's accuracy and commission error.
--- classical_confusion: classical confusion matrix
--- classes: validation classes
--- adjusted_accuracy: adjusted accuracy metrics
--- standard_error: standard error metrics (modified)
+++ Return: void
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void compute_confidence_of_users_accuracy(confusion_t *classical_confusion, class_t *classes, accuracy_t *adjusted_accuracy, accuracy_t *standard_error){

  // Olofsson et al. 2014, eq. 6

  for (int i=0; i<classes->n; i++){
    standard_error->class[i][_ACC_UA_] = 
      sqrt(
        (adjusted_accuracy->class[i][_ACC_UA_] * 
        (1.0 - adjusted_accuracy->class[i][_ACC_UA_]) / 
        (classical_confusion->row_sum[i] - 1.0))
      ) * 1.96;
      standard_error->class[i][_ACC_CE_] = standard_error->class[i][_ACC_UA_];
  }

  return;
}


/** This function computes the confidence intervals for producer's accuracy and omission error.
--- classical_confusion: classical confusion matrix
--- classes: validation classes
--- adjusted_accuracy: adjusted accuracy metrics
--- standard_error: standard error metrics (modified)
+++ Return: void
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void compute_confidence_of_producers_accuracy(confusion_t *classical_confusion, class_t *classes, accuracy_t *adjusted_accuracy, accuracy_t *standard_error){

  // Olofsson et al. 2014, eq. 7

  for (int j=0; j<classes->n; j++){

    double marginal_total = 0.0;

    for (int i=0; i<classes->n; i++){
      marginal_total += 
        classes->count[i] / 
        classical_confusion->row_sum[i] * 
        classical_confusion->matrix[i][j];
    }

    double term0 = 1.0 / (marginal_total * marginal_total);

    double term1 = classes->count[j] * classes->count[j] *
      (1.0 - adjusted_accuracy->class[j][_ACC_PA_]) * 
      (1.0 - adjusted_accuracy->class[j][_ACC_PA_]) *
      adjusted_accuracy->class[j][_ACC_UA_] * 
      (1.0 - adjusted_accuracy->class[j][_ACC_UA_]) /
      (classical_confusion->col_sum[j] - 1.0);

    double term2 = 0.0;

    for (int i=0; i<classes->n; i++){
      if (i == j) continue; // do not use diagonal elements
      term2 += classes->count[i] * classes->count[i] *
        classical_confusion->matrix[i][j] /
        classical_confusion->row_sum[i] *
        (1.0 - classical_confusion->matrix[i][j] / classical_confusion->row_sum[i]) /
        (classical_confusion->row_sum[i] - 1.0);
    }

    term2 *= adjusted_accuracy->class[j][_ACC_PA_] * adjusted_accuracy->class[j][_ACC_PA_];

    standard_error->class[j][_ACC_PA_] = sqrt(term0 * (term1 + term2)) * 1.96;
    standard_error->class[j][_ACC_OE_] = standard_error->class[j][_ACC_PA_];

  }

  return;
}


/** This function generates the validation report and writes it to a file.
--- file_output: output file path
--- classes: validation classes
--- classical_confusion: classical confusion matrix
--- adjusted_confusion: area-adjusted confusion matrix
--- classical_accuracy: classical accuracy metrics
--- adjusted_accuracy: adjusted accuracy metrics
--- standard_error: standard error metrics
+++ Return: void
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void generate_validation_report(char *file_output, class_t *classes, confusion_t *classical_confusion, confusion_t *adjusted_confusion, accuracy_t *classical_accuracy, accuracy_t *adjusted_accuracy, accuracy_t *standard_error){

  FILE *fout = fopen(file_output, "w");
  if (fout == NULL){
    fprintf(stderr, "Could not open output file for writing: %s\n", file_output);
    exit(FAILURE);
  }

  fprintf(fout, "# Traditional Accuracy assessment\n");
  fprintf(fout, "\n");
  fprintf(fout, "## Confusion matrix\n");
  fprintf(fout, "\n");
  fprintf(fout, "The confusion matrix is expressed in terms of pixel counts,\n");
  fprintf(fout, "the rows represent the predicted/map classes and the columns represent the reference classes.\n");
  fprintf(fout, "\n");

  fprintf(fout, "| |");
  for (int j=0; j<classes->n; j++) fprintf(fout, " %d |", classes->id[j]);
  fprintf(fout, "\n| --- |");
  for (int j=0; j<classes->n; j++) fprintf(fout, " --- |");
  for (int i=0; i<classes->n; i++){
    fprintf(fout, "\n| %d |", classes->id[i]);
    for (int j=0; j<classes->n; j++){
      fprintf(fout, " %d |", (int)classical_confusion->matrix[i][j]);
    }
  }
  fprintf(fout, "\n");
  fprintf(fout, "\n");
  
  fprintf(fout, "Overall Accuracy (OA): %.3f%%\n", 
    classical_accuracy->overall * 100.0);
  fprintf(fout, "\n");

  fprintf(fout, "## Class accuracies\n");
  fprintf(fout, "\n");
  fprintf(fout,  "Accuracy metrics are expressed in %%.\n");
  fprintf(fout, "\n");
  fprintf(fout, "| | Producer's Accuracy | User's Accuracy | Error of Omission | Error of Commission |\n");
  fprintf(fout, "| --- | --- | --- | --- | --- |");
  for (int i=0; i<classes->n; i++){
    fprintf(fout, "\n| %d | %.3f | %.3f | %.3f | %.3f |", 
      classes->id[i],
      classical_accuracy->class[i][_ACC_PA_] * 100.0,
      classical_accuracy->class[i][_ACC_UA_] * 100.0,
      classical_accuracy->class[i][_ACC_OE_] * 100.0,
      classical_accuracy->class[i][_ACC_CE_] * 100.0
    );
  }
  fprintf(fout, "\n");



  fprintf(fout, "\n\n");
  fprintf(fout, "# Area-Adjusted Accuracy\n");
  fprintf(fout, "\n");
  fprintf(fout, "## Confusion matrix\n");
  fprintf(fout, "\n");
  fprintf(fout, "The confusion matrix is expressed in terms of area percentage,\n");
  fprintf(fout, "the rows represent the predicted/map classes and the columns represent the reference classes.\n");
  fprintf(fout, "\n");

  fprintf(fout, "| |");
  for (int j=0; j<classes->n; j++) fprintf(fout, " %d |", classes->id[j]);
  fprintf(fout, "\n| --- |");
  for (int j=0; j<classes->n; j++) fprintf(fout, " --- |");
  for (int i=0; i<classes->n; i++){
    fprintf(fout, "\n| %d |", classes->id[i]);
    for (int j=0; j<classes->n; j++){
      fprintf(fout, " %.3f |", adjusted_confusion->matrix[i][j]*100.0);
    }
  }
  fprintf(fout, "\n");
  fprintf(fout, "\n");

  fprintf(fout, "Overall Accuracy (OA): %.3f%% \u00b1 %.3f\n", 
    adjusted_accuracy->overall * 100.0, standard_error->overall * 100.0);
  fprintf(fout, "\n");
  
  fprintf(fout, "## Class accuracies\n");
  fprintf(fout, "\n");
  fprintf(fout,  "Accuracy metrics are expressed in %%.\n");
  fprintf(fout, "\n");
  fprintf(fout, "| | Producer's Accuracy | User's Accuracy | Error of Omission | Error of Commission |\n");
  fprintf(fout, "| --- | --- | --- | --- | --- |");
  for (int i=0; i<classes->n; i++){
    fprintf(fout, "\n| %d | %.3f \u00b1 %.3f | %.3f \u00b1 %.3f | %.3f \u00b1 %.3f | %.3f \u00b1 %.3f |", 
      classes->id[i],
      adjusted_accuracy->class[i][_ACC_PA_] * 100.0, standard_error->class[i][_ACC_PA_] * 100.0,
      adjusted_accuracy->class[i][_ACC_UA_] * 100.0, standard_error->class[i][_ACC_UA_] * 100.0,
      adjusted_accuracy->class[i][_ACC_OE_] * 100.0, standard_error->class[i][_ACC_OE_] * 100.0,
      adjusted_accuracy->class[i][_ACC_CE_] * 100.0, standard_error->class[i][_ACC_CE_] * 100.0
    );
  }
  fprintf(fout, "\n");
  fprintf(fout, "\n");

  fprintf(fout, "## Area estimates\n");
  fprintf(fout, "\n");
  fprintf(fout, "| | Mapped Area | Estimated Area |\n");
  fprintf(fout, "| --- | --- | --- |");
  for (int i=0; i<classes->n; i++){
    fprintf(fout, "\n| %d | %.3f | %.3f \u00b1 %.3f |", 
      classes->id[i], 
      classes->area[i], 
      classes->adjusted_area[i], 
      classes->confidence_adjusted_area[i]);
  }
  fprintf(fout, "\n");

  fclose(fout);

  return;
}


/** This function frees all dynamically allocated memory used in validation.
--- classical_confusion: classical confusion matrix
--- adjusted_confusion: area-adjusted confusion matrix
--- classes: validation classes
--- labels: validation labels
--- classical_accuracy: classical accuracy metrics
--- adjusted_accuracy: adjusted accuracy metrics
--- standard_error: standard error metrics
+++ Return: void
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void free_validation_memory(confusion_t *classical_confusion, confusion_t *adjusted_confusion, class_t *classes, label_t *labels, accuracy_t *classical_accuracy, accuracy_t *adjusted_accuracy, accuracy_t *standard_error){

  if (classical_confusion->matrix != NULL) free_2D((void**)classical_confusion->matrix, classical_confusion->n);
  if (classical_confusion->row_sum != NULL) free((void*)classical_confusion->row_sum);
  if (classical_confusion->col_sum != NULL) free((void*)classical_confusion->col_sum);
  classical_confusion->matrix = NULL;
  classical_confusion->row_sum = NULL;
  classical_confusion->col_sum = NULL;

  if (adjusted_confusion->matrix != NULL) free_2D((void**)adjusted_confusion->matrix, adjusted_confusion->n);
  if (adjusted_confusion->row_sum != NULL) free((void*)adjusted_confusion->row_sum);
  if (adjusted_confusion->col_sum != NULL) free((void*)adjusted_confusion->col_sum);
  adjusted_confusion->matrix = NULL;
  adjusted_confusion->row_sum = NULL;
  adjusted_confusion->col_sum = NULL;

  if (classes->id != NULL) free((void*)classes->id);
  if (classes->count != NULL) free((void*)classes->count);
  if (classes->area != NULL) free((void*)classes->area);
  if (classes->weight != NULL) free((void*)classes->weight);
  if (classes->adjusted_area != NULL) free((void*)classes->adjusted_area);
  if (classes->confidence_adjusted_area != NULL) free((void*)classes->confidence_adjusted_area);
  classes->id = NULL;
  classes->count = NULL;
  classes->area = NULL;
  classes->weight = NULL;
  classes->adjusted_area = NULL;
  classes->confidence_adjusted_area = NULL;

  if (labels->map != NULL) free((void*)labels->map);
  if (labels->reference != NULL) free((void*)labels->reference);
  labels->map = NULL;
  labels->reference = NULL;

  if (classical_accuracy->class != NULL) free_2D((void**)classical_accuracy->class, classes->n);
  if (adjusted_accuracy->class != NULL) free_2D((void**)adjusted_accuracy->class, classes->n);
  if (standard_error->class != NULL) free_2D((void**)standard_error->class, classes->n);
  classical_accuracy->class = NULL;
  adjusted_accuracy->class = NULL;
  standard_error->class = NULL;

  return;
}

/**+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

This file is part of FORCE - Framework for Operational Radiometric 
Correction for Environmental monitoring.

Copyright (C) 2013-2024 David Frantz

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
This program computes map accuracy and area statistics
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


#include <stdio.h>   // core input and output functions
#include <stdlib.h>  // standard general utilities library

//#include <ctype.h>   // testing and mapping characters
//#include <unistd.h>  // standard symbolic constants and types 

//#include <time.h>

#include "../../modules/cross-level/const-cl.h"
//#include "../../modules/cross-level/utils-cl.h"
#include "../../modules/cross-level/konami-cl.h"
#include "../../modules/cross-level/string-cl.h"
#include "../../modules/cross-level/table-cl.h"
#include "../../modules/cross-level/dir-cl.h"

/** Geospatial Data Abstraction Library (GDAL) **/
#include "gdal.h"           // public (C callable) GDAL entry points
//#include "ogr_spatialref.h" // coordinate systems services
#include "ogr_api.h"        // OGR geometry and feature definition


typedef struct{
  char file_input_count[NPOW_10];
  char file_input_sample[NPOW_10];
  char file_output[NPOW_10];
  float pixel_area;
} args_t;


void usage(char *exe, int exit_code){


  printf("Usage: %s [-h] [-v] [-i] [-o output-file] -c count-file -s sample-file -a pixel-area\n", exe);
  printf("\n");
  printf("  -h  = show this help\n");
  printf("  -v  = show version\n");
  printf("  -i  = show program's purpose\n");
  printf("\n");
  printf("  -o output-file  = output file path with extension,\n");
  printf("     defaults to './accuracy-assessment.md'\n");
  printf("  -c count-file  = csv table with pixel counts per class\n");
  printf("     2 columns named class and count\n");
  printf("  -s sample-file = vector file (or csv table) with predicted and reference class labels\n");
  printf("     2 columns named label_map and label_reference\n");
  printf("  -a pixel-area  = area of one pixel in desired reporting unit, e.g.\n");
  printf("      100 for a Sentinel-2 based map to be reported in m², or\n");
  printf("     0.01 for a Sentinel-2 based map to be reported in ha\n");
  printf("\n");

  exit(exit_code);
  return;
}


void parse_args(int argc, char *argv[], args_t *args){
int opt;
bool given_c = false;
bool given_s = false;
bool given_a = false;


  opterr = 0;

  // default parameters
  copy_string(args->file_output,  NPOW_10, "accuracy-assessment.md");
 
  // optional parameters
  while ((opt = getopt(argc, argv, "hvio:c:s:a:")) != -1){
    switch(opt){
      case 'h':
        usage(argv[0], SUCCESS);
      case 'v':
        get_version(NULL, 0);
        exit(SUCCESS);
      case 'i':
        printf("Compute map accuracy and area statistics\n");
        exit(SUCCESS);
      case 'o':
        copy_string(args->file_output, NPOW_10, optarg);
        if (fileexist(args->file_output)){
          fprintf(stderr, "output report already exists: %s.\n", args->file_output); 
          usage(argv[0], FAILURE);
        }
        break;
      case 'c':
        copy_string(args->file_input_count, NPOW_10, optarg);
        if (!fileexist(args->file_input_count)){
          fprintf(stderr, "count file does not exist: %s.\n", args->file_input_count); 
          usage(argv[0], FAILURE);
        }
        given_c = true;
        break;
      case 's':
        copy_string(args->file_input_sample, NPOW_10, optarg);
        if (!fileexist(args->file_input_sample)){
          fprintf(stderr, "sample file does not exist: %s.\n", args->file_input_sample); 
          usage(argv[0], FAILURE);
        }
        given_s = true;
        break;
      case 'a':
        args->pixel_area = atof(optarg);
        given_a = true;
        if (args->pixel_area <= 0.0){
          fprintf(stderr, "Pixel area must be > 0.0\n");
          usage(argv[0], FAILURE);  
        }
        break;
      case '?':
        if (isprint(optopt)){
          fprintf(stderr, "Unknown option `-%c'.\n", optopt);
        } else{
          fprintf(stderr, "Unknown option character `\\x%x'.\n", optopt);
        }
        usage(argv[0], FAILURE);
      default:
        fprintf(stderr, "Error parsing arguments.\n");
        usage(argv[0], FAILURE);
    }
  }

  if (!given_c){
    fprintf(stderr, "count file argument is missing.\n");
    usage(argv[0], FAILURE);
  }

  if (!given_s){
    fprintf(stderr, "sample file argument is missing.\n");
    usage(argv[0], FAILURE);
  }

  if (!given_a){
    fprintf(stderr, "pixel area argument is missing.\n");
    usage(argv[0], FAILURE);
  }


  return;
}





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

void compile_classes(char *file_input_count, double pixel_area, class_t *classes){

  
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
  printf("table has %d classes\n", n_classes);
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

void compile_labels(char *file_input_sample, label_t *labels){

  
  GDALAllRegister();

  GDALDatasetH hDS;

  hDS = GDALOpenEx(file_input_sample, GDAL_OF_VECTOR, NULL, NULL, NULL);
  if(hDS == NULL){
    fprintf(stderr, "Open failed.\n");
    exit(FAILURE);
  }

  if (GDALDatasetGetLayerCount(hDS) != 1){
    fprintf(stderr, "Dataset has more than one layer.\n");
    GDALClose(hDS);
    exit(FAILURE);
  }

  OGRLayerH hLayer = GDALDatasetGetLayer(hDS, 0);
  if (hLayer == NULL){
    fprintf(stderr, "Could not get layer from dataset.\n");
    GDALClose(hDS);
    exit(FAILURE);
  }

  // Get layer definition
  OGRFeatureDefnH hFDefn = OGR_L_GetLayerDefn(hLayer);
  if (hFDefn == NULL){
    fprintf(stderr, "Could not get layer definition.\n");
    GDALClose(hDS);
    exit(FAILURE);
  }

  // Get field indices
  int idx_label_map = OGR_FD_GetFieldIndex(hFDefn, "label_map");
  if (idx_label_map < 0){
    fprintf(stderr, "Could not find field 'label_map' in layer.\n");
    GDALClose(hDS);
    exit(FAILURE);
  }

  int idx_label_reference = OGR_FD_GetFieldIndex(hFDefn, "label_reference");
  if (idx_label_reference < 0){
    fprintf(stderr, "Could not find field 'label_reference' in layer.\n");
    GDALClose(hDS);
    exit(FAILURE);
  }

  // Count features
  int feature_count = (int)OGR_L_GetFeatureCount(hLayer, TRUE);
  if (feature_count < 1){
    fprintf(stderr, "No features found in layer.\n");
    GDALClose(hDS);
    exit(FAILURE);
  }


  alloc((void**)&labels->map, feature_count, sizeof(int));
  alloc((void**)&labels->reference, feature_count, sizeof(int));
  labels->n = feature_count;

  // Read features and extract attributes
  OGR_L_ResetReading(hLayer);
  OGRFeatureH hFeature;
  int i = 0;
  while ((hFeature = OGR_L_GetNextFeature(hLayer)) != NULL){
    labels->map[i] = (int)OGR_F_GetFieldAsDouble(hFeature, idx_label_map);
    labels->reference[i] = (int)OGR_F_GetFieldAsDouble(hFeature, idx_label_reference);
    OGR_F_Destroy(hFeature);
    i++;
  }

  GDALClose(hDS);

  return;
}

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

void compute_confidence_of_adjusted_area(confusion_t *classical_confusion, class_t *classes, confusion_t *adjusted_confusion){

  alloc((void**)&classes->confidence_adjusted_area, classes->n, sizeof(double));
  
  // Olofsson et al. 2013, eq. 3-5
  for (int j=0; j<classes->n; j++){
  
    double sum = 0.0;

    for (int i=0; i<classes->n; i++){

      sum += classes->weight[i] * classes->weight[i] * 
        classical_confusion->matrix[i][j] / classical_confusion->row_sum[i] * 
        (1.0 - classical_confusion->matrix[i][j] / classical_confusion->row_sum[i]) / 
        (classical_confusion->row_sum[j] - 1.0);

    }

    classes->confidence_adjusted_area[j] = 
      sqrt(sum) * // eq 3
      classes->adjusted_area[j] * // eq 4
      1.96; // eq 5
  
  }

  return;
}

// confusion.matrix[map][ref]
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

void compute_confidence_of_users_accuracy(confusion_t *classical_confusion, class_t *classes, accuracy_t *adjusted_accuracy, accuracy_t *standard_error){

  // Olofsson et al. 2014, eq. 6

  for (int i=0; i<classes->n; i++){
    standard_error->class[i][_ACC_UA_] = 
      sqrt(
        (adjusted_accuracy->class[i][_ACC_UA_] * 
        (1.0 - adjusted_accuracy->class[i][_ACC_UA_]) / 
        (classical_confusion->row_sum[i] - 1.0))
      ) * 1.96;
  }

  return;
}

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

    standard_error->class[j][_ACC_PA_] = sqrt(term0 * (term1 + term2)) * 1.96;

  }

  return;
}

void generate_report(char *file_output, class_t *classes, confusion_t *classical_confusion, confusion_t *adjusted_confusion, accuracy_t *classical_accuracy, accuracy_t *adjusted_accuracy, accuracy_t *standard_error){

  FILE *fout = fopen(file_output, "w");
  if (fout == NULL){
    fprintf(stderr, "Could not open output file for writing: %s\n", file_output);
    exit(FAILURE);
  }

  fprintf(fout, "# Traditional Accuracy assessment\n");
  fprintf(fout, "\n");
  fprintf(fout, "## Traditional confusion matrix, expressed in terms of pixel counts:\n");
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
  fprintf(fout, "Overall Accuracy (OA): %.3f\n", 
    classical_accuracy->overall);
  fprintf(fout, "\n");
  fprintf(fout, "| | Producer's Accuracy | User's Accuracy | Error of Omission | Error of Commission |");
  fprintf(fout, "| --- | --- | --- | --- | --- |");
  for (int i=0; i<classes->n; i++){
    fprintf(fout, "\n| %d | %.3f | %.3f | %.3f | %.3f |", 
      classes->id[i],
      classical_accuracy->class[i][_ACC_PA_],
      classical_accuracy->class[i][_ACC_UA_],
      classical_accuracy->class[i][_ACC_OE_],
      classical_accuracy->class[i][_ACC_CE_]
    );
  }
  fprintf(fout, "\n");



  fprintf(fout, "\n\n");
  fprintf(fout, "# Area-Adjusted Accuracy\n");
  fprintf(fout, "-----------------------------------------------------------------\n");
  fprintf(fout, "\n");
  fprintf(fout, "## Confusion matrix, expressed in terms of proportion of area:\n");
  fprintf(fout, "\n");

  fprintf(fout, "| |");
  for (int j=0; j<classes->n; j++) fprintf(fout, " %d |", classes->id[j]);
  fprintf(fout, "\n| --- |");
  for (int j=0; j<classes->n; j++) fprintf(fout, " --- |");
  for (int i=0; i<classes->n; i++){
    fprintf(fout, "\n| %d |", classes->id[i]);
    for (int j=0; j<classes->n; j++){
      fprintf(fout, " %.3f |", adjusted_confusion->matrix[i][j]);
    }
  }
  fprintf(fout, "\n");

  fprintf(fout, "\n");
  fprintf(fout, "Overall Accuracy (OA): %.3f \u00b1 %.3f\n", 
    adjusted_accuracy->overall, standard_error->overall);
  fprintf(fout, "\n");
  fprintf(fout, "| | Producer's Accuracy | User's Accuracy | Error of Omission | Error of Commission |");
  fprintf(fout, "| --- | --- | --- | --- | --- |");
  for (int i=0; i<classes->n; i++){
    fprintf(fout, "\n| %d | %.3f \u00b1 %.3f | %.3f \u00b1 %.3f | %.3f \u00b1 %.3f | %.3f \u00b1 %.3f |", 
      classes->id[i],
      adjusted_accuracy->class[i][_ACC_PA_], standard_error->class[i][_ACC_PA_],
      adjusted_accuracy->class[i][_ACC_UA_], standard_error->class[i][_ACC_UA_],
      adjusted_accuracy->class[i][_ACC_OE_], standard_error->class[i][_ACC_OE_],
      adjusted_accuracy->class[i][_ACC_CE_], standard_error->class[i][_ACC_CE_]
    );
  }
  fprintf(fout, "\n");


  fprintf(fout, "\n");
  fprintf(fout, "| Mapped Area | Estimated Area |\n");
  fprintf(fout, "| --- | --- |");
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

void free_memory(confusion_t *classical_confusion, confusion_t *adjusted_confusion, class_t *classes, label_t *labels, accuracy_t *classical_accuracy, accuracy_t *adjusted_accuracy, accuracy_t *standard_error){

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

int main(int argc, char *argv[]){
args_t args;


  parse_args(argc, argv, &args);

  
  // read count file
  
  class_t classes = {0};
  compile_classes(args.file_input_count, args.pixel_area, &classes);
  
  label_t labels = {0};
  compile_labels(args.file_input_sample, &labels);
  
  confusion_t classical_confusion = {0};
  compile_confusion_matrix(&labels, &classes, &classical_confusion);
  
  // Olofsson et al. 2013, eq. 1
  confusion_t adjusted_confusion = {0};
  compute_estimated_area_proportions(&classical_confusion, &classes, &adjusted_confusion);

  // Olofsson et al. 2013, eq. 2
  compute_unbiased_area(&classes, &adjusted_confusion);
  
  // Olofsson et al. 2013, eq. 3-5
  compute_confidence_of_adjusted_area(&classical_confusion, &classes, &adjusted_confusion);

  // Olofsson et al. 2013, eq. 6-8
  accuracy_t classical_accuracy = {0};
  accuracy_metrics(&classical_confusion, &classical_accuracy);
  
  // Olofsson et al. 2013, eq. 6-8
  accuracy_t adjusted_accuracy = {0};
  accuracy_metrics(&adjusted_confusion, &adjusted_accuracy);


  accuracy_t standard_error = {0};
  alloc_2D((void***)&standard_error.class, classes.n, _ACC_LENGTH_, sizeof(double));

  // Olofsson et al. 2014, eq. 5
  compute_confidence_of_overall_accuracy(&classical_confusion, &classes, &adjusted_accuracy, &standard_error);

  // Olofsson et al. 2014, eq. 6
  compute_confidence_of_users_accuracy(&classical_confusion, &classes, &adjusted_accuracy, &standard_error);

  // Olofsson et al. 2014, eq. 7
  compute_confidence_of_producers_accuracy(&classical_confusion, &classes, &adjusted_accuracy, &standard_error);

  generate_report(args.file_output, &classes, &classical_confusion, &adjusted_confusion, &classical_accuracy, &adjusted_accuracy, &standard_error);

  free_memory(&classical_confusion, &adjusted_confusion, &classes, &labels, &classical_accuracy, &adjusted_accuracy, &standard_error);


  return SUCCESS;
}

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
This program computes a histogram of the given image
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


#include <stdio.h>   // core input and output functions
#include <stdlib.h>  // standard general utilities library

#include <ctype.h>   // testing and mapping characters
#include <unistd.h>  // standard symbolic constants and types 

#include <time.h>

#include "../../modules/cross-level/const-cl.h"
#include "../../modules/cross-level/utils-cl.h"
#include "../../modules/cross-level/konami-cl.h"
#include "../../modules/cross-level/string-cl.h"
#include "../../modules/cross-level/table-cl.h"
#include "../../modules/cross-level/dir-cl.h"

/** Geospatial Data Abstraction Library (GDAL) **/
#include "gdal.h"           // public (C callable) GDAL entry points
#include "ogr_spatialref.h" // coordinate systems services
#include "ogr_api.h"        // OGR geometry and feature definition


typedef struct {
  int n;
  int band;
  char file_input_image[NPOW_10];
  char file_input_sample_size[NPOW_10];
  char file_output[NPOW_10];
  char column_sample_size[NPOW_10];
  char format[NPOW_10];
} args_t;


void usage(char *exe, int exit_code){


  printf("Usage: %s [-h] [-v] [-i] [-b band] [-o output-file] [-f format] \n", exe);
  printf("       classification sample-size-table sample-size-column \n");
  printf("\n");
  printf("  -h  = show this help\n");
  printf("  -v  = show version\n");
  printf("  -i  = show program's purpose\n");
  printf("\n");
  printf("  -b band = band to use in classification image,\n");
  printf("     defaults to 1\n");
  printf("  -o output-file  = output file path with extension,\n");
  printf("     defaults to './sample.gpkg'\n");
  printf("  -f format  = output format (GDAL vector driver short name)\n");
  printf("     defaults to GPKG\n");
  printf("\n");  
  printf("  Positional arguments:\n");
  printf("  - 'classification':     classification image\n");
  printf("  - 'sample-size-table':  table with sample size per class with at least 3 named columns\n");
  printf("                          class: class ID, needs to match classes in classification\n");
  printf("                          count: number of pixels per class\n");
  printf("                          ***:   number of sample points per class;.\n");
  printf("                                 column name is variable (see next argument)\n");
  printf("  - 'sample-size-column': column name of sample size\n");
  printf("\n");

  exit(exit_code);
  return;
}


void parse_args(int argc, char *argv[], args_t *args){
int opt;
bool o, f;


  opterr = 0;

  // default parameters
  copy_string(args->file_output,  NPOW_10, "sample.gpkg");
  copy_string(args->format, NPOW_10, "GPKG");
  args->band = 1;
  o = f = false;

  // optional parameters
  while ((opt = getopt(argc, argv, "hvio:f:b:")) != -1){
    switch(opt){
      case 'h':
        usage(argv[0], SUCCESS);
      case 'v':
        get_version(NULL, 0);
        exit(SUCCESS);
      case 'i':
        printf("Compute image histogram\n");
        exit(SUCCESS);
      case 'o':
        copy_string(args->file_output, NPOW_10, optarg);
        break;
      case 'f':
        copy_string(args->format, NPOW_10, optarg);
        f = true;
        break;
      case 'b':
        args->band = atoi(optarg);
        if (args->band < 1){
          fprintf(stderr, "Band must be >= 1\n");
          usage(argv[0], FAILURE);  
        }
        break;
      case '?':
        if (isprint(optopt)){
          fprintf(stderr, "Unknown option `-%c'.\n", optopt);
        } else {
          fprintf(stderr, "Unknown option character `\\x%x'.\n", optopt);
        }
        usage(argv[0], FAILURE);
      default:
        fprintf(stderr, "Error parsing arguments.\n");
        usage(argv[0], FAILURE);
    }
  }

  // non-optional parameters
  args->n = 3;

  if (optind < argc){
    konami_args(argv[optind]);
    if (argc-optind == args->n){
      copy_string(args->file_input_image, NPOW_10, argv[optind++]);
      copy_string(args->file_input_sample_size, NPOW_10, argv[optind++]);
      copy_string(args->column_sample_size, NPOW_10, argv[optind++]);
    } else if (argc-optind < args->n){
      fprintf(stderr, "some non-optional arguments are missing.\n");
      usage(argv[0], FAILURE);
    } else if (argc-optind > args->n){
      fprintf(stderr, "too many non-optional arguments.\n");
      usage(argv[0], FAILURE);
    }
  } else {
    fprintf(stderr, "non-optional arguments are missing.\n");
    usage(argv[0], FAILURE);
  }

  if ((!o && f) || (!f && o)){
    fprintf(stderr, "If -f is given, -o need to be given, too.\n"); usage(argv[0], FAILURE);
  }

  if (fileexist(args->file_output)){
    fprintf(stderr, "sample already exists: %s.\n", args->file_output); usage(argv[0], FAILURE);
  }

  return;
}


int main(int argc, char *argv[]){
args_t args;
GDALDatasetH  fp;
GDALRasterBandH band;
int i, j, nx, ny, nbands;
short *line = NULL;
short nodata;
int has_nodata;
table_t sample_size;
int col_size, col_class, col_count, row;
char *wkt = NULL;
char projection[NPOW_10];
double geotran[6];

int offset = SHRT_MAX+1;
int length = USHRT_MAX+1;
double **dictionary = NULL;
enum { _DICT_CLASS_, _DICT_SIZE_, _DICT_PROBABILITY_, _DICT_COUNT_, _DICT_LENGTH_ };

table_t sample;
int count = 0;
enum { _SAMPLE_CLASS_, _SAMPLE_X_, _SAMPLE_Y_, _SAMPLE_PROBABILITY_, _SAMPLE_LENGTH_ };

double probability;


int class_ID, size_target, size_sampled;
int row2, row_max;
double max;
OGRSFDriverH driver;
OGRSpatialReferenceH srs;
GDALDatasetH fp_out;
OGRLayerH layer;
OGRFieldDefnH field;
OGRFeatureH feature;
OGRGeometryH point;


  parse_args(argc, argv, &args);

  sample_size = read_table(args.file_input_sample_size, false, true);
  if ((col_size = find_table_col(&sample_size, args.column_sample_size)) < 0){
    printf("could not find column name %s in file-sample-size\n", args.column_sample_size); exit(FAILURE);}
  if ((col_class = find_table_col(&sample_size, "class")) < 0){
    printf("could not find column name %s in file-sample-size\n", "class"); exit(FAILURE);}
  if ((col_count = find_table_col(&sample_size, "count")) < 0){
    printf("could not find column name %s in file-sample-size\n", "count"); exit(FAILURE);}

  #ifdef FORCE_DEBUG
  print_table(&sample_size, false, false);
  printf("column %s in column %d\n", "class", col_class);
  printf("column %s in column %d\n", "count", col_count);
  printf("column %s in column %d\n", args.column_sample_size, col_size);
  printf("min/max class: %d/%d\n", (int)sample_size.min[col_class], (int)sample_size.max[col_class]);
  #endif

  alloc_2D((void***)&dictionary, length, _DICT_LENGTH_, sizeof(double));

  for (row=0; row<sample_size.nrow; row++){
    dictionary[(int)sample_size.data[row][col_class] + offset][_DICT_CLASS_] = sample_size.data[row][col_class];
    dictionary[(int)sample_size.data[row][col_class] + offset][_DICT_SIZE_] = sample_size.data[row][col_size];
    dictionary[(int)sample_size.data[row][col_class] + offset][_DICT_PROBABILITY_] = sample_size.data[row][col_size] / sample_size.data[row][col_count] * 2;
  }

  #ifdef FORCE_DEBUG
  for (row=0; row<sample_size.nrow; row++){
    printf("%.2f %.2f %.12f\n", 
      dictionary[(int)sample_size.data[row][col_class] + offset][_DICT_CLASS_],
      dictionary[(int)sample_size.data[row][col_class] + offset][_DICT_SIZE_],
      dictionary[(int)sample_size.data[row][col_class] + offset][_DICT_PROBABILITY_]);
  }
  #endif



  sample = allocate_table(sample_size.sum[col_size] * 10, _SAMPLE_LENGTH_, false, false);
  memset(sample.row_mask, 0, sizeof(bool) * sample_size.sum[col_size] * 10);

  GDALAllRegister();
  if ((fp = GDALOpen(args.file_input_image, GA_ReadOnly)) == NULL){
    fprintf(stderr, "could not open %s.\n", args.file_input_image); exit(1);}

  nx     = GDALGetRasterXSize(fp);
  ny     = GDALGetRasterYSize(fp);
  nbands = GDALGetRasterCount(fp);

  GDALGetGeoTransform(fp, geotran);
  wkt = (char*)GDALGetProjectionRef(fp);
  copy_string(projection, NPOW_10, wkt);
  wkt = projection;
  
  if (args.band > nbands){
    fprintf(stderr, "Input image has %d band(s), band %d was requested.\n", nbands, args.band); exit(1);}
  alloc((void**)&line, nx, sizeof(short));

  band = GDALGetRasterBand(fp, args.band);

  nodata = (short)GDALGetRasterNoDataValue(band, &has_nodata);
  if (!has_nodata){
    fprintf(stderr, "input image has no nodata value.\n"); 
    exit(1);
  }


  srand(time(NULL)); 

  for (i=0; i<ny; i++){

    if (GDALRasterIO(band, GF_Read, 0, i, nx, 1, 
      line, nx, 1, GDT_Int16, 0, 0) == CE_Failure){
      fprintf(stderr, "could not read line %d.\n", i+1); exit(1);}

    for (j=0; j<nx; j++){

      if (line[j] == nodata) continue;

      probability = (double)rand()/(double)RAND_MAX;

      if (dictionary[line[j] + offset][_DICT_PROBABILITY_] >= probability){
        dictionary[line[j] + offset][_DICT_COUNT_]++;
        //printf("%d samples pre-selected (row %d, col %d)\n", count, i, j);

        if (count < sample.nrow){
          sample.data[count][_SAMPLE_CLASS_] = line[j];
          sample.data[count][_SAMPLE_X_] = geotran[0] + j * geotran[1] + geotran[1] * 0.5;
          sample.data[count][_SAMPLE_Y_] = geotran[3] + i * geotran[5] + geotran[5] * 0.5;
          sample.data[count][_SAMPLE_PROBABILITY_] = probability;
          sample.row_mask[count] = true;
          count++;
        }

      }

    }

  }


  for (row=0; row<sample_size.nrow; row++){
    printf("%.2f, %.2f, %.12f, %.0f,\n", 
      dictionary[(int)sample_size.data[row][col_class] + offset][_DICT_CLASS_],
      dictionary[(int)sample_size.data[row][col_class] + offset][_DICT_SIZE_],
      dictionary[(int)sample_size.data[row][col_class] + offset][_DICT_PROBABILITY_],
      dictionary[(int)sample_size.data[row][col_class] + offset][_DICT_COUNT_]);
  }


  GDALClose(fp);

  free((void*)line);

  for (row=0; row<sample_size.nrow; row++){

    class_ID     = dictionary[(int)sample_size.data[row][col_class] + offset][_DICT_CLASS_];
    size_target  = dictionary[(int)sample_size.data[row][col_class] + offset][_DICT_SIZE_];
    size_sampled = dictionary[(int)sample_size.data[row][col_class] + offset][_DICT_COUNT_];

    while (size_target < size_sampled){

      max = -1.0;
      row_max = 0;

      for (row2=0; row2<count; row2++){

        if (sample.data[row2][_SAMPLE_CLASS_] != class_ID) continue;
        if (!sample.row_mask[row2]) continue;

        if (sample.data[row2][_SAMPLE_PROBABILITY_] > max){
          max = sample.data[row2][_SAMPLE_PROBABILITY_];
          row_max = row2;
        }

      }

      printf("class %d, size %d <-> %d, row_max %d, max_prob %f\n", 
        class_ID, size_target, size_sampled, row_max, max);

      sample.row_mask[row_max] = false;
      
      size_sampled--;

    }

  }



  OGRRegisterAll();

  // get driver
  if ((driver = OGRGetDriverByName(args.format)) == NULL){
    fprintf(stderr, "%s driver is not available.\n", args.format); usage(argv[0], FAILURE);
  }

  
  // create file
  if ((fp_out = OGR_Dr_CreateDataSource(driver, args.file_output, NULL)) == NULL){
    printf("Error creating output file.\n"); return FAILURE;}
    

  // set output projection
  srs = OSRNewSpatialReference(NULL);
  OSRImportFromWkt(srs, &wkt);

  // create layer
  if ((layer = OGR_DS_CreateLayer(fp_out, "sample", srs, 
        wkbPoint, NULL)) == NULL){
    printf("Error creating layer.\n"); return FAILURE;}

  // add field
  field = OGR_Fld_Create("FID", OFTInteger);
  OGR_Fld_SetWidth(field, 4);
  if (OGR_L_CreateField(layer, field, TRUE) != OGRERR_NONE){
      printf("Error creating field.\n"); return FAILURE;}
  OGR_Fld_Destroy(field);

  // add field
  field = OGR_Fld_Create("label_map", OFTInteger);
  OGR_Fld_SetWidth(field, 4);
  if (OGR_L_CreateField(layer, field, TRUE) != OGRERR_NONE){
      printf("Error creating field.\n"); return FAILURE;}
  OGR_Fld_Destroy(field);

  // add field
  field = OGR_Fld_Create("label_reference", OFTInteger);
  OGR_Fld_SetWidth(field, 4);
  if (OGR_L_CreateField(layer, field, TRUE) != OGRERR_NONE){
      printf("Error creating field.\n"); return FAILURE;}
  OGR_Fld_Destroy(field);

  count = 0;

  for (row=0; row<sample.nrow; row++){

    if (!sample.row_mask[row]) continue;

    // create feature
    feature = OGR_F_Create(OGR_L_GetLayerDefn(layer));

    // set fields
    OGR_F_SetFieldInteger(feature, OGR_F_GetFieldIndex(feature, "FID"), count);
    OGR_F_SetFieldInteger(feature, OGR_F_GetFieldIndex(feature, "label_map"), sample.data[row][_SAMPLE_CLASS_]);
    OGR_F_SetFieldInteger(feature, OGR_F_GetFieldIndex(feature, "label_reference"), 0);

    // create local geometry
    point = OGR_G_CreateGeometry(wkbPoint);
    OGR_G_SetPoint_2D(point, 0, sample.data[row][_SAMPLE_X_], sample.data[row][_SAMPLE_Y_]);
    OGR_F_SetGeometry(feature, point);
    OGR_G_DestroyGeometry(point);

    // create feature in the file
    if (OGR_L_CreateFeature(layer, feature) != OGRERR_NONE){
      printf("Error creating feature in file.\n"); return FAILURE;}
    OGR_F_Destroy(feature);

    count++;

  }

  OGR_DS_Destroy(fp_out);


  free_table(&sample);
  free_table(&sample_size);
  free_2D((void**)dictionary, length);

  return SUCCESS;
}


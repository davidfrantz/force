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
This program stacks images
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/

#include <stdio.h>   // core input and output functions
#include <stdlib.h>  // standard general utilities library

#include "../cross-level/const-cl.h"
#include "../cross-level/konami-cl.h"
#include "../cross-level/alloc-cl.h"
#include "../cross-level/dir-cl.h"


/** Geospatial Data Abstraction Library (GDAL) **/
#include "gdal.h"           // public (C callable) GDAL entry points
#include "cpl_conv.h"       // various convenience functions for CPL
#include "cpl_string.h"       // various convenience functions for strings


typedef struct {
  char fname[NPOW_10];   // file name
  char dname[NPOW_10];   // directory name
  char bname[NPOW_10];   // base name
  int nx, ny, nb;        // dimensions
  char proj[NPOW_10];    // projection
  double geotran[6];     // geotransformation
  int b;                 // band
} img_t;


int main ( int argc, char *argv[] ){
int f, nf;
int b, nb;
int k;
int nx, ny;
int nodata;
GDALDriverH     driver = NULL;
GDALDatasetH    src    = NULL;
GDALDatasetH    dst    = NULL;
GDALRasterBandH bsrc   = NULL;
GDALRasterBandH bdst   = NULL;

img_t *inp = NULL;
img_t *out = NULL;



char *f_dst = NULL;
char d_out[NPOW_10];
char e_out[NPOW_10];

char **meta  = NULL;
char **bmeta = NULL;
const char *bname = NULL;
const char *proj_ = NULL;


char source[NPOW_16];


int interleave;
enum { _BYFILE_, _BYBAND_, _INTERLEN_ };



  if (argc >= 2) check_arg(argv[1]);
  if (argc  < 3){ printf("Usage: %s file1 file2 [etc] outfile.\n", argv[0]);
                  printf("  At least two input files need to be given\n\n"); 
                  return FAILURE;}

  // number of input files
  nf = argc-2;

  // output name
  f_dst = argv[argc-1];
  directoryname(f_dst, d_out, NPOW_10);
  extension(f_dst,     e_out, NPOW_10);

  if (strcmp(e_out, "VRT") != 0){
    printf("Output file must have .vrt extension\n\n"); 
    return FAILURE;}

  if (fileexist(f_dst)){
    printf("Output file already exists: %s\n", f_dst);
    printf("Delete or user another filename\n\n"); 
    return FAILURE;}

  if (chdir(d_out) != 0){
    printf("Couldn't change to output directory\n\n"); 
    return FAILURE;}


  GDALAllRegister();

  alloc((void**)&inp, nf, sizeof(img_t));

  // check if we stack file after file, or band after band
  for (f=0, nb=0, interleave=_BYBAND_; f<nf; f++){

    if (strlen(argv[f+1]) > NPOW_10-1){
      printf("cannot copy, string too long\n\n"); return FAILURE;
    } else { 
      strncpy(inp[f].fname, argv[f+1], strlen(argv[f+1])); 
      inp[f].fname[strlen(argv[f+1])] = '\0';
    }
    directoryname(inp[f].fname, inp[f].dname, NPOW_10);
    basename_with_ext(inp[f].fname, inp[f].bname, NPOW_10);

    if ((src = GDALOpenEx(inp[f].fname, GDAL_OF_READONLY, NULL, NULL, NULL)) == NULL){
      printf("Unable to open %s\n\n", inp[f].fname); return FAILURE;}

    nx  = (inp[f].nx = GDALGetRasterXSize(src));
    ny  = (inp[f].ny = GDALGetRasterYSize(src));
    nb += (inp[f].nb = GDALGetRasterCount(src));
    GDALGetGeoTransform(src, inp[f].geotran);
    
    proj_ = GDALGetProjectionRef(src);
    if (strlen(proj_) > NPOW_10-1){
      printf("cannot copy, string too long\n\n"); return FAILURE;
    } else { 
      strncpy(inp[f].proj, proj_, strlen(proj_)); 
      inp[f].proj[strlen(proj_)] = '\0';
    }

    GDALClose(src);


    printf("file %d:\n", f+1);
    printf("  %s\n", inp[f].dname);
    printf("  %s\n", inp[f].bname);
    printf("  %d %d %d\n", inp[f].nx, inp[f].ny, inp[f].nb);


    // tests for consistency and interleave type
    if (f > 0){
      if (strcmp(inp[f].dname, inp[0].dname) != 0){
        printf("Directories are different. This is not allowed.\n");
        printf("Dir 1: %s\nDir %d: %s\n\n", inp[0].dname, f+1, inp[f].dname); 
        return FAILURE;}
      if (inp[f].nx != inp[0].nx){
        printf("Number of columns are different. This is not allowed.\n");
        printf("File 1: %d\nFile %d: %d\n\n", inp[0].nx, f+1, inp[f].nx); 
        return FAILURE;}
      if (inp[f].ny != inp[0].ny){
        printf("Number of rows are different. This is not allowed.\n");
        printf("File 1: %d\nFile %d: %d\n\n", inp[0].ny, f+1, inp[f].ny); 
        return FAILURE;}
      if (inp[f].nb != inp[0].nb) interleave = _BYFILE_;
    }

  }

  if (strcmp(d_out, inp[0].dname) != 0){
    printf("Directories are different. This is not allowed.\n");
    printf("Dir input:  %s\nDir output: %s\n\n", inp[0].dname, d_out); return FAILURE;}



  alloc((void**)&out, nb, sizeof(img_t));

  // choose interleave type, and build band order
  switch (interleave){
    case _BYFILE_:
      printf("\nDifferent number of bands detected. Stacking by file.\n\n");
      for (f=0, k=0; f<nf; f++){
      for (b=0; b<inp[f].nb; b++, k++){
        memcpy(&out[k], &inp[f], sizeof(img_t));
        out[k].b = b+1;
        printf("Band %04d: %s band %d\n", k+1, out[k].bname, out[k].b);
      }
      }
      break;
    case _BYBAND_:
      printf("\nSame number of bands detected. Stacking by band.\n\n");
      for (b=0, k=0; b<inp[0].nb; b++){
      for (f=0; f<nf; f++, k++){
        memcpy(&out[k], &inp[f], sizeof(img_t));
        out[k].b = b+1;
        printf("Band %04d: %s band %d\n", k+1, out[k].bname, out[k].b);
      }
      }
      break;
    default:
      printf("unknown interleave\n\n");
      return FAILURE;
  }


  // create file with VRT driver
  if ((driver = GDALGetDriverByName("VRT")) == NULL){
    printf("Error getting VRT driver.\n\n"); return FAILURE;}

  if ((dst = GDALCreate(driver, f_dst, nx, ny, 0, GDT_Int16, NULL)) == NULL){
    printf("Error creating file %s\n\n", f_dst); return FAILURE;}


  // copy file-level metadata
  if ((src = GDALOpenEx(inp[0].fname, GDAL_OF_READONLY, NULL, NULL, NULL)) == NULL){
    printf("Unable to open %s\n\n", inp[0].fname); return FAILURE;}
  meta = GDALGetMetadata(src, "FORCE");
  GDALSetGeoTransform(dst, out[0].geotran);
  GDALSetProjection(dst,   out[0].proj);
  GDALSetMetadata(dst,     meta, "FORCE");
  GDALClose(src);


  // add the bands to vrt
  for (b=0; b<nb; b++){

    // get band-level metadata
    if ((src = GDALOpenEx(out[b].fname, GDAL_OF_READONLY, NULL, NULL, NULL)) == NULL){
      printf("Unable to open %s\n\n", out[b].fname); return FAILURE;}
    bsrc = GDALGetRasterBand(src, out[b].b);
    bmeta = GDALGetMetadata(bsrc, "FORCE");
    bname = GDALGetDescription(bsrc);
    nodata = (int)GDALGetRasterNoDataValue(bsrc, NULL);

    // add new band
    GDALAddBand(dst, GDT_Int16, NULL);
    bdst = GDALGetRasterBand(dst, GDALGetRasterCount(dst));

    // set source
    sprintf(source,
      "<ComplexSource>"
      "  <SourceFilename relativeToVRT=\"1\">%s</SourceFilename>"
      "  <SourceBand>%d</SourceBand>"
      "  <SrcRect xOff=\"0\" yOff=\"0\" xSize=\"%d\" ySize=\"%d\" />"
      "  <DstRect xOff=\"0\" yOff=\"0\" xSize=\"%d\" ySize=\"%d\" />"
      "  <NODATA>%d</NODATA>"
      "</ComplexSource>", 
      out[b].bname, out[b].b, out[b].nx, out[b].ny, out[b].nx, out[b].ny, nodata);

    // update source and metadata
    GDALSetMetadataItem(bdst, "source_0", source, "new_vrt_sources");
    GDALSetMetadata(bdst, bmeta, "FORCE");
    GDALSetDescription(bdst, bname);

    GDALClose(src);

  }

  GDALClose(dst);

  free((void*)inp);
  free((void*)out);

  return SUCCESS; 
}


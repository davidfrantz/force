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
This file contains functions for handling MODIS Atmospheric Water Vapor
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


#include "modwvp-ll.h"

/** Geospatial Data Abstraction Library (GDAL) **/
#include "cpl_string.h"     // various convenience functions for strings
#include "gdal.h"           // public (C callable) GDAL entry points


void int2bit(int integer, bool *bin, int start, int size);


/** Retrieve LAADS App Key from $HOME/.laads
--- auth:   authentification header including key
--- size:   buffer size of auth
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void get_laads_key(char auth[], int size){
char user[NPOW_10];
char fkey[NPOW_10];
char  key[NPOW_10];
int nchar;
FILE *fk = NULL;


  if (getlogin_r(user, NPOW_10) != 0){
    printf("couldn't retrieve user..\n"); exit(1);}

  nchar = snprintf(fkey, NPOW_10, "/home/%s/.laads", user);
  if (nchar < 0 || nchar >= NPOW_10){ 
    printf("Buffer Overflow in assembling filename\n"); exit(1);}

  if (!fileexist(fkey)){
    printf("LAADS authentification does not exist: %s\n", fkey); exit(1);}

    if ((fk = fopen(fkey, "r")) == NULL){
    printf("Unable to open LAADS authentification: %s\n", fkey);  exit(1);}

  if (fgets(key, NPOW_10, fk) == NULL){
    printf("Unable to read LAADS authentification from %s\n", fkey);  exit(1);}
    
  nchar = snprintf(auth, size, "Authorization: Bearer %s", key);
  if (nchar < 0 || nchar >= size){ 
    printf("Buffer Overflow in assembling app key\n"); exit(1);}

  fclose(fk);

  return;
}


/** Convert integer to bits
+++ This function converts an integer value to bit fields
--- integer: integer to be converted
--- bin:     bit field array (modified)
--- start:   starting bin (usually 0)
--- size:    number of bits (e.g. 16)
+++ Return:  void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void int2bit(int integer, bool *bin, int start, int size){
int x = integer, bit;

  for (bit=0; bit<size; bit++){
    bin[start+bit]= x % 2;
    x /= 2;
  }

  return;
}


/** This function reads coordinates from a text file. Put X-coords in 1st
+++ column and Y-coords in 2nd column. Coordinates must be in geographic 
+++ decimal degree notation (South and West coordinates are negative). Do
+++ not use a header.
--- fcoords: text file containing the coordinates
--- COORD:   array for storing the coordinates
+++ Return:  number of coordinate pairs
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int parse_coord_list(char *fcoords, float ***COORD){
FILE *fp;
char  buffer[NPOW_10] = "\0";
char *tokenptr = NULL;
char *separator = " ";
int c, nc = 0;
float **coord = NULL;


  // open file
  if (!(fp = fopen(fcoords, "r"))){
    printf("unable to open coordinate file. "); return FAILURE;}

  // count lines
  while (fgets(buffer, NPOW_10, fp) != NULL) nc++;
  fseek(fp, 0, SEEK_SET);

  #ifdef DEBUG
  printf("number of coordinate pairs: %d\n", nc);
  #endif

  alloc_2D((void***)&coord, 2, nc, sizeof(float));

  // read line by line
  for (c=0; c<nc; c++){
  
    if (fgets(buffer, NPOW_10, fp) == NULL){
      printf("error reading coordinate file.\n"); exit(1); }

    tokenptr = strtok(buffer, separator);
    coord[0][c] = atof(tokenptr); // X
    tokenptr = strtok(NULL, separator);
    coord[1][c] = atof(tokenptr); // Y

  }

  fclose(fp);
  
  *COORD = coord;
  return nc;
}


/** This function contains a listing of day-long sensor outages of the MO-
+++ DIS sensor aboard Terra. If a specified date matches one of the listed
+++ dates, true is returned, otherwise false.
+++-----------------------------------------------------------------------
+++ http://modaps.nascom.nasa.gov/services/production/outages_terra.html
+++-----------------------------------------------------------------------
--- d:      any date
+++ Return: true/false
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int failure_terra(date_t d){
int i;
int yv[47] = {  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,
                1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  2,
                2,  2,  2,  2,  2,  2,  2,  2,  3,  3,  3,  3,  3,  3,  3,
                8,  8 };
int mv[47] = {  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  6,  6,  6,
                6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  7,  7,  3,
                3,  3,  3,  3,  3,  3,  3,  4, 12, 12, 12, 12, 12, 12, 12,
               12, 12 };
int dv[47] = {  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 16, 17, 18,
               19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,  1,  2, 20,
               21, 22, 23, 24, 25, 26, 27, 15, 17, 18, 19, 20, 21, 22, 23,
               21, 22 };

  for (i=0; i<47; i++){

    if (d.year == 2000+yv[i] && d.month == mv[i] && d.day == dv[i]){
      #ifdef FORCE_DEBUG
      printf("TERRA outage. ");
      #endif
      return true;
    }
  }

  return false;
}


/** This function contains a listing of day-long sensor outages of the MO-
+++ DIS sensor aboard Aqua. If a specified date matches one of the listed
+++ dates, true is returned, otherwise false.
+++-----------------------------------------------------------------------
+++ http://modaps.nascom.nasa.gov/services/production/outages_aqua.html
+++-----------------------------------------------------------------------
--- d:      any date
+++ Return: true/false
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int failure_aqua(date_t d){
int i;
int yv[10] = {  2,  2,  2,  2,  2,  2,  2,  2,  2,  2 };
int mv[10] = {  7,  7,  8,  8,  8,  8,  8,  8,  8,  9 };
int dv[10] = { 30, 31,  1,  2,  3,  4,  5,  6,  7, 13 };

  for (i=0; i<10; i++){

    if (d.year == 2000+yv[i] && d.month == mv[i] && d.day == dv[i]){
      #ifdef FORCE_DEBUG
      printf("AQUA outage. ");
      #endif
      return true;
    }
  }

  return false;
}


/** This function reads MODIS geometa tables (MOD03/MYD03) and performs an
+++ initial selection. Night granules are discarded (flagged invalid). 
+++ Granules at high latitudes are also discarded. Gring coordinates are
+++ sorted if not in UL-UR-LR-LL order. Gring coordinates of granules that
+++ extend between the Datum boundary are also modified.
--- fname: filename of geometa table
--- fid:   MODIS file IDs (returned)
--- gr:    MODIS Gring coordinates (returned)
--- v:     valid/invalid state (returned)
--- nl:    number of files (returned)
--- nv:    number of valid files (returned)
+++ Return:  SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int read_modis_geometa(char *fname, char ***fid, float ****gr, bool **v, int *nl, int *nv){
char buffer[NPOW_10] = "\0";
char *tokenptr = NULL;
char *separator = ",";
FILE *fp = NULL;
int l, g, k, nline = 0, nvalid = 0;
char **id = NULL;
bool *valid = NULL;
float ***bbox = NULL;
float ***gring = NULL;
float tmp, min[3];


  // allocate data
  alloc_2D((void***)&id, NPOW_10, NPOW_10, sizeof(char));
  alloc_3D((void****)&bbox,  NPOW_10, 2, 2, sizeof(float));
  alloc_3D((void****)&gring, NPOW_10, 2, 4, sizeof(float));
  alloc((void**)&valid, NPOW_10, sizeof(bool));


  // open file
  if (!(fp = fopen(fname, "r"))){
    printf("unable to open geometa file. "); return FAILURE;}


  // read line by line
  while (fgets(buffer, NPOW_10, fp) != NULL){

    // skip header
    if (buffer[0] == '#') continue;

    k = 0;

    tokenptr = strtok(buffer, separator);

    while (tokenptr != NULL){

      if (k == 0){ 
        if (strlen(tokenptr) > NPOW_10-1){
          printf("cannot copy, string too long.\n"); return FAILURE;
        } else { strncpy(id[nline], tokenptr, strlen(tokenptr)); id[nline][strlen(tokenptr)] = '\0';}
      } else if (k == 4){
        if (strcmp(tokenptr, "D") == 0 || strcmp(tokenptr, "B") == 0){
          valid[nline] = true;  // day image
          nvalid++;
        } else {
          valid[nline] = false; // night image
        }
      } else if (k ==  5){ bbox[nline][0][1]  = atof(tokenptr); // E
      } else if (k ==  6){ bbox[nline][1][0]  = atof(tokenptr); // N
      } else if (k ==  7){ bbox[nline][1][1]  = atof(tokenptr); // S
      } else if (k ==  8){ bbox[nline][0][0]  = atof(tokenptr); // W
      } else if (k ==  9){ gring[nline][0][0] = atof(tokenptr); // Gring X 1
      } else if (k == 10){ gring[nline][0][1] = atof(tokenptr); // Gring X 2
      } else if (k == 11){ gring[nline][0][2] = atof(tokenptr); // Gring X 3
      } else if (k == 12){ gring[nline][0][3] = atof(tokenptr); // Gring X 4
      } else if (k == 13){ gring[nline][1][0] = atof(tokenptr); // Gring Y 1
      } else if (k == 14){ gring[nline][1][1] = atof(tokenptr); // Gring Y 2
      } else if (k == 15){ gring[nline][1][2] = atof(tokenptr); // Gring Y 3
      } else if (k == 16){ gring[nline][1][3] = atof(tokenptr); // Gring Y 4
      }

      tokenptr = strtok(NULL, separator);
      k++;
    }

    nline++;

  }

  // swap coordinates if in reverse order, 
  // Aqua GRING coords are in reverse order because of S-N orbit
  for (l=0; l<nline; l++){

    if (!valid[l]) continue;

    // Granules at high latitudes are problematic.. Skip them
    if ((bbox[l][0][0] == -180 && bbox[l][0][1] == 180) || 
            (fabs(bbox[l][1][0]) > 89 || fabs(bbox[l][1][1]) > 89)){
      valid[l] = false;
      nvalid--;
      continue;
    }

    // Granules that extend between the Datum boundary are problematic
    // Correct for this; note that all coordinates are stored relative to GRing1
    // GRing x-coords can be > 180° or < -180° after this
    for (g=1; g<4; g++){

      min[0] = fabs(gring[l][0][0]-(gring[l][0][g]));
      min[1] = fabs(gring[l][0][0]-(gring[l][0][g]+360));
      min[2] = fabs(gring[l][0][0]-(gring[l][0][g]-360));

      if (min[1] < min[0] && min[1] < min[2]) gring[l][0][g] += 360;
      if (min[2] < min[0] && min[2] < min[1]) gring[l][0][g] -= 360;

    }

    // re-order GRing coordinates, such that they all are in UL-UR-LR-LL order
    if (gring[l][0][2] < gring[l][0][0] && gring[l][1][2] > gring[l][1][0]){
      tmp = gring[l][0][0]; gring[l][0][0] = gring[l][0][2]; gring[l][0][2] = tmp;
      tmp = gring[l][1][0]; gring[l][1][0] = gring[l][1][2]; gring[l][1][2] = tmp;
      tmp = gring[l][0][1]; gring[l][0][1] = gring[l][0][3]; gring[l][0][3] = tmp;
      tmp = gring[l][1][1]; gring[l][1][1] = gring[l][1][3]; gring[l][1][3] = tmp;
    }
    
    //#ifdef FORCE_DEBUG
    //printf("(reordered) GRING:\n");
    //for (g=0; g<4; g++) printf("GRING %d: X/Y %+.2f/%+.2f\n", g, gring[l][0][g], gring[l][1][g]);
    //#endif

  }

  #ifdef FORCE_DEBUG
  printf("%d lines, %d valid.\n", nline, nvalid);
  #endif

  fclose(fp);

  free_3D((void***)bbox, NPOW_10, 2);

  *nl  = nline;
  *nv  = nvalid;
  *gr  = gring;
  *v   = valid;
  *fid = id;

  return SUCCESS;
}


/** This function identifies MODIS granules that intersect with a pair of
+++ coordinates (+- 0.75°). The spatial selection is based on the Gring 
+++ coordinates of the MODIS granules. The function returns the number of
+++ intersecting granules, as well as a pointer to these elements.
--- lat:    latitude of input coordinate
--- lon:    longitude of input coordinate
--- gr:     MODIS Gring coordinates
--- v:      valid/invalid state
--- nl:     number of granules
--- nv:     number of valid granules
--- ptr:    pointer to intersecting elements (returned)
+++ Return: number of intersecting elements
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int modis_intersect(float lat, float lon, float ***gr, bool *v, int nl, int nv, int **ptr){
int *p = NULL;
int l, j, k = 0;
float wlon, elon, nlat, slat, dboff[3] = { 0, 360, -360 };


  alloc((void**)&p, nv, sizeof(int));

  for (l=0; l<nl; l++){

    if (!v[l]) continue;

    for (j=0; j<3; j++){

      wlon = (lon-0.75)+dboff[j]; elon = (lon+0.75)+dboff[j];
      nlat = lat+0.75; slat = lat-0.75;

      if ((wlon < (gr[l][0][1]*(gr[l][1][2]-nlat)+
          gr[l][0][2]*(nlat-gr[l][1][1]))/(gr[l][1][2]-gr[l][1][1]) || 
         wlon < (gr[l][0][1]*(gr[l][1][2]-slat)+
          gr[l][0][2]*(slat-gr[l][1][1]))/(gr[l][1][2]-gr[l][1][1]))
          &&
        (elon > (gr[l][0][0]*(gr[l][1][3]-nlat)+
          gr[l][0][3]*(nlat-gr[l][1][0]))/(gr[l][1][3]-gr[l][1][0]) || 
         elon > (gr[l][0][0]*(gr[l][1][3]-slat)+
          gr[l][0][3]*(slat-gr[l][1][0]))/(gr[l][1][3]-gr[l][1][0]))
          &&
        (nlat > (gr[l][1][3]*(gr[l][0][2]-wlon)+
          gr[l][1][2]*(wlon-gr[l][0][3]))/(gr[l][0][2]-gr[l][0][3]) || 
         nlat > (gr[l][1][3]*(gr[l][0][2]-elon)+
          gr[l][1][2]*(elon-gr[l][0][3]))/(gr[l][0][2]-gr[l][0][3]))
          &&
        (slat < (gr[l][1][0]*(gr[l][0][1]-wlon)+
          gr[l][1][1]*(wlon-gr[l][0][0]))/(gr[l][0][1]-gr[l][0][0]) || 
         slat < (gr[l][1][0]*(gr[l][0][1]-elon)+
          gr[l][1][1]*(elon-gr[l][0][0]))/(gr[l][0][1]-gr[l][0][0]))){
        p[k] = l;
        k++;
      }
    }
  }

  *ptr = p;

  return k;
}


/** This function identifies an intersecting MODIS granule that completely
+++ contains a pair of coordinates (+- 0.75°). The spatial selection is 
+++ based on the Gring coordinates of the MODIS granules. If no such gra-
+++ nule exists, -1 is returned, otherwise the function returns this ele-
+++ ment.
--- lat:    latitude of input coordinate
--- lon:    longitude of input coordinate
--- gr:     MODIS Gring coordinates
--- ni:     number of intersecting granules
--- ptr:    pointer to intersecting elements
+++ Return: number of intersecting elements
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int modis_inside(float lat, float lon, float ***gr, int nintersect, int *ptr){
int l, j;
float wlon, elon, nlat, slat, dboff[3] = { 0, 360, -360 };


  // return the first granule that completely contains the frame
  for (l=0; l<nintersect; l++){

    for (j=0; j<3; j++){

      wlon = (lon-0.75)+dboff[j]; elon = (lon+0.75)+dboff[j];
      nlat = lat+0.75; slat = lat-0.75;

      if ((wlon > (gr[ptr[l]][0][0]*(gr[ptr[l]][1][3]-nlat)+
          gr[ptr[l]][0][3]*(nlat-gr[ptr[l]][1][0]))/(gr[ptr[l]][1][3]-gr[ptr[l]][1][0])
          && 
        wlon > (gr[ptr[l]][0][0]*(gr[ptr[l]][1][3]-slat)+
          gr[ptr[l]][0][3]*(slat-gr[ptr[l]][1][0]))/(gr[ptr[l]][1][3]-gr[ptr[l]][1][0])
          && 
        elon < (gr[ptr[l]][0][1]*(gr[ptr[l]][1][2]-nlat)+
          gr[ptr[l]][0][2]*(nlat-gr[ptr[l]][1][1]))/(gr[ptr[l]][1][2]-gr[ptr[l]][1][1])
          && 
        elon < (gr[ptr[l]][0][1]*(gr[ptr[l]][1][2]-slat)+
          gr[ptr[l]][0][2]*(slat-gr[ptr[l]][1][1]))/(gr[ptr[l]][1][2]-gr[ptr[l]][1][1])
          && 
        nlat < (gr[ptr[l]][1][0]*(gr[ptr[l]][0][1]-wlon)+
          gr[ptr[l]][1][1]*(wlon-gr[ptr[l]][0][0]))/(gr[ptr[l]][0][1]-gr[ptr[l]][0][0])
          && 
        slat > (gr[ptr[l]][1][3]*(gr[ptr[l]][0][2]-wlon)+
          gr[ptr[l]][1][2]*(wlon-gr[ptr[l]][0][3]))/(gr[ptr[l]][0][2]-gr[ptr[l]][0][3])
          && 
        nlat < (gr[ptr[l]][1][0]*(gr[ptr[l]][0][1]-elon)+
          gr[ptr[l]][1][1]*(elon-gr[ptr[l]][0][0]))/(gr[ptr[l]][0][1]-gr[ptr[l]][0][0])
          && 
        slat > (gr[ptr[l]][1][3]*(gr[ptr[l]][0][2]-elon)+
          gr[ptr[l]][1][2]*(elon-gr[ptr[l]][0][3]))/(gr[ptr[l]][0][2]-gr[ptr[l]][0][3]))){

        return ptr[l];

      }

    }
  }

  // if no granule completely contains the frame: return -1
  return -1;
}


/** This function identifies the MODIS data that should be processed, then
+++ downloads them, reads them, and computes the average for each reques-
+++ ted coordinate.
--- dir_geo:   Directory of MODIS geometa tables
--- dir_hdf:   Directory of MODIS water vapor data
--- d_now:     Date
--- sen:       Sensor (TERRA/AQUA MODIS)
--- nc:        Number of coordinates
--- COO:       Coordinate array
--- avg:       Water vapor averages
--- count:     Number of pixels for averaging
+++ Return:    void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void compile_modis_wvp(char *dir_geo, char *dir_hdf, date_t d_now, char *sen, int nc, float **COO, double **avg, double **count, char *key){
char geoname[NPOW_10];
char pattern[NPOW_10], ftp_pattern[NPOW_10], doy[4];
char httplist[NPOW_10];
char loclist[NPOW_10];
char basename[NPOW_10];
char httpname[NPOW_10];
char fullname[NPOW_10];
char buffer[NPOW_10];
int nchar;
int curl;
float ***gr = NULL; // gring coords: (0/0-3): Lon/ul-ur-lr-ll, (1/0-3): Lat/ul-ur-lr-ll
char **id = NULL;
bool *v = NULL;
int *ptr = NULL;
char *str = NULL;
int nl, nv, ni;
int i, c, p;
int y, yc, yct, x, xc, xct, pos, posc;
int try__;
GDALDatasetH hdfDS, subDS;
GDALRasterBandH band;
char **sds = NULL, **metadata = NULL;
char KeyName[NPOW_10];
char *sdsname = NULL;
int nx, ny, nxc, nyc;
int yoff, xoff;
float voff, vscl, val;
int fill;
short *PW = NULL;
small *CLD = NULL;
float *LAT = NULL;
float *LON = NULL;
int pcld = 6, pwvp = 7, plat = 12, plon = 13;
bool bit[8];
FILE *fp = NULL;
double sumwvp, ctrwvp, *average = NULL, *ctrall = NULL;
float wlon, elon, nlat, slat;
bool ok;
const char *separator = ",";


  // allocate and initialize wvp averages and pixel counts
  alloc((void**)&average, nc, sizeof(double));
  alloc((void**)&ctrall,  nc, sizeof(double));
  for (c=0; c<nc; c++) average[c] = 9999;

  // geomate filename
  nchar = snprintf(geoname, NPOW_10, "%s/%s03_%4d-%02d-%02d.txt", 
    dir_geo, sen, d_now.year, d_now.month, d_now.day);
  if (nchar < 0 || nchar >= NPOW_10){ 
    printf("Buffer Overflow in assembling filename\n"); exit(1);}


  // if geometa file doesn't exist, return
  if (!fileexist(geoname)){ *avg = average; *count = ctrall; return;}


  GDALAllRegister();

  // read the geometa table
  if (read_modis_geometa(geoname, &id, &gr, &v, &nl, &nv) != SUCCESS){
    printf("compiling wvp failed.\n"); exit(1);}

  #ifdef FORCE_DEBUG
  printf("%d valid lines in geometa\n", nv);
  #endif

  // if no granule is valid, return
  if (nv == 0){ *avg = average; *count = ctrall; return;}


  // do for each requested coordinate
  for (c=0; c<nc; c++){

    sumwvp = ctrwvp = 0;

    // estimate average of wvp in a 1.5° x 1.5° box
    wlon = COO[0][c]-0.75; elon = COO[0][c]+0.75;
    nlat = COO[1][c]+0.75; slat = COO[1][c]-0.75;

    // get intersecting granules
    if ((ni = modis_intersect(COO[1][c], COO[0][c], gr, v, 
                                  nl, nv, &ptr)) < 1) continue;

    #ifdef FORCE_DEBUG
    printf("requested box: UL %.2f/%.2f, LR %.2f/%.2f\n", wlon, nlat, elon, slat);
    printf("%d intersecting granules\n", ni);
    #endif

    // if more than one was found, check if one is completely inside, then use only this one
    if (ni > 1){
      if ((p = modis_inside(COO[1][c], COO[0][c], gr, 
                         ni, ptr)) > -1){
        ptr[0] = p;
        ni = 1;

        #ifdef FORCE_DEBUG
        printf("one granule contains the box completely.\n");
        #endif

      }
    }


    // for every remaining granule: get correct HDF name, download if not there and process
    for (i=0; i<ni; i++){

      // appr. HDF name with wildcards
      strncpy(pattern, sen, 3);
      strncpy(pattern+3, "05_L2", 5);
      strncpy(pattern+8, id[ptr[i]]+5, 14);
      pattern[22] = '\0';
      strncpy(doy, pattern+14, 3); doy[3] = '\0';
      nchar = snprintf(ftp_pattern, NPOW_10, 
        "https://ladsweb.modaps.eosdis.nasa.gov/archive/allData/61/%s05_L2/%4d/%s/%s*", 
        sen, d_now.year, doy, pattern);
      if (nchar < 0 || nchar >= NPOW_10){ 
        printf("Buffer Overflow in assembling filename\n"); exit(1);}

      // download file listing
      try__ = 0;
      nchar = snprintf(httplist, NPOW_10, 
        "https://ladsweb.modaps.eosdis.nasa.gov/archive/allData/61/%s05_L2/%4d/%s.csv",
        sen, d_now.year, doy);
      if (nchar < 0 || nchar >= NPOW_10){ 
        printf("Buffer Overflow in assembling filename\n"); exit(1);}

      nchar = snprintf(loclist, NPOW_10, 
        "%s/%s05_L2-%04d-%s.csv", dir_hdf, sen, d_now.year, doy);
      if (nchar < 0 || nchar >= NPOW_10){ 
        printf("Buffer Overflow in assembling filename\n"); exit(1);}

      while (try__ < 10 && !fileexist(loclist)){
        curl = download_file(httplist, loclist, key);
        printf("\b\b%2d", try__+1);
        if (curl != 0) printf("; curl: %02d. fail. ", curl);
        if (curl == 78) try__ = 99;
        try__++; fflush(stdout);
      }

      if (!fileexist(loclist)) continue;


      if ((fp = fopen(loclist, "r")) == NULL){
        printf("Unable to open file list!\n"); continue;}
      ok = false;

      while (fgets(buffer, NPOW_10, fp) != NULL){
        if (strstr(buffer, pattern) != NULL){
          str = strtok(buffer, separator);
          if (strlen(str) > NPOW_10-1){
            printf("cannot copy, string too long.\n"); exit(1);
          } else { strncpy(basename, str, strlen(str)); basename[strlen(str)] = '\0';}
          ok = true;
        }
      }

      fclose(fp);

      if (!ok){
        printf("Unable to locate name in filelist!\n"); continue;}

      nchar = snprintf(fullname, NPOW_10, "%s/%s", dir_hdf, basename);
      if (nchar < 0 || nchar >= NPOW_10){ 
        printf("Buffer Overflow in assembling filename\n"); exit(1);}
      
      nchar = snprintf(httpname, NPOW_10, 
        "https://ladsweb.modaps.eosdis.nasa.gov/archive/allData/61/%s05_L2/%4d/%s/%s", 
        sen, d_now.year, doy, basename);
      if (nchar < 0 || nchar >= NPOW_10){ 
        printf("Buffer Overflow in assembling filename\n"); exit(1);}


      for (try__=0; try__ < 10; try__++){

        if (fileexist(fullname)) break;

        #ifdef FORCE_DEBUG
        printf("Download file (%s). Try # %2d\n", pattern, try__);
        #endif

        curl = download_file(httpname, fullname, key);
        printf("\b\b%2d", try__+1);
        if (curl != 0) printf("; curl: %02d. fail. ", curl);
        if (curl == 78) try__ = 99;

        if (fileexist(fullname)) break;

      }


      if (!fileexist(fullname)) continue;

      // open input dataset
      if ((hdfDS = GDALOpen(fullname, GA_ReadOnly)) == NULL){
        printf("unable to open image\n"); exit(1);
      } else {
        //free((void*)hdfname); hdfname = NULL;
      }

      // get SDS listing
      sds = GDALGetMetadata(hdfDS, "SUBDATASETS");
      if (CSLCount(sds) == 0){
        printf("unable to retrieve SDS list.\n"); exit(1);}



      // read NIR water vapour retrieval
      nchar = snprintf(KeyName, NPOW_10, "SUBDATASET_%d_NAME", pwvp);
      if (nchar < 0 || nchar >= NPOW_10){ 
        printf("Buffer Overflow in assembling hdf sds name\n"); exit(1);}

      sdsname = CPLStrdup(CSLFetchNameValue(sds, KeyName));
      subDS = GDALOpen(sdsname, GA_ReadOnly);
      nx = GDALGetRasterXSize(subDS); 
      ny = GDALGetRasterYSize(subDS);
      alloc((void**)&PW, nx*ny, sizeof(short));
      band = GDALGetRasterBand(subDS, 1);
      if (GDALRasterIO(band, GF_Read,  0, 0, nx, ny, 
            PW, nx, ny, GDT_Int16, 0, 0) == CE_Failure){
        printf("could not read image. "); return;}
      CPLFree(sdsname);

      // read some metadata
      metadata = GDALGetMetadata(subDS, "GEOLOCATION");
      yoff = atoi(CSLFetchNameValue(metadata,"LINE_OFFSET"));
      xoff = atoi(CSLFetchNameValue(metadata,"PIXEL_OFFSET"));
      //CSLDestroy(metadata);
      
      metadata = GDALGetMetadata(subDS, NULL);
      voff = atof(CSLFetchNameValue(metadata,"add_offset"));
      vscl = atof(CSLFetchNameValue(metadata,"scale_factor"));
      fill = atoi(CSLFetchNameValue(metadata,"_FillValue"));
      //CSLDestroy(metadata);
      GDALClose(subDS);

      // read latitude
      nchar = snprintf(KeyName, NPOW_10, "SUBDATASET_%d_NAME", plat);
      if (nchar < 0 || nchar >= NPOW_10){ 
        printf("Buffer Overflow in assembling hdf sds name\n"); exit(1);}

      sdsname = CPLStrdup(CSLFetchNameValue(sds, KeyName));
      subDS = GDALOpen(sdsname, GA_ReadOnly);
      nxc = GDALGetRasterXSize(subDS); 
      nyc = GDALGetRasterYSize(subDS);
      alloc((void**)&LAT, nxc*nyc, sizeof(float));
      band = GDALGetRasterBand(subDS, 1);
      if (GDALRasterIO(band, GF_Read,  0, 0, nxc, nyc, 
            LAT, nxc, nyc, GDT_Float32, 0, 0) == CE_Failure){
        printf("could not read image. "); return;}
      GDALClose(subDS);
      CPLFree(sdsname);

      // read longitude
      nchar = snprintf(KeyName, NPOW_10, "SUBDATASET_%d_NAME", plon);
      if (nchar < 0 || nchar >= NPOW_10){ 
        printf("Buffer Overflow in assembling hdf sds name\n"); exit(1);}
        
      sdsname = CPLStrdup(CSLFetchNameValue(sds, KeyName));
      subDS = GDALOpen(sdsname, GA_ReadOnly);
      alloc((void**)&LON, nxc*nyc, sizeof(float));
      band = GDALGetRasterBand(subDS, 1);
      if (GDALRasterIO(band, GF_Read,  0, 0, nxc, nyc, 
            LON, nxc, nyc, GDT_Float32, 0, 0) == CE_Failure){
        printf("could not read image. "); return;}
      GDALClose(subDS);
      CPLFree(sdsname);

      // read cloud state
      nchar = snprintf(KeyName, NPOW_10, "SUBDATASET_%d_NAME", pcld);
      if (nchar < 0 || nchar >= NPOW_10){ 
        printf("Buffer Overflow in assembling hdf sds name\n"); exit(1);}

      sdsname = CPLStrdup(CSLFetchNameValue(sds, KeyName));
      subDS = GDALOpen(sdsname, GA_ReadOnly);
      alloc((void**)&CLD, nx*ny, sizeof(small));
      band = GDALGetRasterBand(subDS, 1);
      if (GDALRasterIO(band, GF_Read,  0, 0, nx, ny, 
            CLD, nx, ny, GDT_Byte, 0, 0) == CE_Failure){
        printf("could not read image. "); return;}
      GDALClose(subDS);
      CPLFree(sdsname);

      //CSLDestroy(sds);
      GDALClose(hdfDS);


      // sum up precipitable water for mean
      for (y=yoff, yc=0, yct=1; y<ny; y++){
      for (x=xoff, xc=0, xct=1; x<nx; x++){
       
        pos  = nx*y+x;
        posc = nxc*yc+xc;

        //if in frame
        if (LON[posc] >= wlon && LON[posc] <= elon &&
          LAT[posc] <= nlat && LAT[posc] >= slat){

          if (PW[pos] != fill){

            int2bit(CLD[pos], bit, 0, 8);

            if (bit[2] && bit[4]){
              val = vscl*(PW[pos]-voff);
              sumwvp += val;
              ctrwvp++;
            }

          }
          ctrall[c]++;
        }

        if ((xct++) >= 5){ xct = 1; xc++;}
      }
      if ((yct++) >= 5){ yct = 1; yc++;}
      }

      free((void*)PW); free((void*)CLD);
      free((void*)LAT); free((void*)LON);

    }

    // only calculate average if at least 10% of in-frame pixels were valid
    if (ctrall[c] > 0 && ctrwvp >= ctrall[c]/10){
      average[c] = sumwvp/ctrwvp;
    } else average[c] = 9999;

    free((void*)ptr);
  }

  // free memory
  free_3D((void***)gr, NPOW_10, 2);
  free_2D((void**)id, NPOW_10);
  free((void*)v);

  *avg   = average;
  *count = ctrall;
  
  return;
}


/** This function chooses, which water vapor estimate to use. This is 
+++ based on data availability and the number of pixels that were used
+++ for averaging.
--- aqua:   AQUA commissioning flag
--- nc:     Number of coordinates
--- WVP:    Water vapor array
--- SEN:    Sensor source array
--- modavg: TERRA averages
--- mydavg: AQUA  averages
--- modctr: TERRA number of pixels
--- mydctr: AQUA  number of pixels
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void choose_modis_wvp(bool aqua, int nc, float *WVP, char **SEN, double *modavg, double *mydavg, double *modctr, double *mydctr){
int c;
float ctr = 0;


  for (c=0; c<nc; c++){
    if (!aqua){
      WVP[c] = modavg[c];
      if (WVP[c] < 9999){ strncpy(SEN[c], "MOD", 3); SEN[c][3] = '\0';}
    } else {
      if (modavg[c] < 9999 && mydavg[c] < 9999){
        if (modctr[c] >= mydctr[c]){
          WVP[c] = modavg[c];
          if (WVP[c] < 9999){ strncpy(SEN[c], "MOD", 3); SEN[c][3] = '\0';}
        } else {
          WVP[c] = mydavg[c];
          if (WVP[c] < 9999){ strncpy(SEN[c], "MYD", 3); SEN[c][3] = '\0';}
        }
      } else if (modavg[c] < 9999 && mydavg[c] >= 9999){
        WVP[c] = modavg[c];
        if (WVP[c] < 9999){ strncpy(SEN[c], "MOD", 3); SEN[c][3] = '\0';}
      } else if (modavg[c] >= 9999 && mydavg[c] < 9999){
        WVP[c] = mydavg[c];
        if (WVP[c] < 9999){ strncpy(SEN[c], "MYD", 3); SEN[c][3] = '\0';}
      }
    }
    if (WVP[c] < 9999) ctr++;
  }

  printf("%5.1f%% of useful frames.\n", ctr/nc*100.0);

  return;
}


/** This function writes a daily water vapor Look-Up-Table.
--- fname:  LUT filename
--- nc:     Number of coordinates
--- COO:    Coordinate array
--- WVP:    Water vapor array
--- SEN:    Sensor source array
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void write_wvp_lut(char *fname, int nc, float **COO, float *WVP, char **SEN){
FILE *fp;
int c;

  if ((fp = fopen(fname, "w")) == NULL){
    printf("couldn't open file for writing.\n"); exit(1);}

  for (c=0; c<nc; c++){
    fprintf(fp, "%.4f %.4f %f %s\n", COO[0][c], COO[1][c], WVP[c], SEN[c]);
  }

  fclose(fp);

  return;
}


/** This function writes a climatology water vapor Look-Up-Table.
--- dir_wvp: Output directory
--- nc:      Number of coordinates
--- COO:     Coordinate array
--- AVG:     Water vapor average array
+++ Return:  void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void write_avg_table(char *dir_wvp, int nc, float **COO, double ***AVG){
FILE *fp;
int c, m;
char fname[NPOW_10];
int nchar; 


  for (m=0; m<12; m++){

    nchar = snprintf(fname, NPOW_10, "%s/WVP_0000-%02d-00.txt", dir_wvp, m+1);
    if (nchar < 0 || nchar >= NPOW_10){ 
      printf("Buffer Overflow in assembling file name\n"); exit(1);}
    
    if ((fp = fopen(fname, "w")) == NULL){
      printf("couldn't open file for writing.\n"); exit(1);}
    for (c=0; c<nc; c++){
      fprintf(fp, "%.4f %.4f %f %f %.0f\n", COO[0][c], COO[1][c], 
        AVG[0][m][c], AVG[1][m][c], AVG[2][m][c]);
    }
    fclose(fp);
  }

  return;
}


/** This function reads a water vapor Look-Up-Table (daily or climatology)
--- fname:  LUT filename
--- nc:     Number of coordinates
--- COO:    Coordinate array
--- WVP:    Water vapor array
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void read_wvp_lut(char *fname, int nc, float **COO, float *WVP){
FILE *fp;
char  buffer[NPOW_10] = "\0";
char *tokenptr = NULL;
char *separator = " ";
int c;
float x, y;

  fp = fopen(fname, "r");

  // process line by line
  for (c=0; c<nc; c++){

    if (fgets(buffer, NPOW_10, fp) == NULL){
      printf("invalid wvp table!\n"); exit(1);}

    tokenptr = strtok(buffer, separator);
    x = atof(tokenptr); tokenptr = strtok(NULL, separator);
    y = atof(tokenptr); tokenptr = strtok(NULL, separator);

    if (fabs(x-COO[0][c]) > 0.0001 || fabs(y-COO[1][c]) > 0.0001){
      printf("Invalid wvp table! Coordinates are messed up..\n"); exit(1);}

    WVP[c] = atof(tokenptr);

  }

  fclose(fp);

  return;
}


/** This function creates a daily water vapor Look-Up-Table for the given
+++ date from MODIS data, which are downloaded from LAADS.
--- dir_geo:   Directory of MODIS geometa tables
--- dir_hdf:   Directory of MODIS water vapor data
--- tablename: Name of LUT
--- d_now:     Date
--- nc:        Number of coordinates
--- COO:       Coordinate array
--- WVP:       Water vapor array
--- SEN:       Sensor source array
+++ Return:    void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void create_wvp_lut(char *dir_geo, char *dir_hdf, char *tablename, date_t d_now, int nc, float **COO, float *WVP, char **SEN, char *key){
char ftpname[NPOW_10], locname[NPOW_10];
int nchar;
int c, try__, nvalid, curl;
bool aqua = false;
double *modavg, *mydavg, *modctr, *mydctr;


  // test if AQUA was commissioned
  if (d_now.year*10000+d_now.month*100+d_now.day  >= 20020703) aqua = true;

  // initialize precipitable water with fill
  for (c=0; c<nc; c++){
    WVP[c] = 9999;
    strncpy(SEN[c], "TBD", 3); SEN[c][3] = '\0';
  }

  // if TERRA geometa doesn't exist: download
  if (!failure_terra(d_now)) try__ = 0; else try__ = 99;

  nchar = snprintf(ftpname, NPOW_10, 
    "https://ladsweb.modaps.eosdis.nasa.gov/archive/geoMeta/6/TERRA/%4d/MOD03_%4d-%02d-%02d.txt", 
    d_now.year, d_now.year, d_now.month, d_now.day);
  if (nchar < 0 || nchar >= NPOW_10){ 
    printf("Buffer Overflow in assembling file name\n"); exit(1);}

  nchar = snprintf(locname, NPOW_10, 
    "%s/MOD03_%4d-%02d-%02d.txt", 
    dir_geo, d_now.year, d_now.month, d_now.day);
  if (nchar < 0 || nchar >= NPOW_10){ 
    printf("Buffer Overflow in assembling file name\n"); exit(1);}


  if (try__ < 1){
    if (!fileexist(locname)){ printf("Download TERRA geometa. Try #   ");
    } else printf("TERRA geometa exists.");
  }

  while (try__ < 10 && !fileexist(locname)){
    curl = download_file(ftpname, locname, key);
    printf("\b\b%2d", try__+1);
    if (curl != 0) printf("; curl: %02d. fail. ", curl);
    if (curl == 78) try__ = 99;
    try__++; fflush(stdout);
  }
  printf("\n");

  // compile water vapour
  compile_modis_wvp(dir_geo, dir_hdf, d_now, "MOD", nc, COO, &modavg, &modctr, key);

  if (aqua){

    // if AQUA geometa doesn't exist: download
    if (!failure_aqua(d_now)) try__ = 0; else try__ = 99;

    nchar = snprintf(ftpname, NPOW_10, 
      "https://ladsweb.modaps.eosdis.nasa.gov/archive/geoMeta/6/AQUA/%4d/MYD03_%4d-%02d-%02d.txt", 
      d_now.year, d_now.year, d_now.month, d_now.day);
    if (nchar < 0 || nchar >= NPOW_10){ 
      printf("Buffer Overflow in assembling file name\n"); exit(1);}

    nchar = snprintf(locname, NPOW_10, 
      "%s/MYD03_%4d-%02d-%02d.txt", 
      dir_geo, d_now.year, d_now.month, d_now.day);
    if (nchar < 0 || nchar >= NPOW_10){ 
      printf("Buffer Overflow in assembling file name\n"); exit(1);}
      
    if (try__ < 1){
      if (!fileexist(locname)){ printf("Download AQUA geometa. Try #   ");
      } else printf("AQUA geometa exists.");
    }

    while (try__ < 10 && !fileexist(locname)){
      curl = download_file(ftpname, locname, key);
      printf("\b\b%2d", try__+1);
      if (curl != 0) printf("; curl: %02d. fail. ", curl);
      if (curl == 78) try__ = 99;
      try__++; fflush(stdout);
    }
  }
  printf("\n");

  // compile water vapour
  if (aqua) compile_modis_wvp(dir_geo, dir_hdf, d_now, "MYD", nc, COO, &mydavg, &mydctr, key);

  // choose between TERRA and AQUA
  choose_modis_wvp(aqua, nc, WVP, SEN, modavg, mydavg, modctr, mydctr);

  // clean
  free((void*)modavg); free((void*)modctr);
  if (aqua){ free((void*)mydavg); free((void*)mydctr);}

  // write table
  for (c=0, nvalid=0; c<nc; c++) if (WVP[c] != 9999) nvalid++;
  if (nvalid > 0) write_wvp_lut(tablename, nc, COO, WVP, SEN);
  
  return;
}


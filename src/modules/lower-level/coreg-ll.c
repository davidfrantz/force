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
This file contains functions for supporting coregistration
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


/** The following code was adopted from the LSReg code, developed by Lin 
+++ Yan and David Roy at the Geospatial Sciences Center of Excellence, 
+++ South Dakota State University under NASA grant NNX17AB34G. 
+++ LSReg, Version: 2.0
+++ Copyright (C) 2018 Lin Yan & David Roy
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
+++ Yan, L., Roy, D.P., Zhang, H.K., Li, J., Huang, H. (2016). An automated
+++ approach for sub-pixel registration of Landsat-8 Operational Land 
+++ Imager (OLI) and Sentinel-2 Multi Spectral Instrument (MSI) imagery. 
+++ Remote Sensing, 8(6), 520. 
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


#include "coreg-ll.h"

#define MAX_PYRAMID_LAYER (10)
#define MIN_TIES_NUM (12)
#define MIN_SAM_THRESHOLD (0.985)

#define ABS(x)        ((x>=0)? (x):-(x))
#define MAX(a,b) (((a)>(b))? (a) : (b))
#define MIN(a,b) (((a)<(b))? (a) : (b))


typedef enum { enum_TRANSLATION = 1, enum_AFFINE = 2, enum_POLYNOMIAL = 3, enum_AUTO = 4 } enum_tranformation_type;

typedef struct{
  int n;
  double *i;
  double *j;
} poi_t;

typedef struct{
  poi_t target;
  poi_t base;
} tie_t;

typedef struct{
  int ntie;
  double coefs[12];
  double rmse;
} match_t;

int coreg(short **target, short *base, brick_t *QAI, float res, int nx, int ny, int nb, int band, short nodata);
int cumulative_scale(int toplayer, int *scales);
void free_pyramids(short ***pyramids_, int nlayer);
void build_pyramids(short *image, brick_t *QAI, int nx, int ny, short nodata, int nlayer, int *scales, short ***pyramids, int **nx_pyr, int **ny_pyr);
void build_pyramidlayer(int nx, int ny, short nodata, int scale, int *nx_new_, int *ny_new_, short *image_, short **pyramid);
poi_t points_of_interest(short ***pyramids_, int *nx_pyr, int *ny_pyr, short nodata, int iLayer, int iMinPOINum);
bool mask_and_base(short *target, short *base, int nx, int ny, small *mask, bool *pbImage1);
tie_t initial_matching(short ***pyramids_, int *nx_pyr, int *ny_pyr, int iLayer, int h, int max_h, double corr_threshold, poi_t *poi);
match_t depth_first_matching(short ***pyramids_, int *nx_pyr, int *ny_pyr, int toplayer, int *scales, int h_, int max_h_, double SAM, tie_t *init);
enum_tranformation_type choose_transform(int ntie); 
int transform_from_df(enum_tranformation_type transform_, tie_t *tie, double *coefs, double *rmse);
int transform_from_dm(enum_tranformation_type transform, float *parallaxmap_x, float *parallaxmap_y, float *corrmap, int nx_new, int ny_new, int step, double *coefs, double *rmse);
short *register_band(enum_tranformation_type transform, match_t *dm, short *image, int nx, int ny, short nodata);
short *register_quality(enum_tranformation_type transform, match_t *dm, short *image, int nx, int ny, short nodata);
match_t dense_matching(short ***pyramids_, int *nx_pyr, int *ny_pyr, short nodata, int layer, int toplayer, int *scales, int step, int h, int max_h_, double SAM, enum_tranformation_type transform, match_t *df);


/** This function is the private entry point to the LSReg coregistration.
--- target:  target image (to be registered)
--- base:    base image
--- QAI:     Quality Assurance Information
--- res:     resolution
--- nx:      number of columns
--- ny:      number of rows
--- nb:      number of bands
--- band:    band for tie point detection
--- nodata:  nodata value
+++ Return:  SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int coreg(short **target, short *base, brick_t *QAI, float res, int nx, int ny, int nb, int band, short nodata){
int h, max_h, b;
int i, j, p;
double SAM, SAM_original;
bool success = false;
short ***pyramids_ = NULL;
short **qai_ = NULL, qai_nodata;
int *nx_pyr = NULL;
int *ny_pyr = NULL;
poi_t poi;
tie_t init;
match_t df, dm;
int dmstep;
int nlayer, toplayer, scales[MAX_PYRAMID_LAYER];
enum_tranformation_type transform;
short band_value_thr = 1000;
int nland = 0;


  #ifdef FORCE_CLOCK 
  time_t TIME; time(&TIME);
  #endif

  if ((qai_ = get_bands_short(QAI)) == NULL) return FAILURE;
  qai_nodata = (short)get_brick_nodata(QAI, 0);
  
  // number of valid land pixels
  #pragma omp parallel private(j,p) shared(ny,nx,QAI,target,base,band,band_value_thr) reduction(+: nland) default(none)
  {

    #pragma omp for
    for (i=0; i<ny; i++){
    for (j=0; j<nx; j++){
      p = i*nx+j;
      if (get_off(QAI, p) || get_shadow(QAI, p) || get_cloud(QAI, p) > 0) continue;
      if (target[band][p] > band_value_thr && base[p] > band_value_thr) nland++;
    }
    }

  }

  // pixel step for dense matching
  dmstep = (int)(MIN(nx, ny) / 36.6 + 0.5f); // = 300 if image is sized 10980 x 10980
  dmstep = (dmstep != 0) ? dmstep : 1;
  
  // adjust with land proportion in image
  dmstep = (int)(nland/(float)(nx*ny)*dmstep);
  if (dmstep < 3) dmstep = 3; // at least 3 px

  #ifdef FORCE_DEBUG
  printf("dense matching step: %d\n", dmstep);
  #endif


  nlayer = 4;
  toplayer = nlayer-1;
  scales[0] = 1;
  scales[1] = 2; // changed from 3 -> pyramids: 10 20 40 80 meters
  scales[2] = 2;
  scales[3] = 2;

  h = 10;
  max_h = 20;
  SAM = 0.99;

  transform = enum_AFFINE;
  SAM_original = SAM;
  

  alloc((void**)&pyramids_, 2, sizeof(short**));
  build_pyramids(target[band], QAI, nx, ny, nodata, nlayer, scales, &pyramids_[0], &nx_pyr, &ny_pyr);
  free((void*)nx_pyr); free((void*)ny_pyr);
  build_pyramids(base,         QAI, nx, ny, nodata, nlayer, scales, &pyramids_[1], &nx_pyr, &ny_pyr);


  while (success == false){

    // detect POI on top layer
    poi = points_of_interest(pyramids_, nx_pyr, ny_pyr, nodata, toplayer, 2000);

    // initial matching on top layer
    init = initial_matching(pyramids_, nx_pyr, ny_pyr, toplayer, h, max_h, SAM, &poi);      // initail matching on top layer
    if (poi.n > 0){
      free((void*)poi.j); free((void*)poi.i); poi.n = 0;}

    // depth-first matching
    df = depth_first_matching(pyramids_, nx_pyr, ny_pyr, toplayer, scales, h, max_h, SAM, &init);    // depth-first matching on initial matched points; results stored at [pLSReg->pdCoefs, pLSReg->n_tie, pLSReg->dRMSE]

    if (init.target.n >0){
      free((void*)init.target.j); free((void*)init.target.i); init.target.n = 0;}
    if (init.base.n   >0){
      free((void*)init.base.j);   free((void*)init.base.i);   init.base.n   = 0;}

    #ifdef FORCE_DEBUG
    printf("Depth-First matching:\n");
    printf("%d tie points, RMSE: %f\n", df.ntie, df.rmse);
    #endif


    if (df.ntie >= MIN_TIES_NUM){
      success = true; // sufficient tie points found
      break;
    } else {
      if (SAM - 0.005 >= MIN_SAM_THRESHOLD){
        #ifdef FORCE_DEBUG
        printf(" Insufficient tie points: %d. SAM threshold = %.3f. Reset to %.3f.\n\n", df.ntie, SAM, SAM-0.005);
        #endif
        SAM -= 0.005; // reduce SAM threshold by 0.005 and redo matching
        if (ABS(SAM - MIN_SAM_THRESHOLD) <= 0.0001 && toplayer > 2){
          #ifdef FORCE_DEBUG
          printf(" Final try: pyramid layer reset to 2.\n\n");
          #endif
          SAM = SAM_original;
          toplayer = 2;
        }
      } else {
        break; // insufficient tie points found, report fail (success = false)
      }
    }

  }


  if (success == false){

    dm.ntie = 0;
    dm.coefs[0] = 0.0;
    dm.coefs[3] = 0.0;
    dm.rmse = 0.0;

  } else {

    // dense grid matching on bottom layer
    dm = dense_matching(pyramids_, nx_pyr, ny_pyr, nodata, 0, toplayer, scales, dmstep, 20, max_h, SAM, transform, &df);

    //if (dm.rmse > 0.5 || (dm.rmse > 0.3 && dm.ntie < 40)) success = false;
    if (sqrt(dm.coefs[0]*dm.coefs[0] + dm.coefs[3]*dm.coefs[3]) > 6) success = false;
    //if (dm.ntie < MIN_TIES_NUM) success = false;

  }

  free_pyramids(pyramids_, nlayer);
  free((void*)nx_pyr);
  free((void*)ny_pyr);

  printf("coreg (#tie, x/y shift, rmse): %d/%.2f/%.2f/%.2f", dm.ntie, dm.coefs[0]*res, dm.coefs[3]*res, dm.rmse);

  if (success){
    printf(" - good, ");
    for (b=0; b<nb; b++) target[b] = register_band(transform, &dm, target[b], nx, ny, nodata);
    qai_[0] = register_quality(transform, &dm, qai_[0], nx, ny, qai_nodata);
  } else {
    //printf(" - fail, ");
    printf(" coreg failed. Exit.\n");
    exit(1);
  }
  

  #ifdef FORCE_CLOCK
  proctime_print("coreg core (LSReg)", TIME);
  #endif

  if (success) return SUCCESS; else return CANCEL;
}


/** This function computes the product of all layer scales
--- toplayer: number of layers
--- scales:   scale of each layer
+++ Return:   product of all layer scales
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int cumulative_scale(int toplayer, int *scales){
int layer;
int cumscale = 1;

  for (layer=0; layer<toplayer; layer++)  cumscale *= scales[layer];

  return cumscale;
}


/** This function frees the pyramids
--- pyramids_: pyramids
--- nlayer:    number of pyramid layers
+++ Return:    void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void free_pyramids(short ***pyramids_, int nlayer){
int type, layer;  

  for (type=0; type<2; type++){
    for (layer=0; layer<nlayer; layer++) free((void*)pyramids_[type][layer]);
    free((void*)pyramids_[type]);
  }
  free((void*)pyramids_);
  
  return;
}
  

/** This function builds the pyramids
--- image:    full res image
--- QAI:      Quality Assurance Information
--- nx:       number of columns
--- ny:       number of rows
--- nodata:   nodata value
--- nlayer:   number of pyramid layers
--- scales:   scale of each layer
--- pyramids: pyramids
--- nx_pyr:   number of columns of pyramid layers
--- ny_pyr:   number of rows of pyramid layers
+++ Return:   void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void build_pyramids(short *image, brick_t *QAI, int nx, int ny, short nodata, int nlayer, int *scales, short ***pyramids, int **nx_pyr, int **ny_pyr){
int layer, p;
short **pyramids_ = NULL;
int *nx_ = NULL;
int *ny_ = NULL;


  #ifdef FORCE_CLOCK 
  time_t TIME; time(&TIME);
  #endif

  alloc((void**)&pyramids_, nlayer, sizeof(short*));
  alloc((void**)&nx_, nlayer, sizeof(int));
  alloc((void**)&ny_, nlayer, sizeof(int));

  alloc((void**)&pyramids_[0], nx*ny, sizeof(short));
  memmove(pyramids_[0], image, nx*ny*sizeof(short));

  #pragma omp parallel shared(ny,nx,QAI,pyramids_,nodata) default(none)
  {

    #pragma omp for
    for (p=0; p<nx*ny; p++){
      if (get_off(QAI, p) || get_shadow(QAI, p) || get_cloud(QAI, p) > 0) pyramids_[0][p] = nodata;
    }

  }
  
  nx_[0] = nx; ny_[0] = ny;
  
  #ifdef FORCE_DEBUG
  printf("dims of pyramids: %d %d\n", nx_[0], ny_[0]);
  #endif

  // generate pyramids
  for (layer=1; layer<nlayer; layer++){
    build_pyramidlayer(nx_[layer-1], ny_[layer-1], nodata, scales[layer], &nx_[layer], &ny_[layer], pyramids_[layer-1], &pyramids_[layer]);
    #ifdef FORCE_DEBUG
    printf("dims of pyramids: %d %d\n", nx_[layer], ny_[layer]);
    #endif
  }

  #ifdef FORCE_CLOCK
  proctime_print("building pyramids", TIME);
  #endif

  *nx_pyr = nx_;
  *ny_pyr = ny_;
  *pyramids = pyramids_;
  return;
}


/** This function builds one pyramid layer
--- nx:      number of columns
--- ny:      number of rows
--- nodata:  nodata value
--- scales:  scale of the layers
--- nx_new_: number of columns of pyramid layer
--- ny_new_: number of rows of pyramid layer
--- image_:  previous pyramid layer
--- pyramid: current pyramid layer
+++ Return:  void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void build_pyramidlayer(int nx, int ny, short nodata, int scale, int *nx_new_, int *ny_new_, short *image_, short **pyramid){
short *blurred_ = NULL, *pyramid_ = NULL;
int    FilterWidth;
float  *pfFilter = NULL;
double dFilterSigma;
int i, j, p, i_new, j_new, p_new;
int nx_new, ny_new;

  nx_new = (int)(nx / (float)scale + 1e-10);
  ny_new = (int)(ny / (float)scale + 1e-10);

  alloc((void**)&blurred_, nx*ny, sizeof(short));
  alloc((void**)&pyramid_, nx_new*ny_new, sizeof(short));

  // get Gaussian filter
  dFilterSigma = sqrt(scale / 2.0);
  pfFilter = GetGaussian(dFilterSigma, &FilterWidth);
  Conv2same(image_, blurred_, nx, ny, nodata, pfFilter, FilterWidth, scale);

  // resample
  #pragma omp parallel private(j,p,i_new,j_new,p_new) shared(nx,ny,nx_new,scale,image_,pyramid_,blurred_,nodata) default(none)
  {

    #pragma omp for
    for (i=0; i<=ny-scale; i+=scale){
    
      i_new = i/scale;

      for (j=0; j<=nx-scale; j+=scale){
        
        j_new = j/scale;
        p = i*nx+j;
        p_new = i_new*nx_new+j_new;

        if (image_[p] == nodata){
          pyramid_[p_new] = nodata;
        } else {
          pyramid_[p_new] = blurred_[p];
        }
        //if ((++j_new) >= nx_new) break;
      }
      //if ((++i_new) >= ny_new) break;
    }
    
  }

  free(blurred_); free(pfFilter);

  *nx_new_ = nx_new;
  *ny_new_ = ny_new;
  *pyramid = pyramid_;
  return;
}


/** This function builds a nodata mask and selects the base image from
+++ target or base
--- target:  target image
--- base:   base image
--- nx:       number of columns
--- ny:       number of rows
--- mask:     nodata mask
--- pbImage1: flag whether base is target
+++ Return:   logical value indicating if there is any valid value
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
bool mask_and_base(short *target, short *base, int nx, int ny, small *mask, bool *pbImage1){
int p;
float fStd1, fStd2;
int n = 0;


  *pbImage1 = true;

  for (p=0; p<ny*nx; p++){
    if (target[p] > 0 && base[p] > 0){
      n++;
      mask[p] = true;
    }
  }

  if (n == 0) return false;

  fStd1 = GetStd(target, ny*nx, mask);
  fStd2 = GetStd(base,   ny*nx, mask);

  *pbImage1 = (fStd1 <= fStd2) ? true : false;

  return true;
}


/** This function detects initial points of interest on the top layer
--- pyramids:   pyramids
--- nx_pyr:     number of columns of pyramid layers
--- ny_pyr:     number of rows of pyramid layers
--- nodata:     nodata value
--- layer:      layer for POI detection
--- iMinPOINum: minimum number of POIs
+++ Return:     points of interest 
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
poi_t points_of_interest(short ***pyramids_, int *nx_pyr, int *ny_pyr, short nodata, int layer, int iMinPOINum){
int i, j, p;
int wFilter7, wFilter2;
int h = 7, npoi;
float  *dFilter7 = NULL, *dFilter2 = NULL;
float  dMin, dMax, dMean;
short *target = NULL;
short *base   = NULL;
short *image = NULL;
float *img0 = NULL, *img1 = NULL, *img2 = NULL, *img3 = NULL;
float img_max;
int iIterNum;
bool base_is_target;
int nx, ny, nc;
small *mask = NULL;
poi_t poi;


  #ifdef FORCE_CLOCK 
  time_t TIME; time(&TIME);
  #endif
  
  //  printf("\npoi detection\n");

  target = pyramids_[0][layer];
  base   = pyramids_[1][layer];

  nx = nx_pyr[layer];
  ny = ny_pyr[layer];
  nc = nx*ny;


  alloc((void**)&mask, nc, sizeof(small));

  // get overlappping area mask and the image for POI detection (the one with smaller std)
  if (mask_and_base(target, base, nx, ny, mask, &base_is_target) == false){
    free(mask);
    poi.n = 0;
    return poi;
  }

  image = (base_is_target) ? target : base;
  ApplyMask(image, nc, mask, nodata);

  alloc((void**)&img0, nc, sizeof(float));
  alloc((void**)&img1, nc, sizeof(float));
  alloc((void**)&img2, nc, sizeof(float));
  alloc((void**)&img3, nc, sizeof(float));


  // convert input data to [0, 255] saved in img0
  img_max = 0;
  for (p=0; p<nc; p++){
    if (image[p] > img_max) img_max = image[p];
  }
  for (p=0; p<nc; p++) img0[p] = (image[p] < 0) ? 0 : image[p] / img_max * 255;

  dFilter7 = GetGaussian(0.7, &wFilter7);
  dFilter2 = GetGaussian(2,   &wFilter2);

  Conv2same_FLT_T(img0, img1, nx, ny, dFilter7, wFilter7);

  // calculate gradient x-y
  CalcGradient2D(img1, img2, img3, nx, ny); //img0 img1 img2 img3

  // calculate the 1-variable coefficients in the Forstner equations  
  CalcMultiply(img2, img3, img1, nc);
  CalcMultiply(img2, img2, img2, nc);
  CalcMultiply(img3, img3, img3, nc);

  Conv2same_FLT_T(img1, img0, nx, ny, dFilter2, wFilter2);  //img0=img2ab
  Conv2same_FLT_T(img2, img1, nx, ny, dFilter2, wFilter2);  //img1=img2a  
  Conv2same_FLT_T(img3, img2, nx, ny, dFilter2, wFilter2);  //img2=img2b      

  // calculate the Forstner equations
  CalcAdd(img1, img2, img3, nc);    //img3=(img2a+img2b)
  AddConst(img3, 0.01f, nc);      //img3=(img2a+img2b+0.01)
  CalcMultiply(img0, img0, img0, nc);    //img0 =img2ab^2
  CalcMultiply(img1, img2, img1, nc);  //img1=img2a.*img2b
  CalcSubtract(img1, img0, img2, nc);  //img2=(img2a.*img2b-img2ab^2)
  CalcDivide(img2, img3, img1, nc);    //img1=(img2a.*img2b-img2ab^2)/(img2a+img2b+0.01)

  // find local minima
  Conv2same_FLT_T(img1, img0, nx, ny, dFilter2, wFilter2);

  memset(img1, 0, sizeof(float));
  memset(img2, 0, sizeof(float));
  CalcGradient2D(img0, img1, img2, nx, ny);

  // find zero-crossing    
  CalcMinMaxMeanWithMask(img0, mask, &dMin, &dMax, &dMean, nc);

  // determine proper dMean value
  iIterNum = 0;
  do {
    npoi = 0;
    for (i=h; i<ny-h; i++){
    for (j=h; j<nx-h; j++){
      if (FindTargetValueInWindow(image, nx, ny, j, i, h, nodata)) continue;
      if (img1[i*nx + j] > 0 && img1[i*nx + j + 1] < 0
        && img2[i*nx + j] > 0 && img2[(i + 1)*nx + j] < 0
        && img0[i*nx + j] > dMean){
        npoi++;
      }
    }
    }
    dMean /= 2;
    iIterNum += 1;
  } while (npoi < iMinPOINum && iIterNum < 10);
  
  
  // one iteration more to have the same npoi as original LSReg code
  npoi = 0;
  for (i=h; i<ny-h; i++){
  for (j=h; j<nx-h; j++){
    if (FindTargetValueInWindow(image, nx, ny, j, i, h, nodata)) continue;
    if (img1[i*nx + j] > 0 && img1[i*nx + j + 1] < 0
      && img2[i*nx + j] > 0 && img2[(i + 1)*nx + j] < 0
      && img0[i*nx + j] > dMean){
      npoi++;
    }
  }
  }

  
  alloc((void**)&poi.j, npoi, sizeof(double));
  alloc((void**)&poi.i, npoi, sizeof(double));
  
  // detect POI and output
  poi.n = 0;
  for (i=h; i<ny-h; i++){
  for (j=h; j<nx-h; j++){
    if (FindTargetValueInWindow(image, nx, ny, j, i, h, nodata)) continue;
    if (img1[i*nx + j] > 0 && img1[i*nx + j + 1]   < 0
      && img2[i*nx + j] > 0 && img2[(i + 1)*nx + j] < 0
      && img0[i*nx + j] > dMean)      {
      poi.j[poi.n] = j;
      poi.i[poi.n] = i;
      poi.n++;
    }
  }
  }

  #ifdef FORCE_DEBUG
  printf(" %d POIs detected.\n", poi.n);
  #endif

  free(dFilter7);
  free(dFilter2);
  free(img0);
  free(img1);
  free(img2);
  free(img3);
  free(mask);
  
  
  #ifdef FORCE_CLOCK
  proctime_print("detecting POIs", TIME);
  #endif

  return poi;
}


/** This function performs initial tie point matching on the top layer
--- pyramids:       pyramids
--- nx_pyr:         number of columns of pyramid layers
--- ny_pyr:         number of rows of pyramid layers
--- h:              window size
--- max_h:          maximum window size
--- corr_threshold: correlation threshold
--- poi:            points of interest 
+++ Return:         tie points 
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
tie_t initial_matching(short ***pyramids_, int *nx_pyr, int *ny_pyr, int layer, int h, int max_h, double corr_threshold, poi_t *poi){
double cor, maxcor;
float corr;
float x1n, y1n, x2n, y2n;
double xdif, ydif;
int ws, w, ww;
int n;
int dx, dy;
int x, y, x2, y2, x1 = 0, y1 = 0;
int nx, ny;
short *target     = NULL;
short *base       = NULL;
short *target_sub = NULL;
short *base_sub   = NULL;
double diff_threshold;
int ntie;
tie_t init;


  #ifdef FORCE_CLOCK 
  time_t TIME; time(&TIME);
  #endif
  
  //  printf("\nImage matching\n");
  
  if (poi->n == 0){
    init.target.n = 0;
    init.base.n   = 0;
    return init;
  }

  target = pyramids_[0][layer];
  base   = pyramids_[1][layer];

  nx = nx_pyr[layer];
  ny = ny_pyr[layer];


  // set matching parameters
  diff_threshold = 1.51;  // dislocation threshold; least-squares matched position should not deviate from initial cross-correlation matched position by over this threshold

  // set maximum allowed offset, i.e. the search window size for first-step cross-correlation matching; usually = 1, resulting in 3 x 3 search window
  dx = 1;
  dy = 1;

  w = 2 * h + 1;
  ww = w*w;


  alloc((void**)&target_sub, ww, sizeof(short));
  alloc((void**)&base_sub,   ww, sizeof(short));
  
  alloc((void**)&init.target.j, poi->n, sizeof(double));
  alloc((void**)&init.target.i, poi->n, sizeof(double));
  alloc((void**)&init.base.j,   poi->n, sizeof(double));
  alloc((void**)&init.base.i,   poi->n, sizeof(double));


  ntie = 0;
  for (n=0; n<poi->n; n++){
    
    x2 = (int)poi->j[n];
    y2 = (int)poi->i[n];

    // initial cross-correlation matching
    if (!imsub(target, nx, ny, x2, y2, h, target_sub))      continue;

    // get cross-correlation matched position (x1, y1)
    maxcor = 0;
    for (y=y2-dy; y<=y2+dy; y++){
    for (x=x2-dx; x<=x2+dx; x++){
      if (imsub(base, nx, ny, x, y, h, base_sub)){
        cor = corr2(target_sub, base_sub, ww);
        if (cor > maxcor){
          x1 = x;
          y1 = y;
          maxcor = cor;
        }
      }
    }
    }

    // (x1, y1) is initially matched to (x2n, y2n), i.e. (x2, y2)
    // do least-squares matching (LSM) to get new matched position (x1n, y1n)
    // iteratively increasing matching window is used to increase matching ratio
    corr = (float)(maxcor);
    x2n = (float)(x2);
    y2n = (float)(y2);
    xdif = 0.f;
    ydif = 0.f;
    ws = w;
    do {
      // get LSM matched position (x1n, y1n)
      x1n = (float)(x1);
      y1n = (float)(y1);
      LSMatching_SAM(target, nx, ny, base, nx, ny, ws, ws, x2n, y2n, &x1n, &y1n, &corr, SHRT_MAX);

      // increment matching window size
      ws = ws + 4;

      // compare (x1n, y1n) with initial position (x1, y1)
      xdif = ABS(x1n - x1);
      ydif = ABS(y1n - y1);
    } while (xdif < 1e-10 && ydif < 1e-10 && ws / 2 < max_h && corr > 0);

    if ((xdif<1e-10 && ydif<1e-10) || xdif>diff_threshold || ydif>diff_threshold || corr<corr_threshold)
      continue; // match unsuccessful

    // output matched points
    //fprintf(fout, "%d\t%d\t%.2f\t%.2f\t%.2f\t%.2f\t%.3f\n", x2, y2, x1n, y1n, x1n - x2, y1n - y2, corr);
    init.target.j[ntie] = x2;
    init.target.i[ntie] = y2;
    init.base.j[ntie]   = x1n;
    init.base.i[ntie]   = y1n;
    ntie++;
  }

  init.target.n = ntie;
  init.base.n   = ntie;

  #ifdef FORCE_DEBUG
  printf(" %d points initially matched on top layer.\n", ntie);
  #endif


  free(base_sub);
  free(target_sub);
  
  #ifdef FORCE_CLOCK
  proctime_print("initial matching", TIME);
  #endif

  return init;
}


/** This function performs depth first matching
--- pyramids: pyramids
--- nx_pyr:   number of columns of pyramid layers
--- ny_pyr:   number of rows of pyramid layers
--- toplayer: number of layers
--- scales:   scale of the layers
--- h_:       window size
--- max_h_:   maximum window size
--- SAM:      Spectral angle mapper threshold
--- init:     tie points 
+++ Return:   matched tie points 
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
match_t depth_first_matching(short ***pyramids_, int *nx_pyr, int *ny_pyr, int toplayer, int *scales, int h_, int max_h_, double SAM, tie_t *init){
int nx, ny;
int n;
double **i_new = NULL, **j_new = NULL;    // matched coordinates on all lower layers
int layerscale;
int cumscale;
bool *invalid = NULL;          // set to 1 if not passing depth-first match
// variables for image matching
float corr;
float x1n, y1n, x2n, y2n;
float xdif, ydif;
int ws, w;
int h, max_h;
int x2, y2;
float x1, y1;
int ninit, ntie;
double diff_threshold;
int layer;
int nlayer = toplayer+1;
int value_threshold = 1000; // do not do matching if band value is smaller than this threshold; used to filter out water
short *target = NULL;
short *base   = NULL;
tie_t tie;
match_t df;


  #ifdef FORCE_CLOCK 
  time_t TIME; time(&TIME);
  #endif
  
  if (toplayer <= 0 || init->base.n == 0){
    df.ntie = 0;
    return df;
  }


  ntie = ninit = init->base.n;

  alloc_2D((void***)&i_new, nlayer, ninit, sizeof(double));
  alloc_2D((void***)&j_new, nlayer, ninit, sizeof(double));
  alloc((void**)&invalid, ninit, sizeof(bool));

  alloc((void**)&tie.target.j, ninit, sizeof(double));
  alloc((void**)&tie.target.i, ninit, sizeof(double));
  alloc((void**)&tie.base.j,   ninit, sizeof(double));
  alloc((void**)&tie.base.i,   ninit, sizeof(double));


  // save results on top layer
  for (n = 0; n < ninit; n++){
    i_new[toplayer][n] = init->base.i[n];
    j_new[toplayer][n] = init->base.j[n];
  }



  // start depth-first matching
  cumscale = 1;

  for (layer = toplayer-1; layer >= 0; layer--){

    target = pyramids_[0][layer];
    base   = pyramids_[1][layer];

    nx = nx_pyr[layer];
    ny = ny_pyr[layer];
    

    // get scale between current layer and higher (previous) layer
    layerscale = scales[layer+1];

    cumscale *= layerscale; // scale between current layer and top layer

    // set matching parameters
    h = (int)(h_*sqrt((double)(cumscale)) + 0.5);  // enlarge matching window (half window)
    max_h = (int)(max_h_*sqrt((double)(cumscale)) + 0.5);
    w = 2 * h + 1;

    diff_threshold = (layer > 0) ? 0.35 : 0.4; // empiracal dislocation thresholds

 
    // do matching on current layer
    for (n=0; n<ninit; n++){
      
      if (invalid[n]) continue;  // point n have been detected as mismatch

      x2 = (int)init->target.j[n] * cumscale;
      y2 = (int)init->target.i[n] * cumscale;

      x1 = (float)init->base.j[n] * layerscale; 
      y1 = (float)init->base.i[n] * layerscale;

      // skip possible water pixels; added 1/27/2017
      if (target[y2*nx + x2] <= value_threshold || base[(int)(y1 + 1e-10)*nx + (int)(x1 + 1e-10)] <= value_threshold){
        invalid[n] = true;
        ntie -= 1;
        continue;
      }

      // (x1, y1) is initially matched to (x2n, y2n), i.e. (x2, y2)
      // do least-squares matching (LSM) to get new matched position (x1n, y1n)
      corr = -1;
      x2n = (float)(x2);
      y2n = (float)(y2);
      xdif = 0.f;
      ydif = 0.f;
      ws = w;
      do {
        // get LSM matched position (x1n, y1n)
        x1n = x1;
        y1n = y1;
        LSMatching_SAM(target, nx, ny, base, nx, ny, ws, ws, x2n, y2n, &x1n, &y1n, &corr, SHRT_MAX);

        // increment matching window size
        ws = ws + 4 * (int)(sqrt((double)(cumscale)) + 0.5);

        // compare (x1n, y1n) with initial position (x1, y1)
        xdif = ABS(x1n - x1);
        ydif = ABS(y1n - y1);
      } while (xdif < 1e-10 && ydif < 1e-10 && ws / 2 < max_h && corr > 0);

      j_new[layer][n] = x1n;
      i_new[layer][n] = y1n;

      if (((xdif<1e-10 && ydif<1e-10) || sqrt(xdif*xdif + ydif*ydif)>diff_threshold || corr < SAM) && !invalid[n]){
        // mark point n as a mismatch
        invalid[n] = true;
        ntie -= 1;
      }

      // update pdPreCols1 and pdPreRows1, i.e. LSM matched positions
      init->base.j[n] = x1n;
      init->base.i[n] = y1n;
    }

  }
  

  cumscale = 1;
  for (layer=toplayer; layer>=0; layer--){
    cumscale *= scales[layer];
  }

  layer = 0; 
  layerscale = scales[layer+1];
  ntie = 0;
  for (n=0; n<ninit; n++){
    if (invalid[n] == true) continue;

    tie.target.j[ntie] = init->target.j[n] * cumscale;
    tie.target.i[ntie] = init->target.i[n] * cumscale;
    tie.base.j[ntie]   = j_new[layer][n];
    tie.base.i[ntie]   = i_new[layer][n];
    
    //printf("%d %.0f %.0f %.0f %.0f\n", ntie,tie.img2.j[ntie],tie.img2.i[ntie],tie.img1.j[ntie],tie.img1.i[ntie]);
    
    ntie++;

  }

  tie.target.n = ntie;
  tie.base.n   = ntie;

  
  #ifdef FORCE_DEBUG
  printf(" %d tie points pass depth-first matching.\n", ntie);
  #endif

  
  df.ntie = transform_from_df(enum_AUTO, &tie, df.coefs, &df.rmse);
 

  free((void*)tie.target.j); free((void*)tie.target.i); tie.target.n  = 0;
  free((void*)tie.base.j);   free((void*)tie.base.i);   tie.base.n = 0;
  free_2D((void**)i_new, nlayer);
  free_2D((void**)j_new, nlayer);
  free(invalid);
  
  #ifdef FORCE_CLOCK
  proctime_print("depth first matching", TIME);
  #endif

  return df;
}


/** Compute transformation from matched tie points (depth first)
--- transform_: transformation type
--- tie:        matched tie points 
--- coefs:      coefficients of the transformation
--- rmse:       RMSE of the transformation
+++ Return:     number of good tie points 
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int transform_from_df(enum_tranformation_type transform_, tie_t *tie, double *coefs, double *rmse){
int ntie, n, nout;
double *target_i  = NULL, *target_j  = NULL;
double *base_i    = NULL, *base_j    = NULL;
double *target_i_ = NULL, *target_j_ = NULL;
double *base_i_   = NULL, *base_j_   = NULL;
double *residuals = NULL;
double mean_residual, fitting_rmse;
enum_tranformation_type transform;

  
  memset(coefs, 0, 12 * sizeof(double));

  if ((ntie = tie->base.n) < 2) return 0;

  alloc((void**)&residuals, ntie, sizeof(double));

  target_j = tie->target.j;
  target_i = tie->target.i;
  base_j   = tie->base.j;
  base_i   = tie->base.i;

  target_i_ = target_i;
  target_j_ = target_j;
  base_i_   = base_i;
  base_j_   = base_j;

  if (transform_ == enum_AUTO){
    transform = choose_transform(ntie);
  } else {
    transform = transform_;
  }
  

  // fit transformation
  switch (transform)  {
  case enum_TRANSLATION:
    FitTranslationTransform(
      target_j_, target_i_, base_j_, base_i_, ntie, coefs, residuals, &mean_residual, &fitting_rmse);
    break;
  case enum_AFFINE:
    FitAffineTransform(
      target_j_, target_i_, base_j_, base_i_, ntie, coefs, residuals, &mean_residual, &fitting_rmse);
    break;
  case enum_POLYNOMIAL:
    FitPolynomialTransform(
      target_j_, target_i_, base_j_, base_i_, ntie, coefs, residuals, &mean_residual, &fitting_rmse);
    break;
  default:
    FitTranslationTransform(
      target_j_, target_i_, base_j_, base_i_, ntie, coefs, residuals, &mean_residual, &fitting_rmse);
    break;
  }


  // remove outliers
  nout = 0;
  for (n=0; n<ntie; n++){
    if (residuals[n] == -1) break;
    if (residuals[n] > 2 * fitting_rmse){
      memmove(base_j    + n, base_j    + n + 1, (size_t)(ntie - n - 1 - nout)*sizeof(double));
      memmove(base_i    + n, base_i    + n + 1, (size_t)(ntie - n - 1 - nout)*sizeof(double));
      memmove(target_j  + n, target_j  + n + 1, (size_t)(ntie - n - 1 - nout)*sizeof(double));
      memmove(target_i  + n, target_i  + n + 1, (size_t)(ntie - n - 1 - nout)*sizeof(double));
      memmove(residuals + n, residuals + n + 1, (size_t)(ntie - n - 1 - nout)*sizeof(double));
      base_j[ntie    - nout - 1] = -1;
      base_i[ntie    - nout - 1] = -1;
      target_j[ntie  - nout - 1] = -1;
      target_i[ntie  - nout - 1] = -1;
      residuals[ntie - nout - 1] = -1;
      nout++;
      n--;
    }
  }

  #ifdef FORCE_DEBUG
  printf("%d outliers\n", nout);
  #endif

  
  // refit transformation
  ntie -= nout;

  if (transform_ == enum_AUTO){
    transform = choose_transform(ntie);
  } else {
    transform = transform_;
  }

  switch (transform)  {
  case enum_TRANSLATION:
    FitTranslationTransform(
      target_j_, target_i_, base_j_, base_i_, ntie, coefs, residuals, &mean_residual, &fitting_rmse);
    break;
  case enum_AFFINE:
    FitAffineTransform(
      target_j_, target_i_, base_j_, base_i_, ntie, coefs, residuals, &mean_residual, &fitting_rmse);
    break;
  case enum_POLYNOMIAL:
    FitPolynomialTransform(
      target_j_, target_i_, base_j_, base_i_, ntie, coefs, residuals, &mean_residual, &fitting_rmse);
    break;
  default:
    FitTranslationTransform(
      target_j_, target_i_, base_j_, base_i_, ntie, coefs, residuals, &mean_residual, &fitting_rmse);
    break;
  }


  free(residuals);
  
  *rmse = fitting_rmse;
  return ntie;
}


/** Compute transformation from matched tie points (dense matching)
--- transform_:    transformation type
--- parallaxmap_x: matched tie points x
--- parallaxmap_y: matched tie points y
--- corrmap:       correlation of tie points
--- nx_new:        columns of dense matching layer
--- ny_new:        rows of dense matching layer
--- step:          step for the dense matching
--- coefs:         coefficients of the transformation
--- rmse:          RMSE of the transformation
+++ Return:        number of good tie points 
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int transform_from_dm(enum_tranformation_type transform, float *parallaxmap_x, float *parallaxmap_y, float *corrmap, int nx_new, int ny_new, int step, double *coefs, double *rmse){
int i, j;
int ntie, n, nout;
double *target_i  = NULL, *target_j  = NULL;
double *base_i = NULL, *base_j = NULL;
double *target_i_ = NULL, *target_j_ = NULL;
double *base_i_   = NULL, *base_j_   = NULL;
double *residuals = NULL;
double mean_residual, fitting_rmse;


  memset(coefs, 0, 12 * sizeof(double));


  // get number of matched points
  ntie = 0;
  for (n=0; n<ny_new*nx_new; n++){
    if (corrmap[n] > 0.1f) ntie++;
  }

  if (ntie < MIN_TIES_NUM) return ntie;

  alloc((void*)&target_j,   ntie, sizeof(double));
  alloc((void*)&target_i,   ntie, sizeof(double));
  alloc((void*)&base_j,     ntie, sizeof(double));
  alloc((void*)&base_i,     ntie, sizeof(double));
  alloc((void**)&residuals, ntie, sizeof(double));


  // get coordinates of matched points
  ntie = 0;
  for (i=0; i<ny_new; i++){
  for (j=0; j<nx_new; j++){
    n = i*nx_new + j;
    if (corrmap[n] > 0.1f){
      target_i[ntie] = i*step;
      target_j[ntie] = j*step;
      base_i[ntie]   = target_i[ntie] + parallaxmap_y[n];
      base_j[ntie]   = target_j[ntie] + parallaxmap_x[n];
      ntie++;
    }
  }
  }
  
  if (ntie < MIN_TIES_NUM) return ntie;


  base_i_   = base_i;
  base_j_   = base_j;
  target_i_ = target_i;
  target_j_ = target_j;

  // fit transformation
  switch (transform){
  case enum_TRANSLATION:
    FitTranslationTransform(
      base_j_, base_i_, target_j_, target_i_, ntie, coefs, residuals, &mean_residual, &fitting_rmse);
    break;
  case enum_AFFINE:
    FitAffineTransform(
      base_j_, base_i_, target_j_, target_i_, ntie, coefs, residuals, &mean_residual, &fitting_rmse);
    break;
  case enum_POLYNOMIAL:
    FitPolynomialTransform(
      base_j_, base_i_, target_j_, target_i_, ntie, coefs, residuals, &mean_residual, &fitting_rmse);
    break;
  default:
    FitAffineTransform(
      base_j_, base_i_, target_j_, target_i_, ntie, coefs, residuals, &mean_residual, &fitting_rmse);
    break;
  }

  // remove outliers
  nout = 0;
  for (n = 0; n < ntie; n++){
    if (residuals[n] == -1) break;
    if (residuals[n] > 2 * fitting_rmse){
      memmove(base_j    + n, base_j    + n + 1, (ntie - n - 1 - nout)*sizeof(double));
      memmove(base_i    + n, base_i    + n + 1, (ntie - n - 1 - nout)*sizeof(double));
      memmove(target_j  + n, target_j  + n + 1, (ntie - n - 1 - nout)*sizeof(double));
      memmove(target_i  + n, target_i  + n + 1, (ntie - n - 1 - nout)*sizeof(double));
      memmove(residuals + n, residuals + n + 1, (ntie - n - 1 - nout)*sizeof(double));
      base_j[ntie    - nout - 1] = -1;
      base_i[ntie    - nout - 1] = -1;
      target_j[ntie  - nout - 1] = -1;
      target_i[ntie  - nout - 1] = -1;
      residuals[ntie - nout - 1] = -1;
      nout += 1;
      n -= 1;
    }
  }
  
  #ifdef FORCE_DEBUG
  printf("%d outliers\n", nout);
  #endif

  // refit transformation
  ntie -= nout;
  if (ntie < MIN_TIES_NUM) return ntie;

  switch (transform){
  case enum_TRANSLATION:
    FitTranslationTransform(
      base_j_, base_i_, target_j_, target_i_, ntie, coefs, residuals, &mean_residual, &fitting_rmse);
    break;
  case enum_AFFINE:
    FitAffineTransform(
      base_j_, base_i_, target_j_, target_i_, ntie, coefs, residuals, &mean_residual, &fitting_rmse);
    break;
  case enum_POLYNOMIAL:
    FitPolynomialTransform(
      base_j_, base_i_, target_j_, target_i_, ntie, coefs, residuals, &mean_residual, &fitting_rmse);
    break;
  default:
    FitAffineTransform(
      base_j_, base_i_, target_j_, target_i_, ntie, coefs, residuals, &mean_residual, &fitting_rmse);
    break;
  }

  free(target_i);
  free(target_j);
  free(base_i);
  free(base_j);
  free(residuals);

  *rmse = fitting_rmse;
  return ntie;
}


/** Choose transformation type based on the number of tie points
--- ntie:   number of tie points
+++ Return: transformation type
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
enum_tranformation_type choose_transform(int ntie){

  if (ntie > 36){
    return enum_POLYNOMIAL;
  } else if (ntie > 18){
    return enum_AFFINE;
  } else {
    return enum_TRANSLATION;
  }
}


/** Re-register a band (apply transformation to image)
--- transform: transformation type
--- dm:        dense matching tie points
--- target:    image to be registered
--- nx:        columns of image
--- ny:        rows of image
--- nodata:    nodata value
+++ Return:    registered image
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
short *register_band(enum_tranformation_type transform, match_t *dm, short *target, int nx, int ny, short nodata){
int i, j, p_;
short *warped = NULL;
double di_, dj_;
int i_, j_;
float avg, weight, weightsum;
float dx, dy;


  #ifdef FORCE_CLOCK 
  time_t TIME; time(&TIME);
  #endif


  // allocate memory for new registered (warped) image
  alloc((void**)&warped, ny*nx, sizeof(short));

  #pragma omp parallel private(j,j_,i_,dj_,di_,dx,dy,p_,weightsum,avg,weight) shared(nx,ny,transform,dm,target,warped,nodata) default(none)
  {

    #pragma omp for
    for (i=0; i<ny; i++){
    for (j=0; j<nx; j++){
      
      GetTransformedCoords((double)j, (double)i, transform, dm->coefs, &dj_, &di_);

      // for bilinear resampling
      weightsum = 0;
      avg = 0;

      // top left point
      j_ = (int)dj_;
      i_ = (int)di_;

      dx = (float)(dj_ - j_);
      dy = (float)(di_ - i_);
      if (j_ >= 0 && j_ < nx && i_ >= 0 && i_ < ny){
        p_ = i_*nx + j_;
        if (target[p_] != nodata){
          weight = (1 - dx)*(1 - dy);
          avg += target[p_] * weight;
          weightsum += weight;
        }
      }

      // top right point
      j_ += 1;
      if (j_ >= 0 && j_ < nx && i_ >= 0 && i_ < ny){
        p_ = i_*nx + j_;
        if (target[p_] != nodata){
          weight = dx*(1 - dy);
          avg += target[p_] * weight;
          weightsum += weight;
        }
      }

      // bottom right point
      i_ += 1;
      if (j_ >= 0 && j_ < nx && i_ >= 0 && i_ < ny){
        p_ = i_*nx + j_;
        if (target[p_] != nodata){
          weight = dx*dy;
          avg += target[p_] * weight;
          weightsum += weight;
        }
      }

      // bottom left point
      j_ -= 1;
      if (j_ >= 0 && j_ < nx && i_ >= 0 && i_ < ny){
        p_ = i_*nx + j_;
        if (target[p_] != nodata){
          weight = (1 - dx)*dy;
          avg += target[p_] * weight;
          weightsum += weight;
        }
      }

      if (weightsum > 0){
        avg /= weightsum;
        warped[i*nx + j] = (short)(avg + 0.5f);
      } else {
        warped[i*nx + j] = nodata;
      }

    }
    }
    
  }

  free((void*)target);

  #ifdef FORCE_CLOCK
  proctime_print("registering band", TIME);
  #endif

  return warped;
}


/** Re-register the quality band (apply transformation to image, 
+++ using NN resampling)
--- transform: transformation type
--- dm:        dense matching tie points
--- image:     image to be registered
--- nx:        columns of image
--- ny:        rows of image
--- nodata:    nodata value
+++ Return:    registered image
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
short *register_quality(enum_tranformation_type transform, match_t *dm, short *image, int nx, int ny, short nodata){
int i, j, p_;
short *warped = NULL;
double di_, dj_;
int i_, j_;


  #ifdef FORCE_CLOCK 
  time_t TIME; time(&TIME);
  #endif


  // allocate memory for new registered (warped) image
  alloc((void**)&warped, ny*nx, sizeof(short));

  #pragma omp parallel private(j,j_,i_,dj_,di_,p_) shared(nx,ny,transform,dm,image,warped,nodata) default(none)
  {

    #pragma omp for
    for (i=0; i<ny; i++){
    for (j=0; j<nx; j++){
      
      GetTransformedCoords((double)j, (double)i, transform, dm->coefs, &dj_, &di_);
    
      j_ = (int)(dj_ + 0.5 + 1e-10);
      i_ = (int)(di_ + 0.5 + 1e-10);
      if (j_ >= 0 && j_ < nx && i_ >= 0 && i_ < ny){
        p_ = i_*nx + j_;
        warped[i*nx + j] = image[p_];
      } else {
        warped[i*nx + j] = nodata;
      }
    
    }
    }
  
  }

  free((void*)image);

  #ifdef FORCE_CLOCK
  proctime_print("registering quality", TIME);
  #endif

  return warped;
}


/** This function performs dense matching
--- pyramids:  pyramids
--- nx_pyr:    number of columns of pyramid layers
--- ny_pyr:    number of rows of pyramid layers
--- nodata:    nodata value
--- layer:     layer for dense matching
--- toplayer:  number of layers
--- scales:    scale of the layers
--- step:      step for the dense matching
--- h:         window size
--- max_h:     maximum window size
--- SAM:       Spectral angle mapper threshold
--- transform: transformation type
--- df:        depth first matching tie points 
+++ Return:    matched tie points 
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
match_t dense_matching(short ***pyramids_, int *nx_pyr, int *ny_pyr, short nodata, int layer, int toplayer, int *scales, int step, int h, int max_h_, double SAM, enum_tranformation_type transform, match_t *df){
int cumscale;
float corr;
float x1n, y1n, x2n, y2n, x1d, y1d;
double xdif, ydif;
int ws, w;
int  max_h;
float x1, y1;
double diff_threshold;
double corr_threshold;
double x1_predict, y1_predict;
short mean_diff_thr;   // do not do matching if the mean difference of two matching windows is larger than the threshold
short band_value_thr;  // do not do matching if band value is smaller than this threshold; used to filter out water
int i, j;
int i_new, j_new;
float *parallaxmap_x = NULL, *parallaxmap_y = NULL;
float *corrmap = NULL;
int nx_new, ny_new;
int iMatchedNum = 0, iPointNum = 0;
int autotransform;
int nx, ny;
short *target = NULL;
short *base   = NULL;
match_t dm;


  #ifdef FORCE_CLOCK 
  time_t TIME; time(&TIME);
  #endif

  if (step == 0 || df->ntie == 0){
    dm.ntie = 0;
    return dm;
  }


  // do not do matching if the mean difference of two matching windows is larger than the threshold
  mean_diff_thr = 800;
  
  // do not do matching if band value is smaller than this threshold; used to filter out water
  band_value_thr = 1000;

  //printf(" Dense matching. ");

  
  autotransform = choose_transform(df->ntie);  // added 8/31/2016

  target = pyramids_[0][layer];
  base   = pyramids_[1][layer];

  nx = nx_pyr[layer];
  ny = ny_pyr[layer];
  
  ny_new = ny / step;
  nx_new = nx / step;

  // set matching parameters
  cumscale = cumulative_scale(toplayer, scales) / cumulative_scale(layer, scales); // scale between current layer and top layer
  w = 2 * h + 1;


  max_h = (int)(max_h_*sqrt((double)(cumscale)) + 0.5); // same as the window size on this layer in depth-first matching

  corr_threshold = SAM - 0.005f;  // smaller than the default SAM threshold
  if (corr_threshold < MIN_SAM_THRESHOLD)    corr_threshold = MIN_SAM_THRESHOLD;  
  diff_threshold = MAX(0.5, df->rmse * 3);        // three times of fitting RMSE, but no smaller than 0.5 pixels; use 1.5 for pair 6
  diff_threshold = MIN(diff_threshold, 2.0);      // no larger than 2


  alloc((void**)&parallaxmap_x, ny_new*nx_new, sizeof(float));
  alloc((void**)&parallaxmap_y, ny_new*nx_new, sizeof(float));
  alloc((void**)&corrmap, ny_new*nx_new, sizeof(float));


  #pragma omp parallel private(j_new,i,j,x1_predict,y1_predict,x1,y1,corr,x2n,y2n,xdif,ydif,ws,x1n,y1n,x1d,y1d) shared(nx,ny,nx_new,ny_new,step,max_h,h,w,target,base,nodata,band_value_thr,diff_threshold,mean_diff_thr,corr_threshold,autotransform,df,parallaxmap_x,parallaxmap_y,corrmap) reduction(+: iMatchedNum,iPointNum) default(none)
  {

    #pragma omp for
    for (i_new=0; i_new<ny_new; i_new++){
    for (j_new=0; j_new<nx_new; j_new++){
      
      // get a sampled grid point (i, j) on image 2
      i = i_new*step;
      j = j_new*step;

      // skip possible water pixels
      if (target[i*nx + j] <= band_value_thr || base[i*nx + j] <= band_value_thr) continue;

      // skip if matching window contain fill values
      if (FindTargetValueInWindow(target, nx, ny, j, i, h, nodata)) continue;

      iPointNum++; // base for matching ratio calculation (water pixels and fill value pixels excluded)

      // get predicted coordinates (x1, y1) on image 1
      GetTransformedCoords((double)j, (double)i, autotransform, df->coefs, &x1_predict, &y1_predict);

      x1 = (float)x1_predict;
      y1 = (float)y1_predict;

      // least-square matching
      corr = -1;
      x2n = (float)(j);
      y2n = (float)(i);
      xdif = 0.f;
      ydif = 0.f;
      ws = w;
      do {
        x1n = x1;
        y1n = y1;
        LSMatching_SAM(target, nx, ny, base, nx, ny, ws, ws, x2n, y2n, &x1n, &y1n, &corr, mean_diff_thr);
        x1d = x1n;
        y1d = y1n;
        ws = ws + 4;
        xdif = ABS(x1d - x1);
        ydif = ABS(y1d - y1);
      } while (xdif < 1e-10 && ydif < 1e-10 && ws / 2 < max_h && corr > 0); // corr condition added 3/3/2016


      if (!((xdif < 1e-10 && ydif<1e-10) || sqrt(xdif*xdif + ydif*ydif)>diff_threshold || corr < corr_threshold)){
        parallaxmap_x[i_new*nx_new + j_new] = x1n - x2n;
        parallaxmap_y[i_new*nx_new + j_new] = y1n - y2n;
        corrmap[i_new*nx_new + j_new] = corr;

        iMatchedNum++;
      }
    }
    }

  }

  dm.ntie = transform_from_dm(transform, parallaxmap_x, parallaxmap_y, corrmap, nx_new, ny_new,  step, dm.coefs, &dm.rmse);



  // release memory
  free(parallaxmap_x);
  free(parallaxmap_y);
  free(corrmap);

  #ifdef FORCE_DEBUG
  printf(" %d (%.2f%%) grid points matched\n", dm.ntie, iMatchedNum*1.0f / iPointNum * 100);
  #endif
  
  #ifdef FORCE_CLOCK
  proctime_print("dense matching", TIME);
  #endif

  return dm;
}


/** public functions
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


/** This function is the public interface to the LSReg coregistration.
+++ Only Sentinel-2 images will be co-registered. The tie point detection
+++ is performed on the NIR band.
--- mission: mission ID
--- pl2:     L2 parameters
--- meta:    metadata
--- TOA:     Top of Atmosphere reflectance
--- QAI:     Quality Assurance Information
+++ Return:  SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int coregister(int mission, par_ll_t *pl2, brick_t *TOA, brick_t *QAI){
int p, nx, ny, nc, nb, band, err, year, month, dy;
float res;
char fname[NPOW_10], cyear[NPOW_03];
int nchar;
short nodata;
short  **target = NULL;
short   *base   = NULL;
brick_t *BASE   = NULL;
int success = FAILURE;


  #ifdef FORCE_CLOCK
  time_t TIME; time(&TIME);
  #endif


  if (strcmp(pl2->d_coreg, "NULL") == 0) return SUCCESS;

  
  #ifdef FORCE_DEBUG
  printf("doing coreg\n");
  #endif

  cite_me(_CITE_COREG_);

  nx  = get_brick_ncols(TOA);
  ny  = get_brick_nrows(TOA);
  nc  = get_brick_nrows(TOA);
  nb  = get_brick_nbands(TOA);
  res = get_brick_res(TOA);

  // import target
  if ((target = get_bands_short(TOA)) == NULL) return FAILURE;
  if (mission == SENTINEL2){
    if ((band = find_domain(TOA, "BROADNIR")) < 0) return FAILURE;
  } else if (mission == LANDSAT){
    if ((band = find_domain(TOA, "NIR")) < 0) return FAILURE;
  } else {
    printf("unknown mission in coreg.\n"); return FAILURE;
  }
  nodata = get_brick_nodata(TOA, band);
  year  =  get_brick_year(TOA, band);
  month =  get_brick_month(TOA, band);

  // get base
  dy = 0;
  while (success == FAILURE && dy < 50){
    
    nchar = snprintf(cyear, NPOW_03, "%04d-", year-dy);
    if (nchar < 0 || nchar >= NPOW_03){
      printf("Buffer Overflow in assembling pattern\n"); return FAILURE;}

    success = findfile(pl2->d_coreg, cyear, NULL, fname, NPOW_10);
    dy++;
  }
  
  //printf("%s %d %d\n", cyear, year, month-1);

  if (!fileexist(fname)){
    printf("could not retrieve base image. First 5 digits = 'YYYY-'. "); return FAILURE;}

  #ifdef FORCE_DEBUG
  printf("reference image: %s\n", fname);
  #endif


  BASE = copy_brick(TOA, 1, _DT_SHORT_);
  if ((warp_from_disc_to_known_brick(2, pl2->nthread, fname, BASE, month-1, 0, pl2->coreg_nodata)) != SUCCESS){
    printf("Warping base failed! "); return FAILURE;}
  if ((base = get_band_short(BASE, 0)) == NULL) return FAILURE;
  
  
  err = coreg(target, base, QAI, res, nx, ny, nb, band, nodata);
  free_brick(BASE);
  if (err == FAILURE){
    printf("error in coregistering image.\n"); return FAILURE;
  } else if (err == CANCEL){
    #ifdef FORCE_DEBUG
    printf("coregistering image unsuccessfull. proceed anyway.\n");
    #endif
  }


  // go through QAI, and reset boundary if neccessary
 
  #pragma omp parallel shared(nc, target, QAI, nodata, band) default(none) 
  {

    #pragma omp for schedule(static)
    for (p=0; p<nc; p++){

      if (target[band][p] == nodata && !get_off(QAI, p)) set_off(QAI, p, true);
      if (target[band][p] != nodata &&  get_off(QAI, p)) set_off(QAI, p, false);

    }
  }


  #ifdef FORCE_DEBUG
  set_brick_filename(TOA, "TOA-COREG");
  print_brick_info(TOA); set_brick_open(TOA, OPEN_CREATE); write_brick(TOA);
  #endif

  #ifdef FORCE_CLOCK
  proctime_print("coregistering image", TIME);
  #endif

  return SUCCESS;
}


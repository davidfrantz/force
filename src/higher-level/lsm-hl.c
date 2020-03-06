/**+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

This file is part of FORCE - Framework for Operational Radiometric 
Correction for Environmental monitoring.

Copyright (C) 2013-2019 David Frantz

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
This file contains functions for computing landscape metrics
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Copyright (C) 2013-2019 Franz Schug, David Frantz
Contact: franz.schug@geo.hu-berlin.de
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


#include "lsm-hl.h"


stack_t **compile_lsm(ard_t *features, lsm_t *lsm, par_hl_t *phl, cube_t *cube, int *nproduct);
stack_t *compile_lsm_stack(stack_t *ard, int nb, bool write, char *prodname, par_hl_t *phl);

int test_objects(small *cld_, int nx, int ny, int **OBJ, int **SIZE, int *nobj);


/** This function compiles the stacks, in which LSM results are stored. 
+++ It also sets metadata and sets pointers to instantly useable image 
+++ arrays.
--- features: input features
--- lsm:      pointer to instantly useable LSM image arrays
--- phl:      HL parameters
--- cube:     datacube definition
--- nproduct: number of output stacks (returned)
+++ Return:   stacks for LSM results
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
stack_t **compile_lsm(ard_t *features, lsm_t *lsm, par_hl_t *phl, cube_t *cube, int *nproduct){
stack_t **LSM = NULL;
int o, nprod = 10;
int error = 0;
enum{ _mpa_, _uci_, _fdi_, _wed_, _nbr_, _ems_, _avg_, _std_, _geo_, _max_ };
int prodlen[10] ={ phl->ftr.nfeature, phl->ftr.nfeature, phl->ftr.nfeature, phl->ftr.nfeature, phl->ftr.nfeature, phl->ftr.nfeature, phl->ftr.nfeature, phl->ftr.nfeature, phl->ftr.nfeature, phl->ftr.nfeature };
char prodname[10][NPOW_02] ={ "MPA", "UCI", "FDI", "WED", "NBR", "EMS", "AVG", "STD", "GEO", "MAX" };
                             
int prodtype[10] ={ _mpa_, _uci_, _fdi_, _wed_, _nbr_, _ems_, _avg_, _std_, _geo_, _max_ };

bool enable[10] ={ phl->lsm.ompa, phl->lsm.ouci, phl->lsm.ofdi, phl->lsm.owed, phl->lsm.onbr, 
                    phl->lsm.oems, phl->lsm.oavg, phl->lsm.ostd, phl->lsm.ogeo, phl->lsm.omax };

bool write[10]  ={ phl->lsm.ompa, phl->lsm.ouci, phl->lsm.ofdi, phl->lsm.owed, phl->lsm.onbr, 
                    phl->lsm.oems, phl->lsm.oavg, phl->lsm.ostd, phl->lsm.ogeo, phl->lsm.omax };

short ***ptr[10] ={ &lsm->mpa_, &lsm->uci_, &lsm->fdi_, &lsm->wed_, &lsm->nbr_, &lsm->ems_, &lsm->avg_, &lsm->std_, &lsm->geo_, &lsm->max_};


  alloc((void**)&LSM, nprod, sizeof(stack_t*));

  for (o=0; o<nprod; o++){
    if (enable[o]){
      if ((LSM[o] = compile_lsm_stack(features[0].DAT, prodlen[prodtype[o]], write[o], prodname[o], phl)) == NULL || (*ptr[o] = get_bands_short(LSM[o])) == NULL){
        printf("Error compiling %s product. ", prodname[o]); error++;
      }
    } else{
      LSM[o]  = NULL;
      *ptr[o] = NULL;
    }
  }

  if (error > 0){
    printf("%d compiling LSM product errors.\n", error);
    for (o=0; o<nprod; o++) free_stack(LSM[o]);
    free((void*)LSM);
    return NULL;
  }

  *nproduct = nprod;
  return LSM;
}


/** This function compiles a LSM stack
--- from:      stack from which most attributes are copied
--- nb:        number of bands in stack
--- write:     should this stack be written, or only used internally?
--- prodname:  product name
--- phl:       HL parameters
+++ Return:    stack for LSM result
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
stack_t *compile_lsm_stack(stack_t *from, int nb, bool write, char *prodname, par_hl_t *phl){
int b;
stack_t *stack = NULL;
char fname[NPOW_10];
char dname[NPOW_10];
int nchar;

  if ((stack = copy_stack(from, nb, _DT_SHORT_)) == NULL) return NULL;

  set_stack_name(stack, "FORCE Landscape Metrics");
  set_stack_product(stack, prodname);
  
  //printf("dirname should be assemlbed in write_stack, check with L2\n");
  nchar = snprintf(dname, NPOW_10, "%s/X%04d_Y%04d", phl->d_higher, 
    get_stack_tilex(stack), get_stack_tiley(stack));
  if (nchar < 0 || nchar >= NPOW_10){ 
    printf("Buffer Overflow in assembling dirname\n"); return NULL;}
  set_stack_dirname(stack, dname);

  nchar = snprintf(fname, NPOW_10, "%s_HL_LSM_%s", phl->lsm.base, prodname);
  if (nchar < 0 || nchar >= NPOW_10){ 
    printf("Buffer Overflow in assembling filename\n"); return NULL;}
  set_stack_filename(stack, fname);
  
  
  if (write){
    set_stack_open(stack, OPEN_BLOCK);
  } else{
    set_stack_open(stack, OPEN_FALSE);
  }
  set_stack_format(stack, phl->format);
  set_stack_par(stack, phl->params->log);

  for (b=0; b<nb; b++){
    set_stack_save(stack, b, true);
  }

  return stack;
}


/** public functions
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


/** This function is the entry point to the landscape metrics module
--- features:  input features
--- mask:      mask image
--- nf:        number of features
--- phl:       HL parameters
--- cube:      datacube definition
--- nproduct:  number of output stacks (returned)
+++ Return:    stacks with TXT results
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
stack_t **landscape_metrics(ard_t *features, stack_t *mask, int nf, par_hl_t *phl, cube_t *cube, int *nproduct){
lsm_t lsm;
stack_t **LSM;
small *mask_ = NULL;
int nprod = 0;
int f, i, j, p, t, nx, ny, nc, ii, jj, ni, nj, np, u;
bool exists = false;
int nobj = 0;
short nodata;
float share = 0;
int *CCL = NULL;
int *SIZE = NULL;
int numberOfUniqueClasses = 0;
int *uniqueClasses;
int *perimeters;
int *patchAreas;
int totalClassArea = 0;
int edgeLength = 0;
int sumFractalDims = 0;
int sumPatchAreas = 0;
float logSum = 0;
int maxVal = 0;
int logCounter = 0;
int validDataPixels = 0;
int res = cube->res;
int kernelSize = (phl->lsm.radius * 2 + 1) * (phl->lsm.radius * 2 + 1);
small *newFeatures = NULL;
double mx, vx;


  // import stacks
  nx = get_stack_chunkncols(features[0].DAT);
  ny = get_stack_chunknrows(features[0].DAT);
  nc = get_stack_chunkncells(features[0].DAT);
  nodata = get_stack_nodata(features[0].DAT, 0);

  // import mask (if available)
  if (mask != NULL){
    if ((mask_ = get_band_small(mask, 0)) == NULL){
      printf("Error getting processing mask."); 
      *nproduct = 0;
      return NULL;}
  }
  
  // compile products + stacks
  if ((LSM = compile_lsm(features, &lsm, phl, cube, &nprod)) == NULL || nprod == 0){
    printf("Unable to compile LSM products!\n"); 
    *nproduct = 0;
    return NULL;
  }
  

  // based on parameter input: range of numbers that are considered one class 
  // --> derive new array with binaries from input features
  // newFeatures is one-layered
  alloc((void**)&newFeatures, nc, sizeof(small));

  for (f=0; f<nf; f++){

    memset(newFeatures, 0, sizeof(small)*nc);
  
    #pragma omp parallel shared(newFeatures,phl,features,mask_,nc,f) default(none)
   {

      #pragma omp for
      for (p=0; p<nc; p++){

//FRANZ, PLEASE CHECK, ok? processing masks added 
        if (mask_ != NULL && !mask_[p]) continue;

        switch (phl->lsm.query[f]){
          case _QUERY_EQ_:
            if (features[f].dat[0][p] == phl->lsm.threshold[f]) newFeatures[p] = true;
            break;
          case _QUERY_GT_:
            if (features[f].dat[0][p] >= phl->lsm.threshold[f]) newFeatures[p] = true;
            break;
          case _QUERY_LT_:
            if (features[f].dat[0][p] <= phl->lsm.threshold[f]) newFeatures[p] = true;
            break;
          default:
            newFeatures[p] = false;
        }
      }
    
    }

    /** make cloud objects and delete small clouds **/
    binary_to_objects(newFeatures, nx, ny, 3, &CCL, &SIZE, &nobj);
    if (nobj < 1) continue;

    
    #pragma omp parallel firstprivate(kernelSize) private(p,numberOfUniqueClasses,u,uniqueClasses,perimeters,patchAreas,totalClassArea,edgeLength,sumFractalDims,sumPatchAreas,mx,vx,validDataPixels,maxVal,logSum,logCounter,ii,jj,ni,nj,np,exists,t,share) shared(ny,nx,res,f,nodata,lsm,mask_,CCL,features,phl,newFeatures) default(none)
   {

      alloc((void**)&uniqueClasses, kernelSize, sizeof(int));
      alloc((void**)&perimeters,    kernelSize, sizeof(int));
      alloc((void**)&patchAreas,    kernelSize, sizeof(int));

      #pragma omp for collapse(2) schedule(guided)
      for (i=0; i<ny; i++){
      for (j=0; j<nx; j++){

        p = i*nx+j;

//FRANZ, PLEASE CHECK, I initiated everything with NA, makes it easier to handle exceptions in the following
//FRANZ, PLEASE CHECK, this new "if" is necessary bc the metrics can now be
// chosen individually, if one is not chosen, its memory is not allocated and
// you'd get a coredump when accessing iz
        if (phl->lsm.ompa) lsm.mpa_[f][p] = nodata;
        if (phl->lsm.ouci) lsm.uci_[f][p] = nodata;
        if (phl->lsm.ofdi) lsm.fdi_[f][p] = nodata;
        if (phl->lsm.owed) lsm.wed_[f][p] = nodata;
        if (phl->lsm.onbr) lsm.nbr_[f][p] = nodata;
        if (phl->lsm.oems) lsm.ems_[f][p] = nodata;
        if (phl->lsm.oavg) lsm.avg_[f][p] = nodata;
        if (phl->lsm.ostd) lsm.std_[f][p] = nodata;
        if (phl->lsm.ogeo) lsm.geo_[f][p] = nodata;
//FRANZ, PLEASE CHECK (max_ added)
        if (phl->lsm.omax) lsm.max_[f][p] = nodata;
        
//FRANZ, PLEASE CHECK, ok? features[f].msk[p] shpuld be empty, 
// mask_ holds the processing masks
        if (mask_ != NULL && !mask_[p]) continue;
        if (!features[f].msk[p] && phl->ftr.exclude) continue;

//FRANZ, PLEASE CHECK, I added this here, makes sense to skip these pixels before going into the costly kernel loop, right?
//FRANZ, PLEASE CHECK, you set to 0, why not nodata? If nodata, a simple continue suffices
        if (!newFeatures[p] && !phl->lsm.allpx){
          if (phl->lsm.ompa) lsm.mpa_[f][p] = 0;
          if (phl->lsm.ouci) lsm.uci_[f][p] = 0;
          if (phl->lsm.ofdi) lsm.fdi_[f][p] = 0;
//FRANZ, PLEASE CHECK (wed_ added)
          if (phl->lsm.owed) lsm.wed_[f][p] = 0;
          if (phl->lsm.onbr) lsm.nbr_[f][p] = 0;
          if (phl->lsm.oems) lsm.ems_[f][p] = 0;
          if (phl->lsm.oavg) lsm.avg_[f][p] = 0;
          if (phl->lsm.ostd) lsm.std_[f][p] = 0;
          if (phl->lsm.ogeo) lsm.geo_[f][p] = 0;
          if (phl->lsm.omax) lsm.max_[f][p] = 0;
          continue;
        }

//FRANZ, PLEASE CHECK, I reduced the indendation levels by working with continues, 
// also in some cases below
// it might be a personal preference, but if there is an if without an else,
// i often prefer the continue, bc of indendation depth

        if (phl->lsm.ouci) lsm.uci_[f][p] = CCL[p];

        // loop through a kernel of given size
        numberOfUniqueClasses = 0;
        for (u=0; u<kernelSize; u++) uniqueClasses[u] = 0;
        for (u=0; u<kernelSize; u++) perimeters[u] = 0;
        for (u=0; u<kernelSize; u++) patchAreas[u] = 0;
        totalClassArea = 0;
        edgeLength = 0;
        sumFractalDims = 0;
        sumPatchAreas = 0;
        logSum = 0;
        logCounter = 0;
        validDataPixels = 0;
        maxVal = 0;
        mx = vx = 0;

        for (ii=-phl->lsm.radius; ii<=phl->lsm.radius; ii++){
        for (jj=-phl->lsm.radius; jj<=phl->lsm.radius; jj++){
            
          ni = i+ii; nj = j+jj;
// FRANZ, PLEASE CHECK, added a new variable np
          np = ni*nx+nj;
          
          // if invalid pixel 
          if (ni < 0 || nj < 0 || ni >= nx || nj >= ny) continue;

          // if not in processing mask
          if (mask_ != NULL && !mask_[np]) continue;
          
          // if inactive pixel
          if (!newFeatures[np] && !phl->lsm.allpx) continue;

          // either if allpixels must be processed or if the pixel is part of a patch
          //if (phl->lsm.allpx || newFeatures[np]){
// FRANZ, PLEASE CHECK, this continue works as you expect, right?

// FRANZ, PLEASE CHECK, better use "+=" and "++", this is easier to read
// FRANZ, PLEASE CHECK, I changed this here to a 1-pass implementation for estimating statistics,
// this way, you don't need the second costly loop over the kernel
          //do not incorporate nodata values in sum
          if (features[f].dat[0][np] != nodata){

            validDataPixels++;
            if (validDataPixels == 1){
              mx = features[f].dat[0][np];
            } else {
              var_recurrence(features[f].dat[0][np], &mx, &vx, validDataPixels);
            }
            
            if (maxVal < features[f].dat[0][np]) maxVal = features[f].dat[0][np];

          }

// FRANZ, PLEASE CHECK, this won't work for RADAR
          // natural log is undefined for 0 or negative values
          if (features[f].dat[0][np] > 0){
            logSum += (float)log(features[f].dat[0][np]);
            logCounter++;
          }
        
        
          exists = false;
// FRANZ, PLEASE CHECK, I don't see why this makes sense, numberOfUniqueClasses is 0 at this point
// will be incremented afterwards
// vector to test gets longer and longer, no linear performance..
// numberOfUniqueClasses is not the number of unique classes..
// actually I am surprised, that you don't get a segfault, as numberOfUniqueClasses might become twice as high as the number of elements allocatd in perimeters etc.

// why do you reset perimeters? It was initialized to 0 before going into the kernel loop, is this necessary here?
          //get Unique Classes
          for (t=0; t<numberOfUniqueClasses; t++){
            if (CCL[np] == uniqueClasses[t]) exists = true;
          }

          if (!exists && newFeatures[np]){
            uniqueClasses[numberOfUniqueClasses] = CCL[np];
            perimeters[numberOfUniqueClasses] = 0;
            patchAreas[numberOfUniqueClasses] = 0;
            numberOfUniqueClasses++;
          }
// FRANZ, PLEASE CHECK, no sense up until here?
          
          // add total class area
          if (newFeatures[np]) totalClassArea += 10;
          
          // add to total length of edges
// FRANZ, PLEASE CHECK, these longer ifs ok?
          if (jj != phl->lsm.radius &&
              newFeatures[np] != newFeatures[ni*nx+(nj+1)]){
              edgeLength += (res * 100);}
          
          if (ii != phl->lsm.radius && ni < nx-1 &&
              newFeatures[np] != newFeatures[(ni+1)*nx+nj]){
              edgeLength += (res * 100);}
          
          
          // add edges of current pixel to the patch perimeter of that class

          if (numberOfUniqueClasses == 0) continue;
          
          // check of which class the pixel is and at which position of 
          // uniqueClasses it is lcoated
          for (t=0; t<numberOfUniqueClasses; t++){
            
// FRANZ, PLEASE CHECK, another sort of indexing might be faster and you might get rid of this loop
            if (uniqueClasses[t] != CCL[np]) continue;

            // Increase patch area for that class
            patchAreas[t] += 10;

            // At that same position in the perimeters array, add edge length
            if (ii != phl->lsm.radius && ni < nx-1 &&
                newFeatures[np] != newFeatures[(ni+1)*nx+nj]){
                perimeters[t] += res;}

            if (ii != -phl->lsm.radius && ni > 0 &&
                newFeatures[np] != newFeatures[(ni-1)*nx+nj]){
                perimeters[t] += res;}

            if (jj != phl->lsm.radius &&
                newFeatures[np] != newFeatures[ni*nx+(nj+1)]){
                perimeters[t] += res;}

            if (jj != -phl->lsm.radius &&
                newFeatures[np] != newFeatures[ni*nx+(nj-1)]){
                perimeters[t] += res;}

          }

        }
        }



//FRANZ, PLEASE CHECK, I shifted this to down here, to set all the layers in the same section
// probably, it would even be better to use variables for the computations, and then set all layers at the end (for readability)
        if (phl->lsm.ouci) lsm.uci_[f][p] = CCL[p];

        if (phl->lsm.oavg) lsm.avg_[f][p] = mx;
        if (phl->lsm.ogeo) lsm.geo_[f][p] = exp(logSum / (float)logCounter);
        if (phl->lsm.omax) lsm.max_[f][p] = maxVal;

//FRANZ, PLEASE CHECK, better use 1-pass implementation for the standard deviation. Much faster bc kernel loop is the costly thing
        if (phl->lsm.ostd) lsm.std_[f][p] = standdev(vx, validDataPixels);


        // TODO: change unit to px or percent of area
//FRANZ, PLEASE CHECK, I swapped r with t to use the same letter for iterating the same thing throughout the code
//FRANZ, PLEASE CHECK, not sure if this makes sense, numberOfUniqueClasses points to the last class in the lower right of the kernel..
// shouldn't this be the class of the central pixel? you need to check against CCL, and better use another type of indexing
// as above, I think there is a bug in the class-table logic
        if (numberOfUniqueClasses != 0){
          for (t=0; t<numberOfUniqueClasses; t++){
            // area weight. share of patch area in total class area
            share = (float)patchAreas[t] / totalClassArea;
            if (phl->lsm.ompa) lsm.mpa_[f][p] += share * patchAreas[t] / 10;
          }
        } else{
          if (phl->lsm.ompa) lsm.mpa_[f][p] = 0;
        }

        if (phl->lsm.owed) lsm.wed_[f][p] = edgeLength / (float)kernelSize;
        if (phl->lsm.onbr) lsm.nbr_[f][p] = numberOfUniqueClasses;
        if (phl->lsm.oems) lsm.ems_[f][p] = totalClassArea * totalClassArea / kernelSize / 1000;

        // weighted sum of fractal dimensions
//FRANZ, PLEASE CHECK, I swapped r with t to use the same letter for iterating the same thing throughout the code
//FRANZ, PLEASE CHECK, not sure if this makes sense, numberOfUniqueClasses points to the last class in the lower right of the kernel..
// shouldn't this be the class of the central pixel? you need to check against CCL, and better use another type of indexing
// as above, I think there is a bug in the class-table logic
        for (t=0; t<numberOfUniqueClasses; t++){
          sumFractalDims += patchAreas[t] * (((2.0 * log(0.25 * perimeters[t]) * 1000) / (log(patchAreas[t]) * 1000)) * 10);
          sumPatchAreas += patchAreas[t];
        }

        //  weighted mean fractal index
        if (numberOfUniqueClasses != 0){
          if (sumFractalDims>0){
//FRANZ, PLEASE CHECK, (float) cast is incorrect here, as fdi_ is short 
            if (phl->lsm.ofdi) lsm.fdi_[f][p] = (short) (sumFractalDims * 10 / totalClassArea);
          }
        } else{
          if (phl->lsm.ofdi) lsm.fdi_[f][p] = 0;
        }


      }
      }
      
      free((void*)uniqueClasses);
      free((void*)perimeters);
      free((void*)patchAreas);
      
    }
      
    free((void*)CCL);
    free((void*)SIZE);
    
  }

  free((void*)newFeatures);
    
  *nproduct = nprod;
  return LSM;
}


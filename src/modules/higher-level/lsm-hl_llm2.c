#include "lsm-hl.h"

// GLCM quantisation (same idea as n_grey / min_x / max_x in azvoleff/glcm)
#define GLCM_NLEVELS 32
#define GLCM_MIN    -10000
#define GLCM_MAX     10000

static inline int glcm_quantize(short value){
  int level = (int)(((double)value - GLCM_MIN) / (GLCM_MAX - GLCM_MIN) * GLCM_NLEVELS);
  if (level < 0) level = 0;
  if (level > GLCM_NLEVELS-1) level = GLCM_NLEVELS-1;
  return level;
}

brick_t **compile_lsm(ard_t *features, lsm_t *lsm, par_hl_t *phl, cube_t *cube, int *nproduct);
brick_t *compile_lsm_brick(brick_t *ard, int nb, bool write, char *prodname, par_hl_t *phl);

int test_objects(small *cld_, int nx, int ny, int **OBJ, int **SIZE, int *nobj);


/** This function compiles the bricks, in which LSM results are stored.
+++ It also sets metadata and sets pointers to instantly useable image
+++ arrays.
--- features: input features
--- lsm:      pointer to instantly useable LSM image arrays
--- phl:      HL parameters
--- cube:     datacube definition
--- nproduct: number of output bricks (returned)
+++ Return:   bricks for LSM results
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
brick_t **compile_lsm(ard_t *features, lsm_t *lsm, par_hl_t *phl, cube_t *cube, int *nproduct){
brick_t **LSM = NULL;
int b, o, nprod = 16;
int error = 0;
int nchar;
char bname[NPOW_10];
char domain[NPOW_10];
enum{ _mpa_, _uci_, _fdi_, _edd_, _nbr_, _ems_, _avg_, _std_, _geo_, _max_, _are_, _dis_, _hom_, _con_, _ent_, _uni_ };
int prodlen[16] ={ phl->ftr.nfeature, phl->ftr.nfeature, phl->ftr.nfeature, phl->ftr.nfeature, phl->ftr.nfeature, phl->ftr.nfeature, phl->ftr.nfeature, phl->ftr.nfeature, phl->ftr.nfeature, phl->ftr.nfeature, phl->ftr.nfeature, phl->ftr.nfeature, phl->ftr.nfeature, phl->ftr.nfeature, phl->ftr.nfeature, phl->ftr.nfeature };
char prodname[16][NPOW_10] ={ "MPA", "UCI", "FDI", "EDD", "NBR", "EMS", "AVG", "STD", "GEO", "MAX", "ARE", "DIS", "HOM", "CON", "ENT", "UNI" };

int prodtype[16] ={ _mpa_, _uci_, _fdi_, _edd_, _nbr_, _ems_, _avg_, _std_, _geo_, _max_, _are_, _dis_, _hom_, _con_, _ent_ , _uni_};

bool enable[16] ={ phl->lsm.ompa, phl->lsm.ouci, phl->lsm.ofdi, phl->lsm.oedd, phl->lsm.onbr,
                    phl->lsm.oems, phl->lsm.oavg, phl->lsm.ostd, phl->lsm.ogeo, phl->lsm.omax, phl->lsm.oare,
                    phl->lsm.odis, phl->lsm.ohom, phl->lsm.ocon, phl->lsm.oent, phl->lsm.ouni };

bool write[16]  ={ phl->lsm.ompa, phl->lsm.ouci, phl->lsm.ofdi, phl->lsm.oedd, phl->lsm.onbr,
                    phl->lsm.oems, phl->lsm.oavg, phl->lsm.ostd, phl->lsm.ogeo, phl->lsm.omax, phl->lsm.oare,
                    phl->lsm.odis, phl->lsm.ohom, phl->lsm.ocon, phl->lsm.oent, phl->lsm.ouni };

short ***ptr[16] ={ &lsm->mpa_, &lsm->uci_, &lsm->fdi_, &lsm->edd_, &lsm->nbr_, &lsm->ems_, &lsm->avg_, &lsm->std_, &lsm->geo_, &lsm->max_, &lsm->are_, &lsm->dis_, &lsm->hom_, &lsm->con_, &lsm->ent_, &lsm->uni_ };


  alloc((void**)&LSM, nprod, sizeof(brick_t*));

  for (o=0; o<nprod; o++){
    if (enable[o]){
      if ((LSM[o] = compile_lsm_brick(features[0].DAT, prodlen[prodtype[o]], write[o], prodname[o], phl)) == NULL || (*ptr[o] = get_bands_short(LSM[o])) == NULL){
        printf("Error compiling %s product. ", prodname[o]); error++;
      } else {
        for (b=0; b<prodlen[o]; b++){
          basename_without_ext(phl->ftr.bname[b], bname, NPOW_10);
          if (strlen(bname) > NPOW_10-1){
            nchar = snprintf(domain, NPOW_10, "FEATURE-%04d", b+1);
            if (nchar < 0 || nchar >= NPOW_10){ 
              printf("Buffer Overflow in assembling domain\n"); error++;}
          } else {
            nchar = snprintf(domain, NPOW_10, "%s_B%04d", bname, phl->ftr.band[b]);
            if (nchar < 0 || nchar >= NPOW_10){ 
              printf("Buffer Overflow in assembling domain\n"); error++;}
          }
          set_brick_domain(LSM[o],   b, domain);
          set_brick_bandname(LSM[o], b, domain);
        }
      }
    } else{
      LSM[o]  = NULL;
      *ptr[o] = NULL;
    }
  }

  if (error > 0){
    printf("%d compiling LSM product errors.\n", error);
    for (o=0; o<nprod; o++) free_brick(LSM[o]);
    free((void*)LSM);
    return NULL;
  }

  *nproduct = nprod;
  return LSM;
}


/** This function compiles a LSM brick
--- from:     brick from which most attributes are copied
--- nb:       number of bands in brick
--- write:    should this brick be written, or only used internally?
--- prodname: product name
--- phl:      HL parameters
+++ Return:   brick for LSM result
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
brick_t *compile_lsm_brick(brick_t *from, int nb, bool write, char *prodname, par_hl_t *phl){
int b;
brick_t *brick = NULL;
char fname[NPOW_10];
char dname[NPOW_10];
char subname[NPOW_10];
int nchar;

  if ((brick = copy_brick(from, nb, _DT_SHORT_)) == NULL) return NULL;

  set_brick_name(brick, "FORCE Landscape Metrics");
  set_brick_product(brick, prodname);

  if (phl->subfolders){
    copy_string(subname, NPOW_10, prodname);
  } else {
    subname[0] = '\0';
  }

  //printf("dirname should be assemlbed in write_brick, check with L2\n");
  nchar = snprintf(dname, NPOW_10, "%s/X%04d_Y%04d/%s", phl->d_higher,
    get_brick_tilex(brick), get_brick_tiley(brick), subname);
  if (nchar < 0 || nchar >= NPOW_10){
    printf("Buffer Overflow in assembling dirname\n"); return NULL;}
  set_brick_dirname(brick, dname);
  set_brick_provdir(brick, phl->d_prov);

  nchar = snprintf(fname, NPOW_10, "%s_HL_LSM_%s", phl->lsm.base, prodname);
  if (nchar < 0 || nchar >= NPOW_10){
    printf("Buffer Overflow in assembling filename\n"); return NULL;}
  set_brick_filename(brick, fname);


  if (write){
    set_brick_open(brick, OPEN_CHUNK);
  } else{
    set_brick_open(brick, OPEN_FALSE);
  }
  set_brick_format(brick, &phl->gdalopt);
  set_brick_explode(brick, phl->explode);
  set_brick_par(brick, phl->params->log);

  for (b=0; b<nb; b++){
    set_brick_save(brick, b, true);
  }

  return brick;
}


/** public functions
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


/** This function is the entry point to the landscape metrics module
--- features: input features
--- mask:     mask image
--- nf:       number of features
--- phl:      HL parameters
--- cube:     datacube definition
--- nproduct: number of output bricks (returned)
+++ Return:   bricks with TXT results
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
brick_t **landscape_metrics(ard_t *features, brick_t *mask, int nf, par_hl_t *phl, cube_t *cube, int *nproduct){
lsm_t lsm;
brick_t **LSM;
small *mask_ = NULL;
int nprod = 0;
int f, i, j, d, p, t, nx, ny, nc, ni, nj, np, u;
int ki, kj; // iterator for KDIST
int ii, jj; // iterator for kernel relative to center
int nobj = 0;
short nodata;
float share = 0;
int ccl = 0;
int *CCL = NULL;
int *SIZE = NULL;
int *exists = NULL;
float **KDIST = NULL;
int numberOfuniquePatches = 0;
int *uniquePatches;
float *patchperimeters;
float *patchAreas;
float totalClassArea = 0;
float totaledgelength = 0;
double sumFractalDims = 0;
float logSum = 0;
float sumarea = 0;
float sumshare = 0;
int maxVal = 0;
int logCounter = 0;
int validDataPixels = 0;
int kernelSize = (phl->lsm.radius * 2 + 1) * (phl->lsm.radius * 2 + 1);
int minPatchSize = phl->lsm.minpatchsize;
int width;
small *newFeatures = NULL;
double mx, vx;
float kfraction, sqkfraction;
float unit_area = 0;
float unit_perim = 0;

// GLCM local accumulators
double glcm_dis_sum = 0.0;
double glcm_hom_sum = 0.0;
double glcm_con_sum = 0.0;
int glcm_count = 0;
int ni2, nj2, np2;
double diff = 0.0;

  cite_me(_CITE_LSM_);

  // import bricks
  nx = get_brick_chunkncols(features[0].DAT);
  ny = get_brick_chunknrows(features[0].DAT);
  nc = get_brick_chunkncells(features[0].DAT);
  nodata = get_brick_nodata(features[0].DAT, 0);

  // import mask (if available)
  if (mask != NULL){
    if ((mask_ = get_band_small(mask, 0)) == NULL){
      printf("Error getting processing mask.");
      *nproduct = 0;
      return NULL;}
  }

  // compile products + bricks
  if ((LSM = compile_lsm(features, &lsm, phl, cube, &nprod)) == NULL || nprod == 0){
    printf("Unable to compile LSM products!\n");
    *nproduct = 0;
    return NULL;
  }


  // pre-compute kernel distances
  width = phl->lsm.radius*2+1;
  distance_kernel(width, &KDIST);

  kfraction = 1.0/(float)kernelSize;
  sqkfraction = sqrt(kfraction);


  // based on parameter input: range of numbers that are considered one class
  // --> derive new array with binaries from input features
  // newFeatures is one-layered
  alloc((void**)&newFeatures, nc, sizeof(small));

  
  for (f=0; f<nf; f++){

    if (phl->lsm.odis || phl->lsm.ohom || phl->lsm.ocon ||
        phl->lsm.oent || phl->lsm.ouni) {

#pragma omp parallel \
    private(i,j,p,ii,jj,ni,nj,np,d,ni2,nj2,np2,diff, \
            glcm_dis_sum,glcm_hom_sum,glcm_con_sum,glcm_count) \
    shared(ny,nx,f,nodata,lsm,mask_,features,phl) \
    default(none)
{
    static const int dr[4] = {0, 1, 1, 1};   // 0°, 90°, 45°, 135°
    static const int dc[4] = {1, 0, 1, -1};
    int    *glcm = NULL, *touched = NULL;    // per-thread matrix + list of used cells
    int    ntouched = 0, k, cell, gi, gj;
    double glcm_ent, glcm_uni, prob;

    if (phl->lsm.oent || phl->lsm.ouni){
      alloc((void**)&glcm,    GLCM_NLEVELS*GLCM_NLEVELS, sizeof(int));
      alloc((void**)&touched, GLCM_NLEVELS*GLCM_NLEVELS, sizeof(int));
      memset(glcm, 0, GLCM_NLEVELS*GLCM_NLEVELS*sizeof(int));
    }

    #pragma omp for collapse(2) schedule(guided)
    for (i = 0; i < ny; i++) {
    for (j = 0; j < nx; j++) {

        p = i * nx + j;

        if (phl->lsm.odis) lsm.dis_[f][p] = nodata;
        if (phl->lsm.ohom) lsm.hom_[f][p] = nodata;
        if (phl->lsm.ocon) lsm.con_[f][p] = nodata;
        if (phl->lsm.oent) lsm.ent_[f][p] = nodata;
        if (phl->lsm.ouni) lsm.uni_[f][p] = nodata;

        if (mask_ != NULL && !mask_[p]) continue;
        if (phl->ftr.exclude && !features[f].msk[p]) continue;
        if (features[f].dat[0][p] == nodata) continue;

        glcm_dis_sum = glcm_hom_sum = glcm_con_sum = 0.0;
        glcm_count = 0;

        for (ii = -phl->lsm.radius; ii <= phl->lsm.radius; ii++) {
        for (jj = -phl->lsm.radius; jj <= phl->lsm.radius; jj++) {

            ni = i + ii;
            nj = j + jj;

            if (ni < 0 || nj < 0 || ni >= ny || nj >= nx) continue;

            np = ni * nx + nj;

            if (mask_ != NULL && !mask_[np]) continue;
            if (phl->ftr.exclude && !features[f].msk[np]) continue;
            if (features[f].dat[0][np] == nodata) continue;

            for (d = 0; d < 4; d++) {

                ni2 = ni + dr[d];
                nj2 = nj + dc[d];

                if (ni2 < i - phl->lsm.radius || ni2 > i + phl->lsm.radius ||
                    nj2 < j - phl->lsm.radius || nj2 > j + phl->lsm.radius)
                    continue;

                if (ni2 < 0 || nj2 < 0 || ni2 >= ny || nj2 >= nx) continue;

                np2 = ni2 * nx + nj2;

                if (mask_ != NULL && !mask_[np2]) continue;
                if (phl->ftr.exclude && !features[f].msk[np2]) continue;
                if (features[f].dat[0][np2] == nodata) continue;

                // grey levels, not raw values
                gi = glcm_quantize(features[f].dat[0][np]);
                gj = glcm_quantize(features[f].dat[0][np2]);
                diff = (double)(gi - gj);

                glcm_dis_sum += fabs(diff);
                glcm_con_sum += diff * diff;
                glcm_hom_sum += 1.0 / (1.0 + diff * diff);

                // symmetric co-occurrence matrix (only touched cells are tracked)
                if (glcm != NULL){
                    cell = gi*GLCM_NLEVELS + gj;
                    if (glcm[cell] == 0) touched[ntouched++] = cell;
                    glcm[cell]++;
                    cell = gj*GLCM_NLEVELS + gi;
                    if (glcm[cell] == 0) touched[ntouched++] = cell;
                    glcm[cell]++;
                }

                glcm_count++;
            }
        }
        }

        if (glcm_count > 0) {

            // grey-level units: DIS 0..31, CON 0..961, HOM 0..1
            if (phl->lsm.odis) lsm.dis_[f][p] = (short)(glcm_dis_sum / glcm_count * 100.0);
            if (phl->lsm.ocon) lsm.con_[f][p] = (short)(glcm_con_sum / glcm_count * 10.0);
            if (phl->lsm.ohom) lsm.hom_[f][p] = (short)(glcm_hom_sum / glcm_count * 10000.0);

            if (glcm != NULL){
                glcm_ent = glcm_uni = 0.0;

                for (k = 0; k < ntouched; k++){
                    cell = touched[k];
                    prob = (double)glcm[cell] / (2.0 * glcm_count);
                    glcm_ent -= prob * log(prob);
                    glcm_uni += prob * prob;
                    glcm[cell] = 0;            // reset for the next pixel
                }
                ntouched = 0;

                // max entropy = ln(32*32) = 6.93
                if (phl->lsm.oent) lsm.ent_[f][p] = (short)(glcm_ent * 1000.0);
                if (phl->lsm.ouni) lsm.uni_[f][p] = (short)(glcm_uni * 10000.0);
            }
        }
    }
    }

    free((void*)glcm);      // free(NULL) is fine
    free((void*)touched);
    }
  }
   
   
    memset(newFeatures, 0, sizeof(small)*nc);

    #pragma omp parallel shared(newFeatures,lsm,phl,features,mask_,nc,f,nodata) default(none)
    {

      #pragma omp for
      for (p=0; p<nc; p++){
        
        // if the metric is selected, initiate it with nodata
        if (phl->lsm.ompa) lsm.mpa_[f][p] = nodata;
        if (phl->lsm.ouci) lsm.uci_[f][p] = nodata;
        if (phl->lsm.ofdi) lsm.fdi_[f][p] = nodata;
        if (phl->lsm.oedd) lsm.edd_[f][p] = nodata;
        if (phl->lsm.onbr) lsm.nbr_[f][p] = nodata;
        if (phl->lsm.oems) lsm.ems_[f][p] = nodata;
        if (phl->lsm.oavg) lsm.avg_[f][p] = nodata;
        if (phl->lsm.ostd) lsm.std_[f][p] = nodata;
        if (phl->lsm.ogeo) lsm.geo_[f][p] = nodata;
        if (phl->lsm.omax) lsm.max_[f][p] = nodata;
        if (phl->lsm.oare) lsm.are_[f][p] = nodata;
        
        if (mask_ != NULL && !mask_[p]) continue;
        if (!features[f].msk[p] && phl->ftr.exclude) continue;

        switch (phl->lsm.query[f]){
          case _QUERY_EQ_:
            if (features[f].dat[0][p] == phl->lsm.threshold[f]) newFeatures[p] = true;
            break;
          case _QUERY_GT_:
            if (features[f].dat[0][p] > phl->lsm.threshold[f]) newFeatures[p] = true;
            break;
          case _QUERY_GE_:
            if (features[f].dat[0][p] >= phl->lsm.threshold[f]) newFeatures[p] = true;
            break;
          case _QUERY_LT_:
            if (features[f].dat[0][p] < phl->lsm.threshold[f]) newFeatures[p] = true;
            break;
          case _QUERY_LE_:
            if (features[f].dat[0][p] <= phl->lsm.threshold[f]) newFeatures[p] = true;
            break;
          default:
            newFeatures[p] = false;
        }
      }

    }

    // make objects and delete small ones
    binary_to_objects(newFeatures, nx, ny, minPatchSize, &CCL, &SIZE, &nobj);
    if (nobj < 1) continue;


    #pragma omp parallel firstprivate(kernelSize,numberOfuniquePatches) private(p,u,uniquePatches,ccl,patchperimeters,patchAreas,totalClassArea,totaledgelength,sumFractalDims,mx,vx,validDataPixels,maxVal,logSum,logCounter,ii,jj,ni,nj,np,exists,t,share,kj,ki,sumarea,sumshare,unit_area,unit_perim) shared(ny,nx,nobj,f,nodata,lsm,mask_,CCL,features,phl,newFeatures,KDIST,kfraction,sqkfraction) default(none)
    {

      alloc((void**)&uniquePatches, kernelSize, sizeof(int));
      alloc((void**)&patchperimeters, nobj+1, sizeof(float));
      alloc((void**)&patchAreas,      nobj+1, sizeof(float));
      alloc((void**)&exists,          nobj+1, sizeof(int));

      #pragma omp for collapse(2) schedule(guided)
      for (i=0; i<ny; i++){
      for (j=0; j<nx; j++){

        p = i*nx+j;

        if (mask_ != NULL && !mask_[p]) continue;
        if (!features[f].msk[p] && phl->ftr.exclude) continue;
        if (!newFeatures[p] && !phl->lsm.allpx) continue;

        // loop through a kernel of given size
        for (u=0; u<numberOfuniquePatches; u++){
          patchperimeters[uniquePatches[u]] = 0;
          patchAreas[uniquePatches[u]] = 0;
          exists[uniquePatches[u]]     = 0;
          uniquePatches[u] = 0;
        }
        numberOfuniquePatches = 0;
        totalClassArea = 0;
        totaledgelength = 0;
        sumFractalDims = 0;
        logSum = 0;
        logCounter = 0;
        validDataPixels = 0;
        maxVal = 0;
        mx = vx = 0;


        for (ii=-phl->lsm.radius, ki=0; ii<=phl->lsm.radius; ii++, ki++){
        for (jj=-phl->lsm.radius, kj=0; jj<=phl->lsm.radius; jj++, kj++){

          ni = i+ii; nj = j+jj;
          np = ni*nx+nj;
          
          // if not in circular kernel, skip
          if (phl->lsm.kernel == _KERNEL_CIRCLE_ && 
              KDIST[ki][kj] > phl->lsm.radius) continue;


          // if invalid pixel
          if (ni < 0 || nj < 0 || ni >= ny || nj >= nx) continue;

          // if not in processing mask
          if (mask_ != NULL && !mask_[np]) continue;

          // if nodata
          if (!features[f].msk[np] && phl->ftr.exclude) continue;

          // if inactive pixel
          if (!newFeatures[np]) continue;

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

          // natural log is undefined for 0 or negative values
          if (features[f].dat[0][np] > 0){
            logSum += (float)log(features[f].dat[0][np]);
            logCounter++;
          }


          ccl = CCL[np];

          if (ccl > 0 && !exists[ccl]){
            uniquePatches[numberOfuniquePatches] = ccl;
            numberOfuniquePatches++;
            exists[ccl] = true;
          }
          if (numberOfuniquePatches == 0) continue;
        
          // Increase patch area for that class
          patchAreas[ccl] += kfraction;
        
          // add total class area
          if (newFeatures[np]) totalClassArea += kfraction;


          // At that same position in the patchperimeters array, add edge length
          if (ii != phl->lsm.radius && ni+1 < ny &&
              newFeatures[np] != newFeatures[(ni+1)*nx+nj]){
              patchperimeters[ccl] += sqkfraction;
              totaledgelength += sqkfraction;
          }

          if (ii != -phl->lsm.radius && ni-1 >= 0 &&
              newFeatures[np] != newFeatures[(ni-1)*nx+nj]){
              patchperimeters[ccl] += sqkfraction;
              totaledgelength += sqkfraction;
          }

          if (jj != phl->lsm.radius && nj+1 < nx &&
              newFeatures[np] != newFeatures[ni*nx+(nj+1)]){
              patchperimeters[ccl] += sqkfraction;
              totaledgelength += sqkfraction;
        }
        
          if (jj != -phl->lsm.radius && nj-1 >= 0 &&
              newFeatures[np] != newFeatures[ni*nx+(nj-1)]){
              patchperimeters[ccl] += sqkfraction;
              totaledgelength += sqkfraction;
          }
        
        }
        }


        if (phl->lsm.ouci) lsm.uci_[f][p] = CCL[p];
        if (phl->lsm.oavg) lsm.avg_[f][p] = mx;
        if (phl->lsm.ogeo) lsm.geo_[f][p] = exp(logSum / (float)logCounter);
        if (phl->lsm.omax) lsm.max_[f][p] = maxVal;
        if (phl->lsm.oare){
            if (validDataPixels <= SHRT_MAX) {
                lsm.are_[f][p] = validDataPixels;
            } else {
                lsm.are_[f][p] = SHRT_MAX;
            }
            
        }
        if (phl->lsm.ostd) lsm.std_[f][p] = standdev(vx, validDataPixels);


        // area weight. share of patch area in total class area
        sumarea = sumshare = 0;
        
        for (t=0; t<numberOfuniquePatches; t++){
          share = patchAreas[uniquePatches[t]] / totalClassArea;
        
          sumarea  += share * patchAreas[uniquePatches[t]];
          sumshare += share;
        
        }
        
        if (phl->lsm.ompa){
          if (sumshare > 0){
                lsm.mpa_[f][p] = (short)(sumarea/sumshare*10000);
          } else lsm.mpa_[f][p] = 0;
        }

  
        if (phl->lsm.oedd) lsm.edd_[f][p] = (short)(totaledgelength / sqrt(kernelSize) * 10000);
        if (phl->lsm.onbr) lsm.nbr_[f][p] = numberOfuniquePatches;
        if (phl->lsm.oems) lsm.ems_[f][p] = (short)(totalClassArea * totalClassArea * 10000);

        sumshare = 0;
        // weighted sum of fractal dimensions
        for (t=0; t<numberOfuniquePatches; t++){

          if (patchAreas[uniquePatches[t]]*kernelSize == 1) continue;

            unit_area  = patchAreas[uniquePatches[t]]*kernelSize;
            unit_perim = patchperimeters[uniquePatches[t]]*sqrt(kernelSize);

            sumshare += patchAreas[uniquePatches[t]];
            sumFractalDims += patchAreas[uniquePatches[t]] * 
              (2.0 * log(0.25 * unit_perim)) / log(unit_area);

        }

        //  weighted mean fractal index
        if (numberOfuniquePatches != 0){
          if (phl->lsm.ofdi) lsm.fdi_[f][p] = (short) (sumFractalDims / sumshare * 10000);
        } else{
          if (phl->lsm.ofdi) lsm.fdi_[f][p] = 0;
        }


      }
      }

      free((void*)uniquePatches);
      free((void*)patchperimeters);
      free((void*)patchAreas);
      free((void*)exists);

    }

    free((void*)CCL);
    free((void*)SIZE);

  }

  free((void*)newFeatures);
  free_2D((void**)KDIST, width);

  *nproduct = nprod;
  return LSM;
}
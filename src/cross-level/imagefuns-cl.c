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
This file contains functions for image methods
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


#include "imagefuns-cl.h"


/** This function estimates the width of a Gaussian used to perform Gaus-
+++ sian lowpass filtering.
--- r:      ratio of target and actual resolution
+++ Return: width of Gaussian (sigma)
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
float find_sigma(float r){
float r2, amp, sig, sigma, k, diff;

  r2 = (r/2.0)*(r/2.0);

  sigma = 0.0; diff = INT_MAX;

  while (diff > 0){
    
    sigma += 0.01;

    sig = sigma*sigma*2;
    amp = 1.0/(M_PI*sig);
    k = amp*exp(-r2/sig);

    diff = amp/2.0 - k;

  }

  sigma -= 0.01;

  return sigma;
}


/** This function generates a Gaussian kernel to perform Gaussian lowpass
+++ filtering
--- nk:     kernel width
--- sigma:  width of Gaussian
--- kernel: kernel with weighting coefficients (returned)
+++ Return: SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int gauss_kernel(int nk, float sigma, float ***kernel){
int i, j;
float hx, hy, sig;
float **k = NULL;

  if ((nk % 2) != 1){
    printf("gaussian kernel width must be odd. "); return FAILURE;}

  #ifdef FORCE_DEBUG
  printf("gaussian kernel width %d, sigma: %.2f\n", nk, sigma);
  #endif

  sig = sigma*sigma*2;
  alloc_2D((void***)&k, nk, nk, sizeof(float));

  for (i=0; i<nk; i++){
  for (j=0; j<nk; j++){

    hx = -(nk-1)/2 + j;
    hy = -(nk-1)/2 + i;
    k[i][j] = 1/(M_PI*sig) * exp(-(hx*hx+hy*hy)/(sig));

  }
  }

  *kernel = k;
  return SUCCESS;
}


/** This function generates a distance kernel for kernel-based methods
--- nk:     kernel width
--- kernel: kernel with distance to central pixel (returned)
+++ Return: SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int distance_kernel(int nk, float ***kernel){
int ki, kj, ii, jj, h;
float **k = NULL;

  if ((nk % 2) != 1){
    printf("distance kernel width must be odd. "); return FAILURE;}

  alloc_2D((void***)&k, nk, nk, sizeof(float));

  h = (nk-1)/2;

  // pre-compute kernel distance
  for (ii=-h, ki=0; ii<=h; ii++, ki++){
  for (jj=-h, kj=0; jj<=h; jj++, kj++){
    k[ki][kj] = sqrt(ii*ii+jj*jj);
  }
  }

  *kernel = k;
  return SUCCESS;
}


/** This function buffers all TRUE pixels by r pixels. Brick entry point
--- brick:  brick with binary image (only use with 0/1)
--- b:      band
--- r:      radius
+++ Return: SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int buffer(brick_t *brick, int b, int r){
small *image = NULL;
int nx, ny;

  if ((image = get_band_small(brick, b)) == NULL) return FAILURE;

  nx = get_brick_ncols(brick);
  ny = get_brick_nrows(brick);

  return buffer_(image, nx, ny, r);
}


/** This function buffers all TRUE pixels by r pixels using the midpoint 
+++ circle algorithm.
--- image:  Binary image (only use with 0/1)
--- nx:     number of columns
--- ny:     number of rows
--- r:      radius
+++ Return: SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int buffer_(small *image, int nx, int ny, int r){
int i, j, k, p, d, x, y;
int nc = nx*ny;
small *tmp = NULL;


  alloc((void**)&tmp, nc, sizeof(small));

  // do for every pixel, except boundary
  // boundary width is determined by buffer radius

  #pragma omp parallel private(j, p, k, x, y, d) shared(nx, ny, r, image, tmp) default(none) 
  {

    #pragma omp for schedule(guided)
    for (i=r; i<(ny-r); i++){
    for (j=r; j<(nx-r); j++){

      p = nx*i+j;

      if (!image[p]) continue;

      // midpoint circle algorithm
      d = 3 - r;
      x = 0;
      y = r;

      while (x <= y){

        // draw buffer with radius around point
        for (k=i-y; k<=i+y; k++){ tmp[nx*k+j+x] = true; tmp[nx*k+j-x] = true; }
        for (k=i-x; k<=i+x; k++){ tmp[nx*k+j+y] = true; tmp[nx*k+j-y] = true; }

        if (d < 0){
          d = d + (4 * x) + 6;
        } else {
          d = d + 4 * (x - y) + 10;
          y -= 1;
        }
        x += 1;

      }
    }
    }
    
  }


  // copy buffer to input
  memmove(image, tmp, nc*sizeof(small));

  free((void*)tmp);

  return SUCCESS;
}


/** This function performs a majority filling, i.e. FALSE cells that are 
+++ surrounded by 5 or more TRUE cells will be set to TRUE. This fills
+++ small holes in a binary image. Brick entry point
--- brick:  brick with binary image (only use with 0/1)
--- b:      band
+++ Return: SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int majorfill(brick_t *brick, int b){
small *image = NULL;
int nx, ny;

  if ((image = get_band_small(brick, b)) == NULL) return FAILURE;

  nx = get_brick_ncols(brick);
  ny = get_brick_nrows(brick);

  return majorfill_(image, nx, ny);
}


/** This function performs a majority filling, i.e. FALSE cells that are 
+++ surrounded by 5 or more TRUE cells will be set to TRUE. This fills
+++ small holes in a binary image.
--- image:  Binary image (only use with 0/1)
--- nx:     number of columns
--- ny:     number of rows
+++ Return: SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int majorfill_(small *image, int nx, int ny){
int i, j, ii, jj, ni, nj, p, k;


  #pragma omp parallel private(j, p, ii, jj, ni, nj, k) shared(nx, ny, image) default(none) 
  {

    #pragma omp for schedule(guided)
    for (i=0; i<ny; i++){
    for (j=0; j<nx; j++){
      
      p = i*nx+j;

      if (image[p]) continue;

      k = 0;

      for (ii=-1; ii<=1; ii++){
      for (jj=-1; jj<=1; jj++){

        if (ii == 0 && jj == 0) continue;
        ni = i+ii; nj = j+jj;
        if (ni < 0 || ni >= ny || nj < 0 || nj >= nx) continue;

        k += image[ni*nx+nj];

      }
      }

      if (k >= 5) image[p] = true;

    }
    }
    
  }
  

  return SUCCESS;
}


/** This function computes the distance transformation, i.e. the pixel 
+++ distance of any FALSE cell to its next TRUE cell. Brick entry point
--- brick:  brick with binary image (only use with 0/1)
--- b:      band
+++ Return: distance transformed image
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
ushort *dist_transform(brick_t *brick, int b){
small *image = NULL;
int nx, ny;

  if ((image = get_band_small(brick, b)) == NULL) return NULL;

  nx = get_brick_ncols(brick);
  ny = get_brick_nrows(brick);

  return dist_transform_(image, nx, ny);
}


/** This function computes the distance transformation, i.e. the pixel 
+++ distance of any FALSE cell to its next TRUE cell.
+++-----------------------------------------------------------------------
+++ Meijster, A., Roerdink, J.B.T.M., Hesselink, W.H. (2006). A general 
+++ algorithm for computing distance transforms in linear time. Computa-
+++ tional Imaging and Vision, 18, 331-340.
+++-----------------------------------------------------------------------
--- image:  Binary image (only use with 0/1)
--- nx:     number of columns
--- ny:     number of rows
+++ Return: distance transformed image
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
ushort *dist_transform_(small *image, int nx, int ny){
int x, y, q, w, u;
int *s = NULL;
int *t = NULL;
ushort *G = NULL;
ushort *distance = NULL;


  alloc((void**)&distance, nx*ny, sizeof(ushort));
  alloc((void**)&G, nx*ny, sizeof(ushort));

  /** first phase **/
  
  #pragma omp parallel private(y) shared(nx, ny, image, G) default(none) 
  {

    #pragma omp for schedule(static)
    for (x=0; x<nx; x++){

      // scan 1
      if (image[x]) G[x] = 0; else G[x] = nx+ny;
      
      for (y=1; y<ny; y++){
        if (image[y*nx+x]){
          G[y*nx+x] = 0;
        } else {
          G[y*nx+x] = 1 + G[(y-1)*nx+x];
        }
      }

      // scan 2    
      for (y=ny-2; y>=0; y--){
        if (G[(y+1)*nx+x] < G[y*nx+x]){
          G[y*nx+x] = 1 + G[(y+1)*nx+x];
        }
      }

    }

  }
  
  
  /** second phase **/
  
  #pragma omp parallel private(u, s, t, q, w) shared(nx, ny, image, G, distance) default(none) 
  {
    
    // allocate s and t
    alloc((void**)&s, nx, sizeof(int));
    alloc((void**)&t, nx, sizeof(int));

    #pragma omp for schedule(static)
    for (y=0; y<ny; y++){
    
      q = 0; s[0] = 0; t[0] = 0;
    
      // scan 3
      for (u=1; u<nx; u++){
      
        while (q >= 0 && dt_dfun(nx, t[q], s[q], y, G) > dt_dfun(nx, t[q], u, y, G)) q--;

        if (q < 0){
          q = 0;
          s[0] = u;
        } else {
          w = 1 + dt_Sep(nx, s[q], u, y, G);
          if (w < nx){
            q++;
            s[q] = u;
            t[q] = w;
          }
        }
      
      }
    
      // scan 4
      for (u=nx-1; u>=0; u--){
        distance[y*nx+u] = (ushort) sqrt(dt_dfun(nx, u, s[q], y, G));
        if (u == t[q]) q--;
      }
    
    }
    
    free((void*)s);
    free((void*)t);
    
  }

  // free memory
  free((void*)G);


  return distance;
}


/** Function for euclidian distance
+++ This function is used in the distance transformation (dist_transform)
+++--------------------------------------------------------------------**/
int dt_dfun(int nx, int x, int i, int y, ushort *G){
int d;

  d = (x-i)*(x-i) + G[nx*y+i]*G[nx*y+i];

  return d;
}


/** Sep function for euclidian distance
+++ This function is used in the distance transformation (dist_transform)
+++--------------------------------------------------------------------**/
int dt_Sep(int nx, int i, int u, int y, ushort *G){
int sep;

  // note that integer division is used on purpose
  sep = (u*u - i*i + G[nx*y+u]*G[nx*y+u] - 
                     G[nx*y+i]*G[nx*y+i]) / (2*(u-i));

  return sep;
}


/** This function performs connected components labeling, i.e. it segments
+++ all 8-connected TRUE patches. A unique ID is given to each segment, 
+++ starting at top-left. Brick entry point
--- brick:  brick with binary image (only use with 0/1)
--- b:              band of binary image
--- b_brick:        brick that CCL should go to
--- b_segmentation: band of brick that holds CCL
+++ Return:         number of segments
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int connectedcomponents(brick_t *brick, int b_brick, brick_t *segmentation, int b_segmentation){
small *brick_        = NULL;
int   *segmentation_ = NULL;
int nx, ny;

  if ((brick_        = get_band_small(brick,      b_brick))        == NULL) return -1;
  if ((segmentation_ = get_band_int(segmentation, b_segmentation)) == NULL) return -1;

  nx = get_brick_ncols(brick);
  ny = get_brick_nrows(brick);

  if (nx != get_brick_ncols(segmentation)) return -1;
  if (ny != get_brick_nrows(segmentation)) return -1;

  return connectedcomponents_(brick_, segmentation_, nx, ny);
}


/** This function performs connected components labeling, i.e. it segments
+++ all 8-connected TRUE patches. A unique ID is given to each segment, 
+++ starting at top-left.
+++-----------------------------------------------------------------------
+++ Chang, F., Chen, C.-J., Lu, C.-J. (2004). A linear-time component-la-
+++ beling algorithm using contour tracing technique. Computer Vision and
+++ Image Understanding, 93 (2), 206-220.
+++-----------------------------------------------------------------------
--- image:  Binary image (only use with 0/1)
--- CCL:    Connected components
--- nx:     number of columns
--- ny:     number of rows
+++ Return: number of segments
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int connectedcomponents_(small *image, int *CCL, int nx, int ny){
int i, j, p;
int dir, k = 0, label = 0;

  for (i=0, p=0; i<ny; i++){
  for (j=0, label=0; j<nx; j++, p++){


    if (image[p]){

      if (label != 0){ // use last label

        CCL[p] = label;

      } else {

        label = CCL[p];

        if (label == 0){

          label = ++k;
          dir = 0;
          ccl_contourtrace(i, j, label, dir, image, CCL, nx, ny); // external contour
          CCL[p] = label;

        }

      }

    } else if (label != 0){ // if last pixel was labeled

      if (CCL[p] == 0){
        dir = 1;
        ccl_contourtrace(i, j-1, label, dir, image, CCL, nx, ny); // internal contour
      }

      label = 0;

    }

  }
  }


  // replace -1 with 0
  for (p=0; p<nx*ny; p++){
    if (CCL[p] < 0) CCL[p] = 0;
  }

  return(k);
}


/** Tracer Function for connected components labeling
+++ This function is used in connectedcomponents
+++--------------------------------------------------------------------**/
void ccl_tracer(int *cy, int *cx, int *dir, small *image, int *CCL, int nx, int ny){
int i, y, x, tval;
static int neighbor[8][2] = {{0,1},{1,1},{1,0},{1,-1},{0,-1},{-1,-1},{-1,0},{-1,1}};


  for (i=0; i<7; i++){

    y = *cy + neighbor[*dir][0];
    x = *cx + neighbor[*dir][1];

    if (y>=0 && y<ny && x>=0 && x<nx){

      tval = image[y*nx+x];
      
    } else {

      tval = 0;

    }

    if (tval == 0){

      if (y>=0 && y<ny && x>=0 && x<nx) CCL[y*nx+x] = -1;
      *dir = (*dir + 1) % 8;

    } else {

      *cy = y;
      *cx = x;
      break;

    }

  }

  return;
}


/** Contour tracing Function for connected components labeling
+++ This function is used in connectedcomponents
+++--------------------------------------------------------------------**/
void ccl_contourtrace(int cy, int cx, int label, int dir, 
                      small *image, int *CCL, int nx, int ny){
bool stop = false, search = true;
int fx, fy, sx = cx, sy = cy;

  ccl_tracer(&cy, &cx, &dir, image, CCL, nx, ny);

  if (cx != sx || cy != sy){

    fx = cx;
    fy = cy;

    while (search){

      dir = (dir + 6) % 8;
      CCL[cy*nx+cx] = label;
      ccl_tracer(&cy, &cx, &dir, image, CCL, nx, ny);

      if (cx == sx && cy == sy){

        stop = true;

      } else if (stop){

        if (cx == fx && cy == fy){
          search = false;
        } else {
          stop = false;
        }

      }

    }

  }

  return;
}


/** This function segments a binary layer. Small objects (< nmin) are eli-
+++ minated. The segmentation result (unique ID per object), and the number
+++ of object and object pixels is returned.
--- image:  binary mask
--- nx:     number of columns
--- ny:     number of rows
--- OBJ:    segmentation (returned)
--- SIZE:   object sizes (returned)
--- nobj:   Number of objects (returned)
+++ Return: SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int binary_to_objects(small *image, int nx, int ny, int nmin, int **OBJ, int **SIZE, int *nobj){
int *CCL = NULL; // connected component labelling
int *CCS = NULL; // connected component object size
int id, nc, p, no;


  #ifdef FORCE_CLOCK
  time_t TIME; time(&TIME);
  #endif
  
  
  nc = nx*ny;

  alloc((void**)&CCL, nc, sizeof(int));

  // to ensure that 1st object gets ID=1
  image[0] = false;

  if ((no = connectedcomponents_(image, CCL, nx, ny)) > 0){

    alloc((void**)&CCS, no, sizeof(int));

    for (p=0; p<nc; p++){
      if (!image[p]) continue;
      id = CCL[p]; 
      CCS[id-1]++;
    }

    for (p=0; p<nc; p++){
      if (!image[p]) continue;
      id = CCL[p];
      if (CCS[id-1] < nmin) image[p] = 0;
    }

    for (p=0; p<nc; p++) CCL[p] = 0;
    free((void*)CCS);

    if ((no = connectedcomponents_(image, CCL, nx, ny)) > 0){

      alloc((void**)&CCS, no, sizeof(int));

      for (p=0; p<nc; p++){
        if (!image[p]){ CCL[p] = 0; continue;}
        id = CCL[p]; 
        CCS[id-1]++;
      }

    } else free((void*)CCL);

  } else free((void*)CCL);

  
  #ifdef FORCE_CLOCK
  proctime_print("binary to objects", TIME);
  #endif
  
  *nobj = no;
  *OBJ  = CCL;
  *SIZE = CCS;
  return SUCCESS;
}


/** Fast hybrid greyscale reconstruction algorithm
+++ This function is an efficient implementation of a greyscale flood-fill
+++ procesure. Brick entry point
--- mask:     brick with mask image
--- b_mask:   band of mask image
--- marker:   brick with marker image
--- b_marker: band of marker image
+++ Return:   SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int greyscale_reconstruction(brick_t *mask, int b_mask, brick_t *marker, int b_marker){
short *mask_   = NULL;
short *marker_ = NULL;
int nx, ny;

  if ((mask_   = get_band_short(mask,   b_mask))   == NULL) return FAILURE;
  if ((marker_ = get_band_short(marker, b_marker)) == NULL) return FAILURE;

  nx = get_brick_ncols(mask);
  ny = get_brick_nrows(mask);

  if (nx != get_brick_ncols(marker)) return FAILURE;
  if (ny != get_brick_nrows(marker)) return FAILURE;

  return greyscale_reconstruction_(mask_, marker_, nx, ny);
}


/** Fast hybrid greyscale reconstruction algorithm
+++ This function is an efficient implementation of a greyscale flood-fill
+++ procesure.
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
+++ Vincent, L. (1993). Morphological Grayscale Reconstruction in Image An
+++ alysis: Applications and Efficient Algorithms. IEEE Transactions on Im
+++ age Processing, 2 (2), 176-201.
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
--- MASK:   mask image
--- MARKER: marker image
--- nx:     number of columns
--- ny:     number of rows
+++ Return: SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int greyscale_reconstruction_(short *MASK, short *MARKER, int nx, int ny){
int i, j, p, ii, jj, np, np1[4], np2[4], k;
short min, max;
queue_t fifo;

  np1[0] = -nx-1; np1[1] = -nx; np1[2] = -nx+1; np1[3] = -1;
  np2[0] = nx+1; np2[1] = nx; np2[2] = nx-1; np2[3] = 1;


  // initialize queue
  if ((create_queue(&fifo, (nx*ny)/2)) == FAILURE){
    printf("failed to create new queue!\n"); return FAILURE;}


  /* set border of marker image to mask value
     set rest of image to maximum of mask */
  max = SHRT_MIN;  
  for (p=0; p<nx*ny; p++){
    if (MASK[p] > max) max = MASK[p];
  }
  
  
  for (i=0, p=0; i<ny; i++){
  for (j=0; j<nx; j++, p++){
    
    if (i == 0 || j == 0 || i == ny-1 || j == nx-1){
      MARKER[p] = MASK[p];
    } else {
      MARKER[p] = max;
    }

  }
  }


  // sequential reconstruction in raster order
  for (i=1; i<ny-1; i++){
  for (j=1; j<nx-1; j++){

    p = i*nx+j; 
    min = MARKER[p];
    
    for (k=0; k<4; k++){
      np = p+np1[k];
      if (MARKER[np] < min) min = MARKER[np];
    }

    if (min > MASK[p]) MARKER[p] = min; else MARKER[p] = MASK[p];

  }
  }


  // sequential reconstruction in anti-raster order
  for (i=ny-2; i>=1; i--){
  for (j=nx-2; j>=1; j--){

    p = i*nx+j; 
    min = MARKER[p];

    for (k=0; k<4; k++){
      np = p+np2[k];
      if (MARKER[np] < min) min = MARKER[np];
    }

    if (min > MASK[p]) MARKER[p] = min; else MARKER[p] = MASK[p];

    for (k=0; k<4; k++){
      np = p+np2[k];
      if (MARKER[np] > MARKER[p] && MARKER[np] > MASK[np]){

        if (enqueue(&fifo, j, i) == FAILURE){
          printf("Failed to enqueue another coord. pair!\n"); 
          return FAILURE;
        }
        continue;

      }
    }

  }
  }

  //fprintf("step 2: done. queue length: %d\n", fifo->length);

  // queue propagation
  while (dequeue(&fifo, &j, &i) == SUCCESS){

    p = nx*i+j;

    for (ii=-1; ii<=1; ii++){
    for (jj=-1; jj<=1; jj++){

      if (ii == 0 && jj == 0) continue;

      np = (i+ii)*nx+j+jj;

      if (MARKER[np] > MARKER[p] && MASK[np] != MARKER[np]){

        if (MARKER[p] > MASK[np]) MARKER[np] = MARKER[p]; else MARKER[np] = MASK[np];
        
        if (enqueue(&fifo, j+jj, i+ii) == FAILURE){
          printf("Failed to enqueue another coord. pair!\n");
          return FAILURE;
        }

      }

    }
    }

  }  

  //fprintf("step 3: done. queue length: %d\n", fifo->length);

  // free queue's memory
  destroy_queue(&fifo);
  //fifo = NULL;

  return SUCCESS;
}


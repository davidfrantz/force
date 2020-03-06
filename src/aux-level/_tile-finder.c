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
This program returns the tile ID and pixel that contains the requested 
input coordinate.
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


#include <stdio.h>   // core input and output functions
#include <stdlib.h>  // standard general utilities library

#include "../cross-level/const-cl.h"
#include "../cross-level/konami-cl.h"
#include "../cross-level/cube-cl.h"
#include "../cross-level/warp-cl.h"


int main( int argc, char *argv[] ){
char *dir = NULL;
coord_t geo, map, tile;
int t_ulx, t_uly, ti, tj, ci, cj, chunk;
double res;
cube_t *cube = NULL;


  if (argc >= 2) check_arg(argv[1]);
  if (argc != 5){ printf("usage: %s datacube lon lat res\n\n", argv[0]); return FAILURE;}


  // get command line parameters
  dir    = argv[1];
  geo.x  = atof(argv[2]);
  geo.y  = atof(argv[3]);
  res    = atof(argv[4]);


  // read datacube definition
  if ((cube = read_datacube_def(dir)) == NULL){
    printf("Reading datacube definition failed.\n"); return FAILURE;}
  update_datacube_res(cube, res);


  // get target coordinates in target css coordinates
  if ((warp_geo_to_any(geo.x,  geo.y, &map.x, &map.y, cube->proj)) == FAILURE){
    printf("Computing target coordinates in dst_srs failed!\n"); return FAILURE;}


  // find the tile the target coordinates fall into
  tile_find(map.x, map.y, &tile.x, &tile.y, &t_ulx, &t_uly, cube);

  // find pixel in tile
  tj = (int)((map.x-tile.x)/cube->res);
  ti = (int)((tile.y-map.y)/cube->res);

  // find chunk in tile
  chunk = (int)(ti/cube->cy);

  // find pixel in chunk
  cj = tj;
  ci = ti - chunk*cube->cy;

  // Print to stdout
  printf("Point { LON/LAT (%.2f,%.2f) | X/Y (%.2f,%.2f) }\n"
          "       is in tile X%04d_Y%04d at pixel J/I %d/%d\n"
          "       is in chunk %d at position J/I %d/%d\n",
    geo.x, geo.y, map.x, map.y, 
    t_ulx, t_uly, tj, ti,
    chunk, cj, ci);

          
  free_datacube(cube);

  return SUCCESS;
}


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
This file contains functions for Level 3 processing
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


#include "tsa-hl.h"


stack_t *compile_tsa_stack(stack_t *ard, int nb, int idx, int write, char *prodname, par_hl_t *phl);
stack_t **compile_tsa(ard_t *ard, tsa_t *tsa, par_hl_t *phl, cube_t *cube, int nt, int ni, int idx, int *nproduct);


/** This function compiles the stacks, in which TSA results are stored. 
+++ It also sets metadata and sets pointers to instantly useable image 
+++ arrays.
--- ard:      ARD
--- ts:       pointer to instantly useable TSA image arrays
--- phl:      HL parameters
--- cube:     datacube definition
--- nt:       number of ARD products over time
--- ni:       number of interpolated products over time
--- idx:      spectral index
--- nproduct: number of output stacks (returned)
+++ Return:   stacks for TSA results
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
stack_t **compile_tsa(ard_t *ard, tsa_t *ts, par_hl_t *phl, cube_t *cube, int nt, int ni, int idx, int *nproduct){
stack_t **TSA = NULL;
int t, k;
date_t date;
char fdate[NPOW_10];
char sensor[NPOW_04];
char domain[NPOW_10];
int nchar;
int o, nprod = 98;
int error = 0;
enum { _full_, _stats_, _inter_, _year_, _quarter_, _month_, _week_, _day_, _lsp_, _trd_, _cat_ };
int prodlen[11] = { nt, phl->tsa.stm.sta.nmetrics, ni, 
                    phl->ny, phl->nq, phl->nm, phl->nw, phl->nd,
                    phl->tsa.lsp.ny, _TRD_LENGTH_, _CAT_LENGTH_ };
char prodname[98][NPOW_03] = { 
  "TSS", "RMS", "STM", "TSI", "SPL",
  "FBY", "FBQ", "FBM", "FBW", "FBD",
  "DEM-LSP", "DSS-LSP", "DRI-LSP", "DPS-LSP", "DFI-LSP", "DES-LSP", 
  "DLM-LSP", "LTS-LSP", "LGS-LSP", "VEM-LSP", "VSS-LSP", "VRI-LSP", 
  "VPS-LSP", "VFI-LSP", "VES-LSP", "VLM-LSP", "VBL-LSP", "VSA-LSP", 
  "IST-LSP", "IBL-LSP", "IBT-LSP", "IGS-LSP", "RAR-LSP", "RAF-LSP", 
  "RMR-LSP", "RMF-LSP",
  "DEM-TRP", "DSS-TRP", "DRI-TRP", "DPS-TRP", "DFI-TRP", "DES-TRP", 
  "DLM-TRP", "LTS-TRP", "LGS-TRP", "VEM-TRP", "VSS-TRP", "VRI-TRP", 
  "VPS-TRP", "VFI-TRP", "VES-TRP", "VLM-TRP", "VBL-TRP", "VSA-TRP", 
  "IST-TRP", "IBL-TRP", "IBT-TRP", "IGS-TRP", "RAR-TRP", "RAF-TRP", 
  "RMR-TRP", "RMF-TRP", 
  "TRY", "TRQ", "TRM", "TRW", "TRD",
  "DEM-CAP", "DSS-CAP", "DRI-CAP", "DPS-CAP", "DFI-CAP", "DES-CAP", 
  "DLM-CAP", "LTS-CAP", "LGS-CAP", "VEM-CAP", "VSS-CAP", "VRI-CAP", 
  "VPS-CAP", "VFI-CAP", "VES-CAP", "VLM-CAP", "VBL-CAP", "VSA-CAP", 
  "IST-CAP", "IBL-CAP", "IBT-CAP", "IGS-CAP", "RAR-CAP", "RAF-CAP", 
  "RMR-CAP", "RMF-CAP", 
  "CAY", "CAQ", "CAM", "CAW", "CAD" };

int prodtype[98] = { 
  _full_, _full_, _stats_, _inter_, _inter_,
  _year_, _quarter_, _month_, _week_, _day_, 
  _lsp_, _lsp_, _lsp_, _lsp_, _lsp_, _lsp_, 
  _lsp_, _lsp_, _lsp_, _lsp_, _lsp_, _lsp_, 
  _lsp_, _lsp_, _lsp_, _lsp_, _lsp_, _lsp_, 
  _lsp_, _lsp_, _lsp_, _lsp_, _lsp_, _lsp_, 
  _lsp_, _lsp_, 
  _trd_, _trd_, _trd_, _trd_, _trd_, _trd_, 
  _trd_, _trd_, _trd_, _trd_, _trd_, _trd_, 
  _trd_, _trd_, _trd_, _trd_, _trd_, _trd_, 
  _trd_, _trd_, _trd_, _trd_, _trd_, _trd_, 
  _trd_, _trd_, 
  _trd_, _trd_, _trd_, _trd_, _trd_,
  _cat_, _cat_, _cat_, _cat_, _cat_, _cat_, 
  _cat_, _cat_, _cat_, _cat_, _cat_, _cat_, 
  _cat_, _cat_, _cat_, _cat_, _cat_, _cat_, 
  _cat_, _cat_, _cat_, _cat_, _cat_, _cat_, 
  _cat_, _cat_, 
  _cat_, _cat_, _cat_, _cat_, _cat_ };

int enable[98] = { 
  true, phl->tsa.sma.orms, phl->tsa.stm.ostm, true, phl->tsa.lsp.ospl,
  phl->tsa.fld.ofby+phl->tsa.fld.otry+phl->tsa.fld.ocay, phl->tsa.fld.ofbq+phl->tsa.fld.otrq+phl->tsa.fld.ocaq,
  phl->tsa.fld.ofbm+phl->tsa.fld.otrm+phl->tsa.fld.ocam, phl->tsa.fld.ofbw+phl->tsa.fld.otrw+phl->tsa.fld.ocaw,
  phl->tsa.fld.ofbd+phl->tsa.fld.otrd+phl->tsa.fld.ocad, 
  phl->tsa.lsp.odem*(phl->tsa.lsp.olsp+phl->tsa.lsp.otrd+phl->tsa.lsp.ocat), phl->tsa.lsp.odss*(phl->tsa.lsp.olsp+phl->tsa.lsp.otrd+phl->tsa.lsp.ocat),
  phl->tsa.lsp.odri*(phl->tsa.lsp.olsp+phl->tsa.lsp.otrd+phl->tsa.lsp.ocat), phl->tsa.lsp.odps*(phl->tsa.lsp.olsp+phl->tsa.lsp.otrd+phl->tsa.lsp.ocat),
  phl->tsa.lsp.odfi*(phl->tsa.lsp.olsp+phl->tsa.lsp.otrd+phl->tsa.lsp.ocat), phl->tsa.lsp.odes*(phl->tsa.lsp.olsp+phl->tsa.lsp.otrd+phl->tsa.lsp.ocat),
  phl->tsa.lsp.odlm*(phl->tsa.lsp.olsp+phl->tsa.lsp.otrd+phl->tsa.lsp.ocat), phl->tsa.lsp.olts*(phl->tsa.lsp.olsp+phl->tsa.lsp.otrd+phl->tsa.lsp.ocat),
  phl->tsa.lsp.olgs*(phl->tsa.lsp.olsp+phl->tsa.lsp.otrd+phl->tsa.lsp.ocat), phl->tsa.lsp.ovem*(phl->tsa.lsp.olsp+phl->tsa.lsp.otrd+phl->tsa.lsp.ocat),
  phl->tsa.lsp.ovss*(phl->tsa.lsp.olsp+phl->tsa.lsp.otrd+phl->tsa.lsp.ocat), phl->tsa.lsp.ovri*(phl->tsa.lsp.olsp+phl->tsa.lsp.otrd+phl->tsa.lsp.ocat),
  phl->tsa.lsp.ovps*(phl->tsa.lsp.olsp+phl->tsa.lsp.otrd+phl->tsa.lsp.ocat), phl->tsa.lsp.ovfi*(phl->tsa.lsp.olsp+phl->tsa.lsp.otrd+phl->tsa.lsp.ocat),
  phl->tsa.lsp.oves*(phl->tsa.lsp.olsp+phl->tsa.lsp.otrd+phl->tsa.lsp.ocat), phl->tsa.lsp.ovlm*(phl->tsa.lsp.olsp+phl->tsa.lsp.otrd+phl->tsa.lsp.ocat),
  phl->tsa.lsp.ovbl*(phl->tsa.lsp.olsp+phl->tsa.lsp.otrd+phl->tsa.lsp.ocat), phl->tsa.lsp.ovsa*(phl->tsa.lsp.olsp+phl->tsa.lsp.otrd+phl->tsa.lsp.ocat),
  phl->tsa.lsp.oist*(phl->tsa.lsp.olsp+phl->tsa.lsp.otrd+phl->tsa.lsp.ocat), phl->tsa.lsp.oibl*(phl->tsa.lsp.olsp+phl->tsa.lsp.otrd+phl->tsa.lsp.ocat),
  phl->tsa.lsp.oibt*(phl->tsa.lsp.olsp+phl->tsa.lsp.otrd+phl->tsa.lsp.ocat), phl->tsa.lsp.oigs*(phl->tsa.lsp.olsp+phl->tsa.lsp.otrd+phl->tsa.lsp.ocat),
  phl->tsa.lsp.orar*(phl->tsa.lsp.olsp+phl->tsa.lsp.otrd+phl->tsa.lsp.ocat), phl->tsa.lsp.oraf*(phl->tsa.lsp.olsp+phl->tsa.lsp.otrd+phl->tsa.lsp.ocat),
  phl->tsa.lsp.ormr*(phl->tsa.lsp.olsp+phl->tsa.lsp.otrd+phl->tsa.lsp.ocat), phl->tsa.lsp.ormf*(phl->tsa.lsp.olsp+phl->tsa.lsp.otrd+phl->tsa.lsp.ocat),
  phl->tsa.lsp.otrd*phl->tsa.lsp.odem, phl->tsa.lsp.otrd*phl->tsa.lsp.odss, phl->tsa.lsp.otrd*phl->tsa.lsp.odri, phl->tsa.lsp.otrd*phl->tsa.lsp.odps, phl->tsa.lsp.otrd*phl->tsa.lsp.odfi, phl->tsa.lsp.otrd*phl->tsa.lsp.odes, 
  phl->tsa.lsp.otrd*phl->tsa.lsp.odlm, phl->tsa.lsp.otrd*phl->tsa.lsp.olts, phl->tsa.lsp.otrd*phl->tsa.lsp.olgs, phl->tsa.lsp.otrd*phl->tsa.lsp.ovem, phl->tsa.lsp.otrd*phl->tsa.lsp.ovss, phl->tsa.lsp.otrd*phl->tsa.lsp.ovri, 
  phl->tsa.lsp.otrd*phl->tsa.lsp.ovps, phl->tsa.lsp.otrd*phl->tsa.lsp.ovfi, phl->tsa.lsp.otrd*phl->tsa.lsp.oves, phl->tsa.lsp.otrd*phl->tsa.lsp.ovlm, phl->tsa.lsp.otrd*phl->tsa.lsp.ovbl, phl->tsa.lsp.otrd*phl->tsa.lsp.ovsa, 
  phl->tsa.lsp.otrd*phl->tsa.lsp.oist, phl->tsa.lsp.otrd*phl->tsa.lsp.oibl, phl->tsa.lsp.otrd*phl->tsa.lsp.oibt, phl->tsa.lsp.otrd*phl->tsa.lsp.oigs, phl->tsa.lsp.otrd*phl->tsa.lsp.orar, phl->tsa.lsp.otrd*phl->tsa.lsp.oraf, 
  phl->tsa.lsp.otrd*phl->tsa.lsp.ormr, phl->tsa.lsp.otrd*phl->tsa.lsp.ormf,
  phl->tsa.fld.otry, phl->tsa.fld.otrq, phl->tsa.fld.otrm, phl->tsa.fld.otrw, phl->tsa.fld.otrd, 
  phl->tsa.lsp.ocat*phl->tsa.lsp.odem, phl->tsa.lsp.ocat*phl->tsa.lsp.odss, phl->tsa.lsp.ocat*phl->tsa.lsp.odri, phl->tsa.lsp.ocat*phl->tsa.lsp.odps, phl->tsa.lsp.ocat*phl->tsa.lsp.odfi, phl->tsa.lsp.ocat*phl->tsa.lsp.odes, 
  phl->tsa.lsp.ocat*phl->tsa.lsp.odlm, phl->tsa.lsp.ocat*phl->tsa.lsp.olts, phl->tsa.lsp.ocat*phl->tsa.lsp.olgs, phl->tsa.lsp.ocat*phl->tsa.lsp.ovem, phl->tsa.lsp.ocat*phl->tsa.lsp.ovss, phl->tsa.lsp.ocat*phl->tsa.lsp.ovri, 
  phl->tsa.lsp.ocat*phl->tsa.lsp.ovps, phl->tsa.lsp.ocat*phl->tsa.lsp.ovfi, phl->tsa.lsp.ocat*phl->tsa.lsp.oves, phl->tsa.lsp.ocat*phl->tsa.lsp.ovlm, phl->tsa.lsp.ocat*phl->tsa.lsp.ovbl, phl->tsa.lsp.ocat*phl->tsa.lsp.ovsa, 
  phl->tsa.lsp.ocat*phl->tsa.lsp.oist, phl->tsa.lsp.ocat*phl->tsa.lsp.oibl, phl->tsa.lsp.ocat*phl->tsa.lsp.oibt, phl->tsa.lsp.ocat*phl->tsa.lsp.oigs, phl->tsa.lsp.ocat*phl->tsa.lsp.orar, phl->tsa.lsp.ocat*phl->tsa.lsp.oraf, 
  phl->tsa.lsp.ocat*phl->tsa.lsp.ormr, phl->tsa.lsp.ocat*phl->tsa.lsp.ormf,
  phl->tsa.fld.ocay, phl->tsa.fld.ocaq, phl->tsa.fld.ocam, phl->tsa.fld.ocaw, phl->tsa.fld.ocad };

int write[98]  = { 
  phl->tsa.otss, phl->tsa.sma.orms, phl->tsa.stm.ostm, phl->tsa.tsi.otsi, phl->tsa.lsp.ospl,
  phl->tsa.fld.ofby, phl->tsa.fld.ofbq, phl->tsa.fld.ofbm, phl->tsa.fld.ofbw, phl->tsa.fld.ofbd, 
  phl->tsa.lsp.olsp*phl->tsa.lsp.odem, phl->tsa.lsp.olsp*phl->tsa.lsp.odss, phl->tsa.lsp.olsp*phl->tsa.lsp.odri, phl->tsa.lsp.olsp*phl->tsa.lsp.odps, phl->tsa.lsp.olsp*phl->tsa.lsp.odfi, phl->tsa.lsp.olsp*phl->tsa.lsp.odes, 
  phl->tsa.lsp.olsp*phl->tsa.lsp.odlm, phl->tsa.lsp.olsp*phl->tsa.lsp.olts, phl->tsa.lsp.olsp*phl->tsa.lsp.olgs, phl->tsa.lsp.olsp*phl->tsa.lsp.ovem, phl->tsa.lsp.olsp*phl->tsa.lsp.ovss, phl->tsa.lsp.olsp*phl->tsa.lsp.ovri, 
  phl->tsa.lsp.olsp*phl->tsa.lsp.ovps, phl->tsa.lsp.olsp*phl->tsa.lsp.ovfi, phl->tsa.lsp.olsp*phl->tsa.lsp.oves, phl->tsa.lsp.olsp*phl->tsa.lsp.ovlm, phl->tsa.lsp.olsp*phl->tsa.lsp.ovbl, phl->tsa.lsp.olsp*phl->tsa.lsp.ovsa, 
  phl->tsa.lsp.olsp*phl->tsa.lsp.oist, phl->tsa.lsp.olsp*phl->tsa.lsp.oibl, phl->tsa.lsp.olsp*phl->tsa.lsp.oibt, phl->tsa.lsp.olsp*phl->tsa.lsp.oigs, phl->tsa.lsp.olsp*phl->tsa.lsp.orar, phl->tsa.lsp.olsp*phl->tsa.lsp.oraf, 
  phl->tsa.lsp.olsp*phl->tsa.lsp.ormr, phl->tsa.lsp.olsp*phl->tsa.lsp.ormf,
  phl->tsa.lsp.otrd*phl->tsa.lsp.odem, phl->tsa.lsp.otrd*phl->tsa.lsp.odss, phl->tsa.lsp.otrd*phl->tsa.lsp.odri, phl->tsa.lsp.otrd*phl->tsa.lsp.odps, phl->tsa.lsp.otrd*phl->tsa.lsp.odfi, phl->tsa.lsp.otrd*phl->tsa.lsp.odes, 
  phl->tsa.lsp.otrd*phl->tsa.lsp.odlm, phl->tsa.lsp.otrd*phl->tsa.lsp.olts, phl->tsa.lsp.otrd*phl->tsa.lsp.olgs, phl->tsa.lsp.otrd*phl->tsa.lsp.ovem, phl->tsa.lsp.otrd*phl->tsa.lsp.ovss, phl->tsa.lsp.otrd*phl->tsa.lsp.ovri, 
  phl->tsa.lsp.otrd*phl->tsa.lsp.ovps, phl->tsa.lsp.otrd*phl->tsa.lsp.ovfi, phl->tsa.lsp.otrd*phl->tsa.lsp.oves, phl->tsa.lsp.otrd*phl->tsa.lsp.ovlm, phl->tsa.lsp.otrd*phl->tsa.lsp.ovbl, phl->tsa.lsp.otrd*phl->tsa.lsp.ovsa, 
  phl->tsa.lsp.otrd*phl->tsa.lsp.oist, phl->tsa.lsp.otrd*phl->tsa.lsp.oibl, phl->tsa.lsp.otrd*phl->tsa.lsp.oibt, phl->tsa.lsp.otrd*phl->tsa.lsp.oigs, phl->tsa.lsp.otrd*phl->tsa.lsp.orar, phl->tsa.lsp.otrd*phl->tsa.lsp.oraf, 
  phl->tsa.lsp.otrd*phl->tsa.lsp.ormr, phl->tsa.lsp.otrd*phl->tsa.lsp.ormf,
  phl->tsa.fld.otry, phl->tsa.fld.otrq, phl->tsa.fld.otrm, phl->tsa.fld.otrw, phl->tsa.fld.otrd, 
  phl->tsa.lsp.ocat*phl->tsa.lsp.odem, phl->tsa.lsp.ocat*phl->tsa.lsp.odss, phl->tsa.lsp.ocat*phl->tsa.lsp.odri, phl->tsa.lsp.ocat*phl->tsa.lsp.odps, phl->tsa.lsp.ocat*phl->tsa.lsp.odfi, phl->tsa.lsp.ocat*phl->tsa.lsp.odes, 
  phl->tsa.lsp.ocat*phl->tsa.lsp.odlm, phl->tsa.lsp.ocat*phl->tsa.lsp.olts, phl->tsa.lsp.ocat*phl->tsa.lsp.olgs, phl->tsa.lsp.ocat*phl->tsa.lsp.ovem, phl->tsa.lsp.ocat*phl->tsa.lsp.ovss, phl->tsa.lsp.ocat*phl->tsa.lsp.ovri, 
  phl->tsa.lsp.ocat*phl->tsa.lsp.ovps, phl->tsa.lsp.ocat*phl->tsa.lsp.ovfi, phl->tsa.lsp.ocat*phl->tsa.lsp.oves, phl->tsa.lsp.ocat*phl->tsa.lsp.ovlm, phl->tsa.lsp.ocat*phl->tsa.lsp.ovbl, phl->tsa.lsp.ocat*phl->tsa.lsp.ovsa, 
  phl->tsa.lsp.ocat*phl->tsa.lsp.oist, phl->tsa.lsp.ocat*phl->tsa.lsp.oibl, phl->tsa.lsp.ocat*phl->tsa.lsp.oibt, phl->tsa.lsp.ocat*phl->tsa.lsp.oigs, phl->tsa.lsp.ocat*phl->tsa.lsp.orar, phl->tsa.lsp.ocat*phl->tsa.lsp.oraf, 
  phl->tsa.lsp.ocat*phl->tsa.lsp.ormr, phl->tsa.lsp.ocat*phl->tsa.lsp.ormf,
  phl->tsa.fld.ocay, phl->tsa.fld.ocaq, phl->tsa.fld.ocam, phl->tsa.fld.ocaw, phl->tsa.fld.ocad };

short ***ptr[98] = { 
  &ts->tss_, &ts->rms_, &ts->stm_, &ts->tsi_, &ts->spl_,
  &ts->fby_, &ts->fbq_, &ts->fbm_, &ts->fbw_, &ts->fbd_, 
  &ts->lsp_[0],  &ts->lsp_[1],  &ts->lsp_[2],  &ts->lsp_[3],  &ts->lsp_[4],  &ts->lsp_[5], 
  &ts->lsp_[6],  &ts->lsp_[7],  &ts->lsp_[8],  &ts->lsp_[9],  &ts->lsp_[10], &ts->lsp_[11], 
  &ts->lsp_[12], &ts->lsp_[13], &ts->lsp_[14], &ts->lsp_[15], &ts->lsp_[16], &ts->lsp_[17], 
  &ts->lsp_[18], &ts->lsp_[19], &ts->lsp_[20], &ts->lsp_[21], &ts->lsp_[22], &ts->lsp_[23], 
  &ts->lsp_[24], &ts->lsp_[25], 
  &ts->trp_[0],  &ts->trp_[1],  &ts->trp_[2],  &ts->trp_[3],  &ts->trp_[4],  &ts->trp_[5], 
  &ts->trp_[6],  &ts->trp_[7],  &ts->trp_[8],  &ts->trp_[9],  &ts->trp_[10], &ts->trp_[11], 
  &ts->trp_[12], &ts->trp_[13], &ts->trp_[14], &ts->trp_[15], &ts->trp_[16], &ts->trp_[17], 
  &ts->trp_[18], &ts->trp_[19], &ts->trp_[20], &ts->trp_[21], &ts->trp_[22], &ts->trp_[23], 
  &ts->trp_[24], &ts->trp_[25],
  &ts->try_, &ts->trq_, &ts->trm_, &ts->trw_, &ts->trd_, 
  &ts->cap_[0],  &ts->cap_[1],  &ts->cap_[2],  &ts->cap_[3],  &ts->cap_[4],  &ts->cap_[5], 
  &ts->cap_[6],  &ts->cap_[7],  &ts->cap_[8],  &ts->cap_[9],  &ts->cap_[10], &ts->cap_[11], 
  &ts->cap_[12], &ts->cap_[13], &ts->cap_[14], &ts->cap_[15], &ts->cap_[16], &ts->cap_[17], 
  &ts->cap_[18], &ts->cap_[19], &ts->cap_[20], &ts->cap_[21], &ts->cap_[22], &ts->cap_[23], 
  &ts->cap_[24], &ts->cap_[25], 
  &ts->cay_, &ts->caq_, &ts->cam_, &ts->caw_, &ts->cad_ };



  alloc((void**)&TSA, nprod, sizeof(stack_t*));


//printf("about to allocate %d %d %d %d %d %d %d %d dates\n", 
//nt,ni,phl->ny,phl->nq,phl->nm,phl->nw,phl->nd,phl->tsa.lsp.ny);

  if (nt              > 0) alloc((void**)&ts->d_tss, nt, sizeof(date_t));              else ts->d_tss = NULL;
  if (ni              > 0) alloc((void**)&ts->d_tsi, ni, sizeof(date_t));              else ts->d_tsi = NULL;
  if (phl->ny         > 0) alloc((void**)&ts->d_fby, phl->ny, sizeof(date_t));         else ts->d_fby = NULL;
  if (phl->nq         > 0) alloc((void**)&ts->d_fbq, phl->nq, sizeof(date_t));         else ts->d_fbq = NULL;
  if (phl->nm         > 0) alloc((void**)&ts->d_fbm, phl->nm, sizeof(date_t));         else ts->d_fbm = NULL;
  if (phl->nw         > 0) alloc((void**)&ts->d_fbw, phl->nw, sizeof(date_t));         else ts->d_fbw = NULL;
  if (phl->nd         > 0) alloc((void**)&ts->d_fbd, phl->nd, sizeof(date_t));         else ts->d_fbd = NULL;
  if (phl->tsa.lsp.ny > 0) alloc((void**)&ts->d_lsp, phl->tsa.lsp.ny, sizeof(date_t)); else ts->d_lsp = NULL;

  //printf("scale, date, ts, bandnames, and sensor ID must be set in compile_tsa!!!\n");

  
  for (o=0; o<nprod; o++){
    
    if (enable[o]){
      
      if ((TSA[o] = compile_tsa_stack(ard[0].DAT, prodlen[prodtype[o]], idx, write[o], prodname[o], phl)) == NULL || (  *ptr[o] = get_bands_short(TSA[o])) == NULL){
        printf("Error compiling %s product. ", prodname[o]); error++;
      } else {

        init_date(&date);
        set_date(&date, 2000, 1, 1);

        for (t=0, k=1; t<prodlen[prodtype[o]]; t++){

          switch (prodtype[o]){
            case _full_:
              date = get_stack_date(ard[t].DAT, 0);
              get_stack_sensor(ard[t].DAT, 0, sensor, NPOW_04);
              set_stack_sensor(TSA[o], t, sensor);
              copy_date(&date, &ts->d_tss[t]);
              compact_date(date.year, date.month, date.day, fdate, NPOW_10);
              set_stack_wavelength(TSA[o], t, date.year + (date.doy-1)/365.0);
              set_stack_unit(TSA[o], t, "decimal year");
              nchar = snprintf(domain, NPOW_10, "%s_%s", fdate, sensor);
              if (nchar < 0 || nchar >= NPOW_10){ 
                printf("Buffer Overflow in assembling domain\n"); error++;}
              //set_stack_domain(TSA[o], t, domain);
              set_stack_bandname(TSA[o], t, domain);
              break;
            case _stats_:
              set_stack_sensor(TSA[o], t, "BLEND");
              //set_stack_domain(TSA[o],   t, _TAGGED_ENUM_STA_[phl->tsa.stm.sta.metrics[t]].tag);
              set_stack_bandname(TSA[o], t, _TAGGED_ENUM_STA_[phl->tsa.stm.sta.metrics[t]].tag);
              break;
            case _inter_:
              if (phl->tsa.tsi.method == _INT_NONE_){
                date = get_stack_date(ard[t].DAT, 0);
              } else {
                set_date_ce(&date, phl->date_range[_MIN_].ce + t*phl->tsa.tsi.step);
              }
              set_stack_sensor(TSA[o], t, "BLEND");
              copy_date(&date, &ts->d_tsi[t]);
              compact_date(date.year, date.month, date.day, fdate, NPOW_10);
              set_stack_wavelength(TSA[o], t, date.year + (date.doy-1)/365.0);
              set_stack_unit(TSA[o], t, "decimal year");
              //set_stack_domain(TSA[o], t, fdate);
              set_stack_bandname(TSA[o], t, fdate);
              break;
            case _year_:
              set_date_year(&date, phl->date_range[_MIN_].year+t);
              set_stack_sensor(TSA[o], t, "BLEND");
              copy_date(&date, &ts->d_fby[t]);
              nchar = snprintf(fdate, NPOW_10, "YEAR-%04d", date.year);
              if (nchar < 0 || nchar >= NPOW_10){ 
                printf("Buffer Overflow in assembling domain\n"); error++;}
              set_stack_wavelength(TSA[o], t, date.year);
              set_stack_unit(TSA[o], t, "year");
              //set_stack_domain(TSA[o], t, fdate);
              set_stack_bandname(TSA[o], t, fdate);
              break;
            case  _quarter_:
              while (k < 5 && !phl->date_quarters[k]) k++;
              set_date_quarter(&date, k);
              set_stack_sensor(TSA[o], t, "BLEND");
              copy_date(&date, &ts->d_fbq[t]);
              nchar = snprintf(fdate, NPOW_10, "QUARTER-%01d", date.quarter);
              if (nchar < 0 || nchar >= NPOW_10){ 
                printf("Buffer Overflow in assembling domain\n"); error++;}
              set_stack_wavelength(TSA[o], t, k);
              set_stack_unit(TSA[o], t, "quarter");
              //set_stack_domain(TSA[o], t, fdate);
              set_stack_bandname(TSA[o], t, fdate);
              k++;
              break;
            case _month_: 
              while (k < 13 && !phl->date_months[k]) k++;
              set_date_month(&date, k);
              set_stack_sensor(TSA[o], t, "BLEND");
              copy_date(&date, &ts->d_fbm[t]);
              nchar = snprintf(fdate, NPOW_10, "MONTH-%02d", date.month);
              if (nchar < 0 || nchar >= NPOW_10){ 
                printf("Buffer Overflow in assembling domain\n"); error++;}
              set_stack_wavelength(TSA[o], t, k);
              set_stack_unit(TSA[o], t, "month");
              //set_stack_domain(TSA[o], t, fdate);
              set_stack_bandname(TSA[o], t, fdate);
              k++;
              break;
            case _week_: 
              while (k < 53 && !phl->date_weeks[k]) k++;
              set_date_week(&date, k);
              set_stack_sensor(TSA[o], t, "BLEND");
              copy_date(&date, &ts->d_fbw[t]);
              nchar = snprintf(fdate, NPOW_10, "WEEK-%02d", date.week);
              if (nchar < 0 || nchar >= NPOW_10){ 
                printf("Buffer Overflow in assembling domain\n"); error++;}
              set_stack_wavelength(TSA[o], t, k);
              set_stack_unit(TSA[o], t, "week");
              //set_stack_domain(TSA[o], t, fdate);
              set_stack_bandname(TSA[o], t, fdate);
              k++;
              break;
            case _day_: 
              while (k < 366 && !phl->date_doys[k]) k++;
              set_date_doy(&date, k);
              set_stack_sensor(TSA[o], t, "BLEND");
              copy_date(&date, &ts->d_fbd[t]);
              nchar = snprintf(fdate, NPOW_10, "DOY-%03d", date.doy);
              if (nchar < 0 || nchar >= NPOW_10){ 
                printf("Buffer Overflow in assembling domain\n"); error++;}
              set_stack_wavelength(TSA[o], t, k);
              set_stack_unit(TSA[o], t, "day of year");
              //set_stack_domain(TSA[o], t, fdate);
              set_stack_bandname(TSA[o], t, fdate);
              k++;
              break;
            case _lsp_: 
              set_date_year(&date, phl->date_range[_MIN_].year+t+1);
              set_stack_sensor(TSA[o], t, "BLEND");
              copy_date(&date, &ts->d_lsp[t]);
              nchar = snprintf(fdate, NPOW_10, "YEAR-%04d", date.year);
              if (nchar < 0 || nchar >= NPOW_10){ 
                printf("Buffer Overflow in assembling domain\n"); error++;}
              set_stack_wavelength(TSA[o], t, date.year);
              set_stack_unit(TSA[o], t, "year");
              //set_stack_domain(TSA[o], t, fdate);
              set_stack_bandname(TSA[o], t, fdate);
              break;
            case _trd_:
              set_stack_sensor(TSA[o], t, "BLEND");
              //set_stack_domain(TSA[o], t, _TAGGED_ENUM_TRD_[t].tag);
              set_stack_bandname(TSA[o], t, _TAGGED_ENUM_TRD_[t].tag);
              break;
            case _cat_:
              set_stack_sensor(TSA[o], t, "BLEND");
              //set_stack_domain(TSA[o], t, _TAGGED_ENUM_CAT_[t].tag);
              set_stack_bandname(TSA[o], t, _TAGGED_ENUM_CAT_[t].tag);
              break;
            default:
              printf("unknown tsa type.\n"); error++;
              break;
          }
          
          set_stack_date(TSA[o], t, date);

        }

      }

    } else {
      TSA[o]  = NULL;
      *ptr[o] = NULL;
    }
  }


  if (error > 0){
    printf("%d compiling TSA product errors.\n", error);
    for (o=0; o<nprod; o++) free_stack(TSA[o]);
    free((void*)TSA);
    if (ts->d_tss != NULL){ free((void*)ts->d_tss); ts->d_tss = NULL;}
    if (ts->d_tsi != NULL){ free((void*)ts->d_tsi); ts->d_tsi = NULL;}
    if (ts->d_fby != NULL){ free((void*)ts->d_fby); ts->d_fby = NULL;}
    if (ts->d_fbq != NULL){ free((void*)ts->d_fbq); ts->d_fbq = NULL;}
    if (ts->d_fbm != NULL){ free((void*)ts->d_fbm); ts->d_fbm = NULL;}
    if (ts->d_fbw != NULL){ free((void*)ts->d_fbw); ts->d_fbw = NULL;}
    if (ts->d_fbd != NULL){ free((void*)ts->d_fbd); ts->d_fbd = NULL;}
    if (ts->d_lsp != NULL){ free((void*)ts->d_lsp); ts->d_lsp = NULL;}
    return NULL;
  }

  *nproduct = nprod;
  return TSA;
}


/** This function compiles a TSA stack
--- from:      stack from which most attributes are copied
--- nb:        number of bands in stack
--- idx:       spectral index
--- write:     should this stack be written, or only used internally?
--- prodname:  product name
--- phl:       HL parameters
+++ Return:    stack for TSA result
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
stack_t *compile_tsa_stack(stack_t *from, int nb, int idx, int write, char *prodname, par_hl_t *phl){
int b;
stack_t *stack = NULL;
date_t date;
char fname[NPOW_10];
char dname[NPOW_10];
char domain[NPOW_10];
int nchar;


  if ((stack = copy_stack(from, nb, _DT_SHORT_)) == NULL) return NULL;

  set_stack_name(stack, "FORCE Time Series Analysis");
  set_stack_product(stack, prodname);

  //printf("dirname should be assemlbed in write_stack, check with L2\n");
  nchar = snprintf(dname, NPOW_10, "%s/X%04d_Y%04d", phl->d_higher, 
    get_stack_tilex(stack), get_stack_tiley(stack));
  if (nchar < 0 || nchar >= NPOW_10){ 
    printf("Buffer Overflow in assembling dirname\n"); return NULL;}
  set_stack_dirname(stack, dname);

  nchar = snprintf(fname, NPOW_10, "%04d-%04d_%03d-%03d_HL_TSA_%s_%s_%s", 
    phl->date_range[_MIN_].year, phl->date_range[_MAX_].year, 
    phl->doy_range[_MIN_], phl->doy_range[_MAX_], 
    phl->sen.target, phl->tsa.index_name[idx], prodname);
  if (nchar < 0 || nchar >= NPOW_10){ 
    printf("Buffer Overflow in assembling filename\n"); return NULL;}
  set_stack_filename(stack, fname);
  
  if (write){
    set_stack_open(stack, OPEN_BLOCK);
  } else {
    set_stack_open(stack, OPEN_FALSE);
  }
  set_stack_format(stack, phl->format);
  set_stack_explode(stack, phl->explode);
  set_stack_par(stack, phl->params->log);

  sprintf(domain, "%s_%s", phl->tsa.index_name[idx], prodname);

  for (b=0; b<nb; b++){
    set_stack_save(stack, b, true);
    set_stack_date(stack, b, date);
    set_stack_domain(stack, b, domain);
  }

  return stack;
}


/** public functions
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


/** This function is the entry point to the time series analysis module
--- ard:       ARD
--- mask:      mask image
--- nt:        number of ARD products over time
--- phl:       HL parameters
--- endmember: endmember (if SMA was selected)
--- cube:      datacube definition
--- nproduct:  number of output stacks (returned)
+++ Return:    stacks with TSA results
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
stack_t **time_series_analysis(ard_t *ard, stack_t *mask, int nt, par_hl_t *phl, aux_emb_t *endmember, cube_t *cube, int *nproduct){
tsa_t ts;
stack_t ***TSA;
stack_t **PTR;
small *mask_ = NULL;
int idx;
int o, k, nprod = 0;
int nc;
int ni;
short nodata;


  // import stacks
  nc = get_stack_chunkncells(ard[0].DAT);
  nodata = get_stack_nodata(ard[0].DAT, 0);

  // import mask (if available)
  if (mask != NULL){
    if ((mask_ = get_band_small(mask, 0)) == NULL){
      printf("Error getting processing mask."); return NULL;}
  }

//  printf("allocating %d indices\n", phl->tsa.n);
  alloc((void**)&TSA, phl->tsa.n, sizeof(stack_t**));

  // number of steps for interpolation
  if (phl->tsa.tsi.method == _INT_NONE_){
    ni = nt;
  } else {
    ni = ceil((phl->date_range[_MAX_].ce-phl->date_range[_MIN_].ce+1)/phl->tsa.tsi.step);
  }


  
  for (idx=0; idx<phl->tsa.n; idx++){

    // compile products + stacks
    if ((TSA[idx] = compile_tsa(ard, &ts, phl, cube, nt, ni, idx, &nprod)) == NULL || nprod == 0){
      printf("Unable to compile TSA products!\n"); 
      free((void*)TSA);
      *nproduct = 0;
      return NULL;
    }

    
    tsa_spectral_index(ard, &ts, mask_, nc, nt, idx, nodata, &phl->tsa, &phl->sen, endmember);
    
    tsa_interpolation(&ts, mask_, nc, nt, ni, nodata, &phl->tsa.tsi);
    
    tsa_stm(&ts, mask_, nc, ni, nodata, &phl->tsa.stm);
    
    tsa_fold(&ts, mask_, nc, ni, nodata, phl);
    
    tsa_pheno(&ts, mask_, nc, ni, nodata, phl);

    tsa_trend(&ts, mask_, nc, nodata, phl);
   
    tsa_cat(&ts, mask_, nc, nodata, phl);
    
    tsa_standardize(&ts, mask_, nc, nt, ni, nodata, phl);

      
    // clean temporal information
    if (ts.d_tss != NULL){ free((void*)ts.d_tss); ts.d_tss = NULL;}
    if (ts.d_tsi != NULL){ free((void*)ts.d_tsi); ts.d_tsi = NULL;}
    if (ts.d_fby != NULL){ free((void*)ts.d_fby); ts.d_fby = NULL;}
    if (ts.d_fbq != NULL){ free((void*)ts.d_fbq); ts.d_fbq = NULL;}
    if (ts.d_fbm != NULL){ free((void*)ts.d_fbm); ts.d_fbm = NULL;}
    if (ts.d_fbw != NULL){ free((void*)ts.d_fbw); ts.d_fbw = NULL;}
    if (ts.d_fbd != NULL){ free((void*)ts.d_fbd); ts.d_fbd = NULL;}
    if (ts.d_lsp != NULL){ free((void*)ts.d_lsp); ts.d_lsp = NULL;}

  }
  

  // flatten out TSA stacks for returning to main
  alloc((void**)&PTR, phl->tsa.n*nprod, sizeof(stack_t*));
  
  for (idx=0, k=0; idx<phl->tsa.n; idx++){
    for (o=0; o<nprod; o++, k++) PTR[k] = TSA[idx][o];
  }
  
  for (idx=0; idx<phl->tsa.n; idx++) free((void*)TSA[idx]);
  free((void*)TSA);


  *nproduct = nprod*phl->tsa.n;
  return PTR;
}


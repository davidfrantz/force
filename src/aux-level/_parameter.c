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
This program generates parameter file skeletons, which may be used to
parameterize the FORCE programs
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


#include <stdio.h>   // core input and output functions
#include <stdlib.h>  // standard general utilities library
#include <string.h>  // string handling functions

#include "../cross-level/const-cl.h"
#include "../cross-level/konami-cl.h"
#include "../aux-level/param-aux.h"


void usage(char *prog){
  
  
  printf("usage: %s dir type verbose\n", prog); 
  printf("\n");
  printf("  type can be one of the following:\n");
  printf("    LEVEL2:   Level 2 Processing System\n");
  printf("    LEVEL3:   Level 3 Processing System\n");
  printf("    TSA:      Time Series Analysis\n");
  printf("    CSO:      Clear-Sky Observations\n");
  printf("    L2IMP:    Level 2 ImproPhe\n");
  printf("    CFIMP:    Continuous Field ImproPhe\n");
  printf("    SMP:      Sampling\n");
  printf("    TRAIN:    Train Machine Learner\n");
  printf("    SYNTHMIX: Synthetic Mixing\n");
  printf("    ML:       Machine Learning\n");
  printf("    TXT:      Texture\n");
  printf("    LSM:      Landscape Metrics\n");
  printf("    LIB:      Library Completeness\n");
  printf("  verbose (1) will generate long parameter\n");
  printf("    files with comments for each parameter. \n");
  printf("    verbose (0) will generate compact parameter\n");
  printf("    files without any comments.\n");
  printf("\n");

  exit(FAILURE);
}


int main( int argc, char *argv[] ){
FILE *fp;
char fname[NPOW_10];
char *dname = NULL;
char *ctype = NULL;
char *cverb = NULL;
int nchar;
int level, input_level, type;
bool verbose;


  if (argc >= 2) check_arg(argv[1]);
  if (argc != 4) usage(argv[0]);

  dname = argv[1];
  ctype = argv[2];
  cverb = argv[3];
  puts(ctype);
    
  if (strcmp(ctype, "LEVEL2") == 0){
    level = _LOWER_LEVEL_;
    input_level = _INP_RAW_;
    type = _LL_LEVEL2_;
  } else if (strcmp(ctype, "LEVEL3") == 0){
    level = _HIGHER_LEVEL_;
    input_level = _INP_ARD_;
    type = _HL_BAP_;
  } else if (strcmp(ctype, "TSA") == 0){
    level = _HIGHER_LEVEL_;
    input_level = _INP_ARD_;
    type = _HL_TSA_;
  } else if (strcmp(ctype, "CSO") == 0){
    level = _HIGHER_LEVEL_;
    input_level = _INP_QAI_;
    type = _HL_CSO_;
  } else if (strcmp(ctype, "CFIMP") == 0){
    level = _HIGHER_LEVEL_;
    input_level = _INP_ARD_;
    type = _HL_CFI_;
  } else if (strcmp(ctype, "L2IMP") == 0){
    level = _HIGHER_LEVEL_;
    input_level = _INP_ARD_;
    type = _HL_L2I_;
  } else if (strcmp(ctype, "ML") == 0){
    level = _HIGHER_LEVEL_;
    input_level = _INP_FTR_;
    type = _HL_ML_;
  } else if (strcmp(ctype, "SMP") == 0){
    level = _HIGHER_LEVEL_;
    input_level = _INP_FTR_;
    type = _HL_SMP_;
  } else if (strcmp(ctype, "TXT") == 0){
    level = _HIGHER_LEVEL_;
    input_level = _INP_FTR_;
    type = _HL_TXT_;
  } else if (strcmp(ctype, "LSM") == 0){
    level = _HIGHER_LEVEL_;
    input_level = _INP_FTR_;
    type = _HL_LSM_;
  } else if (strcmp(ctype, "LIB") == 0){
    level = _HIGHER_LEVEL_;
    input_level = _INP_FTR_;
    type = _HL_LIB_;
  } else if (strcmp(ctype, "TRAIN") == 0){
    level = _AUX_LEVEL_;
    input_level = _INP_AUX_;
    type = _AUX_TRAIN_;
  } else if (strcmp(ctype, "SYNTHMIX") == 0){
    level = _AUX_LEVEL_;
    input_level = _INP_AUX_;
    type = _AUX_SYNTHMIX_;
  } else {
    printf("No valid type!\n"); return FAILURE;
  }

  if (atoi(cverb) == 0){
    verbose = false;
  } else if (atoi(cverb) == 1){
    verbose = true;
  } else {
    printf("Unknown verbosity\n"); return FAILURE;
  }


  nchar = snprintf(fname, NPOW_10, "%s/%s-skeleton.prm", dname, ctype);
  if (nchar < 0 || nchar >= NPOW_10){ 
    printf("Buffer Overflow in assembling filename\n"); return FAILURE;}

  if ((fp = fopen(fname, "w")) == NULL){
    printf("Unable to open output file!\n"); return FAILURE; }


  fprintf(fp, "++PARAM_%s_START++\n", ctype);
  
  if (level == _HIGHER_LEVEL_){
    write_par_hl_dirs(fp, verbose);
    write_par_hl_mask(fp, verbose);
    write_par_hl_output(fp, verbose);
    write_par_hl_thread(fp, verbose);
    write_par_hl_extent(fp, verbose);
  }

  if (input_level == _INP_ARD_){
    write_par_hl_psf(fp, verbose);
    write_par_hl_improphed(fp, verbose);
    write_par_hl_sensor(fp, verbose);
    write_par_hl_qai(fp, verbose);
    write_par_hl_noise(fp, verbose);
    write_par_hl_time(fp, verbose);
  }

  if (input_level == _INP_QAI_){
    write_par_hl_sensor(fp, verbose);
    write_par_hl_qai(fp, verbose);
    write_par_hl_time(fp, verbose);
  }

  if (input_level == _INP_FTR_){
    write_par_hl_feature(fp, verbose);
  }

  if (type == _HL_BAP_){
    write_par_hl_bap(fp, verbose);
    write_par_hl_pac(fp, verbose);
  }

  if (type == _HL_TSA_){
    write_par_hl_index(fp, verbose);
    write_par_hl_sma(fp, verbose);
    write_par_hl_tsi(fp, verbose);
    write_par_hl_stm(fp, verbose);
    write_par_hl_fold(fp, verbose);
    write_par_hl_lsp(fp, verbose);
    write_par_hl_trend(fp, verbose);
  }

  if (type == _HL_CSO_){
    write_par_hl_cso(fp, verbose);
  }

  if (type == _HL_CFI_){
    write_par_hl_imp(fp, verbose);
    write_par_hl_cfi(fp, verbose);
  }

  if (type == _HL_L2I_){
    write_par_hl_imp(fp, verbose);
    write_par_hl_l2i(fp, verbose);
  }

  if (type == _HL_TXT_){
    write_par_hl_txt(fp, verbose);
  }

  if (type == _HL_LSM_){
    write_par_hl_lsm(fp, verbose);
  }

  if (type == _HL_LIB_){
    write_par_hl_lib(fp, verbose);
  }

  if (type == _HL_SMP_){
    write_par_hl_smp(fp, verbose);
  }

  if (type == _HL_ML_){
    write_par_hl_ml(fp, verbose);
  }
  
  if (type == _AUX_TRAIN_){
    write_par_hl_train(fp, verbose);
  }

  if (type == _AUX_SYNTHMIX_){
    write_par_hl_synthmix(fp, verbose);
  }
  
  if (level == _LOWER_LEVEL_){
    write_par_ll_dirs(fp, verbose);
    write_par_ll_dem(fp, verbose);
    write_par_ll_cube(fp, verbose);
    write_par_ll_atcor(fp, verbose);
    write_par_ll_wvp(fp, verbose);
    write_par_ll_aod(fp, verbose);
    write_par_ll_cloud(fp, verbose);
    write_par_ll_resmerge(fp, verbose);
    write_par_ll_coreg(fp, verbose);
    write_par_ll_misc(fp, verbose);
    write_par_ll_tier(fp, verbose);
    write_par_ll_thread(fp, verbose);
    write_par_ll_output(fp, verbose);
  }
  
  fprintf(fp, "\n++PARAM_%s_END++\n", ctype);

  fclose(fp);

  
  printf("An empty parameter file skeleton was written to\n  %s\n", fname);
  printf("Note that all parameters need to be given, even though some may not be used\n");
  printf("with your specific parameterization.\n");
  printf("You should rename the file, e.g. my-first-%s.prm.\n", ctype);
  printf("Parameterize according to your needs and run with\n");

  if (level == _LOWER_LEVEL_){
    printf("force-level2 %s/my-first-%s.prm\n", dname, ctype);
    printf(" or for a single image:\n");
    printf("force-l2ps image %s/my-first-%s.prm\n", dname, ctype);
  } else if (level == _HIGHER_LEVEL_){
    printf("force-higher-level %s/my-first-%s.prm\n", dname, ctype);
  } else if (level == _AUX_LEVEL_){
    if (type == _AUX_TRAIN_){
      printf("force-train %s/my-first-%s.prm\n", dname, ctype);
    } else if (type == _AUX_SYNTHMIX_){
      printf("force-synthmix %s/my-first-%s.prm\n", dname, ctype);
    }
  }


  return SUCCESS;
}


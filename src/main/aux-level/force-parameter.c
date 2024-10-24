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
This program generates parameter file skeletons, which may be used to
parameterize the FORCE programs
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


#include <stdio.h>   // core input and output functions
#include <stdlib.h>  // standard general utilities library
#include <string.h>  // string handling functions

#include <ctype.h>   // testing and mapping characters
#include <unistd.h>  // standard symbolic constants and types 

#include "../../modules/cross-level/const-cl.h"
#include "../../modules/cross-level/utils-cl.h"
#include "../../modules/cross-level/konami-cl.h"
#include "../../modules/cross-level/string-cl.h"
#include "../../modules/aux-level/param-aux.h"


typedef struct {
  int  n;
  char fname[NPOW_10];
  char module_tag[NPOW_10];
  int  module;
  int  level;
  int  input_level;
  bool comments;
} args_t;


void usage(char *exe, int exit_code){


  printf("Usage: %s [-h] [-v] [-i] [-m] [-c] parameter-file module\n", exe); 
  printf("\n");
  printf("  -h  = show this help\n");
  printf("  -v  = show version\n");
  printf("  -i  = show program's purpose\n");
  printf("  -m  = show available modules\n");
  printf("  -c  = generate more compact parameter files without comments\n");
  printf("\n");
  printf("  Positional arguments:\n");
  printf("  - 'parameter-file': output file\n");
  printf("  - 'module':         FORCE module.\n");
  printf("                      Use -m to show available modules.\n");
  printf("\n");

  exit(exit_code);
  return;
}


void show_modules(int exit_code){

  printf("  available modules:\n");
  printf("    LEVEL2:   Level 2 Processing System\n");
  printf("    LEVEL3:   Level 3 Processing System\n");
  printf("    TSA:      Time Series Analysis\n");
  printf("    CSO:      Clear-Sky Observations\n");
  printf("    UDF:      Plug-In User Defined Functions\n");
  printf("    L2IMP:    Level 2 ImproPhe\n");
  printf("    CFIMP:    Continuous Field ImproPhe\n");
  printf("    SMP:      Sampling\n");
  printf("    TRAIN:    Train Machine Learner\n");
  printf("    SYNTHMIX: Synthetic Mixing\n");
  printf("    ML:       Machine Learning\n");
  printf("    TXT:      Texture\n");
  printf("    LSM:      Landscape Metrics\n");
  printf("    LIB:      Library Completeness\n");
  printf("\n");

  exit(exit_code);
  return;
}


void parse_module(char *argv, args_t *args){
int m, n_modules = 14;
char module_tags[14][NPOW_10] = {
  "LEVEL2", "LEVEL3", "TSA",   "CSO", 
  "UDF",    "CFIMP",  "L2IMP", "ML",      
  "SMP",    "TXT",    "LSM",   "LIB",
  "TRAIN",  "SYNTHMIX" };
int modules[14] = {
  _LL_LEVEL2_, _HL_BAP_, _HL_TSA_, _HL_CSO_, 
  _HL_UDF_,    _HL_CFI_, _HL_L2I_, _HL_ML_, 
  _HL_SMP_,    _HL_TXT_, _HL_LSM_, _HL_LIB_, 
  _AUX_TRAIN_, _AUX_SYNTHMIX_ };
int levels[14] = {
  _LOWER_LEVEL_,  _HIGHER_LEVEL_, _HIGHER_LEVEL_, _HIGHER_LEVEL_, 
  _HIGHER_LEVEL_, _HIGHER_LEVEL_, _HIGHER_LEVEL_, _HIGHER_LEVEL_, 
  _HIGHER_LEVEL_, _HIGHER_LEVEL_, _HIGHER_LEVEL_, _HIGHER_LEVEL_, 
  _AUX_LEVEL_,    _AUX_LEVEL_ };
int input_levels[14] = {
  _INP_RAW_, _INP_ARD_, _INP_ARD_, _INP_QAI_,
  _INP_ARD_, _INP_ARD_, _INP_ARD_, _INP_FTR_,
  _INP_FTR_, _INP_FTR_, _INP_FTR_, _INP_FTR_,
  _INP_AUX_, _INP_AUX_ };


  copy_string(args->module_tag, NPOW_10, argv);

  for (m=0; m<n_modules; m++){
    if (strcmp(argv, module_tags[m]) == 0){
      args->module = modules[m];
      args->level = levels[m];
      args->input_level = input_levels[m];
      return;
    }
  }

  fprintf(stderr, "No valid module.\n");
  show_modules(FAILURE);

  return;
}


void parse_args(int argc, char *argv[], args_t *args){
int opt;


  opterr = 0;

  // default parameters
  args->comments = true;

  // optional parameters
  while ((opt = getopt(argc, argv, "hvimc")) != -1){
    switch(opt){
      case 'h':
        usage(argv[0], SUCCESS);
      case 'v':
        get_version(NULL, 0);
        exit(SUCCESS);
      case 'i':
        printf("Generation of parameter files\n");
        exit(SUCCESS);
      case 'm':
        show_modules(SUCCESS);
      case 'c':
        args->comments = false;
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
  args->n = 2;

  if (optind < argc){
    konami_args(argv[optind]);
    if (argc-optind == args->n){
      copy_string(args->fname, NPOW_10, argv[optind++]);
      parse_module(argv[optind++], args);
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

  return;
}


int main( int argc, char *argv[] ){
args_t args;
FILE *fp;


  parse_args(argc, argv, &args);

  if ((fp = fopen(args.fname, "w")) == NULL){
    printf("Unable to open output file!\n"); return FAILURE; }


  fprintf(fp, "++PARAM_%s_START++\n", args.module_tag);
  
  if (args.level == _HIGHER_LEVEL_){
    write_par_hl_dirs(fp, args.comments);
    write_par_hl_mask(fp, args.comments);
    write_par_hl_output(fp, args.comments);
    write_par_hl_thread(fp, args.comments);
    write_par_hl_extent(fp, args.comments);
  }

  if (args.input_level == _INP_ARD_){
    write_par_hl_psf(fp, args.comments);
    write_par_hl_improphed(fp, args.comments);
    write_par_hl_sensor(fp, args.comments);
    write_par_hl_qai(fp, args.comments);
    write_par_hl_noise(fp, args.comments);
    write_par_hl_time(fp, args.comments);
  }

  if (args.input_level == _INP_QAI_){
    write_par_hl_sensor(fp, args.comments);
    write_par_hl_qai(fp, args.comments);
    write_par_hl_time(fp, args.comments);
  }

  if (args.input_level == _INP_FTR_){
    write_par_hl_feature(fp, args.comments);
  }

  if (args.module == _HL_BAP_){
    write_par_hl_bap(fp, args.comments);
    write_par_hl_pac(fp, args.comments);
  }

  if (args.module == _HL_TSA_){
    write_par_hl_index(fp, args.comments);
    write_par_hl_sma(fp, args.comments);
    write_par_hl_tsi(fp, args.comments);
    write_par_hl_pyp(fp, args.comments);
    write_par_hl_rsp(fp, args.comments);
    write_par_hl_stm(fp, args.comments);
    write_par_hl_fold(fp, args.comments);
    write_par_hl_lsp(fp, args.comments);
    write_par_hl_pol(fp, args.comments);
    write_par_hl_trend(fp, args.comments);
  }

  if (args.module == _HL_CSO_){
    write_par_hl_cso(fp, args.comments);
  }

  if (args.module == _HL_UDF_){
    write_par_hl_pyp(fp, args.comments);
    write_par_hl_rsp(fp, args.comments);
  }

  if (args.module == _HL_CFI_){
    write_par_hl_imp(fp, args.comments);
    write_par_hl_cfi(fp, args.comments);
  }

  if (args.module == _HL_L2I_){
    write_par_hl_imp(fp, args.comments);
    write_par_hl_l2i(fp, args.comments);
  }

  if (args.module == _HL_TXT_){
    write_par_hl_txt(fp, args.comments);
  }

  if (args.module == _HL_LSM_){
    write_par_hl_lsm(fp, args.comments);
  }

  if (args.module == _HL_LIB_){
    write_par_hl_lib(fp, args.comments);
  }

  if (args.module == _HL_SMP_){
    write_par_hl_smp(fp, args.comments);
  }

  if (args.module == _HL_ML_){
    write_par_hl_ml(fp, args.comments);
  }
  
  if (args.module == _AUX_TRAIN_){
    write_par_aux_train(fp, args.comments);
  }

  if (args.module == _AUX_SYNTHMIX_){
    write_par_aux_synthmix(fp, args.comments);
  }
  
  if (args.level == _LOWER_LEVEL_){
    write_par_ll_dirs(fp, args.comments);
    write_par_ll_dem(fp, args.comments);
    write_par_ll_cube(fp, args.comments);
    write_par_ll_atcor(fp, args.comments);
    write_par_ll_wvp(fp, args.comments);
    write_par_ll_aod(fp, args.comments);
    write_par_ll_cloud(fp, args.comments);
    write_par_ll_resmerge(fp, args.comments);
    write_par_ll_coreg(fp, args.comments);
    write_par_ll_misc(fp, args.comments);
    write_par_ll_tier(fp, args.comments);
    write_par_ll_thread(fp, args.comments);
    write_par_ll_output(fp, args.comments);
  }
  
  fprintf(fp, "\n++PARAM_%s_END++\n", args.module_tag);

  fclose(fp);

  
  printf("An empty parameter file skeleton was written to\n  %s\n", args.fname);
  printf("Note that all parameters need to be given, even though some may not be used\n");
  printf("with your specific parameterization.\n");
  printf("Parameterize according to your needs and run with\n");

  if (args.level == _LOWER_LEVEL_){
    printf("force-level2 %s\n", args.fname);
    printf(" or for a single image:\n");
    printf("force-l2ps image %s\n", args.fname);
  } else if (args.level == _HIGHER_LEVEL_){
    printf("force-higher-level %s\n", args.fname);
  } else if (args.level == _AUX_LEVEL_){
    if (args.module == _AUX_TRAIN_){
      printf("force-train %s\n", args.fname);
    } else if (args.module == _AUX_SYNTHMIX_){
      printf("force-synthmix %s\n", args.fname);
    }
  }


  return SUCCESS;
}


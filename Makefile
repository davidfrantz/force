##########################################################################
# 
# This file is part of FORCE - Framework for Operational Radiometric 
# Correction for Environmental monitoring.
# 
# Copyright (C) 2013-2020 David Frantz
# 
# FORCE is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# FORCE is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with FORCE. If not, see <http://www.gnu.org/licenses/>.
# 
##########################################################################

##########################################################################
# Modify the following lines to match your needs

# Installation directory
BINDIR=/usr/local/bin

# Libraries
GDAL=-I/usr/include/gdal -L/usr/lib -Wl,-rpath=/usr/lib
GSL=-I/usr/include/gsl -L/usr/lib/x86_64-linux-gnu -Wl,-rpath=/usr/lib/x86_64-linux-gnu -DHAVE_INLINE=1 -DGSL_RANGE_CHECK=0
CURL=-I/usr/include/curl -L/usr/lib/x86_64-linux-gnu -Wl,-rpath=/usr/lib/x86_64-linux-gnu -I/usr/include/x86_64-linux-gnu/curl -L/usr/lib/x86_64-linux-gnu -Wl,-rpath=/usr/lib/x86_64-linux-gnu
OPENCV=-I/usr/local/include/opencv4 -L/usr/local/lib -Wl,-rpath=/usr/local/lib
PYTHON != python3-config --includes 
PYTHON2 != python3-config --ldflags
#SPLITS=-I/usr/local/include/splits -L/usr/local/lib -Wl,-rpath=/usr/local/lib

# Linked libs
LDGDAL=-lgdal
LDGSL=-lgsl -lgslcblas
#LDSPLITS=-lsplits -larmadillo
LDOPENCV=-lopencv_core -lopencv_ml -lopencv_imgproc
LDCURL=-lcurl
LDPYTHON != (python3-config --libs --embed || python3-config --libs) | tail -n 1

# NO! changes below this line (unless you know what to do, then go ahead)
##########################################################################


### COMPILER

GCC=gcc
GPP=g++
G11=g++ -std=c++11

CFLAGS=-O3 -Wall -fopenmp
#CFLAGS=-g -Wall -fopenmp


### DIRECTORIES

DB=bash
DP=python
DC=src/cross-level
DL=src/lower-level
DH=src/higher-level
DA=src/aux-level
TB=temp-bin
TC=temp-cross
TL=temp-lower
TH=temp-higher
TA=temp-aux


### TARGETS

all: temp cross lower higher aux exe
cross: string_cl enum_cl cite_cl utils_cl alloc_cl brick_cl imagefuns_cl param_cl date_cl datesys_cl lock_cl cube_cl dir_cl stats_cl pca_cl tile_cl queue_cl warp_cl sun_cl quality_cl sys_cl konami_cl download_cl read_cl
lower: table_ll param_ll meta_ll cube_ll equi7_ll glance7_ll atc_ll sunview_ll read_ll radtran_ll topo_ll cloud_ll gas_ll brdf_ll atmo_ll aod_ll resmerge_ll coreg_ll coregfuns_ll acix_ll modwvp_ll
higher: param_hl progress_hl tasks_hl read-aux_hl read-ard_hl quality_hl bap_hl level3_hl cso_hl tsa_hl index_hl interpolate_hl stm_hl fold_hl standardize_hl pheno_hl polar_hl trend_hl ml_hl texture_hl lsm_hl lib_hl sample_hl imp_hl cfimp_hl l2imp_hl pyp_hl
aux: param_aux param_train_aux train_aux
exe: force force-parameter force-qai-inflate force-tile-finder force-tabulate-grid force-l2ps force-higher-level force-train force-lut-modis force-mdcp force-stack force-import-modis
.PHONY: temp all install install_ bash python clean build


### TEMP

temp:
	mkdir -p $(TB) $(TC) $(TL) $(TH) $(TA)


### CROSS LEVEL COMPILE UNITS

string_cl: temp $(DC)/string-cl.c
	$(GCC) $(CFLAGS) -c $(DC)/string-cl.c -o $(TC)/string_cl.o

enum_cl: temp $(DC)/enum-cl.c
	$(GCC) $(CFLAGS) -c $(DC)/enum-cl.c -o $(TC)/enum_cl.o

cite_cl: temp $(DC)/cite-cl.c
	$(GCC) $(CFLAGS) -c $(DC)/cite-cl.c -o $(TC)/cite_cl.o

utils_cl: temp $(DC)/utils-cl.c
	$(GCC) $(CFLAGS) -c $(DC)/utils-cl.c -o $(TC)/utils_cl.o -lm

alloc_cl: temp $(DC)/alloc-cl.c
	$(GCC) $(CFLAGS) -c $(DC)/alloc-cl.c -o $(TC)/alloc_cl.o

brick_cl: temp $(DC)/brick-cl.c
	$(G11) $(CFLAGS) $(GDAL) -c $(DC)/brick-cl.c -o $(TC)/brick_cl.o $(LDGDAL)

imagefuns_cl: temp $(DC)/imagefuns-cl.c
	$(GCC) $(CFLAGS) -c $(DC)/imagefuns-cl.c -o $(TC)/imagefuns_cl.o

param_cl: temp $(DC)/param-cl.c
	$(GCC) $(CFLAGS) -c $(DC)/param-cl.c -o $(TC)/param_cl.o

date_cl: temp $(DC)/date-cl.c
	$(GCC) $(CFLAGS) -c $(DC)/date-cl.c -o $(TC)/date_cl.o

datesys_cl : temp $(DC)/datesys-cl.c
	$(GCC) $(CFLAGS) -c $(DC)/datesys-cl.c -o $(TC)/datesys_cl.o

lock_cl: temp $(DC)/lock-cl.c
	$(GCC) $(CFLAGS) $(GDAL) -c $(DC)/lock-cl.c -o $(TC)/lock_cl.o $(LDGDAL)

sys_cl: temp $(DC)/sys-cl.c
	$(GCC) $(CFLAGS) -c $(DC)/sys-cl.c -o $(TC)/sys_cl.o

konami_cl: temp $(DC)/konami-cl.c
	$(GCC) $(CFLAGS) -c $(DC)/konami-cl.c -o $(TC)/konami_cl.o

sun_cl: temp $(DC)/sun-cl.c
	$(GCC) $(CFLAGS) -c $(DC)/sun-cl.c -o $(TC)/sun_cl.o

cube_cl: temp $(DC)/cube-cl.c
	$(GCC) $(CFLAGS) -c $(DC)/cube-cl.c -o $(TC)/cube_cl.o

tile_cl: temp $(DC)/tile-cl.c
	$(GCC) $(CFLAGS) $(GDAL) -c $(DC)/tile-cl.c -o $(TC)/tile_cl.o $(LDGDAL)

dir_cl: temp $(DC)/dir-cl.c
	$(GCC) $(CFLAGS) -c $(DC)/dir-cl.c -o $(TC)/dir_cl.o

stats_cl: temp $(DC)/stats-cl.c
	$(GCC) $(CFLAGS) -c $(DC)/stats-cl.c -o $(TC)/stats_cl.o

pca_cl: temp $(DC)/pca-cl.c
	$(GCC) $(CFLAGS) $(GSL) -c $(DC)/pca-cl.c -o $(TC)/pca_cl.o $(LDGSL)

queue_cl: temp $(DC)/queue-cl.c
	$(GCC) $(CFLAGS) -c $(DC)/queue-cl.c -o $(TC)/queue_cl.o

warp_cl: temp $(DC)/warp-cl.cpp
	$(G11) $(CFLAGS) $(GDAL) -c $(DC)/warp-cl.cpp -o $(TC)/warp_cl.o $(LDGDAL)

quality_cl: temp $(DC)/quality-cl.c
	$(GCC) $(CFLAGS) -c $(DC)/quality-cl.c -o $(TC)/quality_cl.o

download_cl: temp $(DC)/download-cl.c
	$(GCC) $(CFLAGS) $(CURL) -c $(DC)/download-cl.c -o $(TC)/download_cl.o $(LDCURL)

read_cl: temp $(DC)/read-cl.c
	$(GCC) $(CFLAGS) -c $(DC)/read-cl.c -o $(TC)/read_cl.o


### LOWER LEVEL COMPILE UNITS

table_ll: temp $(DL)/table-ll.c
	$(GCC) $(CFLAGS) -c $(DL)/table-ll.c -o $(TL)/table_ll.o

param_ll: temp $(DL)/param-ll.c
	$(GCC) $(CFLAGS) -c $(DL)/param-ll.c -o $(TL)/param_ll.o

meta_ll: temp $(DL)/meta-ll.c
	$(GCC) $(CFLAGS) $(GDAL) -c $(DL)/meta-ll.c -o $(TL)/meta_ll.o $(LDGDAL)

atc_ll: temp $(DL)/atc-ll.c
	$(GCC) $(CFLAGS) -c $(DL)/atc-ll.c -o $(TL)/atc.o

sunview_ll: temp $(DL)/sunview-ll.c
	$(GCC) $(CFLAGS) -c $(DL)/sunview-ll.c -o $(TL)/sunview.o

read_ll: temp $(DL)/read-ll.c
	$(GCC) $(CFLAGS) $(GDAL) -c $(DL)/read-ll.c -o $(TL)/read.o $(LDGDAL)

brdf_ll: temp $(DL)/brdf-ll.c
	$(GCC) $(CFLAGS) -c $(DL)/brdf-ll.c -o $(TL)/brdf_ll.o

geo_ll: temp $(DL)/geo-ll.cpp
	$(GCC) $(CFLAGS) -c $(DL)/geo-ll.c -o $(TL)/geo_ll.o

cube_ll: temp $(DL)/cube-ll.c
	$(GCC) $(CFLAGS) -c $(DL)/cube-ll.c -o $(TL)/cube_ll.o
 
equi7_ll: temp $(DL)/equi7-ll.c
	$(GCC) $(CFLAGS) -c $(DL)/equi7-ll.c -o $(TL)/equi7_ll.o

glance7_ll: temp $(DL)/glance7-ll.c
	$(GCC) $(CFLAGS) -c $(DL)/glance7-ll.c -o $(TL)/glance7_ll.o

radtran_ll: temp $(DL)/radtran-ll.c
	$(GCC) $(CFLAGS) -c $(DL)/radtran-ll.c -o $(TL)/radtran_ll.o

topo_ll: temp $(DL)/topo-ll.c
	$(GCC) $(CFLAGS) -c $(DL)/topo-ll.c -o $(TL)/topo_ll.o

cloud_ll: temp $(DL)/cloud-ll.c
	$(GCC) $(CFLAGS) -c $(DL)/cloud-ll.c -o $(TL)/cloud_ll.o

gas_ll: temp $(DL)/gas-ll.c
	$(GCC) $(CFLAGS) $(GSL) -c $(DL)/gas-ll.c -o $(TL)/gas_ll.o $(LDGSL)

atmo_ll: temp $(DL)/atmo-ll.c
	$(GCC) $(CFLAGS) $(GDAL) -c $(DL)/atmo-ll.c -o $(TL)/atmo_ll.o $(LDGDAL)

aod_ll: temp $(DL)/aod-ll.c
	$(GCC) $(CFLAGS) $(GDAL) $(GSL) -c $(DL)/aod-ll.c -o $(TL)/aod_ll.o $(LDGDAL) $(LDGSL)

resmerge_ll: temp $(DL)/resmerge-ll.c
	$(GCC) $(CFLAGS) $(GSL) -c $(DL)/resmerge-ll.c -o $(TL)/resmerge_ll.o $(LDGSL)

coreg_ll: temp $(DL)/coreg-ll.c
	$(GCC) $(CFLAGS) -c $(DL)/coreg-ll.c -o $(TL)/coreg_ll.o

coregfuns_ll: temp $(DL)/coregfuns-ll.c
	$(GCC) $(CFLAGS) -c $(DL)/coregfuns-ll.c -o $(TL)/coregfuns_ll.o

acix_ll: temp $(DL)/acix-ll.c
	$(GCC) $(CFLAGS) -c $(DL)/acix-ll.c -o $(TL)/acix_ll.o

 modwvp_ll: temp $(DL)/modwvp-ll.c
	$(GCC) $(CFLAGS) $(GDAL) -c $(DL)/modwvp-ll.c -o $(TL)/modwvp_ll.o $(LDGDAL)

 
### HIGHER LEVEL COMPILE UNITS
 
param_hl: temp $(DH)/param-hl.c
	$(GCC) $(CFLAGS) -c $(DH)/param-hl.c -o $(TH)/param_hl.o

progress_hl: temp $(DH)/progress-hl.c
	$(GCC) $(CFLAGS) -c $(DH)/progress-hl.c -o $(TH)/progress_hl.o

tasks_hl: temp $(DH)/tasks-hl.c
	$(G11) $(CFLAGS) $(GDAL) $(OPENCV) -c $(DH)/tasks-hl.c -o $(TH)/tasks_hl.o $(LDGDAL) $(LDOPENCV)

read-aux_hl: temp $(DH)/read-aux-hl.c
	$(G11) $(CFLAGS) $(OPENCV) -c $(DH)/read-aux-hl.c -o $(TH)/read-aux_hl.o $(LDOPENCV)

read-ard_hl: temp $(DH)/read-ard-hl.c
	$(GCC) $(CFLAGS) $(GDAL) -c $(DH)/read-ard-hl.c -o $(TH)/read-ard_hl.o $(LDGDAL)

quality_hl: temp $(DH)/quality-hl.c
	$(GCC) $(CFLAGS) -c $(DH)/quality-hl.c -o $(TH)/quality_hl.o

index_hl: temp $(DH)/index-hl.c
	$(GCC) $(CFLAGS) $(GSL) -c $(DH)/index-hl.c -o $(TH)/index_hl.o $(LDGSL)

interpolate_hl: temp $(DH)/interpolate-hl.c
	$(GCC) $(CFLAGS) -c $(DH)/interpolate-hl.c -o $(TH)/interpolate_hl.o

stm_hl: temp $(DH)/stm-hl.c
	$(GCC) $(CFLAGS) -c $(DH)/stm-hl.c -o $(TH)/stm_hl.o

fold_hl: temp $(DH)/fold-hl.c
	$(GCC) $(CFLAGS) -c $(DH)/fold-hl.c -o $(TH)/fold_hl.o

standardize_hl: temp $(DH)/standardize-hl.c
	$(GCC) $(CFLAGS) -c $(DH)/standardize-hl.c -o $(TH)/standardize_hl.o
 
# SPLITS crashes if compiled with C++11
pheno_hl: temp $(DH)/pheno-hl.cpp
	$(GPP) $(CFLAGS) $(SPLITS) -c $(DH)/pheno-hl.cpp -o $(TH)/pheno_hl.o $(LDSPLITS)

polar_hl: temp $(DH)/polar-hl.c
	$(GCC) $(CFLAGS) -c $(DH)/polar-hl.c -o $(TH)/polar_hl.o

trend_hl: temp $(DH)/trend-hl.c
	$(GCC) $(CFLAGS) -c $(DH)/trend-hl.c -o $(TH)/trend_hl.o
 
bap_hl: temp $(DH)/bap-hl.c
	$(GCC) $(CFLAGS) $(GSL) -c $(DH)/bap-hl.c -o $(TH)/bap_hl.o $(LDGSL)

level3_hl: temp $(DH)/level3-hl.c
	$(GCC) $(CFLAGS) -c $(DH)/level3-hl.c -o $(TH)/level3_hl.o

cso_hl: temp $(DH)/cso-hl.c
	$(GCC) $(CFLAGS) -c $(DH)/cso-hl.c -o $(TH)/cso_hl.o

tsa_hl: temp $(DH)/tsa-hl.c
	$(GCC) $(CFLAGS) -c $(DH)/tsa-hl.c -o $(TH)/tsa_hl.o

ml_hl: temp $(DH)/ml-hl.c
	$(G11) $(CFLAGS) $(OPENCV) -c $(DH)/ml-hl.c -o $(TH)/ml_hl.o $(LDOPENCV)

texture_hl: temp $(DH)/texture-hl.c
	$(G11) $(CFLAGS) $(OPENCV) -c $(DH)/texture-hl.c -o $(TH)/texture_hl.o $(LDOPENCV)

lsm_hl: temp $(DH)/lsm-hl.c
	$(GCC) $(CFLAGS) -c $(DH)/lsm-hl.c -o $(TH)/lsm_hl.o

lib_hl: temp $(DH)/lib-hl.c
	$(GCC) $(CFLAGS) $(OPENCV) -c $(DH)/lib-hl.c -o $(TH)/lib_hl.o

sample_hl: temp $(DH)/sample-hl.c
	$(G11) $(CFLAGS) -c $(DH)/sample-hl.c -o $(TH)/sample_hl.o

imp_hl: temp $(DH)/improphe-hl.c
	$(GCC) $(CFLAGS) -c $(DH)/improphe-hl.c -o $(TH)/imp_hl.o
 
cfimp_hl: temp $(DH)/cf-improphe-hl.c
	$(GCC) $(CFLAGS) -c $(DH)/cf-improphe-hl.c -o $(TH)/cfimp_hl.o
 
l2imp_hl: temp $(DH)/l2-improphe-hl.c
	$(GCC) $(CFLAGS) -c $(DH)/l2-improphe-hl.c -o $(TH)/l2imp_hl.o

pyp_hl: temp $(DH)/py-plugin-hl.c
	$(GCC) $(CFLAGS) $(PYTHON) -c $(DH)/py-plugin-hl.c -o $(TH)/pyp_hl.o $(LDPYTHON)


### AUX COMPILE UNITS

param_aux: temp $(DA)/param-aux.c
	$(GCC) $(CFLAGS) -c $(DA)/param-aux.c -o $(TA)/param_aux.o

param_train_aux: temp $(DA)/param-train-aux.c
	$(GCC) $(CFLAGS) -c $(DA)/param-train-aux.c -o $(TA)/param_train_aux.o

train_aux: temp $(DA)/train-aux.cpp
	$(G11) $(CFLAGS) $(OPENCV) -c $(DA)/train-aux.cpp -o $(TA)/train_aux.o $(LDOPENCV)


### EXECUTABLES

force: temp cross $(DA)/_main.c
	$(G11) $(CFLAGS) $(GDAL) $(GSL) $(CURL) -o $(TB)/force $(DA)/_main.c $(TC)/*.o $(LDGDAL) $(LDGSL) $(LDCURL)

force-parameter: temp cross aux $(DA)/_parameter.c
	$(G11) $(CFLAGS) $(GDAL) $(GSL) $(CURL) $(OPENCV) -o $(TB)/force-parameter $(DA)/_parameter.c $(TC)/*.o $(TA)/*.o $(LDGDAL) $(LDGSL) $(LDCURL) $(LDOPENCV)

force-tile-finder: temp cross $(DA)/_tile-finder.c
	$(G11) $(CFLAGS) $(GDAL) $(GSL) $(CURL) -o $(TB)/force-tile-finder $(DA)/_tile-finder.c $(TC)/*.o $(LDGDAL) $(LDGSL) $(LDCURL)

force-tabulate-grid: temp cross $(DA)/_tabulate-grid.c
	$(G11) $(CFLAGS) $(GDAL) $(GSL) $(CURL) -o $(TB)/force-tabulate-grid $(DA)/_tabulate-grid.c $(TC)/*.o $(LDGDAL) $(LDGSL) $(LDCURL)

force-train: temp cross aux $(DA)/_train.cpp
	$(G11) $(CFLAGS) $(GDAL) $(GSL) $(CURL) $(OPENCV) -o $(TB)/force-train $(DA)/_train.cpp $(TC)/*.o $(TA)/*.o $(LDGDAL) $(LDGSL) $(LDCURL) $(LDOPENCV)
 
force-qai-inflate: temp cross higher $(DA)/_quality-inflate.c
	$(G11) $(CFLAGS) $(GDAL) $(GSL) $(CURL) $(SPLITS) $(OPENCV) $(PYTHON) -o $(TB)/force-qai-inflate $(DA)/_quality-inflate.c $(TC)/*.o $(TH)/*.o $(LDGDAL) $(LDGSL) $(LDCURL) $(LDSPLITS) $(LDOPENCV) $(LDPYTHON)
 
force-l2ps: temp cross lower $(DL)/_level2.c
	$(G11) $(CFLAGS) $(GDAL) $(GSL) $(CURL) -o $(TB)/force-l2ps $(DL)/_level2.c $(TC)/*.o $(TL)/*.o $(LDGDAL) $(LDGSL) $(LDCURL)

force-higher-level: temp cross higher $(DH)/_higher-level.c
	$(G11) $(CFLAGS) $(GDAL) $(GSL) $(CURL) $(SPLITS) $(OPENCV) $(PYTHON) $(PYTHON2) -o $(TB)/force-higher-level $(DH)/_higher-level.c $(TC)/*.o $(TH)/*.o $(LDGDAL) $(LDGSL) $(LDCURL) $(LDSPLITS) $(LDOPENCV) $(LDPYTHON)

force-lut-modis: temp cross lower $(DL)/_lut-modis.c
	$(G11) $(CFLAGS) $(GDAL) $(GSL) $(CURL) -o $(TB)/force-lut-modis $(DL)/_lut-modis.c $(TC)/*.o $(TL)/*.o $(LDGDAL) $(LDGSL) $(LDCURL)

force-mdcp: temp cross $(DA)/_md_copy.c
	$(G11) $(CFLAGS) $(GDAL) $(GSL) $(CURL) -o $(TB)/force-mdcp $(DA)/_md_copy.c $(TC)/*.o $(LDGDAL) $(LDGSL) $(LDCURL)

force-stack: temp cross $(DA)/_stack.c
	$(G11) $(CFLAGS) $(GDAL) $(GSL) $(CURL) -o $(TB)/force-stack $(DA)/_stack.c $(TC)/*.o $(LDGDAL) $(LDGSL) $(LDCURL)

force-import-modis: temp cross lower $(DL)/_import-modis.c
	$(G11) $(CFLAGS) $(GDAL) $(GSL) $(CURL) -o $(TB)/force-import-modis $(DL)/_import-modis.c $(TC)/*.o $(TL)/*.o $(LDGDAL) $(LDGSL) $(LDCURL)

### dummy code for testing stuff  

dummy: temp cross aux higher src/dummy.c
	$(G11) $(CFLAGS) $(GDAL) $(GSL) $(CURL) $(SPLITS) $(OPENCV) -o $(TB)/dummy src/dummy.c $(TC)/*.o $(TA)/*.o $(TH)/*.o $(LDGDAL) $(LDGSL) $(LDCURL) $(LDSPLITS) $(LDOPENCV)

  
### MISC

install_:
	chmod 0755 $(TB)/*
	cp $(TB)/* $(BINDIR)

clean:
	rm -rf $(TB) $(TC) $(TL) $(TH) $(TA) 

bash: temp
	cp $(DB)/force-cube.sh $(TB)/force-cube
	cp $(DB)/force-l2ps_.sh $(TB)/force-l2ps_
	cp $(DB)/force-level1-csd.sh $(TB)/force-level1-csd  
	cp $(DB)/force-level1-landsat.sh $(TB)/force-level1-landsat
	cp $(DB)/force-level1-sentinel2.sh $(TB)/force-level1-sentinel2
	cp $(DB)/force-level2.sh $(TB)/force-level2
	cp $(DB)/force-mosaic.sh $(TB)/force-mosaic
	cp $(DB)/force-pyramid.sh $(TB)/force-pyramid
	cp $(DB)/force-procmask.sh $(TB)/force-procmask
	cp $(DB)/force-tile-extent.sh $(TB)/force-tile-extent
	cp $(DB)/force-magic-parameters.sh $(TB)/force-magic-parameters
	sed -i 's+BINDIR=???+BINDIR=$(BINDIR)+g' $(TB)/force-level2

python: temp
	cp $(DP)/force-synthmix.py $(TB)/force-synthmix

install: bash python install_ clean

build:
	$(eval V := $(shell grep '#define _VERSION_' src/cross-level/const-cl.h | cut -d '"' -f 2 | sed 's/ /_/g'))
	$(eval T :=$(shell date +"%Y%m%d%H%M%S"))
	tar -czf force_v$(V)_$(T).tar.gz src bash python images docs Makefile LICENSE README.md

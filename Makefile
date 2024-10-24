##########################################################################
# 
# This file is part of FORCE - Framework for Operational Radiometric 
# Correction for Environmental monitoring.
# 
# Copyright (C) 2013-2024 David Frantz
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
##########################################################################

# Installation directory
BINDIR=/usr/local/bin

# Libraries
GDAL_INCLUDES = -I/usr/include/gdal
GDAL_LIBS = -L/usr/lib -lgdal
GDAL_FLAGS = -Wl,-rpath=/usr/lib

GSL_INCLUDES = -I/usr/include/gsl
GSL_LIBS = -L/usr/lib/x86_64-linux-gnu -lgsl -lgslcblas
GSL_FLAGS = -Wl,-rpath=/usr/lib/x86_64-linux-gnu -DHAVE_INLINE=1 -DGSL_RANGE_CHECK=0

CURL_INCLUDES = -I/usr/include/curl -I/usr/include/x86_64-linux-gnu/curl
CURL_LIBS = -L/usr/lib/x86_64-linux-gnu -lcurl
CURL_FLAGS = -Wl,-rpath=/usr/lib/x86_64-linux-gnu

OPENCV_INCLUDES = -I/usr/local/include/opencv4
OPENCV_LIBS = -L/usr/local/lib -lopencv_core -lopencv_ml -lopencv_imgproc
OPENCV_FLAGS = -Wl,-rpath=/usr/local/lib

PYTHON_INCLUDES = $(shell python3-config --includes)
PYTHON_LIBS = $(shell (python3-config --ldflags --libs --embed || python3-config --ldflags --libs) | tr -d '\n')

RSTATS_INCLUDES = $(shell R CMD config --cppflags)
RSTATS_LIBS = $(shell R CMD config --ldflags | sed 's/ /\n/g' | grep '\-L') -lR

INCLUDES = $(GDAL_INCLUDES) $(GSL_INCLUDES) $(CURL_INCLUDES) $(OPENCV_INCLUDES) $(PYTHON_INCLUDES) $(RSTATS_INCLUDES)
LIBS = $(GDAL_LIBS) $(GSL_LIBS) $(CURL_LIBS) $(OPENCV_LIBS) $(PYTHON_LIBS) $(RSTATS_LIBS)
FLAGS = $(GDAL_FLAGS) $(GSL_FLAGS) $(CURL_FLAGS) $(OPENCV_FLAGS) $(PYTHON_FLAGS) $(RSTATS_FLAGS)


##########################################################################
# NO! changes below this line (unless you know what to do, then go ahead)
##########################################################################

### Compiler

CXX=g++ -std=c++11

# Compilation Flags
CFLAGS=-O3 -Wall -fopenmp
#CFLAGS=-g -Wall -fopenmp

##########################################################################

# Directories
SRCDIR = src
OBJDIR = obj
BINDIR = bin

# Create necessary directories
$(shell mkdir -p $(OBJDIR) $(BINDIR) $(BINDIR)/force-test $(BINDIR)/force-misc)

# Source Files (modules)
CROSS_SRC = $(wildcard $(SRCDIR)/modules/cross-level/*.c)
LOWER_SRC = $(wildcard $(SRCDIR)/modules/lower-level/*.c)
HIGHER_SRC = $(wildcard $(SRCDIR)/modules/higher-level/*.c)
AUX_SRC = $(wildcard $(SRCDIR)/modules/aux-level/*.c)

# Source Files (main executables)
MAIN_AUX_SRC = $(wildcard $(SRCDIR)/main/aux-level/*.c)
MAIN_LOWER_SRC = $(wildcard $(SRCDIR)/main/lower-level/*.c)
MAIN_HIGHER_SRC = $(wildcard $(SRCDIR)/main/higher-level/*.c)

# Source Files (test executables)
TEST_SRC = $(wildcard $(SRCDIR)/tests/*.c)

# Object Files
CROSS_OBJ = $(patsubst $(SRCDIR)/modules/cross-level/%.c, $(OBJDIR)/%.o, $(CROSS_SRC))
LOWER_OBJ = $(patsubst $(SRCDIR)/modules/lower-level/%.c, $(OBJDIR)/%.o, $(LOWER_SRC))
HIGHER_OBJ = $(patsubst $(SRCDIR)/modules/higher-level/%.c, $(OBJDIR)/%.o, $(HIGHER_SRC))
AUX_OBJ = $(patsubst $(SRCDIR)/modules/aux-level/%.c, $(OBJDIR)/%.o, $(AUX_SRC))

# Main executables
MAIN_AUX_EXE = $(patsubst $(SRCDIR)/main/aux-level/%.c, $(BINDIR)/%, $(MAIN_AUX_SRC))
MAIN_LOWER_EXE = $(patsubst $(SRCDIR)/main/lower-level/%.c, $(BINDIR)/%, $(MAIN_LOWER_SRC))
MAIN_HIGHER_EXE = $(patsubst $(SRCDIR)/main/higher-level/%.c, $(BINDIR)/%, $(MAIN_HIGHER_SRC))

# Test executables
TEST_EXE = $(patsubst $(SRCDIR)/tests/%.c, $(BINDIR)/force-test/%, $(TEST_SRC))

# Dependencies
DEP = $(CROSS_OBJ:.o=.d) $(LOWER_OBJ:.o=.d) $(HIGHER_OBJ:.o=.d) $(AUX_OBJ:.o=.d)

# Targets
all: exe tests
exe: $(MAIN_AUX_EXE) $(MAIN_LOWER_EXE) $(MAIN_HIGHER_EXE)
tests: $(TEST_EXE)
dev: $(BINDIR)/force-stratified-sample # specific target for development
#tests: test_utils-cl test_alloc-cl
#exe: force-parameter force-qai-inflate force-tile-finder force-tabulate-grid force-l2ps force-higher-level force-train force-lut-modis force-mdcp force-stack force-import-modis force-cube-init force-hist force-stratified-sample
#all: temp cross lower higher aux exe unit-tests
#.PHONY: temp all install install_ bash python rstats misc external clean build check

# Include dependencies
-include $(DEP)


print-vars:
	@echo "main source files: $(TEST_SRC)"
	@echo "main program files: $(TEST_EXE)"
	@echo "Object files: $(CROSS_OBJ)"
	@echo "Compiler flags: $(CFLAGS)"

##########################################################################

# Modules
$(OBJDIR)/%.o: $(SRCDIR)/modules/cross-level/%.c
	@echo "Compiling $<..."
	$(CXX) $(CFLAGS) $(INCLUDES) $(FLAGS) -c $< -o $@ $(LIBS)

$(OBJDIR)/%.o: $(SRCDIR)/modules/lower-level/%.c
	@echo "Compiling $<..."
	$(CXX) $(CFLAGS) $(INCLUDES) $(FLAGS) -c $< -o $@ $(LIBS)

$(OBJDIR)/%.o: $(SRCDIR)/modules/higher-level/%.c
	@echo "Compiling $<..."
	$(CXX) $(CFLAGS) $(INCLUDES) $(FLAGS) -c $< -o $@ $(LIBS)

$(OBJDIR)/%.o: $(SRCDIR)/modules/aux-level/%.c
	@echo "Compiling $<..."
	$(CXX) $(CFLAGS) $(INCLUDES) $(FLAGS) -c $< -o $@ $(LIBS)

##########################################################################


### DEPENDENCIES

#EXECUTABLES = gcc g++ \
#              parallel \
#              gdalinfo gdal_translate gdaladdo gdalwarp gdalbuildvrt \
#              gdal_merge.py gdal_rasterize gdaltransform gdalsrsinfo \
#              gdal_edit.py gdal_calc.py gdal-config \
#              ogrinfo ogr2ogr \
#              gsl-config curl-config \
#              unzip tar lockfile-create lockfile-remove rename dos2unix \
#              python3 pip3 \
#		       R \
#			  landsatlinks \
#              opencv_version 

#OK := $(foreach exec,$(EXECUTABLES),\
#        $(if $(shell which $(exec)),OK,$(error "No $(exec) in PATH, install dependencies!")))


### EXECUTABLES AND MISC FILES TO BE CHECKED

#FORCE_EXE = force-info force-cube force-higher-level force-import-modis \
#            force-l2ps force-level1-csd force-level1-landsat \
#            force-level2 force-lut-modis \
#            force-magic-parameters force-mdcp force-mosaic force-parameter \
#            force-procmask force-pyramid force-qai-inflate force-stack \
#            force-synthmix force-tabulate-grid force-tile-extent \
#            force-tile-finder force-train force-level2-report force-cube-init \
#			force-init force-datacube-size force-hist force-stratified-sample \
#			force-unit-testing

#FORCE_MISC = force-version.txt force-level2-report.Rmd force-bash-library.sh \
#			force-rstats-library.r




### DIRECTORIES

#DB=bash
#DP=python
#DR=rstats
#DD=misc
#DM=force-misc
#DT=force-test
#DC=src/cross-level
#DL=src/lower-level
#DH=src/higher-level
#DA=src/aux-level
#DU=src/unit-testing
#TB=temp-bin
#TM=$(TB)/$(DM)
#TC=temp-cross
#TL=temp-lower
#TH=temp-higher
#TA=temp-aux
#TU=$(TB)/$(DT)





##########################################################################

# Main executables

$(BINDIR)/%: $(SRCDIR)/main/aux-level/%.c $(CROSS_OBJ) $(AUX_OBJ)
	@echo "Compiling $<..."
	$(CXX) $(CFLAGS) $(INCLUDES) $(FLAGS) -o $@ $^ $(LIBS)

$(BINDIR)/%: $(SRCDIR)/main/lower-level/%.c $(CROSS_OBJ) $(LOWER_OBJ)
	@echo "Compiling $<..."
	$(CXX) $(CFLAGS) $(INCLUDES) $(FLAGS) -o $@ $^ $(LIBS)

#$(BINDIR)/%: $(SRCDIR)/main/higher-level/%.c $(CROSS_OBJ) $(HIGHER_OBJ)
$(BINDIR)/%: $(MAIN_HIGHER_EXE) $(CROSS_OBJ) $(HIGHER_OBJ)
	@echo "Compiling $<..."
	$(CXX) $(CFLAGS) $(INCLUDES) $(FLAGS) -o $@ $^ $(LIBS)

# Test executables

$(BINDIR)/force-test/%: $(SRCDIR)/tests/%.c $(SRCDIR)/tests/unity/unity.c $(CROSS_OBJ)
	@echo "Compiling $<..."
	$(CXX) $(CFLAGS) $(INCLUDES) $(FLAGS) -o $@ $^ $(LIBS)


### dummy code for testing stuff  

#dummy: temp cross aux higher src/dummy.c
#	$(G11) $(CFLAGS) $(GDAL) $(GSL) $(CURL) $(SPLITS) $(OPENCV) $(PYTHON) $(PYTHON2) $(RSTATS) -o $(TB)/dummy src/dummy.c $(TC)/*.o $(TA)/*.o $(TH)/*.o $(LDGDAL) $(LDGSL) $(LDCURL) $(LDSPLITS) $(LDOPENCV) $(LDPYTHON) $(LDRSTATS)


### MISC

#install_:
#	chmod 0755 $(TB)/*
#	chmod 0755 $(TU)/*
#	cp -a $(TB)/. $(BINDIR)

clean:
	rm -rf $(OBJDIR) $(BINDIR)

#check:
#	$(foreach exec,$(FORCE_EXE),\
#      $(if $(shell which $(BINDIR)/$(exec)), \
#	    $(info $(exec) installed), \
#		$(error $(exec) was not installed properly!)))
#	$(foreach miscfiles,$(FORCE_MISC),\
#      $(if $(shell ls $(BINDIR)/$(DM)/$(miscfiles) 2> /dev/null), \
#	    $(info $(miscfiles) installed), \
#		$(error $(miscfiles) was not installed properly!)))

#misc: temp
#	$(foreach miscfiles,$(FORCE_MISC),\
#	  $(shell cp $(DD)/$(miscfiles) -t $(TM)))

#bash: temp
#	cp $(DB)/force-info.sh $(TB)/force-info
#	cp $(DB)/force-cube.sh $(TB)/force-cube
#	cp $(DB)/force-level1-csd.sh $(TB)/force-level1-csd  
#	cp $(DB)/force-level2.sh $(TB)/force-level2
#	cp $(DB)/force-mosaic.sh $(TB)/force-mosaic
#	cp $(DB)/force-pyramid.sh $(TB)/force-pyramid
#	cp $(DB)/force-procmask.sh $(TB)/force-procmask
#	cp $(DB)/force-tile-extent.sh $(TB)/force-tile-extent
#	cp $(DB)/force-magic-parameters.sh $(TB)/force-magic-parameters
#	cp $(DB)/force-level2-report.sh $(TB)/force-level2-report
#	cp $(DB)/force-init.sh $(TB)/force-init
#	cp $(DB)/force-datacube-size.sh $(TB)/force-datacube-size
#	cp $(DB)/force-unit-testing.sh $(TB)/force-unit-testing

#external: temp
#	cp $(shell which landsatlinks) $(TB)/force-level1-landsat

#python: temp
#	cp $(DP)/force-synthmix.py $(TB)/force-synthmix

#rstats: temp
#	cp $(DR)/force-sample-size.r $(TB)/force-sample-size
#	cp $(DR)/force-map-accuracy.r $(TB)/force-map-accuracy

#install: bash python rstats misc external install_ clean check

#build:
#	$(eval V := $(shell grep '#define _VERSION_' src/cross-level/const-cl.h | cut -d '"' -f 2 | sed 's/ /_/g'))
#	$(eval T :=$(shell date +"%Y%m%d%H%M%S"))
#	tar -czf force_v$(V)_$(T).tar.gz src bash python rstats misc external images docs Makefile LICENSE README.md

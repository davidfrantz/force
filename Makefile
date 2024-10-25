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
INSTALLDIR=/usr/local/bin

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
BASHDIR = bash
RSTATSDIR = rstats
PYTHONDIR = python
MISCDIR = misc

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
DEPENDENCIES = $(CROSS_OBJ:.o=.d) $(LOWER_OBJ:.o=.d) $(HIGHER_OBJ:.o=.d) $(AUX_OBJ:.o=.d)

# Targets
all: check-required exe tests scripts misc
exe: aux higher lower
aux: $(MAIN_AUX_EXE)
higher: $(MAIN_HIGHER_EXE)
lower: $(MAIN_LOWER_EXE)
tests: $(TEST_EXE)
scripts: bash rstats python external
dev: $(BINDIR)/force-stratified-sample # specific target for development
.PHONY: check-required bash rstats python external scripts misc install

# Include dependencies
-include $(DEPENDENCIES)


##########################################################################

### Check external tools

REQUIRE = gcc g++ \
          parallel \
          gdalinfo gdal_translate gdaladdo gdalwarp gdalbuildvrt \
          gdal_merge.py gdal_rasterize gdaltransform gdalsrsinfo \
          gdal_edit.py gdal_calc.py gdal-config \
          ogrinfo ogr2ogr \
          gsl-config curl-config \
          unzip tar lockfile-create lockfile-remove rename dos2unix \
          python3 pip3 \
          R \
          landsatlinks \
          opencv_version 

check-required:
	@missing=$$(for exec in $(REQUIRE); do \
		if ! command -v $$exec > /dev/null; then \
			echo $$exec; \
		fi; \
	done); \
	if [ "$$missing" ]; then \
		echo "No executable(s) in PATH: $$missing. Please install the dependencies!"; \
		exit 1; \
	fi


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

# Main executables

$(BINDIR)/%: $(SRCDIR)/main/aux-level/%.c $(CROSS_OBJ) $(AUX_OBJ)
	@echo "Compiling $<..."
	$(CXX) $(CFLAGS) $(INCLUDES) $(FLAGS) -o $@ $^ $(LIBS)

$(BINDIR)/%: $(SRCDIR)/main/lower-level/%.c $(CROSS_OBJ) $(LOWER_OBJ)
	@echo "Compiling $<..."
	$(CXX) $(CFLAGS) $(INCLUDES) $(FLAGS) -o $@ $^ $(LIBS)

$(BINDIR)/%: $(SRCDIR)/main/higher-level/%.c $(CROSS_OBJ) $(HIGHER_OBJ)
	@echo "Compiling $<..."
	$(CXX) $(CFLAGS) $(INCLUDES) $(FLAGS) -o $@ $^ $(LIBS)


# Test executables

$(BINDIR)/force-test/%: $(SRCDIR)/tests/%.c $(SRCDIR)/tests/unity/unity.c $(CROSS_OBJ)
	@echo "Compiling $<..."
	$(CXX) $(CFLAGS) $(INCLUDES) $(FLAGS) -o $@ $^ $(LIBS)


##########################################################################

# Bash scripts

bash:
	@for file in $(BASHDIR)/*.sh; do \
		cp $$file $(BINDIR)/$$(basename $$file .sh); \
	done


# R scripts

rstats:
	@for file in $(RSTATSDIR)/*.r; do \
		cp $$file $(BINDIR)/$$(basename $$file .r); \
	done


# Python scripts

python:
	@for file in $(PYTHONDIR)/*.py; do \
		cp $$file $(BINDIR)/$$(basename $$file .py); \
	done


# re-branded tools [with permission]

external:
	cp $(shell which landsatlinks) $(BINDIR)/force-level1-landsat


# misc files

misc:
	@for file in $(MISCDIR)/*; do \
		cp $$file $(BINDIR)/force-misc/; \
	done


### dummy code for testing stuff  

#dummy: temp cross aux higher src/dummy.c
#	$(G11) $(CFLAGS) $(GDAL) $(GSL) $(CURL) $(OPENCV) $(PYTHON) $(PYTHON2) $(RSTATS) -o $(TB)/dummy src/dummy.c $(TC)/*.o $(TA)/*.o $(TH)/*.o $(LDGDAL) $(LDGSL) $(LDCURL) $(LDOPENCV) $(LDPYTHON) $(LDRSTATS)


##########################################################################

# install the software

install: all
	find $(BINDIR) -type f -exec chmod 0755 {} +
	find $(BINDIR) -type d -exec chmod 0755 {} +
	cp -a $(BINDIR)/. $(INSTALLDIR)


# clean up

clean:
	rm -rf $(OBJDIR) $(BINDIR)


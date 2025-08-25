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


##########################################################################
# NO! changes below this line (unless you know what to do, then go ahead)
##########################################################################

### Compiler

# Compilation Flags
CFLAGS=-O3 -Wall -fopenmp
#CFLAGS=-g -Wall -fopenmp

GCC=gcc $(CFLAGS)
G11=g++ -std=c++11 $(CFLAGS)


##########################################################################

# Directories
SRCDIR = src
OBJDIR = obj
BINDIR = bin
BUILDDIR = build
BASHDIR = bash
RSTATSDIR = rstats
PYTHONDIR = python
MISCDIR = misc


# Targets
all: check-tools prepare exe bash rstats python external misc
with_tests: all tests
.PHONY: prepare check-tools bash rstats python external misc \
  install clean all with_tests exe tests auch higher lower cross

# Compile targets
include $(BUILDDIR)/cross-level.mk
include $(BUILDDIR)/lower-level.mk
include $(BUILDDIR)/higher-level.mk
include $(BUILDDIR)/aux-level.mk
include $(BUILDDIR)/executables.mk
include $(BUILDDIR)/tests.mk

# Prepare build directories
prepare: check-tools
	mkdir -p \
	$(OBJDIR) \
	$(BINDIR) \
	$(BINDIR)/force-misc

# Check installed tools
check-tools:
	./check-required.sh

# Bash scripts
bash: prepare
	@for file in $(BASHDIR)/*.sh; do \
		cp $$file $(BINDIR)/$$(basename $$file .sh); \
	done

# R scripts
rstats: prepare
	@for file in $(RSTATSDIR)/*.r; do \
		cp $$file $(BINDIR)/$$(basename $$file .r); \
	done

# Python scripts
python: prepare
	@for file in $(PYTHONDIR)/*.py; do \
		cp $$file $(BINDIR)/$$(basename $$file .py); \
	done

# re-branded tools [with permission]
external: prepare
	cp $(shell which landsatlinks) $(BINDIR)/force-level1-landsat

# misc files
misc: prepare
	@for file in $(MISCDIR)/*; do \
		cp $$file $(BINDIR)/force-misc/; \
	done

# install the software
install: all
	find $(BINDIR) -type f -exec chmod 0755 {} +
	find $(BINDIR) -type d -exec chmod 0755 {} +
	cp -a $(BINDIR)/. $(INSTALLDIR)


# clean up
clean:
	rm -rf $(OBJDIR) $(BINDIR)


##########################################################################
# 
# This file is part of FORCE - Framework for Operational Radiometric 
# Correction for Environmental monitoring.
# 
# Copyright (C) 2013-2022 David Frantz
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

# Copyright (C) 2020-2022 Gergely Padányi-Gulyás (github user fegyi001),
#                         David Frantz
#                         Fabian Lehmann


# base installation to speed up build process
# https://github.com/davidfrantz/base_image
FROM davidfrantz/base:latest as force_builder

# Environment variables
ENV SOURCE_DIR $HOME/src/force
ENV INSTALL_DIR $HOME/bin

# build args
ARG debug=disable

# Copy src to SOURCE_DIR
RUN mkdir -p $SOURCE_DIR
WORKDIR $SOURCE_DIR
COPY --chown=docker:docker . .

# Build, install, check FORCE
RUN echo "building FORCE" && \
  ./splits.sh enable && \
  ./debug.sh $debug && \
  sed -i "/^BINDIR=/cBINDIR=$INSTALL_DIR/" Makefile && \
  make -j && \
  make install && \
  make clean && \
  cd $HOME && \
  rm -rf $SOURCE_DIR && \
  force-info && \
# clone FORCE UDF
  git clone https://github.com/davidfrantz/force-udf.git

FROM davidfrantz/base:latest as force

COPY --chown=docker:docker --from=force_builder $HOME/bin $HOME/bin
COPY --chown=docker:docker --from=force_builder $HOME/force-udf $HOME/udf

WORKDIR /home/docker

ENV R_HOME=/usr/lib/R
ENV LD_LIBRARY_PATH=$R_HOME/lib

CMD ["force-info"]

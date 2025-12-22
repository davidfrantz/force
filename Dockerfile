# syntax=docker/dockerfile:1
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

# Copyright (C) 2020-2025 Gergely Padányi-Gulyás (github user fegyi001),
#                         David Frantz
#                         Fabian Lehmann


# base installation to speed up build process
# https://github.com/davidfrantz/base_image
FROM davidfrantz/base:latest AS force_builder

# Environment variables
ENV SOURCE_DIR=$HOME/src/force
ENV INSTALL_DIR=$HOME/bin

# build args
ARG debug=disable
ARG build=all

# Copy src to SOURCE_DIR
RUN mkdir -p $SOURCE_DIR
WORKDIR $SOURCE_DIR
COPY --link --chown=1000:1000 . .

# Build, install, check FORCE
RUN echo "building FORCE" && \
  ./debug.sh $debug && \
  make -j$(nproc) $build && \
  make INSTALLDIR=$INSTALL_DIR install && \
  make clean && \
  cd $HOME && \
  rm -rf $SOURCE_DIR && \
  force-info

FROM davidfrantz/base:latest AS force

# Use numerical UID/GID to avoid name resolving issues with the non-default
# Docker builders.
ADD --link --chown=1000:1000 --exclude=.github https://github.com/davidfrantz/force-udf.git $HOME/udf
COPY --link --chown=1000:1000 --from=force_builder $HOME/bin $HOME/bin

WORKDIR /home/ubuntu

ENV R_HOME=/usr/lib/R
ENV LD_LIBRARY_PATH=$R_HOME/lib

CMD ["force-info"]

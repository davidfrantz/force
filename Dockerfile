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
  make -j$(nproc) $build

FROM davidfrantz/base:latest AS force

ADD --link --chown=root:root --exclude=.github https://github.com/davidfrantz/force-udf.git /usr/local/bin/force/force-udf
COPY --link --chown=root:root --from=force_builder $HOME/src/force/bin /usr/local/bin/force

ENV PATH="$PATH:/usr/local/bin/force"

ENV R_HOME=/usr/lib/R
ENV LD_LIBRARY_PATH=$R_HOME/lib

CMD ["force-info"]

FROM ubuntu:18.04 as builder

# Install folder
ENV INSTALL_DIR /opt/install/src
RUN mkdir -p $INSTALL_DIR

# Refresh package list & upgrade existing packages 
RUN apt-get -y update && apt-get -y upgrade  

# Add PPA for Python 3.x
RUN apt -y install software-properties-common
RUN add-apt-repository -y ppa:deadsnakes/ppa

# Install libraries
RUN apt-get -y install wget unzip curl git build-essential libgdal-dev gdal-bin python-gdal \ 
  libarmadillo-dev libfltk1.3-dev libgsl0-dev lockfile-progs rename \
  parallel libfltk1.3-dev apt-utils cmake \
  libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev \
  python3.8 python3-pip

# Set python aliases for Python 3.x
RUN echo 'alias python=python3' >> ~/.bashrc \
  && echo 'alias pip=pip3' >> ~/.bashrc
# NumPy is needed for OpenCV
RUN pip3 install numpy==1.18.1 

# Build OpenCV from source
RUN mkdir -p $INSTALL_DIR/opencv
WORKDIR $INSTALL_DIR/opencv
RUN wget https://github.com/opencv/opencv/archive/4.1.0.zip \
  && unzip 4.1.0.zip
WORKDIR $INSTALL_DIR/opencv/opencv-4.1.0
RUN mkdir -p $INSTALL_DIR/opencv/opencv-4.1.0/build
WORKDIR $INSTALL_DIR/opencv/opencv-4.1.0/build
RUN cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local .. \
  && make -j7 \
  && make install \
  && make clean

# Build SPLITS from source
RUN mkdir -p $INSTALL_DIR/splits
WORKDIR $INSTALL_DIR/splits
RUN wget http://sebastian-mader.net/wp-content/uploads/2017/11/splits-1.9.tar.gz
RUN tar -xzf splits-1.9.tar.gz
WORKDIR $INSTALL_DIR/splits/splits-1.9
RUN ./configure CPPFLAGS="-I /usr/include/gdal" CXXFLAGS=-fpermissive \
  && make \
  && make install \
  && make clean

# Build FORCE from source
RUN mkdir -p $INSTALL_DIR/force
WORKDIR $INSTALL_DIR/force
COPY . . 
# Conditionally enable SPLITS which is disabled by default
ARG splits=false 
RUN if [ "$splits" = "true" ] ; then ./splits.sh enable; else ./splits.sh disable; fi
RUN make -j7 \
  && make install \
  && make clean

# Cleanup after successfull builds
RUN rm -rf $INSTALL_DIR
RUN apt-get purge -y --auto-remove apt-utils cmake git build-essential software-properties-common

# Test FORCE run
RUN force
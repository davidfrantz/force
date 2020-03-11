FROM ubuntu:18.04 as builder

# Install folder
ENV INSTALL_DIR /opt/install/src
RUN mkdir -p $INSTALL_DIR

# Install libraries
RUN apt-get -y update && apt-get -y upgrade  
RUN apt-get -y install wget unzip curl git build-essential libgdal-dev gdal-bin python-gdal \ 
  libarmadillo-dev libfltk1.3-dev libgsl0-dev lockfile-progs rename \
  parallel libfltk1.3-dev apt-utils cmake \
  libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev 

# Install OpenCV
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

# Install SPLITS
RUN mkdir -p $INSTALL_DIR/splits
WORKDIR $INSTALL_DIR/splits
RUN wget http://sebastian-mader.net/wp-content/uploads/2017/11/splits-1.9.tar.gz
RUN tar -xzf splits-1.9.tar.gz
WORKDIR $INSTALL_DIR/splits/splits-1.9
RUN ./configure CPPFLAGS="-I /usr/include/gdal" CXXFLAGS=-fpermissive \
  && make \
  && make install \
  && make clean

# Install FORCE
WORKDIR $INSTALL_DIR
RUN git clone https://github.com/davidfrantz/force.git --branch v3.0  --single-branch
WORKDIR $INSTALL_DIR/force
COPY ./const-cl.h ./src/cross-level/
COPY ./Makefile .
RUN make -j7 \
  && make install \
  && make clean

# Copy SCIHIB test credentials
COPY ./.scihub /root/
WORKDIR /root/
RUN chmod 400 ./.scihub

# Cleanup
RUN rm -rf $INSTALL_DIR
RUN apt-get purge -y --auto-remove apt-utils cmake git build-essential 

# Test FORCE run
RUN force
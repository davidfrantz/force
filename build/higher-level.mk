### HIGHER LEVEL COMPILE UNITS

HIGHER_DIR=$(SRCDIR)/modules/higher-level

higher: \
    bap-hl \
    cf-improphe-hl \
    cso-hl \
    fold-hl \
    improphe-hl \
    index-compute-hl \
    index-parse-hl \
    interpolate-hl \
    l2-improphe-hl \
    level3-hl \
    lib-hl \
    lsm-hl \
    ml-hl \
    param-hl \
    polar-hl \
    progress-hl \
    py-udf-hl \
    quality-hl \
    r-udf-hl \
    read-ard-hl \
    read-aux-hl \
    sample-hl \
	sensor-hl \
    spec-adjust-hl \
    standardize-hl \
    stm-hl \
    tasks-hl \
    texture-hl \
    trend-hl \
    tsa-hl \
    udf-hl

bap-hl: prepare $(HIGHER_DIR)/bap-hl.c
	$(GCC) -c $(HIGHER_DIR)/bap-hl.c -o $(OBJDIR)/bap-hl.o

cf-improphe-hl: prepare $(HIGHER_DIR)/cf-improphe-hl.c
	$(GCC) -c $(HIGHER_DIR)/cf-improphe-hl.c -o $(OBJDIR)/cf-improphe-hl.o

cso-hl: prepare $(HIGHER_DIR)/cso-hl.c
	$(GCC) -c $(HIGHER_DIR)/cso-hl.c -o $(OBJDIR)/cso-hl.o

fold-hl: prepare $(HIGHER_DIR)/fold-hl.c
	$(GCC) -c $(HIGHER_DIR)/fold-hl.c -o $(OBJDIR)/fold-hl.o

improphe-hl: prepare $(HIGHER_DIR)/improphe-hl.c
	$(GCC) -c $(HIGHER_DIR)/improphe-hl.c -o $(OBJDIR)/improphe-hl.o

index-compute-hl: prepare $(HIGHER_DIR)/index-compute-hl.c
	$(GCC) $(GSL) -c $(HIGHER_DIR)/index-compute-hl.c -o $(OBJDIR)/index-compute-hl.o $(LDGSL)

index-parse-hl: prepare $(HIGHER_DIR)/index-parse-hl.c
	$(GCC) $(GSL) -c $(HIGHER_DIR)/index-parse-hl.c -o $(OBJDIR)/index-parse-hl.o $(LDGSL)

interpolate-hl: prepare $(HIGHER_DIR)/interpolate-hl.c
	$(GCC) -c $(HIGHER_DIR)/interpolate-hl.c -o $(OBJDIR)/interpolate-hl.o

l2-improphe-hl: prepare $(HIGHER_DIR)/l2-improphe-hl.c
	$(GCC) -c $(HIGHER_DIR)/l2-improphe-hl.c -o $(OBJDIR)/l2-improphe-hl.o

level3-hl: prepare $(HIGHER_DIR)/level3-hl.c
	$(GCC) -c $(HIGHER_DIR)/level3-hl.c -o $(OBJDIR)/level3-hl.o

lib-hl: prepare $(HIGHER_DIR)/lib-hl.c
	$(GCC) -c $(HIGHER_DIR)/lib-hl.c -o $(OBJDIR)/lib-hl.o

lsm-hl: prepare $(HIGHER_DIR)/lsm-hl.c
	$(GCC) -c $(HIGHER_DIR)/lsm-hl.c -o $(OBJDIR)/lsm-hl.o

ml-hl: prepare $(HIGHER_DIR)/ml-hl.c
	$(G11) $(OPENCV_INCLUDES) $(OPENCV_FLAGS) -c $(HIGHER_DIR)/ml-hl.c -o $(OBJDIR)/ml-hl.o $(OPENCV_LIBS)

param-hl: prepare $(HIGHER_DIR)/param-hl.c
	$(GCC) -c $(HIGHER_DIR)/param-hl.c -o $(OBJDIR)/param-hl.o

polar-hl: prepare $(HIGHER_DIR)/polar-hl.c
	$(GCC) -c $(HIGHER_DIR)/polar-hl.c -o $(OBJDIR)/polar-hl.o

progress-hl: prepare $(HIGHER_DIR)/progress-hl.c
	$(GCC) -c $(HIGHER_DIR)/progress-hl.c -o $(OBJDIR)/progress-hl.o

py-udf-hl: prepare $(HIGHER_DIR)/py-udf-hl.c
	$(G11) $(PYTHON_INCLUDES) -c $(HIGHER_DIR)/py-udf-hl.c -o $(OBJDIR)/py-udf-hl.o $(PYTHON_LIBS)

quality-hl: prepare $(HIGHER_DIR)/quality-hl.c
	$(GCC) -c $(HIGHER_DIR)/quality-hl.c -o $(OBJDIR)/quality-hl.o

r-udf-hl: prepare $(HIGHER_DIR)/r-udf-hl.c
	$(GCC) $(RSTATS_INCLUDES) -c $(HIGHER_DIR)/r-udf-hl.c -o $(OBJDIR)/r-udf-hl.o $(RSTATS_LIBS)

read-ard-hl: prepare $(HIGHER_DIR)/read-ard-hl.c
	$(GCC) $(GDAL_INCLUDES) $(GDAL_FLAGS) -c $(HIGHER_DIR)/read-ard-hl.c -o $(OBJDIR)/read-ard-hl.o $(GDAL_LIBS)

read-aux-hl: prepare $(HIGHER_DIR)/read-aux-hl.cpp
	$(G11) $(OPENCV_INCLUDES) $(OPENCV_FLAGS) -c $(HIGHER_DIR)/read-aux-hl.cpp -o $(OBJDIR)/read-aux-hl.o $(OPENCV_LIBS)

sample-hl: prepare $(HIGHER_DIR)/sample-hl.c
	$(GCC) -c $(HIGHER_DIR)/sample-hl.c -o $(OBJDIR)/sample-hl.o

sensor-hl: prepare $(HIGHER_DIR)/sensor-hl.c
	$(GCC) -c $(HIGHER_DIR)/sensor-hl.c -o $(OBJDIR)/sensor-hl.o

spec-adjust-hl: prepare $(HIGHER_DIR)/spec-adjust-hl.c
	$(GCC) -c $(HIGHER_DIR)/spec-adjust-hl.c -o $(OBJDIR)/spec-adjust-hl.o

standardize-hl: prepare $(HIGHER_DIR)/standardize-hl.c
	$(GCC) -c $(HIGHER_DIR)/standardize-hl.c -o $(OBJDIR)/standardize-hl.o

stm-hl: prepare $(HIGHER_DIR)/stm-hl.c
	$(GCC) -c $(HIGHER_DIR)/stm-hl.c -o $(OBJDIR)/stm-hl.o

tasks-hl: prepare $(HIGHER_DIR)/tasks-hl.c
	$(G11) $(GDAL_INCLUDES) $(GDAL_FLAGS) $(OPENCV_INCLUDES) $(OPENCV_FLAGS) -c $(HIGHER_DIR)/tasks-hl.c -o $(OBJDIR)/tasks-hl.o $(GDAL_LIBS) $(OPENCV_LIBS)

texture-hl: prepare $(HIGHER_DIR)/texture-hl.c
	$(G11) $(OPENCV_INCLUDES) $(OPENCV_FLAGS) -c $(HIGHER_DIR)/texture-hl.c -o $(OBJDIR)/texture-hl.o $(OPENCV_LIBS)

trend-hl: prepare $(HIGHER_DIR)/trend-hl.c
	$(GCC) -c $(HIGHER_DIR)/trend-hl.c -o $(OBJDIR)/trend-hl.o

tsa-hl: prepare $(HIGHER_DIR)/tsa-hl.c
	$(GCC) -c $(HIGHER_DIR)/tsa-hl.c -o $(OBJDIR)/tsa-hl.o

udf-hl: prepare $(HIGHER_DIR)/udf-hl.c
	$(GCC) -c $(HIGHER_DIR)/udf-hl.c -o $(OBJDIR)/udf-hl.o


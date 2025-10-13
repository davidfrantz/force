## CROSS LEVEL COMPILE UNITS

CROSS_DIR=$(SRCDIR)/modules/cross-level

cross: \
    alloc-cl \
    brick_base-cl \
    brick_io-cl \
    cite-cl \
    cube-cl \
    date-cl \
    datesys-cl \
    dir-cl \
    download-cl \
    enum-cl \
    gdalopt-cl \
    imagefuns-cl \
    konami-cl \
    lock-cl \
    param-cl \
    pca-cl \
    quality-cl \
    queue-cl \
    read-cl \
    stats-cl \
    string-cl \
    sun-cl \
    sys-cl \
    table-cl \
    tile-cl \
    utils-cl \
    vector-cl \
    warp-cl

alloc-cl: prepare $(CROSS_DIR)/alloc-cl.c
	$(GCC) -c $(CROSS_DIR)/alloc-cl.c -o $(OBJDIR)/alloc-cl.o

brick_base-cl: prepare $(CROSS_DIR)/brick_base-cl.c
	$(G11) $(GDAL_INCLUDES) $(GDAL_FLAGS) -c $(CROSS_DIR)/brick_base-cl.c -o $(OBJDIR)/brick_base-cl.o $(GDAL_LIBS)

brick_io-cl: prepare $(CROSS_DIR)/brick_io-cl.c
	$(G11) $(GDAL_INCLUDES) $(GDAL_FLAGS) -c $(CROSS_DIR)/brick_io-cl.c -o $(OBJDIR)/brick_io-cl.o $(GDAL_LIBS)

cite-cl: prepare $(CROSS_DIR)/cite-cl.c
	$(GCC) -c $(CROSS_DIR)/cite-cl.c -o $(OBJDIR)/cite-cl.o

cube-cl: prepare $(CROSS_DIR)/cube-cl.c
	$(GCC) -c $(CROSS_DIR)/cube-cl.c -o $(OBJDIR)/cube-cl.o

date-cl: prepare $(CROSS_DIR)/date-cl.c
	$(GCC) -c $(CROSS_DIR)/date-cl.c -o $(OBJDIR)/date-cl.o

datesys-cl : prepare $(CROSS_DIR)/datesys-cl.c
	$(GCC) -c $(CROSS_DIR)/datesys-cl.c -o $(OBJDIR)/datesys-cl.o

dir-cl: prepare $(CROSS_DIR)/dir-cl.c
	$(GCC) -c $(CROSS_DIR)/dir-cl.c -o $(OBJDIR)/dir-cl.o

download-cl: prepare $(CROSS_DIR)/download-cl.c
	$(GCC) $(CURL_INCLUDES) $(CURL_FLAGS) -c $(CROSS_DIR)/download-cl.c -o $(OBJDIR)/download-cl.o $(CURL_LIBS)

enum-cl: prepare $(CROSS_DIR)/enum-cl.c
	$(GCC) -c $(CROSS_DIR)/enum-cl.c -o $(OBJDIR)/enum-cl.o

gdalopt-cl: prepare $(CROSS_DIR)/gdalopt-cl.c
	$(GCC) -c $(CROSS_DIR)/gdalopt-cl.c -o $(OBJDIR)/gdalopt-cl.o

imagefuns-cl: prepare $(CROSS_DIR)/imagefuns-cl.c
	$(GCC) -c $(CROSS_DIR)/imagefuns-cl.c -o $(OBJDIR)/imagefuns-cl.o

konami-cl: prepare $(CROSS_DIR)/konami-cl.c
	$(GCC) -c $(CROSS_DIR)/konami-cl.c -o $(OBJDIR)/konami-cl.o

lock-cl: prepare $(CROSS_DIR)/lock-cl.c
	$(GCC) $(GDAL_INCLUDES) $(GDAL_FLAGS) -c $(CROSS_DIR)/lock-cl.c -o $(OBJDIR)/lock-cl.o $(GDAL_LIBS)

param-cl: prepare $(CROSS_DIR)/param-cl.c
	$(GCC) -c $(CROSS_DIR)/param-cl.c -o $(OBJDIR)/param-cl.o

pca-cl: prepare $(CROSS_DIR)/pca-cl.c
	$(GCC) $(GSL) -c $(CROSS_DIR)/pca-cl.c -o $(OBJDIR)/pca-cl.o $(LDGSL)

quality-cl: prepare $(CROSS_DIR)/quality-cl.c
	$(GCC) -c $(CROSS_DIR)/quality-cl.c -o $(OBJDIR)/quality-cl.o

queue-cl: prepare $(CROSS_DIR)/queue-cl.c
	$(GCC) -c $(CROSS_DIR)/queue-cl.c -o $(OBJDIR)/queue-cl.o

read-cl: prepare $(CROSS_DIR)/read-cl.c
	$(GCC) -c $(CROSS_DIR)/read-cl.c -o $(OBJDIR)/read-cl.o

stats-cl: prepare $(CROSS_DIR)/stats-cl.c
	$(GCC) -c $(CROSS_DIR)/stats-cl.c -o $(OBJDIR)/stats-cl.o

string-cl: prepare $(CROSS_DIR)/string-cl.c
	$(GCC) -c $(CROSS_DIR)/string-cl.c -o $(OBJDIR)/string-cl.o

sun-cl: prepare $(CROSS_DIR)/sun-cl.c
	$(GCC) -c $(CROSS_DIR)/sun-cl.c -o $(OBJDIR)/sun-cl.o

sys-cl: prepare $(CROSS_DIR)/sys-cl.c
	$(GCC) -c $(CROSS_DIR)/sys-cl.c -o $(OBJDIR)/sys-cl.o

table-cl: prepare $(CROSS_DIR)/table-cl.c
	$(GCC) -c $(CROSS_DIR)/table-cl.c -o $(OBJDIR)/table-cl.o

tile-cl: prepare $(CROSS_DIR)/tile-cl.c
	$(GCC) $(GDAL_INCLUDES) $(GDAL_FLAGS) -c $(CROSS_DIR)/tile-cl.c -o $(OBJDIR)/tile-cl.o $(GDAL_LIBS)

utils-cl: prepare $(CROSS_DIR)/utils-cl.c
	$(GCC) -c $(CROSS_DIR)/utils-cl.c -o $(OBJDIR)/utils-cl.o -lm

vector-cl: prepare $(CROSS_DIR)/vector-cl.c
	$(G11) $(GDAL_INCLUDES) $(GDAL_FLAGS) -c $(CROSS_DIR)/vector-cl.c -o $(OBJDIR)/vector-cl.o $(GDAL_LIBS)

warp-cl: prepare $(CROSS_DIR)/warp-cl.cpp
	$(G11) $(GDAL_INCLUDES) $(GDAL_FLAGS) -c $(CROSS_DIR)/warp-cl.cpp -o $(OBJDIR)/warp-cl.o $(GDAL_LIBS)


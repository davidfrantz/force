### LOWER LEVEL COMPILE UNITS

LOWER_DIR=$(SRCDIR)/modules/lower-level

lower: \
    aod-ll \
    atc-ll \
    atmo-ll \
    brdf-ll \
    cloud-ll \
    coreg-ll \
    coregfuns-ll \
    cube-ll \
    equi7-ll \
    gas-ll \
    glance7-ll \
    meta-ll \
    modwvp-ll \
    param-ll \
    radtran-ll \
    read-ll \
    resmerge-ll \
    sunview-ll \
    table-ll \
    topo-ll

aod-ll: prepare $(LOWER_DIR)/aod-ll.c
	$(GCC) $(GDAL_INCLUDES) $(GDAL_FLAGS) -c $(LOWER_DIR)/aod-ll.c -o $(OBJDIR)/aod-ll.o $(GDAL_LIBS)

atc-ll: prepare $(LOWER_DIR)/atc-ll.c
	$(GCC) -c $(LOWER_DIR)/atc-ll.c -o $(OBJDIR)/atc-ll.o

atmo-ll: prepare $(LOWER_DIR)/atmo-ll.c
	$(GCC) $(GDAL_INCLUDES) $(GDAL_FLAGS) -c $(LOWER_DIR)/atmo-ll.c -o $(OBJDIR)/atmo-ll.o $(GDAL_LIBS)

brdf-ll: prepare $(LOWER_DIR)/brdf-ll.c
	$(GCC) -c $(LOWER_DIR)/brdf-ll.c -o $(OBJDIR)/brdf-ll.o

cloud-ll: prepare $(LOWER_DIR)/cloud-ll.c
	$(GCC) -c $(LOWER_DIR)/cloud-ll.c -o $(OBJDIR)/cloud-ll.o

coreg-ll: prepare $(LOWER_DIR)/coreg-ll.c
	$(GCC) -c $(LOWER_DIR)/coreg-ll.c -o $(OBJDIR)/coreg-ll.o

coregfuns-ll: prepare $(LOWER_DIR)/coregfuns-ll.c
	$(GCC) -c $(LOWER_DIR)/coregfuns-ll.c -o $(OBJDIR)/coregfuns-ll.o

cube-ll: prepare $(LOWER_DIR)/cube-ll.c
	$(GCC) -c $(LOWER_DIR)/cube-ll.c -o $(OBJDIR)/cube-ll.o

equi7-ll: prepare $(LOWER_DIR)/equi7-ll.c
	$(GCC) -c $(LOWER_DIR)/equi7-ll.c -o $(OBJDIR)/equi7-ll.o

gas-ll: prepare $(LOWER_DIR)/gas-ll.c
	$(GCC) -c $(LOWER_DIR)/gas-ll.c -o $(OBJDIR)/gas-ll.o

glance7-ll: prepare $(LOWER_DIR)/glance7-ll.c
	$(GCC) -c $(LOWER_DIR)/glance7-ll.c -o $(OBJDIR)/glance7-ll.o

meta-ll: prepare $(LOWER_DIR)/meta-ll.c
	$(GCC) $(GDAL_INCLUDES) $(GDAL_FLAGS) -c $(LOWER_DIR)/meta-ll.c -o $(OBJDIR)/meta-ll.o $(GDAL_LIBS)

modwvp-ll: prepare $(LOWER_DIR)/modwvp-ll.c
	$(GCC) $(GDAL_INCLUDES) $(GDAL_FLAGS) -c $(LOWER_DIR)/modwvp-ll.c -o $(OBJDIR)/modwvp-ll.o $(GDAL_LIBS)

param-ll: prepare $(LOWER_DIR)/param-ll.c
	$(GCC) -c $(LOWER_DIR)/param-ll.c -o $(OBJDIR)/param-ll.o

radtran-ll: prepare $(LOWER_DIR)/radtran-ll.c
	$(GCC) -c $(LOWER_DIR)/radtran-ll.c -o $(OBJDIR)/radtran-ll.o

read-ll: prepare $(LOWER_DIR)/read-ll.c
	$(GCC) $(GDAL_INCLUDES) $(GDAL_FLAGS) -c $(LOWER_DIR)/read-ll.c -o $(OBJDIR)/read-ll.o $(GDAL_LIBS)

resmerge-ll: prepare $(LOWER_DIR)/resmerge-ll.c
	$(GCC) -c $(LOWER_DIR)/resmerge-ll.c -o $(OBJDIR)/resmerge-ll.o

sunview-ll: prepare $(LOWER_DIR)/sunview-ll.c
	$(GCC) -c $(LOWER_DIR)/sunview-ll.c -o $(OBJDIR)/sunview-ll.o

table-ll: prepare $(LOWER_DIR)/table-ll.c
	$(GCC) -c $(LOWER_DIR)/table-ll.c -o $(OBJDIR)/table-ll.o

topo-ll: prepare $(LOWER_DIR)/topo-ll.c
	$(GCC) -c $(LOWER_DIR)/topo-ll.c -o $(OBJDIR)/topo-ll.o


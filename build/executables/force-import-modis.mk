force-import-modis: \
  alloc-cl \
  brick_base-cl \
  brick_io-cl \
  cube-cl \
  date-cl \
  datesys-cl \
  dir-cl \
  gdalopt-cl \
  konami-cl \
  lock-cl \
  quality-cl \
  read-cl \
  string-cl \
  sys-cl \
  utils-cl \
  warp-cl \
  $(EXE_LOWER_DIR)/force-import-modis.c
	$(G11) -o $(BINDIR)/force-import-modis $(EXE_LOWER_DIR)/force-import-modis.c \
	$(OBJDIR)/alloc-cl.o \
	$(OBJDIR)/brick_base-cl.o \
	$(OBJDIR)/brick_io-cl.o \
	$(OBJDIR)/cube-cl.o \
	$(OBJDIR)/date-cl.o \
	$(OBJDIR)/datesys-cl.o \
	$(OBJDIR)/dir-cl.o \
	$(OBJDIR)/gdalopt-cl.o \
	$(OBJDIR)/konami-cl.o \
	$(OBJDIR)/lock-cl.o \
	$(OBJDIR)/quality-cl.o \
	$(OBJDIR)/read-cl.o \
	$(OBJDIR)/string-cl.o \
	$(OBJDIR)/sys-cl.o \
	$(OBJDIR)/utils-cl.o \
	$(OBJDIR)/warp-cl.o \
	$(GDAL_INCLUDES) $(GDAL_FLAGS) $(GDAL_LIBS)

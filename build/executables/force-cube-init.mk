force-cube-init: \
  alloc-cl \
  brick_base-cl \
  brick_io-cl \
  cite-cl \
  cube-cl \
  cube-ll \
  date-cl \
  datesys-cl \
  dir-cl \
  equi7-ll \
  gdalopt-cl \
  glance7-ll \
  konami-cl \
  lock-cl \
  quality-cl \
  read-cl \
  string-cl \
  sys-cl \
  tile-cl \
  utils-cl \
  warp-cl \
  $(EXE_LOWER_DIR)/force-cube-init.c
	$(G11) -o $(BINDIR)/force-cube-init $(EXE_LOWER_DIR)/force-cube-init.c \
	$(OBJDIR)/alloc-cl.o \
	$(OBJDIR)/brick_base-cl.o \
	$(OBJDIR)/brick_io-cl.o \
	$(OBJDIR)/cite-cl.o \
	$(OBJDIR)/cube-cl.o \
	$(OBJDIR)/cube-ll.o \
	$(OBJDIR)/date-cl.o \
	$(OBJDIR)/datesys-cl.o \
	$(OBJDIR)/dir-cl.o \
	$(OBJDIR)/equi7-ll.o \
	$(OBJDIR)/gdalopt-cl.o \
	$(OBJDIR)/glance7-ll.o \
	$(OBJDIR)/konami-cl.o \
	$(OBJDIR)/lock-cl.o \
	$(OBJDIR)/quality-cl.o \
	$(OBJDIR)/read-cl.o \
	$(OBJDIR)/string-cl.o \
	$(OBJDIR)/sys-cl.o \
	$(OBJDIR)/tile-cl.o \
	$(OBJDIR)/utils-cl.o \
	$(OBJDIR)/warp-cl.o \
	$(GDAL_INCLUDES) $(GDAL_FLAGS) $(GDAL_LIBS) \
	$(GSL_INCLUDES) $(GSL_FLAGS) $(GSL_LIBS) \
	-lm

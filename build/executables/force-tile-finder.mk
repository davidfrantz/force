force-tile-finder: \
  alloc-cl \
  cube-cl \
  dir-cl \
  konami-cl \
  lock-cl \
  read-cl \
  string-cl \
  sys-cl \
  utils-cl \
  warp-cl \
  $(EXE_AUX_DIR)/force-tile-finder.c 
	$(G11) -o $(BINDIR)/force-tile-finder $(EXE_AUX_DIR)/force-tile-finder.c \
	$(OBJDIR)/alloc-cl.o \
	$(OBJDIR)/cube-cl.o \
	$(OBJDIR)/dir-cl.o \
	$(OBJDIR)/konami-cl.o \
	$(OBJDIR)/lock-cl.o \
	$(OBJDIR)/read-cl.o \
	$(OBJDIR)/string-cl.o \
	$(OBJDIR)/sys-cl.o \
	$(OBJDIR)/utils-cl.o \
	$(OBJDIR)/warp-cl.o \
	$(GDAL_INCLUDES) $(GDAL_FLAGS) $(GDAL_LIBS) \
	-lm

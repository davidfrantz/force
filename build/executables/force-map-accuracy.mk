force-map-accuracy: \
  alloc-cl \
  cite-cl \
  dir-cl \
  konami-cl \
  lock-cl \
  stats-cl \
  string-cl \
  sys-cl \
  table-cl \
  utils-cl \
  validation-aux \
  $(EXE_AUX_DIR)/force-map-accuracy.c 
	$(GCC) -o $(BINDIR)/force-map-accuracy $(EXE_AUX_DIR)/force-map-accuracy.c \
	$(OBJDIR)/alloc-cl.o \
	$(OBJDIR)/cite-cl.o \
	$(OBJDIR)/dir-cl.o \
	$(OBJDIR)/konami-cl.o \
	$(OBJDIR)/lock-cl.o \
	$(OBJDIR)/stats-cl.o \
	$(OBJDIR)/string-cl.o \
	$(OBJDIR)/sys-cl.o \
	$(OBJDIR)/table-cl.o \
	$(OBJDIR)/utils-cl.o \
	$(OBJDIR)/validation-aux.o \
	$(GDAL_INCLUDES) $(GDAL_FLAGS) $(GDAL_LIBS) \
	$(GSL_INCLUDES) $(GSL_FLAGS) $(GSL_LIBS) \
	-lm

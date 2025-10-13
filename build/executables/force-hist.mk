force-hist: \
  alloc-cl \
  dir-cl \
  konami-cl \
  stats-cl \
  string-cl \
  sys-cl \
  table-cl \
  utils-cl \
  $(EXE_AUX_DIR)/force-hist.c
	$(GCC) -o $(BINDIR)/force-hist $(EXE_AUX_DIR)/force-hist.c \
	$(OBJDIR)/alloc-cl.o \
	$(OBJDIR)/dir-cl.o \
	$(OBJDIR)/konami-cl.o \
	$(OBJDIR)/stats-cl.o \
	$(OBJDIR)/string-cl.o \
	$(OBJDIR)/sys-cl.o \
	$(OBJDIR)/table-cl.o \
	$(OBJDIR)/utils-cl.o \
	$(GDAL_INCLUDES) $(GDAL_FLAGS) $(GDAL_LIBS) \
	$(GSL_INCLUDES) $(GSL_FLAGS) $(GSL_LIBS) \
	-lm

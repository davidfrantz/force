force-mdcp: \
  alloc-cl \
  dir-cl \
  konami-cl \
  string-cl \
  sys-cl \
  utils-cl \
  $(EXE_AUX_DIR)/force-mdcp.c
	$(GCC) -o $(BINDIR)/force-mdcp $(EXE_AUX_DIR)/force-mdcp.c \
	$(OBJDIR)/alloc-cl.o \
	$(OBJDIR)/dir-cl.o \
	$(OBJDIR)/konami-cl.o \
	$(OBJDIR)/string-cl.o \
	$(OBJDIR)/sys-cl.o \
	$(OBJDIR)/utils-cl.o \
	$(GDAL_INCLUDES) $(GDAL_FLAGS) $(GDAL_LIBS) \
	-lm

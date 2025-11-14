force-lut-modis: \
  alloc-cl \
  date-cl \
  datesys-cl \
  dir-cl \
  download-cl \
  konami-cl \
  modwvp-ll \
  stats-cl \
  string-cl \
  sys-cl \
  table-cl \
  utils-cl \
  $(EXE_LOWER_DIR)/force-lut-modis.c
	$(GCC) -o $(BINDIR)/force-lut-modis $(EXE_LOWER_DIR)/force-lut-modis.c \
	$(OBJDIR)/alloc-cl.o \
	$(OBJDIR)/date-cl.o \
	$(OBJDIR)/datesys-cl.o \
	$(OBJDIR)/dir-cl.o \
	$(OBJDIR)/download-cl.o \
	$(OBJDIR)/konami-cl.o \
	$(OBJDIR)/modwvp-ll.o \
	$(OBJDIR)/stats-cl.o \
	$(OBJDIR)/string-cl.o \
	$(OBJDIR)/sys-cl.o \
	$(OBJDIR)/table-cl.o \
	$(OBJDIR)/utils-cl.o \
	$(GDAL_INCLUDES) $(GDAL_FLAGS) $(GDAL_LIBS) \
	$(GSL_INCLUDES) $(GSL_FLAGS) $(GSL_LIBS) \
	$(CURL_INCLUDES) $(CURL_FLAGS) $(CURL_LIBS) \
	-lm

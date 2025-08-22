dummy: \
  cross \
  lower \
  higher \
  aux \
  $(EXE_DIR)/dummy.c
	$(G11) -o $(BINDIR)/dummy $(EXE_DIR)/dummy.c \
	$(OBJDIR)/*.o \
	$(GDAL_INCLUDES) $(GDAL_FLAGS) $(GDAL_LIBS) \
	$(GSL_INCLUDES) $(GSL_FLAGS) $(GSL_LIBS) \
	$(CURL_INCLUDES) $(CURL_FLAGS) $(CURL_LIBS) \
	$(OPENCV_INCLUDES) $(OPENCV_FLAGS) $(OPENCV_LIBS) \
	$(PYTHON_INCLUDES) $(PYTHON_LIBS) \
	$(RSTATS_INCLUDES) $(RSTATS_LIBS) \
	-lm

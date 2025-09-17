force-train: \
  alloc-cl \
  date-cl \
  dir-cl \
  enum-cl \
  konami-cl \
  param-cl \
  param_train-aux \
  stats-cl \
  string-cl \
  sys-cl \
  read-cl \
  utils-cl \
  train-aux \
  $(EXE_AUX_DIR)/force-train.cpp
	$(G11) -o $(BINDIR)/force-train $(EXE_AUX_DIR)/force-train.cpp \
	$(OBJDIR)/alloc-cl.o \
	$(OBJDIR)/dir-cl.o \
	$(OBJDIR)/date-cl.o \
	$(OBJDIR)/enum-cl.o \
	$(OBJDIR)/konami-cl.o \
	$(OBJDIR)/param-cl.o \
	$(OBJDIR)/param_train-aux.o \
	$(OBJDIR)/stats-cl.o \
	$(OBJDIR)/string-cl.o \
	$(OBJDIR)/sys-cl.o \
	$(OBJDIR)/train-aux.o \
	$(OBJDIR)/read-cl.o \
	$(OBJDIR)/utils-cl.o \
	$(GSL_INCLUDES) $(GSL_FLAGS) $(GSL_LIBS) \
	$(OPENCV_INCLUDES) $(OPENCV_FLAGS) $(OPENCV_LIBS) \
	-lm

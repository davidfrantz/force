force-parameter: \
  alloc-cl \
  date-cl \
  dir-cl \
  enum-cl \
  gdalopt-cl \
  konami-cl \
  param-aux \
  param-cl \
  param-hl \
  param-ll \
  read-cl \
  string-cl \
  sys-cl \
  utils-cl \
  $(EXE_AUX_DIR)/force-parameter.c
	$(GCC) -o $(BINDIR)/force-parameter $(EXE_AUX_DIR)/force-parameter.c \
	$(OBJDIR)/alloc-cl.o \
	$(OBJDIR)/date-cl.o \
	$(OBJDIR)/dir-cl.o \
	$(OBJDIR)/enum-cl.o \
	$(OBJDIR)/gdalopt-cl.o \
	$(OBJDIR)/konami-cl.o \
	$(OBJDIR)/param-aux.o \
	$(OBJDIR)/param-cl.o \
	$(OBJDIR)/param-hl.o \
	$(OBJDIR)/param-ll.o \
	$(OBJDIR)/read-cl.o \
	$(OBJDIR)/string-cl.o \
	$(OBJDIR)/sys-cl.o \
	$(OBJDIR)/utils-cl.o \
	-lm

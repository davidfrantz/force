force-runtime-data: \
  alloc-cl \
  dir-cl \
  index-parse-hl \
  konami-cl \
  string-cl \
  utils-cl \
  sensor-hl \
  sys-cl \
  $(EXE_AUX_DIR)/force-runtime-data.c
	$(GCC) -o $(BINDIR)/force-runtime-data $(EXE_AUX_DIR)/force-runtime-data.c \
	$(OBJDIR)/alloc-cl.o \
	$(OBJDIR)/dir-cl.o \
	$(OBJDIR)/index-parse-hl.o \
	$(OBJDIR)/konami-cl.o \
	$(OBJDIR)/string-cl.o \
	$(OBJDIR)/utils-cl.o \
	$(OBJDIR)/sensor-hl.o \
	$(OBJDIR)/sys-cl.o \
	-lm \
	-ljansson

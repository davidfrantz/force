force-info: \
  alloc-cl \
  dir-cl \
  konami-cl \
  string-cl \
  sys-cl \
  utils-cl \
  $(EXE_AUX_DIR)/force-info.c
	$(GCC) -o $(BINDIR)/force-info $(EXE_AUX_DIR)/force-info.c \
	$(OBJDIR)/alloc-cl.o \
	$(OBJDIR)/dir-cl.o \
	$(OBJDIR)/konami-cl.o \
	$(OBJDIR)/string-cl.o \
	$(OBJDIR)/sys-cl.o \
	$(OBJDIR)/utils-cl.o \
	-lm

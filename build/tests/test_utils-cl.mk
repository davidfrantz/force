test_utils-cl: \
  prepare_tests \
  alloc-cl \
  dir-cl \
  string-cl \
  sys-cl \
  utils-cl \
  $(TEST_DIR)/test_utils-cl.c \
  $(UNITY)
	$(GCC) -o $(BINDIR)/force-test/test_utils-cl $(TEST_DIR)/test_utils-cl.c $(UNITY) \
	$(OBJDIR)/alloc-cl.o \
	$(OBJDIR)/dir-cl.o \
	$(OBJDIR)/string-cl.o \
	$(OBJDIR)/sys-cl.o \
	$(OBJDIR)/utils-cl.o

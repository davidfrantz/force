test_alloc-cl: \
  prepare_tests \
  alloc-cl \
  $(TEST_DIR)/test_alloc-cl.c \
  $(UNITY)
	$(GCC) -o $(BINDIR)/force-test/test_alloc-cl $(TEST_DIR)/test_alloc-cl.c $(UNITY) \
	$(OBJDIR)/alloc-cl.o

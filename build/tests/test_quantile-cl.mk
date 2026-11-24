test_quantile-cl: prepare_tests
	$(GCC) -o $(BINDIR)/force-test/test_quantile-cl \
	$(TEST_DIR)/test_quantile-cl.c $(UNITY) \
	$(SRCDIR)/modules/cross-level/stats-cl.c \
	$(SRCDIR)/modules/cross-level/alloc-cl.c \
	$(GSL_INCLUDES) $(GSL_FLAGS) $(GSL_LIBS) -lm
	$(BINDIR)/force-test/test_quantile-cl

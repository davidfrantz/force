# UNIT TESTS

TEST_DIR=$(SRCDIR)/tests
UNITY=$(TEST_DIR)/unity/unity.c

include $(BUILDDIR)/tests/test_alloc-cl.mk
include $(BUILDDIR)/tests/test_utils-cl.mk

prepare_tests: prepare
	mkdir -p $(BINDIR)/force-test

tests: \
  test_alloc-cl \
  test_utils-cl


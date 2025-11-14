# EXECUTABLES

EXE_DIR=$(SRCDIR)/main
EXE_AUX_DIR=$(EXE_DIR)/aux-level
EXE_LOWER_DIR=$(EXE_DIR)/lower-level
EXE_HIGHER_DIR=$(EXE_DIR)/higher-level

include $(BUILDDIR)/executables/dummy.mk
include $(BUILDDIR)/executables/force-cube-init.mk
include $(BUILDDIR)/executables/force-higher-level.mk
include $(BUILDDIR)/executables/force-hist.mk
include $(BUILDDIR)/executables/force-import-modis.mk
include $(BUILDDIR)/executables/force-info.mk
include $(BUILDDIR)/executables/force-l2ps.mk
include $(BUILDDIR)/executables/force-lut-modis.mk
include $(BUILDDIR)/executables/force-mdcp.mk
include $(BUILDDIR)/executables/force-parameter.mk
include $(BUILDDIR)/executables/force-qai-inflate.mk
include $(BUILDDIR)/executables/force-stack.mk
include $(BUILDDIR)/executables/force-stratified-sample.mk
include $(BUILDDIR)/executables/force-tabulate-grid.mk
include $(BUILDDIR)/executables/force-tile-finder.mk
include $(BUILDDIR)/executables/force-train.mk

exe: \
  force-cube-init \
  force-higher-level \
  force-hist \
  force-import-modis \
  force-info \
  force-l2ps \
  force-lut-modis \
  force-mdcp \
  force-parameter \
  force-qai-inflate \
  force-stack \
  force-stratified-sample \
  force-tabulate-grid \
  force-tile-finder \
  force-train

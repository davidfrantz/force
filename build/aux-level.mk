## AUX LEVEL COMPILE UNITS

AUX_DIR=$(SRCDIR)/modules/aux-level

aux: \
    param-aux \
    param_train-aux \
    train-aux

param-aux: prepare $(AUX_DIR)/param-aux.c
	$(GCC) -c $(AUX_DIR)/param-aux.c -o $(OBJDIR)/param-aux.o

param_train-aux: prepare $(AUX_DIR)/param-train-aux.c
	$(GCC) -c $(AUX_DIR)/param-train-aux.c -o $(OBJDIR)/param_train-aux.o

train-aux: prepare $(AUX_DIR)/train-aux.cpp
	$(G11) $(OPENCV_INCLUDES) $(OPENCV_FLAGS) -c $(AUX_DIR)/train-aux.cpp -o $(OBJDIR)/train-aux.o $(OPENCV_LIBS)

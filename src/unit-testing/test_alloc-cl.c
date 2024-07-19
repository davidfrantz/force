#include "unity/unity.h"
#include "../cross-level/alloc-cl.h"

void setUp(void) { }

void tearDown(void) { }

// Test cases for alloc
void test_alloc_should_AllocateMemory(void) {
int *ptr = NULL;
  alloc((void**)&ptr, 2, sizeof(int));
  TEST_ASSERT_NOT_NULL(ptr);
  free((void*)ptr); // Free the allocated memory
}

void test_alloc_should_InitializeMemoryToZero(void) {
int *ptr = NULL;
  alloc((void**)&ptr, 2, sizeof(int));
  for (int i = 0; i < 2; i++) {
    TEST_ASSERT_EQUAL(0, ptr[i]);
  }
  free((void*)ptr); // Free the allocated memory
}

// Test cases for alloc_2D
void test_alloc_2D_should_Allocate2DArray(void) {
int **ptr = NULL;
  alloc_2D((void***)&ptr, 2, 3, sizeof(int));
  TEST_ASSERT_NOT_NULL(ptr);
  for (int i = 0; i < 2; i++) {
    TEST_ASSERT_NOT_NULL(ptr[i]);
  }
  free_2D((void**)ptr, 2); // Free the allocated memory
}

void test_alloc_2D_should_Initialize2DArrayToZero(void) {
int **ptr = NULL;
  alloc_2D((void***)&ptr, 2, 3, sizeof(int));
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 3; j++) {
      TEST_ASSERT_EQUAL(0, ptr[i][j]);
    }
  }
  free_2D((void**)ptr, 2); // Free the allocated memory
}

// Test cases for alloc_2DC
void test_alloc_2DC_should_AllocateContiguous2DArray(void) {
int **ptr = NULL;
  alloc_2DC((void***)&ptr, 2, 3, sizeof(int));
  TEST_ASSERT_NOT_NULL(ptr);
  TEST_ASSERT_NOT_NULL(ptr[0]);
  free_2DC((void**)ptr); // Free the allocated memory
}

void test_alloc_2DC_should_InitializeContiguous2DArrayToZero(void) {
int **ptr = NULL;
  alloc_2DC((void***)&ptr, 2, 3, sizeof(int));
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 3; j++) {
      TEST_ASSERT_EQUAL(0, ptr[i][j]);
    }
  }
  free_2DC((void**)ptr); // Free the allocated memory
}

// Test cases for alloc_3D
void test_alloc_3D_should_Allocate3DArray(void) {
int ***ptr = NULL;
  alloc_3D((void****)&ptr, 2, 3, 4, sizeof(int));
  TEST_ASSERT_NOT_NULL(ptr);
  for (int i = 0; i < 2; i++) {
    TEST_ASSERT_NOT_NULL(ptr[i]);
    for (int j = 0; j < 3; j++) {
      TEST_ASSERT_NOT_NULL(ptr[i][j]);
    }
  }
  free_3D((void***)ptr, 2, 3); // Free the allocated memory
}

void test_alloc_3D_should_Initialize3DArrayToZero(void) {
int ***ptr = NULL;
  alloc_3D((void****)&ptr, 2, 3, 4, sizeof(int));
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 3; j++) {
      for (int k = 0; k < 4; k++) {
        TEST_ASSERT_EQUAL(0, ptr[i][j][k]);
      }
    }
  }
  free_3D((void***)ptr, 2, 3); // Free the allocated memory
}

// Test cases for re_alloc
void test_re_alloc_should_ReallocateMoreMemory(void) {
int *ptr = NULL;
  alloc((void**)&ptr, 5, sizeof(int));
  re_alloc((void**)&ptr, 5, 10, sizeof(int));
  TEST_ASSERT_NOT_NULL(ptr);
  free(ptr); // Free the allocated memory
}

// Test cases for re_alloc
void test_re_alloc_should_ReallocateLessMemory(void) {
int *ptr = NULL;
  alloc((void**)&ptr, 10, sizeof(int));
  re_alloc((void**)&ptr, 10, 5, sizeof(int));
  TEST_ASSERT_NOT_NULL(ptr);
  free(ptr); // Free the allocated memory
}

void test_re_alloc_should_InitializeMoreMemoryToZero(void) {
int *ptr = NULL;
  alloc((void**)&ptr, 5, sizeof(int));
  re_alloc((void**)&ptr, 5, 10, sizeof(int));
  for (int i = 0; i < 10; i++) {
      TEST_ASSERT_EQUAL(0, ptr[i]);
  }
  free(ptr); // Free the allocated memory
}

void test_re_alloc_should_InitializeLessMemoryToZero(void) {
int *ptr = NULL;
  alloc((void**)&ptr, 10, sizeof(int));
  re_alloc((void**)&ptr, 10, 5, sizeof(int));
  for (int i = 0; i < 5; i++) {
      TEST_ASSERT_EQUAL(0, ptr[i]);
  }
  free(ptr); // Free the allocated memory
}

// Test cases for re_alloc_2D
void test_re_alloc_2D_should_Reallocate2DArray(void) {
int **ptr = NULL;
  alloc_2D((void***)&ptr, 2, 3, sizeof(int));
  re_alloc_2D((void***)&ptr, 2, 3, 4, 6, sizeof(int));
  TEST_ASSERT_NOT_NULL(ptr);
  for (int i = 0; i < 4; i++) {
    TEST_ASSERT_NOT_NULL(ptr[i]);
  }
  free_2D((void**)ptr, 4); // Free the allocated memory
}

void test_re_alloc_2D_should_InitializeNew2DArrayMemoryToZero(void) {
int **ptr = NULL;
  alloc_2D((void***)&ptr, 2, 3, sizeof(int));
  re_alloc_2D((void***)&ptr, 2, 3, 4, 6, sizeof(int));
  for (int i = 0; i < 4; i++) {
      for (int j = 0; j < 6; j++) {
          TEST_ASSERT_EQUAL(0, ptr[i][j]);
      }
  }
  free_2D((void**)ptr, 4); // Free the allocated memory
}

// Test cases for re_alloc_2DC
void test_re_alloc_2DC_should_ReallocateContiguous2DArray(void) {
int **ptr = NULL;
  alloc_2DC((void***)&ptr, 2, 3, sizeof(int));
  re_alloc_2DC((void***)&ptr, 2, 3, 4, 6, sizeof(int));
  TEST_ASSERT_NOT_NULL(ptr);
  TEST_ASSERT_NOT_NULL(ptr[0]);
  free_2DC((void**)ptr); // Free the allocated memory
}

void test_re_alloc_2DC_should_InitializeNewContiguous2DArrayMemoryToZero(void) {
int **ptr = NULL;
  alloc_2DC((void***)&ptr, 2, 3, sizeof(int));
  re_alloc_2DC((void***)&ptr, 2, 3, 4, 6, sizeof(int));
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 6; j++) {
      TEST_ASSERT_EQUAL(0, ptr[i][j]);
    }
  }
  free_2DC((void**)ptr); // Free the allocated memory
}

// Test cases for re_alloc_3D
void test_re_alloc_3D_should_Reallocate3DArray(void) {
int ***ptr = NULL;
  alloc_3D((void****)&ptr, 2, 3, 5, sizeof(int));
  re_alloc_3D((void****)&ptr, 2, 3, 5, 4, 6, 10, sizeof(int));
  TEST_ASSERT_NOT_NULL(ptr);
  for (int i = 0; i < 4; i++) {
    TEST_ASSERT_NOT_NULL(ptr[i]);
    for (int j = 0; j < 6; j++) {
      TEST_ASSERT_NOT_NULL(ptr[i][j]);
    }
  }
  free_3D((void***)ptr, 4, 6); // Free the allocated memory
}

void test_re_alloc_3D_should_InitializeNew3DArrayMemoryToZero(void) {
int ***ptr = NULL;
  alloc_3D((void****)&ptr, 2, 3, 5, sizeof(int));
  re_alloc_3D((void****)&ptr, 2, 3, 5, 4, 6, 10, sizeof(int));
  for (int i = 0; i < 4; i++) {
      for (int j = 0; j < 6; j++) {
          for (int k = 0; k < 10; k++) {
              TEST_ASSERT_EQUAL(0, ptr[i][j][k]);
          }
      }
  }
  free_3D((void***)ptr, 4, 6); // Free the allocated memory
}

int main(void) {

  UNITY_BEGIN();

  RUN_TEST(test_alloc_should_AllocateMemory);
  RUN_TEST(test_alloc_should_InitializeMemoryToZero);

  RUN_TEST(test_alloc_2D_should_Allocate2DArray);
  RUN_TEST(test_alloc_2D_should_Initialize2DArrayToZero);

  RUN_TEST(test_alloc_2DC_should_AllocateContiguous2DArray);
  RUN_TEST(test_alloc_2DC_should_InitializeContiguous2DArrayToZero);

  RUN_TEST(test_alloc_3D_should_Allocate3DArray);
  RUN_TEST(test_alloc_3D_should_Initialize3DArrayToZero);

  RUN_TEST(test_re_alloc_should_ReallocateMoreMemory);
  RUN_TEST(test_re_alloc_should_InitializeMoreMemoryToZero);
  RUN_TEST(test_re_alloc_should_ReallocateLessMemory);
  RUN_TEST(test_re_alloc_should_InitializeLessMemoryToZero);

  RUN_TEST(test_re_alloc_2D_should_Reallocate2DArray);
  RUN_TEST(test_re_alloc_2D_should_InitializeNew2DArrayMemoryToZero);

  RUN_TEST(test_re_alloc_2DC_should_ReallocateContiguous2DArray);
  RUN_TEST(test_re_alloc_2DC_should_InitializeNewContiguous2DArrayMemoryToZero);
  
  RUN_TEST(test_re_alloc_3D_should_Reallocate3DArray);
  RUN_TEST(test_re_alloc_3D_should_InitializeNew3DArrayMemoryToZero);

  return UNITY_END();

}


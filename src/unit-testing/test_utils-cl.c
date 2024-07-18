#include "unity/unity.h"
#include <limits.h>
#include "../cross-level/utils-cl.h"

void setUp(void) { }

void tearDown(void) { }

void test_fequal_Should_ReturnTrue_ForEqualFloats(void) {
  TEST_ASSERT_TRUE(fequal(1.0f, 1.0f));
  TEST_ASSERT_TRUE(fequal(0.0f, 0.0f));
  TEST_ASSERT_TRUE(fequal(-1.0f, -1.0f));
}

void test_fequal_Should_ReturnTrue_ForNearlyEqualFloats(void) {
  TEST_ASSERT_TRUE(fequal(1.0f, 1.0f + FLT_EPSILON / 2.0));
  TEST_ASSERT_TRUE(fequal(1.0f, 1.0f - FLT_EPSILON / 2.0));
}

void test_fequal_Should_ReturnFalse_ForDifferentFloats(void) {
  TEST_ASSERT_FALSE(fequal(1.0f, 2.0f));
  TEST_ASSERT_FALSE(fequal(0.0f, FLT_EPSILON));
}

void test_dequal_Should_ReturnTrue_ForEqualDoubles(void) {
  TEST_ASSERT_TRUE(dequal(1.0, 1.0));
  TEST_ASSERT_TRUE(dequal(0.0, 0.0));
  TEST_ASSERT_TRUE(dequal(-1.0, -1.0));
}

void test_dequal_Should_ReturnTrue_ForNearlyEqualDoubles(void) {
  TEST_ASSERT_TRUE(dequal(1.0, 1.0 + DBL_EPSILON / 2.0));
  TEST_ASSERT_TRUE(dequal(1.0, 1.0 - DBL_EPSILON / 2.0));
}

void test_dequal_Should_ReturnFalse_ForDifferentDoubles(void) {
  TEST_ASSERT_FALSE(dequal(1.0, 2.0));
  TEST_ASSERT_FALSE(dequal(0.0, DBL_EPSILON));
}


void test_num_decimal_places_SingleDigit(void) {
  TEST_ASSERT_EQUAL_INT(1, num_decimal_places(0));
  TEST_ASSERT_EQUAL_INT(1, num_decimal_places(5));
}

void test_num_decimal_places_MultipleDigit(void) {
  TEST_ASSERT_EQUAL_INT(2, num_decimal_places(50));
  TEST_ASSERT_EQUAL_INT(3, num_decimal_places(500));
  TEST_ASSERT_EQUAL_INT(4, num_decimal_places(5000));
}

void test_num_decimal_places_NegativeNumbers(void) {
  TEST_ASSERT_EQUAL_INT(1, num_decimal_places(-5));
  TEST_ASSERT_EQUAL_INT(2, num_decimal_places(-50));
  TEST_ASSERT_EQUAL_INT(3, num_decimal_places(-500));
  TEST_ASSERT_EQUAL_INT(4, num_decimal_places(-5000));
}

void test_num_decimal_places_LargeNumbers(void) {
    TEST_ASSERT_EQUAL_INT(10, num_decimal_places(1234567890));
    TEST_ASSERT_EQUAL_INT(10, num_decimal_places(INT_MAX));
    TEST_ASSERT_EQUAL_INT(10, num_decimal_places(INT_MIN));
}

int main(void) {

  UNITY_BEGIN();

  RUN_TEST(test_fequal_Should_ReturnTrue_ForEqualFloats);
  RUN_TEST(test_fequal_Should_ReturnTrue_ForNearlyEqualFloats);
  RUN_TEST(test_fequal_Should_ReturnFalse_ForDifferentFloats);
  
  RUN_TEST(test_dequal_Should_ReturnTrue_ForEqualDoubles);
  RUN_TEST(test_dequal_Should_ReturnTrue_ForNearlyEqualDoubles);
  RUN_TEST(test_dequal_Should_ReturnFalse_ForDifferentDoubles);
  
  RUN_TEST(test_num_decimal_places_SingleDigit);
  RUN_TEST(test_num_decimal_places_MultipleDigit);
  RUN_TEST(test_num_decimal_places_NegativeNumbers);
  RUN_TEST(test_num_decimal_places_LargeNumbers);

  return UNITY_END();

}

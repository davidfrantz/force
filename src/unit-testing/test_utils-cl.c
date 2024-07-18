#include "unity/unity.h"
#include <limits.h>
#include "../cross-level/utils-cl.h"

void setUp(void) { }

void tearDown(void) { }

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

  RUN_TEST(test_num_decimal_places_SingleDigit);
  RUN_TEST(test_num_decimal_places_MultipleDigit);
  RUN_TEST(test_num_decimal_places_NegativeNumbers);
  RUN_TEST(test_num_decimal_places_LargeNumbers);

  return UNITY_END();

}

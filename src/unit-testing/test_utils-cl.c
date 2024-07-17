#include "unity/unity.h"
#include "../cross-level/utils-cl.h"


void setUp(void) { }

void tearDown(void) { }

void test_num_decimal_places_negative_1(void) {
  int result = num_decimal_places(-5);
  TEST_ASSERT_EQUAL(1, result);
}

void test_num_decimal_places_negative_2(void) {
  int result = num_decimal_places(-50);
  TEST_ASSERT_EQUAL(2, result);
}

void test_num_decimal_places_positive_1(void) {
  int result = num_decimal_places(5);
  TEST_ASSERT_EQUAL(1, result);
}

void test_num_decimal_places_positive_2(void) {
  int result = num_decimal_places(50);
  TEST_ASSERT_EQUAL(2, result);
}

void test_num_decimal_places_zero(void) {
  int result = num_decimal_places(0);
  TEST_ASSERT_EQUAL(1, result);
}

int main(void) {

  UNITY_BEGIN();

  RUN_TEST(test_num_decimal_places_negative_1);
  RUN_TEST(test_num_decimal_places_negative_2);
  RUN_TEST(test_num_decimal_places_positive_1);
  RUN_TEST(test_num_decimal_places_positive_2);
  RUN_TEST(test_num_decimal_places_zero);

  return UNITY_END();

}

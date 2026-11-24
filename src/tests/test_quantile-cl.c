#include "unity/unity.h"
#include <limits.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "../modules/cross-level/stats-cl.h"

// quantile probabilities and thread counts that every test case is run with
static const float probabilities[] = {0.01f, 0.25f, 0.99f};
static const int thread_counts[] = {1, 4, 8};

// small deterministic RNG (xorshift), reproducible between runs
static uint32_t rng_state = 12345u;
static uint32_t next_random(void) {
  rng_state ^= rng_state << 13;
  rng_state ^= rng_state >> 17;
  rng_state ^= rng_state << 5;
  return rng_state;
}
static void check_quantiles(int nx, int ny) {
  const int nc = nx * ny;
  const int land_percentages[] = {0, 50, 100};
  short *pixels = calloc((size_t)nc, sizeof(*pixels));
  small *land = calloc((size_t)nc, sizeof(*land));

  TEST_ASSERT_NOT_NULL(pixels);
  TEST_ASSERT_NOT_NULL(land);

  for (int percentage_index = 0; percentage_index < 3; percentage_index++) {
    // the first nland pixels are land, the rest is cloud/not land

    const int nland = nc * land_percentages[percentage_index] / 100;
    double *reference = NULL;
    short expected[3];

    if (nland > 0) {

      // old-style double copy of the land pixels, input for the reference quantile()
      reference = malloc((size_t)nland * sizeof(*reference));
      TEST_ASSERT_NOT_NULL(reference);


      // pick exactly nland random land pixels
      int selected = 0;
      for (int i = 0; i < nc; i++) {
        if (next_random() % (uint32_t)(nc - i) < (uint32_t)(nland - selected)) {
          land[i] = 1;
          switch (selected % 7) {
          case 0: pixels[i] = SHRT_MIN; break;
          case 1: pixels[i] = SHRT_MAX; break;
          default: pixels[i] = (short)(((selected % 5) - 2) * 13); break;
          }
          reference[selected] = pixels[i];

          selected++;
        } else {
          pixels[i] = 1000;
        }
      }
    }

    for (int probability_index = 0; probability_index < 3; probability_index++) {
      expected[probability_index] = nland > 0
        ? (short)quantile(reference, nland, probabilities[probability_index])
        : 0;
    }

    for (int thread_index = 0; thread_index < 3; thread_index++) {
      short results[3] = {SHRT_MIN, SHRT_MIN, SHRT_MIN};
      const int nthreads = thread_counts[thread_index];

      #pragma omp parallel for num_threads(nthreads)
      for (int probability_index = 0; probability_index < 3; probability_index++) {
        uint32_t *histogram = calloc(65536, sizeof(*histogram));
        if (histogram != NULL) {
          results[probability_index] = quantile_short(
            pixels, land, nc, nland, probabilities[probability_index], histogram);
          free(histogram);
        }
      }

      for (int probability_index = 0; probability_index < 3; probability_index++) {
        TEST_ASSERT_EQUAL_INT(expected[probability_index], results[probability_index]);
      }
    }

    free(reference);
    memset(land, 0, (size_t)nc * sizeof(*land));
    memset(pixels, 0, (size_t)nc * sizeof(*pixels));
  }

  free(land);
  free(pixels);
}

void setUp(void) { }
void tearDown(void) { }

void test_quantile_short_matches_gsl_across_land_fractions_probabilities_and_threads(void) {
  const int sizes[][2] = {{109, 109}, {75, 75}};

  for (size_t i = 0; i < sizeof(sizes) / sizeof(sizes[0]); i++) {
    check_quantiles(sizes[i][0], sizes[i][1]);
  }
}

int main(void) {
  UNITY_BEGIN();
  RUN_TEST(test_quantile_short_matches_gsl_across_land_fractions_probabilities_and_threads);
  return UNITY_END();
}

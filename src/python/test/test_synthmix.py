from unittest import TestCase

import numpy as np

from synthmix import synthMix


class TestSynthMix(TestCase):

    def test(self):
        features = np.array([
            [1, 2, 3, 4],
            [10, 20, 30, 40],
            [100, 200, 300, 400],
            [1000, 2000, 3000, 4000]
        ])
        responses = np.array([1, 2, 3, 1])

        mixtureStream = synthMix(
            features=features, responses=responses, target=1, mixingLikelihood={1: 0.2, 2: 0.4, 3: 0.4},
            classLikelihood={1: 0.4, 2: 0.3, 3: 0.3}, includeWithinClassMixtures=True
        )

        for i, mixture in enumerate(mixtureStream):
            print(mixture)

            if i == 3:
                break

from typing import Dict, Union, NamedTuple, Iterator, List

import numpy as np


class Mixture(NamedTuple):
    classIds: List[int]
    indices: List[int]
    fractions: List[float]
    profile: List[float]


def synthMix(
        features: np.ndarray, responses: np.array, target: int, mixingLikelihood: Dict[int, float],
        classLikelihood: Union[Dict[int, float], str], includeWithinClassMixtures=False, targetRange=(0., 1.)
) -> Iterator[Mixture]:
    # prepare parameters and check consistency
    assert isinstance(features, np.ndarray) and features.ndim == 2
    assert isinstance(responses, np.ndarray) and responses.ndim == 1
    assert len(features) == len(responses)
    classIds, counts = np.unique(responses, return_counts=True)
    if classLikelihood is None:
        classLikelihood = 'proportional'
    if classLikelihood == 'proportional':
        classLikelihood = {classId: float(count) / len(responses) for classId, count in zip(classIds, counts)}
    elif classLikelihood == 'equalized':
        classLikelihood = {classId: 1. / len(classIds) for classId in classIds}
    assert isinstance(mixingLikelihood, dict)
    assert isinstance(classLikelihood, dict)
    for classId in classIds:
        assert classId in classLikelihood

    # cache feature locations by class
    indicesByClassId = dict()
    for classId in classIds:
        indicesByClassId[classId] = np.where(responses == classId)[0]

    # remove within class mixtures if requested
    if includeWithinClassMixtures:
        replace = True
    else:
        classLikelihood = {k: v / (1 - classLikelihood[target]) for k, v in classLikelihood.items() if k != target}
        replace = False

    # prepare random sampling
    complexitiesV = list(mixingLikelihood.keys())
    complexitiesP = list(mixingLikelihood.values())
    classP = [classLikelihood[classId] for classId in classIds]

    # generate endless stream of mixtures (caller needs to break out!)
    while True:
        # draw mixing complexity
        complexity = np.random.choice(complexitiesV, p=complexitiesP)

        # draw classes
        drawnClassIds = [target]
        drawnClassIds.extend(np.random.choice(classIds, size=complexity - 1, replace=replace, p=classP))

        # draw profiles
        drawnIndices = [np.random.choice(indicesByClassId[label]) for label in drawnClassIds]

        # draw fractions
        drawnFractions = list()
        for i in range(complexity - 1):
            if i == 0:
                fraction = np.random.random() * (targetRange[1] - targetRange[0]) + targetRange[0]
            else:
                fraction = np.random.random() * (1. - sum(drawnFractions))
            drawnFractions.append(fraction)
        drawnFractions.append(1. - sum(drawnFractions))
        assert sum(drawnFractions) == 1.

        # mix
        mixedProfile = list(np.sum(features[drawnIndices] * np.reshape(drawnFractions, (-1, 1)), axis=0))

        yield Mixture(classIds=drawnClassIds, indices=drawnIndices, fractions=drawnFractions, profile=mixedProfile)

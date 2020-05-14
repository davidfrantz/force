import argparse

from os import makedirs
from os.path import join, exists

import numpy as np


class Mixture(object):
    def __init__(self, classIds, indices, fractions, profile):
        self.classIds = classIds
        self.indices = indices
        self.fractions = fractions
        self.profile = profile


def synthMixCore(
        features, response, target, mixingLikelihood, classLikelihood, includeWithinClassMixtures=False,
        targetRange=(0., 1.)
):
    # prepare parameters and check consistency
    assert isinstance(features, np.ndarray) and features.ndim == 2
    assert isinstance(response, np.ndarray) and response.ndim == 1
    assert len(features) == len(response)
    classIds, counts = np.unique(response, return_counts=True)
    if classLikelihood is None:
        classLikelihood = 'proportional'
    if isinstance(classLikelihood, str):
        if classLikelihood.lower() == 'proportional':
            classLikelihood = {classId: float(count) / len(response) for classId, count in zip(classIds, counts)}
        elif classLikelihood.lower() == 'equalized':
            classLikelihood = {classId: 1. / len(classIds) for classId in classIds}
    assert isinstance(mixingLikelihood, dict)
    assert isinstance(classLikelihood, dict)
    for classId in classIds:
        assert classId in classLikelihood

    # cache feature locations by class
    indicesByClassId = dict()
    for classId in classIds:
        indicesByClassId[classId] = np.where(response == classId)[0]

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


def parsePrm(filenamePrm):
    with open(filenamePrm) as file:
        lines = file.readlines()

    parameters = dict()
    for line in lines:
        if line.startswith('#'):
            continue
        tmp = line.split('=')
        if len(tmp) != 2:
            continue
        key = tmp[0].strip()
        value = tmp[1].strip()
        parameters[key] = value
    return parameters


def synthMixCli(filenamePrm):
    parameters = parsePrm(filenamePrm=filenamePrm)
    features = np.genfromtxt(fname=parameters['FILE_FEATURES'])
    response = np.genfromtxt(fname=parameters['FILE_RESPONSE'])
    targets = [int(v) for v in parameters['TARGET_CLASS'].split(' ')]
    n = int(parameters['SYNTHETIC_MIXTURES'])
    mixingLikelihood = {
        int(complexity): float(likelihood) for complexity, likelihood in zip(
            parameters['MIXING_COMPLEXITY'].split(' '),
            parameters['MIXING_LIKELIHOOD'].split(' ')
        )
    }
    if parameters['CLASS_LIKELIHOOD'].lower() in ['proportional', 'equalized']:
        classLikelihood = parameters['CLASS_LIKELIHOOD'].lower()
    else:
        classLikelihood = {
            int(complexity): float(likelihood) for complexity, likelihood in zip(
                parameters['TARGET_CLASS'].split(' '),
                parameters['CLASS_LIKELIHOOD'].split(' ')
            )
        }
    includeWithinClassMixtures = parameters['WITHIN_CLASS_MIXING'] == 'TRUE'
    iterations = int(parameters['ITERATIONS'])
    if not exists(parameters['DIR_MIXES']):
        makedirs(parameters['DIR_MIXES'])

    for iteration in range(1, iterations + 1):
        for target in targets:
            filenameFeatures = join(parameters['DIR_MIXES'],
                '{}_FEATURES_CLASS-{}_ITERATION-{}.txt'.format(parameters["BASE_MIXES"], str(target).zfill(3),
                    str(iteration).zfill(3)))

            filenameResponse = join(parameters['DIR_MIXES'],
                '{}_RESPONSE_CLASS-{}_ITERATION-{}.txt'.format(parameters["BASE_MIXES"], str(target).zfill(3),
                    str(iteration).zfill(3)))

            mixtureStream = synthMixCore(
                features=features, response=response, target=target, mixingLikelihood=mixingLikelihood,
                classLikelihood=classLikelihood, includeWithinClassMixtures=includeWithinClassMixtures
            )
            with open(filenameFeatures, 'w') as fileFeatures, open(filenameResponse, 'w') as fileResponse:
                for i, mixture in enumerate(mixtureStream, 1):
                    fileFeatures.write(' '.join([str(round(v, 2)) for v in mixture.profile]) + '\n')
                    fileResponse.write(str(round(mixture.fractions[0], 4)) + '\n')
                    if i == n:
                        break
                for profile, classId in zip(features, response):
                    if classId == target:
                        fileFeatures.write(' '.join([str(round(v, 2)) for v in profile]) + '\n')
                        fileResponse.write('1.0\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('parameterFile', type=str, help='parameter file')
    args = parser.parse_args()
    synthMixCli(filenamePrm=args.parameterFile)

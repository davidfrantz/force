#!/usr/bin/env python3

##########################################################################
# 
# This file is part of FORCE - Framework for Operational Radiometric 
# Correction for Environmental monitoring.
# 
# Copyright (C) 2013-2022 David Frantz
# 
# FORCE is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# FORCE is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with FORCE.  If not, see <http://www.gnu.org/licenses/>.
# 
##########################################################################

# This script generates synthetic mixtures from endmember feature vectors

# Copyright (C) 2020 Andreas Rabe
# Contact: andreas.rabe@geo.hu-berlin.de


import argparse
from os.path import join, exists

import numpy as np


class Mixture(object):
    def __init__(self, classIds, indices, fractions, profile):
        self.classIds = classIds
        self.indices = indices
        self.fractions = fractions
        self.profile = profile


def synthMixCore(
        features, response, target, classes, mixingLikelihood, classLikelihood, includeWithinClassMixtures=False,
        targetRange=(0., 1.)
):
    # prepare parameters and check consistency
    assert isinstance(features, np.ndarray) and features.ndim == 2
    assert isinstance(response, np.ndarray) and response.ndim == 1
    assert isinstance(classes, list)
    assert len(features) == len(response)
    _classIds, counts = np.unique(response, return_counts=True)
    classIds = classes
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
        classIds = [v for v in classIds if v != target]
        replace = False

    # prepare random sampling
    complexitiesV = list(mixingLikelihood.keys())
    complexitiesP = list(mixingLikelihood.values())
    complexitiesP = [v / sum(complexitiesP) for v in complexitiesP]
    classP = [classLikelihood[classId] for classId in classIds]
    classP = [v / sum(classP) for v in classP]

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
    if not exists(filenamePrm):
        print('Unable to open parameter file!')
        print('Reading parameter file failed!')
        exit(1)

    with open(filenamePrm) as file:
        lines = file.readlines()

    lines = [line.strip() for line in lines if line.strip() != '']

    if lines[0] != '++PARAM_SYNTHMIX_START++':
        print('Not a synthmix parameter file!')
        exit(1)

    if lines[-1] != '++PARAM_SYNTHMIX_END++':
        print('Not a synthmix parameter file!')
        exit(1)

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

    keys = [
        'FILE_FEATURES', 'FILE_RESPONSE', 'USE_CLASSES', 'SYNTHETIC_MIXTURES', 'INCLUDE_ORIGINAL', 'MIXING_COMPLEXITY',
        'MIXING_LIKELIHOOD', 'WITHIN_CLASS_MIXING', 'CLASS_LIKELIHOOD', 'ITERATIONS', 'TARGET_CLASS', 'DIR_MIXES',
        'BASE_MIXES'
    ]
    missingKey = False
    for key in keys:
        if not key in parameters:
            print('parameter ' + key + ' was not set.')
            missingKey = True
    if missingKey:
        print('Reading parameter file failed!')
        exit(1)

    keys = ['INCLUDE_ORIGINAL', 'WITHIN_CLASS_MIXING']
    wrongKey = False
    for key in keys:
        if parameters[key] not in ['TRUE', 'FALSE']:
            print('parameter ' + key + ' must be TRUE or FALSE.')
            wrongKey = True
    if wrongKey:
        print('Reading parameter file failed!')
        exit(1)

    return parameters


def synthMixCli(filenamePrm):
    parameters = parsePrm(filenamePrm=filenamePrm)

    if not exists(parameters['FILE_FEATURES']):
        print('parameter FILE_FEATURES does not exist in the filesystem.')
        print('Reading parameter file failed!')
        exit(1)

    if not exists(parameters['FILE_RESPONSE']):
        print('parameter FILE_RESPONSE does not exist in the filesystem.')
        print('Reading parameter file failed!')
        exit(1)

    if not exists(parameters['DIR_MIXES']):
        print('parameter DIR_MIXES does not exist in the filesystem.')
        print('Reading parameter file failed!')
        exit(1)

    features = np.genfromtxt(fname=parameters['FILE_FEATURES'])
    response = np.genfromtxt(fname=parameters['FILE_RESPONSE'])
    includeOriginal = parameters['INCLUDE_ORIGINAL'] == 'TRUE'

    if parameters['USE_CLASSES'].upper() == 'ALL':
        classes = list(sorted(set(response)))
    else:
        classes = [float(v) for v in parameters['USE_CLASSES'].split(' ')]

    if not set(classes).issubset(set(response)):
        print('parameter CLASSES must be a subset of classes in parameter FILE_RESPONSE file.')
        print('Reading parameter file failed!')
        exit(1)

    targets = [int(v) for v in parameters['TARGET_CLASS'].split(' ')]
    if not set(targets).issubset(set(classes)):
        print('parameter TARGET_CLASS must be a subset of parameter CLASSES.')
        print('Reading parameter file failed!')
        exit(1)

    n = int(parameters['SYNTHETIC_MIXTURES'])

    if n < 1:
        print('parameter SYNTHETIC_MIXTURES must be greather 0.')
        print('Reading parameter file failed!')
        exit(1)

    parsedMixingComplexity = [int(v) for v in parameters['MIXING_COMPLEXITY'].split(' ')]
    parsedMixingLikelihood = [float(v) for v in parameters['MIXING_LIKELIHOOD'].split(' ')]

    if min(parsedMixingComplexity) < 1:
        print('parameter MIXING_COMPLEXITY must be a list of integers greater 0.')
        print('Reading parameter file failed!')
        exit(1)

    if len(parsedMixingComplexity) != len(parsedMixingLikelihood):
        print('parameters MIXING_COMPLEXITY and MIXING_LIKELIHOOD must have matching length.')
        print('Reading parameter file failed!')
        exit(1)

    mixingLikelihood = {k: v for k, v in zip(parsedMixingComplexity, parsedMixingLikelihood)}

    if abs(sum(mixingLikelihood.values()) - 1) > 0.01:
        print('parameter MIXING_LIKELIHOOD must sum to one')
        print('Reading parameter file failed!')
        exit(1)

    if parameters['CLASS_LIKELIHOOD'].lower() in ['proportional', 'equalized']:
        classLikelihood = parameters['CLASS_LIKELIHOOD'].lower()
    else:
        if parameters['USE_CLASSES'].upper() == 'ALL':
            print('if parameter USE_CLASSES = ALL, CLASS_LIKELIHOOD must be set to EQUALIZED or PROPORTIONAL')
            print('Reading parameter file failed!')
            exit(1)
        parsedClassLikelihood = [float(v) for v in parameters['CLASS_LIKELIHOOD'].split(' ')]

        if len(parsedClassLikelihood) != len(classes):
            print('length of parameter CLASS_LIKELIHOOD and CLASSES must match')
            print('Reading parameter file failed!')
            exit(1)

        if abs(sum(parsedClassLikelihood) - 1) > 0.01:
            print('parameter CLASS_LIKELIHOOD must sum to one')
            print('Reading parameter file failed!')
            exit(1)

        classLikelihood = {int(c): float(likelihood) for c, likelihood in zip(classes, parsedClassLikelihood)}

    includeWithinClassMixtures = parameters['WITHIN_CLASS_MIXING'] == 'TRUE'
    iterations = int(parameters['ITERATIONS'])
    if iterations < 1:
        print('parameter ITERATIONS must be greater 0')
        print('Reading parameter file failed!')
        exit(1)

    for iteration in range(1, iterations + 1):
        for target in targets:
            filenameFeatures = join(parameters['DIR_MIXES'],
                '{}_FEATURES_CLASS-{}_ITERATION-{}.txt'.format(parameters["BASE_MIXES"], str(target).zfill(3),
                    str(iteration).zfill(3)))

            filenameResponse = join(parameters['DIR_MIXES'],
                '{}_RESPONSE_CLASS-{}_ITERATION-{}.txt'.format(parameters["BASE_MIXES"], str(target).zfill(3),
                    str(iteration).zfill(3)))

            mixtureStream = synthMixCore(
                features=features, response=response, target=target, classes=classes, mixingLikelihood=mixingLikelihood,
                classLikelihood=classLikelihood, includeWithinClassMixtures=includeWithinClassMixtures
            )

            with open(filenameFeatures, 'w') as fileFeatures, open(filenameResponse, 'w') as fileResponse:
                for i, mixture in enumerate(mixtureStream, 1):
                    fileFeatures.write(' '.join([str(round(v, 2)) for v in mixture.profile]) + '\n')
                    fileResponse.write(str(round(mixture.fractions[0], 4)) + '\n')
                    if i == n:
                        break

                if includeOriginal:
                    for profile, classId in zip(features, response):
                        fileFeatures.write(' '.join([str(round(v, 2)) for v in profile]) + '\n')
                        if classId == target:
                            fileResponse.write('1.0\n')
                        else:
                            fileResponse.write('0.0\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('parameterFile', type=str, help='parameter file')
    args = parser.parse_args()
    synthMixCli(filenamePrm=args.parameterFile)

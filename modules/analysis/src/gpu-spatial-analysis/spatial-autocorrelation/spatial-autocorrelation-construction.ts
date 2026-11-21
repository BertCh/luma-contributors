// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  snapshotWeightsFields,
  snapshotWeightsPort
} from '../spatial-weights/weights-construction-snapshot';

/** @internal Capture the declared analytic/adapter metadata, preserving resident GPU contents. */
export function snapshotSpatialAutocorrelationProps<T extends object>(value: T): T {
  const props = snapshotWeightsFields(value, [
    'id',
    'weights',
    'values',
    'secondValues',
    'events',
    'populations',
    'parameters',
    'mask',
    'output',
    'islands',
    'minimumNeighbors',
    'multipleTesting',
    'quadrantGating',
    'selfWeight',
    'operation',
    'additionalVariables',
    'quantileParameters',
    'maximumPermutations',
    'maximumNeighbors',
    'residualPower',
    'alternative',
    'mode',
    'statistics',
    'symmetryStrategy'
  ]) as T & {
    weights?: object;
    output: object;
    additionalVariables?: readonly object[];
    statistics?: readonly unknown[];
  };
  return Object.freeze({
    ...props,
    ...(props.weights ? {weights: snapshotWeightsPort(props.weights)} : {}),
    output: snapshotOutput(props.output),
    ...(props.additionalVariables
      ? {
          additionalVariables: Object.freeze(
            props.additionalVariables.map(variable =>
              snapshotWeightsFields(variable, ['values', 'quantileParameters'])
            )
          )
        }
      : {}),
    ...(props.statistics ? {statistics: Object.freeze([...props.statistics])} : {})
  }) as T;
}

function snapshotOutput(value: object): object {
  const output = snapshotWeightsFields(value, [
    'status',
    'zScores',
    'localI',
    'spatialLag',
    'quadrants',
    'pValues',
    'adjustedPValues',
    'neighborCounts',
    'globalStatistics',
    'globalCounts',
    'bins',
    'statistic',
    'standardizedRates',
    'statistics',
    'membership',
    'boundaries',
    'validCount',
    'classes',
    'invalidQuantile',
    'exceedances',
    'pseudoPValues',
    'observed',
    'significant',
    'islandCount',
    'analysisCount',
    'normalizedResiduals',
    'residualCount',
    'inference',
    'summary',
    'joinCounts',
    'counts'
  ]) as {status?: object; inference?: object};
  return Object.freeze({
    ...output,
    // Descriptive LOSH's status is a row view; other contributors use a map of scalar views.
    ...(output.status && !('buffer' in output.status)
      ? {
          status: snapshotWeightsFields(output.status, [
            'overflow',
            'candidateOverflow',
            'invalidCount',
            'weightsOverflow',
            'neighborOverflow',
            'invalidInput'
          ])
        }
      : {}),
    ...(output.inference ? {inference: snapshotOutput(output.inference)} : {})
  });
}

// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {validateRequirement} from '../../analysis-core/kernels/gpu-contributor-utils';
import {snapshotConstructionFields} from '../../analysis-core/kernels/construction-snapshot';

/** Capture compile-time envelope settings and output presence for contributors and operations. @internal */
export function snapshotEnvelopeConstruction<
  T extends {
    output: {
      lower: unknown;
      upper: unknown;
      statistics: unknown;
      status: unknown;
      pValueInterval?: unknown;
    };
  }
>(props: T): T {
  const captured = snapshotConstructionFields(props, [
    'id',
    'observed',
    'simulations',
    'reference',
    'integrationWeights',
    'curveRowCount',
    'simulationCount',
    'rank',
    'statistic',
    'significanceLevel',
    'nullModel',
    'fittedParameters',
    'output'
  ] as readonly (keyof T)[]);
  return Object.freeze({
    ...captured,
    output: snapshotConstructionFields(captured.output, [
      'lower',
      'upper',
      'statistics',
      'status',
      'pValueInterval'
    ])
  }) as T;
}

/** Resolve an exact replicate-major curve count without discarding trailing ordinates. @internal */
export function resolveEnvelopeSimulationCount(
  id: string,
  observedRows: number,
  simulationRows: number,
  explicitCount?: number
): number {
  const inferred = simulationRows / observedRows;
  validateRequirement(
    id,
    observedRows > 0 && Number.isInteger(inferred) && inferred >= 2,
    `simulations.length ${simulationRows} must be an exact nonempty multiple (at least 2) of observed.length ${observedRows}`
  );
  validateRequirement(
    id,
    explicitCount === undefined || explicitCount === inferred,
    `simulationCount ${explicitCount} must equal simulations.length / observed.length (${inferred})`
  );
  return explicitCount ?? inferred;
}

/** Shared default and accepted statistic values for both backends and operation forms. @internal */
export function resolveEnvelopeStatistic(
  id: string,
  statistic:
    | 'maximum-deviation'
    | 'integrated-squared-deviation'
    | 'extreme-rank' = 'maximum-deviation'
) {
  validateRequirement(
    id,
    ['maximum-deviation', 'integrated-squared-deviation', 'extreme-rank'].includes(statistic),
    `statistic must be maximum-deviation, integrated-squared-deviation or extreme-rank; received ${statistic}`
  );
  return statistic;
}

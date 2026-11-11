// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {makeGPURasterMetadata} from '@luma-contributors/analysis/raster-analysis';
import type {RasterFractionalZonalDistributionOperationProps} from '../../../src/raster-analysis/raster-zonal-statistics/gpu-raster-fractional-zonal-distribution-operation';
import type {CPURasterFractionalZonalDistributionInput} from '../../../src/raster-analysis/raster-zonal-statistics/cpu-raster-fractional-zonal-distribution-operation';
import type {OperationGraphProps} from '../../utils/mixed-execution-test-utils';

export const DISTRIBUTION_PROPS: RasterFractionalZonalDistributionOperationProps = {
  id: 'distribution',
  grid: makeGPURasterMetadata({width: 3, height: 1}),
  input: {
    id: 'band',
    format: 'float32',
    storage: {kind: 'buffer', values: 'values'},
    validity: 'inputValidity'
  },
  zoneCount: 1,
  probabilityCount: 6,
  observationWeights: 'observationWeights',
  cellAreas: 'cellAreas',
  parameters: 'parameters',
  coverage: {
    zoneIds: 'zoneIds',
    cellIds: 'cellIds',
    fractions: 'fractions',
    zoneValidity: 'zoneValidity',
    status: {count: 'count', requiredCount: 'requiredCount', overflow: 'overflow'}
  },
  probabilities: 'probabilities',
  output: {
    means: 'means',
    populationVariances: 'populationVariances',
    sampleVariances: 'sampleVariances',
    effectiveSampleSizes: 'effectiveSampleSizes',
    weightSums: 'weightSums',
    squaredWeightSums: 'squaredWeightSums',
    coveredArea: 'coveredArea',
    validArea: 'validArea',
    quantiles: 'quantiles',
    quantileValidity: 'quantileValidity',
    validity: 'validity',
    status: 'status'
  }
};

export function createDistributionInput(empty = false): CPURasterFractionalZonalDistributionInput {
  return {
    input: {
      id: 'band',
      format: 'float32',
      values: new Float32Array([0, 10, 20]),
      validity: new Uint32Array([1, 1, 1])
    },
    coverage: {
      zoneIds: new Uint32Array([0, 0, 0]),
      cellIds: new Uint32Array([0, 1, 2]),
      fractions: new Float32Array([0.5, 1, 0.5]),
      zoneValidity: new Uint32Array([1]),
      count: new Uint32Array([empty ? 0 : 3]),
      requiredCount: new Uint32Array([empty ? 0 : 3]),
      overflow: new Uint32Array([0])
    },
    observationWeights: new Float32Array([2, 2, 2]),
    cellAreas: new Float32Array([2, 4, 6]),
    parameters: new Float32Array([0, 0, 1, 0, 0, -1, -1, 0]),
    probabilities: new Float32Array([0, 0.25, 0.5, 0.75, 1, NaN])
  };
}

export function createDistributionGraphInputs(empty = false): OperationGraphProps['inputs'] {
  const input = createDistributionInput(empty);
  return getDistributionGraphInputs(input);
}

/** Unit coverage/support rows for the hand-derived nodata, singleton and asymmetric-weight cases. */
export function createUnitDistributionInput(
  values: Float32Array,
  probabilities: Float32Array
): CPURasterFractionalZonalDistributionInput {
  const count = values.length;
  return {
    input: {id: 'band', format: 'float32', values, validity: new Uint32Array(count).fill(1)},
    coverage: {
      zoneIds: new Uint32Array(count),
      cellIds: Uint32Array.from({length: count}, (_, cell) => cell),
      fractions: new Float32Array(count).fill(1),
      zoneValidity: new Uint32Array([1]),
      count: new Uint32Array([count]),
      requiredCount: new Uint32Array([count]),
      overflow: new Uint32Array([0])
    },
    observationWeights: new Float32Array(count).fill(1),
    cellAreas: new Float32Array(count).fill(1),
    parameters: new Float32Array([0, 0, 1, 0, 0, -1, -1, 0]),
    probabilities
  };
}

/** Converts the scalar rows to the named ports shared by both distribution operations. */
export function getDistributionGraphInputs(
  input: CPURasterFractionalZonalDistributionInput
): OperationGraphProps['inputs'] {
  return {
    values: {format: 'float32', values: input.input.values},
    inputValidity: {format: 'uint32', values: input.input.validity!},
    zoneIds: {format: 'uint32', values: input.coverage.zoneIds},
    cellIds: {format: 'uint32', values: input.coverage.cellIds},
    fractions: {format: 'float32', values: input.coverage.fractions},
    zoneValidity: {format: 'uint32', values: input.coverage.zoneValidity},
    count: {format: 'uint32', values: input.coverage.count},
    requiredCount: {format: 'uint32', values: input.coverage.requiredCount},
    overflow: {format: 'uint32', values: input.coverage.overflow},
    observationWeights: {format: 'float32', values: input.observationWeights!},
    cellAreas: {format: 'float32', values: input.cellAreas!},
    parameters: {format: 'float32', values: input.parameters!},
    probabilities: {format: 'float32', values: input.probabilities}
  };
}

// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {RasterWalkingCostOperationProps} from '../../src/raster-analysis/cost-distance/gpu-raster-walking-cost-operation';
import type {CPURasterWalkingCostInput} from '../../src/raster-analysis/cost-distance/cpu-raster-walking-cost-operation';
import {getGPURasterWalkingCostParameterValues} from '../../src/raster-analysis/gpu-raster-walking-cost';
import type {OperationGraphInput} from '../utils/mixed-execution-test-utils';
import fixtures from './raster-walking-cost-operation-fixtures.json';

export function createWalkingProps(kind: 'grass' | 'tobler'): RasterWalkingCostOperationProps {
  return {
    id: 'walking',
    grid: {
      width: fixtures.width,
      height: fixtures.height,
      affine: fixtures.affine as [number, number, number, number, number, number],
      pixelInterpretation: 'area'
    },
    model:
      kind === 'grass'
        ? {kind, coefficients: [2, 3, -0.5, -2], downhillSlopeThreshold: -0.25}
        : {kind},
    maximumIterations: 64,
    elevation: {
      id: 'elevation',
      format: 'float32',
      storage: {kind: 'buffer', values: 'elevationValues'},
      validity: 'elevationValidity'
    },
    friction: {
      id: 'friction',
      format: 'float32',
      storage: {kind: 'buffer', values: 'frictionValues'},
      validity: 'frictionValidity'
    },
    sources: 'sources',
    parameters: 'parameters',
    output: {
      costs: 'costs',
      validity: 'validity',
      status: {
        converged: 'converged',
        iterationCount: 'rounds',
        iterationLimitReached: 'limit',
        invalidCount: 'invalid'
      }
    }
  };
}

export function createWalkingInput(
  props: RasterWalkingCostOperationProps
): CPURasterWalkingCostInput {
  const sources = new Uint32Array(fixtures.width * fixtures.height);
  for (const cell of fixtures.sources) sources[cell] = 1;
  return {
    elevation: {
      id: 'elevation',
      format: 'float32',
      values: Float32Array.from(fixtures.elevation),
      validity: new Uint32Array(sources.length).fill(1)
    },
    friction: {
      id: 'friction',
      format: 'float32',
      values: Float32Array.from(fixtures.friction),
      validity: new Uint32Array(sources.length).fill(1)
    },
    parameters: getGPURasterWalkingCostParameterValues(
      {model: props.model, frictionWeight: 0.75},
      props.grid
    ),
    sources
  };
}

export function createWalkingGraphInputs(
  input: CPURasterWalkingCostInput
): Record<string, OperationGraphInput> {
  return {
    elevationValues: {format: input.elevation.format, values: input.elevation.values},
    elevationValidity: {format: 'uint32', values: input.elevation.validity!},
    frictionValues: {format: input.friction.format, values: input.friction.values},
    frictionValidity: {format: 'uint32', values: input.friction.validity!},
    sources: {format: 'uint32', values: input.sources},
    parameters: {format: 'float32', values: input.parameters}
  };
}

export function createSingletonWalkingProps(): RasterWalkingCostOperationProps {
  return {
    ...createWalkingProps('grass'),
    grid: {width: 1, height: 1, affine: [1, 0, 0, 0, -1, 1], pixelInterpretation: 'area'}
  };
}

export function createSingletonWalkingInput(
  sources: Uint32Array = Uint32Array.of(1, 0)
): CPURasterWalkingCostInput {
  const props = createSingletonWalkingProps();
  return {
    elevation: {
      id: 'elevation',
      format: 'float32',
      values: Float32Array.of(2),
      validity: Uint32Array.of(1)
    },
    friction: {
      id: 'friction',
      format: 'float32',
      values: Float32Array.of(3),
      validity: Uint32Array.of(1)
    },
    sources,
    parameters: getGPURasterWalkingCostParameterValues({model: props.model}, props.grid)
  };
}

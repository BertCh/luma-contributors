// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {StreamlinesOperationProps} from '../../../src/raster-analysis/flow-texture/gpu-streamlines-operation';
import type {ProfilePeaksOperationProps} from '../../../src/gpu-terrain/terrain-features/gpu-profile-peaks-operation';
import {getGPUProfilePeaksParameterValues} from '../../../src/gpu-terrain/terrain-features/gpu-profile-peaks';
import {
  getGPUStreamlinesParameterValues,
  getGPUStreamlinesUint32ParameterValues
} from '../../../src/raster-analysis/flow-texture/streamlines-parameters';

export const COMPACT_STATUS_OUTPUTS = ['count', 'overflow', 'candidateOverflow'];
const status = {count: 'count', overflow: 'overflow', candidateOverflow: 'candidateOverflow'};

export function getCompactStreamlinesFixture(maximumIterations: number) {
  const props: StreamlinesOperationProps = {
    fieldWidth: 16,
    fieldHeight: 1,
    gridWidth: 16,
    gridHeight: 1,
    seedColumns: 3,
    seedRows: 1,
    stepsPerDirection: 2,
    maximumIterations,
    lineCapacity: 3,
    pointCapacity: 15,
    velocities: 'velocities',
    parameters: 'parameters',
    uint32Parameters: 'words',
    output: {
      lines: {ids: 'ids', status},
      positions: 'positions',
      lineOffsets: 'offsets',
      positionCount: 'positionCount'
    }
  };
  const settings = {
    fieldPlacement: {origin: [0, 0], cellSize: [1, 1]} as const,
    gridPlacement: {origin: [0, 0], cellSize: [1, 1]} as const,
    stepLength: 1,
    seed: 3,
    minimumPoints: 2
  };
  return {
    props,
    inputs: {
      velocities: {
        format: 'float32x2' as const,
        values: Float32Array.from({length: 32}, (_, component) => (component % 2 === 0 ? 1 : 0))
      },
      parameters: {format: 'float32' as const, values: getGPUStreamlinesParameterValues(settings)},
      words: {format: 'uint32' as const, values: getGPUStreamlinesUint32ParameterValues(settings)}
    }
  };
}

export function getCompactPeaksFixture() {
  const props: ProfilePeaksOperationProps = {
    values: 'values',
    offsets: 'offsets',
    parameters: 'parameters',
    sampleCount: 3,
    window: 1,
    minimumSide: 1,
    suppressionRadius: 0,
    maximumSuppressionIterations: 0,
    output: {peaks: {ids: 'ids', capacity: 3, status}}
  };
  return {
    props,
    inputs: {
      values: {format: 'float32' as const, values: Float32Array.of(0, 1, 0)},
      offsets: {format: 'uint32' as const, values: Uint32Array.of(0, 3)},
      parameters: {
        format: 'float32' as const,
        values: getGPUProfilePeaksParameterValues({minimumProminence: 0})
      }
    }
  };
}

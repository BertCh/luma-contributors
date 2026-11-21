// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {expect, test} from 'vitest';
import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import type {
  GPUContributorOperation,
  GPUContributorOperationViewGetter
} from '../../src/analysis-core/operations/gpu-contributor-operation';
import {CPUProfilePeaksOperation} from '../../src/gpu-terrain/terrain-features/cpu-profile-peaks-operation';
import {
  GPUProfilePeaksOperation,
  type ProfilePeaksOperationProps
} from '../../src/gpu-terrain/terrain-features/gpu-profile-peaks-operation';
import {CPUStreamlinesOperation} from '../../src/raster-analysis/flow-texture/cpu-streamlines-operation';
import {
  GPUStreamlinesOperation,
  type StreamlinesOperationProps
} from '../../src/raster-analysis/flow-texture/gpu-streamlines-operation';
import {getGPUProfilePeaksParameterValues} from '../../src/gpu-terrain/terrain-features/gpu-profile-peaks';
import {
  getGPUStreamlinesParameterValues,
  getGPUStreamlinesUint32ParameterValues
} from '../../src/raster-analysis/flow-texture/streamlines-parameters';
import {createVortexField} from '../gpu-raster/flow-texture/flow-texture-scenes';
import {createNullWebGPUDevice} from './gpu-contributor-test-utils';

function rows(
  values: Float32Array | Uint32Array,
  format: 'uint32' | 'float32' | 'float32x2' = values instanceof Uint32Array ? 'uint32' : 'float32'
) {
  const length = values.length / (format === 'float32x2' ? 2 : 1);
  return {format, length, chunks: [{values, length}], pack: () => values};
}
function lower(
  operation: GPUContributorOperation,
  graph: GPUCommandGraph,
  inputs: Record<string, {format: 'float32' | 'float32x2' | 'uint32'; length: number}>
) {
  const views = new Map();
  return operation.createContributor(((port: string) => {
    if (!views.has(port)) {
      const declaration = operation.outputs[port] ?? inputs[port];
      views.set(
        port,
        createTransientView(
          graph,
          `${operation.id}-${port}`,
          declaration.format,
          declaration.length
        )
      );
    }
    return views.get(port);
  }) as GPUContributorOperationViewGetter);
}

test.each([
  false,
  true
])('final2 G6 peak operations capture literal/inherited zero budget, inherited=%s', inherited => {
  const values = Float32Array.from({length: 80}, (_, row) =>
    [
      [30, 10],
      [38, 8],
      [46, 6]
    ].reduce(
      (sum, [center, height]) => sum + height * Math.exp(-0.5 * ((row - center) / 2) ** 2),
      0
    )
  );
  let budget = 0;
  const props: ProfilePeaksOperationProps = Object.assign(
    inherited
      ? Object.create({
          get maximumSuppressionIterations() {
            return budget;
          }
        })
      : {maximumSuppressionIterations: 0},
    {
      values: 'values',
      offsets: 'offsets',
      parameters: 'parameters',
      sampleCount: 80,
      window: 12,
      suppressionRadius: 10,
      output: {
        peaks: {
          ids: 'ids',
          capacity: 80,
          status: {count: 'count', overflow: 'overflow', candidateOverflow: 'candidate'}
        },
        status: {iterationLimitReached: 'limit'}
      }
    }
  );
  const cpu = new CPUProfilePeaksOperation(props),
    gpu = new GPUProfilePeaksOperation(props);
  const inputs = {
    values: rows(values),
    offsets: rows(new Uint32Array([0, 80])),
    parameters: rows(getGPUProfilePeaksParameterValues({minimumProminence: 1}))
  };
  const run = () => cpu.execute({inputs}) as Record<string, Uint32Array>;
  expect([run()['count'][0], run()['candidate'][0], run()['limit'][0]]).toEqual([0, 1, 1]);
  budget = 16;
  if (!inherited) props.maximumSuppressionIterations = 16;
  expect([run()['count'][0], run()['candidate'][0], run()['limit'][0]]).toEqual([0, 1, 1]);
  const device = createNullWebGPUDevice();
  try {
    expect(
      (
        lower(gpu, new GPUCommandGraph(device), inputs) as unknown as {
          maximumSuppressionIterations: number;
        }
      ).maximumSuppressionIterations
    ).toBe(0);
  } finally {
    device.destroy();
  }
});

test.each([
  false,
  true
])('final2 G6 streamline operations retain one vortex round after mutation, inherited=%s', inherited => {
  const field = createVortexField(32, 0.1);
  let budget = 1;
  const props: StreamlinesOperationProps = Object.assign(
    inherited
      ? Object.create({
          get maximumIterations() {
            return budget;
          }
        })
      : {maximumIterations: 1},
    {
      fieldWidth: 32,
      fieldHeight: 32,
      gridWidth: 48,
      gridHeight: 48,
      seedColumns: 24,
      seedRows: 24,
      stepsPerDirection: 40,
      lineCapacity: 576,
      pointCapacity: 576 * 81,
      velocities: 'velocities',
      parameters: 'parameters',
      uint32Parameters: 'uint32Parameters',
      output: {
        lines: {
          ids: 'ids',
          status: {count: 'count', overflow: 'overflow', candidateOverflow: 'candidate'}
        },
        lineOffsets: 'offsets',
        positions: 'positions',
        positionCount: 'pointCount',
        status: {iterationLimitReached: 'limit'}
      }
    }
  );
  const cpu = new CPUStreamlinesOperation(props),
    gpu = new GPUStreamlinesOperation(props);
  const settings = {
    fieldPlacement: {origin: [0, 0], cellSize: [1, 1]} as const,
    gridPlacement: {origin: [0, 0], cellSize: [32 / 48, 32 / 48]} as const,
    stepLength: 0.3,
    minimumSpeed: 0.05,
    seed: 5,
    minimumPoints: 6
  };
  const inputs = {
    velocities: rows(field.velocities, 'float32x2'),
    parameters: rows(getGPUStreamlinesParameterValues(settings)),
    uint32Parameters: rows(getGPUStreamlinesUint32ParameterValues(settings))
  };
  const run = () => cpu.execute({inputs}) as Record<string, Uint32Array>;
  expect([run()['count'][0], run()['candidate'][0]]).toEqual([27, 1]);
  budget = 32;
  if (!inherited) props.maximumIterations = 32;
  expect([run()['count'][0], run()['candidate'][0]]).toEqual([27, 1]);
  const device = createNullWebGPUDevice();
  try {
    expect(
      (lower(gpu, new GPUCommandGraph(device), inputs) as unknown as {maximumIterations: number})
        .maximumIterations
    ).toBe(1);
  } finally {
    device.destroy();
  }
});

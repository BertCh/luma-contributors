// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {expect, it} from 'vitest';
import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import {GPUSpatialCrossValidation} from '../../../src/gpu-spatial-analysis/spatial-validation/gpu-spatial-cross-validation';
import {computeSpatialCrossValidationOnCPU} from '../../../src/gpu-spatial-analysis/spatial-validation/cpu-spatial-cross-validation-operation';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {executeSpatialAuditKernel} from './spatial-design-audit.test-utils';
import reference from '../spatial-regression/spatial-validation-advanced-reference.json';
import {GPUSpatialCrossValidationOperation} from '../../../src/gpu-spatial-analysis/spatial-validation/gpu-spatial-cross-validation-operation';
import {CPUSpatialCrossValidationOperation} from '../../../src/gpu-spatial-analysis/spatial-validation/cpu-spatial-cross-validation-operation';

it.each([
  2 ** -13,
  2 ** -27
])('B4 chooses the geometrically nearest defining neighbor across rounded ties (%s)', vertical => {
  const positions = Float32Array.of(0, 0, 1, vertical, 1, 0);
  const cpu = computeSpatialCrossValidationOnCPU(
    {positions, blocks: new Uint32Array(3), predictionPositions: Float32Array.of(0, 0)},
    {foldCount: 3, bufferDistance: 0, design: {mode: 'nndm', maximumDistance: 0}}
  );
  // From row 0: row 2 has d²=1 and row 1 has d²=1+vertical².
  expect(Array.from(cpu.training.slice(0, 3))).toEqual([0, 1, 0]);
  expect(cpu.status[0]).toBe(0);
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends 'uint32' | 'float32x2'>(format: Format, length: number) =>
    createTransientView(graph, `nearest-tie-${serial++}`, format, length);
  try {
    const contributor = new GPUSpatialCrossValidation({
      positions: view('float32x2', 3),
      blocks: view('uint32', 3),
      foldCount: 3,
      bufferDistance: 0,
      design: {mode: 'nndm', predictionPositions: view('float32x2', 1), maximumDistance: 0},
      output: {
        folds: view('uint32', 3),
        training: view('uint32', 9),
        testing: view('uint32', 9),
        counts: view('uint32', 9),
        status: view('uint32', 3)
      }
    });
    const state = new Float32Array(7);
    const values = {
      positions,
      prediction: Float32Array.of(0, 0),
      folds: new Uint32Array(3),
      state,
      designState: state,
      distances: new Float32Array(9),
      adaptiveNeighbors: new Uint32Array(3),
      training: new Uint32Array(9),
      testing: new Uint32Array(9),
      counts: new Uint32Array(9),
      status: new Uint32Array(3)
    };
    const kernels = getGPUCommandNodeKernelSources(contributor.getCommandNodes(graph));
    for (const kernel of kernels.filter(
      kernel => kernel.id.endsWith('-design') || kernel.id.endsWith('-adaptive-neighbors')
    ))
      executeSpatialAuditKernel(kernel, values);
    expect(Array.from(values.adaptiveNeighbors)).toEqual([2, 2, 1]);
    for (const kernel of kernels.filter(
      kernel => kernel.id.endsWith('-masks') || kernel.id.endsWith('-counts')
    ))
      executeSpatialAuditKernel(kernel, values, {
        invocationCount: kernel.id.endsWith('-masks') ? 3 : 1
      });
    expect(Array.from(values.training.slice(0, 3))).toEqual([0, 1, 0]);
    expect(values.status[0]).toBe(0);
    // Once the closer row is deleted, recovery must choose the remaining farther row.
    values.distances[2] = -1;
    for (const kernel of kernels.filter(kernel => kernel.id.endsWith('-adaptive-neighbors')))
      executeSpatialAuditKernel(kernel, values);
    expect(values.adaptiveNeighbors[0]).toBe(1);
  } finally {
    device.destroy();
  }
});

it('NOTE rejects unknown GPU design modes during construction with accepted values', () => {
  const device = createNullWebGPUDevice(),
    graph = new GPUCommandGraph(device);
  const positions = createTransientView(graph, 'bad-mode-positions', 'float32x2', 2),
    blocks = createTransientView(graph, 'bad-mode-blocks', 'uint32', 2);
  const view = (name: string, length: number) => createTransientView(graph, name, 'uint32', length);
  try {
    expect(
      () =>
        new GPUSpatialCrossValidation({
          positions,
          blocks,
          foldCount: 2,
          design: {mode: 'typo'} as never,
          output: {
            folds: view('folds', 2),
            training: view('training', 4),
            testing: view('testing', 4),
            counts: view('counts', 6),
            status: view('status', 2)
          }
        })
    ).toThrow(/groups.*environmental.*nndm.*knndm/);
    const named = {
      rowCount: 2,
      positions: 'positions',
      blocks: 'blocks',
      foldCount: 2,
      design: {mode: 'typo'} as never,
      output: {
        folds: 'folds',
        training: 'training',
        testing: 'testing',
        counts: 'counts',
        status: 'status'
      }
    };
    expect(() => new GPUSpatialCrossValidationOperation(named)).toThrow(
      /groups.*environmental.*nndm.*knndm/
    );
    expect(() => new CPUSpatialCrossValidationOperation(named)).toThrow(
      /groups.*environmental.*nndm.*knndm/
    );
  } finally {
    device.destroy();
  }
});

it.each([
  [1, 1],
  [3, 8]
])('B1 excludes the defining adaptive neighbor (%s,%s)', (horizontal, vertical) => {
  const positions = Float32Array.of(0, 0, horizontal, vertical);
  const cpu = computeSpatialCrossValidationOnCPU(
    {positions, blocks: Uint32Array.of(0, 1), predictionPositions: Float32Array.of(0, 0)},
    {foldCount: 2, design: {mode: 'nndm', maximumDistance: 0}}
  );
  // The sole neighbor defines each exclusion disk, so it lies on the inclusive boundary.
  expect.soft(Array.from(cpu.training)).toEqual([0, 0, 0, 0]);
  const device = createNullWebGPUDevice(),
    graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends 'uint32' | 'float32x2'>(format: Format, length: number) =>
    createTransientView(graph, `adaptive-${serial++}`, format, length);
  try {
    const contributor = new GPUSpatialCrossValidation({
      positions: view('float32x2', 2),
      blocks: view('uint32', 2),
      foldCount: 2,
      design: {mode: 'nndm', maximumDistance: 0, predictionPositions: view('float32x2', 1)},
      output: {
        folds: view('uint32', 2),
        training: view('uint32', 4),
        testing: view('uint32', 4),
        counts: view('uint32', 6),
        status: view('uint32', 2)
      }
    });
    const kernels = getGPUCommandNodeKernelSources(contributor.getCommandNodes(graph));
    const radius = Math.fround(Math.hypot(horizontal, vertical));
    const values = {
      positions,
      folds: Uint32Array.of(0, 1),
      designState: Float32Array.of(0, 0, NaN, NaN, radius, radius),
      distances: Float32Array.of(-1, radius, radius, -1),
      adaptiveNeighbors: new Uint32Array(2),
      training: new Uint32Array(4),
      testing: new Uint32Array(4)
    };
    for (const kernel of kernels.filter(
      kernel => kernel.id.endsWith('-adaptive-neighbors') || kernel.id.endsWith('-masks')
    ))
      executeSpatialAuditKernel(kernel, values);
    expect(Array.from(values.training)).toEqual([0, 0, 0, 0]);
  } finally {
    device.destroy();
  }
});

it('TWN-05/B1 retains the independently recorded blockCV NNDM masks', () => {
  const result = computeSpatialCrossValidationOnCPU(
    {
      positions: Float32Array.from(reference.positions),
      blocks: Uint32Array.from(reference.groups),
      predictionPositions: Float32Array.from(reference.predictionPositions)
    },
    {foldCount: 8, design: {mode: 'nndm', maximumDistance: 4, minimumTrainingFraction: 0.2}}
  );
  expect(Array.from(result.training)).toEqual(reference.nndm.training);
});

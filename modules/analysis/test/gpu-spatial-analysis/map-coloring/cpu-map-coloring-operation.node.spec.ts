// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {
  computeMapColoringOnCPU,
  CPUMapColoringOperation
} from '../../../src/gpu-spatial-analysis/map-coloring/cpu-map-coloring-operation';
import {
  GPUMapColoringOperation,
  type MapColoringOperationProps
} from '../../../src/gpu-spatial-analysis/map-coloring/gpu-map-coloring-operation';
import {getGPUMapColoringParameterValues} from '../../../src/gpu-spatial-analysis/map-coloring/gpu-map-coloring';
import {
  createCliqueCSR,
  createRookLatticeCSR,
  createStarCSR,
  simulateMapColoringRounds
} from './map-coloring-oracle';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {createOperationTestViewGetter} from '../../utils/statistics-dynamics-operation-test-utils';

const PROPS: MapColoringOperationProps = {
  id: 'color-check',
  weights: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
  parameters: 'parameters',
  rowCount: 3,
  output: {colors: 'colors'}
};

it('B3 snapshots inherited loss names once in both operation forms', () => {
  for (const Operation of [CPUMapColoringOperation, GPUMapColoringOperation]) {
    let reads = 0;
    const status = {overflow: 'loss'};
    const weights = Object.create(PROPS.weights, {
      status: {
        get() {
          reads++;
          return status;
        }
      }
    });
    const operation = new Operation({...PROPS, weights});
    status.overflow = 'changed-loss';
    expect(reads).toBe(1);
    expect(operation.inputs['overflow']).toBe('loss');
  }
});

it('sequential CPU coloring agrees with independent synchronous Jones-Plassmann rounds', () => {
  for (const pattern of [createCliqueCSR(35), createRookLatticeCSR(6, 5), createStarCSR(12)]) {
    for (const seed of [0, 17, 0xffffffff]) {
      for (const priority of ['random', 'largest-degree-first'] as const) {
        const expected = simulateMapColoringRounds(pattern, seed, priority).colors;
        const result = computeMapColoringOnCPU(
          Uint32Array.from(pattern.offsets),
          Uint32Array.from(pattern.neighbors),
          getGPUMapColoringParameterValues({seed}),
          priority
        );
        expect(Array.from(result.colors)).toEqual(expected);
        expect(result.colorCount[0]).toBe(Math.max(...expected) + 1);
        expect(result.conflictCount[0]).toBe(0);
        expect(Array.from(result.converged)).toEqual([1]);
        expect(Array.from(result.iterationCount)).toEqual([1]);
      }
    }
  }
});

it('handles isolates, self slots and duplicate neighbors and rejects malformed arrays', () => {
  const result = computeMapColoringOnCPU(
    Uint32Array.of(0, 3, 5, 5),
    Uint32Array.of(0, 1, 1, 0, 0),
    Uint32Array.of(0)
  );
  expect(Array.from(result.colors)).toEqual([0, 1, 0]);
  expect(result.conflictCount[0]).toBe(0);
  expect(() =>
    computeMapColoringOnCPU(Uint32Array.of(0, 2), Uint32Array.of(0), Uint32Array.of(0))
  ).toThrow(/offsets/);
  expect(() =>
    computeMapColoringOnCPU(Uint32Array.of(0, 1), Uint32Array.of(1), Uint32Array.of(0))
  ).toThrow(/neighbors/);
  expect(() =>
    computeMapColoringOnCPU(Uint32Array.of(0, 0), new Uint32Array(0), new Uint32Array(0))
  ).toThrow(/parameters/);
});

it('validates CPU props and GPU contributor construction without imposing GPU round limits on CPU', () => {
  for (const Operation of [CPUMapColoringOperation, GPUMapColoringOperation]) {
    expect(() => new Operation({...PROPS, output: {...PROPS.output, colors: 'neighbors'}})).toThrow(
      /^color-check outputs.colors/
    );
  }
  expect(() => new CPUMapColoringOperation({...PROPS, rowCount: 0})).toThrow(
    /^color-check rowCount/
  );
  expect(() => new GPUMapColoringOperation({...PROPS, rowCount: 0})).toThrow(
    /^color-check outputs.colors.length/
  );
  expect(() => new CPUMapColoringOperation({...PROPS, maximumIterations: 0})).toThrow(
    /^color-check maximumIterations/
  );
  expect(() => new CPUMapColoringOperation({...PROPS, maximumIterations: 2048})).not.toThrow();
  for (const Operation of [CPUMapColoringOperation, GPUMapColoringOperation]) {
    const device = createNullWebGPUDevice();
    try {
      const props = {...PROPS, priority: 'invalid' as 'random'};
      if (Operation === CPUMapColoringOperation) {
        expect(() => new CPUMapColoringOperation(props)).toThrow(/^color-check priority/);
      } else {
        const graph = new GPUCommandGraph(device);
        const getView = createOperationTestViewGetter(graph, {
          offsets: {format: 'uint32', length: 4},
          neighbors: {format: 'uint32', length: 2},
          weights: {format: 'float32', length: 2},
          parameters: {format: 'uint32', length: 1},
          colors: {format: 'uint32', length: 3}
        });
        expect(() => new GPUMapColoringOperation(props).createContributor(getView)).toThrow(
          /^color-check priority/
        );
        expect(() =>
          new GPUMapColoringOperation({...PROPS, maximumIterations: 1025}).createContributor(
            getView
          )
        ).toThrow(/^color-check maximumIterations/);
      }
    } finally {
      device.destroy();
    }
  }
});

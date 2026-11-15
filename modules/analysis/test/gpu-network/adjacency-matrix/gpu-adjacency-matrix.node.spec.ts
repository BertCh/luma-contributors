// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  computeAdjacencyMatrixOrder,
  getGPUAdjacencyMatrixParameterValues,
  getGPUAdjacencyMatrixFixedWeight,
  GPUAdjacencyMatrix,
  GPUAdjacencyMatrixOrder,
  type GPUAdjacencyMatrixProps
} from '../../../src/gpu-network/adjacency-matrix/index';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

function createContext(device: Device) {
  const graph = new GPUCommandGraph(device);
  const view = <F extends 'uint32' | 'float32'>(name: string, format: F, length: number) => {
    const buffer = device.createBuffer({
      id: `${name}-${Math.random()}`,
      byteLength: Math.max(length, 1) * 4,
      usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
    });
    return importGraphBuffer(graph, `${name}-${buffer.id}`, buffer, format, length);
  };
  const props = (overrides: Partial<GPUAdjacencyMatrixProps> = {}): GPUAdjacencyMatrixProps => ({
    topology: {offsets: view('offsets', 'uint32', 7), neighbors: view('neighbors', 'uint32', 10)},
    resolution: 4,
    output: {counts: view('counts', 'uint32', 16)},
    ...overrides
  });
  return {graph, view, props};
}

it('GPUAdjacencyMatrix validates shapes and options', () => {
  const device = createNullWebGPUDevice();
  const {view, props} = createContext(device);
  expect(() => new GPUAdjacencyMatrix(props())).not.toThrow();
  expect(() => new GPUAdjacencyMatrix(props({resolution: 0}))).toThrow(/resolution/);
  expect(() => new GPUAdjacencyMatrix(props({resolution: 4097}))).toThrow(/resolution/);
  expect(() => new GPUAdjacencyMatrix(props({output: {counts: view('c', 'uint32', 15)}}))).toThrow(
    /counts/
  );
  expect(() => new GPUAdjacencyMatrix(props({order: view('order', 'uint32', 5)}))).toThrow(/order/);
  expect(() => new GPUAdjacencyMatrix(props({edgeMask: view('mask', 'uint32', 9)}))).toThrow(
    /edgeMask/
  );
  expect(
    () => new GPUAdjacencyMatrix(props({parameters: view('parameters', 'uint32', 3)}))
  ).toThrow(/parameters/);
  expect(() => new GPUAdjacencyMatrix(props({weightScale: 0}))).toThrow(/weightScale/);
  expect(
    () =>
      new GPUAdjacencyMatrix(
        props({
          output: {
            counts: view('c2', 'uint32', 16),
            weightSums: view('w', 'uint32', 16)
          }
        })
      )
  ).toThrow(/require weights/);
  expect(
    () =>
      new GPUAdjacencyMatrix(
        props({
          topology: {
            offsets: view('offsets', 'uint32', 7),
            neighbors: view('neighbors', 'uint32', 10),
            weights: view('weights', 'float32', 10)
          },
          output: {
            counts: view('c3', 'uint32', 16),
            maximumWeightSum: view('m', 'uint32', 1)
          }
        })
      )
  ).toThrow(/weightSums/);
});

it('GPUAdjacencyMatrix validates its topology', () => {
  const device = createNullWebGPUDevice();
  const {view, props} = createContext(device);
  expect(
    () =>
      new GPUAdjacencyMatrix(
        props({
          topology: {
            offsets: view('offsets', 'uint32', 7),
            neighbors: view('neighbors', 'uint32', 10),
            weights: view('weights', 'float32', 9)
          }
        })
      )
  ).toThrow(/topology\.weights length/);
  expect(
    () =>
      new GPUAdjacencyMatrix(
        props({
          topology: {offsets: view('offsets', 'uint32', 1), neighbors: view('n', 'uint32', 2)}
        })
      )
  ).toThrow(/topology\.offsets/);
});

it('GPUAdjacencyMatrix returns deterministic node ids', () => {
  const device = createNullWebGPUDevice();
  const {graph, view, props} = createContext(device);
  const contributor = new GPUAdjacencyMatrix(
    props({
      topology: {
        offsets: view('offsets', 'uint32', 7),
        neighbors: view('neighbors', 'uint32', 10),
        weights: view('weights', 'float32', 10)
      },
      output: {
        counts: view('c', 'uint32', 16),
        weightSums: view('w', 'uint32', 16),
        maximumCount: view('mc', 'uint32', 1),
        maximumWeightSum: view('mw', 'uint32', 1)
      }
    })
  );
  const ids = contributor.getCommandNodes(graph).map(node => node.id);
  expect(ids).toEqual([
    'adjacency-matrix-zero-counts',
    'adjacency-matrix-zero-weight-sums',
    'adjacency-matrix-zero-max-count',
    'adjacency-matrix-zero-max-weight-sum',
    // Counts and weight sums share one binning kernel while its bindings fit the device limit.
    'adjacency-matrix-bin',
    'adjacency-matrix-maxima'
  ]);
});

it('parameters packing and fixed-point weights follow their documented rules', () => {
  expect(
    Array.from(
      getGPUAdjacencyMatrixParameterValues({
        rowStart: 1,
        rowEnd: 9,
        colStart: 2,
        colEnd: 5,
        resolution: 4
      })
    )
  ).toEqual([1, 9, 2, 5]);
  expect(() =>
    getGPUAdjacencyMatrixParameterValues({
      rowStart: 0,
      rowEnd: 2 ** 31,
      colStart: 0,
      colEnd: 1,
      resolution: 4
    })
  ).toThrow(/2\^32/);
  expect(() =>
    getGPUAdjacencyMatrixParameterValues({
      rowStart: -1,
      rowEnd: 2,
      colStart: 0,
      colEnd: 1,
      resolution: 4
    })
  ).toThrow(/uint32/);
  expect(getGPUAdjacencyMatrixFixedWeight(1.5, 1024)).toBe(1536);
  expect(getGPUAdjacencyMatrixFixedWeight(-3, 1024)).toBe(0);
  expect(getGPUAdjacencyMatrixFixedWeight(NaN, 1024)).toBe(0);
  expect(getGPUAdjacencyMatrixFixedWeight(1e30, 1024)).toBe(0xffffffff);
  // Halves round up, and an odd integer product above 2^23 is kept (8192.0009765625 * 1024).
  expect(getGPUAdjacencyMatrixFixedWeight(1.5 / 1024, 1024)).toBe(2);
  expect(getGPUAdjacencyMatrixFixedWeight(2.5 / 1024, 1024)).toBe(3);
  expect(getGPUAdjacencyMatrixFixedWeight(8192.0009765625, 1024)).toBe(8388609);
  // The product is a float32, as on the GPU: 16.0035 is stored as 16.00349998474121, whose
  // product with 1000 rounds to the float32 16003.5, and that rounds half up.
  expect(getGPUAdjacencyMatrixFixedWeight(16.0035, 1000)).toBe(16004);
  // The largest float32 below 2^32 saturates, as in the kernel.
  expect(getGPUAdjacencyMatrixFixedWeight(4294967040 / 1024, 1024)).toBe(0xffffffff);
});

it('computeAdjacencyMatrixOrder sorts by group, tie key, then vertex', () => {
  expect(Array.from(computeAdjacencyMatrixOrder([1, 0, 1, 0]))).toEqual([2, 0, 3, 1]);
  expect(Array.from(computeAdjacencyMatrixOrder([1, 0, 1, 0], [5, 9, 1, 9]))).toEqual([3, 0, 2, 1]);
});

it('GPUAdjacencyMatrixOrder validates and names its nodes', () => {
  const device = createNullWebGPUDevice();
  const {graph, view} = createContext(device);
  expect(
    () =>
      new GPUAdjacencyMatrixOrder({
        groups: view('g', 'uint32', 4),
        output: {order: view('o', 'uint32', 3)}
      })
  ).toThrow(/order/);
  const contributor = new GPUAdjacencyMatrixOrder({
    groups: view('g2', 'uint32', 4),
    tieKeys: view('t', 'uint32', 4),
    output: {order: view('o2', 'uint32', 4)}
  });
  const ids = contributor.getCommandNodes(graph).map(node => node.id);
  expect(ids[0]).toBe('adjacency-matrix-order-identity');
  expect(ids.at(-1)).toBe('adjacency-matrix-order-scatter');
  expect(ids).toContain('adjacency-matrix-order-sort-pass-1-gather');
});

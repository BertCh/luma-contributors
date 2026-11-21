// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {
  getRegressionLossViews,
  getRegressionWeightsLossNodes
} from '../../../src/gpu-spatial-analysis/spatial-regression/regression-weights-completeness';
import {
  validateGPUChunkedSpatialWeights,
  validateGPUSpatialWeights
} from '../../../src/gpu-spatial-analysis/spatial-weights/index';
import {GPUSpatialTwoStageLeastSquares} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-two-stage-least-squares';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';

it('preserves outer-before-chunk loss order while deduplicating shared views', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    let serial = 0;
    const view = (length = 1) =>
      createTransientView(graph, `loss-order-${serial++}`, 'uint32', length);
    const shared = view();
    const outerCandidate = view();
    const chunkOverflow = view();
    const chunkCandidate = view();
    const createCSR = (name: string) => ({
      offsets: view(2),
      neighbors: view(),
      weights: createTransientView(graph, name, 'float32', 1)
    });
    const first = createCSR('first-values');
    const second = createCSR('second-values');
    const weights = {
      rowCount: 2,
      status: {overflow: shared, candidateOverflow: outerCandidate},
      chunks: [
        {
          rowStart: 0,
          rowCount: 1,
          weights: {...first, status: {overflow: chunkOverflow, candidateOverflow: shared}}
        },
        {
          rowStart: 1,
          rowCount: 1,
          weights: {...second, status: {overflow: shared, candidateOverflow: chunkCandidate}}
        }
      ]
    };
    expect(validateGPUChunkedSpatialWeights('ordered-loss', weights)).toBe(2);
    const memory = createCPUGraphMemory();
    weights.chunks.forEach((chunk, index) => {
      memory.write(chunk.weights.offsets, new Uint32Array([0, 1]));
      memory.write(chunk.weights.neighbors, new Uint32Array([1 - index]));
      memory.write(chunk.weights.weights, new Float32Array([1]));
    });
    expect(getRegressionLossViews(weights)).toEqual([
      shared,
      outerCandidate,
      chunkOverflow,
      chunkCandidate
    ]);
  } finally {
    device.destroy();
  }
});

it('detects count loss independently of clear loss flags and resets on recovery', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const view = (name: string, length = 1) => createTransientView(graph, name, 'uint32', length);
    const offsets = view('offsets', 4);
    const requiredCount = view('required');
    const count = view('count');
    const clear = view('clear');
    const weights = {
      offsets,
      neighbors: view('neighbors', 4),
      weights: createTransientView(graph, 'weights', 'float32', 4),
      status: {requiredCount, count, overflow: clear, candidateOverflow: clear}
    };
    expect(validateGPUSpatialWeights('count-only', weights)).toBe(3);
    const {nodes, loss} = getRegressionWeightsLossNodes(graph, 'count-only', weights);
    const memory = createCPUGraphMemory();
    memory.write(clear, new Uint32Array([0]));
    memory.write(weights.neighbors, new Uint32Array([1, 2, 0, 0]));
    memory.write(weights.weights, new Float32Array([1, 1, 1, 0]));
    memory.write(requiredCount, new Uint32Array([3]));
    memory.write(offsets, new Uint32Array([0, 1, 2, 3]));
    memory.write(count, new Uint32Array([2]));
    executeCommandNodesOnCPU(nodes, memory);
    expect(Array.from(memory.readUint32(loss!))).toEqual([1]);
    memory.write(count, new Uint32Array([3]));
    executeCommandNodesOnCPU(nodes, memory);
    expect(Array.from(memory.readUint32(loss!))).toEqual([0]);
    memory.write(offsets, new Uint32Array([0, 1, 2, 2]));
    executeCommandNodesOnCPU(nodes, memory);
    expect(Array.from(memory.readUint32(loss!))).toEqual([1]);
  } finally {
    device.destroy();
  }
});

for (const location of ['whole', 'root', 'chunk'] as const) {
  for (const withFlags of [false, true]) {
    it(`folds a ${location} completeness certificate with inverse loss polarity and recovery: flags=${withFlags}`, () => {
      const device = createNullWebGPUDevice();
      try {
        const graph = new GPUCommandGraph(device);
        const complete = createTransientView(graph, 'complete', 'uint32', 1);
        const clear = createTransientView(graph, 'clear-flags', 'uint32', 1);
        const offsets = createTransientView(graph, 'valid-offsets', 'uint32', 3);
        const status = withFlags
          ? {complete, overflow: clear, candidateOverflow: clear}
          : {complete};
        const csr = {
          offsets,
          neighbors: createTransientView(graph, 'neighbors', 'uint32', 2),
          weights: createTransientView(graph, 'values', 'float32', 2)
        };
        const input =
          location === 'whole'
            ? {...csr, status}
            : {
                rowCount: 2,
                status: location === 'root' ? status : undefined,
                chunks: [
                  {
                    rowStart: 0,
                    rowCount: 2,
                    weights: {...csr, status: location === 'chunk' ? status : undefined}
                  }
                ]
              };
        const {nodes, loss} = getRegressionWeightsLossNodes(graph, 'certificate', input);
        expect(loss).toBeDefined();
        const memory = createCPUGraphMemory();
        memory.write(offsets, new Uint32Array([0, 1, 2]));
        memory.write(csr.neighbors, new Uint32Array([1, 0]));
        memory.write(csr.weights, new Float32Array([1, 1]));
        memory.write(clear, new Uint32Array([0]));
        for (const value of [0, 1, 0]) {
          memory.write(complete, new Uint32Array([value]));
          executeCommandNodesOnCPU(nodes, memory);
          expect(Array.from(memory.readUint32(loss!))).toEqual([value === 0 ? 1 : 0]);
        }
      } finally {
        device.destroy();
      }
    });
  }
}

it('2SLS captures a supplied certificate and guards its real graph after caller map mutation', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    let serial = 0;
    const view = <Format extends 'float32' | 'uint32'>(format: Format, length: number) =>
      createTransientView(graph, `capture-${serial++}`, format, length);
    const complete = view('uint32', 1);
    const status = {complete};
    const input = {
      offsets: view('uint32', 5),
      neighbors: view('uint32', 4),
      weights: view('float32', 4),
      status
    };
    const contributor = new GPUSpatialTwoStageLeastSquares({
      weights: input,
      predictors: view('float32', 4),
      predictorCount: 1,
      response: view('float32', 4),
      output: {
        table: view('float32', 12),
        summary: view('float32', 32),
        status: {code: view('uint32', 1)}
      }
    });
    status.complete = view('uint32', 1);
    expect(contributor.props.weights.status?.complete).toBe(complete);
    expect(
      contributor.getCommandNodes(graph).some(node => node.id.includes('-weights-complete-'))
    ).toBe(true);
  } finally {
    device.destroy();
  }
});

for (const location of ['root', 'chunk'] as const) {
  it(`GM captures ${location} certificates through nested snapshots`, async () => {
    const {GPUSpatialErrorGM} = await import(
      '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-error-gm'
    );
    const device = createNullWebGPUDevice();
    try {
      const graph = new GPUCommandGraph(device);
      let serial = 0;
      const view = <Format extends 'float32' | 'uint32'>(format: Format, length: number) =>
        createTransientView(graph, `chunk-capture-${serial++}`, format, length);
      const complete = view('uint32', 1);
      const status = {complete};
      const weights = {
        rowCount: 4,
        status: location === 'root' ? status : undefined,
        chunks: [0, 2].map(rowStart => ({
          rowStart,
          rowCount: 2,
          weights: {
            offsets: view('uint32', 3),
            neighbors: view('uint32', 2),
            weights: view('float32', 2),
            status: location === 'chunk' ? status : undefined
          }
        }))
      };
      const contributor = new GPUSpatialErrorGM({
        weights,
        estimator: 'kelejian-prucha',
        predictors: view('float32', 4),
        predictorCount: 1,
        response: view('float32', 4),
        output: {
          table: view('float32', 12),
          summary: view('float32', 32),
          status: {code: view('uint32', 1)}
        }
      });
      status.complete = view('uint32', 1);
      const captured = contributor.props.weights;
      expect(
        location === 'root'
          ? captured.status?.complete
          : 'chunks' in captured
            ? captured.chunks[0].weights.status?.complete
            : undefined
      ).toBe(complete);
      expect(
        contributor.getCommandNodes(graph).some(node => node.id.includes('-weights-complete-'))
      ).toBe(true);
    } finally {
      device.destroy();
    }
  });
}

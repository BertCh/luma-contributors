// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {CPUSpatialWeightsAlgebraOperation} from '../../../src/gpu-spatial-analysis/spatial-weights/cpu-spatial-weights-algebra-operation';
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {GPUSpatialWeightsAlgebra} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-spatial-weights-algebra';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

it('review B2 CPU and direct GPU reject mixed distance units and accept matching units', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const floating = (length: number) =>
    createTransientView(graph, `algebra-float-${serial++}`, 'float32', length);
  const unsigned = (length: number) =>
    createTransientView(graph, `algebra-uint-${serial++}`, 'uint32', length);
  const context = {coordinateSpace: 'longitude-latitude', metric: 'great-circle'} as const;
  try {
    for (const units of ['meters', 'kilometers'] as const) {
      const left = {
        offsets: unsigned(4),
        neighbors: unsigned(1),
        weights: floating(1),
        distances: floating(1),
        spatialContext: context
      };
      const right = {...left, spatialContext: {...context, units}};
      const output = {
        offsets: unsigned(4),
        neighbors: unsigned(2),
        weights: floating(2),
        distances: floating(2),
        status: {overflow: unsigned(1)}
      };
      const cpu = () =>
        new CPUSpatialWeightsAlgebraOperation({
          operation: 'union',
          rowCount: 3,
          capacity: 2,
          left: {
            offsets: 'leftOffsets',
            neighbors: 'leftNeighbors',
            weights: 'leftWeights',
            distances: 'leftDistances',
            spatialContext: context
          },
          right: {
            offsets: 'rightOffsets',
            neighbors: 'rightNeighbors',
            weights: 'rightWeights',
            distances: 'rightDistances',
            spatialContext: right.spatialContext
          },
          output: {
            offsets: 'offsets',
            neighbors: 'neighbors',
            weights: 'weights',
            distances: 'distances',
            status: {overflow: 'overflow'}
          }
        });
      const gpu = () => new GPUSpatialWeightsAlgebra({operation: 'union', left, right, output});
      if (units === 'kilometers') {
        expect(cpu).toThrow(/effective length units/);
        expect(gpu).toThrow(/effective length units/);
      } else {
        expect(cpu).not.toThrow();
        expect(gpu).not.toThrow();
      }
    }
  } finally {
    device.destroy();
  }
});

it('review S1 keeps final overflow distinct from candidate overflow', async () => {
  const operation = new CPUSpatialWeightsAlgebraOperation({
    operation: 'subgraph',
    rowCount: 2,
    capacity: 1,
    mask: 'mask',
    weights: {offsets: 'inputOffsets', neighbors: 'inputNeighbors', weights: 'inputWeights'},
    output: {
      offsets: 'offsets',
      neighbors: 'neighbors',
      weights: 'weights',
      status: {
        overflow: 'overflow',
        requiredCount: 'requiredCount',
        candidateOverflow: 'candidateOverflow'
      }
    }
  });
  const vector = (format: 'uint32' | 'float32', values: Uint32Array | Float32Array) => ({
    format,
    length: values.length,
    chunks: [{values, length: values.length}],
    pack: () => values
  });
  const output = await operation.execute({
    inputs: {
      sourceOffsets: vector('uint32', Uint32Array.of(0, 1, 2)),
      sourceNeighbors: vector('uint32', Uint32Array.of(1, 0)),
      sourceWeights: vector('float32', Float32Array.of(1, 1)),
      mask: vector('uint32', Uint32Array.of(1, 1))
    }
  });
  // Two directed swap links require two slots; one retained slot loses final output only.
  expect(output['overflow']).toEqual(Uint32Array.of(1));
  expect(output['requiredCount']).toEqual(Uint32Array.of(2));
  expect(output['candidateOverflow']).toEqual(Uint32Array.of(0));
});

it('TSC-10 maps selected graph ports to vectors and keeps nested metadata out of the callback', async () => {
  for (const distances of [false, true]) {
    const operation = new CPUSpatialWeightsAlgebraOperation({
      operation: 'selfWeight',
      rowCount: 2,
      capacity: 4,
      selfWeight: 0.25,
      weights: {
        offsets: 'inputOffsets',
        neighbors: 'inputNeighbors',
        weights: 'inputWeights',
        distances: 'inputDistances'
      },
      output: {
        offsets: 'outputOffsets',
        neighbors: 'outputNeighbors',
        weights: 'outputWeights',
        ...(distances ? {distances: 'outputDistances'} : {}),
        status: {
          overflow: 'outputOverflow',
          requiredCount: 'outputRequired',
          candidateOverflow: 'outputCandidate'
        }
      }
    });
    const vector = (format: 'uint32' | 'float32', values: Uint32Array | Float32Array) => ({
      format,
      length: values.length,
      chunks: [{values, length: values.length}],
      pack: () => values
    });
    const output = await operation.execute({
      inputs: {
        sourceOffsets: vector('uint32', Uint32Array.of(0, 1, 2)),
        sourceNeighbors: vector('uint32', Uint32Array.of(1, 0)),
        sourceWeights: vector('float32', Float32Array.of(1, 1)),
        sourceDistances: vector('float32', Float32Array.of(3, 5))
      }
    });
    // Add .25 on the diagonal of the 2x2 swap matrix; distances on new self links are 0.
    expect(output).toEqual({
      outputOffsets: Uint32Array.of(0, 2, 4),
      outputNeighbors: Uint32Array.of(0, 1, 0, 1),
      outputWeights: Float32Array.of(0.25, 1, 1, 0.25),
      ...(distances ? {outputDistances: Float32Array.of(0, 3, 5, 0)} : {}),
      outputOverflow: Uint32Array.of(0),
      outputRequired: Uint32Array.of(4),
      outputCandidate: Uint32Array.of(0)
    });
  }
});

// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {expect, it} from 'vitest';
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  GPUContiguityWeights,
  GPULatticeWeights,
  GPUSpatialWeightsAlgebra,
  GPUChunkedSpatialWeightsAlgebra,
  GPUFuzzyContiguity,
  GPUAnalysisContiguityStatistics,
  GPUSpatialWeightsTranspose
} from '../../../src/gpu-spatial-analysis/spatial-weights/index';

it('OUT-01 and OUT-04 refuse unwritten destinations and declare a count writer on each route', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    let serial = 0;
    const view = <Format extends 'uint32' | 'float32' | 'float32x2'>(
      format: Format,
      length: number
    ) => createTransientView(graph, `output-w3-${serial++}`, format, length);
    const output = () => ({
      offsets: view('uint32', 3),
      neighbors: view('uint32', 2),
      weights: view('float32', 2),
      status: {
        overflow: view('uint32', 1),
        requiredCount: view('uint32', 1),
        candidateOverflow: view('uint32', 1),
        count: view('uint32', 1)
      }
    });
    const polygons = {
      kind: 'polygons' as const,
      positions: view('float32x2', 8),
      ringOffsets: view('uint32', 3),
      polygonOffsets: view('uint32', 3),
      featureOffsets: view('uint32', 3)
    };
    const source = output();
    const chunked = {rowCount: 2, chunks: [{rowStart: 0, rowCount: 2, weights: source}]};
    let producerId = 0;
    const constructors = [
      (destination: ReturnType<typeof output>) =>
        new GPULatticeWeights({
          id: `written-${producerId}`,
          width: 2,
          height: 1,
          criterion: 'rook',
          output: destination
        }),
      (destination: ReturnType<typeof output>) =>
        new GPUContiguityWeights({
          id: `written-${producerId}`,
          criterion: 'rook',
          polygons,
          output: destination
        }),
      (destination: ReturnType<typeof output>) =>
        new GPUSpatialWeightsAlgebra({
          id: `written-${producerId}`,
          operation: 'union',
          left: source,
          right: source,
          output: destination
        }),
      (destination: ReturnType<typeof output>) =>
        new GPUChunkedSpatialWeightsAlgebra({
          id: `written-${producerId}`,
          operation: 'union',
          left: chunked,
          right: chunked,
          output: {chunks: [{rowStart: 0, rowCount: 2, weights: destination}]}
        }),
      (destination: ReturnType<typeof output>) =>
        new GPUFuzzyContiguity({
          id: `written-${producerId}`,
          polygons,
          tolerance: 0,
          output: {...destination, added: view('uint32', 2), featureStatus: view('uint32', 2)}
        }),
      (destination: ReturnType<typeof output>) =>
        new GPUAnalysisContiguityStatistics({
          id: `written-${producerId}`,
          polygons,
          criterion: 'rook',
          output: {
            weights: destination,
            counts: view('uint32', 6),
            cardinality: view('uint32', 2),
            median: view('float32', 1),
            validity: view('uint32', 1)
          }
        })
    ];
    for (const construct of constructors) {
      producerId++;
      const destination = output();
      for (const name of [
        'converged',
        'iterationCount',
        'iterationLimitReached',
        'approximation',
        'invalidCount',
        'uncertainCount',
        'complete',
        'unknown'
      ]) {
        expect(() =>
          construct({
            ...destination,
            status: {...destination.status, [name]: view('uint32', 1)}
          } as never)
        ).toThrow(new RegExp(`written.*output.status.${name}`));
      }
      const nodes = construct(destination).getCommandNodes(graph);
      expect(nodes.some(node => node.id.endsWith('-count'))).toBe(true);
    }
    for (const seed of [
      [0, 0],
      [10, 10]
    ]) {
      // Seed values must not make an unwritten metric into a distance-producing request.
      const distances = view('float32', seed.length);
      expect(
        () =>
          new GPUContiguityWeights({criterion: 'rook', polygons, output: {...output(), distances}})
      ).toThrow(/output.distances/);
    }
    const producer = new GPUContiguityWeights({criterion: 'rook', polygons, output: output()});
    expect(producer.outputWeights.distances).toBeUndefined();
    expect(() =>
      new GPUSpatialWeightsTranspose({
        weights: producer.outputWeights,
        output: {...output(), status: undefined} as never
      }).getCommandNodes(graph)
    ).not.toThrow();
  } finally {
    device.destroy();
  }
});

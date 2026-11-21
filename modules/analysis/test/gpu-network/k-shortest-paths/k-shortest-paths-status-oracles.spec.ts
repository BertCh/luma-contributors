// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPUKShortestPaths} from '../../../src/gpu-network/k-shortest-paths/gpu-k-shortest-paths';
import {GPUKShortestPathsOperation} from '../../../src/gpu-network/k-shortest-paths/gpu-k-shortest-paths-operation';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {
  createInputBuffer,
  createOutputBuffer,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';

for (const scene of ['complete', 'candidate-limit', 'path-limit', 'invalid'] as const) {
  it(`writes every independently derived Yen status word and clears it on reuse: ${scene}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    // The only paths are 0-1-3 (edge 2, cost 2), 0-1-3 (edge 3, cost 3),
    // and 0-2-3 (cost 4). Round 2 proposes costs 4 and 3 simultaneously;
    // capacity 1 cannot hold both. Capacity 2 accepts all routes in 3 rounds.
    // Path limit 2 refuses the first path's 3 vertices in round 1. No shortest
    // search needs more than two hops, so two 64-hop rounds cannot bind here.
    const offsets = Uint32Array.of(0, 2, 4, 5, 5);
    const neighbors = Uint32Array.of(1, 2, 3, 3, 3);
    const weightValues = Float32Array.of(1, 2, 1, 2, 2);
    const sources = Uint32Array.of(scene === 'invalid' ? 99 : 0);
    const targets = Uint32Array.of(3);
    const options = {
      routeCount: 3,
      maximumPathLength: scene === 'path-limit' ? 2 : 3,
      candidateCapacity: scene === 'candidate-limit' ? 1 : 2,
      maximumIterations: 2
    };
    const expected = {
      complete: [3, 1, 0, 0, 3],
      'candidate-limit': [1, 0, 0, 1, 2],
      'path-limit': [0, 0, 1, 0, 1],
      invalid: [0, 1, 0, 0, 0]
    }[scene];
    const graph = new GPUCommandGraph(device);
    const buffers: ReturnType<typeof createOutputBuffer>[] = [];
    let nextView = 0;
    const input = <Format extends 'uint32' | 'float32'>(
      values: Uint32Array | Float32Array,
      format: Format
    ) => {
      const buffer = createInputBuffer(device, values);
      buffers.push(buffer);
      return {
        buffer,
        view: importGraphBuffer(graph, `input-${nextView++}`, buffer, format, values.length)
      };
    };
    const output = <Format extends 'uint32' | 'float32' = 'uint32'>(
      length: number,
      format: Format = 'uint32' as Format
    ) => {
      const buffer = createOutputBuffer(device, length);
      buffers.push(buffer);
      return {
        buffer,
        view: importGraphBuffer(graph, `output-${nextView++}`, buffer, format, length)
      };
    };
    const statusNames = [
      'converged',
      'iterationLimitReached',
      'candidateOverflow',
      'iterationCount'
    ] as const;
    const counts = output(1);
    const converged = output(1);
    const iterationLimitReached = output(1);
    const candidateOverflow = output(1);
    const iterationCount = output(1);
    const checked = [counts, converged, iterationLimitReached, candidateOverflow, iterationCount];
    const weights = input(weightValues, 'float32');
    graph.add(
      new GPUKShortestPaths({
        ...options,
        topology: {
          offsets: input(offsets, 'uint32').view,
          neighbors: input(neighbors, 'uint32').view,
          weights: weights.view
        },
        sources: input(sources, 'uint32').view,
        targets: input(targets, 'uint32').view,
        output: {
          vertices: output(3 * options.maximumPathLength).view,
          edges: output(3 * options.maximumPathLength).view,
          lengths: output(3).view,
          costs: output(3, 'float32').view,
          counts: counts.view,
          status: {
            converged: converged.view,
            iterationLimitReached: iterationLimitReached.view,
            candidateOverflow: candidateOverflow.view,
            iterationCount: iterationCount.view
          }
        }
      })
    );
    const compiled = graph.compile();
    try {
      for (const port of checked) port.buffer.write(Uint32Array.of(0xdeadbeef));
      submitGraph(device, compiled, undefined);
      expect(
        await Promise.all(checked.map(async port => (await readUint32(port.buffer, 1))[0]))
      ).toEqual(expected);
      weights.buffer.write(new Float32Array(5).fill(-1));
      submitGraph(device, compiled, undefined);
      expect(
        await Promise.all(checked.map(async port => (await readUint32(port.buffer, 1))[0]))
      ).toEqual(scene === 'invalid' ? [0, 1, 0, 0, 0] : [0, 1, 0, 0, 1]);
    } finally {
      compiled.destroy();
      for (const buffer of buffers) buffer.destroy();
    }
    // Exercise graph-value lowering separately with the same hand-derived literals.
    const rows = await runOperationGraph(device, {
      inputs: {
        offsets: {format: 'uint32', values: offsets},
        neighbors: {format: 'uint32', values: neighbors},
        weights: {format: 'float32', values: weightValues},
        sources: {format: 'uint32', values: sources},
        targets: {format: 'uint32', values: targets}
      },
      operations: [
        new GPUKShortestPathsOperation({
          ...options,
          queryCount: 1,
          topology: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
          sources: 'sources',
          targets: 'targets',
          output: {
            vertices: 'vertices',
            edges: 'edges',
            lengths: 'lengths',
            costs: 'costs',
            counts: 'counts',
            status: {
              converged: 'converged',
              iterationLimitReached: 'iterationLimitReached',
              candidateOverflow: 'candidateOverflow',
              iterationCount: 'iterationCount'
            }
          }
        })
      ],
      outputs: ['counts', ...statusNames]
    });
    expect(['counts', ...statusNames].map(name => rows[name][0])).toEqual(expected);
  });
}

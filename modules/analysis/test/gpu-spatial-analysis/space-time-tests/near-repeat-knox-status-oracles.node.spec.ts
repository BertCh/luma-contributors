// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPUNearRepeatKnox} from '../../../src/gpu-spatial-analysis/space-time-tests/gpu-near-repeat-knox';
import {GPU_SPACE_TIME_PARAMETER_LENGTH} from '../../../src/gpu-spatial-analysis/space-time-tests/space-time-parameters';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {useCPUExecutorTestHooks} from '../segment-intersection/cpu-executor-test-hooks';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';

useCPUExecutorTestHooks();

it('near-repeat binds both loss flags and suppresses only owned count and summary words', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const overflow = createTransientView(graph, 'pair-overflow', 'uint32', 1);
    const candidateOverflow = createTransientView(graph, 'pair-candidate-loss', 'uint32', 1);
    const counts = createTransientView(graph, 'counts', 'uint32', 18);
    const summary = createTransientView(graph, 'summary', 'float32', 34);
    const pairs = {
      offsets: createTransientView(graph, 'offsets', 'uint32', 4),
      neighbors: createTransientView(graph, 'neighbors', 'uint32', 3),
      weights: createTransientView(graph, 'weights', 'float32', 3),
      distances: createTransientView(graph, 'distances', 'float32', 3),
      status: {overflow, candidateOverflow}
    };
    const contributor = new GPUNearRepeatKnox({
      pairs,
      times: createTransientView(graph, 'times', 'float32', 3),
      parameters: createTransientView(
        graph,
        'parameters',
        'uint32',
        GPU_SPACE_TIME_PARAMETER_LENGTH
      ),
      distanceEdges: [0, 1, 3],
      timeEdges: [0, 1, 2],
      maximumPermutations: 3,
      output: {counts, summary}
    });
    // Mutating caller-owned wrappers must not remove the two already validated loss views.
    expect(Reflect.deleteProperty(pairs.status, 'overflow')).toBe(true);
    expect(Reflect.deleteProperty(pairs.status, 'candidateOverflow')).toBe(true);
    pairs.status = {
      overflow: createTransientView(graph, 'replacement-overflow', 'uint32', 1),
      candidateOverflow: createTransientView(graph, 'replacement-candidate-loss', 'uint32', 1)
    };
    expect(contributor.props.pairs.status?.overflow).toBe(overflow);
    expect(contributor.props.pairs.status?.candidateOverflow).toBe(candidateOverflow);
    const nodes = contributor.getCommandNodes(graph);
    const suppression = nodes.filter(node => node.id.includes('-incomplete-'));
    // The histogram uses workgroup barriers, which this interpreter cannot execute. Execute
    // only the real scalar suppression kernels on independently seeded public output ranges.
    expect(suppression).toHaveLength(2);
    // A newly read loss flag cannot alias a writable output in its suppression dispatch.
    expect(
      () =>
        new GPUNearRepeatKnox({
          ...contributor.props,
          pairs: {
            ...contributor.props.pairs,
            status: {
              overflow: graph.createDataView(counts.buffer, {format: 'uint32', length: 1})
            }
          }
        })
    ).toThrow(/outputs must not share buffers with inputs/);
    const memory = createCPUGraphMemory();
    for (const [finalLoss, candidateLoss] of [
      [0, 0],
      [1, 0],
      [0, 1],
      [0, 0]
    ]) {
      memory.write(overflow, Uint32Array.of(finalLoss));
      memory.write(candidateOverflow, Uint32Array.of(candidateLoss));
      memory.write(counts, new Uint32Array(18).fill(7));
      memory.write(summary, new Float32Array(34).fill(11));
      executeCommandNodesOnCPU(suppression, memory);
      const incomplete = finalLoss !== 0 || candidateLoss !== 0;
      expect(Array.from(memory.readUint32(counts))).toEqual([
        ...Array(16).fill(incomplete ? 0 : 7),
        7,
        7
      ]);
      const actualSummary = Array.from(memory.readFloat32(summary));
      expect(
        actualSummary.slice(0, 32).every(incomplete ? Number.isNaN : value => value === 11)
      ).toBe(true);
      expect(actualSummary.slice(32)).toEqual([11, 11]);
    }
  } finally {
    device.destroy();
  }
});

for (const populated of [false, true]) {
  it(`near-repeat ignores undeclared chunks on a packed CSR (populated=${populated})`, () => {
    const device = createNullWebGPUDevice();
    try {
      const graph = new GPUCommandGraph(device);
      const pairs = {
        offsets: createTransientView(graph, 'offsets', 'uint32', 4),
        neighbors: createTransientView(graph, 'neighbors', 'uint32', 3),
        weights: createTransientView(graph, 'weights', 'float32', 3),
        distances: createTransientView(graph, 'distances', 'float32', 3)
      };
      const extraPairs = {
        ...pairs,
        chunks: populated
          ? [
              {
                rowStart: 0,
                rowCount: 3,
                weights: {
                  ...pairs,
                  status: {
                    candidateOverflow: createTransientView(
                      graph,
                      'unrelated-chunk-loss',
                      'uint32',
                      1
                    )
                  }
                }
              }
            ]
          : undefined
      };
      const nodes = new GPUNearRepeatKnox({
        pairs: extraPairs,
        times: createTransientView(graph, 'times', 'float32', 3),
        parameters: createTransientView(
          graph,
          'parameters',
          'uint32',
          GPU_SPACE_TIME_PARAMETER_LENGTH
        ),
        distanceEdges: [0, 1, 3],
        timeEdges: [0, 1, 2],
        maximumPermutations: 3,
        output: {
          counts: createTransientView(graph, 'counts', 'uint32', 16),
          summary: createTransientView(graph, 'summary', 'float32', 32)
        }
      }).getCommandNodes(graph);
      // This class consumes the packed CSR only: mask, band, histogram, summary. An extra
      // property cannot select chunked semantics or add loss from a neighborhood it never reads.
      expect(nodes).toHaveLength(4);
      expect(nodes.some(node => node.id.includes('-incomplete-'))).toBe(false);
    } finally {
      device.destroy();
    }
  });
}

it('near-repeat alias diagnostics name the public candidate-loss and count ports', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const counts = createTransientView(graph, 'counts', 'uint32', 16);
    expect(
      () =>
        new GPUNearRepeatKnox({
          pairs: {
            offsets: createTransientView(graph, 'offsets', 'uint32', 4),
            neighbors: createTransientView(graph, 'neighbors', 'uint32', 3),
            weights: createTransientView(graph, 'weights', 'float32', 3),
            distances: createTransientView(graph, 'distances', 'float32', 3),
            status: {
              candidateOverflow: graph.createDataView(counts.buffer, {format: 'uint32', length: 1})
            }
          },
          times: createTransientView(graph, 'times', 'float32', 3),
          parameters: createTransientView(
            graph,
            'parameters',
            'uint32',
            GPU_SPACE_TIME_PARAMETER_LENGTH
          ),
          distanceEdges: [0, 1, 3],
          timeEdges: [0, 1, 2],
          maximumPermutations: 3,
          output: {counts, summary: createTransientView(graph, 'summary', 'float32', 32)}
        })
    ).toThrow(/output\.counts conflicts with pairs\.status\.candidateOverflow/);
  } finally {
    device.destroy();
  }
});

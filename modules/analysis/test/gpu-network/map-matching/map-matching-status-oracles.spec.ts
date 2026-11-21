// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {
  GPUMapMatching,
  getGPUMapMatchingParameterValues
} from '../../../src/gpu-network/map-matching/gpu-map-matching';
import {GPUMapMatchingOperation} from '../../../src/gpu-network/map-matching/gpu-map-matching-operation';
import {CPUMapMatchingOperation} from '../../../src/gpu-network/map-matching/cpu-map-matching-operation';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {
  createInputBuffer,
  createOutputBuffer,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {poisonOperationOutputs} from '../adjacency-matrix/network-output-oracle-test-utils';

it('rewrites poisoned matching status words and distinguishes restart from traversal capacity', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  const buffers: ReturnType<typeof createOutputBuffer>[] = [];
  let nextView = 0;
  const input = <Format extends 'uint32' | 'float32' | 'float32x2'>(
    values: Uint32Array | Float32Array,
    format: Format
  ) => {
    const buffer = createInputBuffer(device, values);
    buffers.push(buffer);
    return {
      buffer,
      view: importGraphBuffer(
        graph,
        `input-${nextView++}`,
        buffer,
        format,
        values.length / (format === 'float32x2' ? 2 : 1)
      )
    };
  };
  const checked = new Map<
    string,
    {buffer: ReturnType<typeof createOutputBuffer>; length: number}
  >();
  const output = (name: string, length: number) => {
    const buffer = createOutputBuffer(device, length);
    buffers.push(buffer);
    checked.set(name, {buffer, length});
    return importGraphBuffer(graph, name, buffer, 'uint32', length);
  };
  const values = {
    forward: Float32Array.of(0.25, 0, 0.75, 0, 0.5, 0),
    reversed: Float32Array.of(0.75, 0, 0.25, 0, 0.5, 0),
    unmatched: Float32Array.of(10, 10, 10, 10, 10, 10)
  };
  const positions = input(values.forward, 'float32x2');
  const parameters = getGPUMapMatchingParameterValues({sigma: 0.1, beta: 1, searchRadius: 0.1});
  const offsets = Uint32Array.of(0, 1, 1);
  const neighbors = Uint32Array.of(1);
  const nodes = Float32Array.of(0, 0, 1, 0);
  const tracks = Uint32Array.of(0, 2, 3);
  const bounds = {minimum: [-1, -1] as const, maximum: [2, 2] as const};
  const options = {cellSize: 1, candidateCount: 1, routeNodeBudget: 8, traversedEdgeCapacity: 1};
  graph.add(
    new GPUMapMatching({
      ...options,
      positions: positions.view,
      parameters: input(parameters, 'float32').view,
      trackOffsets: input(tracks, 'uint32').view,
      bounds,
      topology: {
        offsets: input(offsets, 'uint32').view,
        neighbors: input(neighbors, 'uint32').view,
        nodePositions: input(nodes, 'float32x2').view
      },
      output: {
        matchedEdges: output('edges', 3),
        matchedCount: output('matchedCount', 1),
        breakCount: output('breakCount', 1),
        status: {
          overflow: output('overflow', 1),
          candidateOverflow: output('candidateOverflow', 1),
          routeOverflow: output('routeOverflow', 1)
        },
        traversedEdges: {
          edges: output('traversedEdges', 2),
          counts: output('traversedCounts', 2),
          status: {
            overflow: output('traversedOverflow', 1),
            candidateOverflow: output('traversedCandidateOverflow', 1)
          }
        }
      }
    })
  );
  const statusNames = [
    'matchedCount',
    'breakCount',
    'overflow',
    'candidateOverflow',
    'routeOverflow',
    'traversedCounts',
    'traversedOverflow',
    'traversedCandidateOverflow'
  ];
  // One directed edge: forward fixes share it, so each track emits one entry.
  // Reversing the first pair cannot return from node 1 to node 0: it restarts,
  // needs [0,NONE,0], and exceeds capacity 1 while remaining a complete match.
  // Remote fixes match nothing; only the noninitial fix counts as a break.
  // No outgoing edge at node 1 means no search can drop a node. Final matching
  // has one output row per fix, so its capacity overflow is always zero.
  const expected = {
    forward: [3, 0, 0, 0, 0, 1, 1, 0, 0],
    reversed: [3, 1, 0, 0, 0, 3, 1, 1, 0],
    unmatched: [0, 1, 0, 0, 0, 0, 0, 0, 0]
  };
  const compiled = graph.compile();
  try {
    for (const {buffer, length} of checked.values())
      buffer.write(new Uint32Array(length).fill(0xdeadbeef));
    for (const scene of ['forward', 'reversed', 'unmatched', 'forward'] as const) {
      positions.buffer.write(values[scene]);
      submitGraph(device, compiled, undefined);
      const actual: number[] = [];
      for (const name of statusNames) {
        const {buffer, length} = checked.get(name)!;
        actual.push(...(await readUint32(buffer, length)));
      }
      expect(actual, scene).toEqual(expected[scene]);
    }
  } finally {
    compiled.destroy();
    for (const buffer of buffers) buffer.destroy();
  }
  for (const scene of ['forward', 'reversed', 'unmatched'] as const)
    for (const Operation of [GPUMapMatchingOperation, CPUMapMatchingOperation]) {
      const createOperation = (props: ConstructorParameters<typeof GPUMapMatchingOperation>[0]) => {
        const operation = new Operation(props);
        return operation instanceof GPUMapMatchingOperation
          ? poisonOperationOutputs(operation)
          : operation;
      };
      const rows = await runOperationGraph(device, {
        inputs: {
          points: {format: 'float32x2', values: values[scene]},
          parameters: {format: 'float32', values: parameters},
          tracks: {format: 'uint32', values: tracks},
          offsets: {format: 'uint32', values: offsets},
          neighbors: {format: 'uint32', values: neighbors},
          nodes: {format: 'float32x2', values: nodes}
        },
        operations: [
          createOperation({
            ...options,
            bounds,
            pointCount: 3,
            trackCount: 2,
            nodeCount: 2,
            edgeCount: 1,
            positions: 'points',
            parameters: 'parameters',
            trackOffsets: 'tracks',
            topology: {offsets: 'offsets', neighbors: 'neighbors', nodePositions: 'nodes'},
            output: {
              matchedEdges: 'edges',
              matchedCount: 'matchedCount',
              breakCount: 'breakCount',
              status: {
                overflow: 'overflow',
                candidateOverflow: 'candidateOverflow',
                routeOverflow: 'routeOverflow'
              },
              traversedEdges: {
                edges: 'traversedEdges',
                counts: 'traversedCounts',
                status: {
                  overflow: 'traversedOverflow',
                  candidateOverflow: 'traversedCandidateOverflow'
                }
              }
            }
          })
        ],
        outputs: statusNames
      });
      expect(
        statusNames.flatMap(name => Array.from(rows[name])),
        `${scene} ${Operation.name}`
      ).toEqual(expected[scene]);
    }
});

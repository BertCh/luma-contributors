// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPUMapMatchedEdgeVolumes} from '../../../src/gpu-network/flow-aggregation/gpu-map-matched-edge-volumes';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {
  createInputBuffer,
  createOutputBuffer,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';

it('counts fixed traversal movements and rewrites every count and incompleteness word', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  const buffers: ReturnType<typeof createOutputBuffer>[] = [];
  let nextView = 0;
  const input = (values: Uint32Array) => {
    const buffer = createInputBuffer(device, values);
    buffers.push(buffer);
    return {
      buffer,
      view: importGraphBuffer(graph, `input-${nextView++}`, buffer, 'uint32', values.length)
    };
  };
  const outputs = new Map<
    string,
    {buffer: ReturnType<typeof createOutputBuffer>; length: number}
  >();
  const output = (name: string, length: number) => {
    const buffer = createOutputBuffer(device, length);
    buffers.push(buffer);
    outputs.set(name, {buffer, length});
    return importGraphBuffer(graph, name, buffer, 'uint32', length);
  };
  // Track A: [0,1,separator,2]; B: [0,1,2] with an unused trailing 2.
  // Exactly three movements exist: 0->1 twice and 1->2 once. No movement
  // crosses a separator or reads past a track's count. All edge volumes are 2.
  const edges = input(Uint32Array.of(0, 1, 0xffffffff, 2, 0, 1, 2, 2));
  const counts = input(Uint32Array.of(4, 3));
  const candidateOverflow = input(Uint32Array.of(0));
  const overflow = input(Uint32Array.of(0));
  graph.add(
    new GPUMapMatchedEdgeVolumes({
      traversedEdges: {
        edges: edges.view,
        counts: counts.view,
        status: {candidateOverflow: candidateOverflow.view, overflow: overflow.view}
      },
      trackCount: 2,
      traversedEdgeCapacity: 4,
      edgeCount: 3,
      transitionPairCapacity: 8,
      output: {
        edgeVolumes: output('volumes', 3),
        status: {candidateOverflow: output('loss', 1)},
        transitions: {
          ids: output('ids', 3),
          flowCounts: output('flowCounts', 3),
          zoneOutCounts: output('zoneOutCounts', 3),
          zoneInCounts: output('zoneInCounts', 3),
          status: {
            count: output('count', 1),
            overflow: output('overflow', 1),
            requiredCount: output('requiredCount', 1),
            candidateOverflow: output('transitionLoss', 1)
          }
        }
      }
    })
  );
  const compiled = graph.compile();
  try {
    for (const {buffer, length} of outputs.values())
      buffer.write(new Uint32Array(length).fill(0xdeadbeef));
    for (const scene of [
      'complete',
      'candidate-loss',
      'count-truncation',
      'input-overflow',
      'empty'
    ] as const) {
      candidateOverflow.buffer.write(Uint32Array.of(scene === 'candidate-loss' ? 1 : 0));
      overflow.buffer.write(Uint32Array.of(scene === 'input-overflow' ? 1 : 0));
      counts.buffer.write(
        scene === 'empty'
          ? Uint32Array.of(0, 0)
          : Uint32Array.of(scene === 'count-truncation' ? 5 : 4, 3)
      );
      submitGraph(device, compiled, undefined);
      const actual: Record<string, number[]> = {};
      for (const [name, {buffer, length}] of outputs)
        actual[name] = await readUint32(buffer, length);
      const empty = scene === 'empty';
      const loss =
        scene === 'candidate-loss' || scene === 'count-truncation' || scene === 'input-overflow';
      // Two distinct pairs fit both the eight-slot hash and three-row output, so
      // transition loss here comes only from the explicit input incompleteness.
      expect(actual).toEqual({
        volumes: empty ? [0, 0, 0] : [2, 2, 2],
        loss: [Number(loss)],
        ids: empty ? [0xffffffff, 0xffffffff, 0xffffffff] : [1, 5, 0xffffffff],
        flowCounts: empty ? [0, 0, 0] : [2, 1, 0],
        zoneOutCounts: empty ? [0, 0, 0] : [2, 1, 0],
        zoneInCounts: empty ? [0, 0, 0] : [0, 2, 1],
        count: [empty ? 0 : 2],
        requiredCount: [empty ? 0 : 2],
        overflow: [0],
        transitionLoss: [Number(loss)]
      });
    }
  } finally {
    compiled.destroy();
    for (const buffer of buffers) buffer.destroy();
  }
});

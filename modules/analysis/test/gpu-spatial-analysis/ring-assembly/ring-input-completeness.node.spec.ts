// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {getRingInputCompletenessNodes} from '../../../src/analysis-core/ring-assembly/ring-input-completeness';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';

it('CPL-12 and UGE-02 publish counted polygon completeness and reset inherited loss on reuse', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const scalar = (length = 1) => createTransientView(graph, `word-${serial++}`, 'uint32', length);
  const input = {count: scalar(), overflow: scalar(), candidateOverflow: scalar()};
  const counters = scalar(6),
    flags = scalar(8);
  const output = {
    positions: createTransientView(graph, 'positions', 'float32x2', 20),
    ringOffsets: scalar(4),
    polygons: {
      kind: 'polygons' as const,
      positions: createTransientView(graph, 'polygon-positions', 'float32x2', 20),
      ringOffsets: scalar(4),
      polygonOffsets: scalar(4),
      featureOffsets: scalar(4)
    },
    status: {
      count: scalar(),
      overflow: scalar(),
      candidateOverflow: scalar(),
      complete: scalar(),
      polygonCount: scalar(),
      openSegmentCount: scalar()
    }
  };
  const memory = createCPUGraphMemory();
  memory.write(counters, Uint32Array.of(0, 0, 0, 0, 0, 4));
  memory.write(flags, new Uint32Array(8));
  memory.write(output.status.count, Uint32Array.of(1));
  memory.write(output.status.overflow, Uint32Array.of(0));
  memory.write(output.polygons.featureOffsets, Uint32Array.of(0, 1, 1, 1));
  const nodes = getRingInputCompletenessNodes(graph, {
    id: 'prefix',
    input,
    output,
    counters,
    segmentFlags: flags
  });
  // Four retained edges close one unit square; a second square's four edges were lost upstream.
  for (const [overflow, candidate, complete] of [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]
  ]) {
    memory.write(input.overflow, Uint32Array.of(overflow));
    memory.write(input.candidateOverflow, Uint32Array.of(candidate));
    executeCommandNodesOnCPU(nodes, memory);
    expect(memory.readUint32(output.status.candidateOverflow)[0]).toBe(1 - complete);
    expect(memory.readUint32(output.status.complete)[0]).toBe(complete);
    expect(memory.readUint32(output.status.polygonCount)[0]).toBe(1);
    expect(memory.readUint32(output.status.overflow)[0]).toBe(0);
  }
  memory.write(input.overflow, Uint32Array.of(1));
  memory.write(output.status.openSegmentCount, Uint32Array.of(7));
  const {candidateOverflow: _candidate, complete: _complete, ...status} = output.status;
  executeCommandNodesOnCPU(
    getRingInputCompletenessNodes(graph, {
      id: 'suppressed',
      input,
      output: {...output, status},
      counters,
      segmentFlags: flags
    }),
    memory
  );
  expect(memory.readUint32(status.count)[0]).toBe(0);
  expect(memory.readUint32(status.polygonCount)[0]).toBe(0);
  expect(Array.from(memory.readUint32(output.polygons.featureOffsets))).toEqual([0, 0, 0, 0]);
  expect(memory.readUint32(status.overflow)[0]).toBe(0);
  expect(memory.readUint32(status.openSegmentCount)[0]).toBe(7);
  device.destroy();
});

for (const target of ['mappings', 'scalar-padding'] as const) {
  it(`B3/B4 suppression preserves its output contracts: ${target}`, () => {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    let serial = 0;
    const words = (length = 1) =>
      createTransientView(graph, `suppression-${serial++}`, 'uint32', length);
    const input = {count: words(), overflow: words()};
    const output = {
      positions: createTransientView(graph, 'positions', 'float32x2', 4),
      ringOffsets: words(2),
      segmentRings: words(3),
      segmentVertices: words(3),
      polygons: {
        kind: 'polygons' as const,
        positions: createTransientView(graph, 'polygons', 'float32x2', 4),
        ringOffsets: words(2),
        polygonOffsets: words(2),
        featureOffsets: words(2),
        sourceIds: words(1)
      },
      status: {count: words(2), overflow: words(), openSegmentCount: words()}
    };
    const memory = createCPUGraphMemory();
    memory.write(input.overflow, Uint32Array.of(1));
    memory.write(output.status.count, Uint32Array.of(7, 55));
    memory.write(output.status.openSegmentCount, Uint32Array.of(0));
    memory.write(output.segmentRings, Uint32Array.of(0, 0, 0));
    memory.write(output.segmentVertices, Uint32Array.of(0, 1, 2));
    memory.write(output.polygons.sourceIds, Uint32Array.of(42));
    executeCommandNodesOnCPU(
      getRingInputCompletenessNodes(graph, {
        id: 'suppress',
        input,
        output,
        counters: words(6),
        segmentFlags: words(3)
      }),
      memory
    );
    if (target === 'mappings') {
      expect(Array.from(memory.readUint32(output.segmentRings))).toEqual([
        0xffffffff, 0xffffffff, 0xffffffff
      ]);
      expect(Array.from(memory.readUint32(output.segmentVertices))).toEqual([
        0xffffffff, 0xffffffff, 0xffffffff
      ]);
      expect(Array.from(memory.readUint32(output.polygons.sourceIds))).toEqual([0xffffffff]);
    } else expect(Array.from(memory.readUint32(output.status.count))).toEqual([0, 55]);
    expect(memory.readUint32(output.status.openSegmentCount)[0]).toBe(0);
    device.destroy();
  });
}

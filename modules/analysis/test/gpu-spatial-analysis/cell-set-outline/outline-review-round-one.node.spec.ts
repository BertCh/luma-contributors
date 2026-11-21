// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readFileSync} from 'node:fs';
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPUCellSetOutline} from '../../../src/gpu-spatial-analysis/cell-set-outline';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';
import {outlineQuadbinOnCPU} from './cell-set-outline-oracle';

function createFixture() {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends 'uint32' | 'uint32x2' | 'float32x2' | 'float32x4'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `review-${serial++}`, format, length);
  const output = {
    rows: view('uint32', 4),
    cells: view('uint32x2', 4),
    edgeIndices: view('uint32', 4),
    endpoints: view('float32x4', 4),
    status: {count: view('uint32', 1), overflow: view('uint32', 1)}
  };
  const rings = {
    positions: view('float32x2', 5),
    ringOffsets: view('uint32', 3),
    status: {
      count: view('uint32', 3),
      overflow: view('uint32', 1),
      openSegmentCount: view('uint32', 1),
      touchingSegmentCount: view('uint32', 1)
    }
  };
  return {device, graph, view, output, rings, cells: view('uint32x2', 1)};
}

it('B1 suppresses one scalar word while clearing complete offset arrays and preserving clean replay', () => {
  const {device, graph, output, rings, cells} = createFixture();
  try {
    const nodes = new GPUCellSetOutline({
      family: 'quadbin',
      cells,
      output,
      rings: {output: rings}
    }).getCommandNodes(graph);
    const memory = createCPUGraphMemory();
    for (const loss of [1, 0, 1, 0]) {
      memory.write(output.status.count, Uint32Array.of(4));
      memory.write(output.status.overflow, Uint32Array.of(loss));
      memory.write(rings.status.count, Uint32Array.of(1, 91, 9898));
      memory.write(rings.ringOffsets, Uint32Array.of(0, 5, 5));
      executeCommandNodesOnCPU(
        nodes.filter(node => /-rings-(capture-loss|prepare-count)$/.test(node.id)),
        memory
      );
      executeCommandNodesOnCPU(
        nodes.filter(node => node.id.includes('-suppress-incomplete-')),
        memory
      );
      expect(Array.from(memory.readUint32(rings.status.count))).toEqual([loss ? 0 : 1, 91, 9898]);
      expect(Array.from(memory.readUint32(rings.ringOffsets))).toEqual(
        loss ? [0, 0, 0] : [0, 5, 5]
      );
    }
  } finally {
    device.destroy();
  }
});

it('B2 keeps a bare-cell outline valid when local and ring overflow share disjoint buffer regions', () => {
  const {device, graph, view, output, rings, cells} = createFixture();
  const backing = view('uint32', 128);
  output.status.overflow = graph.createDataView(backing.buffer, {
    format: 'uint32',
    length: 1,
    byteOffset: 0
  });
  rings.status.overflow = graph.createDataView(backing.buffer, {
    format: 'uint32',
    length: 1,
    byteOffset: 256
  });
  try {
    const outline = new GPUCellSetOutline({
      family: 'quadbin',
      cells,
      output,
      rings: {output: rings}
    });
    expect(() => outline.getCommandNodes(graph)).not.toThrow();
  } finally {
    device.destroy();
  }
});

it('F4 suppresses rings for an incomplete counted source and recovers on clean replay', () => {
  const {device, graph, view, output, rings, cells} = createFixture();
  try {
    const complete = view('uint32', 1);
    const source = {
      cells,
      status: {count: view('uint32', 1), overflow: view('uint32', 1), complete}
    };
    const nodes = new GPUCellSetOutline({
      family: 'quadbin',
      source,
      output,
      rings: {output: rings}
    }).getCommandNodes(graph);
    const memory = createCPUGraphMemory();
    for (const certificate of [1, 0, 1]) {
      memory.write(complete, Uint32Array.of(certificate));
      memory.write(source.status.overflow, Uint32Array.of(0));
      memory.write(output.status.count, Uint32Array.of(4));
      memory.write(output.status.overflow, Uint32Array.of(0));
      memory.write(rings.status.count, Uint32Array.of(1, 91, 9898));
      memory.write(rings.ringOffsets, Uint32Array.of(0, 5, 5));
      executeCommandNodesOnCPU(
        nodes.filter(node => /-rings-(capture-loss|prepare-count)$/.test(node.id)),
        memory
      );
      executeCommandNodesOnCPU(
        nodes.filter(node => node.id.includes('-suppress-incomplete-')),
        memory
      );
      // A closed square has one ring, but its retained boundary cannot certify a whole source.
      expect(Array.from(memory.readUint32(rings.status.count))).toEqual([certificate, 91, 9898]);
      expect(Array.from(memory.readUint32(rings.ringOffsets))).toEqual(
        certificate ? [0, 5, 5] : [0, 0, 0]
      );
    }
  } finally {
    device.destroy();
  }
});

it('S1 allows adjacent local and candidate flags written in separate dispatches', () => {
  const {device, graph, view, output, cells} = createFixture();
  const backing = view('uint32', 64);
  const overflow = graph.createDataView(backing.buffer, {
    format: 'uint32',
    length: 1,
    byteOffset: 0
  });
  const candidateOverflow = graph.createDataView(backing.buffer, {
    format: 'uint32',
    length: 1,
    byteOffset: 4
  });
  try {
    expect(() =>
      new GPUCellSetOutline({
        family: 'quadbin',
        cells,
        output: {...output, status: {count: output.status.count, overflow, candidateOverflow}}
      }).getCommandNodes(graph)
    ).not.toThrow();
  } finally {
    device.destroy();
  }
});

it('B3 documents retained-input sizing rather than a monotone bound on the full result', () => {
  // All 16 sorted zoom-2 keys cover the globe: only four north and four south
  // edges remain (longitude wraps). Removing the last corner exposes two extra edges.
  const keys = Array.from({length: 16}, (_, row) => 0x4820ffffffffffffn + (BigInt(row) << 48n));
  expect(outlineQuadbinOnCPU(keys.slice(0, 15))).toHaveLength(10);
  expect(outlineQuadbinOnCPU(keys)).toHaveLength(8);
  for (const file of [
    'cell-set-outline/gpu-cell-set-outline.ts',
    'cell-topology/gpu-cell-compaction.ts'
  ]) {
    const source = readFileSync(
      new URL(`../../../src/gpu-spatial-analysis/${file}`, import.meta.url),
      'utf8'
    );
    expect(source).not.toMatch(/requiredCount[^\n]*lower bound/);
  }
});

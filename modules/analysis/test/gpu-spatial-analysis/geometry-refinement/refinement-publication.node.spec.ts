// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph, type GraphDataView} from '@luma.gl/graph/gpu-core';
import type {GPUGraphVectorFormat} from '@luma.gl/graph/gpu-core';
import {expect, it, vi} from 'vitest';
import {GPULineLoopRemoval} from '../../../src/gpu-spatial-analysis/geometry-refinement/gpu-line-loop-removal';
import {GPUPolygonPrecisionReduction} from '../../../src/gpu-spatial-analysis/geometry-refinement/gpu-precision-reduction';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';
import {useCPUExecutorTestHooks} from '../segment-intersection/cpu-executor-test-hooks';
import {createTopologyOutput} from './topology-output';

useCPUExecutorTestHooks();

function createHarness() {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const memory = createCPUGraphMemory();
  let serial = 0;
  const view = <Format extends GPUGraphVectorFormat>(format: Format, length: number) =>
    createTransientView(graph, `publication-${serial++}`, format, length);
  const views: GraphDataView[] = [];
  const createDataView = graph.createDataView.bind(graph);
  const observe = vi.spyOn(graph, 'createDataView').mockImplementation((...parameters) => {
    const result = createDataView(...parameters);
    views.push(result);
    return result;
  });
  const transient = (id: string) => {
    const found = views.find(candidate => candidate.buffer.id === id);
    if (!found) throw new Error(`missing test scratch ${id}`);
    return found;
  };
  return {device, graph, memory, view, observe, transient};
}

it('loop-removal capacity selection crosses a dispatch boundary and preserves later fitting lines', () => {
  const {device, graph, memory, view, observe, transient} = createHarness();
  const lineCount = 258;
  const output = {
    positions: view('float32x2', 6),
    lineOffsets: view('uint32', lineCount + 1),
    sourceEdges: view('uint32', 6),
    status: view('uint32x4', 1)
  };
  const nodes = new GPULineLoopRemoval({
    lines: {
      kind: 'lines',
      positions: view('float32x2', lineCount * 2),
      lineOffsets: view('uint32', lineCount + 1)
    },
    intersectionCapacity: 1,
    output
  }).getCommandNodes(graph);
  observe.mockRestore();
  // Post-erasure lengths 4,4,...,2,... with capacity 6: keep line 0, skip line 1,
  // then keep line 256. Required=10, written=6; a simple clamp of the original prefix is wrong.
  const state = new Uint32Array(lineCount * 2);
  state[1] = 4;
  state[3] = 4;
  state[256 * 2 + 1] = 2;
  expect(
    Array.from(state)
      .filter((value, index) => index % 2 === 1 && value > 0)
      .every(value => value >= 2)
  ).toBe(true);
  memory.write(transient('line-loop-removal-line-state'), state);
  memory.write(transient('line-loop-removal-split-overflow'), Uint32Array.of(0));
  memory.write(transient('line-loop-removal-split-candidate-overflow'), Uint32Array.of(0));
  const publication = nodes.filter(node =>
    /-(clear-selection|select-lines-\d+|offset-prefix.*|publish-offsets|publish-status)$/.test(
      node.id
    )
  );
  try {
    executeCommandNodesOnCPU(publication, memory);
    expect(Array.from(memory.readUint32(output.status))).toEqual([6, 10, 1, 0]);
    const offsets = memory.readUint32(output.lineOffsets);
    expect(Array.from(offsets.slice(0, 3))).toEqual([0, 4, 4]);
    expect(Array.from(offsets.slice(255))).toEqual([4, 4, 6, 6]);
    memory.write(transient('line-loop-removal-line-state'), new Uint32Array(lineCount * 2));
    executeCommandNodesOnCPU(publication, memory);
    expect(Array.from(memory.readUint32(output.status))).toEqual([0, 0, 0, 0]);
    expect(Array.from(memory.readUint32(output.lineOffsets))).toEqual(Array(lineCount + 1).fill(0));
  } finally {
    device.destroy();
  }
});

it('empty-ring publication reduces every ring and resets its occupancy flag on reuse', () => {
  const {device, graph, memory, view, observe} = createHarness();
  const ringCount = 513;
  const offsets = view('uint32', ringCount + 1);
  const lattice = view('float32', 4);
  const output = createTopologyOutput(view);
  const nodes = new GPUPolygonPrecisionReduction({
    polygons: {
      kind: 'polygons',
      positions: view('float32x2', 1),
      ringOffsets: offsets,
      polygonOffsets: view('uint32', 2),
      featureOffsets: view('uint32', 2)
    },
    gridSize: 1,
    lattice,
    intersectionCapacity: 1,
    output
  }).getCommandNodes(graph);
  observe.mockRestore();
  const publication = nodes.filter(node =>
    /-(clear-nonempty|flag-nonempty|empty-status-\d+)$/.test(node.id)
  );
  try {
    for (const empty of [false, true]) {
      const values = new Uint32Array(ringCount + 1);
      if (!empty) values[ringCount] = 1; // The only nonempty ring is the last one.
      memory.write(offsets, values);
      memory.write(lattice, Float32Array.of(99, 99, 99, 99));
      for (const status of [
        output.status,
        output.nodedSegments.status,
        output.directedEdges.status
      ])
        for (const value of Object.values(status))
          if (value) memory.write(value, Uint32Array.of(99));
      executeCommandNodesOnCPU(publication, memory);
      for (const status of [
        output.status,
        output.nodedSegments.status,
        output.directedEdges.status
      ])
        for (const value of Object.values(status))
          if (value) expect(Array.from(memory.readUint32(value))).toEqual([empty ? 0 : 99]);
      expect(Array.from(memory.readFloat32(lattice))).toEqual(
        empty ? [0, 0, 1, 0] : [99, 99, 99, 99]
      );
    }
  } finally {
    device.destroy();
  }
});

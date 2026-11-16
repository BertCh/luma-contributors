// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {afterAll, afterEach, expect, it, vi} from 'vitest';
import {
  GPULineSmooth,
  getGPULineSmoothParameterValues
} from '../../../src/gpu-spatial-analysis/line-segmentize/gpu-line-smooth';
import * as prefixKernels from '../../../src/gpu-spatial-analysis/line-segmentize/line-segmentize-kernels';
import {createWGSLKernelNode} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources,
  getGPUCommandNodePipelineKeys
} from '../../utils/gpu-contributor-test-utils';
import {
  findReservedIdentifiers,
  stripWGSLComments
} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';
import {useCPUExecutorTestHooks} from '../segment-intersection/cpu-executor-test-hooks';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';

useCPUExecutorTestHooks();
const device = createNullWebGPUDevice();
afterAll(() => device.destroy());
afterEach(() => vi.restoreAllMocks());
function createProps(graph: GPUCommandGraph, closed = false, iterationCount = 2) {
  const capacity = closed ? 9 : 8;
  return {
    id: 'smooth-metadata',
    closed,
    iterationCount,
    positions: createTransientView(graph, 'positions', 'float32x2', closed ? 4 : 2),
    lineOffsets: createTransientView(graph, 'lines', 'uint32', 4),
    parameters: createTransientView(graph, 'parameters', 'float32', 4),
    output: {
      positions: createTransientView(graph, 'output', 'float32x2', capacity),
      lineOffsets: createTransientView(graph, 'output-offsets', 'uint32', 4),
      sourceRows: createTransientView(graph, 'sources', 'uint32', capacity),
      measures: createTransientView(graph, 'measures', 'float32', capacity),
      status: {
        count: createTransientView(graph, 'count', 'uint32', 1),
        overflow: createTransientView(graph, 'overflow', 'uint32', 1)
      }
    }
  };
}

function emulatePrefix() {
  // The serial interpreter cannot synchronize the shared prefix workgroup. Keep this test's
  // independent, small-input prefix stand-in separate from the smoothing kernels being executed.
  vi.spyOn(prefixKernels, 'createLinePrefixNode').mockImplementation((graph, props) =>
    createWGSLKernelNode(graph, {
      id: 'cpu-prefix',
      operation: 'TestPrefix',
      invocationCount: props.lineOffsets.length - 1,
      bindings: [
        {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
        {name: 'offsets', view: props.lineOffsets, type: 'u32', access: 'read'},
        {name: 'measures', view: props.rowMeasures!, type: 'f32', access: 'read_write'}
      ],
      body: `let first = offsets[offsetsOffset + index]; let last = offsets[offsetsOffset + index + 1u]; var distance = 0.0;
for (var row = first; row < last; row++) { if (row > first) { distance += length(vec2f(positions[positionsOffset + row * 2u], positions[positionsOffset + row * 2u + 1u]) - vec2f(positions[positionsOffset + (row - 1u) * 2u], positions[positionsOffset + (row - 1u) * 2u + 1u])); } measures[measuresOffset + row] = distance; }`
    })
  );
}

it('pins source rows, source chainage and every open-line point after two iterations', () => {
  emulatePrefix();
  const graph = new GPUCommandGraph(device);
  const props = createProps(graph);
  const memory = createCPUGraphMemory();
  memory.write(props.positions, new Float32Array([0, 0, 4, 0]));
  memory.write(props.lineOffsets, new Uint32Array([0, 0, 2, 2]));
  memory.write(props.parameters, getGPULineSmoothParameterValues());
  const nodes = new GPULineSmooth(props).getCommandNodes(graph);
  executeCommandNodesOnCPU(nodes, memory);
  const expected = [0, 0.25, 0.75, 1.5, 2.5, 3.25, 3.75, 4];
  expect(Array.from(memory.readFloat32(props.output.positions))).toEqual(
    expected.flatMap(value => [value, 0])
  );
  expect(Array.from(memory.readFloat32(props.output.measures))).toEqual(expected);
  expect(Array.from(memory.readUint32(props.output.sourceRows))).toEqual([0, 0, 0, 0, 0, 0, 0, 1]);
  expect(Array.from(memory.readUint32(props.output.lineOffsets))).toEqual([0, 0, 8, 8]);
});

it('keeps ordered ring provenance and unwraps chainage through the closing corner', () => {
  emulatePrefix();
  const graph = new GPUCommandGraph(device);
  const props = createProps(graph, true, 1);
  const memory = createCPUGraphMemory();
  memory.write(props.positions, new Float32Array([0, 0, 4, 0, 4, 4, 0, 4]));
  memory.write(props.lineOffsets, new Uint32Array([0, 0, 4, 4]));
  memory.write(props.parameters, getGPULineSmoothParameterValues());
  executeCommandNodesOnCPU(new GPULineSmooth(props).getCommandNodes(graph), memory);
  expect(Array.from(memory.readUint32(props.output.sourceRows))).toEqual([
    0, 0, 1, 1, 2, 2, 3, 3, 0
  ]);
  expect(Array.from(memory.readFloat32(props.output.measures))).toEqual([
    1, 3, 5, 7, 9, 11, 13, 15, 17
  ]);
  expect(Array.from(memory.readFloat32(props.output.positions))).toEqual([
    1, 0, 3, 0, 4, 1, 4, 3, 3, 4, 1, 4, 0, 3, 0, 1, 1, 0
  ]);
});

it.each([
  false,
  true
])('shares one level shader and pipeline at every iteration (closed=%s)', closed => {
  const counts = [];
  for (const iterationCount of [2, 3, 10]) {
    const graph = new GPUCommandGraph(device);
    const nodes = new GPULineSmooth(createProps(graph, closed, iterationCount)).getCommandNodes(
      graph
    );
    const levels = nodes.filter(node => /(?:metadata-)?level-\d+$/.test(node.id));
    const sources = getGPUCommandNodeKernelSources(levels);
    counts.push(new Set(getGPUCommandNodePipelineKeys(levels)).size);
    expect(new Set(sources.map(({source}) => source)).size).toBe(2);
    for (const {source} of getGPUCommandNodeKernelSources(nodes))
      expect(findReservedIdentifiers(stripWGSLComments(source))).toEqual([]);
  }
  expect(counts).toEqual([2, 2, 2]);
});

it('preflights metadata scratch before allocation and accepts packed separate-pass status words', () => {
  const graph = new GPUCommandGraph(device);
  const props = createProps(graph);
  const packed = createTransientView(graph, 'packed', 'uint32', 65);
  props.output.status.count = graph.createDataView(packed.buffer, {format: 'uint32', length: 1});
  props.output.status.overflow = graph.createDataView(packed.buffer, {
    format: 'uint32',
    length: 1,
    byteOffset: 4
  });
  expect(() => new GPULineSmooth(props).getCommandNodes(graph)).not.toThrow();
  const other = new GPUCommandGraph(device);
  const oversized = createProps(other);
  oversized.output.positions = createTransientView(other, 'large-output', 'float32x2', 8_388_609);
  oversized.output.sourceRows = createTransientView(other, 'large-source', 'uint32', 8_388_609);
  oversized.output.measures = createTransientView(other, 'large-measure', 'float32', 8_388_609);
  const contributor = new GPULineSmooth(oversized);
  const allocate = vi.spyOn(other, 'createTransientBuffer');
  expect(() => contributor.getCommandNodes(other)).toThrow(/levelMetadata.*bytes/);
  expect(allocate).not.toHaveBeenCalled();
});

it('preserves a one-row closed prefix through multiple levels with repeated closure and unequal edges', () => {
  const graph = new GPUCommandGraph(device);
  const props = createProps(graph, true, 2);
  props.positions = createTransientView(graph, 'repeated-ring', 'float32x2', 5);
  props.output.positions = createTransientView(graph, 'short-output', 'float32x2', 1);
  props.output.sourceRows = createTransientView(graph, 'short-sources', 'uint32', 1);
  props.output.measures = createTransientView(graph, 'short-measures', 'float32', 1);
  emulatePrefix();
  const nodes = new GPULineSmooth(props).getCommandNodes(graph);
  const memory = createCPUGraphMemory();
  memory.write(props.positions, new Float32Array([0, 0, 6, 0, 6, 2, 0, 2, 0, 0]));
  memory.write(props.lineOffsets, new Uint32Array([0, 0, 5, 5]));
  memory.write(props.parameters, getGPULineSmoothParameterValues({ratio: 0.25}));
  executeCommandNodesOnCPU(nodes, memory);
  expect(Array.from(memory.readFloat32(props.output.positions))).toEqual([2.25, 0]);
  expect(Array.from(memory.readFloat32(props.output.measures))).toEqual([2.25]);
  expect(Array.from(memory.readUint32(props.output.sourceRows))).toEqual([0]);
  expect(Array.from(memory.readUint32(props.output.lineOffsets))).toEqual([0, 0, 1, 1]);
  expect(memory.readUint32(props.output.status.overflow)[0]).toBe(1);
});

it('matches all closed two-iteration coordinates, source rows and unequal-edge measures', () => {
  const graph = new GPUCommandGraph(device);
  const props = createProps(graph, true, 2);
  props.positions = createTransientView(graph, 'repeated-ring', 'float32x2', 5);
  props.output.positions = createTransientView(graph, 'full-output', 'float32x2', 17);
  props.output.sourceRows = createTransientView(graph, 'full-sources', 'uint32', 17);
  props.output.measures = createTransientView(graph, 'full-measures', 'float32', 17);
  emulatePrefix();
  const nodes = new GPULineSmooth(props).getCommandNodes(graph);
  const memory = createCPUGraphMemory();
  memory.write(props.positions, new Float32Array([0, 0, 6, 0, 6, 2, 0, 2, 0, 0]));
  memory.write(props.lineOffsets, new Uint32Array([0, 0, 5, 5]));
  memory.write(props.parameters, getGPULineSmoothParameterValues({ratio: 0.25}));
  executeCommandNodesOnCPU(nodes, memory);
  expect(Array.from(memory.readFloat32(props.output.positions))).toEqual([
    2.25, 0, 3.75, 0, 4.875, 0.125, 5.625, 0.375, 6, 0.75, 6, 1.25, 5.625, 1.625, 4.875, 1.875,
    3.75, 2, 2.25, 2, 1.125, 1.875, 0.375, 1.625, 0, 1.25, 0, 0.75, 0.375, 0.375, 1.125, 0.125,
    2.25, 0
  ]);
  expect(Array.from(memory.readFloat32(props.output.measures))).toEqual([
    2.25, 3.75, 5, 6, 6.75, 7.25, 8, 9, 10.25, 11.75, 13, 14, 14.75, 15.25, 16, 17, 18.25
  ]);
  expect(Array.from(memory.readUint32(props.output.sourceRows))).toEqual([
    0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 0, 0
  ]);
  expect(memory.readUint32(props.output.status.overflow)[0]).toBe(0);
});

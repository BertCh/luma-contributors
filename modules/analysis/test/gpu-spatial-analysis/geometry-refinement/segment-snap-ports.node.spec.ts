// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {computeSegmentSnapOnCPU} from '../../../src/gpu-spatial-analysis/geometry-refinement/cpu-segment-snap-operation';
import {decodeGPUSegmentSnapIssueCode} from '../../../src/gpu-spatial-analysis/geometry-refinement/gpu-segment-snap';
import {GPUSegmentSnap} from '../../../src/gpu-spatial-analysis/geometry-refinement/gpu-segment-snap';
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  recordWGSLKernelNodes,
  getWGSLKernelNodeRecord
} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';

function createFixture() {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  recordWGSLKernelNodes(graph);
  const props = {
    id: 'port-review',
    positions: createTransientView(graph, 'input', 'float32x2', 2),
    lineOffsets: createTransientView(graph, 'offsets', 'uint32', 2),
    referencePositions: createTransientView(graph, 'references', 'float32x2', 0),
    tolerance: 0,
    maximumVerticesPerLine: 2,
    output: {
      positions: createTransientView(graph, 'output', 'float32x2', 2),
      sourceEdges: createTransientView(graph, 'edges', 'uint32', 2),
      counts: createTransientView(graph, 'counts', 'uint32x4', 1),
      issueCode: createTransientView(graph, 'issues', 'uint32', 1),
      validity: createTransientView(graph, 'validity', 'uint32', 1)
    }
  };
  return {graph, props};
}

it.each([
  {candidateCapacity: 1, maximumVerticesPerLine: 8, counts: [0, 0, 0, 0], candidate: 1},
  {candidateCapacity: 2, maximumVerticesPerLine: 2, counts: [2, 4, 1, 0], candidate: 0}
])('separates candidate loss from vertex capacity: %o', options => {
  const result = computeSegmentSnapOnCPU(
    {
      positions: Float32Array.of(0, 0, 3, 0),
      lineOffsets: Uint32Array.of(0, 2),
      referencePositions: Float32Array.of(1, 0, 2, 0)
    },
    {tolerance: 0, candidateOverflowPolicy: 'incomplete', ...options}
  );
  expect(Array.from(result.counts)).toEqual(options.counts);
  expect(Array.from(result.candidateOverflow)).toEqual([options.candidate]);
  expect(Array.from(result.issueCode)).toEqual([4]);
  expect(Array.from(result.validity)).toEqual([0]);
});

it('keeps insertion and diagnostics on construction-time output views after caller mutation', () => {
  const {graph, props} = createFixture();
  const original = props.output.counts;
  const contributor = new GPUSegmentSnap(props);
  props.output.counts = createTransientView(graph, 'replacement', 'uint32x4', 1);
  const memory = createCPUGraphMemory();
  memory.write(props.positions, Float32Array.of(5, 5, 5, 5));
  memory.write(props.lineOffsets, Uint32Array.of(0, 2));
  executeCommandNodesOnCPU(contributor.getCommandNodes(graph), memory);
  expect(Array.from(memory.readUint32(original))).toEqual([1, 1, 0, 0]);
  expect(Array.from(memory.readUint32(props.output.counts))).toEqual([0, 0, 0, 0]);
  expect(Array.from(memory.readUint32(props.output.issueCode))).toEqual([1]);
  expect(Array.from(memory.readUint32(props.output.validity))).toEqual([0]);
});

it('uses compatible storage access when counts and validity share aligned backing storage', () => {
  const {graph, props} = createFixture();
  const backing = createTransientView(graph, 'packed-status', 'uint32', 128);
  props.output.counts = graph.createDataView(backing.buffer, {
    format: 'uint32x4',
    length: 1,
    byteOffset: 0
  });
  props.output.validity = graph.createDataView(backing.buffer, {
    format: 'uint32',
    length: 1,
    byteOffset: 256
  });
  new GPUSegmentSnap(props).getCommandNodes(graph);
  const record = getWGSLKernelNodeRecord(graph, 'port-review-line-validity')!;
  expect(record.bindings.find(binding => binding.name === 'counts')!.access).toBe('read_write');
});

it('keeps successful lines with boolean validity and separates collapse and incomplete codes', () => {
  const input = {
    positions: Float32Array.of(0, 0, 2, 0, 5, 5, 5, 5, Number.NaN, 0),
    lineOffsets: Uint32Array.of(0, 2, 4, 5, 5),
    referencePositions: new Float32Array()
  };
  const result = computeSegmentSnapOnCPU(input, {tolerance: 0, maximumVerticesPerLine: 2});
  expect(Array.from(result.validity)).toEqual([1, 0, 0, 1]);
  expect(Array.from(result.issueCode)).toEqual([0, 1, 2, 0]);
  const truncated = computeSegmentSnapOnCPU(input, {tolerance: 0, maximumVerticesPerLine: 1});
  expect(Array.from(truncated.issueCode)).toEqual([4, 1, 2, 0]);
  expect(Array.from(truncated.validity)).toEqual([0, 0, 0, 1]);
  expect(decodeGPUSegmentSnapIssueCode(5)).toEqual({
    collapsed: true,
    nonFinite: false,
    incomplete: true
  });
});

// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {afterAll, expect, it} from 'vitest';
import {
  GPUOutlineGeometry,
  getGPUOutlineGeometryParameterValues
} from '../../../src/gpu-spatial-analysis/outline-geometry/gpu-outline-geometry';
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
function createProps(graph: GPUCommandGraph, rows = 2, capacity = 21) {
  return {
    id: 'compact-outline',
    geometryType: 'lines' as const,
    joinSegments: 4,
    positions: createTransientView(graph, 'positions', 'float32x2', rows),
    lineOffsets: createTransientView(graph, 'lines', 'uint32', 2),
    parameters: createTransientView(graph, 'parameters', 'float32', 4),
    output: {
      layout: 'compact' as const,
      positions: createTransientView(graph, 'output', 'float32x2', capacity),
      vertexOffsets: createTransientView(graph, 'offsets', 'uint32', rows + 1),
      sourceRows: createTransientView(graph, 'sources', 'uint32', capacity),
      status: {
        count: createTransientView(graph, 'count', 'uint32', 1),
        requiredCount: createTransientView(graph, 'required', 'uint32', 1),
        overflow: createTransientView(graph, 'overflow', 'uint32', 1)
      }
    }
  };
}

export const COMPACT_OUTLINE_EXPECTED = [
  0, 0, 1, 0, 0, 1, 0, 0, 0, 1, -1, 0, 0, 0, -1, 0, 0, -1, 0, 0, 0, -1, 1, 0, 0, 1, 0, -1, 2, -1, 0,
  1, 2, -1, 2, 1, 2, 0, 3, 0, 2, 1
];

it('removes end quads, preserves ordered triangles, and truncates only at triangle boundaries', () => {
  const graph = new GPUCommandGraph(device);
  const props = createProps(graph);
  const contributor = new GPUOutlineGeometry(props);
  const nodes = contributor.getCommandNodes(graph);
  expect(nodes.length).toBeLessThanOrEqual(contributor.estimatedCommandNodeCount);
  const memory = createCPUGraphMemory();
  memory.write(props.positions, new Float32Array([0, 0, 2, 0]));
  memory.write(props.lineOffsets, new Uint32Array([0, 2]));
  memory.write(props.parameters, getGPUOutlineGeometryParameterValues({distance: 1}));
  executeCommandNodesOnCPU(nodes, memory);
  const actual = memory.readFloat32(props.output.positions);
  COMPACT_OUTLINE_EXPECTED.forEach((value, index) => expect(actual[index]).toBeCloseTo(value, 5));
  expect(Array.from(memory.readUint32(props.output.vertexOffsets))).toEqual([0, 18, 21]);
  expect(Array.from(memory.readUint32(props.output.sourceRows))).toEqual([
    ...Array(18).fill(0),
    1,
    1,
    1
  ]);
  expect(memory.readUint32(props.output.status.count)[0]).toBe(21);
  expect(memory.readUint32(props.output.status.requiredCount)[0]).toBe(30);
  expect(memory.readUint32(props.output.status.overflow)[0]).toBe(1);
  memory.write(props.parameters, getGPUOutlineGeometryParameterValues({distance: 0}));
  executeCommandNodesOnCPU(nodes, memory);
  expect(Array.from(memory.readUint32(props.output.vertexOffsets))).toEqual([0, 0, 0]);
  expect(memory.readUint32(props.output.status.count)[0]).toBe(0);
  expect(memory.readUint32(props.output.status.requiredCount)[0]).toBe(0);
  expect(memory.readUint32(props.output.status.overflow)[0]).toBe(0);
});

it('validates triangle capacities and co-bound output aliases', () => {
  const graph = new GPUCommandGraph(device);
  expect(() => new GPUOutlineGeometry(createProps(graph, 2, 20))).toThrow(/divisible by three/);
  const validGraph = new GPUCommandGraph(device);
  const props = createProps(validGraph);
  expect(
    () =>
      new GPUOutlineGeometry({
        ...props,
        output: {
          ...props.output,
          sourceRows: validGraph.createDataView(props.output.positions.buffer, {
            format: 'uint32',
            length: 21
          })
        }
      })
  ).toThrow(/byte ranges must not overlap/);
});

it('uses a constant number of generation pipelines as geometry grows', () => {
  for (const rows of [2, 256, 65_537]) {
    const graph = new GPUCommandGraph(device);
    const contributor = new GPUOutlineGeometry(createProps(graph, rows));
    const nodes = contributor.getCommandNodes(graph);
    const generation = nodes.filter(
      node => node.id === 'compact-outline-count' || node.id === 'compact-outline-generate'
    );
    expect(getGPUCommandNodePipelineKeys(generation).size).toBe(2);
    for (const {source} of getGPUCommandNodeKernelSources(nodes))
      expect(findReservedIdentifiers(stripWGSLComments(source))).toEqual([]);
  }
});

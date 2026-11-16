// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it, vi} from 'vitest';
import {WgslReflect} from 'wgsl_reflect';
import {GPUConstrainedDelaunayTriangulation} from '../../../src/gpu-spatial-analysis/polygon-triangulation/gpu-constrained-delaunay-triangulation';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources,
  getGPUCommandNodePipelineKeys
} from '../../utils/gpu-contributor-test-utils';
import {WGSL_RESERVED_WORDS} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';

import {useCPUExecutorTestHooks} from '../segment-intersection/cpu-executor-test-hooks';
import {createCPUGraphMemory} from '../segment-intersection/wgsl-graph-executor';
import {executeGeometryNodesOnCPU} from '../polygon-triangulation/geometry-cpu-executor';
import {getConstrainedDelaunayParallelNodes} from '../../../src/gpu-spatial-analysis/polygon-triangulation/constrained-delaunay-parallel';
import {computePolygonTriangulationOnCPU} from '../../../src/analysis-core/polygon-triangulation/cpu-polygon-triangulation-operation';
import {computeConstrainedDelaunayTriangulationOnCPU} from '../../../src/gpu-spatial-analysis/polygon-triangulation/cpu-constrained-delaunay-triangulation-operation';

useCPUExecutorTestHooks();

function createContributor(
  graph: GPUCommandGraph,
  maximumFlips: number,
  indices = 12,
  pointCount = 6
) {
  return new GPUConstrainedDelaunayTriangulation({
    id: 'cdt',
    flipMethod: 'parallel',
    maximumFlips,
    largePolygonMethod: 'none',
    positions: createTransientView(graph, 'positions', 'float32x2', pointCount),
    featureOffsets: createTransientView(graph, 'features', 'uint32', 2),
    polygonOffsets: createTransientView(graph, 'polygons', 'uint32', 2),
    ringOffsets: createTransientView(graph, 'rings', 'uint32', 2),
    output: {
      indices: createTransientView(graph, 'indices', 'uint32', indices),
      valid: createTransientView(graph, 'valid', 'uint32', 1),
      status: {converged: createTransientView(graph, 'converged', 'uint32', 1)}
    }
  });
}

it('keeps a constant shader/pipeline count across flip rounds and bounds the node estimate', () => {
  const counts: number[] = [];
  for (const maximumFlips of [1, 7, 31]) {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    const contributor = createContributor(graph, maximumFlips);
    const nodes = contributor.getCommandNodes(graph);
    expect(nodes.length).toBeLessThanOrEqual(contributor.estimatedCommandNodeCount);
    expect(() => graph.add(nodes)).not.toThrow();
    counts.push(new Set(getGPUCommandNodePipelineKeys(nodes)).size);
    for (const {source} of getGPUCommandNodeKernelSources(nodes)) {
      const identifiers =
        source.replace(/\/\/[^\n]*|\/\*[\s\S]*?\*\//g, '').match(/[A-Za-z_][A-Za-z_0-9]*/g) ?? [];
      expect(identifiers.filter(word => WGSL_RESERVED_WORDS.has(word))).toEqual([]);
      expect(source.replace(/\/\/[^\n]*|\/\*[\s\S]*?\*\//g, '')).not.toMatch(
        /\b(?:nan|inf|NaN|Infinity)\b/
      );
      expect(() => new WgslReflect(source)).not.toThrow();
    }
    device.destroy();
  }
  expect(new Set(counts).size).toBe(1);
});

it('preflights output bindings and parallel scratch before triangulation mutates the graph', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const contributor = createContributor(graph, 1, 33_554_433);
  const allocate = vi.spyOn(graph, 'createTransientBuffer');
  expect(() => contributor.getCommandNodes(graph)).toThrow(/needs 134217732 bytes/);
  expect(allocate).not.toHaveBeenCalled();
  vi.restoreAllMocks();
  device.destroy();
});

it('executes parallel WGSL flips and adjacency repair against the independent CPU oracle', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const contributor = createContributor(graph, 8);
  const props = contributor.props;
  const input = {
    positions: new Float32Array([0, 0, 4, 0, 4, 4, 0, 1, 10, 10, 12, 12]),
    ringOffsets: new Uint32Array([0, 4]),
    polygonOffsets: new Uint32Array([0, 1]),
    featureOffsets: new Uint32Array([0, 1])
  };
  const initial = computePolygonTriangulationOnCPU(input, {indexCapacity: 12});
  const expected = computeConstrainedDelaunayTriangulationOnCPU(input, {indexCapacity: 12});
  expect(Array.from(initial.indices)).not.toEqual(Array.from(expected.indices));
  const offsets = createTransientView(graph, 'offsets', 'uint32', 2);
  const memory = createCPUGraphMemory();
  memory.write(props.positions, input.positions);
  memory.write(props.output.indices, initial.indices);
  memory.write(props.output.valid, initial.valid);
  memory.write(offsets, initial.triangleOffsets);
  const nodes = getConstrainedDelaunayParallelNodes(graph, props, offsets);
  executeGeometryNodesOnCPU(nodes, memory);
  expect(Array.from(memory.readUint32(props.output.indices))).toEqual(Array.from(expected.indices));
  expect(Array.from(memory.readUint32(props.output.status.converged))).toEqual([1]);
  expect(memory.workgroupCounts.get('cdt-parallel-round-8-evaluate')).toBe(0);
  device.destroy();
});

it('repairs indexed adjacency through several ordered flips', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const contributor = createContributor(graph, 32, 18, 8);
  const props = contributor.props;
  const input = {
    positions: new Float32Array([0, 0, 4, 0, 6, 1, 7, 4, 5, 7, 2, 8, -1, 5, -2, 2]),
    ringOffsets: new Uint32Array([0, 8]),
    polygonOffsets: new Uint32Array([0, 1]),
    featureOffsets: new Uint32Array([0, 1])
  };
  const initial = computePolygonTriangulationOnCPU(input, {indexCapacity: 18});
  const expected = computeConstrainedDelaunayTriangulationOnCPU(input, {indexCapacity: 18});
  const offsets = createTransientView(graph, 'offsets', 'uint32', 2);
  const memory = createCPUGraphMemory();
  memory.write(props.positions, input.positions);
  memory.write(props.output.indices, initial.indices);
  memory.write(props.output.valid, initial.valid);
  memory.write(offsets, initial.triangleOffsets);
  const createView = vi.spyOn(graph, 'createDataView');
  executeGeometryNodesOnCPU(getConstrainedDelaunayParallelNodes(graph, props, offsets), memory);
  const iterations = createView.mock.results.find(
    result => result.type === 'return' && result.value.buffer.id === 'cdt-parallel-iterations'
  )!.value;
  expect(memory.readUint32(iterations)[0]).toBeGreaterThan(1);
  expect(Array.from(memory.readUint32(props.output.indices))).toEqual(Array.from(expected.indices));
  expect(Array.from(memory.readUint32(props.output.status.converged))).toEqual([1]);
  vi.restoreAllMocks();
  device.destroy();
});

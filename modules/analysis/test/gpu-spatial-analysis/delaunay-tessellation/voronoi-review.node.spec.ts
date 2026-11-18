// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPUVoronoiDiagram} from '../../../src/gpu-spatial-analysis/delaunay-tessellation/gpu-voronoi-diagram';
import {TRIANGLE_CIRCUMCIRCLE_WGSL} from '../../../src/gpu-spatial-analysis/shared/triangle-circumcircle-wgsl';
import {createWGSLKernelNode} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {auditWGSLModule} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';

function getResourceView(node: unknown, index: number): GraphDataView {
  return (node as {resources: {buffer: GraphDataView}[]}).resources[index].buffer;
}

function createFixture(graph: GPUCommandGraph, positionsValues: number[], triangle = [0, 1, 2]) {
  const props = {
    positions: createTransientView(graph, 'positions', 'float32x2', 3),
    triangles: createTransientView(graph, 'triangles', 'uint32x3', 1),
    triangleCount: createTransientView(graph, 'triangle-count', 'uint32', 1),
    clipBounds: createTransientView(graph, 'bounds', 'float32x4', 1),
    output: {
      segments: createTransientView(graph, 'segments', 'float32x4', 3),
      siteIds: createTransientView(graph, 'sites', 'uint32x2', 3),
      status: {
        count: createTransientView(graph, 'count', 'uint32', 1),
        requiredCount: createTransientView(graph, 'required', 'uint32', 1),
        overflow: createTransientView(graph, 'overflow', 'uint32', 1)
      }
    }
  };
  const memory = createCPUGraphMemory();
  memory.write(props.positions, Float32Array.from(positionsValues));
  memory.write(props.triangles, Uint32Array.from(triangle));
  memory.write(props.triangleCount, Uint32Array.of(1));
  const nodes = new GPUVoronoiDiagram(props).getCommandNodes(graph);
  return {
    props,
    memory,
    centers: nodes.find(node => node.id.endsWith('-circumcenters'))!,
    clip: nodes.find(node => node.id.endsWith('-clip-segments'))!
  };
}

it('review circumcircle and ray-clipping shader modules pass the static WGSL audit', () => {
  const device = createNullWebGPUDevice();
  try {
    const fixture = createFixture(new GPUCommandGraph(device), [0, 0, 1, 0, 0, 1]);
    for (const kernel of getGPUCommandNodeKernelSources([fixture.centers, fixture.clip])) {
      expect(auditWGSLModule(kernel.source)).toEqual([]);
    }
  } finally {
    device.destroy();
  }
});

it('R2 independently scaled edge equations preserve the short-axis center component', () => {
  const device = createNullWebGPUDevice();
  try {
    const fixture = createFixture(new GPUCommandGraph(device), [0, 0, 2 ** 40, 0, 0, 2 ** -40]);
    executeCommandNodesOnCPU([fixture.centers], fixture.memory);
    const center = fixture.memory.readFloat32(getResourceView(fixture.centers, 3));
    expect(center[2]).toBe(1);
    expect(center[0] / 2 ** 40).toBe(0.5);
    expect(center[1] / 2 ** -40).toBe(0.5);
  } finally {
    device.destroy();
  }
});

it('R3 restores a finite center even when its origin-relative offset and radius overflow', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const positions = createTransientView(graph, 'positions', 'float32x2', 3);
    const output = createTransientView(graph, 'circle', 'float32', 3);
    const memory = createCPUGraphMemory();
    const scale = 2 ** 127;
    memory.write(
      positions,
      Float32Array.of(-1.5 * scale, 0, -scale, 1.5 * scale, -scale, -1.5 * scale)
    );
    // Isolate circumcircle arithmetic with the analytic clockwise sign. The interpreter cannot
    // reliably execute the existing integer orientation fallback at this exponent; headless uses it unchanged.
    const node = createWGSLKernelNode(graph, {
      id: 'circle',
      operation: 'CircumcircleReview',
      invocationCount: 1,
      bindings: [
        {name: 'positions', view: positions, type: 'f32', access: 'read'},
        {name: 'output', view: output, type: 'f32', access: 'read_write'}
      ],
      declarations: `${TRIANGLE_CIRCUMCIRCLE_WGSL}
fn orientSign(first: vec2f, second: vec2f, third: vec2f) -> i32 { return -1; }
fn orientIsFinite(value: f32) -> bool { return abs(value) <= 3.402823466e38; }`,
      body: `let circle = getTriangleCircumcircle(vec2f(positions[0],positions[1]),vec2f(positions[2],positions[3]),vec2f(positions[4],positions[5]));
output[0] = circle.center.x; output[1] = circle.center.y; output[2] = circle.radius;`
    });
    executeCommandNodesOnCPU([node], memory);
    const result = memory.readFloat32(output);
    expect(Number.isFinite(result[0])).toBe(true);
    expect(result[0] / scale).toBeCloseTo(1, 5);
    expect(result[1] / scale).toBeCloseTo(0, 5);
    expect(result[2]).toBe(Infinity);
  } finally {
    device.destroy();
  }
});

it('R4 overflowing hull-edge differences preserve all three outward rays in either winding', () => {
  const device = createNullWebGPUDevice();
  try {
    for (const triangle of [
      [0, 1, 2],
      [0, 2, 1]
    ]) {
      const scale = 2 ** 127;
      const fixture = createFixture(
        new GPUCommandGraph(device),
        [-scale, 0, scale, 0, 0, scale],
        triangle
      );
      fixture.memory.write(fixture.props.clipBounds, Float32Array.of(-1, -1, 1, 1));
      fixture.memory.write(
        getResourceView(fixture.clip, 3),
        Float32Array.of(0, 0, triangle[1] === 1 ? 1 : -1)
      );
      fixture.memory.write(getResourceView(fixture.clip, 4), Uint32Array.of(1, 1, 1));
      fixture.memory.write(getResourceView(fixture.clip, 5), Uint32Array.of(1, 1, 1));
      executeCommandNodesOnCPU([fixture.clip], fixture.memory);
      expect(Array.from(fixture.memory.readUint32(getResourceView(fixture.clip, 5)))).toEqual([
        1, 1, 1
      ]);
      const edge = triangle[1] === 1 ? 0 : 2;
      const segments = fixture.memory.readFloat32(getResourceView(fixture.clip, 6));
      expect(Array.from(segments.slice(4 * edge, 4 * edge + 4))).toEqual([0, 0, 0, -1]);
    }
  } finally {
    device.destroy();
  }
});

it('R5 hull clipping reaches finite bounds farther than the former ray cap', () => {
  const device = createNullWebGPUDevice();
  try {
    const fixture = createFixture(new GPUCommandGraph(device), [0, 0, 1, 0, 0, 1]);
    fixture.memory.write(fixture.props.clipBounds, Float32Array.of(0, -3.25e38, 1, -3.2e38));
    fixture.memory.write(getResourceView(fixture.clip, 3), Float32Array.of(0.5, 0.5, 1));
    fixture.memory.write(getResourceView(fixture.clip, 4), Uint32Array.of(1, 1, 1));
    fixture.memory.write(getResourceView(fixture.clip, 5), Uint32Array.of(1, 1, 1));
    executeCommandNodesOnCPU([fixture.clip], fixture.memory);
    expect(Array.from(fixture.memory.readUint32(getResourceView(fixture.clip, 5)))).toEqual([
      1, 0, 0
    ]);
    const segment = fixture.memory.readFloat32(getResourceView(fixture.clip, 6));
    expect(segment[0]).toBe(0.5);
    expect(segment[2]).toBe(0.5);
    expect(segment[1] / Math.fround(-3.2e38)).toBeCloseTo(1, 5);
    expect(segment[3] / Math.fround(-3.25e38)).toBeCloseTo(1, 5);
  } finally {
    device.destroy();
  }
});

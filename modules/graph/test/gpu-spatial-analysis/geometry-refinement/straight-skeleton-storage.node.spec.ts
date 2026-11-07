// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUCommandGraph,
  type GraphDataView
} from '@luma-contributors/graph/gpu-core';
import {expect, it} from 'vitest';
import {
  GPUConvexStraightSkeleton,
  GPUVectorStraightSkeleton
} from '../../../src/gpu-spatial-analysis/geometry-refinement';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';
import {useCPUExecutorTestHooks} from '../segment-intersection/cpu-executor-test-hooks';
import {auditWGSLModule} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';
import fixtures from '../../oracles/fixtures/geometry-straight-skeleton.json';

useCPUExecutorTestHooks();

function createFixture(
  coordinates: number[],
  rings: number[],
  polygons: number[],
  maximumVertices = 32,
  maximumEvents = 64
) {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const memory = createCPUGraphMemory();
  const polygonCount = polygons.length - 1;
  const slots = maximumVertices + 2 * maximumEvents;
  const segmentCapacity = 128;
  const positions = createTransientView(graph, 'positions', 'float32x2', coordinates.length / 2);
  const ringOffsets = createTransientView(graph, 'rings', 'uint32', rings.length);
  const polygonOffsets = createTransientView(graph, 'polygons', 'uint32', polygons.length);
  const featureOffsets = createTransientView(graph, 'features', 'uint32', polygons.length);
  memory.write(positions, Float32Array.from(coordinates));
  memory.write(ringOffsets, Uint32Array.from(rings));
  memory.write(polygonOffsets, Uint32Array.from(polygons));
  memory.write(featureOffsets, Uint32Array.from(polygons.map((_, index) => index)));
  const output = {
    segments: createTransientView(graph, 'segments', 'float32x4', polygonCount * segmentCapacity),
    sourceEdges: createTransientView(
      graph,
      'source-edges',
      'uint32x2',
      polygonCount * segmentCapacity
    ),
    offsetPositions: createTransientView(graph, 'offsets', 'float32x2', polygonCount * slots),
    offsetLinks: createTransientView(graph, 'links', 'uint32x2', polygonCount * slots),
    status: createTransientView(graph, 'status', 'uint32x4', polygonCount)
  };
  const contributor = new GPUVectorStraightSkeleton({
    positions,
    ringOffsets,
    polygonOffsets,
    featureOffsets,
    maximumVertices,
    maximumEvents,
    segmentCapacity,
    offsetDistance: 0.25,
    output
  });
  const nodes = contributor.getCommandNodes(graph);
  return {graph, memory, nodes, output, positions, ringOffsets, slots, segmentCapacity};
}

it.each(
  fixtures.cases
)('compact wavefront preserves complete CGAL arcs and offsets: $name', fixture => {
  const {memory, nodes, output, slots} = createFixture(fixture.positions, fixture.ringOffsets, [
    0,
    fixture.ringOffsets.length - 1
  ]);
  executeCommandNodesOnCPU(nodes, memory);
  const status = memory.readUint32(output.status);
  expect(status[3]).toBe(0);
  const segments = memory.readFloat32(output.segments);
  const actual = Array.from({length: status[0]}, (_, index) =>
    Array.from(segments.slice(index * 4, index * 4 + 4))
  );
  const expected = fixture.arcs
    .map(arc => arc.slice(0, 4))
    .filter(arc => Math.hypot(arc[0] - arc[2], arc[1] - arc[3]) > 1e-5);
  expect(actual.length).toBe(expected.length);
  for (const arc of expected) {
    expect(
      actual.some(
        candidate =>
          Math.min(
            Math.hypot(
              candidate[0] - arc[0],
              candidate[1] - arc[1],
              candidate[2] - arc[2],
              candidate[3] - arc[3]
            ),
            Math.hypot(
              candidate[0] - arc[2],
              candidate[1] - arc[3],
              candidate[2] - arc[0],
              candidate[3] - arc[1]
            )
          ) < 1e-3
      )
    ).toBe(true);
  }
  const links = memory.readUint32(output.offsetLinks);
  const coordinates = memory.readFloat32(output.offsetPositions);
  const expectedPositions = fixture.offsets.flat();
  let activeCount = 0;
  for (let vertex = 0; vertex < slots; vertex++) {
    const previous = links[vertex * 2];
    const next = links[vertex * 2 + 1];
    if (previous === 0xffffffff) {
      expect(next).toBe(0xffffffff);
      continue;
    }
    activeCount++;
    expect(links[previous * 2 + 1]).toBe(vertex);
    expect(links[next * 2]).toBe(vertex);
    expect(
      expectedPositions.some(
        point =>
          Math.hypot(point[0] - coordinates[vertex * 2], point[1] - coordinates[vertex * 2 + 1]) <
          1e-3
      )
    ).toBe(true);
  }
  expect(activeCount).toBe(expectedPositions.length);
  expect(status[2]).toBe(activeCount);
});

it('packed wavefront ranges isolate mixed closed rings and clear old snapshots after reuse', () => {
  const fixture = fixtures.cases.find(value => value.name === 'concave-C')!;
  const firstCoordinates = [...fixture.positions, ...fixture.positions.slice(0, 2)];
  const secondCoordinates = [20, 0, 24, 0, 24, 2, 20, 2];
  const firstLength = firstCoordinates.length / 2;
  const {memory, nodes, output, positions, ringOffsets, slots} = createFixture(
    [...firstCoordinates, ...secondCoordinates],
    [0, firstLength, firstLength + 4],
    [0, 1, 2]
  );
  for (const {source, constants} of getGPUCommandNodeKernelSources(nodes)) {
    expect(auditWGSLModule(source, constants)).toEqual([]);
  }
  executeCommandNodesOnCPU(nodes, memory);
  expect(memory.readUint32(output.status)[3]).toBe(0);
  expect(Array.from(memory.readUint32(output.status).slice(4))).toEqual([5, 5, 4, 0]);
  // Shrink both polygons inside their retained input allocation, changing the packed base.
  memory.write(positions, Float32Array.from([0, 0, 4, 0, 4, 2, 0, 2, ...secondCoordinates]));
  memory.write(ringOffsets, Uint32Array.from([0, 4, 8]));
  executeCommandNodesOnCPU(nodes, memory);
  expect(Array.from(memory.readUint32(output.status))).toEqual([5, 5, 4, 0, 5, 5, 4, 0]);
  const links = memory.readUint32(output.offsetLinks);
  for (let polygon = 0; polygon < 2; polygon++) {
    for (let vertex = 4; vertex < slots; vertex++) {
      expect(
        Array.from(links.slice((polygon * slots + vertex) * 2, (polygon * slots + vertex + 1) * 2))
      ).toEqual([0xffffffff, 0xffffffff]);
    }
  }
});

it('scratch fits the source range rather than maximum-vertex padding and retains no unused fields', () => {
  const {nodes} = createFixture([0, 0, 4, 0, 4, 2, 0, 2], [0, 4], [0, 1], 128, 128);
  const resourceNodes = nodes as {resources?: {buffer?: GraphDataView}[]}[];
  const workspace = resourceNodes
    .flatMap(node => node.resources ?? [])
    .map(resource => resource.buffer)
    .find(view => view?.buffer.id === 'vector-straight-skeleton-wavefront');
  expect(workspace).toBeDefined();
  expect(workspace!.length * workspace!.rowByteLength).toBe(4 + (4 + 256) * 76);
});

it.each([
  4, 128
])('fixed and packed scratch preserve a zero-event budget (maximumVertices=%s)', maximumVertices => {
  const {memory, nodes, output} = createFixture(
    [0, 0, 4, 0, 4, 2, 0, 2],
    [0, 4],
    [0, 1],
    maximumVertices,
    0
  );
  executeCommandNodesOnCPU(nodes, memory);
  expect(Array.from(memory.readUint32(output.status))).toEqual([0, 0, 4, 4]);
  expect(Array.from(memory.readUint32(output.offsetLinks).slice(0, 8))).toEqual([
    3, 1, 0, 2, 1, 3, 2, 0
  ]);
});

it('a 64-vertex collinear rectangle keeps all offset vertices and completes simultaneous ridges', () => {
  const coordinates: number[] = [];
  for (let vertex = 0; vertex < 16; vertex++) coordinates.push(vertex * 4, 0);
  for (let vertex = 0; vertex < 16; vertex++) coordinates.push(64, vertex * 2);
  for (let vertex = 0; vertex < 16; vertex++) coordinates.push(64 - vertex * 4, 32);
  for (let vertex = 0; vertex < 16; vertex++) coordinates.push(0, 32 - vertex * 2);
  const {memory, nodes, output} = createFixture(coordinates, [0, 64], [0, 1], 64, 128);
  executeCommandNodesOnCPU(nodes, memory);
  const status = memory.readUint32(output.status);
  expect(status[3]).toBe(0);
  expect(status[2]).toBe(64);
  expect(status[0]).toBeGreaterThanOrEqual(64);
  const links = memory.readUint32(output.offsetLinks);
  const offsets = memory.readFloat32(output.offsetPositions);
  for (let vertex = 0; vertex < 64; vertex++) {
    expect(links[vertex * 2]).toBe((vertex + 63) % 64);
    expect(links[vertex * 2 + 1]).toBe((vertex + 1) % 64);
    const coordinateX = offsets[vertex * 2];
    const coordinateY = offsets[vertex * 2 + 1];
    // Every shifted supporting edge remains exactly 0.25 from the original boundary.
    expect(Math.min(coordinateX, 64 - coordinateX, coordinateY, 32 - coordinateY)).toBeCloseTo(
      0.25,
      5
    );
  }
});

it('in-place convex compaction preserves later events and source attribution', () => {
  const coordinates = [0, 0, 8, 0, 10, 3, 6, 8, 1, 6];
  const {graph, memory, nodes, output, positions, ringOffsets} = createFixture(
    coordinates,
    [0, 5],
    [0, 1]
  );
  executeCommandNodesOnCPU(nodes, memory);
  const referenceStatus = memory.readUint32(output.status);
  expect(referenceStatus[3]).toBe(0);
  const referenceSegments = memory.readFloat32(output.segments);
  const referenceSources = memory.readUint32(output.sourceEdges);
  for (const maximumIterations of [1, 64]) {
    const convexOutput = {
      segments: createTransientView(graph, `convex-segments-${maximumIterations}`, 'float32x4', 10),
      sourceEdges: createTransientView(
        graph,
        `convex-sources-${maximumIterations}`,
        'uint32x2',
        10
      ),
      offsetPositions: createTransientView(
        graph,
        `convex-offsets-${maximumIterations}`,
        'float32x2',
        5
      ),
      status: createTransientView(graph, `convex-status-${maximumIterations}`, 'uint32x4', 1)
    };
    const convexNodes = new GPUConvexStraightSkeleton({
      positions,
      ringOffsets,
      maximumVertices: 5,
      maximumIterations,
      offsetDistance: 0.25,
      output: convexOutput
    }).getCommandNodes(graph);
    executeCommandNodesOnCPU(convexNodes, memory);
    const status = memory.readUint32(convexOutput.status);
    if (maximumIterations === 1) {
      expect(status[3]).toBe(4);
      continue;
    }
    expect(status[3]).toBe(0);
    expect(status[0]).toBe(referenceStatus[0]);
    const segments = memory.readFloat32(convexOutput.segments);
    const sources = memory.readUint32(convexOutput.sourceEdges);
    for (let arc = 0; arc < status[0]; arc++) {
      const matches = Array.from({length: referenceStatus[0]}, (_, index) => index).filter(
        index =>
          Math.hypot(
            ...Array.from(
              {length: 4},
              (_, coordinate) =>
                segments[arc * 4 + coordinate] - referenceSegments[index * 4 + coordinate]
            )
          ) < 1e-3
      );
      expect(matches).toHaveLength(1);
      expect(Array.from(sources.slice(arc * 2, arc * 2 + 2))).toEqual(
        Array.from(referenceSources.slice(matches[0] * 2, matches[0] * 2 + 2))
      );
    }
  }
});

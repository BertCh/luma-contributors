// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {
  GPURatioConcaveHull,
  GPUSubdividePolygons,
  GPULinePrecisionReduction,
  GPUPolygonPrecisionReduction,
  GPUConvexStraightSkeleton,
  GPULineLoopRemoval,
  GPUCanonicalPolygonEquality
} from '../../../src/gpu-spatial-analysis/geometry-refinement/index';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {createTopologyOutput} from './topology-output';
import fixtures from '../../oracles/fixtures/geometry-refinement.json';

const COMPONENTS = {
  uint32: 1,
  uint32x2: 2,
  uint32x3: 3,
  uint32x4: 4,
  float32: 1,
  float32x2: 2,
  float32x4: 4
} as const;
function createHarness(device: Device) {
  const graph = new GPUCommandGraph(device);
  const buffers = new Map<GraphDataView, Buffer>();
  let serial = 0;
  function view<Format extends keyof typeof COMPONENTS>(
    format: Format,
    length: number,
    values?: number[]
  ) {
    const buffer = values
      ? createInputBuffer(
          device,
          format.startsWith('uint') ? Uint32Array.from(values) : Float32Array.from(values)
        )
      : createOutputBuffer(device, length * COMPONENTS[format]);
    const result = importGraphBuffer(graph, `view-${serial++}`, buffer, format, length);
    buffers.set(result, buffer);
    return result;
  }
  return {
    graph,
    view,
    floats: (value: GraphDataView) =>
      readFloat32(
        buffers.get(value)!,
        value.length * COMPONENTS[value.format as keyof typeof COMPONENTS]
      ),
    integers: (value: GraphDataView) =>
      readUint32(
        buffers.get(value)!,
        value.length * COMPONENTS[value.format as keyof typeof COMPONENTS]
      ),
    destroy: () => buffers.forEach(buffer => buffer.destroy())
  };
}

it('GPU ratio hull endpoints and convex subdivision conserve independent fixture area', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const fixture of fixtures.ratioHulls) {
    const ratio = fixture.ratio;
    const harness = createHarness(device);
    const {graph, view} = harness;
    const coordinates = fixture.positions;
    const pointCount = coordinates.length / 2;
    const capacity = 2 * pointCount - 5;
    const output = {
      triangles: view('uint32x3', capacity),
      triangleValidity: view('uint32', capacity),
      boundaryEdges: view('uint32x2', capacity * 3),
      edgeValidity: view('uint32', capacity * 3),
      status: {converged: view('uint32', 1)}
    };
    graph.add(
      new GPURatioConcaveHull({
        positions: view('float32x2', pointCount, coordinates),
        allowHoles: 'allowHoles' in fixture && fixture.allowHoles,
        ratio,
        output
      })
    );
    const compiled = graph.compile();
    try {
      submitGraph(device, compiled);
      expect(await harness.integers(output.status.converged)).toEqual([1]);
      const triangles = await harness.integers(output.triangles);
      const validity = await harness.integers(output.triangleValidity);
      let area = 0;
      for (let row = 0; row < capacity; row++) {
        if (!validity[row]) {
          continue;
        }
        const [first, second, third] = triangles.slice(row * 3, row * 3 + 3);
        area +=
          Math.abs(
            (coordinates[second * 2] - coordinates[first * 2]) *
              (coordinates[third * 2 + 1] - coordinates[first * 2 + 1]) -
              (coordinates[second * 2 + 1] - coordinates[first * 2 + 1]) *
                (coordinates[third * 2] - coordinates[first * 2])
          ) / 2;
      }
      expect(area).toBeCloseTo(fixture.expectedArea, 4);
    } finally {
      compiled.destroy();
      harness.destroy();
    }
  }
  const harness = createHarness(device);
  const {graph, view} = harness;
  const fixture = fixtures.polygons[1].input;
  const output = {
    positions: view('float32x2', 128),
    pieces: view('uint32x4', 32),
    status: view('uint32x4', 1)
  };
  graph.add(
    new GPUSubdividePolygons({
      positions: view('float32x2', fixture.positions.length / 2, fixture.positions),
      ringOffsets: view('uint32', fixture.ringOffsets.length, fixture.ringOffsets),
      polygonOffsets: view('uint32', fixture.polygonOffsets.length, fixture.polygonOffsets),
      featureOffsets: view('uint32', fixture.featureOffsets.length, fixture.featureOffsets),
      maximumVertices: 4,
      output
    })
  );
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled);
    const status = await harness.integers(output.status);
    expect(status.slice(2)).toEqual([0, 0]);
    const pieces = await harness.integers(output.pieces);
    const positions = await harness.floats(output.positions);
    let area = 0;
    for (let piece = 0; piece < status[0]; piece++) {
      const count = pieces[piece * 4];
      expect(count).toBeLessThanOrEqual(4);
      let doubled = 0;
      for (let vertex = 0; vertex < count; vertex++) {
        const next = (vertex + 1) % count;
        const base = piece * 8;
        doubled +=
          positions[base + vertex * 2] * positions[base + next * 2 + 1] -
          positions[base + next * 2] * positions[base + vertex * 2 + 1];
      }
      area += Math.abs(doubled) / 2;
    }
    expect(area).toBe(99);
  } finally {
    compiled.destroy();
    harness.destroy();
  }
});

it('GPU precision policies preserve collapsed lines explicitly and reconstruct snapped polygon topology', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const mode of ['valid-output', 'keep-collapsed'] as const) {
    const harness = createHarness(device);
    const {graph, view} = harness;
    const output = {
      positions: view('float32x2', 2),
      lineOffsets: view('uint32', 2),
      status: view('uint32', 1)
    };
    graph.add(
      new GPULinePrecisionReduction({
        positions: view('float32x2', 2, [0.1, 0.1, 0.2, 0.2]),
        lineOffsets: view('uint32', 2, [0, 2]),
        gridSize: 1,
        mode,
        output
      })
    );
    const compiled = graph.compile();
    try {
      submitGraph(device, compiled);
      expect(await harness.integers(output.lineOffsets)).toEqual([
        0,
        mode === 'keep-collapsed' ? 2 : 0
      ]);
      expect(await harness.integers(output.status)).toEqual([mode === 'keep-collapsed' ? 2 : 1]);
    } finally {
      compiled.destroy();
      harness.destroy();
    }
  }
  const harness = createHarness(device);
  const {graph, view} = harness;
  const output = createTopologyOutput(view);
  const lattice = view('float32', 4);
  graph.add(
    new GPUPolygonPrecisionReduction({
      gridSize: 1,
      lattice,
      intersectionCapacity: 8,
      polygons: {
        kind: 'polygons',
        positions: view('float32x2', 4, [0.2, 0.2, 3.6, 0.2, 3.6, 3.6, 0.2, 3.6]),
        ringOffsets: view('uint32', 2, [0, 4]),
        polygonOffsets: view('uint32', 2, [0, 1]),
        featureOffsets: view('uint32', 2, [0, 1])
      },
      output
    })
  );
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled);
    expect(await harness.integers(output.status.overflow)).toEqual([0]);
    expect(await harness.integers(output.status.count)).toEqual([1]);
    expect((await harness.floats(lattice))[2]).toBe(1);
    const offsets = await harness.integers(output.polygons.ringOffsets);
    const coordinates = await harness.floats(output.polygons.positions);
    let area = 0;
    for (let vertex = offsets[0]; vertex < offsets[1]; vertex++) {
      const next = vertex + 1 === offsets[1] ? offsets[0] : vertex + 1;
      area +=
        coordinates[vertex * 2] * coordinates[next * 2 + 1] -
        coordinates[next * 2] * coordinates[vertex * 2 + 1];
    }
    expect(Math.abs(area) / 2).toBe(16);
  } finally {
    compiled.destroy();
    harness.destroy();
  }
});

it('GPU convex wavefront handles simultaneous rectangle events and exact inward offsets', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createHarness(device);
  const {graph, view} = harness;
  const output = {
    segments: view('float32x4', 8),
    sourceEdges: view('uint32x2', 8),
    offsetPositions: view('float32x2', 4),
    status: view('uint32x4', 1)
  };
  graph.add(
    new GPUConvexStraightSkeleton({
      positions: view('float32x2', 4, [0, 0, 4, 0, 4, 2, 0, 2]),
      ringOffsets: view('uint32', 2, [0, 4]),
      maximumVertices: 4,
      offsetDistance: 0.5,
      output
    })
  );
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled);
    expect(await harness.integers(output.status)).toEqual([5, 5, 4, 0]);
    expect(await harness.floats(output.offsetPositions)).toEqual([
      0.5, 0.5, 3.5, 0.5, 3.5, 1.5, 0.5, 1.5
    ]);
    const segments = await harness.floats(output.segments);
    expect(segments.slice(16, 20)).toEqual([1, 1, 3, 1]);
  } finally {
    compiled.destroy();
    harness.destroy();
  }
});

it('GPU convex wavefront accepts rotated float32 quadrilaterals in either winding', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const clockwisePositions = [
    -49.123355865478516, -28.567359924316406, -50.025718688964844, 28.959735870361328,
    47.715301513671875, 29.230628967285156, 48.62384033203125, -27.533666610717773
  ];
  const quadrilaterals = [
    clockwisePositions,
    Array.from({length: 4}, (_, vertexIndex) => {
      const reversedVertexIndex = 3 - vertexIndex;
      return clockwisePositions.slice(reversedVertexIndex * 2, reversedVertexIndex * 2 + 2);
    }).flat()
  ];

  for (const positions of quadrilaterals) {
    const harness = createHarness(device);
    const {graph, view} = harness;
    const output = {
      segments: view('float32x4', 8),
      sourceEdges: view('uint32x2', 8),
      offsetPositions: view('float32x2', 4),
      status: view('uint32x4', 1)
    };
    graph.add(
      new GPUConvexStraightSkeleton({
        positions: view('float32x2', 4, positions),
        ringOffsets: view('uint32', 2, [0, 4]),
        maximumVertices: 4,
        offsetDistance: 1,
        output
      })
    );
    const compiled = graph.compile();
    try {
      submitGraph(device, compiled);
      const [writtenArcs, requiredArcs, offsetVertexCount, flags] = await harness.integers(
        output.status
      );
      expect(flags).toBe(0);
      expect(writtenArcs).toBe(requiredArcs);
      expect(writtenArcs).toBeGreaterThan(0);
      expect(offsetVertexCount).toBe(4);

      const segments = (await harness.floats(output.segments)).slice(0, writtenArcs * 4);
      const offsetPositions = await harness.floats(output.offsetPositions);
      expect(segments.every(Number.isFinite)).toBe(true);
      expect(offsetPositions.every(Number.isFinite)).toBe(true);
    } finally {
      compiled.destroy();
      harness.destroy();
    }
  }
});

it('GPU loop erasure nodes crossings and preserves the remaining source path', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createHarness(device);
  const {graph, view} = harness;
  const output = {
    positions: view('float32x2', 8),
    lineOffsets: view('uint32', 2),
    sourceEdges: view('uint32', 8),
    status: view('uint32x4', 1)
  };
  graph.add(
    new GPULineLoopRemoval({
      lines: {
        kind: 'lines',
        positions: view('float32x2', 4, [0, 0, 4, 4, 0, 4, 4, 0]),
        lineOffsets: view('uint32', 2, [0, 4])
      },
      intersectionCapacity: 8,
      output
    })
  );
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled);
    expect(await harness.integers(output.status)).toEqual([3, 3, 0, 0]);
    expect((await harness.floats(output.positions)).slice(0, 6)).toEqual([0, 0, 2, 2, 4, 0]);
    expect((await harness.integers(output.sourceEdges)).slice(0, 2)).toEqual([0, 2]);
  } finally {
    compiled.destroy();
    harness.destroy();
  }
});

it('GPU bounded geometry reports output capacity and iteration exhaustion separately', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const maximumIterations of [0, 4]) {
    const harness = createHarness(device);
    const {graph, view} = harness;
    const output = {
      segments: view('float32x4', 2),
      sourceEdges: view('uint32x2', 2),
      offsetPositions: view('float32x2', 4),
      status: view('uint32x4', 1)
    };
    graph.add(
      new GPUConvexStraightSkeleton({
        positions: view('float32x2', 4, [0, 0, 4, 0, 4, 2, 0, 2]),
        ringOffsets: view('uint32', 2, [0, 4]),
        maximumVertices: 4,
        maximumIterations,
        segmentCapacity: 2,
        offsetDistance: 0.5,
        output
      })
    );
    const compiled = graph.compile();
    try {
      submitGraph(device, compiled);
      expect(await harness.integers(output.status)).toEqual(
        maximumIterations === 0 ? [0, 0, 4, 4] : [2, 5, 4, 2]
      );
    } finally {
      compiled.destroy();
      harness.destroy();
    }
  }
  const harness = createHarness(device);
  const {graph, view} = harness;
  const output = {
    positions: view('float32x2', 3),
    pieces: view('uint32x4', 1),
    status: view('uint32x4', 1)
  };
  graph.add(
    new GPUSubdividePolygons({
      positions: view('float32x2', 4, [0, 0, 4, 0, 4, 4, 0, 4]),
      ringOffsets: view('uint32', 2, [0, 4]),
      polygonOffsets: view('uint32', 2, [0, 1]),
      featureOffsets: view('uint32', 2, [0, 1]),
      maximumVertices: 3,
      output
    })
  );
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled);
    expect(await harness.integers(output.status)).toEqual([1, 2, 1, 0]);
  } finally {
    compiled.destroy();
    harness.destroy();
  }
});

it('GPU canonical equality observes structural and coordinate differences', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createHarness(device);
  const {graph, view} = harness;
  const left = {
    positions: view('float32x2', 4, [0, 0, 0, 4, 4, 4, 4, 0]),
    ringOffsets: view('uint32', 2, [0, 4]),
    polygonOffsets: view('uint32', 2, [0, 1]),
    featureOffsets: view('uint32', 2, [0, 1])
  };
  const right = {...left, positions: view('float32x2', 4, [0, 0, 0, 4, 4, 4, 3, 0])};
  const equal = view('uint32', 1);
  const unequal = view('uint32', 1);
  graph.add(new GPUCanonicalPolygonEquality({id: 'same', left, right: left, output: equal}));
  graph.add(new GPUCanonicalPolygonEquality({id: 'different', left, right, output: unequal}));
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled);
    expect(await harness.integers(equal)).toEqual([1]);
    expect(await harness.integers(unequal)).toEqual([0]);
  } finally {
    compiled.destroy();
    harness.destroy();
  }
});

it('loop erasure propagates exhausted intersection candidates even with sufficient output capacity', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const intersectionCapacity of [1, 32]) {
    const harness = createHarness(device);
    const {graph, view} = harness;
    const output = {
      positions: view('float32x2', 64),
      lineOffsets: view('uint32', 2),
      sourceEdges: view('uint32', 64),
      status: view('uint32x4', 1)
    };
    graph.add(
      new GPULineLoopRemoval({
        lines: {
          kind: 'lines',
          positions: view('float32x2', 6, [0, 0, 4, 4, 0, 4, 4, 0, 0, 2, 4, 2]),
          lineOffsets: view('uint32', 2, [0, 6])
        },
        intersectionCapacity,
        output
      })
    );
    const compiled = graph.compile();
    try {
      submitGraph(device, compiled);
      const status = await harness.integers(output.status);
      expect(status[2]).toBe(0);
      expect(status[3]).toBe(intersectionCapacity === 1 ? 1 : 0);
    } finally {
      compiled.destroy();
      harness.destroy();
    }
  }
});

// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {
  GPUSubdividePolygons,
  GPURatioConcaveHull,
  GPUMaximumInscribedCircle,
  GPUNormalizeGeometry,
  GPUCanonicalGeometryEquality,
  GPUPolygonPrecisionReduction,
  GPULinePrecisionReduction,
  GPUVectorStraightSkeleton,
  GPUOffsetCurveComponents
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
import fixtures from '../../oracles/fixtures/geometry-advanced.json';
import postgisFixtures from '../../oracles/fixtures/geometry-subdivision-postgis.json';
import {getDelaunayMaximumTriangleCount} from '../../../src/gpu-spatial-analysis/delaunay-tessellation/gpu-delaunay-tessellation';
import {
  GPUOffsetCurve,
  getGPUOffsetCurveRowsPerVertex
} from '../../../src/gpu-spatial-analysis/outline-geometry/gpu-offset-curve';
import skeletonFixtures from '../../oracles/fixtures/geometry-straight-skeleton.json';
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
    const buffer = values?.length
      ? createInputBuffer(
          device,
          format.startsWith('uint') ? Uint32Array.from(values) : Float32Array.from(values)
        )
      : createOutputBuffer(device, Math.max(1, length) * COMPONENTS[format]);
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

it('normalizes nested collections, points, lines and empty structures against direct Shapely, idempotently', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const fixture of fixtures.collections) {
    const harness = createHarness(device);
    const {graph, view} = harness;
    const upload = (input: typeof fixture.input) => ({
      positions: view('float32x2', input.positions.length / 2, input.positions),
      nodes: view('uint32x4', input.nodes.length / 4, input.nodes),
      children: view('uint32', input.children.length, input.children)
    });
    const input = upload(fixture.input);
    const reference = upload(fixture.normalized);
    const output = {
      positions: view('float32x2', input.positions.length),
      children: view('uint32', input.children.length),
      status: view('uint32', input.nodes.length)
    };
    const second = {
      positions: view('float32x2', input.positions.length),
      children: view('uint32', input.children.length),
      status: view('uint32', input.nodes.length)
    };
    graph.add(new GPUNormalizeGeometry({...input, output}));
    graph.add(
      new GPUNormalizeGeometry({
        ...input,
        positions: output.positions,
        children: output.children,
        id: 'again',
        output: second
      })
    );
    const equality = view('uint32', 1);
    graph.add(
      new GPUCanonicalGeometryEquality({
        left: {...input, positions: output.positions, children: output.children},
        right: reference,
        pairs: view('uint32x2', 1, [fixture.input.root, fixture.normalized.root]),
        output: equality
      })
    );
    const compiled = graph.compile();
    try {
      submitGraph(device, compiled);
      expect(Array.from(await harness.integers(output.status)), fixture.wkt).toEqual(
        new Array(input.nodes.length).fill(0)
      );
      const positions = Array.from(await harness.floats(output.positions));
      const children = Array.from(await harness.integers(output.children));
      const serialize = (node: number): unknown => {
        const [kind, first, count] = fixture.input.nodes.slice(node * 4, node * 4 + 3);
        return [
          kind,
          [0, 2, 3].includes(kind)
            ? positions.slice(first * 2, (first + count) * 2)
            : children.slice(first, first + count).map(serialize)
        ];
      };
      expect(serialize(fixture.input.root), fixture.wkt).toEqual(fixture.canonical);
      expect(Array.from(await harness.integers(equality)), fixture.wkt).toEqual([1]);
      expect(Array.from(await harness.floats(second.positions))).toEqual(positions);
      expect(Array.from(await harness.integers(second.children))).toEqual(children);
    } finally {
      compiled.destroy();
      harness.destroy();
    }
  }
});

it('reconstructs arbitrary precision grids and collapse policies against direct Shapely area and emptiness', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const fixture of fixtures.precision) {
    const harness = createHarness(device);
    const {graph, view} = harness;
    const output = createTopologyOutput(view);
    const lattice = view('float32', 4);
    graph.add(
      new GPUPolygonPrecisionReduction({
        polygons: {
          kind: 'polygons',
          positions: view('float32x2', fixture.positions.length / 2, fixture.positions),
          ringOffsets: view('uint32', fixture.ringOffsets.length, fixture.ringOffsets),
          polygonOffsets: view('uint32', 2, [0, fixture.ringOffsets.length - 1]),
          featureOffsets: view('uint32', 2, [0, 1])
        },
        gridSize: fixture.gridSize,
        mode: fixture.mode as 'valid-output' | 'keep-collapsed',
        intersectionCapacity: 32,
        lattice,
        output
      })
    );
    const repeated = createTopologyOutput(view);
    graph.add(
      new GPUPolygonPrecisionReduction({
        id: 'precision-again',
        polygons: output.polygons,
        gridSize: fixture.gridSize,
        mode: fixture.mode as 'valid-output' | 'keep-collapsed',
        intersectionCapacity: 32,
        lattice: view('float32', 4),
        output: repeated
      })
    );
    const compiled = graph.compile();
    try {
      submitGraph(device, compiled);
      expect(
        Array.from(await harness.integers(output.status.overflow)),
        JSON.stringify(fixture)
      ).toEqual([0]);
      const count = (await harness.integers(output.status.count))[0];
      expect(count === 0).toBe(fixture.empty);
      const positions = await harness.floats(output.polygons.positions);
      const ringOffsets = await harness.integers(output.polygons.ringOffsets);
      const polygonOffsets = await harness.integers(output.polygons.polygonOffsets);
      let area = 0;
      for (let polygon = 0; polygon < count; polygon++) {
        for (let ring = polygonOffsets[polygon]; ring < polygonOffsets[polygon + 1]; ring++) {
          let twiceArea = 0;
          for (let vertex = ringOffsets[ring]; vertex < ringOffsets[ring + 1]; vertex++) {
            const next = vertex + 1 === ringOffsets[ring + 1] ? ringOffsets[ring] : vertex + 1;
            twiceArea +=
              positions[vertex * 2] * positions[next * 2 + 1] -
              positions[next * 2] * positions[vertex * 2 + 1];
          }
          area += ((ring === polygonOffsets[polygon] ? 1 : -1) * Math.abs(twiceArea)) / 2;
        }
      }
      expect(area, JSON.stringify(fixture)).toBeCloseTo(fixture.area, 4);
      expect(
        await harness.integers(repeated.status.overflow),
        JSON.stringify({
          fixture,
          count,
          firstRings: ringOffsets,
          firstPolygons: polygonOffsets,
          repeatedCount: await harness.integers(repeated.status.count),
          noding: await harness.integers(repeated.nodedSegments.status.overflow),
          candidate: await harness.integers(repeated.nodedSegments.status.candidateOverflow),
          required: await harness.integers(repeated.nodedSegments.status.requiredCount)
        })
      ).toEqual([0]);
      expect(await harness.integers(repeated.status.count)).toEqual([count]);
      expect(await harness.integers(repeated.polygons.ringOffsets)).toEqual(ringOffsets);
      expect(await harness.integers(repeated.polygons.polygonOffsets)).toEqual(polygonOffsets);
      expect(
        (await harness.floats(repeated.polygons.positions)).slice(0, ringOffsets.at(-1)! * 2)
      ).toEqual(positions.slice(0, ringOffsets.at(-1)! * 2));
      expect((await harness.floats(lattice))[2]).toBeCloseTo(fixture.gridSize, 6);
    } finally {
      compiled.destroy();
      harness.destroy();
    }
  }
});

it.each(skeletonFixtures.cases)('event wavefront agrees with CGAL 6.0.1: $name', async fixture => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createHarness(device);
  const {graph, view} = harness;
  const maximumVertices = 32;
  const maximumEvents = 64;
  const slots = maximumVertices + 2 * maximumEvents;
  const output = {
    segments: view('float32x4', 128),
    sourceEdges: view('uint32x2', 128),
    offsetPositions: view('float32x2', slots),
    offsetLinks: view('uint32x2', slots),
    status: view('uint32x4', 1)
  };
  graph.add(
    new GPUVectorStraightSkeleton({
      positions: view('float32x2', fixture.positions.length / 2, fixture.positions),
      ringOffsets: view('uint32', fixture.ringOffsets.length, fixture.ringOffsets),
      polygonOffsets: view('uint32', 2, [0, fixture.ringOffsets.length - 1]),
      featureOffsets: view('uint32', 2, [0, 1]),
      maximumVertices,
      maximumEvents,
      segmentCapacity: 128,
      offsetDistance: fixture.offsetDistance,
      output
    })
  );
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled);
    const status = await harness.integers(output.status);
    const segments = await harness.floats(output.segments);
    const actual = Array.from({length: status[0]}, (_, index) =>
      Array.from(segments.slice(index * 4, index * 4 + 4))
    );
    expect(status[3], JSON.stringify({status, actual})).toBe(0);
    const expected = fixture.arcs
      .map(arc => arc.slice(0, 4))
      .filter(arc => Math.hypot(arc[0] - arc[2], arc[1] - arc[3]) > 1e-5);
    const matching = (left: number[], right: number[]) =>
      Math.hypot(left[0] - right[0], left[1] - right[1], left[2] - right[2], left[3] - right[3]) <
        1e-3 ||
      Math.hypot(left[0] - right[2], left[1] - right[3], left[2] - right[0], left[3] - right[1]) <
        1e-3;
    expect(actual.length, JSON.stringify({actual, expected, status})).toBe(expected.length);
    for (const arc of expected)
      expect(
        actual.some(candidate => matching(candidate, arc)),
        JSON.stringify({arc, actual})
      ).toBe(true);
    const coordinates = await harness.floats(output.offsetPositions);
    const links = await harness.integers(output.offsetLinks);
    const active = Array.from({length: slots}, (_, index) => index).filter(
      index => links[index * 2] !== 0xffffffff
    );
    expect(active.length).toBe(fixture.offsets.reduce((sum, ring) => sum + ring.length, 0));
    const unvisited = new Set(active);
    const actualRings: number[][][] = [];
    while (unvisited.size) {
      const first = unvisited.values().next().value!;
      const ring: number[][] = [];
      let current = first;
      do {
        expect(unvisited.has(current)).toBe(true);
        unvisited.delete(current);
        ring.push(coordinates.slice(current * 2, current * 2 + 2));
        const previous = links[current * 2];
        const next = links[current * 2 + 1];
        expect(links[previous * 2 + 1]).toBe(current);
        expect(links[next * 2]).toBe(current);
        current = next;
      } while (current !== first);
      actualRings.push(ring);
    }
    expect(actualRings.length).toBe(fixture.offsets.length);
    for (const expectedRing of fixture.offsets) {
      const matchingRing = actualRings.findIndex(
        ring =>
          ring.length === expectedRing.length &&
          ring.some((_, start) =>
            [1, -1].some(direction =>
              expectedRing.every((point, vertex) => {
                const actualPoint = ring[(start + direction * vertex + ring.length) % ring.length];
                return Math.hypot(actualPoint[0] - point[0], actualPoint[1] - point[1]) < 1e-3;
              })
            )
          )
      );
      expect(matchingRing).toBeGreaterThanOrEqual(0);
      actualRings.splice(matchingRing, 1);
    }
  } finally {
    compiled.destroy();
    harness.destroy();
  }
});

it.each(
  fixtures.offsets
)('offset components match Shapely side/distance $distance, $joinStyle, $positions', async fixture => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createHarness(device);
  const {graph, view} = harness;
  const joinStyle = fixture.joinStyle as 'round' | 'mitre' | 'bevel';
  const rowsPerVertex = getGPUOffsetCurveRowsPerVertex(joinStyle, 8, 'buffer');
  const original = {
    positions: view('float32x2', fixture.positions.length / 2, fixture.positions),
    lineOffsets: view('uint32', 2, [0, fixture.positions.length / 2])
  };
  const raw = {
    positions: view('float32x2', original.positions.length * rowsPerVertex),
    lineOffsets: view('uint32', 2, [0, original.positions.length * rowsPerVertex])
  };
  graph.add(
    new GPUOffsetCurve({
      ...original,
      joinStyle,
      quadSegments: 8,
      insideTurns: 'buffer',
      parameters: view('float32', 4, [fixture.distance, 5, 0, 0]),
      output: {positions: raw.positions}
    })
  );
  const output = {
    positions: view('float32x2', 1024),
    sourceEdges: view('uint32', 1024),
    components: view('uint32x4', 128),
    status: view('uint32x4', 2)
  };
  graph.add(
    new GPUOffsetCurveComponents({
      original,
      raw,
      rowsPerVertex,
      distance: fixture.distance,
      tolerance: Math.abs(fixture.distance) * 0.005,
      segmentCapacity: 512,
      intersectionCapacity: 256,
      output
    })
  );
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled);
    const status = await harness.integers(output.status);
    expect(status.slice(4)).toEqual([0, 0, 0, 0]);
    const positions = await harness.floats(output.positions);
    const components = await harness.integers(output.components);
    let length = 0;
    const actual: number[][][] = [];
    for (let component = 0; component < status[0]; component++) {
      const first = components[component * 4];
      const count = components[component * 4 + 1];
      const points = Array.from({length: count}, (_, vertex) =>
        positions.slice((first + vertex) * 2, (first + vertex) * 2 + 2)
      );
      actual.push(points);
      for (let vertex = 1; vertex < count; vertex++)
        length += Math.hypot(
          points[vertex][0] - points[vertex - 1][0],
          points[vertex][1] - points[vertex - 1][1]
        );
    }
    expect(length, JSON.stringify({fixture, actual, status})).toBeCloseTo(fixture.length, 3);
    expect(status[0], JSON.stringify({fixture, actual})).toBe(fixture.expected.length);
    const separation = (point: number[], lines: number[][][]) =>
      Math.min(
        ...lines.flatMap(line =>
          line.slice(1).map((finish, index) => {
            const start = line[index];
            const delta = [finish[0] - start[0], finish[1] - start[1]];
            const denominator = delta[0] ** 2 + delta[1] ** 2;
            const fraction = denominator
              ? Math.max(
                  0,
                  Math.min(
                    1,
                    ((point[0] - start[0]) * delta[0] + (point[1] - start[1]) * delta[1]) /
                      denominator
                  )
                )
              : 0;
            return Math.hypot(
              point[0] - start[0] - fraction * delta[0],
              point[1] - start[1] - fraction * delta[1]
            );
          })
        )
      );
    for (const point of actual.flat())
      expect(separation(point, fixture.expected)).toBeLessThan(0.002);
    for (const point of fixture.expected.flat())
      expect(separation(point, actual)).toBeLessThan(0.002);
    const provenance = await harness.integers(output.sourceEdges);
    for (const source of provenance.slice(0, status[1])) {
      expect(source).toBeLessThan(original.positions.length);
    }
    for (const line of actual) {
      if (Math.hypot(line[0][0] - line.at(-1)![0], line[0][1] - line.at(-1)![1]) < 1e-4) continue;
      expect(
        fixture.expected.some(
          expected =>
            Math.hypot(line[0][0] - expected[0][0], line[0][1] - expected[0][1]) < 0.002 &&
            Math.hypot(
              line.at(-1)![0] - expected.at(-1)![0],
              line.at(-1)![1] - expected.at(-1)![1]
            ) < 0.002
        )
      ).toBe(true);
    }
  } finally {
    compiled.destroy();
    harness.destroy();
  }
});

it('rectilinear queue clips holes and concavities into bounded pieces with exact reference union membership', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const fixture of fixtures.subdivision) {
    const harness = createHarness(device);
    const {graph, view} = harness;
    const output = {
      positions: view('float32x2', 256 * 4),
      pieces: view('uint32x4', 256),
      status: view('uint32x4', 1)
    };
    graph.add(
      new GPUSubdividePolygons({
        positions: view('float32x2', fixture.positions.length / 2, fixture.positions),
        ringOffsets: view('uint32', fixture.ringOffsets.length, fixture.ringOffsets),
        polygonOffsets: view('uint32', 2, [0, fixture.ringOffsets.length - 1]),
        featureOffsets: view('uint32', 2, [0, 1]),
        method: 'rectilinear',
        maximumVertices: 4,
        output
      })
    );
    const compiled = graph.compile();
    try {
      submitGraph(device, compiled);
      const status = await harness.integers(output.status);
      expect(status.slice(2)).toEqual([0, 0]);
      const positions = await harness.floats(output.positions);
      const pieces = await harness.integers(output.pieces);
      let area = 0;
      const rings: number[][][] = [];
      for (let piece = 0; piece < status[0]; piece++) {
        const count = pieces[piece * 4];
        expect(count).toBeLessThanOrEqual(4);
        expect(pieces.slice(piece * 4 + 1, piece * 4 + 3)).toEqual([0, 0]);
        const ring = Array.from({length: count}, (_, vertex) =>
          positions.slice((piece * 4 + vertex) * 2, (piece * 4 + vertex) * 2 + 2)
        );
        rings.push(ring);
        area +=
          Math.abs(
            ring.reduce((sum, point, index) => {
              const next = ring[(index + 1) % count];
              return sum + point[0] * next[1] - point[1] * next[0];
            }, 0)
          ) / 2;
      }
      expect(area).toBeCloseTo(fixture.area, 4);
      const contains = (ring: number[][], query: number[]) => {
        let inside = false;
        for (let first = 0, last = ring.length - 1; first < ring.length; last = first++) {
          const start = ring[first],
            finish = ring[last];
          if (
            start[1] > query[1] !== finish[1] > query[1] &&
            query[0] <
              ((finish[0] - start[0]) * (query[1] - start[1])) / (finish[1] - start[1]) + start[0]
          )
            inside = !inside;
        }
        return inside;
      };
      expect(fixture.queries.map(query => rings.some(ring => contains(ring, query)))).toEqual(
        fixture.contains
      );
    } finally {
      compiled.destroy();
      harness.destroy();
    }
  }
});

it('ratio endpoint and hole policies match direct Shapely area for duplicates, collinear inputs and separated clusters', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const fixture of fixtures.hulls) {
    const harness = createHarness(device);
    const {graph, view} = harness;
    const capacity = getDelaunayMaximumTriangleCount(fixture.positions.length / 2);
    const output = {
      triangles: view('uint32x3', capacity),
      triangleValidity: view('uint32', capacity),
      boundaryEdges: view('uint32x2', capacity * 3),
      edgeValidity: view('uint32', capacity * 3),
      status: {converged: view('uint32', 1)}
    };
    graph.add(
      new GPURatioConcaveHull({
        positions: view('float32x2', fixture.positions.length / 2, fixture.positions),
        ratio: fixture.ratio,
        allowHoles: fixture.allowHoles,
        output
      })
    );
    const compiled = graph.compile();
    try {
      submitGraph(device, compiled);
      expect(await harness.integers(output.status.converged)).toEqual([1]);
      const triangles = await harness.integers(output.triangles),
        validity = await harness.integers(output.triangleValidity);
      let area = 0;
      for (let row = 0; row < capacity; row++) {
        if (!validity[row]) continue;
        const [first, second, third] = triangles.slice(row * 3, row * 3 + 3);
        const positions = fixture.positions;
        area +=
          Math.abs(
            (positions[second * 2] - positions[first * 2]) *
              (positions[third * 2 + 1] - positions[first * 2 + 1]) -
              (positions[second * 2 + 1] - positions[first * 2 + 1]) *
                (positions[third * 2] - positions[first * 2])
          ) / 2;
      }
      expect(area, JSON.stringify(fixture)).toBeCloseTo(fixture.area, 4);
    } finally {
      compiled.destroy();
      harness.destroy();
    }
  }
});

it('circle bounds enclose direct Shapely optimum and distinguish work exhaustion from tolerance success', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const fixture of fixtures.circles)
    for (const maximumIterations of [0, 8192]) {
      const harness = createHarness(device);
      const {graph, view} = harness;
      const output = {
        circles: view('float32x4', 1),
        witnesses: view('float32x2', 1),
        status: view('uint32', 1)
      };
      graph.add(
        new GPUMaximumInscribedCircle({
          positions: view('float32x2', fixture.positions.length / 2, fixture.positions),
          ringOffsets: view('uint32', fixture.ringOffsets.length, fixture.ringOffsets),
          polygonOffsets: view('uint32', 2, [0, fixture.ringOffsets.length - 1]),
          featureOffsets: view('uint32', 2, [0, 1]),
          tolerance: fixture.tolerance,
          maximumIterations,
          cellCapacity: 4096,
          output
        })
      );
      const compiled = graph.compile();
      try {
        submitGraph(device, compiled);
        const circle = await harness.floats(output.circles),
          witness = await harness.floats(output.witnesses),
          status = await harness.integers(output.status);
        expect(circle[2]).toBeLessThanOrEqual(fixture.radius + 2e-5);
        expect(circle[2] + circle[3] + 2e-5).toBeGreaterThanOrEqual(fixture.radius);
        expect(Math.hypot(witness[0] - circle[0], witness[1] - circle[1])).toBeCloseTo(
          circle[2],
          4
        );
        if (maximumIterations === 0) expect(status).toEqual([1]);
        else {
          expect(status).toEqual([0]);
          expect(circle[3]).toBeLessThanOrEqual(fixture.tolerance);
          expect(Math.abs(circle[2] - fixture.radius)).toBeLessThanOrEqual(
            fixture.tolerance + 2e-5
          );
        }
      } finally {
        compiled.destroy();
        harness.destroy();
      }
    }
});

it('bounded geometry contracts reject invalid trees and publish only complete offset components', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createHarness(device);
  const {graph, view} = harness;
  const normalization = {
    positions: view('float32x2', 1),
    children: view('uint32', 2),
    status: view('uint32', 3)
  };
  graph.add(
    new GPUNormalizeGeometry({
      positions: view('float32x2', 1, [Number.NaN, 0]),
      nodes: view('uint32x4', 3, [0, 0, 1, 7, 7, 0, 1, 8, 7, 1, 1, 9]),
      children: view('uint32', 2, [0, 2]),
      output: normalization
    })
  );
  const offset = {
    positions: view('float32x2', 2),
    sourceEdges: view('uint32', 2),
    components: view('uint32x4', 1),
    status: view('uint32x4', 2)
  };
  graph.add(
    new GPUOffsetCurveComponents({
      original: {
        positions: view('float32x2', 3, [0, 0, 4, 0, 4, 4]),
        lineOffsets: view('uint32', 2, [0, 3])
      },
      raw: {
        positions: view('float32x2', 3, [0, 0.5, 3.5, 0.5, 3.5, 4]),
        lineOffsets: view('uint32', 2, [0, 3])
      },
      distance: 0.5,
      tolerance: 0.001,
      segmentCapacity: 16,
      intersectionCapacity: 16,
      output: offset
    })
  );
  const fixture = fixtures.subdivision[2];
  const subdivision = {
    positions: view('float32x2', 4),
    pieces: view('uint32x4', 1),
    status: view('uint32x4', 1)
  };
  graph.add(
    new GPUSubdividePolygons({
      positions: view('float32x2', fixture.positions.length / 2, fixture.positions),
      ringOffsets: view('uint32', fixture.ringOffsets.length, fixture.ringOffsets),
      polygonOffsets: view('uint32', 2, [0, 1]),
      featureOffsets: view('uint32', 2, [0, 1]),
      method: 'rectilinear',
      maximumVertices: 4,
      maximumSplits: 1,
      output: subdivision
    })
  );
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled);
    expect(await harness.integers(normalization.status)).toEqual([1, 2, 2]);
    const offsetStatus = await harness.integers(offset.status);
    expect(offsetStatus.slice(0, 2)).toEqual([0, 0]);
    expect(offsetStatus[2]).toBe(1);
    expect(offsetStatus[3]).toBeGreaterThan(2);
    expect(offsetStatus[4]).toBe(1);
    expect(await harness.integers(offset.components)).toEqual([0, 0, 0, 0]);
    expect((await harness.integers(subdivision.status))[3]).toBe(0x80000000);
  } finally {
    compiled.destroy();
    harness.destroy();
  }
});

it('rectilinear pieces preserve multi-feature PostGIS ST_Subdivide unions at three vertex limits', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const postgisLimit of [5, 9, 17]) {
    const references = postgisFixtures.cases.filter(
      fixture => fixture.maximumVertices === postgisLimit
    );
    expect(references).toHaveLength(4);
    const maximumVertices = postgisLimit - 1;
    const coordinates: number[] = [];
    const ringOffsets = [0];
    const polygonOffsets = [0];
    for (const fixture of references) {
      const firstVertex = coordinates.length / 2;
      coordinates.push(...fixture.positions);
      ringOffsets.push(...fixture.ringOffsets.slice(1).map(offset => firstVertex + offset));
      polygonOffsets.push(ringOffsets.length - 1);
      expect(fixture.valid).toBe(true);
      expect(fixture.reconstructsInput).toBe(true);
      expect(fixture.largestPieceVertices).toBeLessThanOrEqual(postgisLimit);
    }
    const harness = createHarness(device);
    const {graph, view} = harness;
    const output = {
      positions: view('float32x2', 256 * maximumVertices),
      pieces: view('uint32x4', 256),
      status: view('uint32x4', 1)
    };
    graph.add(
      new GPUSubdividePolygons({
        positions: view('float32x2', coordinates.length / 2, coordinates),
        ringOffsets: view('uint32', ringOffsets.length, ringOffsets),
        polygonOffsets: view('uint32', polygonOffsets.length, polygonOffsets),
        featureOffsets: view('uint32', 5, [0, 1, 2, 3, 4]),
        method: 'rectilinear',
        maximumVertices,
        output
      })
    );
    const compiled = graph.compile();
    try {
      submitGraph(device, compiled);
      const status = await harness.integers(output.status);
      expect(status.slice(2)).toEqual([0, 0]);
      const positions = await harness.floats(output.positions);
      const pieces = await harness.integers(output.pieces);
      const byFeature: number[][][][] = references.map(() => []);
      for (let piece = 0; piece < status[0]; piece++) {
        const [count, feature, polygon] = pieces.slice(piece * 4, piece * 4 + 3);
        expect(count).toBeLessThanOrEqual(maximumVertices);
        expect(feature).toBeLessThan(references.length);
        expect(polygon).toBe(feature);
        byFeature[feature].push(
          Array.from({length: count}, (_, vertex) =>
            positions.slice(
              (piece * maximumVertices + vertex) * 2,
              (piece * maximumVertices + vertex) * 2 + 2
            )
          )
        );
      }
      const contains = (ring: number[][], query: number[]) => {
        let inside = false;
        for (let first = 0, last = ring.length - 1; first < ring.length; last = first++) {
          const start = ring[first],
            finish = ring[last];
          if (
            start[1] > query[1] !== finish[1] > query[1] &&
            query[0] <
              ((finish[0] - start[0]) * (query[1] - start[1])) / (finish[1] - start[1]) + start[0]
          )
            inside = !inside;
        }
        return inside;
      };
      for (const [feature, reference] of references.entries()) {
        expect(reference.sourceId).toBe(feature);
        const rings = byFeature[feature];
        const area = rings.reduce(
          (total, ring) =>
            total +
            Math.abs(
              ring.reduce((sum, point, index) => {
                const next = ring[(index + 1) % ring.length];
                return sum + point[0] * next[1] - point[1] * next[0];
              }, 0)
            ) /
              2,
          0
        );
        expect(area).toBeCloseTo(reference.area, 4);
        expect(reference.queries.map(query => rings.some(ring => contains(ring, query)))).toEqual(
          reference.contains
        );
      }
    } finally {
      compiled.destroy();
      harness.destroy();
    }
  }
});

it('geometry refinement reports finite-input arithmetic overflow instead of successful invalid results', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createHarness(device);
  const {graph, view} = harness;
  const circleOutput = {
    circles: view('float32x4', 1),
    witnesses: view('float32x2', 1),
    status: view('uint32', 1)
  };
  graph.add(
    new GPUMaximumInscribedCircle({
      positions: view('float32x2', 4, [0, 0, 1e20, 0, 1e20, 1e20, 0, 1e20]),
      ringOffsets: view('uint32', 2, [0, 4]),
      polygonOffsets: view('uint32', 2, [0, 1]),
      featureOffsets: view('uint32', 2, [0, 1]),
      tolerance: 1,
      output: circleOutput
    })
  );
  const lineOutput = {
    positions: view('float32x2', 4),
    lineOffsets: view('uint32', 3),
    status: view('uint32', 2)
  };
  graph.add(
    new GPULinePrecisionReduction({
      positions: view('float32x2', 4, [1e20, 0, 2e20, 0, 0, 0, 1e-30, 0]),
      lineOffsets: view('uint32', 3, [0, 2, 4]),
      gridSize: 1e-30,
      mode: 'valid-output',
      output: lineOutput
    })
  );
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled);
    expect(await harness.integers(circleOutput.status)).toEqual([3]);
    expect((await harness.floats(circleOutput.circles)).every(Number.isNaN)).toBe(true);
    expect((await harness.floats(circleOutput.witnesses)).every(Number.isNaN)).toBe(true);
    expect(await harness.integers(lineOutput.status)).toEqual([3, 0]);
    expect(await harness.integers(lineOutput.lineOffsets)).toEqual([0, 0, 2]);
    expect((await harness.floats(lineOutput.positions)).slice(0, 4)).toEqual([
      0,
      0,
      Math.fround(1e-30),
      0
    ]);
  } finally {
    compiled.destroy();
    harness.destroy();
  }
});

it('normalizes translated shell and hole windings without float32 area cancellation', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const reversed of [false, true]) {
    const harness = createHarness(device);
    const {graph, view} = harness;
    const translation = 100_000_000;
    const shell = [
      [0, 0],
      [64, 0],
      [64, 64],
      [0, 64]
    ];
    const hole = [
      [16, 16],
      [16, 48],
      [48, 48],
      [48, 16]
    ];
    if (reversed) {
      shell.reverse();
      hole.reverse();
    }
    const coordinates = [...shell, ...hole].flatMap(point =>
      point.map(value => value + translation)
    );
    const output = {
      positions: view('float32x2', 8),
      children: view('uint32', 2),
      status: view('uint32', 3)
    };
    graph.add(
      new GPUNormalizeGeometry({
        positions: view('float32x2', 8, coordinates),
        nodes: view('uint32x4', 3, [3, 0, 4, 10, 3, 4, 4, 11, 5, 0, 2, 12]),
        children: view('uint32', 2, [0, 1]),
        output
      })
    );
    const compiled = graph.compile();
    try {
      submitGraph(device, compiled);
      expect(await harness.integers(output.status)).toEqual([0, 0, 0]);
      expect(await harness.floats(output.positions)).toEqual(
        [0, 0, 0, 64, 64, 64, 64, 0, 16, 16, 48, 16, 48, 48, 16, 48].map(
          value => value + translation
        )
      );
    } finally {
      compiled.destroy();
      harness.destroy();
    }
  }
});

it('normalizes a wide collection by ranking children in parallel and bounds nesting depth', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // 40 single-point leaves with repeated coordinates, listed in a scrambled order, under a collection.
  const pointCount = 40;
  const coordinates: number[] = [];
  let seed = 7;
  const next = () => {
    seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
    return seed / 4294967296;
  };
  for (let point = 0; point < pointCount; point++) {
    coordinates.push(Math.floor(next() * 6), Math.floor(next() * 6));
  }
  const nodes: number[] = [];
  for (let point = 0; point < pointCount; point++) {
    nodes.push(0, point, 1, point);
  }
  nodes.push(7, 0, pointCount, 99);
  const children = Array.from({length: pointCount}, (_, slot) => (slot * 17) % pointCount);
  const run = async (
    nodeValues: number[],
    childValues: number[],
    positions: number[],
    depth?: number
  ) => {
    const harness = createHarness(device);
    const {graph, view} = harness;
    const input = {
      positions: view('float32x2', positions.length / 2, positions),
      nodes: view('uint32x4', nodeValues.length / 4, nodeValues),
      children: view('uint32', childValues.length, childValues)
    };
    const output = {
      positions: view('float32x2', input.positions.length),
      children: view('uint32', input.children.length),
      status: view('uint32', input.nodes.length)
    };
    graph.add(new GPUNormalizeGeometry({...input, maximumNestingDepth: depth, output}));
    const compiled = graph.compile();
    try {
      submitGraph(device, compiled);
      return {
        children: await harness.integers(output.children),
        status: await harness.integers(output.status)
      };
    } finally {
      compiled.destroy();
      harness.destroy();
    }
  };
  const wide = await run(nodes, children, coordinates);
  expect(wide.status.every(status => status === 0)).toBe(true);
  // Descending by (x, y), ties in source order (the order of the child list).
  const expected = [...children].sort((left, right) => {
    const dx = coordinates[2 * right] - coordinates[2 * left];
    if (dx !== 0) return dx;
    const dy = coordinates[2 * right + 1] - coordinates[2 * left + 1];
    return dy !== 0 ? dy : children.indexOf(left) - children.indexOf(right);
  });
  expect(wide.children).toEqual(expected);

  // A chain of collections around one point: the longest allowed chain has 64 nodes.
  const buildChain = (collectionCount: number) => {
    const chainNodes = [0, 0, 1, 0];
    const chainChildren: number[] = [];
    for (let collection = 0; collection < collectionCount; collection++) {
      chainNodes.push(7, chainChildren.length, 1, collection + 1);
      chainChildren.push(collection);
    }
    return {chainNodes, chainChildren};
  };
  const allowed = buildChain(63);
  const allowedResult = await run(allowed.chainNodes, allowed.chainChildren, [1, 2]);
  expect(allowedResult.status.every(status => status === 0)).toBe(true);
  const tooDeep = buildChain(64);
  const tooDeepResult = await run(tooDeep.chainNodes, tooDeep.chainChildren, [1, 2]);
  expect(tooDeepResult.status[64]).toBe(2);
  expect(tooDeepResult.status.slice(0, 64).every(status => status === 0)).toBe(true);
  const shallow = await run(allowed.chainNodes, allowed.chainChildren, [1, 2], 8);
  expect(shallow.status[7]).toBe(0);
  expect(shallow.status[8]).toBe(2);
  expect(shallow.status[63]).toBe(2);
});

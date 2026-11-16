// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GraphDataView} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {addBuildingTessellationRecipe} from '../../../src/gpu-spatial-analysis/urban-morphology/building-tessellation';
import {createWGSLKernelNode} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {requireTestDevice} from '../../utils/require-test-device';
import {RecipeTestFixture} from '../recipes/recipe-harness';
import reference from '../../oracles/fixtures/urban-ecology-advanced.json';

function publish(fixture: RecipeTestFixture, name: string, source: GraphDataView) {
  const output = fixture.output(name, source.format as 'float32', source.length);
  const components = source.format.endsWith('x4') ? 4 : source.format.endsWith('x2') ? 2 : 1;
  const type = source.format.startsWith('float') ? 'f32' : 'u32';
  fixture.graph.add(
    createWGSLKernelNode(fixture.graph, {
      id: name,
      operation: 'tessellation-grid-readback',
      invocationCount: source.length * components,
      bindings: [
        {name: 'source', view: source, type, access: 'read'},
        {name: 'output', view: output.view, type, access: 'read_write'}
      ],
      body: 'output[outputOffset+index]=source[sourceOffset+index];'
    })
  );
  return output;
}

function getRingArea(positions: number[], first: number, end: number): number {
  let area = 0;
  for (let vertex = first; vertex < end; vertex++) {
    const next = vertex + 1 === end ? first : vertex + 1;
    area +=
      0.5 *
      (positions[2 * vertex] * positions[2 * next + 1] -
        positions[2 * vertex + 1] * positions[2 * next]);
  }
  return area;
}

async function readTessellation(
  searchStructure: 'grid' | 'brute-force' | undefined,
  enclosureMode: 'convex' | 'simple',
  sampleCapacity = 8,
  input = {
    segments: reference.tessellation.segments as number[],
    owners: reference.tessellation.owners as number[],
    enclosure:
      enclosureMode === 'simple'
        ? (reference.tessellation.enclosure as number[])
        : [0, 0, 8, 0, 8, 8, 0, 8],
    enclosureRingOffsets: undefined as number[] | undefined,
    sampleSpacing: 1
  }
) {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = new RecipeTestFixture(device, `tessellation-${searchStructure}`);
  try {
    const enclosure = input.enclosure;
    const result = addBuildingTessellationRecipe(fixture.graph, {
      segments: fixture.input(
        'walls',
        Float32Array.from(input.segments),
        'float32x4',
        input.owners.length
      ),
      segmentBuildingIds: fixture.input(
        'owners',
        Uint32Array.from(input.owners),
        'uint32',
        input.owners.length
      ),
      buildingCount: Math.max(...input.owners) + 1,
      enclosure: fixture.input(
        'enclosure',
        Float32Array.from(enclosure),
        'float32x2',
        enclosure.length / 2
      ),
      enclosureMode,
      sampleSpacing: input.sampleSpacing,
      enclosureRingOffsets: input.enclosureRingOffsets
        ? fixture.input(
            'enclosure-rings',
            Uint32Array.from(input.enclosureRingOffsets),
            'uint32',
            input.enclosureRingOffsets.length
          )
        : undefined,
      sampleCapacity,
      vertexTolerance: 1 / 1024,
      capacity: {intersections: 4096, nodedSegments: 1024, vertices: 1024, rings: 128},
      searchStructure
    });
    const sampling = publish(fixture, 'sampling-status', result.status.sampling);
    const complete = publish(fixture, 'complete', result.status.complete);
    const fragments = publish(fixture, 'fragments', result.intermediates.fragments);
    const fragmentCounts = publish(fixture, 'fragment-counts', result.intermediates.fragmentCounts);
    const readbacks = result.outputs.map((output, building) => ({
      count: publish(fixture, `count-${building}`, output.status.count),
      positions: publish(fixture, `positions-${building}`, output.geometry.positions),
      features: publish(fixture, `features-${building}`, output.geometry.featureOffsets),
      polygons: publish(fixture, `polygons-${building}`, output.geometry.polygonOffsets),
      rings: publish(fixture, `rings-${building}`, output.geometry.ringOffsets)
    }));
    fixture.run();
    const fragmentPositions = await fixture.readFloat32(
      fragments,
      result.intermediates.fragments.length * 2
    );
    const counts = await fixture.readUint32(fragmentCounts, sampleCapacity);
    const stride = sampleCapacity + enclosure.length / 2;
    const fragmentAreas = counts.map((count, row) =>
      count === 0xffffffff
        ? Number.NaN
        : getRingArea(fragmentPositions, row * stride, row * stride + count)
    );
    const fragmentGeometry = counts.map((count, row) =>
      count === 0xffffffff
        ? []
        : fragmentPositions.slice(row * stride * 2, (row * stride + count) * 2)
    );
    const regions: {
      features: number[];
      polygons: number[];
      rings: number[];
      positions: number[];
    }[] = [];
    const buildingAreas: number[] = [];
    for (const output of readbacks) {
      const [count] = await fixture.readUint32(output.count, 1);
      const features = await fixture.readUint32(output.features, count + 1);
      const polygons = await fixture.readUint32(output.polygons, features[count] + 1);
      const ringCount = polygons.at(-1)!;
      const rings = await fixture.readUint32(output.rings, ringCount + 1);
      const positions = await fixture.readFloat32(output.positions, rings[ringCount] * 2);
      let area = 0;
      for (let ring = 0; ring < ringCount; ring++)
        area += getRingArea(positions, rings[ring], rings[ring + 1]);
      buildingAreas.push(Math.abs(area));
      regions.push({features, polygons, rings, positions});
    }
    return {
      fragmentAreas,
      fragmentGeometry,
      fragmentCounts: counts,
      regions,
      buildingAreas,
      complete: await fixture.readUint32(complete, Math.max(...input.owners) + 1),
      sampling: await fixture.readUint32(sampling, 4)
    };
  } finally {
    fixture.destroy();
  }
}

function containsRing(coordinates: number[], horizontal: number, vertical: number): boolean {
  let inside = false;
  for (
    let vertex = 0, previous = coordinates.length / 2 - 1;
    vertex < coordinates.length / 2;
    previous = vertex++
  ) {
    const firstX = coordinates[2 * vertex],
      firstY = coordinates[2 * vertex + 1];
    const secondX = coordinates[2 * previous],
      secondY = coordinates[2 * previous + 1];
    if (
      firstY > vertical !== secondY > vertical &&
      horizontal < ((secondX - firstX) * (vertical - firstY)) / (secondY - firstY) + firstX
    )
      inside = !inside;
  }
  return inside;
}
function expectCyclicRing(first: number[], second: number[], tolerance: number) {
  expect(first.length).toBe(second.length);
  if (first.length === 0) return;
  const count = first.length / 2;
  let smallestError = Infinity;
  for (let rotation = 0; rotation < count; rotation++) {
    let largestError = 0;
    for (let vertex = 0; vertex < count; vertex++) {
      const other = (vertex + rotation) % count;
      largestError = Math.max(
        largestError,
        Math.hypot(
          first[2 * vertex] - second[2 * other],
          first[2 * vertex + 1] - second[2 * other + 1]
        )
      );
    }
    smallestError = Math.min(smallestError, largestError);
  }
  // Preserve edge order and winding, allowing only a cyclic start shift.
  expect(smallestError).toBeLessThan(tolerance);
}
function removeRepeatedVertices(ring: number[], tolerance: number): number[] {
  const points: number[][] = [];
  for (let vertex = 0; vertex < ring.length / 2; vertex++) {
    const point = ring.slice(vertex * 2, vertex * 2 + 2);
    const previous = points.at(-1);
    if (!previous || Math.hypot(point[0] - previous[0], point[1] - previous[1]) > tolerance)
      points.push(point);
  }
  if (
    points.length > 1 &&
    Math.hypot(points[0][0] - points.at(-1)![0], points[0][1] - points.at(-1)![1]) <= tolerance
  )
    points.pop();
  return points.flat();
}
function compareGeometry(
  indexed: Awaited<ReturnType<typeof readTessellation>>,
  exact: Awaited<ReturnType<typeof readTessellation>>
) {
  // Grid traversal is opt-in: redundant vertices may differ. Compare oriented regions while
  // the square regression separately locks down raw fragment counts and repeated coordinates.
  indexed.fragmentGeometry.forEach((ring, row) =>
    expectCyclicRing(
      removeRepeatedVertices(ring, 1 / 1024),
      removeRepeatedVertices(exact.fragmentGeometry[row], 1 / 1024),
      1 / 1024
    )
  );
  indexed.regions.forEach((region, building) => {
    const expected = exact.regions[building];
    expect(region.features).toEqual(expected.features);
    expect(region.polygons).toEqual(expected.polygons);
    expect(region.rings).toEqual(expected.rings);
    for (let ring = 0; ring + 1 < region.rings.length; ring++)
      expectCyclicRing(
        region.positions.slice(2 * region.rings[ring], 2 * region.rings[ring + 1]),
        expected.positions.slice(2 * expected.rings[ring], 2 * expected.rings[ring + 1]),
        2 / 1024
      );
  });
}
function checkOwnership(
  result: Awaited<ReturnType<typeof readTessellation>>,
  segments: number[],
  owners: number[],
  enclosure: number[],
  ringOffsets: number[],
  spacing: number
) {
  const sites: {horizontal: number; vertical: number; owner: number}[] = [];
  owners.forEach((owner, row) => {
    const start = segments.slice(4 * row, 4 * row + 2),
      finish = segments.slice(4 * row + 2, 4 * row + 4);
    const count = Math.ceil(Math.hypot(finish[0] - start[0], finish[1] - start[1]) / spacing);
    for (let part = 0; part < count; part++)
      sites.push({
        horizontal: start[0] + (part / count) * (finish[0] - start[0]),
        vertical: start[1] + (part / count) * (finish[1] - start[1]),
        owner
      });
  });
  // Independent nearest-boundary-site oracle, away from both lattice and bisector boundaries.
  for (let row = 0; row < 31; row++)
    for (let column = 0; column < 31; column++) {
      const horizontal = 0.119 + column * 0.251,
        vertical = 0.173 + row * 0.247;
      let inEnclosure = false;
      for (let ring = 0; ring + 1 < ringOffsets.length; ring++)
        if (
          containsRing(
            enclosure.slice(2 * ringOffsets[ring], 2 * ringOffsets[ring + 1]),
            horizontal,
            vertical
          )
        )
          inEnclosure = !inEnclosure;
      const ordered = sites
        .map(site => ({
          ...site,
          distance: Math.hypot(horizontal - site.horizontal, vertical - site.vertical)
        }))
        .sort((first, second) => first.distance - second.distance);
      const other = ordered.find(site => site.owner !== ordered[0].owner);
      if (other && other.distance - ordered[0].distance < 0.01) continue;
      result.regions.forEach((region, building) => {
        let inside = false;
        for (let ring = 0; ring + 1 < region.rings.length; ring++)
          if (
            containsRing(
              region.positions.slice(2 * region.rings[ring], 2 * region.rings[ring + 1]),
              horizontal,
              vertical
            )
          )
            inside = !inside;
        expect(inside, `point ${horizontal},${vertical} owner ${building}`).toBe(
          inEnclosure && ordered[0].owner === building
        );
      });
    }
}

it.each([
  'convex',
  'simple'
] as const)('indexed clipping preserves oriented fragment regions and dissolved %s geometry', async enclosureMode => {
  const exact = await readTessellation('brute-force', enclosureMode);
  const indexed = await readTessellation('grid', enclosureMode);
  compareGeometry(indexed, exact);
  checkOwnership(
    indexed,
    reference.tessellation.segments,
    reference.tessellation.owners,
    enclosureMode === 'simple' ? reference.tessellation.enclosure : [0, 0, 8, 0, 8, 8, 0, 8],
    [0, enclosureMode === 'simple' ? 6 : 4],
    1
  );
  expect(indexed.sampling).toEqual(exact.sampling);
  expect(indexed.complete).toEqual([1, 1]);
  expect(indexed.complete).toEqual(exact.complete);
  indexed.fragmentAreas.forEach((area, row) =>
    expect(area).toBeCloseTo(exact.fragmentAreas[row], 4)
  );
  indexed.buildingAreas.forEach((area, row) =>
    expect(area).toBeCloseTo(exact.buildingAreas[row], 2)
  );
  if (enclosureMode === 'simple')
    indexed.buildingAreas.forEach((area, row) =>
      expect(area).toBeCloseTo(reference.tessellation.areas[row], 2)
    );
});

it('sample-grid construction excludes unused capacity rows', async () => {
  const exact = await readTessellation('brute-force', 'convex', 32);
  const indexed = await readTessellation('grid', 'convex', 32);
  expect(indexed.sampling).toEqual([8, 8, 0, 0]);
  expect(indexed.complete).toEqual(exact.complete);
  indexed.fragmentAreas.forEach((area, row) =>
    expect(area).toBeCloseTo(exact.fragmentAreas[row], 4)
  );
  indexed.buildingAreas.forEach((area, row) =>
    expect(area).toBeCloseTo(exact.buildingAreas[row], 2)
  );
});

it('sampling overflow stays incomplete with either candidate route', async () => {
  const exact = await readTessellation('brute-force', 'convex', 4);
  const indexed = await readTessellation('grid', 'convex', 4);
  expect(indexed.sampling).toEqual([4, 8, 1, 0]);
  expect(indexed.complete).toEqual([0, 0]);
  expect(indexed).toEqual(exact);
});

it.each([
  'irregular',
  'nearly-cocircular',
  'holes'
] as const)('indexed/reference geometry and independent ownership: %s', async shape => {
  const centers =
    shape === 'nearly-cocircular'
      ? [
          [2, 2],
          [6, 2.00001],
          [6, 6],
          [2, 5.99999]
        ]
      : [
          [1.3, 2.1],
          [5.7, 1.9],
          [4.3, 6.2]
        ];
  const segments: number[] = [],
    owners: number[] = [];
  centers.forEach(([horizontal, vertical], owner) => {
    const corners = [
      [horizontal - 0.15, vertical - 0.12],
      [horizontal + 0.17, vertical - 0.12],
      [horizontal + 0.17, vertical + 0.19],
      [horizontal - 0.15, vertical + 0.19]
    ];
    corners.forEach((corner, edge) => {
      segments.push(...corner, ...corners[(edge + 1) % 4]);
      owners.push(owner);
    });
  });
  const enclosure =
    shape === 'holes' ? [0, 0, 8, 0, 8, 8, 0, 8, 3, 3, 3, 5, 5, 5, 5, 3] : [0, 0, 8, 0, 8, 8, 0, 8];
  const input = {
    segments,
    owners,
    enclosure,
    enclosureRingOffsets: shape === 'holes' ? [0, 4, 8] : undefined,
    sampleSpacing: 1e20
  };
  const exact = await readTessellation(
    'brute-force',
    shape === 'holes' ? 'simple' : 'convex',
    owners.length,
    input
  );
  const indexed = await readTessellation(
    'grid',
    shape === 'holes' ? 'simple' : 'convex',
    owners.length,
    input
  );
  compareGeometry(indexed, exact);
  expect(indexed.complete).toEqual(centers.map(() => 1));
  checkOwnership(
    indexed,
    segments,
    owners,
    enclosure,
    input.enclosureRingOffsets ?? [0, 4],
    input.sampleSpacing
  );
});
it('duplicate boundary sites across owners invalidate both candidate routes', async () => {
  const input = {
    segments: [
      1, 1, 2, 1, 2, 1, 2, 2, 2, 2, 1, 2, 1, 2, 1, 1, 2, 2, 3, 2, 3, 2, 3, 3, 3, 3, 2, 3, 2, 3, 2, 2
    ],
    owners: [0, 0, 0, 0, 1, 1, 1, 1],
    enclosure: [0, 0, 8, 0, 8, 8, 0, 8],
    enclosureRingOffsets: undefined,
    sampleSpacing: 1e20
  };
  const exact = await readTessellation('brute-force', 'convex', 8, input);
  const indexed = await readTessellation('grid', 'convex', 8, input);
  expect(indexed.complete).toEqual([0, 0]);
  expect(indexed.fragmentCounts.filter(count => count === 0xffffffff)).toHaveLength(2);
  compareGeometry(indexed, exact);
});

it.each([
  2, 1e20
])('the default reference preserves exhaustive square fragments at spacing %s', async sampleSpacing => {
  const input = {
    segments: [0, 0, 2, 0, 2, 0, 2, 2, 2, 2, 0, 2, 0, 2, 0, 0],
    owners: [0, 0, 0, 0],
    enclosure: [-1, -1, 3, -1, 3, 3, -1, 3],
    enclosureRingOffsets: undefined,
    sampleSpacing
  };
  const exhaustive = await readTessellation('brute-force', 'convex', 4, input);
  const implicit = await readTessellation(undefined, 'convex', 4, input);
  const indexed = await readTessellation('grid', 'convex', 4, input);
  expect(implicit).toEqual(exhaustive);
  expect(exhaustive.fragmentCounts).toEqual([5, 4, 5, 5]);
  expect(indexed.fragmentCounts).toEqual([4, 5, 5, 5]);
  expectCyclicRing(exhaustive.fragmentGeometry[0], [-1, -1, 1, -1, 1, 1, 1, 1, -1, 1], 1e-5);
  expectCyclicRing(indexed.fragmentGeometry[0], [-1, -1, 1, -1, 1, 1, -1, 1], 1e-5);
  // Different repeated-vertex multiplicities, with identical four rectangular sample regions.
  const boxes = [
    [-1, -1, 1, -1, 1, 1, -1, 1],
    [1, -1, 3, -1, 3, 1, 1, 1],
    [1, 1, 3, 1, 3, 3, 1, 3],
    [-1, 1, 1, 1, 1, 3, -1, 3]
  ];
  for (const result of [exhaustive, indexed]) {
    result.fragmentAreas.forEach(area => expect(area).toBeCloseTo(4, 5));
    expect(result.buildingAreas[0]).toBeCloseTo(16, 5);
    expect(result.complete).toEqual([1]);
    expect(result.sampling).toEqual([4, 4, 0, 0]);
    result.fragmentGeometry.forEach((ring, row) => {
      expectCyclicRing(removeRepeatedVertices(ring, 1e-5), boxes[row], 1e-5);
    });
    checkOwnership(result, input.segments, input.owners, input.enclosure, [0, 4], sampleSpacing);
  }
  expect(indexed.regions).toEqual(exhaustive.regions);
});

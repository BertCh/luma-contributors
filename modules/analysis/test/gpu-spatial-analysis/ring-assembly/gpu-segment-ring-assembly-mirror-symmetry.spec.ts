// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// A scene and its mirror image with the interior on the other side must give the same rings up
// to the reflection. Finding frc-geometry-07 of the re-review in
// `dev-docs/research/math-correctness-review-2026-11-10/fix-review-closing-geometry.md`.
//
// `GPUSegmentRingAssembly` continues a segment into the candidate that turns tightest toward the
// interior side. A candidate that exactly reverses the incoming segment (a u-turn) is the
// tightest turn for either side: it is the first edge met when turning from the reversed
// incoming direction toward the interior. The kernel ranked the u-turn first with the interior
// on the left and last with the interior on the right, so the mirror image of a scene could lose
// rings that the scene had.
//
// Expected results: the reflection itself (the same segment rows on the same rings), and for
// the scenes with a known answer the rings of the face rule worked out by hand (the spike) or
// the two parts of shapely.union_all (the polar scenes of
// test/oracles/ring-assembly-float32-choices.py, Shapely 2.2.0 on GEOS 3.14.1).

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import type {GPUSpatialContext} from '../../../src/analysis-core/contracts/spatial-context';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  GPUSegmentRingAssembly,
  GPU_SEGMENT_RING_ASSEMBLY_NONE
} from '../../../src/analysis-core/ring-assembly/index';
import {
  createInputBuffer,
  createOutputBuffer,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import fixtureJSON from '../../oracles/fixtures/ring-assembly-float32-choices.json';

type Segment = [number, number, number, number];
type Scene = {name: string; segments: Segment[]};
type InteriorSide = 'left' | 'right';

/** Two triangles that meet within the tolerance beside the antimeridian and the pole. */
type PolarTurnScene = Scene & {
  form: 'apart' | 'closing';
  /** `shapely.union_all` of the two triangles: 2. */
  partCount: number;
  /** A candidate is exactly opposite to the segment that arrives at the tips. */
  oppositeCandidate: boolean;
};

const POLAR_SCENES = (fixtureJSON as unknown as {polarTurns: PolarTurnScene[]}).polarTurns;

const LONGITUDE_LATITUDE: GPUSpatialContext = {
  coordinateSpace: 'longitude-latitude',
  metric: 'none'
};

/** The mirror image in the y axis: the interior changes side, the rows keep their order. */
function mirrorScene(scene: Scene): Scene {
  return {
    name: `${scene.name}, mirrored`,
    segments: scene.segments.map(([x0, y0, x1, y1]): Segment => [-x0, y0, -x1, y1])
  };
}

/** Every segment reversed: the interior changes side, the rows keep their order. */
function reverseScene(scene: Scene): Scene {
  return {
    name: `${scene.name}, reversed`,
    segments: scene.segments.map(([x0, y0, x1, y1]): Segment => [x1, y1, x0, y0])
  };
}

/**
 * Assembles every scene in one run, each in its own group. Returns, per scene and segment row,
 * the lowest row of the ring the segment is on, or -1 when it is on no ring: the rings of a
 * scene as a partition of its rows, which a reflection must keep.
 */
async function getRingPartitions(
  device: Device,
  scenes: readonly Scene[],
  options: {
    vertexTolerance: number;
    interiorSide: InteriorSide;
    spatialContext?: GPUSpatialContext;
  }
): Promise<number[][]> {
  const rows = scenes.reduce((sum, scene) => sum + scene.segments.length, 0);
  const ringCapacity = rows;
  const buffers: Buffer[] = [];
  const graph = new GPUCommandGraph(device, {id: 'ring-assembly-mirror-symmetry'});
  const output = <Format extends 'uint32' | 'float32x2'>(
    name: string,
    format: Format,
    length: number
  ) => {
    const buffer = createOutputBuffer(device, length * (format === 'float32x2' ? 2 : 1));
    buffers.push(buffer);
    return {buffer, view: importGraphBuffer(graph, name, buffer, format, length)};
  };
  const endpoints = createInputBuffer(
    device,
    Float32Array.from(scenes.flatMap(scene => scene.segments.flat()))
  );
  const groups = createInputBuffer(
    device,
    Uint32Array.from(scenes.flatMap((scene, index) => scene.segments.map(() => index)))
  );
  buffers.push(endpoints, groups);
  const ringOffsets = output('ring-offsets', 'uint32', ringCapacity + 1);
  const positions = output('positions', 'float32x2', rows + ringCapacity);
  const segmentRings = output('segment-rings', 'uint32', rows);
  const count = output('count', 'uint32', 1);
  const overflow = output('overflow', 'uint32', 1);
  graph.add(
    new GPUSegmentRingAssembly({
      endpoints: importGraphBuffer(graph, 'endpoints', endpoints, 'float32x4', rows),
      groups: importGraphBuffer(graph, 'groups', groups, 'uint32', rows),
      vertexTolerance: options.vertexTolerance,
      interiorSide: options.interiorSide,
      spatialContext: options.spatialContext,
      output: {
        ringOffsets: ringOffsets.view,
        positions: positions.view,
        segmentRings: segmentRings.view,
        status: {count: count.view, overflow: overflow.view}
      }
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const ringOfSegment = await readUint32(segmentRings.buffer, rows);
  const overflowCount = (await readUint32(overflow.buffer, 1))[0];
  compiled.destroy();
  for (const buffer of buffers) {
    buffer.destroy();
  }
  expect(overflowCount).toBe(0);
  const partitions: number[][] = [];
  let firstRow = 0;
  for (const scene of scenes) {
    const rings = Array.from(ringOfSegment.slice(firstRow, firstRow + scene.segments.length));
    partitions.push(
      rings.map(ring => (ring === GPU_SEGMENT_RING_ASSEMBLY_NONE ? -1 : rings.indexOf(ring)))
    );
    firstRow += scene.segments.length;
  }
  return partitions;
}

/** Names of the scenes whose partition differs from `expected` (one per scene, or one for all). */
function getDepartures(
  scenes: readonly Scene[],
  partitions: readonly number[][],
  expected: readonly number[][] | readonly number[]
): string[] {
  const departures: string[] = [];
  scenes.forEach((scene, index) => {
    const wanted = Array.isArray(expected[0])
      ? (expected as readonly number[][])[index]
      : (expected as readonly number[]);
    if (JSON.stringify(partitions[index]) !== JSON.stringify(wanted)) {
      departures.push(`${scene.name}: rings ${partitions[index]}, expected ${wanted}`);
    }
  });
  return departures;
}

it('a square with a zero-width spike is the square and the spike for either interior side', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // Counter-clockwise square (0, 0) - (10, 0) - (10, 10) - (0, 10) with a spike from (10, 0) to
  // (20, 0) and back. Row 2 arrives at (10, 0) from the spike; row 1 exactly reverses it and row
  // 3 turns away from the interior. By the face rule row 2 continues into row 1 (the spike is a
  // ring of two segments) and row 0 into row 3 (the square).
  const spike: Scene = {
    name: 'spike square',
    segments: [
      [0, 0, 10, 0],
      [10, 0, 20, 0],
      [20, 0, 10, 0],
      [10, 0, 10, 10],
      [10, 10, 0, 10],
      [0, 10, 0, 0]
    ]
  };
  const expected = [0, 1, 1, 0, 0, 0];
  const placements: [Scene, InteriorSide][] = [
    [spike, 'left'],
    [mirrorScene(spike), 'right'],
    [reverseScene(spike), 'right'],
    [mirrorScene(reverseScene(spike)), 'left']
  ];
  for (const [scene, interiorSide] of placements) {
    const [partition] = await getRingPartitions(device, [scene], {
      vertexTolerance: 2 ** -8,
      interiorSide
    });
    expect(partition, `${scene.name}, interior ${interiorSide}`).toEqual(expected);
  }
  device.destroy?.();
});

it('the fixture holds polar scenes with a candidate exactly opposite to the incoming segment', () => {
  expect(POLAR_SCENES.length).toBe(240);
  expect(POLAR_SCENES.filter(scene => scene.oppositeCandidate).map(scene => scene.name)).toEqual([
    'closing-182',
    'closing-186',
    'closing-200'
  ]);
  for (const scene of POLAR_SCENES) {
    expect(scene.partCount, scene.name).toBe(2);
    for (const value of scene.segments.flat()) {
      expect(Math.fround(value), scene.name).toBe(value);
    }
  }
});

it('two triangles beside the pole come back as two rings, as generated and mirrored', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // Rows 0 to 2 are the sliver and rows 3 to 5 the triangle (shapely.union_all: two parts). The
  // tips are at most 0.3 steps of 2^-12 degrees apart per axis; every other vertex is 4 steps or
  // more from any other.
  const expected = [0, 0, 0, 3, 3, 3];
  const mirrored = POLAR_SCENES.map(mirrorScene);
  for (const spatialContext of [undefined, LONGITUDE_LATITUDE]) {
    const label = spatialContext ? 'longitude/latitude' : 'planar';
    const asGenerated = await getRingPartitions(device, POLAR_SCENES, {
      vertexTolerance: 2 ** -13,
      interiorSide: 'left',
      spatialContext
    });
    expect(getDepartures(POLAR_SCENES, asGenerated, expected), `${label}, interior left`).toEqual(
      []
    );
    const asMirrored = await getRingPartitions(device, mirrored, {
      vertexTolerance: 2 ** -13,
      interiorSide: 'right',
      spatialContext
    });
    expect(getDepartures(mirrored, asMirrored, expected), `${label}, interior right`).toEqual([]);
  }
  device.destroy?.();
}, 120_000);

/** A seeded generator of numbers in [0, 1). */
function createRandom(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 4294967296;
  };
}

/**
 * Three to six counter-clockwise lattice triangles with one corner at the origin and the other
 * corners on a few rays, so that edges of different triangles leave the origin in the same
 * direction, in exactly opposite directions, and back along an edge that arrives. The triangles
 * may overlap and repeat: the scene is no planar boundary, most of its segments end up open, and
 * the reflection must still hold row for row. With
 * `isMoved`, each triangle is translated by 0 or 1 lattice steps per axis, so the corners at the
 * origin match within a tolerance of 1.5 without being equal.
 */
function createFanScene(seed: number, isMoved: boolean): Scene {
  const random = createRandom(seed);
  const integer = (low: number, high: number) => low + Math.floor(random() * (high - low + 1));
  const rays: [number, number][] = [];
  const rayCount = integer(4, 8);
  for (let attempt = 0; rays.length < rayCount && attempt < 400; attempt++) {
    let ray: [number, number] = [integer(-60, 60), integer(-60, 60)];
    if (rays.length > 0 && random() < 0.3) {
      // Exactly opposite to an earlier ray.
      const other = rays[integer(0, rays.length - 1)];
      ray = [-other[0], -other[1]];
    }
    const isFar = Math.max(Math.abs(ray[0]), Math.abs(ray[1])) >= 30;
    const isApart = rays.every(
      other => Math.max(Math.abs(ray[0] - other[0]), Math.abs(ray[1] - other[1])) >= 8
    );
    if (isFar && isApart) {
      rays.push(ray);
    }
  }
  const segments: Segment[] = [];
  const triangleCount = integer(3, 6);
  for (let attempt = 0; segments.length < 3 * triangleCount && attempt < 200; attempt++) {
    let first = rays[integer(0, rays.length - 1)];
    let second = rays[integer(0, rays.length - 1)];
    const cross = first[0] * second[1] - first[1] * second[0];
    if (cross === 0) {
      continue;
    }
    if (cross < 0) {
      [first, second] = [second, first];
    }
    const shift: [number, number] = isMoved ? [integer(0, 1), integer(0, 1)] : [0, 0];
    const corners: [number, number][] = [
      shift,
      [first[0] + shift[0], first[1] + shift[1]],
      [second[0] + shift[0], second[1] + shift[1]]
    ];
    for (let corner = 0; corner < 3; corner++) {
      segments.push([...corners[corner], ...corners[(corner + 1) % 3]]);
    }
  }
  return {name: `fan ${seed}${isMoved ? ' moved' : ''}`, segments};
}

it('fans of triangles at one vertex give the same rings as their mirror images with the interior on the other side', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const isMoved of [false, true]) {
    const scenes = Array.from({length: 300}, (_, index) => createFanScene(index + 1, isMoved));
    const mirrored = scenes.map(mirrorScene);
    // Exact matches for the shared corner; 1.5 for corners up to one lattice step apart (every
    // other vertex is at least 7 from any other).
    const vertexTolerance = isMoved ? 1.5 : 2 ** -8;
    for (const [interiorSide, mirroredSide] of [
      ['left', 'right'],
      ['right', 'left']
    ] as const) {
      const partitions = await getRingPartitions(device, scenes, {vertexTolerance, interiorSide});
      const mirroredPartitions = await getRingPartitions(device, mirrored, {
        vertexTolerance,
        interiorSide: mirroredSide
      });
      expect(
        getDepartures(mirrored, mirroredPartitions, partitions),
        `${isMoved ? 'moved' : 'exact'}, interior ${interiorSide} against its mirror image`
      ).toEqual([]);
      // The scenes do exercise the rule: in many of them a segment continues into its exact
      // reversal and the two form a ring of two segments.
      const scenesWithReversalRing = partitions.filter(partition =>
        partition.some(ring => ring >= 0 && partition.filter(other => other === ring).length === 2)
      );
      expect(scenesWithReversalRing.length).toBeGreaterThan(50);
    }
  }
  device.destroy?.();
}, 120_000);

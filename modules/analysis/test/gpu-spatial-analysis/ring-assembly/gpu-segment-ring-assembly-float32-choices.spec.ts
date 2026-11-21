// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  GPUSegmentRingAssembly,
  GPU_SEGMENT_RING_ASSEMBLY_NONE
} from '../../../src/analysis-core/ring-assembly/index';
import {GPUCoverageDissolve} from '../../../src/gpu-spatial-analysis/polygon-coverage-topology/index';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {WeightsRig} from '../spatial-weights/spatial-weights-harness';
import {flattenPolygons, type OraclePolygons} from '../spatial-weights/spatial-weights-oracle';
import fixtureJSON from '../../oracles/fixtures/ring-assembly-float32-choices.json';

// Two choices of GPUSegmentRingAssembly that float32 arithmetic used to decide (closing pass of
// the math correctness review, 2026-11-11), on the inputs that broke them:
//
// 1. The shell of a hole. It was the containing shell of smallest float32 area among those larger
//    than the hole. In the `wedges` scenes consecutive nested rings differ in area by a few units
//    in the last place of the products that make the areas up, and the float32 areas come out in
//    the wrong order: a hole was left without a shell, or given to the outer of two shells.
// 2. The continuation at a vertex that two rings share only within the tolerance. It was ordered
//    by a float32 turn key, which cannot separate directions 2^-43 radians apart.
//
// Expected results are GEOS results (shapely.build_area and shapely.union_all, Shapely 2.2.0 on
// GEOS 3.14.1) written by test/oracles/ring-assembly-float32-choices.py.

const NONE = GPU_SEGMENT_RING_ASSEMBLY_NONE;

type Point = [number, number];
type Segment = [number, number, number, number];

/** Nested rings and, per ring, the index of the shell that owns it (GEOS BuildArea). */
type NestingScene = {name: string; rings: Point[][]; shellOfRing: number[]};
/** Two disjoint triangles whose tips are a quarter unit apart. */
type TurnScene = {name: string; rings: Point[][]; partCount: number; separation: number};

const FIXTURE = fixtureJSON as unknown as {
  wedges: NestingScene[];
  bands: NestingScene[];
  turns: TurnScene[];
};

type AssemblyRun = {
  count: number;
  overflow: number;
  open: number;
  isHole: number[];
  shells: number[];
  areas: number[];
  /** Vertices of every written ring, without the closing vertex. */
  rings: Point[][];
};

function getRingSegments(ring: readonly Point[]): Segment[] {
  return ring.map((point, index) => [...point, ...ring[(index + 1) % ring.length]] as Segment);
}

/** Identity of a ring: its vertex set. */
function getRingKey(ring: readonly Point[]): string {
  return ring
    .map(([x, y]) => `${x},${y}`)
    .sort()
    .join(' ');
}

function readRings(
  offsets: ArrayLike<number>,
  coordinates: ArrayLike<number>,
  count: number
): Point[][] {
  const rings: Point[][] = [];
  for (let ring = 0; ring < count; ring++) {
    const points: Point[] = [];
    // The last vertex of a written ring repeats its first.
    for (let vertex = offsets[ring]; vertex + 1 < offsets[ring + 1]; vertex++) {
      points.push([coordinates[2 * vertex], coordinates[2 * vertex + 1]]);
    }
    rings.push(points);
  }
  return rings;
}

async function runAssembly(
  device: Device,
  segments: readonly Segment[],
  options: {vertexTolerance: number; interiorSide?: 'left' | 'right'}
): Promise<AssemblyRun> {
  const rows = segments.length;
  const ringCapacity = 8;
  const vertexCapacity = rows + ringCapacity;
  const buffers: Buffer[] = [];
  const graph = new GPUCommandGraph(device, {id: 'ring-assembly-float32-choices'});
  const output = <Format extends 'uint32' | 'float32' | 'float32x2'>(
    name: string,
    format: Format,
    length: number
  ) => {
    const buffer = createOutputBuffer(device, length * (format === 'float32x2' ? 2 : 1));
    buffers.push(buffer);
    return {buffer, view: importGraphBuffer(graph, name, buffer, format, length)};
  };
  const input = createInputBuffer(device, Float32Array.from(segments.flat()));
  buffers.push(input);
  const ringOffsets = output('ring-offsets', 'uint32', ringCapacity + 1);
  const positions = output('positions', 'float32x2', vertexCapacity);
  const isHole = output('is-hole', 'uint32', ringCapacity);
  const shells = output('shells', 'uint32', ringCapacity);
  const areas = output('areas', 'float32', ringCapacity);
  const count = output('count', 'uint32', 1);
  const overflow = output('overflow', 'uint32', 1);
  const open = output('open', 'uint32', 1);
  graph.add(
    new GPUSegmentRingAssembly({
      endpoints: importGraphBuffer(graph, 'endpoints', input, 'float32x4', rows),
      vertexTolerance: options.vertexTolerance,
      interiorSide: options.interiorSide ?? 'left',
      output: {
        ringOffsets: ringOffsets.view,
        positions: positions.view,
        ringIsHole: isHole.view,
        ringShells: shells.view,
        ringAreas: areas.view,
        status: {count: count.view, overflow: overflow.view, openSegmentCount: open.view}
      }
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const ringCount = (await readUint32(count.buffer, 1))[0];
  const result: AssemblyRun = {
    count: ringCount,
    overflow: (await readUint32(overflow.buffer, 1))[0],
    open: (await readUint32(open.buffer, 1))[0],
    isHole: await readUint32(isHole.buffer, ringCapacity),
    shells: await readUint32(shells.buffer, ringCapacity),
    areas: Array.from(await readFloat32(areas.buffer, ringCapacity)),
    rings: readRings(
      await readUint32(ringOffsets.buffer, ringCapacity + 1),
      await readFloat32(positions.buffer, 2 * vertexCapacity),
      Math.min(ringCount, ringCapacity)
    )
  };
  compiled.destroy();
  for (const buffer of buffers) {
    buffer.destroy();
  }
  return result;
}

async function runDissolve(
  device: Device,
  polygons: OraclePolygons,
  options: {snapTolerance?: number} = {}
): Promise<AssemblyRun> {
  const layout = flattenPolygons(polygons);
  const vertexCount = layout.positions.length / 2;
  const ringCapacity = 8;
  const vertexCapacity = vertexCount + ringCapacity;
  const rig = new WeightsRig(device);
  const out = {
    ringOffsets: rig.output('uint32', ringCapacity + 1),
    positions: rig.output('float32x2', vertexCapacity),
    areas: rig.output('float32', ringCapacity),
    isHole: rig.output('uint32', ringCapacity),
    shells: rig.output('uint32', ringCapacity),
    count: rig.output('uint32', 1),
    overflow: rig.output('uint32', 1),
    open: rig.output('uint32', 1)
  };
  rig.run(
    new GPUCoverageDissolve({
      positions: rig.input(layout.positions, 'float32x2', vertexCount),
      ringOffsets: rig.input(layout.ringOffsets, 'uint32', layout.ringOffsets.length),
      polygonOffsets: rig.input(layout.polygonOffsets, 'uint32', layout.polygonOffsets.length),
      featureOffsets: rig.input(layout.featureOffsets, 'uint32', layout.featureOffsets.length),
      // One label: the features dissolve into one region.
      labels: rig.input(new Uint32Array(polygons.length), 'uint32', polygons.length),
      snapGridSize: options.snapTolerance,
      output: {
        ringOffsets: out.ringOffsets.view,
        positions: out.positions.view,
        ringAreas: out.areas.view,
        ringIsHole: out.isHole.view,
        ringShells: out.shells.view,
        status: {
          count: out.count.view,
          overflow: out.overflow.view,
          openSegmentCount: out.open.view
        }
      }
    })
  );
  const ringCount = (await readUint32(out.count.buffer, 1))[0];
  const result: AssemblyRun = {
    count: ringCount,
    overflow: (await readUint32(out.overflow.buffer, 1))[0],
    open: (await readUint32(out.open.buffer, 1))[0],
    isHole: await readUint32(out.isHole.buffer, ringCapacity),
    shells: await readUint32(out.shells.buffer, ringCapacity),
    areas: Array.from(await readFloat32(out.areas.buffer, ringCapacity)),
    rings: readRings(
      await readUint32(out.ringOffsets.buffer, ringCapacity + 1),
      await readFloat32(out.positions.buffer, 2 * vertexCapacity),
      Math.min(ringCount, ringCapacity)
    )
  };
  rig.destroy();
  return result;
}

/** Shell of every scene ring as the run assigned it, as scene ring indices (NONE when unowned). */
function getAssignedShells(run: AssemblyRun, scene: NestingScene): number[] {
  const sceneRingOfKey = new Map(scene.rings.map((ring, index) => [getRingKey(ring), index]));
  const sceneRingOfRun = run.rings.map(ring => sceneRingOfKey.get(getRingKey(ring)) ?? -1);
  const assigned: number[] = scene.rings.map(() => -1);
  sceneRingOfRun.forEach((sceneRing, runRing) => {
    if (sceneRing >= 0) {
      const shell = run.shells[runRing];
      assigned[sceneRing] = shell === NONE ? NONE : sceneRingOfRun[shell];
    }
  });
  return assigned;
}

function expectNesting(run: AssemblyRun, scene: NestingScene, label: string) {
  expect(run.overflow, label).toBe(0);
  expect(run.open, label).toBe(0);
  expect(run.count, label).toBe(scene.rings.length);
  // Every ring comes back with its own vertices, and each hole with the shell GEOS gives it.
  expect(getAssignedShells(run, scene), label).toEqual(scene.shellOfRing);
  const sceneRingOfKey = new Map(scene.rings.map((ring, index) => [getRingKey(ring), index]));
  run.rings.forEach((ring, runRing) => {
    const sceneRing = sceneRingOfKey.get(getRingKey(ring))!;
    expect(run.isHole[runRing], label).toBe(scene.shellOfRing[sceneRing] === sceneRing ? 0 : 1);
  });
}

it('GPUSegmentRingAssembly gives each hole the shell GEOS gives it when float32 areas are out of order', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  let holeCount = 0;
  for (const [scenes, vertexTolerance] of [
    // Below the float32 step of the far side (2^-11) and of the apex coordinates.
    [FIXTURE.wedges, 2 ** -30],
    // Below the grid of projected metres (1/32 m east, 1/2 m north).
    [FIXTURE.bands, 2 ** -8]
  ] as const) {
    for (const scene of scenes) {
      for (const value of scene.rings.flat(2)) {
        expect(Math.fround(value)).toBe(value);
      }
      // Ring order follows the lowest segment row: outermost ring first, then innermost first.
      for (const order of ['outer-first', 'inner-first'] as const) {
        const rings = order === 'outer-first' ? scene.rings : [...scene.rings].reverse();
        const run = await runAssembly(device, rings.flatMap(getRingSegments), {vertexTolerance});
        expectNesting(run, scene, `${scene.name} ${order}`);
      }
      holeCount += scene.shellOfRing.filter((shell, ring) => shell !== ring).length;
    }
  }
  expect(holeCount).toBe(27);
  device.destroy?.();
});

it('GPUCoverageDissolve keeps the holes of thin nested polygons (GEOS build_area)', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const scene of [...FIXTURE.wedges, ...FIXTURE.bands]) {
    // One polygon per shell, with the holes GEOS gives it: a valid coverage of one label.
    const polygons: OraclePolygons = [];
    scene.shellOfRing.forEach((shell, ring) => {
      if (shell === ring) {
        polygons.push([
          scene.rings[ring],
          ...scene.rings.filter((_, other) => other !== ring && scene.shellOfRing[other] === ring)
        ]);
      }
    });
    expectNesting(await runDissolve(device, polygons), scene, scene.name);
  }
  device.destroy?.();
});

/** Negates x so that counter-clockwise rings become clockwise (interior on the right). */
function mirrorRing(ring: readonly Point[]): Point[] {
  return ring.map(([x, y]) => [-x, y]);
}

function expectBothTriangles(run: AssemblyRun, rings: readonly Point[][], label: string) {
  expect(run.overflow, label).toBe(0);
  // Before: one ring and three open segments. The edge into the moved tip took the sliver's
  // outgoing edge, because the two outgoing edges had equal float32 turn keys.
  expect(run.open, label).toBe(0);
  expect(run.count, label).toBe(2);
  expect(run.rings.map(getRingKey).sort(), label).toEqual(rings.map(getRingKey).sort());
  expect(run.isHole.slice(0, 2), label).toEqual([0, 0]);
}

it('GPUSegmentRingAssembly orders turns 2^-43 radians apart at a vertex matched within the tolerance', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const scene of FIXTURE.turns) {
    // GEOS: the triangles are disjoint (0.18 apart) and their union has two parts.
    expect(scene.partCount).toBe(2);
    expect(scene.separation).toBeGreaterThan(0.17);
    for (const interiorSide of ['left', 'right'] as const) {
      const rings = interiorSide === 'left' ? scene.rings : scene.rings.map(mirrorRing);
      for (const value of rings.flat(2)) {
        expect(Math.fround(value)).toBe(value);
      }
      // The tips are 0.25 apart and every other vertex is more than a unit from any other.
      const run = await runAssembly(device, rings.flatMap(getRingSegments), {
        vertexTolerance: 0.5,
        interiorSide
      });
      expectBothTriangles(run, rings, `${scene.name} ${interiorSide}`);
    }
  }
  device.destroy?.();
});

it('GPUCoverageDissolve keeps both of two triangles whose tips snap to neighbouring cells', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // The scene whose triangle is moved a quarter unit to the left: tips (1, 1) and (0.75, 1).
  const scene = FIXTURE.turns.find(entry => entry.name === 'tips-a-quarter-apart-left')!;
  const [sliver, triangle] = scene.rings;
  expect(sliver[0]).toEqual([1, 1]);
  expect(triangle[0]).toEqual([0.75, 1]);
  // Snap cells of 2 units anchored at the first vertex of the input, (2^21, 2^21 - 1): cell
  // floor((x - 2^21) / 2 + 0.5) is -1048575 for x = 1 and -1048576 for x = 0.75. The two tips
  // stay two points, 0.25 apart, inside the ring assembly's matching distance of
  // snapTolerance / 4 = 0.5, so the assembly matches them without making them equal.
  const polygons: OraclePolygons = [[[sliver[1], sliver[2], sliver[0]]], [triangle]];
  expect(polygons[0][0][0]).toEqual([2 ** 21, 2 ** 21 - 1]);
  const run = await runDissolve(device, polygons, {snapTolerance: 2});
  expectBothTriangles(run, scene.rings, scene.name);
  device.destroy?.();
});

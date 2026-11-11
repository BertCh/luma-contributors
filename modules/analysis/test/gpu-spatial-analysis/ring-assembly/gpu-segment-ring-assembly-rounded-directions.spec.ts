// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// The continuation at a vertex that two rings share only within the tolerance, when the float32
// differences `end - start` of the segments round. Finding frc-geometry-06 of
// `dev-docs/research/math-correctness-review-2026-11-10/fix-review-closing-geometry.md`.
//
// `GPUSegmentRingAssembly` orders the candidates at such a vertex by the directions of the
// segments. It took the directions from float32 differences, which are not the directions when
// the two ends of a segment differ in magnitude: here the tips are near (1, 1), the far vertices
// one to five million away, and the two edges out of the tips about 1e-7 radians apart. The
// order must be that of the exact directions, for planar input and for longitude/latitude input
// (whose cos(latitude) scale of the longitude differences changes no orientation sign).
//
// Expected results: the successors given by exact rational directions and the two parts of
// shapely.union_all of the two triangles (Shapely 2.2.0 on GEOS 3.14.1), written by
// test/oracles/ring-assembly-float32-choices.py (`roundedTurns`).

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

/** Two triangles that meet within the tolerance: rows 0 to 2 the sliver, rows 3 to 5 the triangle. */
type RoundedTurnScene = {
  name: string;
  form: 'apart' | 'closing';
  segments: Segment[];
  /** `shapely.union_all` of the two triangles: 2. */
  partCount: number;
  /** Successors of rows 2 and 5 from the float32 differences; the exact ones are rows 0 and 3. */
  float32Choice: [number, number];
  roundingChangesChoice: boolean;
};

const SCENES = (fixtureJSON as unknown as {roundedTurns: RoundedTurnScene[]}).roundedTurns;

/**
 * Assembles every scene in one run, each in its own group, and returns the names of the scenes
 * that did not come back as two rings (the sliver's three segments on one, the triangle's on
 * another).
 */
async function getScenesWithoutBothRings(
  device: Device,
  scenes: readonly {name: string; segments: Segment[]}[],
  options: {
    vertexTolerance: number;
    interiorSide: 'left' | 'right';
    spatialContext?: GPUSpatialContext;
  }
): Promise<string[]> {
  const rows = scenes.length * 6;
  const ringCapacity = scenes.length * 2 + 8;
  const buffers: Buffer[] = [];
  const graph = new GPUCommandGraph(device, {id: 'ring-assembly-rounded-directions'});
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
    Uint32Array.from(scenes.flatMap((_, index) => [index, index, index, index, index, index]))
  );
  buffers.push(endpoints, groups);
  const ringOffsets = output('ring-offsets', 'uint32', ringCapacity + 1);
  const positions = output('positions', 'float32x2', rows + ringCapacity);
  const segmentRings = output('segment-rings', 'uint32', rows);
  const count = output('count', 'uint32', 1);
  const overflow = output('overflow', 'uint32', 1);
  const open = output('open', 'uint32', 1);
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
        status: {count: count.view, overflow: overflow.view, openSegmentCount: open.view}
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
  const failures: string[] = [];
  scenes.forEach((scene, index) => {
    const rings = Array.from(ringOfSegment.slice(index * 6, index * 6 + 6));
    const hasBothRings =
      rings[0] !== GPU_SEGMENT_RING_ASSEMBLY_NONE &&
      rings[3] !== GPU_SEGMENT_RING_ASSEMBLY_NONE &&
      rings[0] === rings[1] &&
      rings[1] === rings[2] &&
      rings[3] === rings[4] &&
      rings[4] === rings[5] &&
      rings[0] !== rings[3];
    if (!hasBothRings) {
      failures.push(scene.name);
    }
  });
  return failures;
}

it('the fixture holds scenes in which the float32 direction vectors give the wrong continuation', () => {
  expect(SCENES.length).toBe(64);
  expect(SCENES.filter(scene => scene.roundingChangesChoice).length).toBe(48);
  for (const scene of SCENES) {
    expect(scene.partCount, scene.name).toBe(2);
    expect(scene.roundingChangesChoice, scene.name).toBe(
      scene.float32Choice[0] !== 0 || scene.float32Choice[1] !== 3
    );
    for (const value of scene.segments.flat()) {
      expect(Math.fround(value), scene.name).toBe(value);
    }
  }
});

it('GPUSegmentRingAssembly orders turns at a tolerance match by the exact directions of the segments', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // The tips are at most 0.3 apart per axis and every other vertex is at least 4 from any other.
  expect(
    await getScenesWithoutBothRings(device, SCENES, {vertexTolerance: 0.5, interiorSide: 'left'})
  ).toEqual([]);
  // Mirrored in x: the rings are clockwise and the interior is on the right.
  const mirrored = SCENES.map(scene => ({
    name: scene.name,
    segments: scene.segments.map(([x0, y0, x1, y1]): Segment => [-x0, y0, -x1, y1])
  }));
  expect(
    await getScenesWithoutBothRings(device, mirrored, {vertexTolerance: 0.5, interiorSide: 'right'})
  ).toEqual([]);
  device.destroy?.();
});

it('GPUSegmentRingAssembly gives the same exact order for longitude/latitude input', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // The scenes scaled by 2^-16 are degrees: tips near 1.5e-5 degrees and far vertices up to 80
  // degrees of longitude and latitude. The scale is a power of two, so every coordinate stays a
  // float32 value and every orientation keeps its sign.
  const scale = 2 ** -16;
  const scaled = SCENES.map(scene => ({
    name: scene.name,
    segments: scene.segments.map(segment => segment.map(value => value * scale) as Segment)
  }));
  for (const value of scaled.flatMap(scene => scene.segments.flat())) {
    expect(Math.fround(value)).toBe(value);
    expect(Math.abs(value)).toBeLessThan(85);
  }
  expect(
    await getScenesWithoutBothRings(device, scaled, {
      vertexTolerance: 0.5 * scale,
      interiorSide: 'left',
      spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'none'}
    })
  ).toEqual([]);
  device.destroy?.();
});

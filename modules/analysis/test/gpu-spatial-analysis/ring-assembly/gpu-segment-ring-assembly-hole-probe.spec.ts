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
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

// Hole-to-shell assignment of GPUSegmentRingAssembly on float32 world coordinates (math
// correctness review 2026-11-10, polygon-topology-01: the same side probe as GPUMakeValid).
// Expected values are read off the drawing: one shell, one hole inside it.

type Point = [number, number];

type AssemblyRun = {
  count: number;
  overflow: number;
  open: number;
  isHole: number[];
  shells: number[];
  segmentRings: number[];
  areas: number[];
  /** Number of polygon features of the `polygons` layout. */
  polygonCount: number;
  /** Vertices of the first polygon's shell in the `polygons` layout. */
  firstShell: Point[];
};

function getRingSegments(ring: readonly Point[]): number[] {
  return ring.flatMap((point, index) => [...point, ...ring[(index + 1) % ring.length]]);
}

async function runAssembly(device: Device, endpoints: readonly number[]): Promise<AssemblyRun> {
  const segments = endpoints.length / 4;
  const ringCapacity = 8;
  const vertexCapacity = 64;
  const buffers: Buffer[] = [];
  const graph = new GPUCommandGraph(device, {id: 'ring-assembly-hole-probe'});
  const output = <Format extends 'uint32' | 'float32' | 'float32x2'>(
    name: string,
    format: Format,
    length: number
  ) => {
    const buffer = createOutputBuffer(device, length * (format === 'float32x2' ? 2 : 1));
    buffers.push(buffer);
    return {buffer, view: importGraphBuffer(graph, name, buffer, format, length)};
  };
  const input = createInputBuffer(device, Float32Array.from(endpoints));
  buffers.push(input);
  const ringOffsets = output('ring-offsets', 'uint32', ringCapacity + 1);
  const positions = output('positions', 'float32x2', vertexCapacity);
  const isHole = output('is-hole', 'uint32', ringCapacity);
  const shells = output('shells', 'uint32', ringCapacity);
  const segmentRings = output('segment-rings', 'uint32', segments);
  const areas = output('areas', 'float32', ringCapacity);
  const polygonPositions = output('polygon-positions', 'float32x2', vertexCapacity);
  const polygonRingOffsets = output('polygon-ring-offsets', 'uint32', ringCapacity + 1);
  const polygonOffsets = output('polygon-offsets', 'uint32', ringCapacity + 1);
  const featureOffsets = output('feature-offsets', 'uint32', ringCapacity + 1);
  const count = output('count', 'uint32', 1);
  const overflow = output('overflow', 'uint32', 1);
  const open = output('open', 'uint32', 1);
  graph.add(
    new GPUSegmentRingAssembly({
      endpoints: importGraphBuffer(graph, 'endpoints', input, 'float32x4', segments),
      // Below one float32 step of the coordinates, so vertices one step apart stay distinct.
      vertexTolerance: 2 ** -26,
      interiorSide: 'left',
      output: {
        ringOffsets: ringOffsets.view,
        positions: positions.view,
        ringIsHole: isHole.view,
        ringShells: shells.view,
        ringAreas: areas.view,
        segmentRings: segmentRings.view,
        polygons: {
          kind: 'polygons',
          positions: polygonPositions.view,
          ringOffsets: polygonRingOffsets.view,
          polygonOffsets: polygonOffsets.view,
          featureOffsets: featureOffsets.view
        },
        status: {count: count.view, overflow: overflow.view, openSegmentCount: open.view}
      }
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const features = await readUint32(featureOffsets.buffer, ringCapacity + 1);
  const polygonRings = await readUint32(polygonRingOffsets.buffer, ringCapacity + 1);
  const polygonCoordinates = await readFloat32(polygonPositions.buffer, 2 * vertexCapacity);
  const firstShell: Point[] = [];
  for (let vertex = polygonRings[0]; vertex < polygonRings[1]; vertex++) {
    firstShell.push([polygonCoordinates[2 * vertex], polygonCoordinates[2 * vertex + 1]]);
  }
  const result: AssemblyRun = {
    areas: Array.from(await readFloat32(areas.buffer, ringCapacity)),
    polygonCount: features[ringCapacity],
    firstShell,
    count: (await readUint32(count.buffer, 1))[0],
    overflow: (await readUint32(overflow.buffer, 1))[0],
    open: (await readUint32(open.buffer, 1))[0],
    isHole: await readUint32(isHole.buffer, ringCapacity),
    shells: await readUint32(shells.buffer, ringCapacity),
    segmentRings: await readUint32(segmentRings.buffer, segments)
  };
  compiled.destroy();
  for (const buffer of buffers) {
    buffer.destroy();
  }
  return result;
}

it('GPUSegmentRingAssembly assigns a hole whose first edge is one float32 step tall', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // One float32 step at 4.
  const step = 2 ** -21;
  // Counter-clockwise shell: a square with a notch from the left whose tip (3, 4) touches the hole.
  const shell: Point[] = [
    [0, 0],
    [10, 0],
    [10, 10],
    [0, 10],
    [0, 5],
    [3, 4],
    [0, 3]
  ];
  // Clockwise hole. Its first edge climbs one float32 step from the notch tip, so the float32
  // midpoint of that edge is the tip itself. The point beside the true midpoint is inside the
  // shell (the notch is 1.5 steps narrower there); a ray at the tip's own height runs along the
  // lower notch edge instead.
  const hole: Point[] = [
    [3, 4],
    [3, 4 + step],
    [3, 7],
    [6, 7],
    [6, 4]
  ];
  // Hole rows first: the hole's leader is its one-step edge.
  const result = await runAssembly(device, [...getRingSegments(hole), ...getRingSegments(shell)]);
  expect(result.overflow).toBe(0);
  expect(result.open).toBe(0);
  expect(result.count).toBe(2);
  const holeRing = result.segmentRings[0];
  const shellRing = result.segmentRings[hole.length];
  expect(holeRing).not.toBe(shellRing);
  expect(result.isHole[holeRing]).toBe(1);
  expect(result.isHole[shellRing]).toBe(0);
  expect(result.shells[shellRing]).toBe(shellRing);
  expect(result.shells[holeRing]).toBe(shellRing);
  expect(result.shells[holeRing]).not.toBe(GPU_SEGMENT_RING_ASSEMBLY_NONE);
  device.destroy?.();
});

it('GPUSegmentRingAssembly keeps a ring whole when a zero-length segment sits on it', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const square: Point[] = [
    [0, 0],
    [4, 0],
    [4, 4],
    [0, 4]
  ];
  // A zero-length row at a corner, before and after the square's own rows. It has no direction,
  // so it must neither be chosen as a continuation nor take one from a real segment.
  for (const endpoints of [
    [4, 0, 4, 0, ...getRingSegments(square)],
    [...getRingSegments(square), 4, 0, 4, 0],
    [0, 0, 0, 0, 4, 4, 4, 4, ...getRingSegments(square)]
  ]) {
    const result = await runAssembly(device, endpoints);
    const label = JSON.stringify(endpoints);
    expect(result.overflow, label).toBe(0);
    expect(result.count, label).toBe(1);
    expect(result.isHole[0], label).toBe(0);
    // Every real segment is on the ring; the zero-length rows are on none and count as open.
    const zeroLengthRows = endpoints.length / 4 - square.length;
    expect(result.open, label).toBe(zeroLengthRows);
    endpoints.forEach((_, index) => {
      if (index % 4 !== 0) {
        return;
      }
      const row = index / 4;
      const isZeroLength =
        endpoints[index] === endpoints[index + 2] && endpoints[index + 1] === endpoints[index + 3];
      expect(result.segmentRings[row], `${label} row ${row}`).toBe(
        isZeroLength ? GPU_SEGMENT_RING_ASSEMBLY_NONE : 0
      );
    });
  }
  device.destroy?.();
});

it('GPUSegmentRingAssembly keeps a sliver face whose float32 area is exactly zero', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // Counter-clockwise slivers of exact doubled area 1 (the mirror walks of the slivers of
  // gpu-coverage-topology-float32.spec.ts). The ring starts at the origin, where its float32
  // shoelace sum is one cross product of two long vectors: 0 unless a fused multiply-add keeps
  // the right product exact, and the two slivers need opposite products. A ring of area 0 used to
  // be left out of the polygon layout as an empty walk.
  const slivers: Point[][] = [
    [
      [0, 0],
      [16777215, 16777214],
      [16777216, 16777215]
    ],
    [
      [0, 0],
      [16769025, 16773119],
      [16773121, 16777216]
    ]
  ];
  for (const sliver of slivers) {
    const label = JSON.stringify(sliver);
    const result = await runAssembly(device, getRingSegments(sliver));
    expect(result.overflow, label).toBe(0);
    expect(result.count, label).toBe(1);
    expect(result.isHole[0], label).toBe(0);
    expect(result.polygonCount, label).toBe(1);
    expect(result.firstShell, label).toEqual(sliver);
    // The measure is never zero for a ring with a certified turn, and its sign is the winding.
    expect(result.areas[0], label).toBeGreaterThan(0);
  }
  // A walk out and back along one segment pair is empty: it has no turn at its lowest vertex.
  const slit = await runAssembly(device, [0, 0, 4, 3, 4, 3, 0, 0]);
  expect(slit.count).toBe(1);
  expect(slit.areas[0]).toBe(0);
  expect(slit.polygonCount).toBe(0);
  // So is a bent path walked out and back (a -> b -> c -> b -> a, b lowest): the tightest turn
  // at b and at every tip is the way back, so it comes out as two such walks, both empty.
  const bentSlit = await runAssembly(device, [
    ...[-3, 4, 0, 0],
    ...[0, 0, 5, 2],
    ...[5, 2, 0, 0],
    ...[0, 0, -3, 4]
  ]);
  expect(bentSlit.count).toBe(2);
  expect(bentSlit.areas.slice(0, 2)).toEqual([0, 0]);
  expect(bentSlit.polygonCount).toBe(0);
  device.destroy?.();
});

// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUS2PolygonCovering} from '../../../src/gpu-spatial-analysis/cell-topology/gpu-s2-polygon-covering';
import type {GPUS2PolygonCoveringRefinement} from '../../../src/gpu-spatial-analysis/cell-topology/gpu-s2-polygon-covering';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {readUint32, submitGraph} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import recorded from './fixtures/s2-covering-ordered-parity.json';

type Snapshot = {cells: number[]; counts: number[]; status: number[]; visited: number[]};
type Case = {
  name: string;
  positions: number[];
  offsets: number[];
  mode: 'exterior' | 'interior';
  complement: boolean;
  maximumLevel: number;
  minimumLevel?: number;
  capacity: number;
  maximumVisitedCells: number;
  snapshots?: Snapshot[];
};

/** Runs one covering over the initial, a rotated and an invalid vertex set, then the initial again. */
async function executeCase(
  device: Device,
  reference: Case,
  refinement: GPUS2PolygonCoveringRefinement
): Promise<Snapshot[]> {
  const graph = new GPUCommandGraph(device);
  const buffers: Buffer[] = [];
  const allocate = (bytes: number) => {
    const buffer = device.createBuffer({
      byteLength: bytes,
      usage: Buffer.STORAGE | Buffer.COPY_DST | Buffer.COPY_SRC
    });
    buffers.push(buffer);
    return buffer;
  };
  const positions = allocate(reference.positions.length * 4);
  const offsets = allocate(reference.offsets.length * 4);
  offsets.write(new Uint32Array(reference.offsets));
  const cells = allocate(reference.capacity * 8);
  const counts = allocate(8);
  const status = allocate(4);
  const visited = allocate(4);
  graph.add(
    new GPUS2PolygonCovering({
      positions: importGraphBuffer(
        graph,
        'positions',
        positions,
        'float32x3',
        reference.positions.length / 3
      ),
      ringOffsets: importGraphBuffer(graph, 'offsets', offsets, 'uint32', reference.offsets.length),
      maximumLevel: reference.maximumLevel,
      minimumLevel: reference.minimumLevel,
      mode: reference.mode,
      complement: reference.complement,
      capacity: reference.capacity,
      maximumVisitedCells: reference.maximumVisitedCells,
      refinement,
      output: {
        cells: importGraphBuffer(graph, 'cells', cells, 'uint32x2', reference.capacity),
        counts: importGraphBuffer(graph, 'counts', counts, 'uint32x2', 1),
        status: importGraphBuffer(graph, 'status', status, 'uint32', 1),
        visitedCount: importGraphBuffer(graph, 'visited', visited, 'uint32', 1)
      }
    })
  );
  const compiled = graph.compile();
  const initial = new Float32Array(reference.positions);
  const rotated = Float32Array.from(initial, (value, index) =>
    index % 3 === 0 ? -initial[index + 1]! : index % 3 === 1 ? initial[index - 1]! : value
  );
  const invalid = initial.slice();
  invalid[0] = NaN;
  const snapshots: Snapshot[] = [];
  try {
    for (const values of [initial, rotated, invalid, initial]) {
      positions.write(values);
      submitGraph(device, compiled, undefined);
      snapshots.push({
        cells: await readUint32(cells, reference.capacity * 2),
        counts: await readUint32(counts, 2),
        status: await readUint32(status, 1),
        visited: await readUint32(visited, 1)
      });
    }
    return snapshots;
  } finally {
    compiled.destroy();
    for (const buffer of buffers) buffer.destroy();
  }
}

/** A convex ring of `count` vertices on the circle of angular `radius` around `center`. */
function createCircleRing(count: number, radius: number, center: number[]): number[] {
  const axis = Math.abs(center[2]!) < 0.9 ? [0, 0, 1] : [1, 0, 0];
  const first = normalizeVector(crossVectors(axis, center));
  const second = crossVectors(center, first);
  return Array.from({length: count}, (_, index) => {
    const angle = (2 * Math.PI * index) / count;
    return [0, 1, 2].map(
      axisIndex =>
        Math.cos(radius) * center[axisIndex]! +
        Math.sin(radius) *
          (Math.cos(angle) * first[axisIndex]! + Math.sin(angle) * second[axisIndex]!)
    );
  }).flat();
}
function crossVectors(left: number[], right: number[]): number[] {
  return [
    left[1]! * right[2]! - left[2]! * right[1]!,
    left[2]! * right[0]! - left[0]! * right[2]!,
    left[0]! * right[1]! - left[1]! * right[0]!
  ];
}
function normalizeVector(vector: number[]): number[] {
  const length = Math.hypot(...vector);
  return vector.map(value => value / length);
}

it('level-synchronous S2 polygon covering equals the recorded serial covers, in order', {
  timeout: 300000
}, async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const cases = recorded.cases as Case[];
  expect(cases.length).toBeGreaterThanOrEqual(50);
  for (const reference of cases) {
    const [initial, rotated, invalid, replay] = await executeCase(
      device,
      reference,
      'level-synchronous'
    );
    const expected = reference.snapshots!;
    const note = reference.name;
    if ((expected[0]!.status[0]! & 4) !== 0) {
      // Exhausted budget: the serial traversal keeps a depth-first prefix, this one keeps nothing.
      for (const snapshot of [initial!, rotated!, replay!]) {
        expect(snapshot.status[0]! & 4, note).toBe(4);
        expect(snapshot.counts, note).toEqual([0, 0]);
        expect(
          snapshot.cells.every(word => word === 0),
          note
        ).toBe(true);
        expect(snapshot.visited, note).toEqual(expected[0]!.visited);
      }
    } else {
      for (const [snapshot, recordedSnapshot] of [
        [initial!, expected[0]!],
        [rotated!, expected[1]!],
        [replay!, expected[3]!]
      ] as const) {
        expect(snapshot.status, note).toEqual(recordedSnapshot.status);
        expect(snapshot.counts, note).toEqual(recordedSnapshot.counts);
        expect(snapshot.visited, note).toEqual(recordedSnapshot.visited);
        const written = recordedSnapshot.counts[0]! * 2;
        expect(snapshot.cells.slice(0, written), note).toEqual(recordedSnapshot.cells);
        expect(
          snapshot.cells.slice(written).every(word => word === 0),
          note
        ).toBe(true);
      }
    }
    expect(invalid, note).toEqual({
      cells: invalid!.cells.map(() => 0),
      counts: [0, 0],
      status: [1],
      visited: [0]
    });
  }
});

it('level-synchronous S2 polygon covering matches the serial traversal on a 96-vertex polygon at level 7', {
  timeout: 120000
}, async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const center = normalizeVector([0.4, -0.7, 0.55]);
  const positions = createCircleRing(96, 0.35, center);
  for (const mode of ['exterior', 'interior'] as const) {
    const reference: Case = {
      name: `circle-${mode}`,
      positions,
      offsets: [0, 96],
      mode,
      complement: false,
      minimumLevel: 2,
      maximumLevel: 7,
      capacity: 8192,
      maximumVisitedCells: 65536
    };
    const serial = (await executeCase(device, reference, 'serial'))[0]!;
    const parallel = (await executeCase(device, reference, 'level-synchronous'))[0]!;
    expect(serial.status, mode).toEqual([0]);
    expect(serial.counts[1]!, mode).toBeGreaterThan(50);
    expect(parallel, mode).toEqual(serial);
  }
  // A ring that crosses itself is invalid in both traversals.
  const bowtie = [1, 0.2, 0, 1, -0.2, 0.2, 1, 0.2, 0.2, 1, -0.2, 0];
  const lengths = bowtie.slice();
  const crossing: Case = {
    name: 'bowtie',
    positions: lengths.map((value, index) => (index % 3 === 0 ? value : value)),
    offsets: [0, 4],
    mode: 'exterior',
    complement: false,
    maximumLevel: 5,
    capacity: 64,
    maximumVisitedCells: 4096
  };
  crossing.positions = normalizeRows(crossing.positions);
  const [serialInvalid, parallelInvalid] = [
    (await executeCase(device, crossing, 'serial'))[0]!,
    (await executeCase(device, crossing, 'level-synchronous'))[0]!
  ];
  expect(serialInvalid.status).toEqual([1]);
  expect(parallelInvalid.status).toEqual([1]);
  expect(parallelInvalid.counts).toEqual([0, 0]);
});

function normalizeRows(values: number[]): number[] {
  const result: number[] = [];
  for (let row = 0; row < values.length; row += 3) {
    result.push(...normalizeVector(values.slice(row, row + 3)));
  }
  return result;
}

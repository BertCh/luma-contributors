// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma-contributors/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUS2PolygonCovering} from '../../../src/gpu-spatial-analysis/cell-topology/gpu-s2-polygon-covering';
import {importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {readUint32, submitGraph} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import sourceCases from './fixtures/s2-polygon-covering.json';
import recorded from './fixtures/s2-covering-ordered-parity.json';

type Snapshot = {cells: number[]; counts: number[]; status: number[]; visited: number[]};
type Case = {
  name: string;
  positions: number[];
  offsets: number[];
  mode: 'exterior' | 'interior';
  complement: boolean;
  maximumLevel: number;
  capacity: number;
  maximumVisitedCells: number;
  snapshots?: Snapshot[];
};
const environment = (import.meta as unknown as {env?: Record<string, string | undefined>}).env;
const capture = environment?.['VITE_CAPTURE_S2_PARITY'] === 'true';

async function executeCase(device: Device, reference: Case): Promise<Snapshot[]> {
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
  const counts = allocate(8),
    status = allocate(4),
    visited = allocate(4);
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
      mode: reference.mode,
      complement: reference.complement,
      capacity: reference.capacity,
      maximumVisitedCells: reference.maximumVisitedCells,
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
    index % 3 === 0 ? -initial[index + 1] : index % 3 === 1 ? initial[index - 1] : value
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
    for (const snapshot of snapshots) {
      const writtenWords = snapshot.counts[0] * 2;
      expect(snapshot.cells.slice(writtenWords).every(value => value === 0)).toBe(true);
      snapshot.cells = snapshot.cells.slice(0, writtenWords);
    }
    expect(snapshots[2]).toEqual({
      cells: [],
      counts: [0, 0],
      status: [1],
      visited: [0]
    });
    expect(snapshots[3]).toEqual(snapshots[0]);
    return snapshots;
  } finally {
    compiled.destroy();
    for (const buffer of buffers) buffer.destroy();
  }
}

function createCaptureCases(): Case[] {
  const polar = [0, 120, 240].map(longitude => {
    const angle = (longitude * Math.PI) / 180,
      latitude = (80 * Math.PI) / 180;
    return [
      Math.cos(latitude) * Math.cos(angle),
      Math.cos(latitude) * Math.sin(angle),
      Math.sin(latitude)
    ];
  });
  const geometries = [
    {
      name: 'octant-boundary',
      rings: [
        [
          [1, 0, 0],
          [0, 1, 0],
          [0, 0, 1]
        ]
      ]
    },
    {name: 'polar', rings: [polar]},
    ...sourceCases.cases.map((value, index) => ({name: `holes-${index}`, rings: value.rings}))
  ];
  return geometries.flatMap(geometry =>
    (['exterior', 'interior'] as const).flatMap(mode =>
      (geometry.name === 'octant-boundary' ? [false, true] : [false]).map(complement => {
        const offsets = [0];
        for (const ring of geometry.rings) offsets.push(offsets[offsets.length - 1] + ring.length);
        return {
          name: `${geometry.name}-${mode}-${complement}`,
          positions: geometry.rings.flat(2),
          offsets,
          mode,
          complement,
          maximumLevel: 3,
          capacity: 512,
          maximumVisitedCells: 65536
        };
      })
    )
  );
}

it('S2 classification preserves ordered outputs, exact budgets and changed-input replay', {
  timeout: 300000
}, async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  if (capture) {
    const references: Case[] =
      environment?.['VITE_CAPTURE_S2_BENCHMARK_ONLY'] === 'true'
        ? [4, 6].map(maximumLevel => ({
            name: `benchmark-level${maximumLevel}`,
            positions: [1, 0, 0, 0, 1, 0, 0, 0, 1],
            offsets: [0, 3],
            mode: 'exterior',
            complement: false,
            maximumLevel,
            capacity: 4096,
            maximumVisitedCells: 65536
          }))
        : createCaptureCases();
    for (const reference of references) {
      const snapshots = await executeCase(device, reference);
      expect(snapshots[0].status).toEqual([0]);
      console.log(`S2_ORDERED_PARITY ${JSON.stringify({...reference, snapshots})}`);
      const required = snapshots[0].counts[1],
        visited = snapshots[0].visited[0];
      const limits = [
        ...new Set([1, Math.max(1, required - 1), Math.max(1, required), required + 1])
      ].map(capacity => ({capacity, maximumVisitedCells: 65536}));
      for (const maximumVisitedCells of new Set([
        6,
        Math.max(6, visited - 1),
        Math.max(6, visited)
      ]))
        limits.push({capacity: 512, maximumVisitedCells});
      for (const limit of limits) {
        const bounded = {
          ...reference,
          ...limit,
          name: `${reference.name}-${limit.capacity}-${limit.maximumVisitedCells}`
        };
        const boundedSnapshots = await executeCase(device, bounded);
        console.log(
          `S2_ORDERED_PARITY ${JSON.stringify({...bounded, snapshots: boundedSnapshots})}`
        );
      }
    }
  } else {
    const cases = recorded.cases as Case[];
    expect(cases.length).toBeGreaterThanOrEqual(50);
    for (const reference of cases)
      expect(await executeCase(device, reference), reference.name).toEqual(reference.snapshots);
  }
});

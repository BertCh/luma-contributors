// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {
  getDelaunayMaximumTriangleCount,
  GPUDelaunayTessellation
} from '../../../src/gpu-spatial-analysis/delaunay-tessellation/index';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {checkLocallyDelaunay, mulberry32, toTriangleRows} from './delaunay-reference';
import {requireTestDevice} from '../../utils/require-test-device';

type RunResult = {
  triangles: number[][];
  counts: number[];
  offsets: number[];
  converged: number[];
  invalid: number;
  duplicates: number;
};

/** Runs the tessellation with or without groups; without groups the group arrays are empty. */
async function runDelaunay(
  device: Device,
  coordinates: Float32Array,
  groupIds?: Uint32Array,
  groupCount = 0
): Promise<RunResult> {
  const graph = new GPUCommandGraph(device, {id: 'delaunay-groups'});
  const buffers: Buffer[] = [];
  const pointCount = coordinates.length / 2;
  const capacity = Math.max(getDelaunayMaximumTriangleCount(pointCount), 1);
  const positionsBuffer = createInputBuffer(device, coordinates);
  const triangleBuffer = createOutputBuffer(device, 3 * capacity);
  buffers.push(positionsBuffer, triangleBuffer);
  const output = (name: string, length: number) => {
    const buffer = createOutputBuffer(device, Math.max(length, 1));
    buffers.push(buffer);
    return {buffer, view: importGraphBuffer(graph, name, buffer, 'uint32', Math.max(length, 1))};
  };
  const count = output('count', 1);
  const requiredCount = output('required-count', 1);
  const overflow = output('overflow', 1);
  const invalidCount = output('invalid-count', 1);
  const duplicateCount = output('duplicate-count', 1);
  const groupViews = groupIds
    ? {
        counts: output('group-counts', groupCount),
        offsets: output('group-offsets', groupCount),
        converged: output('group-converged', groupCount)
      }
    : undefined;
  let groups: {ids: GraphDataView<'uint32'>; count: number} | undefined;
  if (groupIds) {
    const idsBuffer = createInputBuffer(device, groupIds);
    buffers.push(idsBuffer);
    groups = {
      ids: importGraphBuffer(graph, 'group-ids', idsBuffer, 'uint32', groupIds.length),
      count: groupCount
    };
  }
  graph.add(
    new GPUDelaunayTessellation({
      positions: importGraphBuffer(graph, 'positions', positionsBuffer, 'float32x2', pointCount),
      groups,
      output: {
        triangles: importGraphBuffer(graph, 'triangles', triangleBuffer, 'uint32x3', capacity),
        status: {
          count: count.view,
          requiredCount: requiredCount.view,
          overflow: overflow.view,
          invalidCount: invalidCount.view,
          duplicateCount: duplicateCount.view
        },
        groups: groupViews
          ? {
              triangleCounts: groupViews.counts.view,
              triangleOffsets: groupViews.offsets.view,
              status: {converged: groupViews.converged.view}
            }
          : undefined
      }
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const written = (await readUint32(count.buffer, 1))[0];
  const result: RunResult = {
    triangles: toTriangleRows(await readUint32(triangleBuffer, 3 * written)),
    counts: groupViews ? Array.from(await readUint32(groupViews.counts.buffer, groupCount)) : [],
    offsets: groupViews ? Array.from(await readUint32(groupViews.offsets.buffer, groupCount)) : [],
    converged: groupViews
      ? Array.from(await readUint32(groupViews.converged.buffer, groupCount))
      : [],
    invalid: (await readUint32(invalidCount.buffer, 1))[0],
    duplicates: (await readUint32(duplicateCount.buffer, 1))[0]
  };
  compiled.destroy();
  for (const buffer of buffers) {
    buffer.destroy();
  }
  return result;
}

function canonicalSet(rows: readonly (readonly number[])[]): string[] {
  return rows.map(row => [...row].sort((a, b) => a - b).join(',')).sort();
}

/** Four interleaved groups of random points, plus a group of two points and a stray label. */
function createGroupedFixture() {
  const random = mulberry32(2024);
  const coordinates: number[] = [];
  const groupIds: number[] = [];
  for (let row = 0; row < 120; row++) {
    coordinates.push(Math.fround(random() * 10), Math.fround(random() * 10));
    groupIds.push(row % 4);
  }
  // Group 4 has two points (no triangle). Row 122 has a label outside every group.
  coordinates.push(1, 1, 2, 2, 5, 5);
  groupIds.push(4, 4, 99);
  // The same coordinates in two groups are not duplicates of each other.
  coordinates.push(3, 3, 3, 3);
  groupIds.push(0, 1);
  return {coordinates: Float32Array.from(coordinates), groupIds: Uint32Array.from(groupIds)};
}

it('GPUDelaunayTessellation groups equal independent per-group triangulations', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const {coordinates, groupIds} = createGroupedFixture();
  const groupCount = 5;
  const grouped = await runDelaunay(device, coordinates, groupIds, groupCount);
  expect(grouped.invalid).toBe(0);
  expect(grouped.converged).toEqual([1, 1, 1, 1, 1]);
  expect(grouped.counts.reduce((sum, value) => sum + value, 0)).toBe(grouped.triangles.length);
  expect(grouped.offsets[0]).toBe(0);
  for (let group = 0; group < groupCount; group++) {
    const rows = Array.from({length: groupIds.length}, (_, row) => row).filter(
      row => groupIds[row] === group
    );
    const groupCoordinates = Float32Array.from(
      rows.flatMap(row => [coordinates[2 * row], coordinates[2 * row + 1]])
    );
    const alone = await runDelaunay(device, groupCoordinates);
    // Alone, rows are local indices into `rows`.
    const expected = canonicalSet(alone.triangles.map(triangle => triangle.map(v => rows[v])));
    const slice = grouped.triangles.slice(
      grouped.offsets[group],
      grouped.offsets[group] + grouped.counts[group]
    );
    expect(canonicalSet(slice), `group ${group}`).toEqual(expected);
    // Every triangle uses only rows of its group and is locally Delaunay among them.
    for (const triangle of slice) {
      for (const vertex of triangle) {
        expect(groupIds[vertex]).toBe(group);
      }
    }
    if (slice.length) {
      const local = slice.map(triangle => triangle.map(vertex => rows.indexOf(vertex)));
      expect(checkLocallyDelaunay(groupCoordinates, local).circleViolations).toBe(0);
    }
  }
  // Packed order: group-major with ascending canonical rows inside each group.
  expect(grouped.triangles.length).toBeGreaterThan(100);
});

it('GPUDelaunayTessellation invalidates only the group with a non-finite site', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const {coordinates, groupIds} = createGroupedFixture();
  const broken = Float32Array.from(coordinates);
  broken[2 * 1] = NaN; // a member of group 1
  const clean = await runDelaunay(device, coordinates, groupIds, 5);
  const grouped = await runDelaunay(device, broken, groupIds, 5);
  expect(grouped.converged).toEqual([1, 0, 1, 1, 1]);
  expect(grouped.counts[1]).toBe(0);
  for (const group of [0, 2, 3, 4]) {
    expect(
      canonicalSet(
        grouped.triangles.slice(
          grouped.offsets[group],
          grouped.offsets[group] + grouped.counts[group]
        )
      )
    ).toEqual(
      canonicalSet(
        clean.triangles.slice(clean.offsets[group], clean.offsets[group] + clean.counts[group])
      )
    );
  }
});

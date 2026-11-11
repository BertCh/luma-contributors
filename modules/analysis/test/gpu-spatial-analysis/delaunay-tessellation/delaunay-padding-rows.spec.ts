// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Coverage of this file, case by case.
//  - GPUDelaunayTessellation (with and without groups): the caller's output views are pre-filled
//    with stale words (valid-looking triples, set words, 0xffffffff, out-of-range point indices)
//    and the counted results must equal those of zero-filled output. Delaunay only writes its
//    triangle rows and never reads them, so these cases do NOT test any read of padding rows. What
//    they do exercise is the words the class accumulates into (`requiredCount` and the per-group
//    `triangleCounts`): each must be reset before use, or the stale value would show in the result.
//    No mutation of the class was run to confirm that these cases fail without the resets.
//  - GPUVoronoiDiagram: the triangle input carries stale rows after the `triangleCount` prefix
//    (valid-looking triples, an out-of-range index, an all-ones row) and the result must equal the
//    prefix-only result. This is the case that tests a read of padding rows.

import type {Buffer} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {
  getDelaunayMaximumTriangleCount,
  GPUDelaunayTessellation,
  GPUVoronoiDiagram
} from '../../../src/gpu-spatial-analysis/delaunay-tessellation/index';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

const POSITIONS = [0, 0, 4, 0, 4, 4, 0, 4, 2, 2, 6, 2, 9, 9, 9, 0, 1, 1.5];
const POINT_COUNT = POSITIONS.length / 2;
const GROUP_IDS = [0, 0, 0, 0, 0, 1, 1, 1, 1];
const STALE_WORDS = [0xffffffff, 1, 0, 3, 1, 2, 7, 4, 5, 0];

function stale(length: number, enabled: boolean): Uint32Array {
  return Uint32Array.from({length}, (_, index) => (enabled ? STALE_WORDS[index % 10] : 0));
}

async function runDelaunay(grouped: boolean, poisoned: boolean) {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device, {id: 'delaunay-padding-rows'});
  const buffers: Buffer[] = [];
  let serial = 0;
  const capacity = getDelaunayMaximumTriangleCount(POINT_COUNT);
  const output = (length: number, components = 1) => {
    const buffer = createOutputBuffer(device, length * components);
    buffer.write(stale(length * components, poisoned));
    buffers.push(buffer);
    return buffer;
  };
  const bind = <Format extends 'uint32' | 'uint32x3'>(
    buffer: Buffer,
    format: Format,
    length: number
  ) => importGraphBuffer(graph, `view-${serial++}`, buffer, format, length);
  const triangles = output(capacity, 3);
  const count = output(1);
  const requiredCount = output(1);
  const overflow = output(1);
  const groupCounts = output(2);
  const groupOffsets = output(2);
  const groupConverged = output(2);
  const positionsBuffer = createInputBuffer(device, Float32Array.from(POSITIONS));
  const groupIdsBuffer = createInputBuffer(device, Uint32Array.from(GROUP_IDS));
  buffers.push(positionsBuffer, groupIdsBuffer);
  graph.add(
    new GPUDelaunayTessellation({
      positions: importGraphBuffer(graph, 'positions', positionsBuffer, 'float32x2', POINT_COUNT),
      groups: grouped ? {ids: bind(groupIdsBuffer, 'uint32', POINT_COUNT), count: 2} : undefined,
      output: {
        triangles: bind(triangles, 'uint32x3', capacity),
        status: {
          count: bind(count, 'uint32', 1),
          requiredCount: bind(requiredCount, 'uint32', 1),
          overflow: bind(overflow, 'uint32', 1)
        },
        groups: grouped
          ? {
              triangleCounts: bind(groupCounts, 'uint32', 2),
              triangleOffsets: bind(groupOffsets, 'uint32', 2),
              status: {converged: bind(groupConverged, 'uint32', 2)}
            }
          : undefined
      }
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const written = (await readUint32(count, 1))[0];
  const result = {
    written,
    required: await readUint32(requiredCount, 1),
    overflow: await readUint32(overflow, 1),
    triangles: (await readUint32(triangles, written * 3)).slice(0, written * 3),
    groups: grouped
      ? [
          await readUint32(groupCounts, 2),
          await readUint32(groupOffsets, 2),
          await readUint32(groupConverged, 2)
        ]
      : []
  };
  compiled.destroy();
  buffers.forEach(buffer => buffer.destroy());
  return result;
}

for (const grouped of [false, true]) {
  it(`GPUDelaunayTessellation${grouped ? ' (groups)' : ''} resets its accumulated words and writes its rows over stale output`, async () => {
    const reference = await runDelaunay(grouped, false);
    expect(reference.written).toBeGreaterThan(0);
    expect(reference.overflow).toEqual([0]);
    expect(await runDelaunay(grouped, true)).toEqual(reference);
  });
}

/** The fan of four triangles around site 4 of the unit square. */
const FAN = [0, 1, 4, 1, 2, 4, 2, 3, 4, 3, 0, 4];
/** Rows after the counted prefix: valid-looking triples, an out-of-range index, an all-ones row. */
const STALE_TRIANGLES = [
  [0, 2, 4],
  [1, 3, 4],
  [0, 1, 2],
  [0, 1, 99],
  [0xffffffff, 0xffffffff, 0xffffffff]
];

async function runVoronoi(trailingRows: number[][]) {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device, {id: 'voronoi-padding-rows'});
  const buffers: Buffer[] = [];
  const input = <Format extends 'float32x2' | 'float32x4' | 'uint32x3' | 'uint32'>(
    name: string,
    format: Format,
    values: Float32Array | Uint32Array,
    length: number
  ) => {
    const buffer = createInputBuffer(device, values);
    buffers.push(buffer);
    return importGraphBuffer(graph, name, buffer, format, length);
  };
  const output = <Format extends 'float32x4' | 'uint32x2' | 'uint32'>(
    name: string,
    format: Format,
    length: number
  ) => {
    const components = format === 'float32x4' ? 4 : format === 'uint32x2' ? 2 : 1;
    const buffer = createOutputBuffer(device, components * length);
    buffers.push(buffer);
    return {buffer, view: importGraphBuffer(graph, name, buffer, format, length)};
  };
  const segments = output('segments', 'float32x4', 32);
  const siteIds = output('site-ids', 'uint32x2', 32);
  const count = output('count', 'uint32', 1);
  const requiredCount = output('required-count', 'uint32', 1);
  const overflow = output('overflow', 'uint32', 1);
  const invalidCount = output('invalid-count', 'uint32', 1);
  const triangleWords = [...FAN, ...trailingRows.flat()];
  graph.add(
    new GPUVoronoiDiagram({
      positions: input(
        'positions',
        'float32x2',
        Float32Array.from([0, 0, 1, 0, 1, 1, 0, 1, 0.5, 0.5]),
        5
      ),
      triangles: input(
        'triangles',
        'uint32x3',
        Uint32Array.from(triangleWords),
        triangleWords.length / 3
      ),
      triangleCount: input('triangle-count', 'uint32', Uint32Array.of(4), 1),
      clipBounds: input('clip-bounds', 'float32x4', Float32Array.from([-1, -1, 2, 2]), 1),
      output: {
        segments: segments.view,
        siteIds: siteIds.view,
        status: {
          count: count.view,
          requiredCount: requiredCount.view,
          overflow: overflow.view,
          invalidCount: invalidCount.view
        }
      }
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const written = (await readUint32(count.buffer, 1))[0];
  const result = {
    written,
    required: await readUint32(requiredCount.buffer, 1),
    overflow: await readUint32(overflow.buffer, 1),
    invalid: await readUint32(invalidCount.buffer, 1),
    segments: await readFloat32(segments.buffer, 4 * written),
    siteIds: await readUint32(siteIds.buffer, 2 * written)
  };
  compiled.destroy();
  buffers.forEach(buffer => buffer.destroy());
  return result;
}

it('GPUVoronoiDiagram reads only the triangles selected by triangleCount', async () => {
  const reference = await runVoronoi([]);
  expect(reference.written).toBeGreaterThan(0);
  expect(reference.invalid).toEqual([0]);
  for (let rows = 1; rows <= STALE_TRIANGLES.length; rows++) {
    expect(await runVoronoi(STALE_TRIANGLES.slice(0, rows)), `${rows} stale rows`).toEqual(
      reference
    );
  }
});

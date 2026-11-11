// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Coverage of this file, case by case. Rows of `groupIds` past the last position are rejected at
// construction (the class accepted them until 2026-11-11 and then threw from the Delaunay stage).
// Each hull case runs once on zero-filled outputs and once on outputs pre-filled with stale rows
// (valid-looking triangles, set validity words, 0xffffffff) and expects the same valid triangles,
// valid boundary edges and convergence words.
//  - GPUConcaveHull: the caller's `triangles` view is Delaunay's output, so rows past the written
//    count reach the retain, adjacency and boundary kernels as stale input. The case proves those
//    kernels gate on the count. This is the meaningful padding case.
//  - GPUGroupConcaveHull: Delaunay, retain, adjacency and boundary run on transient packed views
//    that the caller cannot pre-fill, and the kernels that fill the caller's outputs clear them
//    first. The case therefore proves only that the caller's output rows are cleared (a stale
//    caller row cannot survive). It does NOT prove that retain or boundary ignore padding rows of
//    the packed triangles: those kernels are not exported, so no spec can hand them a poisoned view.

import type {Buffer} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {
  GPUConcaveHull,
  GPUGroupConcaveHull
} from '../../../src/gpu-spatial-analysis/concave-hull/gpu-concave-hull';
import {getDelaunayMaximumTriangleCount} from '../../../src/gpu-spatial-analysis/delaunay-tessellation/gpu-delaunay-tessellation';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

const POSITIONS = [0, 0, 1, 0, 0, 1, 10, 0, 11, 0, 10, 1, 0.5, 0.4, 10.5, 0.4];
const POINT_COUNT = POSITIONS.length / 2;
const GROUP_IDS = [0, 0, 0, 1, 1, 1, 0, 1];
const STALE_WORDS = [0xffffffff, 1, 0, 3, 1, 2, 7];

function stale(length: number, enabled: boolean): Uint32Array {
  return Uint32Array.from({length}, (_, index) => (enabled ? STALE_WORDS[index % 7] : 0));
}

async function runHull(grouped: boolean, poisoned: boolean) {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device, {id: 'concave-hull-padding-rows'});
  const buffers: Buffer[] = [];
  let serial = 0;
  const groupCount = grouped ? 3 : 1;
  const capacity = getDelaunayMaximumTriangleCount(POINT_COUNT) * groupCount;
  const output = <Format extends 'uint32' | 'uint32x2' | 'uint32x3'>(
    length: number,
    format: Format
  ) => {
    const components = format.endsWith('x2') ? 2 : format.endsWith('x3') ? 3 : 1;
    const buffer = createOutputBuffer(device, length * components);
    buffer.write(stale(length * components, poisoned));
    buffers.push(buffer);
    return {buffer, view: importGraphBuffer(graph, `output-${serial++}`, buffer, format, length)};
  };
  const input = <Format extends 'uint32' | 'float32x2'>(values: number[], format: Format) => {
    const buffer = createInputBuffer(
      device,
      format === 'uint32' ? Uint32Array.from(values) : Float32Array.from(values)
    );
    buffers.push(buffer);
    return importGraphBuffer(
      graph,
      `input-${serial++}`,
      buffer,
      format,
      values.length / (format === 'float32x2' ? 2 : 1)
    );
  };
  const triangles = output(capacity, 'uint32x3');
  const triangleValidity = output(capacity, 'uint32');
  const boundaryEdges = output(capacity * 3, 'uint32x2');
  const edgeValidity = output(capacity * 3, 'uint32');
  const converged = output(groupCount, 'uint32');
  const outputViews = {
    triangles: triangles.view,
    triangleValidity: triangleValidity.view,
    boundaryEdges: boundaryEdges.view,
    edgeValidity: edgeValidity.view,
    status: {converged: converged.view}
  };
  const positions = input(POSITIONS, 'float32x2');
  graph.add(
    grouped
      ? new GPUGroupConcaveHull({
          positions,
          groupIds: input(GROUP_IDS, 'uint32'),
          groupCount,
          maximumCircumradius: 1,
          output: outputViews
        })
      : new GPUConcaveHull({positions, maximumCircumradius: 1, output: outputViews})
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const triangleWords = await readUint32(triangles.buffer, capacity * 3);
  const triangleValid = await readUint32(triangleValidity.buffer, capacity);
  const edgeWords = await readUint32(boundaryEdges.buffer, capacity * 6);
  const edgeValid = await readUint32(edgeValidity.buffer, capacity * 3);
  const result = {
    converged: await readUint32(converged.buffer, groupCount),
    triangles: triangleValid.flatMap((valid, row) =>
      valid ? [`${row}:${triangleWords.slice(row * 3, row * 3 + 3).join(',')}`] : []
    ),
    edges: edgeValid.flatMap((valid, row) =>
      valid ? [`${row}:${edgeWords.slice(row * 2, row * 2 + 2).join(',')}`] : []
    ),
    validWords: [...new Set(triangleValid), ...new Set(edgeValid)].sort()
  };
  compiled.destroy();
  buffers.forEach(buffer => buffer.destroy());
  return result;
}

for (const grouped of [false, true]) {
  it(`${grouped ? 'GPUGroupConcaveHull clears stale caller output rows' : 'GPUConcaveHull ignores stale rows outside the written triangles'}`, async () => {
    const reference = await runHull(grouped, false);
    expect(reference.triangles.length).toBeGreaterThan(0);
    expect(reference.edges.length).toBeGreaterThan(0);
    expect(await runHull(grouped, true)).toEqual(reference);
  });
}

it('GPUGroupConcaveHull rejects groupIds with rows beyond the positions', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device, {id: 'concave-hull-group-ids'});
  const buffer = createInputBuffer(device, Uint32Array.from({length: POINT_COUNT + 2}));
  const view = <Format extends 'uint32' | 'float32x2'>(format: Format, length: number) =>
    importGraphBuffer(graph, `view-${format}-${length}`, buffer, format, length);
  expect(
    () =>
      new GPUGroupConcaveHull({
        positions: view('float32x2', 4) as never,
        groupIds: view('uint32', 6),
        groupCount: 2,
        maximumCircumradius: 1,
        output: {} as never
      })
  ).toThrow(/^group-concave-hull groupIds must have exactly 4 rows/);
  buffer.destroy();
});

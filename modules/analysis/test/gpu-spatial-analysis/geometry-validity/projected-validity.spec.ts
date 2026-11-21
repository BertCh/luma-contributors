// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  GPU_GEOMETRY_VALIDITY_BIT,
  GPUGeometryValidity
} from '../../../src/gpu-spatial-analysis/geometry-validity/index';
import {expectProjectedUsability} from '../geometry-refinement/geometry-output-review-test-utils';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {createOutputBuffer, readUint32, submitGraph} from '../../utils/gpu-contributor-test-utils';
import {createSegmentGeometry} from '../segment-intersection/segment-geometry-harness';

it.each([
  [1000000, 5000000],
  [10000000, 10000000]
])('projected validity distinguishes a square and bowtie at (%s,%s)', async (originX, originY) => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  const buffers: Buffer[] = [];
  // Integer corners and translations are exact in f32; a translation cannot create a crossing.
  const square: [number, number][] = [
    [0, 0],
    [8, 0],
    [8, 8],
    [0, 8]
  ];
  const bowtie: [number, number][] = [
    [0, 0],
    [8, 8],
    [0, 8],
    [8, 0]
  ];
  const geometry = createSegmentGeometry(
    device,
    graph,
    'projected',
    [square, bowtie].map(ring => ({
      kind: 'polygons',
      polygons: [[ring.map(([x, y]) => [x + originX, y + originY] as [number, number])]]
    })),
    buffers
  );
  if (geometry.kind !== 'polygons') throw new Error('expected polygon geometry');
  const output = (name: string, length: number) => {
    const buffer = createOutputBuffer(device, length);
    buffers.push(buffer);
    return {buffer, view: importGraphBuffer(graph, name, buffer, 'uint32', length)};
  };
  const issue = output('issue', 2),
    validity = output('validity', 2),
    overflow = output('overflow', 1);
  graph.add(
    new GPUGeometryValidity({
      polygons: geometry,
      orientation: 'ignore',
      intersectionCapacity: 32,
      output: {issueCode: issue.view, validity: validity.view, status: {overflow: overflow.view}}
    })
  );
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    expect(await readUint32(overflow.buffer, 1)).toEqual([0]);
    expectProjectedUsability(await readUint32(validity.buffer, 2));
    expect(await readUint32(issue.buffer, 2)).toEqual([
      0,
      GPU_GEOMETRY_VALIDITY_BIT.selfIntersection
    ]);
  } finally {
    compiled.destroy();
    buffers.forEach(buffer => buffer.destroy());
  }
});

// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {type GraphDataView} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPUSphericalPolygonContains} from '../../../src/gpu-spatial-analysis/global-coordinates/gpu-spherical-polygon-contains';
import {createVectorView} from '../../utils/gpu-contributor-test-utils';
import {createPaddedQueryFixture} from '../linear-referencing/gpu-linear-referencing-chunks.test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

async function runContains(
  device: Device,
  chunked: boolean,
  boundary: 'closed' | 'open',
  complement: boolean,
  chunkLengths = [1, 3, 2],
  mixedRings = false
) {
  const fixture = createPaddedQueryFixture(device);
  const {graph, input, output} = fixture;
  const inside = 1 / Math.sqrt(3);
  const edge = 1 / Math.sqrt(2);
  const unit = (point: number[]) => {
    const norm = Math.hypot(...point);
    return point.map(value => value / norm);
  };
  const shell = [1, 0, 0, 0, 1, 0, 0, 0, 1];
  // Polygon 0 is an octant shell with a small central triangular hole. Polygon 1 has a
  // repeated edge endpoint and is invalid. All points and answers are geometrically explicit.
  const geometry = mixedRings
    ? [
        ...shell,
        ...unit([3, 1, 1]),
        ...unit([1, 3, 1]),
        ...unit([1, 1, 3]),
        1,
        0,
        0,
        1,
        0,
        0,
        0,
        1,
        0
      ]
    : shell;
  const ringOffsets = mixedRings ? [0, 3, 6, 9] : [0, 3];
  const polygonOffsets = input(
    'polygon-offsets',
    'uint32',
    new Uint32Array(mixedRings ? [0, 2, 3] : [0, 1])
  );
  const points = mixedRings
    ? [
        [inside, inside, inside],
        unit([10, 1, 1]),
        [edge, edge, 0],
        unit([3, 1, 1]),
        [inside, inside, inside],
        [-1, 0, 0]
      ]
    : [
        [inside, inside, inside],
        [-1, 0, 0],
        [edge, edge, 0],
        [Number.NaN, 0, 0],
        [inside, inside, inside],
        [inside, inside, inside]
      ];
  const selectors = mixedRings ? [0, 0, 0, 0, 1, 0] : [0, 0, 0, 0, 1, 0];
  const positions = input('positions', 'float32x3', new Float32Array(geometry));
  const offsets = input('offsets', 'uint32', new Uint32Array(ringOffsets));
  const total = chunkLengths.reduce((sum, length) => sum + length, 0);
  const lengths = chunked ? chunkLengths : [total];
  const queries: GraphDataView<'float32x3'>[] = [];
  const polygonIndices: GraphDataView<'uint32'>[] = [];
  const contains: GraphDataView<'uint32'>[] = [];
  const statuses: GraphDataView<'uint32'>[] = [];
  let rowBase = 0;
  for (const [chunkIndex, length] of lengths.entries()) {
    queries.push(
      input(
        `queries-${chunkIndex}`,
        'float32x3',
        new Float32Array(Array.from({length}, (_, row) => points[(rowBase + row) % 6]).flat())
      )
    );
    polygonIndices.push(
      input(
        `selectors-${chunkIndex}`,
        'uint32',
        new Uint32Array(Array.from({length}, (_, row) => selectors[(rowBase + row) % 6]))
      )
    );
    contains.push(output(`contains-${chunkIndex}`, 'uint32', length));
    statuses.push(output(`statuses-${chunkIndex}`, 'uint32', length));
    rowBase += length;
  }
  graph.add(
    new GPUSphericalPolygonContains({
      positions,
      ringOffsets: offsets,
      polygonOffsets,
      boundary,
      complement,
      queries: chunked ? createVectorView('queries', 'float32x3', queries) : queries[0],
      polygonIndices: chunked
        ? createVectorView('selectors', 'uint32', polygonIndices)
        : polygonIndices[0],
      output: {
        contains: chunked ? createVectorView('contains', 'uint32', contains) : contains[0],
        status: chunked ? createVectorView('statuses', 'uint32', statuses) : statuses[0]
      }
    })
  );
  const compiled = graph.compile();
  try {
    return await fixture.submitTwice(compiled, async () => ({
      contains: await fixture.readRows(contains),
      status: await fixture.readRows(statuses)
    }));
  } finally {
    compiled.destroy();
    fixture.destroy();
  }
}

for (const boundary of ['closed', 'open'] as const)
  for (const complement of [false, true]) {
    it(`shares ring validation across uneven chunks and matches the octant oracle (${boundary}, complement=${complement})`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const packed = await runContains(device, false, boundary, complement);
      const chunked = await runContains(device, true, boundary, complement);
      expect(chunked).toEqual(packed);
      expect(chunked.status).toEqual([0, 0, 2, 1, 1, 0]);
      expect(chunked.contains).toEqual([
        Number(!complement),
        Number(complement),
        Number(boundary === 'closed'),
        0,
        0,
        Number(!complement)
      ]);
    });
  }

it('preserves offsets and padding across 255/256/257-row spherical dispatches', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const lengths = [255, 256, 257];
  const packed = await runContains(device, false, 'closed', false, lengths);
  const chunked = await runContains(device, true, 'closed', false, lengths);
  expect(chunked).toEqual(packed);
  for (let row = 0; row < 768; row++) {
    expect(chunked.contains[row]).toBe([1, 0, 1, 0, 0, 1][row % 6]);
    expect(chunked.status[row]).toBe([0, 0, 2, 1, 1, 0][row % 6]);
  }
});

for (const boundary of ['closed', 'open'] as const)
  it(`reuses polygon scratch in 513/1/257-row chunks with holes, boundaries and invalid rings (${boundary})`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const lengths = [513, 1, 257];
    const packed = await runContains(device, false, boundary, false, lengths, true);
    const chunked = await runContains(device, true, boundary, false, lengths, true);
    expect(chunked).toEqual(packed);
    // Center is in the hole, near the x vertex is in the shell, next two points are
    // shell/hole boundaries, selector 1 uses the invalid ring, and -x is outside.
    for (let row = 0; row < 771; row++) {
      expect(chunked.contains[row]).toBe(
        [0, 1, Number(boundary === 'closed'), Number(boundary === 'closed'), 0, 0][row % 6]
      );
      expect(chunked.status[row]).toBe([0, 0, 2, 2, 1, 0][row % 6]);
    }
  });

it('registers and executes query/scatter stages with interleaved empty chunks', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const lengths = [0, 1, 0, 3, 0, 2, 0];
  const packed = await runContains(device, false, 'closed', false, lengths);
  const chunked = await runContains(device, true, 'closed', false, lengths);
  expect(chunked).toEqual(packed);
  expect(chunked.contains).toEqual([1, 0, 1, 0, 0, 1]);
  expect(chunked.status).toEqual([0, 0, 2, 1, 1, 0]);
});

// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import * as arrow from 'apache-arrow';
import {writeWKB} from '@math.gl/arrow/wkb';
import {
  makeKeplerGeometryGPUVectors,
  prepareKeplerGeometrySourceFromArrow
} from '../../../examples/deck/kepler-compute/src/geometry-preparation';
import type {GPUVector} from '@luma.gl/graph/gpu-data';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {requireTestDevice} from '../../analysis/test/utils/require-test-device';
import {GPUKeplerSelectionScopes} from '../src/selection/gpu-selection-scopes';

async function readWords(vector: GPUVector<'uint32'>): Promise<number[]> {
  const values: number[] = [];
  for (const chunk of vector.data) {
    if (!chunk.length) continue;
    const bytes = await chunk.buffer.readAsync(chunk.byteOffset, chunk.length * 4);
    values.push(...new Uint32Array(bytes.buffer, bytes.byteOffset, chunk.length));
  }
  return values;
}

it('raw WKB resident selection preserves chunks and nulls while applying its distinct vertex mean and native bounds policy', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const polygon = writeWKB({
    type: 'Polygon',
    coordinates: [
      [
        [0, 0],
        [10, 0],
        [0, 10],
        [0, 0]
      ]
    ]
  });
  const point = writeWKB({type: 'Point', coordinates: [2.5, 2.5]});
  const first = arrow.vectorFromArray([null, polygon], new arrow.Binary());
  const empty = arrow.vectorFromArray([], new arrow.Binary());
  const last = arrow.vectorFromArray([point, new Uint8Array([1, 2])], new arrow.Binary());
  const source = prepareKeplerGeometrySourceFromArrow(
    new arrow.Vector([...first.data, ...empty.data, ...last.data]),
    {encoding: 'geoarrow.wkb'}
  );
  const geometry = makeKeplerGeometryGPUVectors(device, source);
  const representative = geometry.representative!;
  expect(representative.recipe).toBe('wkb-vertex-mean');
  expect(representative.positions.data.map(chunk => chunk.length)).toEqual([2, 0, 2]);
  expect(await readWords(representative.validity)).toEqual([0, 1, 1, 0]);
  expect(geometry.points.resolveSourceRow(0)).toBe(2);
  const scopes = new GPUKeplerSelectionScopes({
    device,
    rows: representative.positions,
    datasetRevision: 1,
    layers: [{id: 'geometry', geometryRevision: 1, geometry: representative}],
    scalarFilters: [],
    polygonFilters: [
      {id: 'drawn', layerIds: ['geometry'], selection: {kind: 'rectangle', bounds: [2, 2, 3, 3]}}
    ]
  });
  const execute = async () => {
    const encoder = device.createCommandEncoder();
    scopes.encode(encoder);
    device.submit(encoder.finish());
    return readWords(scopes.getLayerMask('geometry'));
  };
  try {
    // WKB mean is [2.5,2.5], Arrow mean is [10/3,10/3], JSON bbox center is [5,5].
    expect(await execute()).toEqual([0, 1, 1, 0]);
    scopes.setFilters(
      [],
      [
        {
          id: 'drawn',
          layerIds: ['geometry'],
          selection: {kind: 'rectangle', bounds: [2.5, 2.5, 3, 3]},
          rectangleBounds: [2.5, 2.5, 3, 3]
        }
      ]
    );
    expect(await execute()).toEqual([0, 1, 1, 0]);
    scopes.setFilters(
      [],
      [
        {
          id: 'drawn',
          layerIds: ['geometry'],
          selection: {kind: 'rectangle', bounds: [2.5, 2.5, 3, 3]}
        }
      ]
    );
    expect(await execute()).toEqual([0, 0, 0, 0]);
  } finally {
    const fence = device.createFence();
    try {
      await scopes.destroy(fence.signaled);
    } finally {
      fence.destroy();
    }
    geometry.destroy();
  }
});

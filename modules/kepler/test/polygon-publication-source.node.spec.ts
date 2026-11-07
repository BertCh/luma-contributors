// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test, vi} from 'vitest';
import {prepareGeometrySource, MAXIMUM_POLYGON_CONTIGUITY_VERTICES} from '@luma.gl/engine/arrow';
import {getPolygonAnalysisSource} from '../../../examples/deck/kepler-compute/src/polygon-publication-source';

function fixture() {
  return prepareGeometrySource([
    {id: 'empty', rows: []},
    {
      id: 'original',
      rows: [
        null,
        {
          type: 'Polygon',
          coordinates: [
            [
              [1.000000000001, 0],
              [2, 0],
              [2, 1],
              [1.000000000001, 0]
            ]
          ]
        }
      ]
    }
  ]);
}

test('polygon analysis borrows existing Float64 Arrow polygons and accounts retained storage without decoding again', () => {
  const geometry = fixture();
  const reads = geometry.batches.map(batch => vi.spyOn(batch.polygons, 'get'));
  const descriptor = getPolygonAnalysisSource(geometry, 'geometry');
  expect(descriptor.reason).toBeUndefined();
  expect(descriptor.byteLength).toBeGreaterThan(64 * 2);
  expect(descriptor.source?.source.metadata).toBe(geometry.metadata);
  expect(
    descriptor.source?.source.batches.map(batch => [
      batch.id,
      batch.rowCount,
      batch.sourceRowOffset
    ])
  ).toEqual([
    ['empty', 0, 0],
    ['original', 2, 0]
  ]);
  descriptor.source?.source.batches.forEach((batch, index) =>
    expect(batch.polygons).toBe(geometry.batches[index].polygons)
  );
  for (const read of reads) {
    expect(read).not.toHaveBeenCalled();
    read.mockRestore();
  }
  const coordinates = geometry.batches[1].polygons.get(1)!.get(0)!.get(0)!.get(0)!;
  expect(coordinates.get(0)!).toBe(1.000000000001);
  expect(coordinates.get(0)!).not.toBe(Math.fround(coordinates.get(0)!));
});

test('polygon analysis limits do not reject or mutate otherwise renderable geometry', () => {
  const geometry = fixture();
  const manyRows = {...geometry, metadata: {...geometry.metadata, rowCount: 4097}};
  expect(getPolygonAnalysisSource(manyRows, 'geometry')).toMatchObject({
    byteLength: 0,
    reason: expect.stringContaining('4096')
  });
  const manyVertices = {
    ...geometry,
    batches: geometry.batches.map((batch, index) =>
      index === 1
        ? {
            ...batch,
            polygonOutlines: {
              ...batch.polygonOutlines,
              positions: new Float32Array((MAXIMUM_POLYGON_CONTIGUITY_VERTICES + 1) * 3)
            }
          }
        : batch
    )
  };
  expect(getPolygonAnalysisSource(manyVertices, 'geometry')).toMatchObject({
    byteLength: 0,
    reason: expect.stringContaining('65536')
  });
  expect(manyVertices.batches[1].polygons).toBe(geometry.batches[1].polygons);
  expect(geometry.metadata.rowCount).toBe(2);
  expect(
    getPolygonAnalysisSource(
      prepareGeometrySource([{id: 'points', rows: [{type: 'Point', coordinates: [0, 0]}]}]),
      'geometry'
    ).reason
  ).toContain('polygon geometry');
});

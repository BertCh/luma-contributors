// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import * as arrow from 'apache-arrow';
import {expect, test} from 'vitest';
import {prepareTabularPointPositions} from '../../../examples/deck/kepler-compute/src/point-position-source';
import {makeResidentSpatialJoinSourcePositions} from '../../../examples/deck/kepler-compute/src/resident-spatial-join-reproject';

// Regression: UTM points joined to EPSG:4326 polygons matched 0 rows because ingestion hides
// out-of-range longitude/latitude as NaN, so the GPU reprojection received NaN positions.
test('projected left coordinates reach the join as raw values, not NaN', () => {
  const table = new arrow.Table({
    longitude: arrow.vectorFromArray([583960.5, 585000, null], new arrow.Float32()),
    latitude: arrow.vectorFromArray([4507523.5, 4510000, 4511000], new arrow.Float32())
  });
  const prepared = {
    table,
    dictionaries: {},
    sourceBatches: [{sourceRowCount: 3, sourceRowIndexOffset: 0}]
  } as never;
  const roles = {longitude: 'longitude', latitude: 'latitude'};
  const ingested = prepareTabularPointPositions(prepared, roles);
  expect(ingested.invalidRowCount).toBe(3);
  expect(Number.isNaN(ingested.batches[0].positions[0])).toBe(true);

  const uploads: Float32Array[] = [];
  const device = {
    lost: new Promise<never>(() => undefined),
    createBuffer: (props: {data?: Float32Array; byteLength?: number}) => {
      if (props.data) uploads.push(props.data);
      return {byteLength: props.data?.byteLength ?? props.byteLength ?? 0, destroy() {}};
    }
  };
  makeResidentSpatialJoinSourcePositions(device as never, prepared, roles);
  expect(Array.from(uploads[0])).toEqual([583960.5, 4507523.5, 585000, 4510000, NaN, 4511000]);
});

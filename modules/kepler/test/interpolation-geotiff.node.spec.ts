// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import type {KeplerAnalysisRecipe} from '../src/recipes/analysis-recipe';
import {encodeInterpolationGeoTIFF} from '../../../examples/deck/kepler-compute/src/interpolation-geotiff';

const bounds = [-74, 40, -70, 43] as const;
const recipe: KeplerAnalysisRecipe = {
  schemaVersion: 1,
  algorithmVersion: 1,
  id: 'interpolation-\u2603',
  source: {datasetId: 'samples-\ud83c\udf0d', revision: 3},
  population: 'selected',
  selectionRevision: 7,
  backendPolicy: 'require-gpu',
  operation: {
    kind: 'idw-points',
    column: 'value',
    longitudeColumn: 'longitude',
    latitudeColumn: 'latitude',
    coordinateSpace: 'longitude-latitude',
    metric: 'great-circle',
    units: 'meters',
    grid: {width: 2, height: 3, bounds},
    searchRadius: 1000,
    power: 2,
    neighborCount: 4,
    minimumNeighborCount: 2,
    taper: 'none'
  }
};

test('interpolation GeoTIFF preserves numeric cells, geographic edges, nodata and provenance', async () => {
  const values = new Float32Array([10.5, -20, NaN, Infinity, -Infinity, 60]);
  const bytes = encodeInterpolationGeoTIFF({width: 2, height: 3, bounds, values, recipe});
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  expect(view.getUint16(0, true)).toBe(0x4949);
  expect(view.getUint16(2, true)).toBe(42);
  const directoryOffset = view.getUint32(4, true);
  const entryCount = view.getUint16(directoryOffset, true);
  const entries = new Map<number, {type: number; count: number; offset: number}>();
  const tags: number[] = [];
  for (let index = 0; index < entryCount; index++) {
    const entryOffset = directoryOffset + 2 + index * 12;
    const tag = view.getUint16(entryOffset, true);
    const type = view.getUint16(entryOffset + 2, true);
    const count = view.getUint32(entryOffset + 4, true);
    const componentBytes = type === 2 ? 1 : type === 3 ? 2 : type === 4 ? 4 : 8;
    const offset =
      count * componentBytes > 4 ? view.getUint32(entryOffset + 8, true) : entryOffset + 8;
    entries.set(tag, {type, count, offset});
    tags.push(tag);
    expect(offset + count * componentBytes).toBeLessThanOrEqual(bytes.length);
  }
  expect(tags).toEqual([...tags].sort((first, second) => first - second));
  expect(view.getUint32(directoryOffset + 2 + entryCount * 12, true)).toBe(0);
  const short = (tag: number) => view.getUint16(entries.get(tag)!.offset, true);
  const long = (tag: number) => view.getUint32(entries.get(tag)!.offset, true);
  expect([
    long(256),
    long(257),
    short(258),
    short(259),
    short(262),
    short(274),
    short(277),
    short(284),
    short(339)
  ]).toEqual([2, 3, 32, 1, 1, 1, 1, 1, 3]);
  const scales = entries.get(33550)!;
  expect(scales.type).toBe(12);
  expect([0, 1, 2].map(index => view.getFloat64(scales.offset + index * 8, true))).toEqual([
    2, 1, 0
  ]);
  const tiepoint = entries.get(33922)!;
  expect(
    [0, 1, 2, 3, 4, 5].map(index => view.getFloat64(tiepoint.offset + index * 8, true))
  ).toEqual([0, 0, 0, -74, 43, 0]);
  const keys = entries.get(34735)!;
  expect(
    Array.from({length: keys.count}, (_, index) => view.getUint16(keys.offset + index * 2, true))
  ).toEqual([1, 1, 1, 3, 1024, 0, 1, 2, 1025, 0, 1, 1, 2048, 0, 1, 4326]);
  const description = entries.get(270)!;
  const descriptionBytes = bytes.subarray(
    description.offset,
    description.offset + description.count
  );
  expect([...descriptionBytes].every(value => value < 128)).toBe(true);
  expect(descriptionBytes.at(-1)).toBe(0);
  const provenance = JSON.parse(new TextDecoder().decode(descriptionBytes.subarray(0, -1)));
  expect(provenance.recipe).toEqual(recipe);
  expect(provenance.metric).toBe('prediction');
  expect(provenance.band).toEqual({
    index: 1,
    name: 'IDW prediction',
    sourceColumn: 'value',
    valueUnitsPower: 1
  });
  const nodata = entries.get(42113)!;
  expect(
    new TextDecoder().decode(bytes.subarray(nodata.offset, nodata.offset + nodata.count))
  ).toBe('nan\0');
  const stripOffset = long(273);
  expect(long(278)).toBe(3);
  expect(long(279)).toBe(24);
  expect(
    Array.from({length: 6}, (_, index) => view.getFloat32(stripOffset + index * 4, true))
  ).toEqual([10.5, -20, NaN, NaN, NaN, 60]);
  for (const index of [2, 3, 4])
    expect(view.getUint32(stripOffset + index * 4, true)).toBe(0x7fc00000);
  expect(values[3]).toBe(Infinity);

  // Independently decode the TIFF directory and raw strip with Three's already installed UTIF.
  const decoderPath = 'three/examples/jsm/libs/utif.module.js';
  const {default: decoder} = await import(/* @vite-ignore */ decoderPath);
  const decoded = decoder.decode(bytes.buffer)[0];
  decoder.decodeImage(bytes.buffer, decoded);
  expect([decoded.width, decoded.height]).toEqual([2, 3]);
  expect(decoded.t33550).toEqual([2, 1, 0]);
  expect(decoded.t33922).toEqual([0, 0, 0, -74, 43, 0]);
  expect(decoded.data).toEqual(bytes.subarray(stripOffset));

  for (const change of [
    {width: 257},
    {height: 0},
    {values: values.subarray(1)},
    {bounds: [0, 0, 0, 1] as const},
    {width: 3, height: 2}
  ])
    expect(() =>
      encodeInterpolationGeoTIFF({width: 2, height: 3, bounds, values, recipe, ...change})
    ).toThrow();
});

test('kriging prediction and variance export distinct single bands with squared-value provenance', async () => {
  const krigingRecipe: KeplerAnalysisRecipe = {
    ...recipe,
    operation: {
      kind: 'kriging-points',
      column: 'response-\u00b5g',
      longitudeColumn: 'longitude',
      latitudeColumn: 'latitude',
      coordinateSpace: 'longitude-latitude',
      metric: 'great-circle',
      units: 'meters',
      grid: {width: 2, height: 3, bounds},
      searchRadiusMeters: 1000,
      neighborCount: 4,
      minimumNeighbors: 2,
      variogram: {model: 'spherical', nugget: 0.1, sill: 1, rangeMeters: 800}
    }
  };
  // Decode both artifacts with the installed TIFF reader, not the encoder's directory helpers.
  const decoderPath = 'three/examples/jsm/libs/utif.module.js';
  const {default: decoder} = await import(/* @vite-ignore */ decoderPath);
  for (const metric of ['prediction', 'variance'] as const) {
    const values = new Float32Array(
      metric === 'prediction' ? [3, -2, NaN, 1, 5, 8] : [0, 2, NaN, 0.5, 4, 3]
    );
    const bytes = encodeInterpolationGeoTIFF({
      width: 2,
      height: 3,
      bounds,
      values,
      recipe: krigingRecipe,
      metric
    });
    const decoded = decoder.decode(bytes.buffer)[0];
    decoder.decodeImage(bytes.buffer, decoded);
    expect([decoded.width, decoded.height]).toEqual([2, 3]);
    expect(decoded.t277).toEqual([1]);
    expect(decoded.t33550).toEqual([2, 1, 0]);
    expect(decoded.t33922).toEqual([0, 0, 0, -74, 43, 0]);
    expect(decoded.t42113[0]).toBe('nan');
    const provenance = JSON.parse(decoded.t270[0]);
    expect(provenance.recipe).toEqual(krigingRecipe);
    expect(provenance.metric).toBe(metric);
    expect(provenance.band).toEqual({
      index: 1,
      name: `Ordinary kriging ${metric}`,
      sourceColumn: 'response-\u00b5g',
      valueUnitsPower: metric === 'variance' ? 2 : 1
    });
    expect(decoded.t42112[0]).toContain(`role="description">Ordinary kriging ${metric}</Item>`);
    expect(decoded.t42112[0]).toContain(
      `name="VALUE_UNITS_POWER" sample="0">${metric === 'variance' ? 2 : 1}</Item>`
    );
    const raster = new DataView(
      decoded.data.buffer,
      decoded.data.byteOffset,
      decoded.data.byteLength
    );
    expect(Array.from({length: 6}, (_, index) => raster.getFloat32(index * 4, true))).toEqual([
      ...values
    ]);
  }
  expect(() =>
    encodeInterpolationGeoTIFF({
      width: 2,
      height: 3,
      bounds,
      values: new Float32Array(6),
      recipe: krigingRecipe
    })
  ).toThrow('Choose a prediction or variance band');
  expect(() =>
    encodeInterpolationGeoTIFF({
      width: 2,
      height: 3,
      bounds,
      values: new Float32Array(6),
      recipe,
      metric: 'variance'
    })
  ).toThrow('IDW exports prediction only');
});

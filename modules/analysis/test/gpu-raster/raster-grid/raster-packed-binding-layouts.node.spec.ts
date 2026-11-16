// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {expect, it} from 'vitest';
import {GPURasterReconstruction} from '../../../src/raster-analysis/gpu-raster-reconstruction';
import {GPURasterStretch} from '../../../src/raster-analysis/raster-stretch/gpu-raster-stretch';
import {validateRasterFractionalWritableViews} from '../../../src/raster-analysis/fractional-coverage/raster-fractional-tile-kernel';

import {createRasterContributorFixture} from './raster-contributor-test-fixture';

it('rejects read-only and writable storage usage of one physical buffer even with disjoint aligned ranges', () => {
  const {graph, grid} = createRasterContributorFixture(1);
  const packed = graph.createTransientBuffer({
    id: 'mixed-storage-scope',
    byteLength: 1024,
    usage: Buffer.STORAGE
  });
  const values = graph.createDataView(packed, {format: 'float32', length: 1});
  const parameters = graph.createDataView(packed, {
    format: 'float32',
    byteOffset: 256,
    length: 16
  });
  const stretched = graph.createDataView(packed, {
    format: 'float32',
    byteOffset: 512,
    length: 1
  });
  expect(
    () =>
      new GPURasterStretch({
        grid,
        values,
        parameters,
        output: {stretched}
      })
  ).toThrow(/read-only and writable storage.*physical buffer.*one dispatch/);
  // The default (all co-bound) helper path must enforce the same physical usage-scope rule.
  expect(() => validateRasterFractionalWritableViews('scope', {stretched}, [values])).toThrow(
    /read-only and writable storage.*physical buffer.*one dispatch/
  );
  // An explicit read-after-write in a later dispatch is valid, as are disjoint writable bindings.
  expect(() =>
    validateRasterFractionalWritableViews(
      'scope',
      {stretched},
      [values],
      [{writes: [], reads: [values]}, {writes: [stretched]}]
    )
  ).not.toThrow();
  expect(() =>
    validateRasterFractionalWritableViews(
      'scope',
      {values, stretched},
      [],
      [{writes: [values, stretched]}]
    )
  ).not.toThrow();
});

it('accepts separate zonal reductions sharing an alignment prefix and ignores an unused stretch palette binding', async () => {
  const {GPURasterZonalStatistics} = await import(
    '../../../src/raster-analysis/raster-zonal-statistics/gpu-raster-zonal-statistics'
  );
  const {graph, grid, view} = createRasterContributorFixture(1);
  const buffer = graph.createTransientBuffer({
    id: 'separate-reductions',
    byteLength: 256,
    usage: Buffer.STORAGE
  });
  const counts = graph.createDataView(buffer, {format: 'uint32', byteOffset: 0, length: 1});
  const sums = graph.createDataView(buffer, {format: 'float32', byteOffset: 4, length: 1});
  expect(() =>
    graph.add(
      new GPURasterZonalStatistics({
        grid,
        zones: view('uint32'),
        values: {
          id: 'values',
          format: 'float32',
          storage: {kind: 'buffer', values: view('float32')}
        },
        zoneCapacity: 1,
        output: {cellCounts: counts, sums}
      })
    )
  ).not.toThrow();
  const paletteBuffer = graph.createTransientBuffer({
    id: 'unused-palette',
    byteLength: 4,
    usage: Buffer.COPY_DST
  });
  const palette = graph.createDataView(paletteBuffer, {format: 'uint32', length: 1});
  expect(() =>
    graph.add(
      new GPURasterStretch({
        grid,
        values: view('float32'),
        parameters: view('float32', 16),
        palette,
        output: {stretched: view('float32')}
      })
    )
  ).not.toThrow();
});
it('accepts packed publication fields and canonicalization inputs in separate dispatches', async () => {
  const {GPURasterContrastEqualization} = await import(
    '../../../src/raster-analysis/raster-contrast-equalization/gpu-raster-contrast-equalization'
  );
  const {GPURasterExtremaPyramid} = await import(
    '../../../src/raster-analysis/raster-pyramid/gpu-raster-extrema-pyramid'
  );
  const {graph, grid, view, band} = createRasterContributorFixture(1);
  const packed = graph.createTransientBuffer({
    id: 'packed-publication',
    byteLength: 768,
    usage: Buffer.STORAGE
  });
  const values = graph.createDataView(packed, {format: 'float32', length: 1});
  const validity = graph.createDataView(packed, {
    format: 'uint32',
    byteOffset: 256,
    length: 1
  });
  const status = graph.createDataView(packed, {format: 'uint32', byteOffset: 512, length: 1});
  const contrastInput = graph.createDataView(packed, {
    format: 'uint32',
    byteOffset: 12,
    length: 1
  });
  expect(() =>
    graph.add(
      new GPURasterContrastEqualization({
        grid,
        input: {...band, storage: {kind: 'buffer', values: contrastInput}},
        parameters: view('float32', 8),
        output: {values, validity, status: {approximation: status}}
      })
    )
  ).not.toThrow();
  expect(() =>
    graph.add(
      new GPURasterReconstruction({
        grid,
        marker: band,
        mask: band,
        output: {
          values,
          validity,
          status: {
            converged: graph.createDataView(packed, {
              format: 'uint32',
              byteOffset: 4,
              length: 1
            })
          }
        }
      })
    )
  ).not.toThrow();
  const inputValues = graph.createDataView(packed, {format: 'uint32', length: 1});
  const maximum = graph.createDataView(packed, {format: 'float32', byteOffset: 4, length: 1});
  expect(() =>
    graph.add(
      new GPURasterExtremaPyramid({
        grid,
        input: {...band, storage: {kind: 'buffer', values: inputValues}},
        output: {maximum}
      })
    )
  ).not.toThrow();
});

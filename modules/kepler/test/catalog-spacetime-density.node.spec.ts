// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {getGPUSpaceTimeKernelDensityParameterValues} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {getCatalogBinding, normalizeCatalogParameters} from '../src/analysis/catalog-bindings';
import {getSpacetimeDensityTimeSlices} from '../src/analysis/catalog-spacetime-density';
import {makeSpacetimeGrid, readSpacetimePoints} from '../src/analysis/catalog-spacetime-common';
import {SPACETIME_SUBSECOND_TEST_POINTS} from './catalog-spacetime-fixture';

for (const operationId of ['space-time-kernel-density', 'space-time-relative-risk']) {
  test(`${operationId} packs two slices 100 ms apart within the observed subsecond range`, () => {
    const binding = getCatalogBinding(operationId)!;
    const values = normalizeCatalogParameters(binding, {
      coordinateSystem: 'planar',
      gridSize: 1,
      sliceCount: 2
    });
    const points = readSpacetimePoints(SPACETIME_SUBSECOND_TEST_POINTS, values);
    const {grid} = makeSpacetimeGrid(points, values);
    const slices = getSpacetimeDensityTimeSlices(points, Number(values['sliceCount']));
    const parameters = getGPUSpaceTimeKernelDensityParameterValues(
      {
        bandwidth: Number(values['spatialBandwidth']),
        temporalBandwidth: Number(values['temporalBandwidth']),
        firstSliceTime: 0,
        sliceInterval: slices.sliceInterval
      },
      grid,
      slices.sliceCount
    );
    expect(parameters[11]).toBeCloseTo(0.1, 7);
    expect(slices.duration).toBe(0.1);
    expect(slices.times).toEqual([1700000000000, 1700000000100]);
    expect(slices.sliceCount).toBe(2);
  });
}

test('equal-time density observations collapse to one slice with a valid unused positive interval', () => {
  const collection = structuredClone(SPACETIME_SUBSECOND_TEST_POINTS);
  collection.features[1].properties!['time'] = 1700000000000;
  const points = readSpacetimePoints(collection, {
    coordinateSystem: 'planar',
    timeProperty: 'time',
    timeUnit: 'milliseconds'
  });
  const slices = getSpacetimeDensityTimeSlices(points, 4);
  expect(slices.sliceCount).toBe(1);
  expect(slices.times).toEqual([1700000000000]);
  expect(slices.sliceInterval).toBeGreaterThan(0);
});

test('subsecond slices also preserve epoch-second input and explicitly requested single slices', () => {
  const collection = structuredClone(SPACETIME_SUBSECOND_TEST_POINTS);
  for (const feature of collection.features)
    feature.properties!['time'] = Number(feature.properties!['time']) / 1000;
  const points = readSpacetimePoints(collection, {
    coordinateSystem: 'planar',
    timeProperty: 'time',
    timeUnit: 'seconds'
  });
  const slices = getSpacetimeDensityTimeSlices(points, 3);
  expect(slices.times[0]).toBe(1700000000);
  expect(slices.times[2]).toBe(1700000000.1);
  expect(getSpacetimeDensityTimeSlices(points, 1).times).toEqual([1700000000]);
});

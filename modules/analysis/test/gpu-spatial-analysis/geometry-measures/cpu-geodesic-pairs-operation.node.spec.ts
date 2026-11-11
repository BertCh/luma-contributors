// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import type {GPUSpatialContext} from '../../../src/analysis-core/contracts/spatial-context';
import {
  computeGeodesicPairsOnCPU,
  CPUGeodesicPairsOperation
} from '../../../src/gpu-spatial-analysis/geometry-measures/cpu-geodesic-pairs-operation';
import {
  GPUGeodesicPairsOperation,
  type GeodesicPairsOperationProps
} from '../../../src/gpu-spatial-analysis/geometry-measures/gpu-geodesic-pairs-operation';
import fixtures from './geodesic-pairs-fixture.json';

it('documents the CPU longitude convention for a coincident north-pole midpoint', () => {
  for (const metric of ['great-circle', 'ellipsoidal'] as const) {
    const result = computeGeodesicPairsOnCPU(
      {origins: new Float32Array([20, 90]), targets: new Float32Array([20, 90])},
      {coordinateSpace: 'longitude-latitude', metric}
    );
    // Both unit vectors are (0, 0, 1). JavaScript atan2(0, 0) chooses longitude 0;
    // the pole's latitude is 90 and distance is 0. WGSL leaves that atan2 unspecified,
    // so this is deliberately a CPU-only test, not a GPU longitude comparison.
    expect(Array.from(result.midpoints)).toEqual([0, 90]);
    expect(Array.from(result.distances)).toEqual([0]);
    expect(Array.from(result.converged)).toEqual([1]);
  }
});

it('matches PROJ geodesics and independent SciPy rhumb quadrature, including a near antipode', () => {
  for (const fixture of fixtures) {
    const spatialContext: GPUSpatialContext = {
      coordinateSpace: 'longitude-latitude',
      metric: fixture.model as GPUSpatialContext['metric'],
      units: 'meters'
    };
    const result = computeGeodesicPairsOnCPU(
      {origins: new Float32Array(fixture.origin), targets: new Float32Array(fixture.target)},
      spatialContext
    );
    expect(result.distances[0]).toBe(Math.fround(fixture.distance));
    expect(Math.abs(result.initialBearings[0] - fixture.initial)).toBeLessThan(1e-5);
    expect(Math.abs(result.finalBearings[0] - fixture.final)).toBeLessThan(1e-5);
    for (let component = 0; component < 2; component++)
      expect(Math.abs(result.midpoints[component] - fixture.midpoint[component])).toBeLessThan(
        1.6e-5
      );
    expect(result.converged[0]).toBe(1);
  }
});

it('preserves tiny seam differences, longitude identity and documented coincident conventions', () => {
  const origins = new Float32Array([-180, 0, 180, 0, 179.999, 0, 0, 0]);
  const targets = new Float32Array([180, 0, -180, 0, -179.999, 0, 0, 0]);
  for (const metric of ['great-circle', 'rhumb', 'ellipsoidal'] as const) {
    const result = computeGeodesicPairsOnCPU(
      {origins, targets},
      {
        coordinateSpace: 'longitude-latitude',
        metric,
        ...(metric === 'ellipsoidal' ? {} : {sphereRadius: 1})
      }
    );
    expect(result.distances[0]).toBe(0);
    expect(result.distances[1]).toBe(0);
    expect(result.distances[3]).toBe(0);
    const radius = metric === 'ellipsoidal' ? 6378137 : 1;
    const expected = (((360 - origins[4] + targets[4]) * Math.PI) / 180) * radius;
    expect(result.distances[2]).toBe(Math.fround(expected));
    expect(Array.from(result.initialBearings)).toEqual([0, 0, 90, 0]);
    expect(result.finalBearings[3]).toBe(metric === 'great-circle' ? 180 : 0);
    expect(Object.is(result.initialBearings[0], -0)).toBe(false);
  }
});

it('uses kilometer units and a caller ellipsoid', () => {
  const result = computeGeodesicPairsOnCPU(
    {origins: new Float32Array([0, 0]), targets: new Float32Array([1, 0])},
    {
      coordinateSpace: 'longitude-latitude',
      metric: 'ellipsoidal',
      units: 'kilometers',
      ellipsoid: {semiMajorAxis: 1000000, flattening: 0}
    }
  );
  expect(result.distances[0]).toBe(Math.fround((1000 * Math.PI) / 180));
  expect(result.initialBearings[0]).toBe(90);
  expect(result.finalBearings[0]).toBe(90);
  expect(Array.from(result.midpoints)).toEqual([0.5, 0]);
});

it('rejects malformed arrays and gives NaN/zero status for unspecified coordinates', () => {
  const context: GPUSpatialContext = {
    coordinateSpace: 'longitude-latitude',
    metric: 'great-circle'
  };
  expect(() =>
    computeGeodesicPairsOnCPU({origins: new Float32Array(1), targets: new Float32Array(1)}, context)
  ).toThrow(/origins/);
  expect(() =>
    computeGeodesicPairsOnCPU({origins: new Float32Array(2), targets: new Float32Array(4)}, context)
  ).toThrow(/targets/);
  const result = computeGeodesicPairsOnCPU(
    {origins: new Float32Array([0, NaN, 0, 91]), targets: new Float32Array(4)},
    context
  );
  expect(result.distances.every(Number.isNaN)).toBe(true);
  expect(result.midpoints.every(Number.isNaN)).toBe(true);
  expect(Array.from(result.converged)).toEqual([0, 0]);
});

it('validates both operation contracts at construction and CPU context/options', () => {
  const props: GeodesicPairsOperationProps = {
    id: 'pair-check',
    origins: 'origins',
    targets: 'targets',
    pairCount: 2,
    spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'ellipsoidal'},
    output: {distances: 'distances', midpoints: 'midpoints', status: {converged: 'converged'}}
  };
  for (const Operation of [CPUGeodesicPairsOperation, GPUGeodesicPairsOperation]) {
    expect(() => new Operation(props)).not.toThrow();
    expect(() => new Operation({...props, pairCount: 0})).toThrow(/^pair-check/);
    expect(() => new Operation({...props, output: {distances: 'origins'}})).toThrow(/^pair-check/);
  }
  expect(() => new CPUGeodesicPairsOperation({...props, maximumIterations: 0})).toThrow(
    /^pair-check.*maximumIterations/
  );
  expect(
    () =>
      new CPUGeodesicPairsOperation({
        ...props,
        spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'great-circle'}
      })
  ).toThrow(/^pair-check.*converged/);
  expect(
    () =>
      new CPUGeodesicPairsOperation({
        ...props,
        spatialContext: {coordinateSpace: 'planar', metric: 'native'}
      })
  ).toThrow(/^pair-check/);
});

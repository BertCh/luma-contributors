// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {computeGeodesicPairsOnCPU} from '../../../src/gpu-spatial-analysis/geometry-measures/cpu-geodesic-pairs-operation';

// Independent one-step stopping witness: equatorial inverse has
// lambda_1-lambda_0=f*pi/180=5.85e-5 > 1e-6. Coincident rows have zero change;
// antipodes are singular before iteration. See adjacent .oracle.py.
it('CPU inverse status matches independent unfinished, coincident and singular cases', () => {
  const result = computeGeodesicPairsOnCPU(
    {
      origins: Float32Array.of(0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
      targets: Float32Array.of(1, 0, 0, 0, 180, 0, 0.01, 0, 0.001, 0)
    },
    {coordinateSpace: 'longitude-latitude', metric: 'ellipsoidal'},
    1
  );
  // Last row's 5.85176e-8 change is below a constant 4.8e-7 but above its
  // relative threshold 8.40567e-12. The adjacent oracle independently derives it.
  expect(Array.from(result.converged)).toEqual([0, 1, 0, 0, 0]);
  expect(Array.from(result.approximation)).toEqual([1, 0, 1, 1, 1]);
  expect(Array.from(result.iterationLimitReached)).toEqual([1, 0, 0, 1, 1]);
});

// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {prepareCRSProjection} from '@luma.gl/graph/gpu-project/crs';
import {createNullWebGPUDevice} from '../../src/testing/gpu-contributor-test-utils';
import {GPUGeometryMeasures} from '../../src/gpu-spatial-analysis/geometry-measures/gpu-geometry-measures';
import {computeGeodesicPairsOnCPU} from '../../src/gpu-spatial-analysis/geometry-measures/cpu-geodesic-pairs-operation';
import {computeLongitudeDelta, wrapLongitude} from '../../src/analysis-core/kernels/cpu-geodesic';
import {createGPUSpatialFrameFromCRS} from '../../src/analysis-core/contracts/spatial-frame-crs';
import {createGPULocalFrame} from '../../src/analysis-core/contracts/local-frame';
import {
  getGPUSpatialContextCompatibility,
  validateGPUSpatialContext,
  type GPUSpatialContext,
  type GPUSpatialFrame
} from '../../src/analysis-core/contracts/spatial-context';
import {getCPUTrajectoryRows} from '../../src/analysis-core/operations/cpu-trajectory-rows';

it('B1 constructor rejects false meter descriptions including metric none and missing unit metadata', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  let serial = 0;
  const create = (frame: GPUSpatialFrame) =>
    new GPUGeometryMeasures({
      id: 'review-measures',
      geometryType: 'points',
      spatialContext: {coordinateSpace: 'planar', metric: 'none', units: 'meters', frame},
      positions: createTransientView(graph, `positions-${serial++}`, 'float32x2', 1),
      output: {centroids: createTransientView(graph, `centroids-${serial++}`, 'float32x2', 1)}
    });
  {
    expect(create(createGPUSpatialFrameFromCRS('EPSG:32618')).measures).toMatchObject({
      lengthUnits: 'meters',
      areaUnits: 'square-meters'
    });
    for (const frame of [
      createGPUSpatialFrameFromCRS('EPSG:4326'),
      createGPUSpatialFrameFromCRS('+proj=utm +zone=18 +units=us-ft'),
      {id: 'manual', kind: 'crs' as const, crs: 'EPSG:4326'}
    ]) {
      expect(() => create(frame)).toThrow(/review-measures.*(units|project)/);
    }
  }
});

it('B2 explicit resident tracks retain unowned prefixes and tails while compact tracks reject them', () => {
  const positions = new Float32Array(6);
  const timestamps = Float32Array.of(0, 1, 2);
  for (const trackOffsets of [Uint32Array.of(1, 3), Uint32Array.of(0, 2)]) {
    expect(() => getCPUTrajectoryRows({positions, timestamps, trackOffsets})).toThrow(
      /trackOffsets/
    );
    expect(
      getCPUTrajectoryRows({positions, timestamps, trackOffsets, storage: 'resident-capacity'})
        .trackOffsets
    ).toEqual(trackOffsets);
  }
});

it('B3 nearest-even half-turn preserves rhumb bearing and midpoint', () => {
  // -180/360=-0.5 rounds to even zero; the westward half-turn midpoint is (110,15).
  expect(computeLongitudeDelta(200, 20)).toBe(-180);
  expect(wrapLongitude(540)).toBe(-180);
  const result = computeGeodesicPairsOnCPU(
    {origins: Float32Array.of(200, 15), targets: Float32Array.of(20, 15)},
    {coordinateSpace: 'longitude-latitude', metric: 'rhumb'}
  );
  expect(result.initialBearings).toEqual(Float32Array.of(-90));
  expect(result.finalBearings).toEqual(Float32Array.of(-90));
  expect(result.midpoints).toEqual(Float32Array.of(110, 15));
  // Constant latitude rhumb length is pi * R * cos(15 degrees).
  expect(Number.isFinite(result.distances[0])).toBe(true);
  expect(Math.abs(result.distances[0] - 19333115.955693815)).toBeLessThan(2);
});

it('B4 native meter frames match explicit meters while a different native scale fails', () => {
  const source: GPUSpatialContext = {
    coordinateSpace: 'planar',
    metric: 'native',
    frame: createGPUSpatialFrameFromCRS('EPSG:32618')
  };
  expect(
    getGPUSpatialContextCompatibility(
      source,
      {...source, units: 'meters'},
      {purpose: 'metric-results'}
    )
  ).toEqual({compatible: true});
  const kilometerFrame = {...source.frame!, unit: {name: 'kilometers', metersPerUnit: 1000}};
  expect(
    getGPUSpatialContextCompatibility(
      source,
      {...source, frame: kilometerFrame},
      {purpose: 'metric-results'}
    ).compatible
  ).toBe(false);
});

it('B5 relative default origin is published and passed into both projection plans', () => {
  const frame = createGPUSpatialFrameFromCRS('EPSG:3857', {anchor: 'relative-origin'});
  expect(frame.origin).toEqual([0, 0]);
  for (const bounds of [
    [10, 10, 11, 11],
    [10010, 10, 10011, 11]
  ] as const) {
    const result = prepareCRSProjection({
      from: 'EPSG:3857',
      to: frame.crs!,
      bounds,
      tolerance: 0.001,
      destinationOrigin: frame.origin
    });
    expect(result.status).toBe('ready');
    if (result.status !== 'ready') throw new Error(JSON.stringify(result.reasons));
    expect(result.program.destinationOrigin).toEqual([0, 0]);
  }
});

it('S2 cloned shared-buffer wrappers preserve complete projected and inverse arrays', () => {
  const frame = createGPULocalFrame({origin: [0, 0]});
  const forwardShared = new SharedArrayBuffer(6 * Float32Array.BYTES_PER_ELEMENT);
  const forwardSource = new Float32Array(forwardShared, 0, 4);
  forwardSource.set([0, 0, 0, 1]);
  const forwardTarget = new Float32Array(
    structuredClone(forwardShared),
    2 * Float32Array.BYTES_PER_ELEMENT,
    4
  );
  expect(frame.projectArray(forwardSource, 2, forwardTarget)).toEqual(
    Float32Array.of(0, 0, 0, 111195.078125)
  );
  const shared = new SharedArrayBuffer(6 * Float64Array.BYTES_PER_ELEMENT);
  const source = new Float64Array(shared, 0, 4);
  source.set([0, 0, 0, 1]);
  const target = new Float32Array(structuredClone(shared), 2 * Float64Array.BYTES_PER_ELEMENT, 4);
  expect(frame.projectArray(source, 2, target)).toEqual(Float32Array.of(0, 0, 0, 111195.078125));
  source.set([0, 0, 0, 111195.0802335329]);
  const inverseTarget = new Float64Array(
    structuredClone(shared),
    2 * Float64Array.BYTES_PER_ELEMENT,
    4
  );
  const actual = frame.unprojectArray(source, inverseTarget);
  expect(actual).toHaveLength(4);
  for (const [index, expected] of [0, 0, 0, 1].entries()) {
    expect(Number.isFinite(actual[index])).toBe(true);
    expect(actual[index]).toBeCloseTo(expected, 10);
  }
});

it('S4 rejects unknown units and local-frame kinds with accepted spellings', () => {
  const context = {
    coordinateSpace: 'longitude-latitude',
    metric: 'great-circle',
    units: 'yards'
  } as unknown as GPUSpatialContext;
  expect(() => validateGPUSpatialContext('review-units', context)).toThrow(
    /review-units.*units.*native.*meters.*kilometers/
  );
  expect(() =>
    createGPULocalFrame({origin: [0, 0], kind: 'unknown' as 'lambert-equal-area'})
  ).toThrow(/kind.*azimuthal-equidistant.*lambert-equal-area/);
});

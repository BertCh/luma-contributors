// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, it} from 'vitest';
import {computeLabelPointOnCPU} from '../../src/gpu-spatial-analysis/label-point/cpu-label-point-operation';
import {computeLongitudeDelta} from '../../src/analysis-core/kernels/cpu-geodesic';
import {computeGeodesicPairsOnCPU} from '../../src/gpu-spatial-analysis/geometry-measures/cpu-geodesic-pairs-operation';
import {createGPUSpatialFrameFromCRS} from '../../src/analysis-core/contracts/spatial-frame-crs';
import {createGPULocalFrame} from '../../src/analysis-core/contracts/local-frame';
import {getCPUTrajectoryRows} from '../../src/analysis-core/operations/cpu-trajectory-rows';
import {validateCPUPolygonRows} from '../../src/gpu-spatial-analysis/shared/cpu-polygon-rows';
import {
  assertCompatibleGPUSpatialContexts,
  getGPUSpatialContextCompatibility,
  resolveGPUSpatialContext,
  type GPUSpatialContext
} from '../../src/analysis-core/contracts/spatial-context';

const geographic: GPUSpatialContext = {
  coordinateSpace: 'longitude-latitude',
  metric: 'great-circle'
};
const planar = (frame: ReturnType<typeof createGPUSpatialFrameFromCRS>): GPUSpatialContext => ({
  coordinateSpace: 'planar',
  metric: 'native',
  frame
});

describe('spatial frames and compact host rows regression', () => {
  it('DGN-07 reduces huge endpoints before subtraction', () => {
    // Exact integer remainder: 2**62 % 360 = 184 = -176 degrees.
    // Equatorial spherical arc: 6371008.8 * 176 * pi/180 = 19570334.12110179 m.
    expect(computeLongitudeDelta(2 ** 62, 0)).toBe(176);
    expect(computeLongitudeDelta(0, 2 ** 62)).toBe(-176);
    const hugeDestination = computeGeodesicPairsOnCPU(
      {origins: Float32Array.of(0, 0), targets: Float32Array.of(2 ** 62, 0)},
      geographic
    );
    expect(hugeDestination.distances).toEqual(Float32Array.of(19570334));
    const huge = computeGeodesicPairsOnCPU(
      {origins: Float32Array.of(2 ** 62, 0), targets: Float32Array.of(0, 0)},
      geographic
    );
    const canonical = computeGeodesicPairsOnCPU(
      {origins: Float32Array.of(-176, 0), targets: Float32Array.of(0, 0)},
      geographic
    );
    expect(Number.isFinite(huge.distances[0])).toBe(true);
    expect(huge.distances[0]).toBeCloseTo(19570334, 0);
    expect(huge.distances).toEqual(canonical.distances);
    expect(huge.converged).toEqual(Uint32Array.of(1));
    expect(huge.approximation).toEqual(Uint32Array.of(0));
    expect(computeLongitudeDelta(179.999999, -179.999999)).toBeCloseTo(0.000002, 10);
  });

  it.each([
    'longlat',
    'latlong',
    'lonlat',
    'latlon'
  ])('UNI-02 resolves %s as angular', projection => {
    expect(
      createGPUSpatialFrameFromCRS(`+proj=${projection} +datum=WGS84`).unit?.degreesPerUnit
    ).toBe(1);
  });
  it('UNI-02 rejects contradictory meters and planar angular measurements', () => {
    for (const crs of ['EPSG:4326', '+proj=utm +zone=33 +units=us-ft']) {
      expect(() =>
        resolveGPUSpatialContext('fixture', {
          ...planar(createGPUSpatialFrameFromCRS(crs)),
          units: 'meters'
        })
      ).toThrow(/fixture.*units/);
    }
    expect(() =>
      resolveGPUSpatialContext('fixture', planar(createGPUSpatialFrameFromCRS('EPSG:4326')))
    ).toThrow(/project/);
    expect(
      resolveGPUSpatialContext('fixture', {coordinateSpace: 'planar', metric: 'native'}).units
    ).toBeUndefined();
  });

  it('UNI-05 compares effective metric parameters only when requested', () => {
    const options = {purpose: 'metric-results' as const};
    const kilometers: GPUSpatialContext = {...geographic, units: 'kilometers'};
    const unitSphere = {...geographic, sphereRadius: 1};
    expect(getGPUSpatialContextCompatibility(geographic, kilometers).compatible).toBe(true);
    expect(getGPUSpatialContextCompatibility(geographic, unitSphere).compatible).toBe(true);
    expect(getGPUSpatialContextCompatibility(geographic, kilometers, options).compatible).toBe(
      false
    );
    expect(getGPUSpatialContextCompatibility(geographic, unitSphere, options).compatible).toBe(
      false
    );
    expect(
      getGPUSpatialContextCompatibility(
        geographic,
        {...geographic, sphereRadius: 6371008.8},
        options
      ).compatible
    ).toBe(true);
    expect(() =>
      assertCompatibleGPUSpatialContexts('fixture', geographic, unitSphere, options)
    ).toThrow(/sphere/);
    const ellipsoidal: GPUSpatialContext = {...geographic, metric: 'ellipsoidal'};
    expect(
      getGPUSpatialContextCompatibility(
        ellipsoidal,
        {...ellipsoidal, ellipsoid: {semiMajorAxis: 6378137, flattening: 1 / 298.257223563}},
        options
      ).compatible
    ).toBe(true);
    expect(
      getGPUSpatialContextCompatibility(
        ellipsoidal,
        {...ellipsoidal, ellipsoid: {semiMajorAxis: 6378137, flattening: 0}},
        options
      ).compatible
    ).toBe(false);
  });

  it('UNI-09 captures storage origins and checks even explicit ids', () => {
    const origin: [number, number] = [500000, 4600000];
    const first = createGPUSpatialFrameFromCRS('EPSG:32633', {origin, anchor: 'relative-origin'});
    const second = createGPUSpatialFrameFromCRS('EPSG:32633', {
      origin: [500000, 4610000],
      anchor: 'relative-origin'
    });
    origin[1] = 0;
    expect(first.origin).toEqual([500000, 4600000]);
    expect(first.id).not.toBe(second.id);
    expect(
      getGPUSpatialContextCompatibility(
        planar({...first, id: 'same'}),
        planar({...second, id: 'same'})
      ).compatible
    ).toBe(false);
    expect(
      getGPUSpatialContextCompatibility(
        planar({...first, anchor: undefined, precision: 'float32-origin-relative', id: 'same'}),
        planar({...second, id: 'same'})
      ).compatible
    ).toBe(false);
  });

  it('UNI-13 refuses partial rows before mutation in both directions', () => {
    const frame = createGPULocalFrame({origin: [0, 0]});
    const forward = Float32Array.of(7, 7, 7, 7);
    const inverse = Float64Array.of(7, 7, 7, 7);
    expect(() => frame.projectArray([0, 0, 1], 2, forward)).toThrow(/length 3.*stride 2/);
    expect(() => frame.unprojectArray([0, 0, 1], inverse)).toThrow(/length 3.*stride 2/);
    expect(() => frame.projectArray([0, 0, 0, 1], 3, forward)).toThrow(/length 4.*stride 3/);
    expect(Array.from(forward)).toEqual([7, 7, 7, 7]);
    expect(Array.from(inverse)).toEqual([7, 7, 7, 7]);
  });
  it('UNI-13 ING-03 snapshots overlapping offset views in both directions', () => {
    const frame = createGPULocalFrame({origin: [0, 0]});
    // On the equator x=R*pi/180, y=0; inverse of [R*pi/180,0] is [1,0].
    const forward = Float32Array.of(99, 0, 0, 1, 0, 99, 99);
    frame.projectArray(forward.subarray(1, 5), 2, forward.subarray(3));
    expect(Array.from(forward)).toEqual([99, 0, 0, 0, 0, 111195.078125, 0]);
    const inverse = Float64Array.of(99, 0, 0, 111195.0802335329, 0, 99, 99);
    frame.unprojectArray(inverse.subarray(1, 5), inverse.subarray(3));
    expect(inverse).toHaveLength(7);
    for (const [index, expected] of [99, 0, 0, 0, 0, 1, 0].entries()) {
      expect(Number.isFinite(inverse[index])).toBe(true);
      expect(inverse[index]).toBeCloseTo(expected, 10);
    }
  });

  it('DGN-09 UGE-10 accepts terminal zero, preserves empty parents, rejects malformed compact hierarchies', () => {
    const empty = {
      positions: new Float32Array(),
      ringOffsets: Uint32Array.of(0),
      polygonOffsets: Uint32Array.of(0),
      featureOffsets: Uint32Array.of(0)
    };
    expect(() => validateCPUPolygonRows(empty)).not.toThrow();
    expect(computeLabelPointOnCPU(empty).points).toHaveLength(0);
    const emptyFeature = computeLabelPointOnCPU({...empty, featureOffsets: Uint32Array.of(0, 0)});
    expect(emptyFeature.points).toHaveLength(2);
    expect(Array.from(emptyFeature.degenerate)).toEqual([1]);
    expect(() =>
      validateCPUPolygonRows({
        ...empty,
        ringOffsets: Uint32Array.of(0, 999, 0),
        storage: 'resident-capacity'
      })
    ).not.toThrow();
    expect(() =>
      validateCPUPolygonRows({...empty, featureOffsets: Uint32Array.of(0, 0)})
    ).not.toThrow();
    for (const offsets of [[1], [0, 999], [0, 1, 0]]) {
      expect(() =>
        validateCPUPolygonRows({...empty, ringOffsets: Uint32Array.from(offsets)})
      ).toThrow(/ringOffsets/);
    }
    expect(() =>
      validateCPUPolygonRows({
        ...empty,
        positions: new Float32Array(6),
        ringOffsets: Uint32Array.of(0, 2, 1, 3),
        polygonOffsets: Uint32Array.of(0, 3),
        featureOffsets: Uint32Array.of(0, 1)
      })
    ).toThrow(/ringOffsets/);
    expect(() =>
      getCPUTrajectoryRows({
        positions: new Float32Array(6),
        timestamps: new Float32Array(3),
        trackOffsets: Uint32Array.of(0, 2, 1, 3)
      })
    ).toThrow(/trackOffsets/);
    const square = {
      ...empty,
      positions: Float32Array.of(0, 0, 1, 0, 1, 1, 0, 1),
      ringOffsets: Uint32Array.of(0, 999),
      polygonOffsets: Uint32Array.of(0, 1),
      featureOffsets: Uint32Array.of(0, 1)
    };
    expect(() => validateCPUPolygonRows(square)).toThrow(/ringOffsets/);
    expect(() => validateCPUPolygonRows(square, {storage: 'resident-capacity'})).not.toThrow();
    expect(() => validateCPUPolygonRows({...empty, positions: Float32Array.of(0, 0)})).toThrow(
      /ringOffsets/
    );
    expect(() => validateCPUPolygonRows({...empty, polygonOffsets: Uint32Array.of(0, 1)})).toThrow(
      /polygonOffsets/
    );
    expect(() => validateCPUPolygonRows({...empty, featureOffsets: Uint32Array.of(0, 1)})).toThrow(
      /featureOffsets/
    );
    const tracks = {
      positions: new Float32Array(),
      timestamps: new Float32Array(),
      trackOffsets: Uint32Array.of(0)
    };
    expect(getCPUTrajectoryRows(tracks).trackOffsets).toEqual(Uint32Array.of(0));
    expect(
      getCPUTrajectoryRows({...tracks, trackOffsets: Uint32Array.of(0, 0)}).trackOffsets
    ).toHaveLength(2);
    expect(() => getCPUTrajectoryRows({...tracks, trackOffsets: Uint32Array.of(0, 9)})).toThrow(
      /trackOffsets/
    );
    expect(
      getCPUTrajectoryRows(
        {...tracks, trackOffsets: Uint32Array.of(0, 9)},
        {storage: 'resident-capacity'}
      ).trackOffsets
    ).toEqual(Uint32Array.of(0, 0));
  });
});

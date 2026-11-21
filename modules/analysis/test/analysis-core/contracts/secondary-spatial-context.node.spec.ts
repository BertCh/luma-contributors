// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, it} from 'vitest';
import {
  getGPUSpatialContextCompatibility,
  type GPUSpatialContextCompatibilityOptions
} from '@luma-contributors/analysis/analysis-core';
import type {GPUSpatialContextCompatibilityOptions as ContractsCompatibilityOptions} from '../../../src/analysis-core/contracts/index';
import {snapshotConstructionSpatialContext} from '../../../src/analysis-core/kernels/construction-snapshot';
import {
  GPU_LINEAR_EDGE_MODELS,
  GPU_PLANAR_SPATIAL_CONTEXT,
  GPU_LONGITUDE_LATITUDE_SPATIAL_CONTEXT,
  GPU_GREAT_CIRCLE_SPATIAL_CONTEXT,
  resolveGPUSpatialContext,
  resolveGPUSecondarySpatialContext,
  type GPUSpatialContext
} from '../../../src/analysis-core/contracts/index';

const PRIMARY: GPUSpatialContext = {
  coordinateSpace: 'longitude-latitude',
  metric: 'great-circle',
  units: 'meters',
  datum: 'EPSG:4326'
};
const PLANAR: GPUSpatialContext = {
  coordinateSpace: 'planar',
  metric: 'native',
  frame: {id: 'primary-frame', kind: 'crs'}
};
const LABEL = 'join polygonsSpatialContext';

it.each([
  undefined,
  GPU_PLANAR_SPATIAL_CONTEXT,
  GPU_LONGITUDE_LATITUDE_SPATIAL_CONTEXT,
  GPU_GREAT_CIRCLE_SPATIAL_CONTEXT
])('review B5 reuses library-owned default snapshots between stages: %s', context => {
  const resolved = resolveGPUSpatialContext('default-stage', context, {
    coordinateSpaces: ['planar', 'longitude-latitude'],
    metrics: ['native', 'none', 'great-circle']
  });
  expect(snapshotConstructionSpatialContext(resolved)).toBe(resolved);
  if (context) expect(resolved).toBe(context);
});

it.each([
  false,
  0,
  '',
  1,
  'planar'
])('review S1 rejects supplied non-object context %s with the caller ID', value => {
  expect(() =>
    resolveGPUSpatialContext('invalid-context', value as unknown as GPUSpatialContext)
  ).toThrow(/invalid-context.*spatialContext.*object/);
});

it('review B3 captures a changing primary before secondary compatibility checking', () => {
  let reads = 0;
  const primary: GPUSpatialContext = {
    coordinateSpace: 'planar',
    get metric() {
      return ++reads === 1 ? 'native' : 'great-circle';
    }
  };
  const secondary: GPUSpatialContext = {coordinateSpace: 'planar', metric: 'native'};
  expect(resolveGPUSecondarySpatialContext('capture-primary', 'other', secondary, primary)).toEqual(
    secondary
  );
  expect(reads).toBe(1);
});

it('review B3 captures one descriptor once when both sides share it', () => {
  let reads = 0;
  const context: GPUSpatialContext = {
    coordinateSpace: 'planar',
    get metric() {
      return ++reads === 1 ? 'native' : 'great-circle';
    }
  };
  expect(
    resolveGPUSecondarySpatialContext('shared-context', 'other', context, context).metric
  ).toBe('native');
  expect(reads).toBe(1);
});

it('review B5 shares a known immutable resolved snapshot between stages', () => {
  const resolved = resolveGPUSpatialContext('first', PLANAR);
  expect(resolved).not.toBe(PLANAR);
  expect(resolveGPUSpatialContext('second', resolved)).toBe(resolved);
  expect(resolveGPUSecondarySpatialContext('third', 'other', resolved, resolved)).toBe(resolved);
});

it('UNI-05 public compatibility options select metric results while coordinates stay the default', () => {
  const options: GPUSpatialContextCompatibilityOptions = {purpose: 'metric-results'};
  const contractsOptions: ContractsCompatibilityOptions = options;
  const unitSphere = {...PRIMARY, sphereRadius: 1};
  expect(getGPUSpatialContextCompatibility(unitSphere, PRIMARY)).toEqual({compatible: true});
  expect(getGPUSpatialContextCompatibility(unitSphere, PRIMARY, contractsOptions)).toEqual({
    compatible: false,
    reason: 'effective sphere radii differ'
  });
});

describe('resolveGPUSecondarySpatialContext', () => {
  it('retains the shared frozen planar default when the primary context is absent', () => {
    expect(resolveGPUSpatialContext('join', undefined)).toBe(GPU_PLANAR_SPATIAL_CONTEXT);
  });

  it('returns the primary object when the secondary context is absent', () => {
    expect(resolveGPUSecondarySpatialContext('join', 'polygons', undefined, PRIMARY)).toBe(PRIMARY);
  });

  it('validates the secondary before checking compatibility', () => {
    expect(() =>
      resolveGPUSecondarySpatialContext('join', 'polygons', {...PRIMARY, datum: ' '}, PRIMARY)
    ).toThrowError(new Error(`${LABEL} datum must be a non-empty string`));
  });

  it('accepts WGS84 as an alias for EPSG:4326', () => {
    const secondary = {...PRIMARY, datum: 'WGS84'};
    expect(resolveGPUSecondarySpatialContext('join', 'polygons', secondary, PRIMARY)).toEqual(
      secondary
    );
  });

  it('accepts an unmeasured secondary with a great-circle primary', () => {
    const secondary: GPUSpatialContext = {...PRIMARY, metric: 'none'};
    expect(resolveGPUSecondarySpatialContext('join', 'polygons', secondary, PRIMARY)).toEqual(
      secondary
    );
  });

  it('rejects nonlinear longitude/latitude edges before checking compatibility', () => {
    expect(() =>
      resolveGPUSecondarySpatialContext(
        'join',
        'polygons',
        {...PRIMARY, edgeModel: 'great-circle'},
        PRIMARY,
        GPU_LINEAR_EDGE_MODELS
      )
    ).toThrowError(
      new Error(`${LABEL} needs linear edges; prepare polygons with GPULonLatPolygonPreparation`)
    );
  });

  const mismatches: {
    name: string;
    primary: GPUSpatialContext;
    secondary: GPUSpatialContext;
    reason: string;
  }[] = [
    {
      name: 'datum',
      primary: PRIMARY,
      secondary: {...PRIMARY, datum: 'EPSG:4267'},
      reason: 'datums differ (EPSG:4267 and EPSG:4326)'
    },
    {
      name: 'planar frame',
      primary: PLANAR,
      secondary: {...PLANAR, frame: {id: 'secondary-frame', kind: 'crs'}},
      reason: 'planar frames differ (secondary-frame and primary-frame)'
    },
    {
      name: 'units',
      primary: PRIMARY,
      secondary: {...PRIMARY, units: 'kilometers'},
      reason: 'units differ (kilometers and meters)'
    },
    {
      name: 'coordinate space',
      primary: PLANAR,
      secondary: PRIMARY,
      reason: 'coordinate spaces differ (longitude-latitude and planar)'
    },
    {
      name: 'metric',
      primary: PRIMARY,
      secondary: {...PRIMARY, metric: 'rhumb'},
      reason: 'metrics differ (rhumb and great-circle)'
    }
  ];
  for (const {name, primary, secondary, reason} of mismatches) {
    it(`rejects a mismatched ${name}, listing the secondary first`, () => {
      expect(() =>
        resolveGPUSecondarySpatialContext('join', 'polygons', secondary, primary)
      ).toThrowError(new Error(`${LABEL} incompatible spatial contexts: ${reason}`));
    });
  }
});

for (const resolver of ['primary', 'secondary'] as const) {
  const resolve = (context: GPUSpatialContext) =>
    resolver === 'primary'
      ? resolveGPUSpatialContext('snapshot', context)
      : resolveGPUSecondarySpatialContext('snapshot', 'other', context, PLANAR);

  it(`R14 ${resolver} captures nested changing context and frame getters once`, () => {
    let metricReads = 0;
    let unitReads = 0;
    let frameReads = 0;
    let frameUnitReads = 0;
    const frame: NonNullable<GPUSpatialContext['frame']> = {
      id: 'primary-frame',
      kind: 'crs',
      get unit() {
        frameUnitReads++;
        return frameUnitReads === 1
          ? {name: 'meter', metersPerUnit: 1}
          : {name: 'degree', degreesPerUnit: 1};
      }
    };
    const context: GPUSpatialContext = {
      coordinateSpace: 'planar',
      get metric() {
        return ++metricReads === 1 ? 'native' : 'great-circle';
      },
      get units() {
        return ++unitReads === 1 ? 'native' : 'kilometers';
      },
      get frame(): GPUSpatialContext['frame'] {
        frameReads++;
        return frameReads === 1 ? frame : {id: 'changed', kind: 'crs'};
      }
    };
    const resolved = resolve(context);
    expect(resolved.metric).toBe('native');
    expect(resolved.units).toBe('native');
    expect(resolved.frame?.unit).toEqual({name: 'meter', metersPerUnit: 1});
    expect([metricReads, unitReads, frameReads, frameUnitReads]).toEqual([1, 1, 1, 1]);
  });

  it(`R14 ${resolver} retains copied frame metadata after caller mutation`, () => {
    const origin: [number, number] = [0, 0];
    const unit = {name: 'meter', metersPerUnit: 1};
    const context: GPUSpatialContext = {
      coordinateSpace: 'planar',
      metric: 'native',
      units: 'native',
      frame: {id: 'primary-frame', kind: 'crs', origin, unit}
    };
    const resolved = resolve(context);
    context.metric = 'none';
    context.frame!.id = 'changed';
    origin[0] = 100;
    unit.metersPerUnit = 1000;
    expect(resolved.metric).toBe('native');
    expect(resolved.frame?.id).toBe('primary-frame');
    expect(resolved.frame?.origin).toEqual([0, 0]);
    expect(resolved.frame?.unit?.metersPerUnit).toBe(1);
  });

  it(`R14 ${resolver} captures ellipsoid getters and retains their values`, () => {
    let axisReads = 0;
    let flatteningReads = 0;
    const context: GPUSpatialContext = {
      coordinateSpace: 'longitude-latitude',
      metric: 'ellipsoidal',
      units: 'meters',
      ellipsoid: {
        get semiMajorAxis() {
          return ++axisReads === 1 ? 6378137 : -1;
        },
        get flattening() {
          return ++flatteningReads === 1 ? 1 / 298.257223563 : 2;
        }
      }
    };
    const primary: GPUSpatialContext = {
      coordinateSpace: 'longitude-latitude',
      metric: 'ellipsoidal'
    };
    const resolved =
      resolver === 'primary'
        ? resolveGPUSpatialContext('snapshot', context, {
            coordinateSpaces: ['longitude-latitude'],
            metrics: ['ellipsoidal']
          })
        : resolveGPUSecondarySpatialContext('snapshot', 'other', context, primary);
    context.ellipsoid = {semiMajorAxis: 1, flattening: 0};
    expect(resolved.ellipsoid).toEqual({semiMajorAxis: 6378137, flattening: 1 / 298.257223563});
    expect([axisReads, flatteningReads]).toEqual([1, 1]);
  });
}

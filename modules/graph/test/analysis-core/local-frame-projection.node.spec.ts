// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, it} from 'vitest';
import {projectionEngine} from '@math.gl/geospatial/projection';
import {
  createGPULocalFrame,
  type GPULocalFrameKind
} from '../../src/analysis-core/contracts/local-frame';

// Forward values recorded from the hand-written binary64 implementation that createGPULocalFrame
// used before delegating to math.gl's aeqd (+R=6371008.8) and laea (+ellps=WGS84) projections.
type PinnedCase = {
  label: string;
  origin: [number, number];
  lngLat: [number, number];
  xy: [number, number];
  /** The inverse longitude is undefined at the pole. */
  skipInverse?: boolean;
  /** Absolute tolerance override where the recorded value lost precision (asin near +-1). */
  tolerance?: number;
};

const PINNED_CASES: Record<GPULocalFrameKind, PinnedCase[]> = {
  'azimuthal-equidistant': [
    {
      label: 'near origin',
      origin: [8.5417, 47.3769],
      lngLat: [8.5427, 47.3779],
      xy: [75.29684330257996, 111.19556373771204]
    },
    {
      label: 'mid range',
      origin: [8.5417, 47.3769],
      lngLat: [9.5417, 49.3769],
      xy: [72409.57336168787, 222860.11359247612]
    },
    {
      label: 'far (about 170 degrees away)',
      origin: [8.5417, 47.3769],
      lngLat: [-171.4583, -40],
      xy: [-1.4024861333572813e-8, 19194839.454660837]
    },
    {
      label: 'north pole',
      origin: [8.5417, 47.3769],
      lngLat: [8.5417, 90],
      xy: [0, 4739479.024301897],
      skipInverse: true
    },
    {
      label: 'mid range west',
      origin: [8.5417, 47.3769],
      lngLat: [-8.4583, 47.3769],
      xy: [-1269864.2097734117, 139646.52665870078]
    },
    {
      label: 'polar origin mid',
      origin: [0, 90],
      lngLat: [45, 80],
      xy: [786267.9526771338, -786267.9526771335]
    },
    {
      label: 'polar origin far',
      origin: [0, 90],
      lngLat: [-120, 10],
      xy: [-7703821.140647071, 4447803.209341314]
    },
    {
      label: 'polar origin near',
      origin: [0, 90],
      lngLat: [0.001, 89.999],
      xy: [0.0019407202619976347, -111.19508021630708]
    },
    {
      label: 'antimeridian crossing',
      origin: [179.5, -17],
      lngLat: [-179.2, -16.4],
      xy: [138673.84459493385, 66262.31973282012]
    },
    {
      label: 'antimeridian near',
      origin: [179.5, -17],
      lngLat: [-179.999, -17.001],
      xy: [53274.18610308792, -179.2952653707971]
    },
    {
      label: 'antimeridian far',
      origin: [179.5, -17],
      lngLat: [-0.5, 16],
      xy: [-1.3425663994380384e-7, -19903919.36180465]
    },
    {
      label: 'near origin southern',
      origin: [151.2093, -33.8688],
      lngLat: [151.2103, -33.8678],
      xy: [92.3281222172156, 111.19463121784273]
    },
    {
      label: 'far southern',
      origin: [151.2093, -33.8688],
      lngLat: [-30, 20],
      xy: [1523543.695000582, -18405410.38400572]
    },
    {
      label: 'antimeridian origin',
      origin: [-179.9, 0],
      lngLat: [179.9, 0.5],
      xy: [-22238.45151028148, 55597.65302116215]
    }
  ],
  'lambert-equal-area': [
    {
      label: 'near origin',
      origin: [8.5417, 47.3769],
      lngLat: [8.5427, 47.3779],
      xy: [75.51808013857428, 111.17868221637922]
    },
    {
      label: 'mid range',
      origin: [8.5417, 47.3769],
      lngLat: [9.5417, 49.3769],
      xy: [72629.5185249931, 222848.35415752768]
    },
    {
      label: 'far (about 170 degrees away)',
      origin: [8.5417, 47.3769],
      lngLat: [-171.4583, -40],
      xy: [-9.314699763602373e-9, 12709093.599520024]
    },
    {
      label: 'north pole',
      origin: [8.5417, 47.3769],
      lngLat: [8.5417, 90],
      xy: [0, 4641801.398754066],
      skipInverse: true
    },
    {
      label: 'mid range west',
      origin: [8.5417, 47.3769],
      lngLat: [-8.4583, 47.3769],
      xy: [-1271493.776254976, 139394.72350965595]
    },
    {
      label: 'polar origin mid',
      origin: [0, 90],
      lngLat: [45, 80],
      xy: [788713.3037299647, -788713.3037299642]
    },
    {
      label: 'polar origin far',
      origin: [0, 90],
      lngLat: [-120, 10],
      xy: [-7096332.8973868005, 4097069.7085654647]
    },
    {
      label: 'polar origin near',
      origin: [0, 90],
      lngLat: [0.001, 89.999],
      xy: [0.001949427268736727, -111.69395495459835],
      tolerance: 1e-4
    },
    {
      label: 'antimeridian crossing',
      origin: [179.5, -17],
      lngLat: [-179.2, -16.4],
      xy: [138861.79110889576, 65945.67527224154]
    },
    {
      label: 'antimeridian near',
      origin: [179.5, -17],
      lngLat: [-179.999, -17.001],
      xy: [53348.90566448736, -178.44686469661661]
    },
    {
      label: 'antimeridian far',
      origin: [179.5, -17],
      lngLat: [-0.5, 16],
      xy: [-8.638591872851706e-8, -12728500.838609783]
    },
    {
      label: 'near origin southern',
      origin: [151.2093, -33.8688],
      lngLat: [151.2103, -33.8678],
      xy: [92.52765448057268, 110.91956253086032]
    },
    {
      label: 'far southern',
      origin: [151.2093, -33.8688],
      lngLat: [-30, 20],
      xy: [1047472.3549033514, -12595623.780925354]
    },
    {
      label: 'antimeridian origin',
      origin: [-179.9, 0],
      lngLat: [179.9, 0.5],
      xy: [-22263.256715295956, 55287.06078056189]
    }
  ]
};

/** 1e-6 m absolute plus 1e-12 relative for projected coordinates up to 2e7 m. */
function getTolerance(value: number): number {
  return 1e-6 + 1e-12 * Math.abs(value);
}

describe('createGPULocalFrame pinned projections', () => {
  for (const kind of Object.keys(PINNED_CASES) as GPULocalFrameKind[]) {
    it.each(PINNED_CASES[kind])(`$label matches the recorded ${kind} value`, pinnedCase => {
      const {origin, lngLat, xy, skipInverse, tolerance} = pinnedCase;
      const frame = createGPULocalFrame({origin, kind});
      const [x, y] = frame.project(lngLat[0], lngLat[1]);
      expect(Math.abs(x - xy[0])).toBeLessThan(tolerance ?? getTolerance(xy[0]));
      expect(Math.abs(y - xy[1])).toBeLessThan(tolerance ?? getTolerance(xy[1]));
      if (!skipInverse) {
        const [longitude, latitude] = frame.unproject(xy[0], xy[1]);
        expect(Math.abs(((longitude - lngLat[0] + 540) % 360) - 180)).toBeLessThan(1e-9);
        expect(Math.abs(latitude - lngLat[1])).toBeLessThan(1e-8);
        expect(longitude).toBeGreaterThanOrEqual(-180);
        expect(longitude).toBeLessThanOrEqual(180);
      }
      const roundTrip = frame.unproject(x, y);
      if (!skipInverse) {
        expect(Math.abs(((roundTrip[0] - lngLat[0] + 540) % 360) - 180)).toBeLessThan(1e-9);
      }
      expect(Math.abs(roundTrip[1] - lngLat[1])).toBeLessThan(1e-6);
    });
  }
});

describe('createGPULocalFrame math.gl behavior', () => {
  it('maps the origin to zero and keeps Float32 projectArray output with strides', () => {
    for (const kind of ['azimuthal-equidistant', 'lambert-equal-area'] as const) {
      for (const origin of [
        [8.5417, 47.3769],
        [0, 90],
        [0, -90]
      ] as [number, number][]) {
        const frame = createGPULocalFrame({origin, kind});
        const [x, y] = frame.project(origin[0], origin[1]);
        expect(Math.abs(x)).toBeLessThan(1e-6);
        expect(Math.abs(y)).toBeLessThan(1e-6);
      }
    }
    const frame = createGPULocalFrame({origin: [8.5417, 47.3769]});
    const projected = frame.projectArray([8.5427, 47.3779, 99, 9.5417, 49.3769, 99], 3);
    expect(projected).toBeInstanceOf(Float32Array);
    expect(projected.length).toBe(4);
    expect(projected[0]).toBeCloseTo(75.29684330257996, 3);
    expect(projected[3]).toBeCloseTo(222860.11359247612, -1);
    const unprojected = frame.unprojectArray(projected);
    expect(unprojected).toBeInstanceOf(Float64Array);
    expect(unprojected[0]).toBeCloseTo(8.5427, 5);
    expect(unprojected[3]).toBeCloseTo(49.3769, 5);
  });

  it('returns NaN at the antipode for both kinds, including inside projectArray', () => {
    for (const kind of ['azimuthal-equidistant', 'lambert-equal-area'] as const) {
      const frame = createGPULocalFrame({origin: [10, 20], kind});
      const [x, y] = frame.project(-170, -20);
      expect(Number.isNaN(x) && Number.isNaN(y)).toBe(true);
      const projected = frame.projectArray([10, 20, -170, -20, 11, 21]);
      expect(projected[0]).toBe(0);
      expect(Number.isNaN(projected[2]) && Number.isNaN(projected[3])).toBe(true);
      expect(Number.isFinite(projected[4])).toBe(true);
    }
  });

  it('keeps near-origin offsets at sub-nanometer relative accuracy', () => {
    // Offsets of 1e-3 and 1e-7 degrees against pyproj-verified recorded values.
    const frame = createGPULocalFrame({origin: [151.2093, -33.8688]});
    const [x, y] = frame.project(151.2093001, -33.8688);
    expect(Math.abs(x - 0.009232703869616882)).toBeLessThan(1e-9);
    expect(Math.abs(y)).toBeLessThan(1e-9);
  });

  it('agrees with a direct math.gl projection for custom sphere radii', () => {
    const sphereRadius = 6378137;
    const frame = createGPULocalFrame({origin: [10, 50], sphereRadius});
    const projection = projectionEngine.createProjection({
      from: '+proj=longlat +ellps=WGS84',
      to: `+proj=aeqd +R=${sphereRadius} +lon_0=10 +lat_0=50`
    });
    const expected = projection.projectSync([11, 51]);
    const [x, y] = frame.project(11, 51);
    expect(Math.abs(x - expected[0])).toBeLessThan(1e-6);
    expect(Math.abs(y - expected[1])).toBeLessThan(1e-6);
  });
});

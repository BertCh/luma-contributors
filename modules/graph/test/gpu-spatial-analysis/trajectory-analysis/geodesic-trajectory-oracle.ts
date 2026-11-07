// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getHaversineDistance} from '../geometry-measures/geodesic-oracle';
import fixtureJson from '../../oracles/fixtures/geodesic-trajectory.json';
import {decodeOracleFixture} from '../../oracles/oracle-fixture';

/**
 * Great-circle oracle for the movement operators.
 *
 * `GEODESIC_ORACLE_CASES` are the pyproj references on the sphere `Geod(a=6371008.8, f=0)`
 * (`GPU_EARTH_MEAN_RADIUS_METERS`) written by `test/oracles/geodesic-trajectory.py`. The CPU
 * functions below are binary64 twins of the WGSL helpers; the node spec checks them against the
 * reference values and the GPU specs compare the kernels with them.
 */

export const GEODESIC_ORACLE_RADIUS = 6371008.8;

export type GeodesicOracleCase = {
  name: string;
  a: [number, number];
  b: [number, number];
  /** pyproj `Geod.inv` distance in meters. */
  distance: number;
  /** pyproj `Geod.inv` forward azimuth in degrees, clockwise from north. */
  bearingDegrees: number;
  /** pyproj `Geod.fwd` points at fractions of the distance, longitude continuous with `a`. */
  points: {fraction: number; lon: number; lat: number}[];
};

/** One lattice cell size and latitude bound with the pyproj-walked reach. */
export type EncounterReachOracleCase = {
  cellDegrees: number;
  maximumLatitude: number;
  maximumDistance: number;
  observedLongitudeReachDegrees: number;
};

const fixture = decodeOracleFixture<{
  generator: string;
  versions: Record<string, string>;
  tolerances: Record<'distanceRelative' | 'bearingDegrees' | 'pointDegrees', number>;
  reference: {
    radius: number;
    cases: GeodesicOracleCase[];
    encounterReach: EncounterReachOracleCase[];
  };
}>(fixtureJson);

export const GEODESIC_ORACLE_CASES: GeodesicOracleCase[] = fixture.reference.cases;

/** Tolerances recorded by the generator, with their derivation in the generator docstring. */
export const GEODESIC_ORACLE_TOLERANCES = fixture.tolerances;

/**
 * Largest searchable distance of the encounter lattice from pyproj: for each case the claimed
 * angle `asin(sin(cell) * cos(L))` was walked in 7201 azimuths from latitude +-L, and the observed
 * longitude and latitude reach stayed within one cell (`observedLongitudeReachDegrees <=
 * cellDegrees`), nearly tight. `maximumDistance` includes the `1 - 1e-6` scale.
 */
export const ENCOUNTER_REACH_ORACLE_CASES: EncounterReachOracleCase[] =
  fixture.reference.encounterReach;

const DEGREES = Math.PI / 180;

export {
  getCentralAngle,
  getInitialBearingDegrees,
  interpolateGreatCircle
} from '../geometry-measures/geodesic-oracle';

/** Great-circle distance in the units of `radius`. */
export function getGreatCircleDistance(
  a: readonly number[],
  b: readonly number[],
  radius: number = GEODESIC_ORACLE_RADIUS
): number {
  return getHaversineDistance(a, b, radius);
}

/** Conservative largest searchable encounter distance of the lattice, see `GPUTrajectoryEncounters`. */
export function getEncounterMaximumDistance(
  cellDegrees: number,
  maximumLatitude: number,
  radius: number = GEODESIC_ORACLE_RADIUS
): number {
  return (
    radius *
    Math.asin(Math.sin(cellDegrees * DEGREES) * Math.cos(maximumLatitude * DEGREES)) *
    (1 - 1e-6)
  );
}

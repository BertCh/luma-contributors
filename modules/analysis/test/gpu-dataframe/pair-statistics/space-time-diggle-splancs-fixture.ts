// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Typed access to `test/oracles/fixtures/space-time-diggle-splancs-port.json`, the NumPy port of
 * the splancs 2.01-45 source (`stkhat`, `stsecal`, `stmctest`, `stdiagn`). It is a port, not a
 * run of R: see the generator `test/oracles/space-time-diggle-splancs-port.py`.
 */

import fixtureJson from '../../oracles/fixtures/space-time-diggle-splancs-port.json';
import type {SpaceTimeKReferenceOptions} from './space-time-k-function-reference';

/** One labelling of the events: `stkhat`, the `stmctest` statistic, `stsecal` and `stdiagn`. */
export type SplancsAnalysis = {
  spatialK: number[];
  temporalK: number[];
  /** Distance-major, like the `GPUSpaceTimeKFunction` outputs. */
  k: number[];
  d: number[];
  /** `sum(kst - outer(ks, kt))`, the `stmctest` statistic. */
  statistic: number;
  /** `stsecal`; absent with fewer than four events. */
  standardError: number[];
  /** `stdiagn` `R = D / stsecal`. */
  standardisedResidual: number[];
};

export type SplancsFixture = {
  provenance: string;
  versions: {python: string; numpy: string; shapely: string; splancsSource: string};
  bounds: [number, number, number, number];
  timeLimits: [number, number];
  maximumDistance: number;
  distanceClassCount: number;
  maximumTimeLag: number;
  timeClassCount: number;
  positions: number[];
  times: number[];
  observed: SplancsAnalysis;
  relabellings: (SplancsAnalysis & {times: number[]})[];
  monteCarlo: {
    permutationCount: number;
    statistics: number[];
    mean: number;
    standardDeviation: number;
    greaterCount: number;
    pseudoPGreater: number;
  };
  /** Whole-day times in `[0, 30]`: the tie rules of `stkhat.f` act on many pairs. */
  integerTimes: {
    timeLimits: [number, number];
    maximumDistance: number;
    distanceClassCount: number;
    positions: number[];
    /** The observed times, then three relabellings of them. */
    labellingTimes: number[][];
    cases: {
      maximumTimeLag: number;
      timeClassCount: number;
      /** Ordered pairs whose lag exactly reaches a time limit (weight 2 in splancs). */
      edgeTieCount: number;
      /** Ordered pairs at exactly the largest lag (dropped by splancs). */
      maximumLagTieCount: number;
      /** Ordered pairs on a class break below the largest (closed above). */
      interiorBreakTieCount: number;
      /** One analysis per entry of `labellingTimes`. */
      labellings: SplancsAnalysis[];
    }[];
  };
  temporalTie: SplancsAnalysis & {
    positions: number[];
    times: number[];
    timeLimits: [number, number];
    distanceBreaks: number[];
    timeBreaks: number[];
  };
  checks: {
    boundaryMargin: number;
    shapelyCircleFractionMaximumDifferenceRectangle: number;
    shapelyCircleFractionMaximumDifferenceLShape: number;
    stsecalVarianceMaximumRelativeErrorAgainstEnumeration: number;
  };
};

export const SPLANCS_FIXTURE = fixtureJson as unknown as SplancsFixture;

/**
 * Options of the binary64 K reference for the continuous-time fixture scene with the given event
 * times, with the splancs rules at exact ties.
 */
export function getSplancsReferenceOptions(times: number[]): SpaceTimeKReferenceOptions {
  return {
    tieRule: 'splancs',
    scene: {positions: SPLANCS_FIXTURE.positions, times},
    bounds: SPLANCS_FIXTURE.bounds,
    timeWindow: SPLANCS_FIXTURE.timeLimits,
    maximumDistance: SPLANCS_FIXTURE.maximumDistance,
    maximumTimeLag: SPLANCS_FIXTURE.maximumTimeLag,
    distanceClassCount: SPLANCS_FIXTURE.distanceClassCount,
    timeClassCount: SPLANCS_FIXTURE.timeClassCount,
    edgeCorrection: 'isotropic'
  };
}

/** Options of the binary64 K reference for one lag grid and labelling of the whole-day scene. */
export function getSplancsIntegerTimeReferenceOptions(
  caseIndex: number,
  labellingIndex: number,
  tieRule: 'stpp' | 'splancs' = 'splancs'
): SpaceTimeKReferenceOptions {
  const scene = SPLANCS_FIXTURE.integerTimes;
  return {
    tieRule,
    scene: {positions: scene.positions, times: scene.labellingTimes[labellingIndex]},
    bounds: SPLANCS_FIXTURE.bounds,
    timeWindow: scene.timeLimits,
    maximumDistance: scene.maximumDistance,
    maximumTimeLag: scene.cases[caseIndex].maximumTimeLag,
    distanceClassCount: scene.distanceClassCount,
    timeClassCount: scene.cases[caseIndex].timeClassCount,
    edgeCorrection: 'isotropic'
  };
}

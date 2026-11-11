// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * The binary64 TypeScript reference of the space-time K family against the port of the splancs
 * source (`test/oracles/space-time-diggle-splancs-port.py`; a port, not a run of R). The GPU
 * specs use the TypeScript reference for the permutations the kernel draws itself, so this spec
 * ties that reference to what splancs `stkhat` and `stmctest` compute (math review 2026-11-10,
 * dataframe-10).
 */

import {expect, it} from 'vitest';
import fixtureJson from '../../oracles/fixtures/space-time-diggle-splancs-port.json';
import {
  getSpaceTimeKReference,
  type SpaceTimeKReferenceOptions
} from './space-time-k-function-reference';
import {
  getSplancsIntegerTimeReferenceOptions,
  getSplancsReferenceOptions,
  SPLANCS_FIXTURE,
  type SplancsAnalysis
} from './space-time-diggle-splancs-fixture';

function expectClose(actual: number[], expected: number[], label: string): void {
  const scale = Math.max(...expected.map(Math.abs), 1);
  expect(actual.length, label).toBe(expected.length);
  actual.forEach((value, index) => {
    expect(Math.abs(value - expected[index]), `${label} ${index}`).toBeLessThanOrEqual(
      1e-9 * scale
    );
  });
}

function expectReferenceMatchesPort(
  options: SpaceTimeKReferenceOptions,
  port: SplancsAnalysis,
  label: string
): void {
  const reference = getSpaceTimeKReference(options);
  expect(reference.included.length).toBe(options.scene.times.length);
  expectClose(reference.spatialK, port.spatialK, `${label} ks`);
  expectClose(reference.temporalK, port.temporalK, `${label} kt`);
  expectClose(reference.k, port.k, `${label} kst`);
  expectClose(reference.d, port.d, `${label} D`);
  const statistic = reference.d.reduce((sum, value) => sum + value, 0);
  expect(Math.abs(statistic - port.statistic), `${label} statistic`).toBeLessThanOrEqual(
    1e-9 * Math.max(...port.k.map(Math.abs))
  );
}

it('the fixture says what it is: a port of the splancs source, checked, not a run of R', () => {
  expect(fixtureJson.provenance).toMatch(/port of the splancs 2\.01-45 source.*not a run of R/);
  expect(SPLANCS_FIXTURE.versions.numpy).toBe('2.5.4');
  expect(SPLANCS_FIXTURE.versions.splancsSource).toBe('2.01-45');
  const {checks} = SPLANCS_FIXTURE;
  expect(checks.boundaryMargin).toBeGreaterThan(1e-4);
  expect(checks.shapelyCircleFractionMaximumDifferenceRectangle).toBeLessThan(2e-5);
  expect(checks.shapelyCircleFractionMaximumDifferenceLShape).toBeLessThan(2e-5);
  expect(checks.stsecalVarianceMaximumRelativeErrorAgainstEnumeration).toBeLessThan(1e-10);
});

it('the binary64 K reference equals the splancs port for the observed pattern and six relabellings', () => {
  const options = getSplancsReferenceOptions(SPLANCS_FIXTURE.times);
  // Every pair is off the class boundaries and the temporal edge rule, so both tie rules agree.
  expect(getSpaceTimeKReference(options).minimumBoundaryMargin).toBeGreaterThan(1e-5);
  expectReferenceMatchesPort(options, SPLANCS_FIXTURE.observed, 'observed');
  expectReferenceMatchesPort({...options, tieRule: 'stpp'}, SPLANCS_FIXTURE.observed, 'stpp rule');
  expect(SPLANCS_FIXTURE.relabellings.length).toBe(6);
  SPLANCS_FIXTURE.relabellings.forEach((relabelling, index) => {
    expectReferenceMatchesPort(
      getSplancsReferenceOptions(relabelling.times),
      relabelling,
      `relabelling ${index}`
    );
  });
});

it('on whole-day times the reference with the splancs tie rules equals the port; the stpp rules give other numbers', () => {
  const {cases, labellingTimes} = SPLANCS_FIXTURE.integerTimes;
  expect(cases.map(entry => [entry.maximumTimeLag, entry.timeClassCount])).toEqual([
    [8.5, 5],
    [7, 7],
    [13, 13]
  ]);
  expect(labellingTimes.length).toBe(4);
  expect(labellingTimes[0].every(Number.isInteger)).toBe(true);
  // Events on both time limits: their edge radius is 0 and splancs counts every partner double.
  expect(Math.min(...labellingTimes[0])).toBe(0);
  expect(Math.max(...labellingTimes[0])).toBe(30);
  // The pairs each rule acts on, as the port counted them for the observed labelling.
  expect(cases.map(entry => entry.edgeTieCount)).toEqual([134, 128, 201]);
  expect(cases.map(entry => entry.maximumLagTieCount)).toEqual([0, 166, 116]);
  expect(cases.map(entry => entry.interiorBreakTieCount)).toEqual([0, 1114, 2038]);
  cases.forEach((entry, caseIndex) => {
    entry.labellings.forEach((port, labellingIndex) => {
      expectReferenceMatchesPort(
        getSplancsIntegerTimeReferenceOptions(caseIndex, labellingIndex),
        port,
        `lag ${entry.maximumTimeLag} labelling ${labellingIndex}`
      );
    });
    // The stpp rules (closed window, last class closed) are those of GPUSpaceTimeKFunction.
    const stpp = getSpaceTimeKReference(
      getSplancsIntegerTimeReferenceOptions(caseIndex, 0, 'stpp')
    );
    const lastLag = entry.timeClassCount - 1;
    expect(
      Math.abs(stpp.temporalK[lastLag] - entry.labellings[0].temporalK[lastLag]),
      `lag ${entry.maximumTimeLag}: K_t differs between the rules`
    ).toBeGreaterThan(0.01);
    const statistic = stpp.d.reduce((sum, value) => sum + value, 0);
    expect(Math.abs(statistic - entry.labellings[0].statistic)).toBeGreaterThan(1);
  });
});

it('the stmctest statistic of the port is the plain sum of D, and R is D over the stsecal error', () => {
  const {observed} = SPLANCS_FIXTURE;
  const sum = observed.d.reduce((total, value) => total + value, 0);
  expect(observed.statistic).toBeCloseTo(sum, 9);
  // For contrast: the sum of standardised residuals is another number.
  const standardised = observed.standardisedResidual.reduce((total, value) => total + value, 0);
  expect(Math.abs(standardised - observed.statistic)).toBeGreaterThan(1);
  observed.d.forEach((value, index) => {
    expect(observed.standardisedResidual[index]).toBeCloseTo(
      value / observed.standardError[index],
      9
    );
  });
});

it('splancs counts a lag that exactly reaches a time limit double; stpp does not', () => {
  const tie = SPLANCS_FIXTURE.temporalTie;
  // Hand computation. Events at t = 3 and t = 6 in [0, 10], lag 3, both circles inside the
  // window. splancs: v = 2 for the first event (3 - 0 <= 3) and 1 for the second, so
  // kt = T / (n (n - 1)) (2 + 1) = 10 / 2 * 3 = 15 and kst = A T / 2 * 3 = 80 * 10 / 2 * 3.
  expect(tie.temporalK).toEqual([15]);
  expect(tie.spatialK).toEqual([80]);
  expect(tie.k).toEqual([1200]);
  const options: SpaceTimeKReferenceOptions = {
    scene: {positions: tie.positions, times: tie.times},
    bounds: SPLANCS_FIXTURE.bounds,
    timeWindow: tie.timeLimits,
    maximumDistance: tie.distanceBreaks[0],
    maximumTimeLag: tie.timeBreaks[0],
    distanceClassCount: 1,
    timeClassCount: 1,
    edgeCorrection: 'isotropic'
  };
  const splancs = getSpaceTimeKReference({...options, tieRule: 'splancs'});
  expect(splancs.temporalK).toEqual([15]);
  expect(splancs.k).toEqual([1200]);
  // stpp 2.0-8 stikfunction.f: vij = 1 when ti - tij >= binft and ti + tij <= bsupt. The window
  // is closed, t - lag = 0 is inside [0, 10], so v = 1 for both events.
  const stpp = getSpaceTimeKReference(options);
  expect(stpp.temporalK).toEqual([10]);
  expect(stpp.spatialK).toEqual([80]);
  expect(stpp.k).toEqual([800]);
  // A lag exactly at the largest lag (3 with the limit 3): dropped by splancs, counted by stpp.
  const atLimit = {...options, maximumTimeLag: 3};
  expect(getSpaceTimeKReference({...atLimit, tieRule: 'splancs'}).temporalK).toEqual([0]);
  expect(getSpaceTimeKReference(atLimit).temporalK).toEqual([10]);
});

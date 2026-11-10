// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPU_TRACK_SIMILARITY_STATUS} from '../../../src/gpu-spatial-analysis/track-similarity/index';
import {requireTestDevice} from '../../utils/require-test-device';
import {
  EDIT_MEASURE_FIXTURE,
  runMeasures,
  type EditMeasureCase
} from './track-similarity-edit-measures-harness';

// Oracle: test/oracles/track-similarity-edit-measures.py (MovingPandas 0.23 Hausdorff and DTW;
// EDR, ERP and SSPD from the paper definitions in binary64; versions in the fixture).
const {tolerances} = EDIT_MEASURE_FIXTURE;
const GREAT_CIRCLE = {coordinateSpace: 'longitude-latitude', metric: 'great-circle'} as const;

function expectNear(
  actual: number,
  expected: number,
  absolute: number,
  relative: number,
  label: string
) {
  expect(Math.abs(actual - expected), `${label}: ${actual} vs ${expected}`).toBeLessThanOrEqual(
    absolute + relative * Math.abs(expected)
  );
}

it('GPUTrackSimilarity Hausdorff, DTW and SSPD match the MovingPandas and paper references', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const cases = EDIT_MEASURE_FIXTURE.planar;
  const result = await runMeasures(
    device,
    cases.map(entry => entry.a),
    cases.map(entry => entry.b),
    ['hausdorff', 'dynamicTimeWarping', 'symmetricSegmentPathDistance']
  );
  cases.forEach((entry, index) => {
    expect(result.status[index], entry.name).toBe(0);
    expectNear(
      result.hausdorff[index],
      entry.hausdorff!,
      tolerances.hausdorffAbsolutePlanar,
      tolerances.hausdorffRelative,
      `${entry.name} hausdorff`
    );
    const steps = entry.a.length + entry.b.length - 1;
    expectNear(
      result.dynamicTimeWarping[index],
      entry.dynamicTimeWarping!,
      steps * tolerances.dynamicTimeWarpingAbsolutePlanar,
      steps * tolerances.dynamicTimeWarpingRelativePerStep,
      `${entry.name} dynamic time warping`
    );
    expectNear(
      result.symmetricSegmentPathDistance[index],
      entry.sspd!,
      tolerances.sspdAbsolutePlanar,
      tolerances.sspdRelative,
      `${entry.name} sspd`
    );
  });
  // Teeth: the cases separate the measures (a constant output would fail).
  expect(new Set(cases.map(entry => entry.sspd)).size).toBeGreaterThan(10);
}, 120000);

async function expectEditMeasures(
  cases: readonly EditMeasureCase[],
  isGreatCircle: boolean
): Promise<void> {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const spatialContext = isGreatCircle ? GREAT_CIRCLE : undefined;
  // EDR tolerance and ERP gap point are compile-time: one run per distinct value.
  const toleranceValues = [...new Set(cases.flatMap(entry => entry.edr.map(edr => edr.tolerance)))];
  for (const tolerance of toleranceValues) {
    const subset = cases.filter(entry => entry.edr.some(edr => edr.tolerance === tolerance));
    const result = await runMeasures(
      device,
      subset.map(entry => entry.a),
      subset.map(entry => entry.b),
      ['editDistanceOnRealSequence'],
      {spatialContext, editDistanceTolerance: tolerance}
    );
    subset.forEach((entry, index) => {
      const expected = entry.edr.find(edr => edr.tolerance === tolerance)!.distance;
      expect(result.status[index], entry.name).toBe(0);
      // A count of edit operations: exact.
      expect(result.editDistanceOnRealSequence[index], `${entry.name} edr ${tolerance}`).toBe(
        expected
      );
    });
  }
  const gapKeys = [...new Set(cases.flatMap(entry => entry.erp.map(erp => erp.gap.join(','))))];
  for (const key of gapKeys) {
    const gap = key.split(',').map(Number) as [number, number];
    const subset = cases.filter(entry => entry.erp.some(erp => erp.gap.join(',') === key));
    const result = await runMeasures(
      device,
      subset.map(entry => entry.a),
      subset.map(entry => entry.b),
      ['editDistanceWithRealPenalty'],
      {spatialContext, editDistanceGapPoint: gap}
    );
    subset.forEach((entry, index) => {
      const expected = entry.erp.find(erp => erp.gap.join(',') === key)!.distance;
      const steps = entry.a.length + entry.b.length;
      expectNear(
        result.editDistanceWithRealPenalty[index],
        expected,
        steps *
          (isGreatCircle ? tolerances.erpAbsoluteMetersPerStep : tolerances.erpAbsolutePlanar),
        steps * tolerances.erpRelativePerStep,
        `${entry.name} erp ${key}`
      );
    });
  }
}

it('GPUTrackSimilarity EDR and ERP match the planar references', async () => {
  await expectEditMeasures(EDIT_MEASURE_FIXTURE.planar, false);
  // Teeth: the cases include both zero and large EDR, so a constant kernel would fail.
  const values = EDIT_MEASURE_FIXTURE.planar.flatMap(entry => entry.edr.map(edr => edr.distance));
  expect(Math.min(...values)).toBe(0);
  expect(Math.max(...values)).toBeGreaterThan(20);
}, 120000);

it('GPUTrackSimilarity EDR and ERP match the great-circle references', async () => {
  await expectEditMeasures(EDIT_MEASURE_FIXTURE.greatCircle, true);
}, 120000);

it('GPUTrackSimilarity EDR and ERP cross the 256-lane strip boundary and report the cap', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // 700 vertices against 300: three strips of 256 lanes, so the carried row is exercised. The
  // expected values are the binary64 recurrences of the oracle file.
  const {computeEditDistanceOnRealSequenceOracle, computeEditDistanceWithRealPenaltyOracle} =
    await import('./track-similarity-edit-oracle');
  const long = Array.from({length: 700}, (_, index) => [
    index * 0.3,
    Math.sin(index * 0.05) * 3
  ]) as [number, number][];
  const short = Array.from({length: 300}, (_, index) => [
    index * 0.7 + 0.1,
    Math.sin(index * 0.12) * 3 + 0.2
  ]) as [number, number][];
  for (const [first, second] of [
    [long, short],
    [short, long]
  ]) {
    const edr = await runMeasures(device, [first], [second], ['editDistanceOnRealSequence'], {
      editDistanceTolerance: 0.8,
      maximumFrechetVertexCount: 512
    });
    expect(edr.editDistanceOnRealSequence[0]).toBe(
      computeEditDistanceOnRealSequenceOracle(first, second, 0.8)
    );
    const erp = await runMeasures(device, [first], [second], ['editDistanceWithRealPenalty'], {
      editDistanceGapPoint: [5, 1],
      maximumFrechetVertexCount: 512
    });
    expectNear(
      erp.editDistanceWithRealPenalty[0],
      computeEditDistanceWithRealPenaltyOracle(first, second, [5, 1]),
      0.05,
      1e-5,
      'strip erp'
    );
  }
  // The shorter track above the cap: NaN and the cap status; SSPD and Hausdorff are uncapped.
  const capped = await runMeasures(
    device,
    [long],
    [long],
    ['editDistanceOnRealSequence', 'editDistanceWithRealPenalty', 'symmetricSegmentPathDistance'],
    {maximumFrechetVertexCount: 256}
  );
  expect(capped.status[0] & GPU_TRACK_SIMILARITY_STATUS.frechetCapExceeded).toBeGreaterThan(0);
  expect(capped.editDistanceOnRealSequence[0]).toBeNaN();
  expect(capped.editDistanceWithRealPenalty[0]).toBeNaN();
  expect(capped.symmetricSegmentPathDistance[0]).toBeCloseTo(0, 6);
}, 120000);

it('GPUTrackSimilarity SSPD is bitwise repeatable (fixed-order sums)', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const track = (seed: number) =>
    Array.from({length: 1500}, (_, index) => [
      index * 0.01 + Math.sin(index * seed),
      Math.cos(index * seed * 1.3)
    ]);
  const runOnce = () =>
    runMeasures(
      device,
      [track(0.31), track(0.7)],
      [track(0.5), track(0.9)],
      ['symmetricSegmentPathDistance']
    );
  const [first, second] = [await runOnce(), await runOnce()];
  expect(Array.from(second.symmetricSegmentPathDistance)).toEqual(
    Array.from(first.symmetricSegmentPathDistance)
  );
  expect(first.symmetricSegmentPathDistance[0]).toBeGreaterThan(0);
}, 120000);

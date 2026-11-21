// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GraphRig} from '../areal-interpolation/areal-interpolation-harness';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {GPUSpatialScanStatistic} from '../../../src/gpu-spatial-analysis/scan-statistics/gpu-spatial-scan-statistic';
import {getGPUSpatialScanStatisticParameterValues} from '../../../src/gpu-spatial-analysis/scan-statistics/scan-statistic-parameters';

for (const exposeStatus of [false, true])
  it(`suppresses zero-exposure Poisson inference and recovers, status=${exposeStatus} (NUM-01)`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const rig = new GraphRig(device);
    const cases = rig.output('uint32', 2);
    cases.buffer.write(Uint32Array.of(1, 2));
    const statistics = rig.output('float32', 20);
    const clusterStatistics = rig.output('float32', 8);
    const summary = rig.output('uint32', 4);
    const invalidCount = rig.output('uint32', 1);
    const clusterIndices = rig.output('uint32', 4);
    try {
      rig.run(
        new GPUSpatialScanStatistic({
          positions: rig.input(Float32Array.of(0, 0, 1, 0), 'float32x2', 2),
          cases: cases.view,
          baseline: rig.input(Float32Array.of(0, 1), 'float32'),
          timeBuckets: 1,
          maximumWindowZones: 2,
          maximumPermutations: 19,
          maximumClusters: 1,
          parameters: rig.input(
            getGPUSpatialScanStatisticParameterValues({seed: 0, permutationCount: 19}),
            'uint32'
          ),
          output: {
            clusterIndices: clusterIndices.view,
            clusterStatistics: clusterStatistics.view,
            statistics: statistics.view,
            summary: summary.view,
            ...(exposeStatus ? {status: {invalidCount: invalidCount.view}} : {})
          }
        })
      );
      for (const invalid of [true, false, true]) {
        cases.buffer.write(invalid ? Uint32Array.of(1, 2) : Uint32Array.of(0, 3));
        rig.resubmit();
        // Under the conditional null a positive case at zero exposure has probability zero.
        if (exposeStatus) expect(await invalidCount.readUint32()).toEqual([Number(invalid)]);
        expect(await summary.readUint32()).toEqual([0, 3, invalid ? 0 : 19, 0]);
        expect(await clusterIndices.readUint32()).toEqual([0, 0, 0, 0]);
        const maxima = await statistics.readFloat32();
        const clusters = await clusterStatistics.readFloat32();
        expect(maxima).toHaveLength(20);
        expect(clusters).toHaveLength(8);
        if (invalid) {
          for (const value of [...maxima, ...clusters]) expect(value).toBeNaN();
        } else {
          expect(maxima).toEqual(new Array(20).fill(0));
          expect(clusters).toEqual(new Array(8).fill(0));
        }
      }
    } finally {
      rig.destroy();
    }
  });

it('parallel margins retain exact totals and reset center overflow each frame (W2-09)', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new GraphRig(device);
  const cases = rig.output('uint32', 6);
  cases.buffer.write(Uint32Array.of(1, 2, 3, 4, 5, 6));
  const settings = (maximumPopulationFraction: number) =>
    getGPUSpatialScanStatisticParameterValues({
      seed: 3,
      permutationCount: 2,
      maximumPopulationFraction,
      windowShape: 'nearest'
    });
  const parameters = rig.output('uint32', 8);
  parameters.buffer.write(settings(1));
  const summary = rig.output('uint32', 4);
  const statistics = rig.output('float32', 3);
  const zoneStatistics = rig.output('float32', 3);
  try {
    rig.run(
      new GPUSpatialScanStatistic({
        model: 'space-time-permutation',
        parallelDraws: true,
        positions: rig.input(Float32Array.of(0, 0, 1, 0, 2, 0), 'float32x2', 3),
        cases: cases.view,
        parameters: parameters.view,
        timeBuckets: 2,
        maximumWindowZones: 1,
        maximumPermutations: 2,
        maximumClusters: 1,
        output: {
          clusterIndices: rig.output('uint32', 4).view,
          clusterStatistics: rig.output('float32', 8).view,
          statistics: statistics.view,
          zoneStatistics: zoneStatistics.view,
          summary: summary.view
        }
      })
    );
    // Sum 1..6 is 21. At fraction 1 each center's next zone fits beyond the one-zone capacity.
    expect(await summary.readUint32()).toEqual([1, 21, 2, 3]);
    // /usr/bin/python3: c*log(c/e)+(21-c)*log((21-c)/(21-e));
    // zone margins [3,7,11], bucket margins [9,12], e = zoneMargin * bucketMargin / 21.
    const expectedScores = [0.02471400553768044, 0, 0.011009266524253214];
    const scores = await zoneStatistics.readFloat32();
    expect(scores).toHaveLength(3);
    for (let row = 0; row < 3; row++) {
      expect(Number.isFinite(scores[row])).toBe(true);
      expect(Math.abs(scores[row] - expectedScores[row])).toBeLessThan(1e-6);
    }
    const initialMaxima = await statistics.readFloat32();
    expect(initialMaxima).toHaveLength(3);
    for (const value of initialMaxima) {
      expect(Number.isFinite(value)).toBe(true);
      expect(value).toBeGreaterThan(0);
    }
    const observed = initialMaxima[0];
    expect(Number.isFinite(observed)).toBe(true);
    expect(Math.abs(observed - expectedScores[0])).toBeLessThan(1e-6);

    cases.buffer.write(Uint32Array.of(1, 1, 1, 1, 1, 1));
    parameters.buffer.write(settings(0.01));
    rig.resubmit();
    // The cap floor(6 * .01) is zero: every positive-population zone exceeds it.
    const suppressedWindows = await summary.readUint32();
    expect(suppressedWindows).toHaveLength(4);
    expect(suppressedWindows).toEqual([0, 6, 2, 0]);
    expect(await statistics.readFloat32()).toEqual([0, 0, 0]);
    cases.buffer.write(Uint32Array.of(1, 2, 3, 4, 5, 6));
    parameters.buffer.write(settings(1));
    rig.resubmit();
    expect(await summary.readUint32()).toEqual([1, 21, 2, 3]);
    expect(await statistics.readFloat32()).toEqual(initialMaxima);
  } finally {
    rig.destroy();
  }
});

it('reports numerical exposure loss instead of certifying a negative scan (review earlier-fix note)', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new GraphRig(device);
  const statistics = rig.output('float32', 2);
  const summary = rig.output('uint32', 4);
  const invalidCount = rig.output('uint32', 1);
  try {
    rig.run(
      new GPUSpatialScanStatistic({
        positions: rig.input(Float32Array.of(0, 0, 1, 0), 'float32x2', 2),
        cases: rig.input(Uint32Array.of(1, 0), 'uint32'),
        baseline: rig.input(Float32Array.of(2 ** -70, 2 ** 70), 'float32'),
        maximumPermutations: 1,
        maximumClusters: 1,
        parameters: rig.input(
          getGPUSpatialScanStatisticParameterValues({seed: 0, permutationCount: 1}),
          'uint32'
        ),
        output: {
          clusterIndices: rig.output('uint32', 4).view,
          clusterStatistics: rig.output('float32', 8).view,
          statistics: statistics.view,
          summary: summary.view,
          status: {invalidCount: invalidCount.view}
        }
      })
    );
    // Its exact positive LLR is ln(1 + 2^140) ~= 97.04060527839235, not zero.
    // This route promises normal-float32 exposure weights, so it reports lost precision instead.
    expect(await invalidCount.readUint32()).toEqual([1]);
    expect(await summary.readUint32()).toEqual([0, 1, 0, 0]);
    const maxima = await statistics.readFloat32();
    expect(maxima).toHaveLength(2);
    for (const value of maxima) expect(value).toBeNaN();
  } finally {
    rig.destroy();
  }
});

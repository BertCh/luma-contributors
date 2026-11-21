// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GraphRig} from '../areal-interpolation/areal-interpolation-harness';
import {GPUSpatialScanStatistic} from '../../../src/gpu-spatial-analysis/scan-statistics/gpu-spatial-scan-statistic';
import {getGPUSpatialScanStatisticParameterValues} from '../../../src/gpu-spatial-analysis/scan-statistics/scan-statistic-parameters';
import {runWGSLModuleKernel} from '../../utils/wgsl-module-harness';
import {SCAN_DRAW_INTERVAL_WGSL} from '../../../src/gpu-spatial-analysis/scan-statistics/scan-draw-interval';
import {requireTestDevice} from '../../utils/require-test-device';

for (const populationOverflow of [false, true])
  it(`scan rejects ${populationOverflow ? 'Bernoulli population' : 'case count'} overflow without wrapped inference`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const rig = new GraphRig(device);
    const summary = rig.output('uint32', 4);
    const invalidCount = rig.output('uint32', 1);
    const statistics = rig.output('float32', 2);
    const clusterStatistics = rig.output('float32', 8);
    try {
      rig.run(
        new GPUSpatialScanStatistic({
          model: populationOverflow ? 'bernoulli' : 'poisson',
          positions: rig.input(Float32Array.of(0, 0, 1, 0), 'float32x2', 2),
          cases: rig.input(Uint32Array.of(populationOverflow ? 1 : 0xfffffffe, 2), 'uint32'),
          ...(populationOverflow
            ? {controls: rig.input(Uint32Array.of(0xffffffff, 1), 'uint32')}
            : {baseline: rig.input(Float32Array.of(1, 1), 'float32')}),
          timeBuckets: 1,
          maximumWindowZones: 2,
          maximumPermutations: 1,
          maximumClusters: 1,
          parameters: rig.input(
            getGPUSpatialScanStatisticParameterValues({
              seed: 1,
              permutationCount: 1,
              maximumPopulationFraction: 0.5,
              maximumTimeBuckets: 1
            }),
            'uint32'
          ),
          output: {
            summary: summary.view,
            statistics: statistics.view,
            clusterIndices: rig.output('uint32', 4).view,
            clusterStatistics: clusterStatistics.view,
            status: {invalidCount: invalidCount.view}
          }
        })
      );
      expect(await invalidCount.readUint32()).toEqual([1]);
      expect((await summary.readUint32())[0]).toBe(0);
      for (const value of await statistics.readFloat32()) expect(value).toBeNaN();
    } finally {
      rig.destroy();
    }
  });

it('scan clears stale clusters across valid, overflowing and valid encodings', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new GraphRig(device);
  const cases = rig.output('uint32', 2);
  const summary = rig.output('uint32', 4);
  const indices = rig.output('uint32', 4);
  const statistics = rig.output('float32', 2);
  const clusters = rig.output('float32', 8);
  const invalid = rig.output('uint32', 1);
  const zones = rig.output('float32', 2);
  const baseline = rig.output('float32', 2);
  cases.buffer.write(Uint32Array.of(10, 0));
  baseline.buffer.write(Float32Array.of(1, 1));
  try {
    rig.run(
      new GPUSpatialScanStatistic({
        model: 'poisson',
        positions: rig.input(Float32Array.of(0, 0, 1, 0), 'float32x2', 2),
        cases: cases.view,
        baseline: baseline.view,
        timeBuckets: 1,
        maximumWindowZones: 2,
        maximumPermutations: 1,
        maximumClusters: 1,
        parameters: rig.input(
          getGPUSpatialScanStatisticParameterValues({
            seed: 1,
            permutationCount: 1,
            maximumPopulationFraction: 0.5,
            maximumTimeBuckets: 1
          }),
          'uint32'
        ),
        output: {
          summary: summary.view,
          clusterIndices: indices.view,
          clusterStatistics: clusters.view,
          statistics: statistics.view,
          zoneStatistics: zones.view,
          status: {invalidCount: invalid.view}
        }
      })
    );
    const firstIndices = await indices.readUint32();
    const firstClusters = await clusters.readFloat32();
    expect((await summary.readUint32())[0]).toBe(1);
    expect(await invalid.readUint32()).toEqual([0]);
    // Ten cases in one half of the baseline: Poisson LLR = 10 log(2).
    expect((await statistics.readFloat32())[0]).toBeCloseTo(10 * Math.log(2), 5);
    cases.buffer.write(Uint32Array.of(0xfffffffe, 2));
    rig.resubmit();
    expect(await invalid.readUint32()).toEqual([1]);
    expect((await summary.readUint32())[0]).toBe(0);
    expect(await indices.readUint32()).toEqual([0, 0, 0, 0]);
    for (const value of [
      ...(await statistics.readFloat32()),
      ...(await clusters.readFloat32()),
      ...(await zones.readFloat32())
    ])
      expect(value).toBeNaN();
    cases.buffer.write(Uint32Array.of(10, 0));
    rig.resubmit();
    expect(await invalid.readUint32()).toEqual([0]);
    expect((await summary.readUint32())[0]).toBe(1);
    expect(await indices.readUint32()).toEqual(firstIndices);
    expect(await clusters.readFloat32()).toEqual(firstClusters);
    expect((await statistics.readFloat32())[0]).toBeCloseTo(10 * Math.log(2), 5);
    const firstStatistics = await statistics.readFloat32();
    for (const scale of [1e-30, 1e30, 1e38, 3e38]) {
      baseline.buffer.write(Float32Array.of(scale, scale));
      rig.resubmit();
      expect(await invalid.readUint32()).toEqual([0]);
      expect((await summary.readUint32())[0], `uniform baseline scale ${scale}`).toBe(1);
      expect(await indices.readUint32()).toEqual(firstIndices);
      expect(await clusters.readFloat32()).toEqual(firstClusters);
      expect(await statistics.readFloat32()).toEqual(firstStatistics);
      expect((await clusters.readFloat32())[2]).toBe(5);
      baseline.buffer.write(Float32Array.of(scale / 2, scale));
      rig.resubmit();
      expect(await invalid.readUint32()).toEqual([0]);
      expect((await summary.readUint32())[0], `unequal baseline scale ${scale}`).toBe(1);
      expect((await clusters.readFloat32())[2]).toBeCloseTo(10 / 3, 5);
      expect((await statistics.readFloat32())[0]).toBeCloseTo(10 * Math.log(3), 5);
    }
    baseline.buffer.write(Uint32Array.of(0x00000001, 0x00000001));
    rig.resubmit();
    expect((await summary.readUint32())[0], 'equal minimum positive subnormal exposures').toBe(1);
    expect((await clusters.readFloat32())[2]).toBe(5);
    expect((await statistics.readFloat32())[0]).toBeCloseTo(10 * Math.log(2), 5);
  } finally {
    rig.destroy();
  }
});

it('Poisson scan reports a negative or non-finite baseline cell as invalid (PTS-107)', async () => {
  // Before the fix the normaliser masked the sign bit and took NaN or infinity as the largest
  // baseline: [1, 1, NaN, 1] normalised to [0, 0, 1.5, 0], -1 to 1, and invalidCount stayed 0.
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new GraphRig(device);
  const summary = rig.output('uint32', 4);
  const statistics = rig.output('float32', 2);
  const clusters = rig.output('float32', 8);
  const invalid = rig.output('uint32', 1);
  const baseline = rig.output('float32', 2);
  baseline.buffer.write(Float32Array.of(1, 1));
  try {
    rig.run(
      new GPUSpatialScanStatistic({
        model: 'poisson',
        positions: rig.input(Float32Array.of(0, 0, 1, 0), 'float32x2', 2),
        cases: rig.input(Uint32Array.of(10, 0), 'uint32'),
        baseline: baseline.view,
        timeBuckets: 1,
        maximumWindowZones: 2,
        maximumPermutations: 1,
        maximumClusters: 1,
        parameters: rig.input(
          getGPUSpatialScanStatisticParameterValues({
            seed: 1,
            permutationCount: 1,
            maximumPopulationFraction: 0.5,
            maximumTimeBuckets: 1
          }),
          'uint32'
        ),
        output: {
          summary: summary.view,
          clusterIndices: rig.output('uint32', 4).view,
          clusterStatistics: clusters.view,
          statistics: statistics.view,
          status: {invalidCount: invalid.view}
        }
      })
    );
    expect(await invalid.readUint32()).toEqual([0]);
    expect((await statistics.readFloat32())[0]).toBeCloseTo(10 * Math.log(2), 5);
    for (const values of [
      [Number.NaN, 1],
      [1, Number.NaN],
      [Number.POSITIVE_INFINITY, 1],
      [1, Number.NEGATIVE_INFINITY],
      [-1, 1],
      [1, -1e-30]
    ]) {
      baseline.buffer.write(Float32Array.from(values));
      rig.resubmit();
      expect(await invalid.readUint32(), `baseline ${values}`).toEqual([1]);
      expect((await summary.readUint32())[0], `baseline ${values}`).toBe(0);
      for (const value of await statistics.readFloat32()) expect(value).toBeNaN();
    }
    // A valid frame afterwards is valid again: the invalid word is cleared every frame.
    baseline.buffer.write(Float32Array.of(1, 1));
    rig.resubmit();
    expect(await invalid.readUint32()).toEqual([0]);
    expect((await summary.readUint32())[0]).toBe(1);
    expect((await statistics.readFloat32())[0]).toBeCloseTo(10 * Math.log(2), 5);
  } finally {
    rig.destroy();
  }
});

it('Poisson binary baseline scale admits an exact half-population boundary', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new GraphRig(device);
  const summary = rig.output('uint32', 4);
  const indices = rig.output('uint32', 4);
  const clusters = rig.output('float32', 8);
  const statistics = rig.output('float32', 2);
  try {
    rig.run(
      new GPUSpatialScanStatistic({
        positions: rig.input(Float32Array.of(0, 0, 1, 0, 2, 0, 3, 0), 'float32x2', 4),
        cases: rig.input(Uint32Array.of(10, 0, 0, 0), 'uint32'),
        baseline: rig.input(Float32Array.of(6, 1, 4, 1), 'float32'),
        timeBuckets: 1,
        maximumWindowZones: 1,
        maximumClusters: 1,
        maximumPermutations: 1,
        parameters: rig.input(
          getGPUSpatialScanStatisticParameterValues({
            seed: 1,
            permutationCount: 1,
            maximumPopulationFraction: 0.5
          }),
          'uint32'
        ),
        output: {
          summary: summary.view,
          clusterIndices: indices.view,
          clusterStatistics: clusters.view,
          statistics: statistics.view
        }
      })
    );
    expect((await summary.readUint32())[0]).toBe(1);
    expect((await indices.readUint32())[0]).toBe(0);
    expect((await clusters.readFloat32())[2]).toBe(5);
    expect((await statistics.readFloat32())[0]).toBeCloseTo(10 * Math.log(2), 5);
  } finally {
    rig.destroy();
  }
});

it('scan draw intervals cover accepted uint32 boundary totals without drawing cases', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const totals = [0, 1, 1023, 1024, 1025, 0xffffff00, 0xfffffffd, 0xfffffffe];
  const blocks = 1024;
  const actual = await runWGSLModuleKernel(device, {
    id: 'scan-interval-boundaries',
    inputs: {totals: Uint32Array.from(totals)},
    outputLength: totals.length * blocks * 2,
    invocationCount: totals.length * blocks,
    declarations: SCAN_DRAW_INTERVAL_WGSL,
    body: `let domain = index / 1024u; let blockIndex = index % 1024u;
let interval = getScanDrawInterval(totals[totalsOffset + domain], 1024u, blockIndex);
output[outputOffset + index * 2u] = interval.first; output[outputOffset + index * 2u + 1u] = interval.stop;`
  });
  for (const [domain, total] of totals.entries()) {
    let previous = 0;
    let covered = 0;
    for (let blockIndex = 0; blockIndex < blocks; blockIndex++) {
      const offset = (domain * blocks + blockIndex) * 2;
      const [first, stop] = [actual[offset], actual[offset + 1]];
      expect(first).toBe(Math.min(blockIndex * Math.ceil(total / blocks), total));
      expect(stop).toBe(Math.min((blockIndex + 1) * Math.ceil(total / blocks), total));
      expect(first).toBe(previous);
      expect(stop).toBeGreaterThanOrEqual(first);
      expect(stop).toBeLessThanOrEqual(total);
      covered += stop - first;
      previous = stop;
    }
    expect(previous).toBe(total);
    expect(covered).toBe(total);
  }
});

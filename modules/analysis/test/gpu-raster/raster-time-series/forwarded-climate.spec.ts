// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {GPURasterClimateAnomaly} from '../../../src/raster-analysis/raster-time-series/gpu-raster-climate-anomaly';
import {
  GPURasterStreamingAccumulator,
  createGPURasterStreamingAccumulatorEncodingPlan,
  getGPURasterStreamingAccumulatorParameterValues
} from '../../../src/raster-analysis/raster-time-series/gpu-raster-streaming-accumulator';
import {createRasterTimeSeriesHarness} from './raster-time-series-test-utils';
import {
  ADJACENT_CLIMATE_OBSERVATIONS,
  CLIMATE_CASES,
  climateWords
} from './forwarded-climate-fixtures';

for (const standardized of [false, true])
  for (const fixture of CLIMATE_CASES)
    for (const loss of [0, 1]) {
      it(`F2-F4 climate ${fixture.name} sticky status ${loss} standardized=${standardized}`, async () => {
        const device = await getWebGPUTestDevice();
        requireTestDevice(device);
        const harness = createRasterTimeSeriesHarness(device);
        try {
          const result = await harness.run(
            new GPURasterClimateAnomaly({
              cellCount: 1,
              slotCount: 1,
              state: {
                counts: harness.integers([fixture.count]),
                sums: harness.integers(climateWords(fixture.sum, fixture.sumUnit)),
                moments: harness.integers(climateWords(fixture.moment, fixture.momentUnit)),
                status: harness.integers([loss])
              },
              accumulatorParameters: harness.floats([0, 0, fixture.sumUnit, fixture.momentUnit, 2]),
              values: harness.floats(fixture.values),
              slotIndices: harness.integers(fixture.values.map(() => 0)),
              chunkSliceCount: fixture.values.length,
              standardized,
              output: {
                climatologyMean: harness.output('mean', 'float32', 1),
                standardDeviation: harness.output('deviation', 'float32', 1),
                anomaly: harness.output('anomaly', 'float32', fixture.values.length)
              }
            })
          );
          expect(result['mean']).toHaveLength(1);
          expect(result['deviation']).toHaveLength(1);
          expect(result['anomaly']).toHaveLength(fixture.values.length);
          if (loss) {
            for (const name of ['mean', 'deviation', 'anomaly'])
              expect(result[name].every(Number.isNaN)).toBe(true);
          } else {
            expect(result['mean'][0]).toBe(fixture.mean);
            if (Number.isFinite(fixture.deviation) && fixture.deviation !== 0) {
              expect(Number.isFinite(result['deviation'][0])).toBe(true);
              expect(Math.abs(result['deviation'][0] / fixture.deviation - 1)).toBeLessThan(5e-6);
            } else {
              expect(result['deviation'][0]).toBe(fixture.deviation);
            }
            (standardized ? fixture.anomalies : fixture.differences).forEach((expected, index) => {
              const actual = result['anomaly'][index];
              if (Number.isFinite(expected) && expected !== 0) {
                expect(Number.isFinite(actual)).toBe(true);
                expect(Math.abs(actual / expected - 1)).toBeLessThan(5e-6);
              } else {
                expect(actual).toBe(expected);
              }
            });
          }
        } finally {
          harness.destroy();
        }
      });
    }

for (const split of [false, true])
  for (const reversed of [false, true]) {
    it(`F2 centered climate from retained streaming chunks split=${split} reversed=${reversed}`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const harness = createRasterTimeSeriesHarness(device);
      try {
        const [lower, upper] = ADJACENT_CLIMATE_OBSERVATIONS;
        const observations = reversed ? [upper, lower] : [lower, upper];
        const plan = createGPURasterStreamingAccumulatorEncodingPlan(
          {maximumValueMagnitude: upper, maximumStreamSliceCount: 2},
          split ? 1 : 2
        );
        const state = {
          counts: harness.output('counts', 'uint32', 1),
          sums: harness.output('sums', 'uint32', 3),
          moments: harness.output('moments', 'uint32', 3)
        };
        const parameters = (reset: boolean) =>
          harness.floats(getGPURasterStreamingAccumulatorParameterValues(plan, {reset}));
        const firstParameters = parameters(true);
        const writers = [
          new GPURasterStreamingAccumulator({
            id: 'first',
            cellCount: 1,
            chunkSliceCount: split ? 1 : 2,
            values: harness.floats(split ? observations.slice(0, 1) : observations),
            parameters: firstParameters,
            state
          })
        ];
        if (split)
          writers.push(
            new GPURasterStreamingAccumulator({
              id: 'second',
              cellCount: 1,
              chunkSliceCount: 1,
              values: harness.floats(observations.slice(1)),
              parameters: parameters(false),
              state
            })
          );
        const reader = new GPURasterClimateAnomaly({
          cellCount: 1,
          slotCount: 1,
          state,
          accumulatorParameters: firstParameters,
          values: harness.floats(observations),
          slotIndices: harness.integers([0, 0]),
          chunkSliceCount: 2,
          standardized: true,
          output: {anomaly: harness.output('anomaly', 'float32', 2)}
        });
        const result = await harness.run([...writers, reader]);
        const expected = reversed ? [Math.SQRT1_2, -Math.SQRT1_2] : [-Math.SQRT1_2, Math.SQRT1_2];
        expect(result['anomaly']).toHaveLength(expected.length);
        expected.forEach((value, index) => {
          expect(Number.isFinite(result['anomaly'][index])).toBe(true);
          expect(result['anomaly'][index]).toBeCloseTo(value, 5);
        });
      } finally {
        harness.destroy();
      }
    });
  }

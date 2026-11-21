// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {WgslReflect} from 'wgsl_reflect';
import {auditWGSLModule} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {GPURasterClimateAnomaly} from '../../../src/raster-analysis/raster-time-series/gpu-raster-climate-anomaly';
import {executeKernel, runHostMomentProbe} from './streaming-moments-w5.test-utils';
import {
  ADJACENT_CLIMATE_OBSERVATIONS,
  CLIMATE_CASES,
  POWER_BOUNDARY_CLIMATE_CASE,
  POWER_BOUNDARY_CLIMATE_OBSERVATIONS,
  climateWords
} from './forwarded-climate-fixtures';

function runHostClimateProbe(
  inputs: {
    counts: Uint32Array;
    sums: Uint32Array;
    moments: Uint32Array;
    stateStatus: Uint32Array;
    accumulatorParameters: Float32Array;
    values: readonly number[];
  },
  standardized: boolean
) {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    let serial = 0;
    const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
      createTransientView(graph, `climate-${serial++}`, format, length);
    const chunkSliceCount = inputs.values.length;
    const producer = new GPURasterClimateAnomaly({
      cellCount: 1,
      slotCount: 1,
      state: {
        counts: view('uint32', 1),
        sums: view('uint32', 3),
        moments: view('uint32', 3),
        status: view('uint32', 1)
      },
      accumulatorParameters: view('float32', inputs.accumulatorParameters.length),
      values: view('float32', chunkSliceCount),
      slotIndices: view('uint32', chunkSliceCount),
      chunkSliceCount,
      standardized,
      output: {
        climatologyMean: view('float32', 1),
        standardDeviation: view('float32', 1),
        anomaly: view('float32', chunkSliceCount)
      }
    });
    const fields = {
      ...inputs,
      values: Float32Array.from(inputs.values),
      slotIndices: new Uint32Array(chunkSliceCount),
      climatologyMean: new Float32Array(1),
      standardDeviation: new Float32Array(1),
      anomalyOut: new Float32Array(chunkSliceCount)
    };
    for (const kernel of getGPUCommandNodeKernelSources(producer.getCommandNodes(graph))) {
      expect(() => new WgslReflect(kernel.source)).not.toThrow();
      expect(auditWGSLModule(kernel.source, kernel.constants)).toEqual([]);
      const invocationCount = kernel.id.endsWith('-anomaly') ? chunkSliceCount : 1;
      // Replace dispatch metadata and avoid the interpreter's helper-local name collisions.
      const source = kernel.source
        .replace(/ _ = &\w+;/g, '')
        .replace(
          /rasterDispatch\.words\[(\d+)u\]\[(\d+)u\]/g,
          (_match, row, column) =>
            `${Number(row) * 4 + Number(column) === 0 ? invocationCount : 0}u`
        )
        .replace(/fn slot\w+\([\s\S]+?\n\}/g, helper =>
          helper.replace(/\bentry\b/g, 'slotEntry').replace(/\bcount\b/g, 'slotCountValue')
        );
      executeKernel(source, kernel.constants, fields);
    }
    return fields;
  } finally {
    device.destroy();
  }
}

for (const standardized of [false, true])
  for (const fixture of CLIMATE_CASES)
    for (const loss of [0, 1]) {
      it(`F2-F4 climate ${fixture.name} sticky status ${loss} standardized=${standardized}`, () => {
        const fields = runHostClimateProbe(
          {
            counts: Uint32Array.of(fixture.count),
            sums: climateWords(fixture.sum, fixture.sumUnit),
            moments: climateWords(fixture.moment, fixture.momentUnit),
            stateStatus: Uint32Array.of(loss),
            accumulatorParameters: Float32Array.of(0, 0, fixture.sumUnit, fixture.momentUnit, 2),
            values: fixture.values
          },
          standardized
        );
        expect(fields.climatologyMean).toHaveLength(1);
        expect(fields.standardDeviation).toHaveLength(1);
        expect(fields.anomalyOut).toHaveLength(fixture.values.length);
        if (loss) {
          for (const values of [
            fields.climatologyMean,
            fields.standardDeviation,
            fields.anomalyOut
          ])
            expect([...values].every(Number.isNaN)).toBe(true);
        } else {
          expect(fields.climatologyMean[0]).toBe(fixture.mean);
          if (Number.isFinite(fixture.deviation) && fixture.deviation !== 0) {
            expect(Number.isFinite(fields.standardDeviation[0])).toBe(true);
            expect(Math.abs(fields.standardDeviation[0] / fixture.deviation - 1)).toBeLessThan(
              5e-6
            );
          } else {
            expect(fields.standardDeviation[0]).toBe(fixture.deviation);
          }
          const expectedAnomalies = standardized ? fixture.anomalies : fixture.differences;
          expectedAnomalies.forEach((expected, index) => {
            const actual = fields.anomalyOut[index];
            if (Number.isFinite(expected) && expected !== 0) {
              expect(Number.isFinite(actual)).toBe(true);
              expect(Math.abs(actual / expected - 1)).toBeLessThan(5e-6);
            } else {
              expect(actual).toBe(expected);
            }
          });
        }
      });
    }

it('F4 captures and validates optional loss views before graph declaration', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    let serial = 0;
    const view = <Format extends 'float32' | 'uint32'>(format: Format, length: number) =>
      createTransientView(graph, `status-${serial++}`, format, length);
    const state = {counts: view('uint32', 1), sums: view('uint32', 3), status: view('uint32', 1)};
    const props = {
      cellCount: 1,
      slotCount: 1,
      state,
      accumulatorParameters: view('float32', 5),
      output: {climatologyMean: view('float32', 1)}
    };
    for (const status of [view('float32', 1), view('uint32', 0), props.output.climatologyMean]) {
      expect(
        () => new GPURasterClimateAnomaly({...props, state: {...state, status: status as never}})
      ).toThrow(/state.status|read-only and writable|overlap/);
    }
    const capturedStatus = state.status;
    const contributor = new GPURasterClimateAnomaly(props);
    state.status = view('uint32', 0);
    expect(contributor.props.state.status).toBe(capturedStatus);
    expect(() => contributor.getCommandNodes(graph)).not.toThrow();
  } finally {
    device.destroy();
  }
});

for (const split of [false, true])
  for (const reversed of [false, true]) {
    it(`F2 streaming state retains adjacent centered residual split=${split} reversed=${reversed}`, () => {
      const [lower, upper] = ADJACENT_CLIMATE_OBSERVATIONS;
      const observations: [number, number] = reversed ? [upper, lower] : [lower, upper];
      const retained = runHostMomentProbe({
        route: 'streaming',
        observations,
        split,
        maximumStreamSliceCount: 2
      });
      expect([...retained.moments]).toEqual([0.5]);
      expect([...retained.fields.sums]).toEqual([
        ...climateWords(16777225, retained.fields.parameters[2])
      ]);
      const climate = runHostClimateProbe(
        {
          counts: retained.fields.counts,
          sums: retained.fields.sums,
          moments: retained.fields.moments,
          stateStatus: Uint32Array.of(0),
          accumulatorParameters: retained.fields.parameters,
          values: observations
        },
        true
      );
      const expected = reversed ? [Math.SQRT1_2, -Math.SQRT1_2] : [-Math.SQRT1_2, Math.SQRT1_2];
      expect(climate.anomalyOut).toHaveLength(expected.length);
      expected.forEach((value, index) => {
        expect(Number.isFinite(climate.anomalyOut[index])).toBe(true);
        expect(climate.anomalyOut[index]).toBeCloseTo(value, 5);
      });
    });
  }

for (const standardized of [false, true]) {
  it(`B1 centers a power-of-two query against streamed observations standardized=${standardized}`, () => {
    const retained = runHostMomentProbe({
      route: 'streaming',
      observations: [...POWER_BOUNDARY_CLIMATE_OBSERVATIONS],
      split: false,
      maximumStreamSliceCount: 2
    });
    expect(retained.fields.parameters[2]).toBe(POWER_BOUNDARY_CLIMATE_CASE.sumUnit);
    const climate = runHostClimateProbe(
      {
        counts: retained.fields.counts,
        sums: retained.fields.sums,
        moments: retained.fields.moments,
        stateStatus: Uint32Array.of(0),
        accumulatorParameters: retained.fields.parameters,
        values: POWER_BOUNDARY_CLIMATE_CASE.values
      },
      standardized
    );
    const expected = standardized
      ? POWER_BOUNDARY_CLIMATE_CASE.anomalies[0]
      : POWER_BOUNDARY_CLIMATE_CASE.differences[0];
    expect(climate.anomalyOut).toHaveLength(1);
    expect(Number.isFinite(climate.anomalyOut[0])).toBe(true);
    expect(climate.anomalyOut[0]).toBeCloseTo(expected, 6);
  });
}

it.each([
  // 50331651 / 3 is exactly halfway between 16777216 and 16777218.
  {name: 'even on a tie', sum: 50331651, count: 3, unit: 0, mean: 16777216},
  // An extra numerator unit puts the rational above that same halfway point.
  {name: 'up past a tie', sum: 50331652, count: 3, unit: 0, mean: 16777218},
  // 2^-118 / (2^32 - 1) is just above half the smallest float32 quantum.
  {name: 'into the subnormal range', sum: -1, count: 0xffffffff, unit: -118, mean: -(2 ** -149)}
])('S1 rounds the exact retained quotient $name', ({sum, count, unit, mean}) => {
  const climate = runHostClimateProbe(
    {
      counts: Uint32Array.of(count),
      sums: climateWords(sum * 2 ** unit, unit),
      moments: new Uint32Array(3),
      stateStatus: Uint32Array.of(0),
      accumulatorParameters: Float32Array.of(0, 0, unit, 0, 2),
      values: [0]
    },
    false
  );
  expect([...climate.climatologyMean]).toEqual([mean]);
});

it('B1 centers a product exactly at the signed minimum', () => {
  // 2 * (-2^24) in units of 2^-70 is -2^95. The retained mean is -16777214.5.
  const climate = runHostClimateProbe(
    {
      counts: Uint32Array.of(2),
      sums: climateWords(-33554429, -70),
      moments: climateWords(0.5, -43),
      stateStatus: Uint32Array.of(0),
      accumulatorParameters: Float32Array.of(0, 0, -70, -43, 2),
      values: [-(2 ** 24)]
    },
    false
  );
  expect([...climate.anomalyOut]).toEqual([-1.5]);
});

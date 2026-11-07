// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPURasterSeasonalBreak} from '../../../src/raster-analysis/remote-sensing/index';
import {requireTestDevice} from '../../utils/require-test-device';
import {createRemoteSensingHarness} from './remote-sensing-test-utils';
import reference from '../../oracles/fixtures/raster-seasonal-extended.json';

async function run(
  entry: (typeof reference.cases)[number],
  options: {
    fitMethod?: 'joint' | 'iterative';
    maximumIterations?: number;
    tolerance?: number;
    missingPolicy?: 'omit' | 'interpolate' | 'reject';
    validity?: number[];
    values?: number[];
  } = {}
) {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createRemoteSensingHarness(device);
  const coefficientCount = entry.seasonality === 'dummy' ? entry.period + 2 : 5;
  return harness.run(
    new GPURasterSeasonalBreak({
      cellCount: 1,
      sliceCount: entry.sliceCount,
      period: entry.period,
      seasonality: entry.seasonality === 'dummy' ? 'dummy' : 'harmonic',
      minimumSegmentLength: entry.minimumSegmentLength,
      missingPolicy:
        options.missingPolicy ?? (entry.missingPolicy === 'interpolate' ? 'interpolate' : 'omit'),
      monitoring: entry.monitoring ?? undefined,
      fitMethod: options.fitMethod ?? 'joint',
      maximumIterations: options.maximumIterations,
      tolerance: options.tolerance,
      values: harness.input(options.values ?? entry.values),
      validity: harness.mask(options.validity ?? entry.validity),
      output: {
        coefficients: harness.output('coefficients', 'float32', coefficientCount),
        components: harness.output('components', 'float32', entry.sliceCount * 3),
        breakIndex: harness.output('breakIndex', 'uint32', 1),
        improvement: harness.output('improvement', 'float32', 1),
        status: harness.output('status', 'uint32', 1),
        diagnostics: harness.output('diagnostics', 'float32', 6)
      }
    })
  );
}

for (const entry of reference.cases) {
  it(`RS11 extended direct NumPy reference: ${entry.name}`, async () => {
    const result = await run(entry);
    expect(result.status).toEqual([0]);
    expect(result.breakIndex).toEqual([entry.breakIndex]);
    result.coefficients.forEach((value, index) =>
      expect(value).toBeCloseTo(entry.coefficients[index], 4)
    );
    expect(result.improvement[0]).toBeCloseTo(entry.improvement, 2);
    for (let slice = 0; slice < entry.sliceCount; slice++) {
      expect(result.components[slice * 3] + result.components[slice * 3 + 1]).toBeCloseTo(
        entry.prediction[slice],
        4
      );
      if (!entry.validity[slice]) expect(Number.isNaN(result.components[slice * 3 + 2])).toBe(true);
    }
    expect(result.diagnostics.slice(0, 3)).toEqual([1, 1, entry.support]);
    expect(result.diagnostics[3]).toBeCloseTo(entry.scale, 4);
    expect(result.diagnostics[4]).toBe(entry.alarm);
    expect(result.diagnostics[5]).toBeCloseTo(entry.maximumScore, 2);
  });
}

it('RS11 cyclic trend/seasonal decomposition converges to independent joint least squares', async () => {
  const entry = reference.cases[0];
  const result = await run(entry, {
    fitMethod: 'iterative',
    maximumIterations: 256,
    tolerance: 1e-5
  });
  expect(result.status).toEqual([0]);
  expect(result.diagnostics[1]).toBe(1);
  expect(result.diagnostics[0]).toBeGreaterThan(1);
  expect(result.breakIndex).toEqual([entry.breakIndex]);
  result.coefficients.forEach((value, index) =>
    expect(value).toBeCloseTo(entry.coefficients[index], 2)
  );
  expect(result.improvement[0]).toBeCloseTo(entry.improvement, 1);
});

it('RS11 exhausted decomposition exposes finite incumbent, incomplete convergence and rejected-gap modes', async () => {
  const entry = reference.cases[0];
  const exhausted = await run(entry, {
    fitMethod: 'iterative',
    maximumIterations: 1,
    tolerance: 1e-8
  });
  expect(exhausted.status).toEqual([8]);
  expect(exhausted.diagnostics.slice(0, 2)).toEqual([1, 0]);
  expect(exhausted.coefficients.every(Number.isFinite)).toBe(true);
  const rejected = await run(entry, {missingPolicy: 'reject'});
  expect(rejected.status).toEqual([4]);
  expect(rejected.coefficients.every(Number.isNaN)).toBe(true);
  const missing = await run(entry, {validity: entry.validity.map(() => 0)});
  expect(missing.status).toEqual([1]);
  expect(missing.diagnostics[2]).toBe(0);
  // Remove an entire phase: enough total support remains but dummy design is rank deficient.
  const singular = await run(entry, {
    validity: entry.validity.map((valid, index) => (index % entry.period === 2 ? 0 : valid))
  });
  expect(singular.status).toEqual([2]);
  expect(singular.coefficients.every(Number.isNaN)).toBe(true);
});

it('RS11 monitoring gaps interrupt persistence and never use future data to interpolate history', async () => {
  const entry = reference.cases[4];
  const validity = entry.validity.slice();
  validity[41] = 0;
  const result = await run(entry, {validity});
  expect(result.status).toEqual([0]);
  expect(result.diagnostics[4]).toBe(42);
  const values = entry.values.slice();
  values.fill(10000, entry.monitoring!.historyLength);
  const altered = await run(entry, {values});
  expect(altered.coefficients).toEqual(result.coefficients);
  expect(altered.diagnostics[4]).toBe(entry.monitoring!.historyLength);
});

it('RS11 eight 128-date cells with twelve dummy seasons preserve historical fits and monitor provenance', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createRemoteSensingHarness(device);
  const cells = 8;
  const slices = 128;
  const result = await harness.run(
    new GPURasterSeasonalBreak({
      cellCount: cells,
      sliceCount: slices,
      period: 12,
      seasonality: 'dummy',
      minimumSegmentLength: 24,
      monitoring: {historyLength: 96, threshold: 4, minimumConsecutive: 3},
      values: harness.input(
        Array.from({length: cells * slices}, (_, index) => {
          const slice = index % slices,
            cell = Math.floor(index / slices);
          return 3 + cell + slice / 16 + (slice % 12) ** 2 / 12 + (slice >= 104 ? 2 : 0);
        })
      ),
      output: {
        coefficients: harness.output('coefficients', 'float32', cells * 14),
        components: harness.output('components', 'float32', cells * slices * 3),
        breakIndex: harness.output('breakIndex', 'uint32', cells),
        improvement: harness.output('improvement', 'float32', cells),
        status: harness.output('status', 'uint32', cells),
        diagnostics: harness.output('diagnostics', 'float32', cells * 6)
      }
    })
  );
  expect(result.status).toEqual(Array(cells).fill(0));
  expect(result.breakIndex).toEqual(Array(cells).fill(0xffffffff));
  for (let cell = 0; cell < cells; cell++) {
    expect(result.diagnostics[cell * 6 + 4]).toBe(104);
    expect(result.coefficients[cell * 14]).toBeCloseTo(3 + cell, 3);
    expect(result.coefficients[cell * 14 + 1]).toBeCloseTo(8, 3);
    for (let phase = 1; phase < 12; phase++)
      expect(result.coefficients[cell * 14 + 1 + phase]).toBeCloseTo(phase ** 2 / 12, 3);
  }
}, 120000);

it('RS11 interpolation never crosses the training boundary and reject policy only rejects training gaps', async () => {
  const entry = reference.cases[4];
  const history = entry.monitoring!.historyLength;
  const validity = entry.validity.slice();
  validity[history - 2] = 0;
  validity[history - 1] = 0;
  const first = await run(entry, {validity});
  const values = entry.values.slice();
  values[history] = 1e6;
  const second = await run(entry, {validity, values});
  expect(first.status).toEqual([0]);
  expect(second.status).toEqual([0]);
  expect(first.coefficients).toEqual(second.coefficients);
  expect(first.diagnostics[2]).toBe(history - 2);
  expect(second.diagnostics[2]).toBe(history - 2);
  expect(Number.isNaN(first.components[(history - 1) * 3 + 2])).toBe(true);
  const monitoringOnly = entry.validity.map((value, index) => (index < history ? 1 : value));
  monitoringOnly[41] = 0;
  const rejectedPolicy = await run(entry, {validity: monitoringOnly, missingPolicy: 'reject'});
  expect(rejectedPolicy.status).toEqual([0]);
  expect(rejectedPolicy.diagnostics[4]).toBe(42);
});

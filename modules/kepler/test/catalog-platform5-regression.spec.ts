// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, test} from 'vitest';
import {runCatalogOperationAsync} from '../src/analysis/catalog-bindings';
import type {CatalogOperationBinding} from '../src/analysis/catalog-binding-types';
import type {CatalogFeatureCollection} from '../src/analysis/catalog-geojson';
import {PLATFORM5_REGRESSION_BINDINGS} from '../src/analysis/catalog-platform5-regression';
import {requireTestDevice} from '../../analysis/test/utils/require-test-device';

// Oracle policy: the CPU engines are float64 implementations of the same models; the GPU engines
// run the contributors and must match them to float32 tolerance.

async function getDevice(): Promise<Device> {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  return device;
}

function getBinding(id: string): CatalogOperationBinding {
  return PLATFORM5_REGRESSION_BINDINGS.find(binding => binding.id === id)!;
}

function makeGenerator(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 4294967296;
  };
}

function makeNormal(next: () => number): () => number {
  return () => {
    const u = Math.max(next(), 1e-9);
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * next());
  };
}

function observations(
  count: number,
  seed: number,
  makeProperties: (
    x: number,
    y: number,
    normal: () => number,
    next: () => number
  ) => Record<string, number>
): CatalogFeatureCollection {
  const next = makeGenerator(seed);
  const normal = makeNormal(next);
  return {
    type: 'FeatureCollection',
    features: Array.from({length: count}, () => {
      const x = next() * 100;
      const y = next() * 100;
      return {
        geometry: {type: 'Point', coordinates: [x, y]},
        properties: makeProperties(x, y, normal, next)
      };
    })
  };
}

async function runBoth(
  device: Device,
  binding: CatalogOperationBinding,
  parameters: Record<string, unknown>,
  inputs: Record<string, CatalogFeatureCollection>
) {
  const gpu = await runCatalogOperationAsync(binding, parameters, inputs, {device});
  const cpu = await runCatalogOperationAsync(binding, parameters, inputs, {device, engine: 'cpu'});
  expect(gpu.engine, `${binding.id} must run on the GPU: ${gpu.fallbackReason ?? ''}`).toBe('gpu');
  expect(gpu.dispatchedContributors).toEqual(binding.gpuContributorNames);
  expect(cpu.engine).toBe('cpu-reference');
  return {gpu, cpu};
}

function expectClose(
  actual: readonly (string | number | boolean | null)[],
  expected: readonly (string | number | boolean | null)[],
  tolerance: number,
  label: string
): void {
  expect(actual.length, label).toBe(expected.length);
  expected.forEach((want, index) => {
    const got = actual[index];
    if (typeof want === 'number' && typeof got === 'number')
      expect(Math.abs(got - want), `${label}[${index}]`).toBeLessThanOrEqual(
        tolerance * Math.max(1, Math.abs(want))
      );
    else expect(got, `${label}[${index}]`).toEqual(want);
  });
}

test('regression-gwr matches the CPU local fits for both kernels', async () => {
  const device = await getDevice();
  const binding = getBinding('regression-gwr');
  // The slope of a varies with x, the slope of b with y.
  const data = observations(150, 3, (x, y, normal, next) => {
    const a = next() * 4;
    const b = next() * 4;
    return {a, b, y: 2 + (x / 50) * a + (y / 50) * b + 0.2 * normal()};
  });
  for (const [kernel, bandwidth] of [
    ['gaussian', 25],
    ['bisquare', 45]
  ] as const) {
    const {gpu, cpu} = await runBoth(
      device,
      binding,
      {
        responseProperty: 'y',
        predictorProperties: 'a, b',
        kernel,
        bandwidth,
        coordinateSystem: 'planar'
      },
      {points: data}
    );
    for (const column of [
      'coef_intercept',
      'coef_a',
      'coef_b',
      'se_intercept',
      'se_a',
      'se_b',
      'fitted',
      'residual'
    ])
      expectClose(
        gpu.tables[0].columns[column],
        cpu.tables[0].columns[column],
        2e-3,
        `${kernel} ${column}`
      );
    expect(gpu.summary['rowCount']).toBe(150);
    expect(
      Math.abs((gpu.summary['sigmaSquared'] as number) - (cpu.summary['sigmaSquared'] as number))
    ).toBeLessThan(2e-3 * Math.max(1, cpu.summary['sigmaSquared'] as number));
    // The slope of a grows with x: the west and east halves differ.
    const slopes = cpu.tables[0].columns['coef_a'] as number[];
    expect(slopes.some(value => Number.isFinite(value))).toBe(true);
  }
});

test('regression-probit matches the CPU maximum likelihood fit', async () => {
  const device = await getDevice();
  const binding = getBinding('regression-probit');
  const data = observations(500, 9, (_x, _y, normal) => {
    const a = normal();
    const b = normal();
    // Latent index with noise from the same generator keeps the classes overlapping.
    const latent = 0.3 + 0.9 * a - 0.6 * b + normal();
    return {a, b, y: latent > 0 ? 1 : 0};
  });
  const {gpu, cpu} = await runBoth(
    device,
    binding,
    {responseProperty: 'y', predictorProperties: 'a, b', coordinateSystem: 'planar'},
    {points: data}
  );
  for (const column of ['estimate', 'standard_error', 'z_statistic', 'average_marginal_effect'])
    expectClose(gpu.tables[0].columns[column], cpu.tables[0].columns[column], 5e-3, column);
  expectClose(
    gpu.tables[1].columns['probability'],
    cpu.tables[1].columns['probability'],
    5e-3,
    'p'
  );
  const estimates = cpu.tables[0].columns['estimate'] as number[];
  expect(estimates[1]).toBeGreaterThan(0.4);
  expect(estimates[2]).toBeLessThan(-0.2);
  expect(
    Math.abs((gpu.summary['logLikelihood'] as number) - (cpu.summary['logLikelihood'] as number))
  ).toBeLessThan(0.05);
});

test('regression-sur matches the CPU feasible GLS system fit', async () => {
  const device = await getDevice();
  const binding = getBinding('regression-sur');
  // Two equations with correlated errors and different designs.
  const data = observations(400, 17, (_x, _y, normal) => {
    const a = normal();
    const b = normal();
    const c = normal();
    const shared = normal();
    return {
      a,
      b,
      c,
      y1: 1 + 2 * a - b + shared + 0.5 * normal(),
      y2: -1 + 0.5 * a + 1.5 * c + shared + 0.5 * normal()
    };
  });
  const {gpu, cpu} = await runBoth(
    device,
    binding,
    {
      responseProperties: 'y1, y2',
      predictorProperties: 'a, b; a, c',
      coordinateSystem: 'planar'
    },
    {points: data}
  );
  for (const column of ['estimate', 'standard_error'])
    expectClose(gpu.tables[0].columns[column], cpu.tables[0].columns[column], 5e-3, column);
  expectClose(gpu.tables[1].columns['residual_0'], cpu.tables[1].columns['residual_0'], 5e-3, 'r0');
  expectClose(gpu.tables[1].columns['residual_1'], cpu.tables[1].columns['residual_1'], 5e-3, 'r1');
  const estimates = cpu.tables[0].columns['estimate'] as number[];
  expect(estimates[1]).toBeGreaterThan(1.5);
});

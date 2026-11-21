// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readFileSync} from 'node:fs';
import {expect, it} from 'vitest';
import {makeGPURasterMetadata} from '@luma-contributors/analysis/raster-analysis';
import {computeTerrainFlowDistanceOnCPU} from '../../../src/gpu-terrain/hydrology/cpu-terrain-flow-distance-operation';
import {computeTerrainStreamOrderOnCPU} from '../../../src/gpu-terrain/hydrology/cpu-terrain-stream-order-operation';
import {computeTerrainCatchmentSummaryOnCPU} from '../../../src/gpu-terrain/terrain-processing/cpu-terrain-catchment-summary-operation';

it('FlowDistance leaves a cycle and its tributary unresolved beside a genuine terminal (RT-123)', () => {
  // 0 <-> 1 <- 2 has no terminal. Separately, 3 -> 4 -> 5 has two unit edges.
  const result = computeTerrainFlowDistanceOnCPU(Uint32Array.of(1, 16, 16, 1, 1, 0), {
    grid: makeGPURasterMetadata({width: 6, height: 1})
  });
  expect(Array.from(result.terminalCells)).toEqual([0xfffffffe, 0xfffffffe, 0xfffffffe, 5, 5, 5]);
  expect(Array.from(result.validity)).toEqual([0, 0, 0, 1, 1, 1]);
  for (const value of result.distance.slice(0, 3)) expect(Number.isNaN(value)).toBe(true);
  expect(Array.from(result.distance.slice(3))).toEqual([2, 1, 0]);
});

it('StreamOrder cannot certify a fabricated cycle root or its tributary (RT-123)', () => {
  // A rootless component has no Euler-tour position; the separate leaf and outlet each have
  // order 1, since neither has two equal-order donors. The gap is not a stream.
  const result = computeTerrainStreamOrderOnCPU(
    Uint32Array.of(1, 16, 16, 0, 1, 0),
    Uint32Array.of(1, 1, 1, 0, 1, 1),
    {grid: makeGPURasterMetadata({width: 6, height: 1})}
  );
  expect(Array.from(result.streamOrder)).toEqual([0xffffffff, 0xffffffff, 0xffffffff, 0, 1, 1]);
  expect(Array.from(result.converged)).toEqual([0]);
  expect(Array.from(result.iterationLimitReached)).toEqual([0]);
});

it('documents the imported stream code distinction and explicit unknown translation (RT-135)', () => {
  const grid = makeGPURasterMetadata({width: 1, height: 1});
  const directions = Uint32Array.of(0);
  const order = (word: number) =>
    Array.from(
      computeTerrainStreamOrderOnCPU(directions, Uint32Array.of(word), {grid}).streamOrder
    );
  const profileStatus = (word: number) =>
    Array.from(
      computeTerrainCatchmentSummaryOnCPU(
        {
          elevation: Float32Array.of(1),
          basinIds: Uint32Array.of(0),
          streams: Uint32Array.of(word),
          flowDirections: directions
        },
        {grid, basinCount: 1, histogramBins: 1}
      ).profileStatus
    );
  // One ordinary stream without donors has Strahler order 1. StreamOrder's sole unknown
  // stream word is 0xffffffff. Catchment profiles reserve both high words (status 2).
  expect(order(0xfffffffe)).toEqual([1]);
  expect(profileStatus(0xfffffffe)).toEqual([2]);
  const imported = Uint32Array.of(0xfffffffe);
  const translated = Uint32Array.from(imported, word => (word === 0xfffffffe ? 0xffffffff : word));
  expect(order(translated[0])).toEqual([0xffffffff]);
  expect(order(0xffffffff)).toEqual([0xffffffff]);
  expect(profileStatus(0xffffffff)).toEqual([2]);
  expect(order(1)).toEqual([1]);
  expect(profileStatus(1)).toEqual([1]);
  expect(order(0)).toEqual([0]);
  expect(profileStatus(0)).toEqual([0]);
});

it('direction precedence and unknown order differ from computation convergence (S2, RT-135)', () => {
  const grid = makeGPURasterMetadata({width: 1, height: 1});
  // Invalid elevation/direction cells are not members, even with the unknown stream word.
  expect(
    Array.from(
      computeTerrainStreamOrderOnCPU(Uint32Array.of(0xffffffff), Uint32Array.of(0xffffffff), {grid})
        .streamOrder
    )
  ).toEqual([0]);
  const order = computeTerrainStreamOrderOnCPU(Uint32Array.of(0), Uint32Array.of(0xffffffff), {
    grid
  });
  expect(Array.from(order.streamOrder)).toEqual([0xffffffff]);
  expect(Array.from(order.converged)).toEqual([1]);
  // An unknown direction prevents certification of a stop, even with known stream membership.
  const distance = computeTerrainFlowDistanceOnCPU(
    Uint32Array.of(0xfffffffe),
    {grid},
    Uint32Array.of(1)
  );
  expect(Array.from(distance.distance)).toEqual([NaN]);
  expect(Array.from(distance.validity)).toEqual([0]);
  expect(Array.from(distance.terminalCells)).toEqual([0xfffffffe]);
});

it('port documentation qualifies stream words by direction and convergence (S2)', () => {
  const orderSource = readFileSync(
    new URL('../../../src/gpu-terrain/hydrology/gpu-terrain-stream-order.ts', import.meta.url),
    'utf8'
  );
  expect(orderSource).not.toContain('not an order a converged run can produce');
  const summarySource = readFileSync(
    new URL(
      '../../../src/gpu-terrain/terrain-processing/gpu-terrain-catchment-summary.ts',
      import.meta.url
    ),
    'utf8'
  );
  expect(summarySource).not.toContain(
    'order, stream network, flow distance and height above drainage.'
  );
  const guide = readFileSync(
    new URL('../../../../../docs/api/gpu-terrain/terrain-processing.md', import.meta.url),
    'utf8'
  );
  expect(guide).toContain('Direction validity takes precedence');
});

it('the actual HAND readback assertion rejects missing and extra NaN rows (S3)', async () => {
  const source = readFileSync(
    new URL('./terrain-hydrology-cycle-followups.spec.ts', import.meta.url),
    'utf8'
  );
  const assertion = source.split('\n').find(line => line.includes("fixture.floats('hand')"));
  expect(assertion).toBeDefined();
  // Execute the assertion from the device spec with controlled readbacks. The four-cell
  // fixture needs exactly four rows; testing its own assertion catches a false-green .every().
  const check = new Function('expect', 'fixture', `return (async () => { ${assertion} })();`) as (
    assertionLibrary: typeof expect,
    fixture: {floats: () => Promise<number[]>}
  ) => Promise<void>;
  for (const values of [[], [NaN], [NaN, NaN, NaN], [NaN, NaN, NaN, NaN, NaN]]) {
    await expect(check(expect, {floats: async () => values})).rejects.toThrow();
  }
  await expect(check(expect, {floats: async () => [NaN, NaN, NaN, NaN]})).resolves.toBeUndefined();
});

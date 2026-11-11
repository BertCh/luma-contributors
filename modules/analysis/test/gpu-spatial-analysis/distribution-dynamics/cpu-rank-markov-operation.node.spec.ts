// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  computeRankMarkovOnCPU,
  CPURankMarkovOperation
} from '../../../src/gpu-spatial-analysis/distribution-dynamics/cpu-rank-markov-operation';
import {
  GPURankMarkovOperation,
  type RankMarkovOperationProps
} from '../../../src/gpu-spatial-analysis/distribution-dynamics/gpu-rank-markov-operation';
import {constructNetworkOperationContributor} from '../../utils/network-operation-test-utils';

const PROPS: RankMarkovOperationProps = {
  id: 'rank-test',
  values: 'values',
  rowCount: 3,
  periodCount: 3,
  output: {ranks: 'ranks', counts: 'counts'}
};

it('matches hand-traced rank cycles, ties, mobility and the uniform stationary distribution', () => {
  // Descending order: [0,1,2], [2,0,1], [1,2,0]. Every rank moves down one cyclically.
  const values = Float32Array.of(3, 2, 1, 2, 1, 3, 1, 3, 2);
  const result = computeRankMarkovOnCPU(values, {periodCount: 3});
  expect([...result.ranks]).toEqual([0, 1, 2, 1, 2, 0, 2, 0, 1]);
  expect([...result.counts]).toEqual([0, 2, 0, 0, 0, 2, 2, 0, 0]);
  expect([...result.probabilities]).toEqual([0, 1, 0, 0, 0, 1, 1, 0, 0]);
  expect([...result.rowTotals]).toEqual([2, 2, 2]);
  expect([...result.steadyStates]).toEqual(Array(3).fill(Math.fround(1 / 3)));
  expect([...result.stateSparsity]).toEqual([3, 0]);
  expect([...result.converged]).toEqual([1]);
  expect([...result.ignored]).toEqual([0]);
  expect([...result.mobility]).toEqual(Array(2).fill(Math.fround(4 / 3)));
  const geographic = computeRankMarkovOnCPU(values, {periodCount: 3, mode: 'geographic-rank'});
  expect([...geographic.counts]).toEqual([0, 0, 2, 2, 0, 0, 0, 2, 0]);
  const ties = Float32Array.of(1, 1, 0, 1, 1, 0);
  expect([...computeRankMarkovOnCPU(ties, {periodCount: 2}).ranks]).toEqual([1, 0, 2, 1, 0, 2]);
  expect([...computeRankMarkovOnCPU(ties, {periodCount: 2, tieOrder: 'row-id'}).ranks]).toEqual([
    0, 1, 2, 0, 1, 2
  ]);
});

it('excludes incomplete locations from every period and preserves empty/reducible diagnostics', () => {
  const values = Float32Array.of(2, NaN, 1, 3, 4, 0);
  const result = computeRankMarkovOnCPU(values, {periodCount: 2}, Uint32Array.of(1, 1, 0));
  expect([...result.ranks]).toEqual([0, 0xffffffff, 0xffffffff, 0, 0xffffffff, 0xffffffff]);
  expect([...result.counts]).toEqual([1, 0, 0, 0, 0, 0, 0, 0, 0]);
  expect([...result.ignored]).toEqual([2]);
  expect(Number.isNaN(result.mobility[0])).toBe(true);
  expect([...result.steadyStates]).toEqual([1, 0, 0]);
  const empty = computeRankMarkovOnCPU(values, {periodCount: 2}, new Uint32Array(3));
  expect(empty.steadyStates.every(Number.isNaN)).toBe(true);
  expect([...empty.converged]).toEqual([1]);
  const reducible = computeRankMarkovOnCPU(Float32Array.of(2, 1, 2, 1), {periodCount: 2});
  expect(reducible.steadyStates.every(Number.isNaN)).toBe(true);
  expect([...reducible.converged]).toEqual([0]);
});

it('validates arrays, CPU construction and deferred GPU contributor construction', () => {
  expect(() => computeRankMarkovOnCPU(new Float32Array(5), {periodCount: 2})).toThrow(/values/);
  expect(() =>
    computeRankMarkovOnCPU(new Float32Array(6), {periodCount: 2}, new Uint32Array(2))
  ).toThrow(/mask/);
  expect(() => new CPURankMarkovOperation({...PROPS, periodCount: 1})).toThrow(
    /rank-test.*periodCount/
  );
  expect(() => new GPURankMarkovOperation({...PROPS, rowCount: 0})).toThrow(/rank-test.*length/);
  expect(() =>
    constructNetworkOperationContributor(new GPURankMarkovOperation({...PROPS, periodCount: 1}), {
      values: {format: 'float32', length: 3}
    })
  ).toThrow(/periodCount/);
  expect(() =>
    constructNetworkOperationContributor(new GPURankMarkovOperation({...PROPS, rowCount: 65}), {
      values: {format: 'float32', length: 195}
    })
  ).toThrow(/rowCount/);
  expect(() => new CPURankMarkovOperation({...PROPS, rowCount: 65})).not.toThrow();
});

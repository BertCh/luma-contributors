// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {assertCostDistanceSubmissionBudget} from './cost-distance-round-diagnostic';
import {emulateTiledRelaxation} from './raster-relaxation-emulator';

it('rejects a converging submission that exceeds its one-round budget', () => {
  expect(() =>
    assertCostDistanceSubmissionBudget(
      0,
      {iterationCount: 2, converged: 1, iterationLimitReached: 0},
      1
    )
  ).toThrow();
  expect(() =>
    assertCostDistanceSubmissionBudget(
      1,
      {iterationCount: 2, converged: 1, iterationLimitReached: 0},
      1
    )
  ).not.toThrow();
});

it('different legal halo visibility gives different counts with the same exact corridor costs', () => {
  // 48 unit edges on a line: cost[c] = c exactly. Forward visibility crosses three tiles
  // in one improving round; reverse/snapshot visibility needs three improving rounds.
  const counts = [false, true].map(reversePlainWorkgroups => {
    const values = new Float64Array(48).fill(Infinity);
    values[0] = 0;
    const result = emulateTiledRelaxation({
      width: 48,
      height: 1,
      values,
      participates: () => true,
      getEdgeCost: () => 1,
      maxIterations: 8,
      activateAllTiles: false,
      plainReadVisibility: 'live',
      reversePlainWorkgroups,
      state: {stamps: Uint32Array.of(1, 0, 0), iterationCount: 0}
    });
    expect(result.converged).toBe(true);
    expect(Array.from(values)).toEqual(Array.from({length: 48}, (_, column) => column));
    return result.iterationCount;
  });
  expect(counts).toEqual([2, 4]);
});

it('the cumulative sweep fixture distinguishes a right sweep from disabled or restarted phases', () => {
  const observations = [1, 512].map(sweepAfterIteration => {
    const values = new Float64Array(128).fill(Infinity);
    values[0] = 0;
    const stamps = new Uint32Array(8);
    stamps[0] = 1;
    const state = {stamps, iterationCount: 0};
    const tail: number[] = [];
    const convergence: boolean[] = [];
    for (let execution = 0; execution < 4; execution++) {
      const result = emulateTiledRelaxation({
        width: 128,
        height: 1,
        values,
        participates: () => true,
        getEdgeCost: () => 1,
        maxIterations: 1,
        activateAllTiles: false,
        sweepAfterIteration,
        state
      });
      tail.push(values[127]);
      convergence.push(result.converged);
    }
    return {tail, convergence};
  });
  expect(observations).toEqual([
    {tail: [Infinity, Infinity, 127, 127], convergence: [false, false, false, true]},
    {tail: [Infinity, Infinity, Infinity, Infinity], convergence: [false, false, false, false]}
  ]);
});

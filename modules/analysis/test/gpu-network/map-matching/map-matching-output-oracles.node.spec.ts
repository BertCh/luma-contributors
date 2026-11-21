// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {expect, it} from 'vitest';
import {computeMapMatchingOnCPU} from '../../../src/gpu-network/map-matching/cpu-map-matching-operation';
import {expectOracleRows} from '../alternative-routes/output-oracle-harness';
import {createMatchingOutputOracle, getMatchingOutputExpected} from './map-matching-output-oracles';

for (const routeNodeBudget of [8, 16])
  it(`pins chain offsets, matched/break counts and exact route drops at budget ${routeNodeBudget}`, () => {
    const {input} = createMatchingOutputOracle(routeNodeBudget);
    const result = computeMapMatchingOnCPU(input, {
      cellSize: 2,
      candidateCount: 1,
      routeNodeBudget
    });
    expect(Array.from(result.matchedOffsets)).not.toEqual(Array.from(result.matchedFractions));
    expect(result.matchedCount[0]).toBeLessThan(input.positions.length / 2);
    expect(result.breakCount[0]).not.toBe(result.routeOverflow[0]);
    for (const [port, expected] of Object.entries(getMatchingOutputExpected(routeNodeBudget)))
      expectOracleRows(result[port as keyof typeof result], expected);
  });

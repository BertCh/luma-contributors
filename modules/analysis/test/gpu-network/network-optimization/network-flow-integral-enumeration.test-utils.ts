// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect} from 'vitest';
import {expectNetworkFlowFeasible} from './network-flow-feasibility-test-utils';

// The first cheapest path 0-1-3-5 costs 1. Completing it greedily costs 101;
// cancelling 1->3 through its reverse residual arc gives paths 0-1-4-5 and 0-2-3-5,
// cost 2 + 2 = 4. Enumerate all 2^8 integral capacity-feasible assignments independently.
export const REVERSE_RESIDUAL_SCENE = {
  nodeCount: 6,
  edges: [0, 1, 0, 2, 1, 3, 1, 4, 2, 3, 2, 4, 3, 5, 4, 5],
  capacities: [1, 1, 1, 1, 1, 1, 1, 1],
  costs: [0, 0, 1, 2, 2, 100, 0, 0]
};

export function enumerateIntegralOptima(required: number) {
  let value = -1;
  let cost = Infinity;
  const flows: number[][] = [];
  for (let bits = 0; bits < 256; bits++) {
    const flow = Array.from({length: 8}, (_, edge) => (bits >>> edge) & 1);
    const balance = new Array<number>(6).fill(0);
    let objective = 0;
    for (let edge = 0; edge < 8; edge++) {
      balance[REVERSE_RESIDUAL_SCENE.edges[2 * edge]] -= flow[edge];
      balance[REVERSE_RESIDUAL_SCENE.edges[2 * edge + 1]] += flow[edge];
      objective += flow[edge] * REVERSE_RESIDUAL_SCENE.costs[edge];
    }
    if (balance.slice(1, 5).some(amount => amount !== 0) || balance[5] > required) continue;
    if (balance[5] > value || (balance[5] === value && objective < cost)) {
      value = balance[5];
      cost = objective;
      flows.length = 0;
    }
    if (balance[5] === value && objective === cost) flows.push(flow);
  }
  return {value, cost, flows};
}

export function expectEnumeratedIntegralFlow(
  result: {
    flow: number[];
    cut: number[];
    summary: number[];
    exactCost: number[];
  },
  required: number
): void {
  const optimum = enumerateIntegralOptima(required);
  expect([optimum.value, optimum.cost]).toEqual(required === 1 ? [1, 1] : [2, 4]);
  expect(result.summary).toHaveLength(8);
  const objectiveWords = new DataView(new ArrayBuffer(4));
  objectiveWords.setUint32(0, result.summary[5], true);
  const publicObjective = objectiveWords.getFloat32(0, true);
  expect(Number.isFinite(publicObjective)).toBe(true);
  expect(publicObjective).toBe(optimum.cost);
  expect(result.cut).toHaveLength(REVERSE_RESIDUAL_SCENE.nodeCount);
  for (let node = 0; node < REVERSE_RESIDUAL_SCENE.nodeCount; node++) {
    expect(Number.isFinite(result.cut[node])).toBe(true);
    expect([0, 1]).toContain(result.cut[node]);
  }
  expect(result.summary.slice(0, 4)).toEqual([1, 1, optimum.value, 0]);
  expectNetworkFlowFeasible(REVERSE_RESIDUAL_SCENE, result.flow, optimum.value);
  const actualCost = result.flow.reduce(
    (sum, amount, edge) => sum + amount * REVERSE_RESIDUAL_SCENE.costs[edge],
    0
  );
  expect(Number.isFinite(actualCost)).toBe(true);
  expect(actualCost).toBe(optimum.cost);
  expect(result.exactCost).toEqual([optimum.cost, 0]);
  // Accept every optimal edge assignment, including a nonunique optimum if the fixture changes.
  expect(optimum.flows).toContainEqual(result.flow);
  expect(result.summary[7]).toBe(required === 1 ? 0 : 1);
  if (result.summary[7] === 1) {
    // Independent residual traversal: unused forward capacity and positive reverse flow.
    const reachable = new Array<number>(REVERSE_RESIDUAL_SCENE.nodeCount).fill(0);
    reachable[0] = 1;
    const frontier = [0];
    for (const node of frontier) {
      for (let edge = 0; edge < REVERSE_RESIDUAL_SCENE.capacities.length; edge++) {
        const source = REVERSE_RESIDUAL_SCENE.edges[2 * edge];
        const target = REVERSE_RESIDUAL_SCENE.edges[2 * edge + 1];
        if (
          node === source &&
          result.flow[edge] < REVERSE_RESIDUAL_SCENE.capacities[edge] &&
          !reachable[target]
        ) {
          reachable[target] = 1;
          frontier.push(target);
        }
        if (node === target && result.flow[edge] > 0 && !reachable[source]) {
          reachable[source] = 1;
          frontier.push(source);
        }
      }
    }
    expect(result.cut).toEqual(reachable);
    expect(result.cut[0]).toBe(1);
    expect(result.cut[5]).toBe(0);
    const capacity = REVERSE_RESIDUAL_SCENE.capacities.reduce(
      (sum, bound, edge) =>
        sum +
        (result.cut[REVERSE_RESIDUAL_SCENE.edges[edge * 2]] &&
        !result.cut[REVERSE_RESIDUAL_SCENE.edges[edge * 2 + 1]]
          ? bound
          : 0),
      0
    );
    expect(capacity).toBe(optimum.value);
    expect(result.summary[6]).toBe(capacity);
  }
}

// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect} from 'vitest';

/** Independent integral capacity and conservation oracle, shared by Node and GPU specs. */
export function expectNetworkFlowFeasible(
  scene: {
    nodeCount: number;
    edges: ArrayLike<number>;
    capacities: ArrayLike<number>;
    source?: number;
    sink?: number;
  },
  flow: ArrayLike<number>,
  value: number
): void {
  expect(flow.length).toBe(scene.capacities.length);
  const balances = new Float64Array(scene.nodeCount);
  for (let edge = 0; edge < scene.capacities.length; edge++) {
    expect(Number.isInteger(flow[edge])).toBe(true);
    expect(flow[edge]).toBeGreaterThanOrEqual(0);
    expect(flow[edge]).toBeLessThanOrEqual(scene.capacities[edge]);
    balances[scene.edges[2 * edge]] -= flow[edge];
    balances[scene.edges[2 * edge + 1]] += flow[edge];
  }
  const source = scene.source ?? 0;
  const sink = scene.sink ?? scene.nodeCount - 1;
  // uint32 flow has one zero. Negating a zero value produces -0 in JavaScript, which
  // Object.is (and toBe) distinguishes from the +0 balance of an untouched vertex.
  expect(balances[source], 'source balance').toBe(value === 0 ? 0 : -value);
  expect(balances[sink], 'sink balance').toBe(value);
  for (let vertex = 0; vertex < scene.nodeCount; vertex++) {
    if (vertex !== source && vertex !== sink) {
      expect(balances[vertex], `vertex ${vertex} balance`).toBe(0);
    }
  }
}

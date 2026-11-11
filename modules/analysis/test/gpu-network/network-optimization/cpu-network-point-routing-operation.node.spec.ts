// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  CPUNetworkPointRoutingOperation,
  computeNetworkPointRoutingOnCPU
} from '../../../src/gpu-network/network-optimization/cpu-network-point-routing-operation';
import {
  GPUNetworkPointRoutingOperation,
  type NetworkPointRoutingOperationProps
} from '../../../src/gpu-network/network-optimization/gpu-network-point-routing-operation';
import {
  enumeratePointRoutingCosts,
  getPointRoutingInput
} from './network-point-routing-operation-test-utils';

it('keeps a finite partial-target cost when only the full-edge float32 sum would overflow', () => {
  const maximumFloat32 = new Float32Array(Uint32Array.of(0x7f7fffff).buffer)[0];
  const result = computeNetworkPointRoutingOnCPU({
    costs: Float32Array.of(maximumFloat32, maximumFloat32),
    turns: Uint32Array.of(0, 1),
    queries: Uint32Array.of(0, 1),
    fractions: Float32Array.of(0, 0)
  });
  // Remaining source cost M plus target fraction 0 is M. The full label 2M is
  // finite in float64, so the CPU completes with parent 0 instead of returning status 1.
  expect(Array.from(result.costs)).toEqual([2 ** 128 - 2 ** 104]);
  expect(Array.from(result.status)).toEqual([0]);
  expect(Array.from(result.predecessorEdges)).toEqual([0xffffffff, 0]);
});

it('matches exhaustive edge-state walks and hand-computed parents, including legal returns', () => {
  const input = getPointRoutingInput();
  const result = computeNetworkPointRoutingOnCPU(input);
  const missing = 0xffffffff;
  expect(result.costs).toEqual(enumeratePointRoutingCosts(input));
  expect(Array.from(result.costs)).toEqual([11, 12, 4, 19, Infinity]);
  expect(Array.from(result.status)).toEqual([0, 0, 0, 0, 0]);
  expect(Array.from(result.predecessorEdges)).toEqual([
    missing,
    0,
    3,
    0,
    3,
    missing,
    2,
    0,
    3,
    0,
    3,
    missing,
    missing,
    0,
    3,
    0,
    3,
    missing,
    2,
    0,
    4,
    0,
    missing,
    missing,
    missing,
    missing,
    missing,
    missing,
    missing,
    missing
  ]);
});

it('retains turn-order ties and status 1 for defined invalid data', () => {
  const tied = computeNetworkPointRoutingOnCPU({
    costs: Float32Array.of(1, 1, 1, 1),
    turns: Uint32Array.of(0, 2, 0, 1, 2, 3, 1, 3),
    queries: Uint32Array.of(0, 3),
    fractions: Float32Array.of(0, 1)
  });
  expect(tied.costs[0]).toBe(3);
  expect(tied.predecessorEdges[3]).toBe(2);
  const input = getPointRoutingInput();
  const invalidCosts = computeNetworkPointRoutingOnCPU({
    ...input,
    costs: Float32Array.of(8, -1, 6, 2, 3, 5)
  });
  expect(invalidCosts.status.every(value => value === 1)).toBe(true);
  expect(invalidCosts.costs.every(value => value === Infinity)).toBe(true);
  expect(invalidCosts.predecessorEdges.every(value => value === 0xffffffff)).toBe(true);
  for (const fractions of [
    Float32Array.of(-1, 0),
    Float32Array.of(0, 1.25),
    Float32Array.of(NaN, 0)
  ])
    expect(
      computeNetworkPointRoutingOnCPU({...input, queries: Uint32Array.of(0, 2), fractions})
        .status[0]
    ).toBe(1);
  expect(
    computeNetworkPointRoutingOnCPU({...input, turns: Uint32Array.of(0, 99)}).status.every(
      value => value === 1
    )
  ).toBe(true);
});

it('rejects malformed packed arrays and validates operation shapes at construction', () => {
  const input = getPointRoutingInput();
  const empty = computeNetworkPointRoutingOnCPU({
    ...input,
    queries: new Uint32Array(0),
    fractions: new Float32Array(0)
  });
  expect(empty.costs.length).toBe(0);
  expect(empty.status.length).toBe(0);
  expect(empty.predecessorEdges.length).toBe(0);
  expect(() => computeNetworkPointRoutingOnCPU({...input, turns: Uint32Array.of(0)})).toThrow(
    /turns/
  );
  expect(() =>
    computeNetworkPointRoutingOnCPU({...input, fractions: Float32Array.of(0, 1)})
  ).toThrow(/fractions/);
  expect(() => computeNetworkPointRoutingOnCPU({...input, queries: Uint32Array.of(0)})).toThrow(
    /queries/
  );
  const props: NetworkPointRoutingOperationProps = {
    id: 'check-point',
    edgeCount: 6,
    queryCount: 5,
    turns: 'turns',
    costs: 'edgeCosts',
    queries: 'queries',
    fractions: 'fractions',
    output: {costs: 'costs', status: 'status', predecessorEdges: 'parents'}
  };
  for (const Operation of [CPUNetworkPointRoutingOperation, GPUNetworkPointRoutingOperation])
    expect(() => new Operation({...props, queryCount: 0})).toThrow(/^check-point/);
  expect(
    () => new CPUNetworkPointRoutingOperation({...props, edgeCount: 129, maximumRelaxations: 0})
  ).not.toThrow();
});

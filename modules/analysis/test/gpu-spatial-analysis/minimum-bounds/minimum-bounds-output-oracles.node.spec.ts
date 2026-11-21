// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {computeMinimumBoundsOnCPU} from '../../../src/gpu-spatial-analysis/minimum-bounds/cpu-minimum-bounds-operation';
import {GPUMinimumBoundsOperation} from '../../../src/gpu-spatial-analysis/minimum-bounds/gpu-minimum-bounds-operation';
import {constructNetworkOperationContributor} from '../../utils/network-operation-test-utils';
import {
  BOUNDS_POSITIONS,
  BOUNDS_LABELS,
  BOUNDS_EXPECTED,
  BOUNDS_OUTPUT_CASES,
  getBoundsOracleOperationProps,
  expectBoundsOutputPorts,
  expectBoundsPort
} from './minimum-bounds-output-oracles-fixture';

it('CPU minimum bounds pins every column to the independent 3-4-5 triangle and singleton', () => {
  const result = computeMinimumBoundsOnCPU({
    positions: Float32Array.from(BOUNDS_POSITIONS),
    labels: Uint32Array.from(BOUNDS_LABELS),
    groupCount: 2
  });
  for (const port of Object.keys(BOUNDS_EXPECTED) as (keyof typeof BOUNDS_EXPECTED)[])
    expectBoundsPort(port, result[port]);
});

it.each(
  BOUNDS_OUTPUT_CASES.map(selected => [selected.join('+'), selected] as const)
)('minimum-bounds headless oracle output case %s satisfies contributor construction', (_name, selected) => {
  const operation = new GPUMinimumBoundsOperation(getBoundsOracleOperationProps(selected));
  expectBoundsOutputPorts(selected, operation.outputs);
  expect(() =>
    constructNetworkOperationContributor(operation, {
      positions: {format: 'float32x2', length: 5},
      labels: {format: 'uint32', length: 5}
    })
  ).not.toThrow();
});

it('requested groupOverflow cannot disappear from the operation output contract', () => {
  const selected = ['sourceBounds', 'groupOverflow'] as const;
  const operation = new GPUMinimumBoundsOperation(getBoundsOracleOperationProps(selected));
  const dropped = {...operation.outputs};
  delete dropped['groupOverflow'];
  Object.defineProperty(operation, 'outputs', {value: dropped});
  // The old construction-only preflight accepts this missing optional destination; the
  // independently requested port list must reject it before numerical assertions can skip it.
  expect(() =>
    constructNetworkOperationContributor(operation, {
      positions: {format: 'float32x2', length: 5},
      labels: {format: 'uint32', length: 5}
    })
  ).not.toThrow();
  expect(() => expectBoundsOutputPorts(selected, operation.outputs)).toThrow();
});

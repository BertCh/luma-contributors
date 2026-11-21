// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  computeAZPRegionsOnCPU,
  CPUAZPRegionsOperation
} from '../../../src/gpu-spatial-analysis/spatial-regionalization/cpu-azp-regions-operation';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {getRegionalizationOperationInputs} from './regionalization-operation-test-utils';
import {createAuditAZPInput, createAuditAZPProps} from './regionalization-audit-azp-fixtures';

it('audits every AZP output against direct NumPy SSD with an island and an empty region', () => {
  for (const [scale, shift, standardize, objective] of [
    [1, 0, false, 24],
    [1, 64, false, 24],
    [3, 0, false, 216],
    [1, 0, true, 0.028328281650734446],
    [3, 64, true, 0.028328281650734446]
  ] as const) {
    const result = computeAZPRegionsOnCPU(createAuditAZPInput(scale, shift), {
      ...createAuditAZPProps(),
      standardize
    });
    expect(Array.from(result.labels)).toEqual([0, 0, 1, 1, 1, 1, 2, 2, 3]);
    expect(Array.from(result.status)).toEqual([1, 1, 0, 0]);
    expect(result.objective[0]).toBe(Math.fround(objective));
  }
});

it('preserves standardized AZP moves and every output for tiny nonconstant columns', () => {
  // Positive powers of two preserve these integer-valued float32 rows exactly. NumPy's
  // population z-scores and direct SSD are invariant: see regionalization-audit-azp.oracle.py.
  for (const exponent of [-80, 64, 120]) {
    const result = computeAZPRegionsOnCPU(createAuditAZPInput(2 ** exponent), {
      ...createAuditAZPProps(),
      standardize: true
    });
    expect(Array.from(result.labels)).toEqual([0, 0, 1, 1, 1, 1, 2, 2, 3]);
    expect(Array.from(result.status)).toEqual([1, 1, 0, 0]);
    expect(result.objective[0]).toBe(Math.fround(0.028328281650734446));
  }
});

it('runs the AZP CPU graph operation on tiny standardized inputs with the same oracle ports', async () => {
  // CPU-only graph: the null device supplies no GPU execution or numerical GPU evidence.
  const device = createNullWebGPUDevice();
  try {
    const result = await runOperationGraph(device, {
      inputs: getRegionalizationOperationInputs(createAuditAZPInput(2 ** -80)),
      operations: [new CPUAZPRegionsOperation({...createAuditAZPProps(), standardize: true})],
      outputs: ['labels', 'objective', 'status']
    });
    expect(Array.from(result['labels'])).toEqual([0, 0, 1, 1, 1, 1, 2, 2, 3]);
    expect(Array.from(result['status'])).toEqual([1, 1, 0, 0]);
    expect(Array.from(result['objective'])).toEqual([Math.fround(0.028328281650734446)]);
  } finally {
    device.destroy();
  }
});

it('audits AZP inclusive floors, disconnected status and a binding round budget', () => {
  const input = createAuditAZPInput();
  const props = createAuditAZPProps();
  const budget = computeAZPRegionsOnCPU(input, {...props, maximumIterations: 1});
  expect(Array.from(budget.labels)).toEqual([0, 0, 1, 1, 1, 1, 2, 2, 3]);
  expect(Array.from(budget.status)).toEqual([1, 0, 1, 0]);
  expect(budget.objective[0]).toBe(24);
  for (const result of [
    computeAZPRegionsOnCPU({...input, extensiveThreshold: Float32Array.of(3)}, props),
    computeAZPRegionsOnCPU(input, {...props, minimumRegionSize: 3})
  ]) {
    expect(result.labels).toEqual(input.initialLabels);
    expect(Array.from(result.status)).toEqual([0, 1, 0, 0]);
    expect(result.objective[0]).toBe(Math.fround(44.66666666666667));
  }
  input.initialLabels[8] = 0;
  const disconnected = computeAZPRegionsOnCPU(input, props);
  expect(disconnected.labels).toEqual(input.initialLabels);
  expect(Array.from(disconnected.status)).toEqual([0, 0, 0, 1]);
  expect(disconnected.objective[0]).toBe(0);
});

it('suppresses AZP labels and objective when supplied adjacency reports lost edges', () => {
  const input = {...createAuditAZPInput(), status: {overflow: Uint32Array.of(1)}};
  const result = computeAZPRegionsOnCPU(input, createAuditAZPProps());
  expect(Array.from(result.labels)).toEqual(new Array(9).fill(4294967295));
  expect(Number.isNaN(result.objective[0])).toBe(true);
  expect(Array.from(result.status)).toEqual([0, 0, 0, 0]);
});

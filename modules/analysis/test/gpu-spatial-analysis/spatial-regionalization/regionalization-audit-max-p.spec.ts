// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUMaxPRegionsOperation} from '../../../src/gpu-spatial-analysis/spatial-regionalization/gpu-max-p-regions-operation';
import {CPUMaxPRegionsOperation} from '../../../src/gpu-spatial-analysis/spatial-regionalization/cpu-max-p-regions-operation';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {
  createAuditMaxPInput,
  createAuditMaxPProps,
  createAuditMaxPSingletons
} from './regionalization-audit-max-p-fixtures';
import {getRegionalizationOperationInputs} from './regionalization-operation-test-utils';
import {
  createMaxPGrowthInput,
  createMaxPRefinementInput
} from './regionalization-operation-test-utils';
import {
  createAuditMaxPRemainderInput,
  evaluateAuditMaxPRemainder
} from './regionalization-audit-max-p-fixtures';

for (const scene of [
  'pairs',
  'island',
  'refined',
  'fallback',
  'fallback-invalid',
  'affine-standardized',
  'selected-start'
]) {
  it(`audits every Max-P output with independent floor and SSD literals: ${scene}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const fallback = scene === 'fallback' || scene === 'fallback-invalid';
    const island = scene === 'island';
    const selectedStart = scene === 'selected-start';
    const input = selectedStart
      ? {...createMaxPGrowthInput(), parameters: Uint32Array.of(2)}
      : fallback
        ? createAuditMaxPSingletons()
        : createAuditMaxPInput(island);
    if (scene === 'fallback-invalid') input.values[0] = NaN;
    if (scene === 'affine-standardized') input.values = input.values.map(value => 3 * value + 64);
    const props = {
      ...createAuditMaxPProps(selectedStart ? 8 : input.values.length),
      columnCount: selectedStart ? 2 : 1,
      standardize: scene === 'affine-standardized',
      maximumGrowthSteps: fallback ? 1 : 64,
      startCount: selectedStart ? 8 : fallback ? 1 : 4,
      maximumIterations: scene === 'refined' ? 4 : 0
    };
    const result = await runOperationGraph(device, {
      inputs: getRegionalizationOperationInputs(input),
      operations: [new GPUMaxPRegionsOperation(props)],
      outputs: Object.values(props.output)
    });
    expect(Array.from(result['labels'])).toEqual(
      selectedStart
        ? [0, 0, 0, 0, 1, 1, 1, 1]
        : fallback
          ? [0, 1, 2, 3, 4, 5, 6, 7]
          : [0, 0, 1, 1, 2, 2, 3, 3, ...(island ? [4] : [])]
    );
    expect(Array.from(result['status'])).toEqual(
      selectedStart
        ? [2, 1, 0, 0, 1, 0]
        : scene === 'fallback-invalid'
          ? [0, 0, 8, 1, 0, 0]
          : fallback
            ? [8, 1, 0, 0, 0, 0]
            : island
              ? [5, 0, 1, 0, 0, 0]
              : [4, 1, 0, 0, 0, 0]
    );
    expect(result['objective']).toHaveLength(1);
    expect(Number.isFinite(result['objective'][0])).toBe(true);
    // Direct NumPy deviations give 28; float32 means/reduction permit 1e-5 absolute.
    expect(
      Math.abs(
        result['objective'][0] -
          (selectedStart ? 150 : fallback ? 0 : props.standardize ? 0.25 : 28)
      )
    ).toBeLessThanOrEqual(1e-5);
  });
}

it('keeps GPU enclave-budget remainders contiguous with independently evaluated outputs', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const result = await runOperationGraph(device, {
    inputs: getRegionalizationOperationInputs(createAuditMaxPRemainderInput()),
    operations: [
      new GPUMaxPRegionsOperation({
        ...createAuditMaxPProps(36),
        startCount: 1,
        maximumGrowthSteps: 4
      })
    ],
    outputs: ['labels', 'objective', 'status']
  });
  const oracle = evaluateAuditMaxPRemainder(result['labels']);
  expect(oracle.connected).toBe(true);
  expect(result['labels'][1]).not.toBe(result['labels'][6]);
  expect(Array.from(new Set(result['labels'])).sort((first, second) => first - second)).toEqual(
    Array.from({length: oracle.regionCount}, (_, row) => row)
  );
  expect(Array.from(result['status'])).toEqual([
    oracle.regionCount,
    0,
    oracle.infeasibleRegionCount,
    0,
    0,
    0
  ]);
  expect(oracle.infeasibleRegionCount).toBeGreaterThan(0);
  expect(result['objective']).toHaveLength(1);
  expect(Number.isFinite(result['objective'][0])).toBe(true);
  // Pairwise-distance SSD is independent of the contributor's means; allow float32 summation.
  expect(Math.abs(result['objective'][0] - oracle.objective)).toBeLessThanOrEqual(
    1e-4 * Math.max(1, oracle.objective)
  );
});

for (const exponent of [-80, 64, 120]) {
  it(`compares all active Max-P refinement ports with NumPy at scale 2^${exponent}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const input = createMaxPRefinementInput(2);
    input.values = input.values.map(value => value * 2 ** exponent);
    for (const Operation of [GPUMaxPRegionsOperation, CPUMaxPRegionsOperation]) {
      const result = await runOperationGraph(device, {
        inputs: getRegionalizationOperationInputs(input),
        operations: [
          new Operation({
            ...createAuditMaxPProps(9),
            standardize: true,
            startCount: 1,
            maximumIterations: 8
          })
        ],
        outputs: ['labels', 'objective', 'status']
      });
      expect(Array.from(result['labels'])).toEqual([0, 1, 1, 0, 1, 2, 0, 2, 2]);
      expect(Array.from(result['status'])).toEqual([3, 1, 0, 0, 0, 1]);
      expect(result['objective']).toHaveLength(1);
      expect(Number.isFinite(result['objective'][0])).toBe(true);
      // Population-standardized NumPy SSD is 7.047; allow 1e-4 relative for float32 kernels.
      expect(Math.abs(result['objective'][0] - 7.047)).toBeLessThanOrEqual(7.047e-4);
    }
  });
}

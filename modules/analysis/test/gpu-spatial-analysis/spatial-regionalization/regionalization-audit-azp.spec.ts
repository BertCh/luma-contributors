// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUAZPRegionsOperation} from '../../../src/gpu-spatial-analysis/spatial-regionalization/gpu-azp-regions-operation';
import {CPUAZPRegionsOperation} from '../../../src/gpu-spatial-analysis/spatial-regionalization/cpu-azp-regions-operation';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {createAuditAZPInput, createAuditAZPProps} from './regionalization-audit-azp-fixtures';
import {getRegionalizationOperationInputs} from './regionalization-operation-test-utils';

for (const scene of ['raw', 'standardized', 'affine', 'floor', 'size', 'budget', 'disconnected']) {
  it(`audits every GPU AZP port against NumPy literals: ${scene}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const input = createAuditAZPInput(scene === 'affine' ? 3 : 1, scene === 'affine' ? 64 : 0);
    const props = {
      ...createAuditAZPProps(),
      standardize: scene === 'standardized' || scene === 'affine',
      minimumRegionSize: scene === 'size' ? 3 : 1,
      maximumIterations: scene === 'budget' ? 1 : 8
    };
    if (scene === 'floor') input.extensiveThreshold![0] = 3;
    if (scene === 'disconnected') input.initialLabels[8] = 0;
    const result = await runOperationGraph(device, {
      inputs: getRegionalizationOperationInputs(input),
      operations: [new GPUAZPRegionsOperation(props)],
      outputs: Object.values(props.output)
    });
    const blocked = scene === 'floor' || scene === 'size';
    const invalid = scene === 'disconnected';
    expect(Array.from(result['labels'])).toEqual(
      blocked || invalid ? Array.from(input.initialLabels) : [0, 0, 1, 1, 1, 1, 2, 2, 3]
    );
    expect(Array.from(result['status'])).toEqual(
      invalid
        ? [0, 0, 0, 1]
        : blocked
          ? [0, 1, 0, 0]
          : scene === 'budget'
            ? [1, 0, 1, 0]
            : [1, 1, 0, 0]
    );
    const objective = invalid
      ? 0
      : blocked
        ? 44.66666666666667
        : props.standardize
          ? 0.028328281650734446
          : 24;
    // Float32 standardization and summation permit 1e-5 relative error (and 1e-6 absolute).
    expect(result['objective']).toHaveLength(1);
    expect(Number.isFinite(result['objective'][0])).toBe(true);
    expect(Math.abs(result['objective'][0] - objective)).toBeLessThanOrEqual(
      Math.max(1e-6, Math.abs(objective) * 1e-5)
    );
  });
}

for (const exponent of [-80, 64, 120]) {
  it(`compares every AZP operation port with NumPy at scale 2^${exponent}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    for (const Operation of [GPUAZPRegionsOperation, CPUAZPRegionsOperation]) {
      const result = await runOperationGraph(device, {
        inputs: getRegionalizationOperationInputs(createAuditAZPInput(2 ** exponent)),
        operations: [new Operation({...createAuditAZPProps(), standardize: true})],
        outputs: ['labels', 'objective', 'status']
      });
      expect(Array.from(result['labels'])).toEqual([0, 0, 1, 1, 1, 1, 2, 2, 3]);
      expect(Array.from(result['status'])).toEqual([1, 1, 0, 0]);
      expect(result['objective']).toHaveLength(1);
      expect(Number.isFinite(result['objective'][0])).toBe(true);
      // Finite-only population standardization and float32 sums: absolute tolerance 1e-6.
      expect(Math.abs(result['objective'][0] - 0.028328281650734446)).toBeLessThanOrEqual(1e-6);
    }
  });
}

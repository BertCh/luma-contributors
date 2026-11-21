// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUSkaterRegionsOperation} from '../../../src/gpu-spatial-analysis/spatial-regionalization/gpu-skater-regions-operation';
import {GPUWardRegionsOperation} from '../../../src/gpu-spatial-analysis/spatial-regionalization/gpu-ward-regions-operation';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {createAuditForest, createAuditWardInput} from './regionalization-audit-twins-fixtures';

function getInputs(input: ReturnType<typeof createAuditForest>) {
  return Object.fromEntries(
    Object.entries(input).map(([name, values]) => [
      name,
      {
        format: values instanceof Float32Array ? ('float32' as const) : ('uint32' as const),
        values
      }
    ])
  );
}

for (const scale of [0, 1, 3]) {
  it(`audits every GPU SKATER port against NumPy cuts at scale ${scale}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const input = createAuditForest();
    input.values = input.values.map(value => value * scale + 64);
    const result = await runOperationGraph(device, {
      inputs: getInputs(input),
      operations: [
        new GPUSkaterRegionsOperation({
          rowCount: 9,
          edgeCount: 12,
          columnCount: 2,
          maximumRegionCount: 6,
          weights: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
          treeEdgeFlags: 'treeEdgeFlags',
          componentLabels: 'componentLabels',
          values: 'values',
          parameters: 'parameters',
          output: {
            labels: 'labels',
            regionCount: 'regionCount',
            cutEdges: 'cutEdges',
            cutGains: 'cutGains'
          }
        })
      ],
      outputs: ['labels', 'regionCount', 'cutEdges', 'cutGains']
    });
    expect(Array.from(result['labels'])).toEqual([0, 0, 2, 2, 4, 4, 6, 6, 8]);
    expect(Array.from(result['regionCount'])).toEqual([5]);
    expect(Array.from(result['cutEdges'])).toEqual([2, 8, 4294967295, 4294967295, 4294967295]);
    expect(result['cutGains']).toHaveLength(5);
    [104 * scale ** 2, 32 * scale ** 2, 0, 0, 0].forEach((expected, index) => {
      expect(Number.isFinite(result['cutGains'][index])).toBe(true);
      // Tour prefix subtraction and float32 gain computation permit 1e-4 relative error.
      expect(Math.abs(result['cutGains'][index] - expected)).toBeLessThanOrEqual(
        1e-4 * Math.max(1, expected)
      );
    });
  });
}

for (const mergeSchedule of ['sequential', 'mutual-nearest'] as const) {
  it(`audits every GPU Ward port against NumPy SSD: ${mergeSchedule}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const input = createAuditWardInput();
    const result = await runOperationGraph(device, {
      inputs: getInputs(input),
      operations: [
        new GPUWardRegionsOperation({
          rowCount: 9,
          targetRegionCount: 5,
          maximumIterations: 4,
          standardize: false,
          mergeSchedule,
          statusLength: mergeSchedule === 'sequential' ? 4 : 5,
          weights: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
          values: 'values',
          parameters: 'parameters',
          output: {labels: 'labels', objective: 'objective', status: 'status'}
        })
      ],
      outputs: ['labels', 'objective', 'status']
    });
    expect(Array.from(result['labels'])).toEqual([0, 0, 2, 2, 4, 4, 6, 6, 8]);
    expect(Array.from(result['status'])).toEqual(
      mergeSchedule === 'sequential' ? [4, 1, 0, 0] : [4, 1, 0, 0, 1]
    );
    expect(result['objective']).toHaveLength(1);
    expect(Number.isFinite(result['objective'][0])).toBe(true);
    // Four small float32 Ward updates permit 1e-4 relative objective error.
    expect(Math.abs(result['objective'][0] - 30)).toBeLessThanOrEqual(0.003);
  });
}

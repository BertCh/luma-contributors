// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {CPUWardRegionsOperation} from '../../../src/gpu-spatial-analysis/spatial-regionalization/cpu-ward-regions-operation';
import {GPUWardRegionsOperation} from '../../../src/gpu-spatial-analysis/spatial-regionalization/gpu-ward-regions-operation';
import {getGPUWardRegionsParameterValues} from '../../../src/gpu-spatial-analysis/spatial-regionalization/gpu-ward-regions';
import {getCSR} from './ward-reference';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {
  createRegionalizationBudgetPath,
  createWardLossPath
} from './regionalization-final1-fixtures';
import {getRegionalizationOperationInputs} from './regionalization-operation-test-utils';

it('spatial-3:F7 keeps the Ward partition unchanged at a zero-round budget on both operations', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const mergeSchedule of ['sequential', 'mutual-nearest'] as const) {
    const props = {
      rowCount: 6,
      weights: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
      values: 'values',
      parameters: 'parameters',
      targetRegionCount: 4,
      standardize: false,
      maximumIterations: 0,
      mergeSchedule,
      output: {labels: 'labels', objective: 'objective', status: 'status'}
    };
    for (const Operation of [GPUWardRegionsOperation, CPUWardRegionsOperation]) {
      const result = await runOperationGraph(device, {
        inputs: getRegionalizationOperationInputs(createRegionalizationBudgetPath()),
        operations: [new Operation(props)],
        outputs: Object.values(props.output)
      });
      expect(Array.from(result['labels'])).toEqual([0, 1, 2, 3, 4, 5]);
      expect(Array.from(result['status'])).toEqual([0, 0, 1, 0]);
      expect(result['objective'][0]).toBe(0);
    }
  }
});

it('spatial-3:G10 reports invalid input without iteration exhaustion at zero rounds on both operations', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const mergeSchedule of ['sequential', 'mutual-nearest'] as const) {
    for (const invalid of [false, true]) {
      const scene = createWardLossPath(false);
      if (invalid) scene.values[1] = NaN;
      const props = {
        rowCount: 3,
        weights: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
        values: 'values',
        parameters: 'parameters',
        targetRegionCount: 2,
        standardize: false,
        maximumIterations: 0,
        mergeSchedule,
        output: {labels: 'labels', objective: 'objective', status: 'status'}
      };
      for (const Operation of [GPUWardRegionsOperation, CPUWardRegionsOperation]) {
        const result = await runOperationGraph(device, {
          inputs: {
            offsets: {format: 'uint32', values: scene.offsets},
            neighbors: {format: 'uint32', values: scene.neighbors},
            weights: {format: 'float32', values: scene.weights},
            values: {format: 'float32', values: scene.values},
            parameters: {format: 'uint32', values: scene.parameters}
          },
          operations: [new Operation(props)],
          outputs: Object.values(props.output)
        });
        expect(Array.from(result['labels'])).toEqual([0, 1, 2]);
        expect(Array.from(result['status'])).toEqual(invalid ? [0, 0, 0, 1] : [0, 0, 1, 0]);
        expect(result['objective'][0]).toBe(0);
      }
    }
  }
});

it('spatial-3:F8 forwards loss flags through both Ward graph operations', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const reason of ['overflow', 'candidateOverflow'] as const) {
    let lossName = 'loss';
    const props = {
      rowCount: 3,
      weights: {
        offsets: 'offsets',
        neighbors: 'neighbors',
        weights: 'weights',
        status: Object.defineProperty({}, reason, {enumerable: false, get: () => lossName})
      },
      values: 'values',
      parameters: 'parameters',
      targetRegionCount: 2,
      standardize: false,
      maximumIterations: 1,
      output: {labels: 'labels', objective: 'objective', status: 'status'}
    };
    const scene = createWardLossPath(true);
    const inputs = {
      offsets: {format: 'uint32' as const, values: scene.offsets},
      neighbors: {format: 'uint32' as const, values: scene.neighbors},
      weights: {format: 'float32' as const, values: scene.weights},
      values: {format: 'float32' as const, values: scene.values},
      parameters: {format: 'uint32' as const, values: scene.parameters},
      loss: {format: 'uint32' as const, values: Uint32Array.of(1)}
    };
    for (const Operation of [GPUWardRegionsOperation, CPUWardRegionsOperation]) {
      lossName = 'loss';
      const operation = new Operation(props);
      lossName = 'unbound';
      const result = await runOperationGraph(device, {
        inputs,
        operations: [operation],
        outputs: Object.values(props.output)
      });
      expect(Array.from(result['labels'])).toEqual([0xffffffff, 0xffffffff, 0xffffffff]);
      expect(Number.isNaN(result['objective'][0])).toBe(true);
      expect(Array.from(result['status'])).toEqual([0, 0, 0, 0]);
    }
  }
});

for (const mergeSchedule of ['sequential', 'mutual-nearest'] as const) {
  for (const scene of ['ordinary', 'disconnected', 'invalid', 'no-merges'] as const) {
    for (const standardize of [false, true]) {
      it(`Ward graph twins agree: ${mergeSchedule}, ${scene}, standardize=${standardize}`, async () => {
        const device = await getWebGPUTestDevice();
        requireTestDevice(device);
        const pattern = getCSR(
          scene === 'disconnected' ? [[1], [0], [3], [2]] : [[1], [0, 2], [1, 3], [2]]
        );
        const values = Float32Array.of(0, 5, 1, 5, 10, 5, 11, scene === 'invalid' ? NaN : 5);
        const targetRegionCount = scene === 'no-merges' ? 4 : scene === 'disconnected' ? 1 : 2;
        const props = {
          weights: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
          rowCount: 4,
          values: 'values',
          parameters: 'parameters',
          columnCount: 2,
          standardize,
          mergeSchedule,
          statusLength: mergeSchedule === 'mutual-nearest' ? (5 as const) : (4 as const),
          targetRegionCount,
          maximumIterations: 4 - targetRegionCount,
          output: {labels: 'labels', objective: 'objective', status: 'status'}
        };
        const results = [];
        for (const Operation of [GPUWardRegionsOperation, CPUWardRegionsOperation]) {
          results.push(
            await runOperationGraph(device, {
              inputs: {
                offsets: {format: 'uint32', values: pattern.offsets},
                neighbors: {format: 'uint32', values: pattern.neighbors},
                weights: {format: 'float32', values: pattern.weights},
                values: {format: 'float32', values},
                parameters: {format: 'uint32', values: getGPUWardRegionsParameterValues({seed: 7})}
              },
              operations: [new Operation(props)],
              outputs: ['labels', 'objective', 'status']
            })
          );
        }
        expect(Array.from(results[0]['labels'])).toEqual(Array.from(results[1]['labels']));
        expect(Array.from(results[0]['status'])).toEqual(Array.from(results[1]['status']));
        // Every output: exact labels/status; objective allows 1e-4 relative (absolute floor 1e-4).
        expect(
          Math.abs(results[0]['objective'][0] - results[1]['objective'][0])
        ).toBeLessThanOrEqual(1e-4 * Math.max(1, Math.abs(results[1]['objective'][0])));
      });
    }
  }
}

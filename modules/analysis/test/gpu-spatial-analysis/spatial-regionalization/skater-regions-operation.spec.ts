// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {CPUSkaterRegionsOperation} from '../../../src/gpu-spatial-analysis/spatial-regionalization/cpu-skater-regions-operation';
import {
  GPUSkaterRegionsOperation,
  type SkaterRegionsOperationProps
} from '../../../src/gpu-spatial-analysis/spatial-regionalization/gpu-skater-regions-operation';
import {getGPUSkaterRegionsParameterValues} from '../../../src/gpu-spatial-analysis/spatial-regionalization/gpu-skater-regions';
import {GPU_MINIMUM_SPANNING_TREE_STATUS} from '../../../src/gpu-spatial-analysis/spatial-regionalization/gpu-spatial-weights-minimum-spanning-tree';
import {runOperationGraph, type OperationGraphInput} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

for (const scene of ['ordinary', 'size-floor', 'no-cuts', 'zero-gain-tie', 'invalid-tree'] as const)
  it(`SKATER CPU/GPU compare every output: ${scene}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const props: SkaterRegionsOperationProps = {
      rowCount: 8,
      edgeCount: 12,
      columnCount: 2,
      maximumRegionCount: 5,
      weights: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
      treeEdgeFlags: 'treeEdgeFlags',
      componentLabels: 'componentLabels',
      values: 'values',
      parameters: 'parameters',
      treeStatus: 'treeStatus',
      output: {
        labels: 'labels',
        regionCount: 'regionCount',
        cutEdges: 'cutEdges',
        cutGains: 'cutGains'
      }
    };
    const treeStatus = new Uint32Array(GPU_MINIMUM_SPANNING_TREE_STATUS.length);
    if (scene === 'invalid-tree') treeStatus[GPU_MINIMUM_SPANNING_TREE_STATUS.invalidCount] = 1;
    const inputs: Record<string, OperationGraphInput> = {
      offsets: {format: 'uint32', values: Uint32Array.of(0, 1, 3, 5, 6, 7, 9, 11, 12)},
      neighbors: {format: 'uint32', values: Uint32Array.of(1, 0, 2, 1, 3, 2, 5, 4, 6, 5, 7, 6)},
      weights: {format: 'float32', values: new Float32Array(12).fill(1)},
      treeEdgeFlags: {format: 'uint32', values: Uint32Array.of(1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0)},
      componentLabels: {format: 'uint32', values: Uint32Array.of(0, 0, 0, 0, 4, 4, 4, 4)},
      values: {
        format: 'float32',
        values:
          scene === 'zero-gain-tie'
            ? new Float32Array(16)
            : Float32Array.of(0, 0, 0, 0, 10, 2, 10, 2, 1, -1, 1, -1, 5, 3, 5, 3)
      },
      parameters: {
        format: 'uint32',
        values: getGPUSkaterRegionsParameterValues({
          regionCount: scene === 'no-cuts' ? 1 : scene === 'ordinary' ? 4 : 5,
          minimumRegionSize: scene === 'size-floor' ? 2 : 1
        })
      },
      treeStatus: {format: 'uint32', values: treeStatus}
    };
    const outputs = Object.values(props.output) as string[];
    const gpu = await runOperationGraph(device, {
      inputs,
      operations: [new GPUSkaterRegionsOperation(props)],
      outputs
    });
    const cpu = await runOperationGraph(device, {
      inputs,
      operations: [new CPUSkaterRegionsOperation(props)],
      outputs
    });
    for (const output of outputs) {
      expect(gpu[output].length).toBe(cpu[output].length);
      for (let slot = 0; slot < cpu[output].length; slot++) {
        const expected = cpu[output][slot];
        if (output !== 'cutGains' || expected === 0)
          expect(gpu[output][slot] === expected, `${output}[${slot}]`).toBe(true);
        // Gains are positive; relative tolerance 2e-5 cannot let a small result pass as zero.
        else expect(Math.abs(gpu[output][slot] - expected) / expected).toBeLessThan(2e-5);
      }
    }
  });

for (const scene of ['valid', 'rowCount', 'transform', 'symmetric'] as const)
  it(`SKATER review metadata regression: ${scene}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const information = {
      rowCount: scene === 'rowCount' ? 3 : 2,
      transform: scene === 'transform' ? ('invalid' as 'raw') : ('raw' as const),
      symmetric: scene === 'symmetric' ? ('invalid' as 'unknown') : true
    };
    const props: SkaterRegionsOperationProps = {
      id: 'review-skater',
      rowCount: 2,
      edgeCount: 2,
      maximumRegionCount: 2,
      weights: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights', information},
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
    };
    const inputs: Record<string, OperationGraphInput> = {
      offsets: {format: 'uint32', values: Uint32Array.of(0, 1, 2)},
      neighbors: {format: 'uint32', values: Uint32Array.of(1, 0)},
      weights: {format: 'float32', values: Float32Array.of(1, 1)},
      treeEdgeFlags: {format: 'uint32', values: Uint32Array.of(1, 0)},
      componentLabels: {format: 'uint32', values: Uint32Array.of(0, 0)},
      values: {format: 'float32', values: Float32Array.of(0, 2)},
      parameters: {
        format: 'uint32',
        values: getGPUSkaterRegionsParameterValues({regionCount: 2, minimumRegionSize: 1})
      }
    };
    for (const createOperation of [
      () => new GPUSkaterRegionsOperation(props),
      () => new CPUSkaterRegionsOperation(props)
    ]) {
      const run = async () =>
        runOperationGraph(device, {
          inputs,
          operations: [createOperation()],
          outputs: ['labels', 'regionCount', 'cutEdges', 'cutGains']
        });
      if (scene !== 'valid')
        await expect(run()).rejects.toThrow(
          new RegExp(`^review-skater weights\\.information\\.${scene}`)
        );
      else {
        const result = await run();
        expect(Array.from(result['labels'])).toEqual([0, 1]);
        expect(Array.from(result['regionCount'])).toEqual([2]);
        expect(Array.from(result['cutEdges'])).toEqual([0]);
        // SSD of [0,2] is 2, and each singleton has zero SSD. Relative tolerance 2e-5.
        expect(result['cutGains'].length).toBe(1);
        expect(Math.abs(result['cutGains'][0] - 2) / 2).toBeLessThan(2e-5);
      }
    }
  });

// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {CPUTerrainDepressionHierarchyOperation} from '../../../src/gpu-terrain/terrain-processing/cpu-terrain-depression-hierarchy-operation';
import {GPUTerrainDepressionHierarchyOperation} from '../../../src/gpu-terrain/terrain-processing/gpu-terrain-depression-hierarchy-operation';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {
  makeHierarchyInput,
  makeHierarchyProps
} from './terrain-depression-hierarchy-operation-fixture';

it('compares every hierarchy output on merges, overflow, masked input and ground-scaled curves', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const scene of ['ordinary', 'overflow', 'empty', 'barrier', 'geographic'] as const) {
    const input = makeHierarchyInput(),
      props = makeHierarchyProps(scene === 'overflow' ? 2 : 8);
    if (scene === 'empty') input.validity.fill(0);
    if (scene === 'barrier') {
      input.validity[3] = 0;
      input.stages[2] = NaN;
    }
    if (scene === 'geographic') {
      props.cellSizeMode = 'geographic';
      props.grid = {...props.grid, affine: [0.01, 0, 0, 0, -0.01, 60]};
    }
    const outputs = [
      'nodes',
      'levels',
      'cellNodes',
      'curves',
      'count',
      'requiredCount',
      'overflow'
    ];
    const results = [];
    for (const Operation of [
      GPUTerrainDepressionHierarchyOperation,
      CPUTerrainDepressionHierarchyOperation
    ])
      results.push(
        await runOperationGraph(device, {
          inputs: {
            elevation: {format: 'float32', values: input.elevation},
            outlets: {format: 'uint32', values: input.outlets},
            stages: {format: 'float32', values: input.stages},
            validity: {format: 'uint32', values: input.validity}
          },
          operations: [new Operation(props)],
          outputs
        })
      );
    for (const port of outputs) {
      expect(results[0][port].length).toBe(results[1][port].length);
      for (let index = 0; index < results[1][port].length; index++) {
        const actual = results[0][port][index],
          expected = results[1][port][index];
        if (Number.isNaN(expected)) expect(actual).toBeNaN();
        else if (port !== 'curves' || expected === 0) expect(actual === expected).toBe(true);
        // Only curve arithmetic is approximate; node IDs, copied levels and all status ports are exact.
        else expect(Math.abs(actual - expected)).toBeLessThanOrEqual(Math.abs(expected) * 2e-5);
      }
    }
  }
});

it('preserves exact activation IDs on plateaus, multifurcations, late outlets and disconnected padding', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const scene of ['plateau', 'merges', 'barriers', 'single', 'overflow'] as const) {
    const width = scene === 'single' ? 1 : 17;
    const height = scene === 'single' ? 1 : 5;
    const count = width * height;
    const props = {
      ...makeHierarchyProps(scene === 'overflow' ? 1 : count + 3),
      grid: {...makeHierarchyProps().grid, width, height},
      stageCount: 2
    };
    // Integer heights provide exact float32 comparisons and many equal-height activation ties.
    const elevation = Float32Array.from({length: count}, (_, cell) =>
      scene === 'plateau' ? 0 : ((cell * 17 + Math.floor(cell / width) * 3) % 11) - 5
    );
    const validity = Uint32Array.from({length: count}, (_, cell) =>
      scene === 'barriers' && (cell % width === 8 || cell >= count - 3) ? 0 : 1
    );
    const outlets = Uint32Array.from({length: count}, (_, cell) => (cell % 7 === 3 ? 9 : 0));
    const inputs = {
      elevation: {format: 'float32' as const, values: elevation},
      validity: {format: 'uint32' as const, values: validity},
      outlets: {format: 'uint32' as const, values: outlets},
      stages: {format: 'float32' as const, values: Float32Array.of(0, 6)}
    };
    const outputs = [
      'nodes',
      'levels',
      'cellNodes',
      'curves',
      'count',
      'requiredCount',
      'overflow'
    ];
    const expected = await runOperationGraph(device, {
      inputs,
      operations: [new CPUTerrainDepressionHierarchyOperation(props)],
      outputs
    });
    // Repeat the merge case to exercise scheduling-independent atomic splicing and metadata.
    for (let repetition = 0; repetition < (scene === 'merges' ? 3 : 1); repetition++) {
      const actual = await runOperationGraph(device, {
        inputs,
        operations: [new GPUTerrainDepressionHierarchyOperation(props)],
        outputs
      });
      for (const port of outputs)
        expect(Array.from(actual[port]), `${scene} ${port}`).toEqual(Array.from(expected[port]));
    }
  }
});

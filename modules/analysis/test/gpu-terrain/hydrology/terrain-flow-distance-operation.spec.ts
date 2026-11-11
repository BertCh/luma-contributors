// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {CPUTerrainFlowDistanceOperation} from '../../../src/gpu-terrain/hydrology/cpu-terrain-flow-distance-operation';
import {
  GPUTerrainFlowDistanceOperation,
  type TerrainFlowDistanceOperationProps
} from '../../../src/gpu-terrain/hydrology/gpu-terrain-flow-distance-operation';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

for (const maximumSteps of [3, 2]) {
  it(`flow-distance operations sum a two-edge chain, maximumSteps=${maximumSteps}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const props: TerrainFlowDistanceOperationProps = {
      grid: {width: 3, height: 1, pixelInterpretation: 'area', affine: [1, 0, 0, 0, -1, 0]},
      flowDirections: 'directions',
      transitionCosts: 'costs',
      maximumSteps,
      output: {distance: 'distance', validity: 'validity', terminalCells: 'terminalCells'}
    };
    const inputs = {
      directions: {format: 'uint32' as const, values: new Uint32Array([1, 1, 0])},
      costs: {format: 'float32' as const, values: new Float32Array([2, 3, NaN])}
    };
    const outputs = ['distance', 'validity', 'terminalCells'];
    const gpu = await runOperationGraph(device, {
      inputs,
      operations: [new GPUTerrainFlowDistanceOperation(props)],
      outputs
    });
    const cpu = await runOperationGraph(device, {
      inputs,
      operations: [new CPUTerrainFlowDistanceOperation(props)],
      outputs
    });
    // These integer costs and their sum are exact in float32: tolerance zero on every output.
    for (const name of outputs) expect(Array.from(gpu[name])).toEqual(Array.from(cpu[name]));
    for (const rows of [gpu, cpu]) {
      expect(Array.from(rows['distance'])).toEqual([maximumSteps === 3 ? 5 : NaN, 3, 0]);
      expect(Array.from(rows['validity'])).toEqual([maximumSteps === 3 ? 1 : 0, 1, 1]);
      expect(Array.from(rows['terminalCells'])).toEqual([
        maximumSteps === 3 ? 2 : 0xfffffffe,
        2,
        2
      ]);
    }
  });
}

for (const testCase of [
  'uniform',
  'geographic',
  'web-mercator',
  'weighted',
  'unresolved'
] as const) {
  it(`flow-distance operations agree on ${testCase}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const weighted = testCase === 'weighted';
    const props: TerrainFlowDistanceOperationProps = {
      grid: {width: 2, height: 2, pixelInterpretation: 'area', affine: [0.01, 0, 0, 0, -0.01, 40]},
      flowDirections: 'directions',
      streams: weighted ? 'streams' : undefined,
      transitionCosts: weighted ? 'costs' : undefined,
      cellSizeMode: testCase === 'geographic' || testCase === 'web-mercator' ? testCase : 'uniform',
      output: {distance: 'distance', validity: 'validity', terminalCells: 'terminalCells'}
    };
    const inputs = {
      directions: {
        format: 'uint32' as const,
        values: new Uint32Array(
          testCase === 'unresolved' ? [1, 16, 0xffffffff, 0xfffffffe] : [2, 4, 1, 0]
        )
      },
      ...(weighted
        ? {
            streams: {format: 'uint32' as const, values: new Uint32Array([0, 1, 0, 0])},
            costs: {format: 'float32' as const, values: new Float32Array([2.5, -1, 7, NaN])}
          }
        : {})
    };
    const outputs = ['distance', 'validity', 'terminalCells'];
    const gpu = await runOperationGraph(device, {
      inputs,
      operations: [new GPUTerrainFlowDistanceOperation(props)],
      outputs
    });
    const cpu = await runOperationGraph(device, {
      inputs,
      operations: [new CPUTerrainFlowDistanceOperation(props)],
      outputs
    });
    expect(Array.from(gpu['validity'])).toEqual(Array.from(cpu['validity']));
    expect(Array.from(gpu['terminalCells'])).toEqual(Array.from(cpu['terminalCells']));
    for (let row = 0; row < 4; row++) {
      const expected = cpu['distance'][row];
      if (Number.isNaN(expected)) expect(gpu['distance'][row]).toBeNaN();
      // f32 georeference evaluation and sums: 2e-5 relative, with 1e-6 absolute floor.
      else
        expect(Math.abs(gpu['distance'][row] - expected)).toBeLessThanOrEqual(
          2e-5 * Math.max(0.05, Math.abs(expected))
        );
    }
  });
}

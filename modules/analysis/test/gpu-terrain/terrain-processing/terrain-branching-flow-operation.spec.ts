// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {CPUTerrainBranchingFlowOperation} from '../../../src/gpu-terrain/terrain-processing/cpu-terrain-branching-flow-operation';
import {
  GPUTerrainBranchingFlowOperation,
  type TerrainBranchingFlowOperationProps
} from '../../../src/gpu-terrain/terrain-processing/gpu-terrain-branching-flow-operation';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {makeBranchingInput, makeBranchingProps} from './terrain-branching-flow-operation-fixture';

for (const mode of [
  'transport',
  'expected-distance',
  'minimum-distance',
  'maximum-distance',
  'fractional-catchment'
] as const) {
  it(`compares every ${mode} output on a weighted split/merge and a cycle`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    for (const scene of ['ordinary', 'cycle', 'masked', 'affine', 'geographic'] as const) {
      const input = makeBranchingInput();
      input.values =
        mode === 'fractional-catchment' ? Float32Array.of(0, 0, 0.25, 0, 0, 0.75) : input.values;
      if (scene === 'cycle') {
        input.fractions.fill(0);
        input.fractions[0] = 1;
        input.fractions[12] = 1;
      }
      const props: TerrainBranchingFlowOperationProps = {
        ...makeBranchingProps(mode),
        validity: 'validity'
      };
      if (scene === 'affine' || scene === 'geographic') props.edgeLengths = undefined;
      if (scene === 'geographic') {
        props.cellSizeMode = 'geographic';
        props.grid = {...props.grid, affine: [0.01, 0, 0, 0, -0.01, 60]};
      }
      const validity = Uint32Array.of(scene === 'masked' ? 0 : 1, 1, 1, 1, 1, 1);
      const results = [];
      for (const Operation of [
        GPUTerrainBranchingFlowOperation,
        CPUTerrainBranchingFlowOperation
      ]) {
        results.push(
          await runOperationGraph(device, {
            inputs: {
              fractions: {format: 'float32', values: input.fractions},
              values: {format: 'float32', values: input.values},
              ...(props.edgeLengths !== undefined
                ? {edgeLengths: {format: 'float32' as const, values: input.edgeLengths}}
                : {}),
              validity: {format: 'uint32', values: validity}
            },
            operations: [new Operation(props)],
            outputs: ['result', 'status', 'unresolvedCount']
          })
        );
      }
      expect(Array.from(results[0]['status'])).toEqual(Array.from(results[1]['status']));
      expect(Array.from(results[0]['unresolvedCount'])).toEqual(
        Array.from(results[1]['unresolvedCount'])
      );
      for (let cell = 0; cell < 6; cell++) {
        const actual = results[0]['result'][cell],
          expected = results[1]['result'][cell];
        if (Number.isNaN(expected)) expect(actual).toBeNaN();
        else if (expected === 0) expect(actual === 0).toBe(true);
        // Relative 2e-5: small positive results cannot pass as zero; zero sign is immaterial.
        else expect(Math.abs(actual - expected)).toBeLessThanOrEqual(2e-5 * Math.abs(expected));
      }
    }
  });
}

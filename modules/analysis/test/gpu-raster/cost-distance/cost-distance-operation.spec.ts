// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {makeGPURasterMetadata} from '../../../src/raster-analysis/raster-grid/index';
import {expect, it} from 'vitest';
import {CPUCostDistanceOperation} from '../../../src/raster-analysis/cost-distance/cpu-cost-distance-operation';
import {
  GPUCostDistanceOperation,
  type CostDistanceOperationProps
} from '../../../src/raster-analysis/cost-distance/gpu-cost-distance-operation';
import {getGPUCostDistanceParameterValues} from '../../../src/raster-analysis/cost-distance/gpu-cost-distance';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

for (const empty of [false, true])
  it(`cost-distance operations agree with ${empty ? 'no active sources' : 'competing sources and a plateau'}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const grid = makeGPURasterMetadata({width: 5, height: 1});
    const props: CostDistanceOperationProps = {
      grid,
      friction: {id: 'friction', format: 'float32', storage: {kind: 'buffer', values: 'friction'}},
      parameters: 'parameters',
      sources: 'sources',
      sourceCosts: 'sourceCosts',
      sourceCount: 'sourceCount',

      costBreaks: 'costBreaks',
      costBreakCount: 3,
      output: {
        costs: 'costs',
        backLinks: 'backLinks',
        bands: 'bands',
        bandCounts: 'bandCounts',
        status: {converged: 'converged', iterationLimitReached: 'iterationLimitReached'}
      }
    };
    const results = [];
    for (const Operation of [GPUCostDistanceOperation, CPUCostDistanceOperation])
      results.push(
        await runOperationGraph(device, {
          inputs: {
            friction: {format: 'float32', values: Float32Array.of(2, 0, 0, 4, 2)},
            parameters: {format: 'float32', values: getGPUCostDistanceParameterValues({}, grid)},
            sources: {format: 'uint32', values: Uint32Array.of(0, 4)},
            sourceCosts: {format: 'float32', values: Float32Array.of(0, 0.5)},
            sourceCount: {format: 'uint32', values: Uint32Array.of(empty ? 0 : 2)},
            costBreaks: {format: 'float32', values: Float32Array.of(0.5, 1, 4)}
          },
          operations: [new Operation(props)],
          outputs: [
            'costs',
            'backLinks',
            'bands',
            'bandCounts',
            'converged',
            'iterationLimitReached'
          ]
        })
      );
    for (const port of Object.keys(results[0]))
      for (let index = 0; index < results[0][port].length; index++) {
        const expected = results[1][port][index];
        const actual = results[0][port][index];
        if (port === 'costs' && Number.isFinite(expected) && expected !== 0)
          expect(Math.abs(actual - expected) / Math.abs(expected)).toBeLessThan(3e-5);
        else expect(actual === expected).toBe(true);
      }
  });

for (const scene of [
  'published-overflow',
  'all-sources',
  'legacy-source-capacity',
  'diagnostics'
] as const)
  it(`cost-distance operations support ${scene}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const width = scene === 'published-overflow' ? 5 : 2;
    const grid = makeGPURasterMetadata({width, height: 1});
    const props: CostDistanceOperationProps = {
      ...(scene === 'legacy-source-capacity' ? {sourceCapacity: 1} : {}),
      grid,
      friction: {id: 'friction', format: 'float32', storage: {kind: 'buffer', values: 'friction'}},
      parameters: 'parameters',
      sources: 'sources',
      costBreaks: 'costBreaks',
      costBreakCount: 1,
      output: {
        costs: 'costs',
        backLinks: 'backLinks',
        bands: 'bands',
        bandCounts: 'bandCounts',
        ...(scene === 'diagnostics'
          ? {settledAt: 'settledAt', status: {iterationCount: 'iterationCount'}}
          : {})
      }
    };
    const outputs = [
      'costs',
      'backLinks',
      'bands',
      'bandCounts',
      ...(scene === 'diagnostics' ? ['settledAt', 'iterationCount'] : [])
    ];
    const results = [];
    for (const Operation of [GPUCostDistanceOperation, CPUCostDistanceOperation])
      results.push(
        await runOperationGraph(device, {
          inputs: {
            friction: {
              format: 'float32',
              values: new Float32Array(width).fill(scene === 'published-overflow' ? 1e38 : 1)
            },
            parameters: {format: 'float32', values: getGPUCostDistanceParameterValues({}, grid)},
            sources: {
              format: 'uint32',
              values:
                scene === 'all-sources' || scene === 'legacy-source-capacity'
                  ? Uint32Array.of(0, 1)
                  : Uint32Array.of(0)
            },
            costBreaks: {format: 'float32', values: Float32Array.of(Infinity)}
          },
          operations: [new Operation(props)],
          outputs
        })
      );
    for (const result of results) {
      if (scene === 'published-overflow') {
        expect(result['costs'][4]).toBe(Infinity);
        expect(result['backLinks'][4]).toBe(0xffffffff);
        expect(result['bands'][4]).toBe(0xffffffff);
        expect(Array.from(result['bandCounts'])).toEqual([4]);
      } else if (scene === 'diagnostics') {
        expect(Array.from(result['settledAt'])).toEqual([0, 1]);
        expect(Array.from(result['iterationCount'])).toEqual([2]);
      } else expect(Array.from(result['costs'])).toEqual([0, 0]);
    }
    for (const port of outputs)
      for (let index = 0; index < results[0][port].length; index++) {
        const actual = results[0][port][index];
        const expected = results[1][port][index];
        if (port === 'costs' && Number.isFinite(expected) && expected !== 0)
          expect(Math.abs(actual - expected) / Math.abs(expected)).toBeLessThan(3e-5);
        else expect(actual === expected).toBe(true);
      }
  });

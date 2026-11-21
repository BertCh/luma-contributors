// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {
  GPUCostDistance,
  getGPUCostDistanceParameterValues
} from '../../../src/raster-analysis/cost-distance/gpu-cost-distance';
import {GPUCostDistanceOperation} from '../../../src/raster-analysis/cost-distance/gpu-cost-distance-operation';
import {CPUCostDistanceOperation} from '../../../src/raster-analysis/cost-distance/cpu-cost-distance-operation';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {createParityGraph} from '../../oracles/parity-wp2-geodesic-harness';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {readFloat32, readUint32, submitGraph} from '../../utils/gpu-contributor-test-utils';

for (const maximumIterations of [1, 2]) {
  it(`cost-distance forms pin all stopping words after one edge relaxation (budget=${maximumIterations})`, async () => {
    const device = await getWebGPUTestDevice('core');
    requireTestDevice(device);
    const grid = {
      width: 2,
      height: 1,
      affine: [1, 0, 0, 0, 1, 0] as const,
      pixelInterpretation: 'area' as const
    };
    const parameters = getGPUCostDistanceParameterValues({}, grid);
    const rig = createParityGraph(device);
    const costs = rig.output('costs', 2, 'float32');
    const bandCounts = rig.output('counts', 2, 'uint32');
    const count = rig.output('iterations', 1, 'uint32');
    const converged = rig.output('done', 1, 'uint32');
    const limit = rig.output('limit', 1, 'uint32');
    rig.graph.add(
      new GPUCostDistance({
        grid,
        maximumIterations,
        maximumTieIterations: 2,
        friction: {
          id: 'friction',
          format: 'float32',
          storage: {kind: 'buffer', values: rig.input('friction', [1, 1], 'float32').view}
        },
        parameters: rig.input('parameters', Array.from(parameters), 'float32').view,
        sources: rig.input('sources', [0], 'uint32').view,
        costBreaks: rig.input('breaks', [0.5, Infinity], 'float32').view,
        output: {
          costs: costs.view,
          bands: rig.output('bands', 2, 'uint32').view,
          bandCounts: bandCounts.view,
          backLinks: rig.output('links', 2, 'uint32').view,
          status: {
            iterationCount: count.view,
            converged: converged.view,
            iterationLimitReached: limit.view
          }
        }
      })
    );
    // A single edge changes in round one; round two must certify no decrease. This tiny
    // one-tile fixture has no scheduling ambiguity. All discrete comparisons are exact.
    const expectedStatus = maximumIterations === 1 ? [1, 0, 1] : [2, 1, 0];
    const expectedCounts = maximumIterations === 1 ? [0xfffffffe, 0xfffffffe] : [1, 1];
    const expectedCosts = maximumIterations === 1 ? [NaN, NaN] : [0, 1];
    const compiled = rig.graph.compile();
    try {
      for (const port of [count, converged, limit]) port.buffer.write(Uint32Array.of(12345));
      bandCounts.buffer.write(Uint32Array.of(12345, 12345));
      submitGraph(device, compiled, undefined);
      expect([
        (await readUint32(count.buffer, 1))[0],
        (await readUint32(converged.buffer, 1))[0],
        (await readUint32(limit.buffer, 1))[0]
      ]).toEqual(expectedStatus);
      expect(await readUint32(bandCounts.buffer, 2)).toEqual(expectedCounts);
      expect(await readFloat32(costs.buffer, 2)).toEqual(expectedCosts);
    } finally {
      compiled.destroy();
      rig.destroy();
    }
    for (const Operation of [GPUCostDistanceOperation, CPUCostDistanceOperation]) {
      const result = await runOperationGraph(device, {
        inputs: {
          friction: {format: 'float32', values: Float32Array.of(1, 1)},
          parameters: {format: 'float32', values: parameters},
          sources: {format: 'uint32', values: Uint32Array.of(0)},
          breaks: {format: 'float32', values: Float32Array.of(0.5, Infinity)}
        },
        operations: [
          new Operation({
            grid,
            maximumIterations,
            maximumTieIterations: 2,
            friction: {
              id: 'friction',
              format: 'float32',
              storage: {kind: 'buffer', values: 'friction'}
            },
            sources: 'sources',
            parameters: 'parameters',
            costBreaks: 'breaks',
            costBreakCount: 2,
            output: {
              costs: 'costs',
              bands: 'bands',
              bandCounts: 'counts',
              backLinks: 'links',
              status: {
                iterationCount: 'iterations',
                converged: 'done',
                iterationLimitReached: 'limit'
              }
            }
          })
        ],
        outputs: ['costs', 'counts', 'iterations', 'done', 'limit']
      });
      expect([result['iterations'][0], result['done'][0], result['limit'][0]]).toEqual(
        expectedStatus
      );
      expect(Array.from(result['counts'])).toEqual(expectedCounts);
      expect(Array.from(result['costs'])).toEqual(expectedCosts);
    }
  });
}

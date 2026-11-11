// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {CPUMapColoringOperation} from '../../../src/gpu-spatial-analysis/map-coloring/cpu-map-coloring-operation';
import {
  GPUMapColoringOperation,
  type MapColoringOperationProps
} from '../../../src/gpu-spatial-analysis/map-coloring/gpu-map-coloring-operation';
import {getGPUMapColoringParameterValues} from '../../../src/gpu-spatial-analysis/map-coloring/gpu-map-coloring';
import {createRookLatticeCSR} from './map-coloring-oracle';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

for (const priority of ['random', 'largest-degree-first'] as const) {
  for (const isolates of [false, true]) {
    it(`map coloring graph twins agree exactly: ${priority}, isolates=${isolates}`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const pattern = createRookLatticeCSR(4, 3);
      const props: MapColoringOperationProps = {
        weights: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
        parameters: 'parameters',
        rowCount: 12,
        priority,
        maximumIterations: 64,
        output: {
          colors: 'colors',
          colorCount: 'colorCount',
          conflictCount: 'conflictCount',
          status: {converged: 'converged', ...(isolates ? {iterationCount: 'iterationCount'} : {})}
        }
      };
      // Compare every exposed output exactly (all uint32). Expose round count only for isolates,
      // where both complete in one sweep; GPU physical round counts otherwise depend on scheduling.
      const outputs = [
        'colors',
        'colorCount',
        'conflictCount',
        'converged',
        ...(isolates ? ['iterationCount'] : [])
      ];
      const results = [];
      for (const Operation of [GPUMapColoringOperation, CPUMapColoringOperation]) {
        results.push(
          await runOperationGraph(device, {
            inputs: {
              offsets: {
                format: 'uint32',
                values: isolates ? new Uint32Array(13) : Uint32Array.from(pattern.offsets)
              },
              neighbors: {format: 'uint32', values: Uint32Array.from(pattern.neighbors)},
              weights: {format: 'float32', values: Float32Array.from(pattern.weights)},
              parameters: {format: 'uint32', values: getGPUMapColoringParameterValues({seed: 19})}
            },
            operations: [new Operation(props)],
            outputs
          })
        );
      }
      for (const output of outputs)
        expect(Array.from(results[0][output]), output).toEqual(Array.from(results[1][output]));
      expect(results[0]['conflictCount'][0]).toBe(0);
      expect(results[0]['converged'][0]).toBe(1);
    });
  }
}

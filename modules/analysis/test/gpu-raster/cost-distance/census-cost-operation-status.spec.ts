// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {it} from 'vitest';
import {GPUCostDistanceOperation} from '../../../src/raster-analysis/cost-distance/gpu-cost-distance-operation';
import {CPUCostDistanceOperation} from '../../../src/raster-analysis/cost-distance/cpu-cost-distance-operation';
import {getGPUCostDistanceParameterValues} from '../../../src/raster-analysis/cost-distance/gpu-cost-distance';
import {makeGPURasterMetadata} from '../../../src/raster-analysis/raster-grid/index';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {expectCensusValues} from '../census-raster-oracle-utils';

for (const withAuxiliary of [false, true])
  for (const Operation of [GPUCostDistanceOperation, CPUCostDistanceOperation]) {
    it(`census ${Operation.name} has explicit complete costs, bands and stopping flags (auxiliary=${withAuxiliary})`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const grid = makeGPURasterMetadata({width: 3, height: 1});
      const result = await runOperationGraph(device, {
        inputs: {
          friction: {format: 'float32', values: Float32Array.of(1, 1, 1)},
          sources: {format: 'uint32', values: Uint32Array.of(0)},
          breaks: {format: 'float32', values: Float32Array.of(0.5, 1.5, Infinity)},
          parameters: {format: 'float32', values: getGPUCostDistanceParameterValues({}, grid)}
        },
        operations: [
          new Operation({
            grid,
            friction: {
              id: 'friction',
              format: 'float32',
              storage: {kind: 'buffer', values: 'friction'}
            },
            sources: 'sources',
            parameters: 'parameters',
            ...(withAuxiliary ? {costBreaks: 'breaks', costBreakCount: 3} : {}),
            output: {
              costs: 'costs',
              ...(withAuxiliary
                ? {backLinks: 'backLinks', bands: 'bands', bandCounts: 'bandCounts'}
                : {}),
              status: {converged: 'converged', iterationLimitReached: 'limit'}
            }
          })
        ],
        outputs: [
          'costs',
          'converged',
          'limit',
          ...(withAuxiliary ? ['backLinks', 'bands', 'bandCounts'] : [])
        ]
      });
      // A three-cell unit-friction line has unique shortest paths of length 0, 1, 2.
      // Upper bounds .5,1.5,Infinity partition it into singleton bands; west is direction bit 16.
      expectCensusValues(result['costs'], [0, 1, 2]);
      if (withAuxiliary) {
        expectCensusValues(result['backLinks'], [0, 16, 16]);
        expectCensusValues(result['bands'], [0, 1, 2]);
        expectCensusValues(result['bandCounts'], [1, 1, 1]);
      }
      expectCensusValues(result['converged'], [1]);
      expectCensusValues(result['limit'], [0]);
    });
  }

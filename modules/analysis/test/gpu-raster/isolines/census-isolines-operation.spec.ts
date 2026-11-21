// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {it} from 'vitest';
import {GPUIsolinesOperation} from '../../../src/raster-analysis/isolines/gpu-isolines-operation';
import {CPUIsolinesOperation} from '../../../src/raster-analysis/isolines/cpu-isolines-operation';
import {getGPUIsolinesParameterValues} from '../../../src/raster-analysis/isolines/isolines-parameters';
import {makeGPURasterMetadata} from '../../../src/raster-analysis/raster-grid/index';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {expectCensusValues} from '../census-raster-oracle-utils';

for (const vertexCapacity of [0, 2, 3]) {
  for (const Operation of [GPUIsolinesOperation, CPUIsolinesOperation]) {
    it(`census ${Operation.name} stitches an exact ramp (vertexCapacity=${vertexCapacity})`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const grid = makeGPURasterMetadata({width: 2, height: 3});
      const result = await runOperationGraph(device, {
        inputs: {
          values: {format: 'float32', values: Float32Array.of(0, 2, 0, 2, 0, 2)},
          levels: {format: 'float32', values: Float32Array.of(1)},
          parameters: {
            format: 'float32',
            values: getGPUIsolinesParameterValues({levelCount: 1, extent: [0, 0, 2, 3]}, grid)
          }
        },
        operations: [
          new Operation({
            grid,
            segmentCapacity: 2,
            values: 'values',
            levels: 'levels',
            parameters: 'parameters',
            output: {
              segments: 'segments',
              segmentLevels: 'segmentLevels',
              status: {
                count: 'count',
                overflow: 'overflow',
                ...(vertexCapacity ? {polylineOverflow: 'polylineOverflow'} : {})
              },
              ...(vertexCapacity
                ? {
                    polylines: {
                      vertexCapacity,
                      vertices: 'vertices',
                      polylineOffsets: 'offsets',
                      polylineLevels: 'polylineLevels',
                      polylineClosed: 'closed',
                      polylineCount: 'polylineCount',
                      vertexCount: 'vertexCount'
                    }
                  }
                : {})
            }
          })
        ],
        outputs: [
          'segments',
          'segmentLevels',
          'count',
          'overflow',
          ...(vertexCapacity
            ? [
                'vertices',
                'offsets',
                'polylineLevels',
                'closed',
                'polylineCount',
                'vertexCount',
                'polylineOverflow'
              ]
            : [])
        ]
      });
      // Linear interpolation of 0 -> 2 at level 1 gives x=1. Each marching cell emits
      // one south-directed segment (the high east side is on its left), joined at y=1.5.
      // All coordinates are dyadic; topology and coordinates are exact, not CPU-twin parity.
      expectCensusValues(result['segments'], [1, 1.5, 1, 0.5, 1, 2.5, 1, 1.5]);
      expectCensusValues(result['segmentLevels'], [0, 0]);
      expectCensusValues(result['count'], [2]);
      expectCensusValues(result['overflow'], [0]);
      if (vertexCapacity) {
        const fits = vertexCapacity === 3;
        expectCensusValues(result['polylineCount'], [fits ? 1 : 0]);
        expectCensusValues(result['vertexCount'], [fits ? 3 : 0]);
        expectCensusValues(result['polylineOverflow'], [fits ? 0 : 1]);
        if (fits) {
          expectCensusValues(result['vertices'], [1, 2.5, 1, 1.5, 1, 0.5]);
          expectCensusValues(result['offsets'].slice(0, 2), [0, 3]);
          expectCensusValues(result['polylineLevels'].slice(0, 1), [0]);
          expectCensusValues(result['closed'].slice(0, 1), [0]);
        }
      }
    });
  }
}

for (const Operation of [GPUIsolinesOperation, CPUIsolinesOperation]) {
  it(`census ${Operation.name} distinguishes closed-chain level and closure`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const grid = makeGPURasterMetadata({width: 3, height: 3});
    const result = await runOperationGraph(device, {
      inputs: {
        values: {format: 'float32', values: Float32Array.of(0, 0, 0, 0, 2, 0, 0, 0, 0)},
        levels: {format: 'float32', values: Float32Array.of(-2, -1, 1)},
        parameters: {
          format: 'float32',
          values: getGPUIsolinesParameterValues({levelCount: 3, extent: [0, 0, 3, 3]}, grid)
        }
      },
      operations: [
        new Operation({
          grid,
          segmentCapacity: 4,
          values: 'values',
          levels: 'levels',
          parameters: 'parameters',
          output: {
            segments: 'segments',
            segmentLevels: 'levelsOut',
            status: {count: 'count', overflow: 'overflow', polylineOverflow: 'polylineOverflow'},
            polylines: {
              vertexCapacity: 5,
              vertices: 'vertices',
              polylineOffsets: 'offsets',
              polylineLevels: 'chainLevel',
              polylineClosed: 'closed',
              polylineCount: 'chainCount',
              vertexCount: 'vertexCount'
            }
          }
        })
      ],
      outputs: [
        'count',
        'overflow',
        'polylineOverflow',
        'offsets',
        'chainLevel',
        'closed',
        'chainCount',
        'vertexCount'
      ]
    });
    // Four midpoint crossings around the high center form a closed diamond. Only level index
    // two (value 1) crosses the raster: [level,closed]=[2,1] rejects zeros and swapped ports.
    expectCensusValues(result['count'], [4]);
    expectCensusValues(result['overflow'], [0]);
    expectCensusValues(result['polylineOverflow'], [0]);
    expectCensusValues(result['chainCount'], [1]);
    expectCensusValues(result['vertexCount'], [5]);
    expectCensusValues(result['offsets'].slice(0, 2), [0, 5]);
    expectCensusValues(result['chainLevel'].slice(0, 1), [2]);
    expectCensusValues(result['closed'].slice(0, 1), [1]);
  });
}

// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {makeGPURasterMetadata} from '../../../src/raster-analysis/raster-grid/index';
import {expect, it} from 'vitest';
import {CPUIsolinesOperation} from '../../../src/raster-analysis/isolines/cpu-isolines-operation';
import {
  GPUIsolinesOperation,
  type IsolinesOperationProps
} from '../../../src/raster-analysis/isolines/gpu-isolines-operation';
import {getGPUIsolinesParameterValues} from '../../../src/raster-analysis/isolines/isolines-parameters';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

for (const capacity of [32, 1])
  it(`isolines operations agree with segment capacity ${capacity}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const grid = makeGPURasterMetadata({width: 4, height: 3});
    const props: IsolinesOperationProps = {
      grid,
      values: 'values',
      levels: 'levels',
      parameters: 'parameters',

      segmentCapacity: capacity,
      output: {
        segments: 'segments',
        segmentLevels: 'segmentLevels',
        segmentEdges: 'segmentEdges',
        status: {
          count: 'count',
          requiredCount: 'requiredCount',
          overflow: 'overflow',
          polylineOverflow: 'polylineOverflow'
        },
        polylines: {
          vertexCapacity: 64,
          vertices: 'vertices',
          polylineOffsets: 'polylineOffsets',
          polylineLevels: 'polylineLevels',
          polylineClosed: 'polylineClosed',
          polylineCount: 'polylineCount',
          vertexCount: 'vertexCount'
        }
      }
    };
    const results = [];
    const outputs = [
      'segments',
      'segmentLevels',
      'segmentEdges',
      'count',
      'requiredCount',
      'overflow',
      'polylineOverflow',
      'vertices',
      'polylineOffsets',
      'polylineLevels',
      'polylineClosed',
      'polylineCount',
      'vertexCount'
    ];
    for (const Operation of [GPUIsolinesOperation, CPUIsolinesOperation])
      results.push(
        await runOperationGraph(device, {
          inputs: {
            values: {
              format: 'float32',
              values: Float32Array.of(0, 3, 0, 0, 3, 0, 3, 0, 0, 0, 0, 0)
            },
            levels: {format: 'float32', values: Float32Array.of(1, 2)},
            parameters: {
              format: 'float32',
              values: getGPUIsolinesParameterValues({levelCount: 2, extent: [0, 0, 4, 3]}, grid)
            }
          },
          operations: [new Operation(props)],
          outputs
        })
      );
    for (const port of outputs) {
      const count = results[1]['count'][0];
      const polylineCount = results[1]['polylineCount'][0];
      const length =
        port === 'segments'
          ? count * 4
          : port === 'segmentEdges'
            ? count * 2
            : port === 'segmentLevels'
              ? count
              : port === 'vertices'
                ? results[1]['vertexCount'][0] * 2
                : port === 'polylineOffsets'
                  ? polylineCount + 1
                  : port === 'polylineLevels' || port === 'polylineClosed'
                    ? polylineCount
                    : 1;
      for (let index = 0; index < length; index++) {
        const actual = results[0][port][index];
        const expected = results[1][port][index];
        // Placement magnitude is at most four; absolute coordinate tolerance 8e-6.
        if (port === 'segments' || port === 'vertices')
          expect(Math.abs(actual - expected)).toBeLessThan(8e-6);
        else expect(actual === expected).toBe(true);
      }
    }
  });

it('isolines operations use every active input level despite a stale maximumLevelCount field', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const grid = makeGPURasterMetadata({width: 2, height: 2});
  const props: IsolinesOperationProps = {
    ...{maximumLevelCount: 1},
    grid,
    segmentCapacity: 2,
    values: 'values',
    levels: 'levels',
    parameters: 'parameters',
    output: {
      segments: 'segments',
      segmentLevels: 'segmentLevels',
      segmentEdges: 'segmentEdges',
      status: {count: 'count', requiredCount: 'requiredCount', overflow: 'overflow'}
    }
  };
  for (const Operation of [GPUIsolinesOperation, CPUIsolinesOperation]) {
    const result = await runOperationGraph(device, {
      inputs: {
        values: {format: 'float32', values: Float32Array.of(0, 2, 0, 2)},
        levels: {format: 'float32', values: Float32Array.of(0.5, 1.5)},
        parameters: {
          format: 'float32',
          values: getGPUIsolinesParameterValues({levelCount: 2, extent: [0, 0, 2, 2]}, grid)
        }
      },
      operations: [new Operation(props)],
      outputs: ['segments', 'segmentLevels', 'segmentEdges', 'count', 'requiredCount', 'overflow']
    });
    expect(Array.from(result['segments'])).toEqual([0.75, 1.5, 0.75, 0.5, 1.25, 1.5, 1.25, 0.5]);
    expect(Array.from(result['segmentLevels'])).toEqual([0, 1]);
    expect(Array.from(result['segmentEdges'])).toEqual([1, 0, 1, 0]);
    expect(Array.from(result['count'])).toEqual([2]);
    expect(Array.from(result['requiredCount'])).toEqual([2]);
    expect(Array.from(result['overflow'])).toEqual([0]);
  }
});

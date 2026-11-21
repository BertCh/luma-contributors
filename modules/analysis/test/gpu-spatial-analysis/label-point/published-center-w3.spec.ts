// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {GPULabelPointOperation} from '../../../src/gpu-spatial-analysis/label-point/gpu-label-point-operation';
import {CPULabelPointOperation} from '../../../src/gpu-spatial-analysis/label-point/cpu-label-point-operation';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';

it('UGE-05 publishes clearance at the rounded point on both label routes and the local-origin control', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const origin of [1048576, 0])
    for (const smallFeatureVertexLimit of [0, 32])
      for (const Operation of [GPULabelPointOperation, CPULabelPointOperation]) {
        const result = await runOperationGraph(device, {
          inputs: {
            positions: {
              format: 'float32x2',
              values: new Float32Array([
                origin,
                origin,
                origin + 0.125,
                origin,
                origin + 0.125,
                origin + 0.125,
                origin,
                origin + 0.125
              ])
            },
            ringOffsets: {format: 'uint32', values: new Uint32Array([0, 4])},
            polygonOffsets: {format: 'uint32', values: new Uint32Array([0, 1])},
            featureOffsets: {format: 'uint32', values: new Uint32Array([0, 1])}
          },
          operations: [
            new Operation({
              positions: 'positions',
              ringOffsets: 'ringOffsets',
              polygonOffsets: 'polygonOffsets',
              featureOffsets: 'featureOffsets',
              featureCount: 1,
              smallFeatureVertexLimit,
              output: {
                points: 'points',
                publishedCenterClearance: 'clearance',
                precisionLimited: 'precisionLimited',
                degenerate: 'degenerate'
              }
            })
          ],
          outputs: ['points', 'clearance', 'precisionLimited', 'degenerate']
        });
        expect(result['points']).toHaveLength(2);
        expect(result['clearance']).toHaveLength(1);
        expect(result['precisionLimited']).toHaveLength(1);
        expect(result['degenerate']).toHaveLength(1);
        // There is no float32 strictly between 2^20 and 2^20+.125. Locally the midpoint is exact.
        expect(Number.isFinite(result['clearance'][0])).toBe(true);
        expect(result['clearance'][0]).toBe(origin === 0 ? 0.0625 : 0);
        expect(result['precisionLimited'][0]).toBe(origin === 0 ? 0 : 1);
        expect(result['degenerate'][0]).toBe(origin === 0 ? 0 : 1);
        if (origin === 0) expect(Array.from(result['points'])).toEqual([0.0625, 0.0625]);
        else
          for (const coordinate of result['points'])
            expect([origin, origin + 0.125]).toContain(coordinate);
      }
});

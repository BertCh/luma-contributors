// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUPolygonOverlay} from '../../../src/gpu-spatial-analysis/polygon-overlay/gpu-polygon-overlay';
import {readUint32} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {WeightsRig} from '../spatial-weights/spatial-weights-harness';
import {packOperand, type OverlayOperand} from './overlay-gpu-harness';

for (const scenario of ['complete', 'boundary-limited', 'candidate-limited'] as const) {
  it(`overlay publishes independent whole boundary status words: ${scenario}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const rig = new WeightsRig(device);
    try {
      const rectangle = (horizontal: number, vertical: number): OverlayOperand[number] => [
        [
          [
            [horizontal, vertical],
            [horizontal + 4, vertical],
            [horizontal + 4, vertical + 4],
            [horizontal, vertical + 4]
          ]
        ]
      ];
      const upload = (operand: OverlayOperand) => {
        const packed = packOperand(operand);
        return {
          kind: 'polygons' as const,
          positions: rig.input(packed.positions, 'float32x2', packed.positions.length / 2),
          ringOffsets: rig.input(packed.ringOffsets, 'uint32', packed.ringOffsets.length),
          polygonOffsets: rig.input(packed.polygonOffsets, 'uint32', packed.polygonOffsets.length),
          featureOffsets: rig.input(packed.featureOffsets, 'uint32', packed.featureOffsets.length)
        };
      };
      const boundaryCount = rig.output('uint32', 1);
      const required = rig.output('uint32', 1);
      const overflow = rig.output('uint32', 1);
      const candidate = rig.output('uint32', 1);
      const candidateLimited = scenario === 'candidate-limited';
      rig.run(
        new GPUPolygonOverlay({
          left: upload(
            candidateLimited
              ? [rectangle(0, 0)]
              : [rectangle(0, 0), rectangle(8, 0), rectangle(16, 0)]
          ),
          right: candidateLimited ? upload([rectangle(2, -1)]) : undefined,
          operation: candidateLimited ? 'intersection' : 'dissolve',
          vertexTolerance: 1,
          capacity: {
            intersections: candidateLimited ? 1 : 48,
            nodedSegments: scenario === 'boundary-limited' ? 4 : 96
          },
          output: {
            geometry: {
              kind: 'polygons',
              positions: rig.output('float32x2', 96).view,
              featureOffsets: rig.output('uint32', 17).view,
              polygonOffsets: rig.output('uint32', 17).view,
              ringOffsets: rig.output('uint32', 17).view
            },
            status: {
              count: rig.output('uint32', 1).view,
              overflow: rig.output('uint32', 1).view,
              candidateOverflow: rig.output('uint32', 1).view
            },
            boundary: {
              endpoints: rig.output('float32x4', 96).view,
              operandIds: rig.output('uint32', 96).view,
              sourceFeatureIds: rig.output('uint32', 96).view,
              status: {
                count: boundaryCount.view,
                requiredCount: required.view,
                overflow: overflow.view,
                candidateOverflow: candidate.view
              }
            }
          }
        })
      );
      // All status buffers start poisoned. Three disjoint rectangles supply exactly 12 edges.
      // A noded-segment capacity of 4 cannot retain all 12. In the crossing case, the two
      // distinct proper intersections (2,0) and (4,3) alone exceed intersection capacity 1.
      expect(await readUint32(overflow.buffer, 1)).toEqual([
        scenario === 'boundary-limited' ? 1 : 0
      ]);
      expect(await readUint32(candidate.buffer, 1)).toEqual([candidateLimited ? 1 : 0]);
      if (scenario === 'complete') {
        expect(await readUint32(boundaryCount.buffer, 1)).toEqual([12]);
        expect(await readUint32(required.buffer, 1)).toEqual([12]);
      }
    } finally {
      rig.destroy();
    }
  });
}

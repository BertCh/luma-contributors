// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUBufferSurface} from '../../../src/gpu-spatial-analysis/polygon-buffer/gpu-buffer-surface';
import {readUint32} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {WeightsRig} from '../spatial-weights/spatial-weights-harness';

for (const vertexCapacity of [8, 9]) {
  it(`buffer surface pins eight retained boundary segments at vertex capacity ${vertexCapacity}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const rig = new WeightsRig(device);
    try {
      // WeightsRig fills every output with 0x7f7f7f7f, including the zero-valued overflow port.
      const count = rig.output('uint32', 1);
      const overflow = rig.output('uint32', 1);
      const boundaryRequiredCount = rig.output('uint32', 1);
      rig.run(
        new GPUBufferSurface({
          geometry: {
            kind: 'lines',
            positions: rig.input(Float32Array.of(0, 0, 2, 0), 'float32x2', 2),
            lineOffsets: rig.input(Uint32Array.of(0, 2), 'uint32', 2)
          },
          parameters: rig.input(Float32Array.of(1, 5, 0, 0), 'float32', 4),
          capStyle: 'square',
          joinStyle: 'bevel',
          quadSegments: 2,
          vertexTolerance: 1e-5,
          capacity: {intersections: 128, nodedSegments: 128},
          output: {
            geometry: {
              kind: 'polygons',
              positions: rig.output('float32x2', vertexCapacity).view,
              featureOffsets: rig.output('uint32', 33).view,
              polygonOffsets: rig.output('uint32', 33).view,
              ringOffsets: rig.output('uint32', 33).view
            },
            status: {
              count: count.view,
              overflow: overflow.view,
              candidateOverflow: rig.output('uint32', 1).view
            },
            boundary: {
              endpoints: rig.output('float32x4', 128).view,
              operandIds: rig.output('uint32', 128).view,
              sourceFeatureIds: rig.output('uint32', 128).view,
              status: {
                count: rig.output('uint32', 1).view,
                requiredCount: boundaryRequiredCount.view,
                overflow: rig.output('uint32', 1).view,
                candidateOverflow: rig.output('uint32', 1).view
              }
            }
          }
        })
      );
      // Union of the center stroke [0,2] x [-1,1] and square caps [-1,0] / [2,3] x [-1,1].
      // Retaining their junctions at x=0 and x=2 gives three top + three bottom + two end
      // segments: eight edges and nine closed-ring positions. Eight positions overflow;
      // nine fit. Shapely unary_union of those three rectangles independently retains the
      // same eight edges (unlike an oracle that removes every collinear vertex).
      expect(await readUint32(boundaryRequiredCount.buffer, 1)).toEqual([8]);
      // Ring assembly writes whole rings: the single feature is unavailable if its ring cannot fit.
      expect(await readUint32(count.buffer, 1)).toEqual([vertexCapacity === 8 ? 0 : 1]);
      expect(await readUint32(overflow.buffer, 1)).toEqual([vertexCapacity === 8 ? 1 : 0]);
    } finally {
      rig.destroy();
    }
  });
}

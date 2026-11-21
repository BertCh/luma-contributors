// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUSegmentRingAssembly} from '../../../src/analysis-core/ring-assembly';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {readUint32} from '../../utils/gpu-contributor-test-utils';
import {WeightsRig} from '../spatial-weights/spatial-weights-harness';

it('CPL-12 marks a closed four-edge prefix incomplete and preserves local assembly diagnostics', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const publishCandidate of [true, false]) {
    const rig = new WeightsRig(device);
    try {
      // Two disjoint unit squares require eight edges. The retained first four close perfectly.
      const endpoints = rig.input(
        Float32Array.of(
          0,
          0,
          1,
          0,
          1,
          0,
          1,
          1,
          1,
          1,
          0,
          1,
          0,
          1,
          0,
          0,
          3,
          0,
          4,
          0,
          4,
          0,
          4,
          1,
          4,
          1,
          3,
          1,
          3,
          1,
          3,
          0
        ),
        'float32x4',
        8
      );
      const sourceLoss = rig.input(Uint32Array.of(1), 'uint32', 1);
      const count = rig.output('uint32', 2),
        overflow = rig.output('uint32', 1),
        open = rig.output('uint32', 1),
        candidate = rig.output('uint32', 1),
        polygonCount = rig.output('uint32', 1);
      count.buffer.write(Uint32Array.of(99, 55));
      const segmentRings = rig.output('uint32', 8),
        segmentVertices = rig.output('uint32', 8),
        sourceIds = rig.output('uint32', 4);
      const producer = new GPUSegmentRingAssembly({
        endpoints,
        groups: rig.input(new Uint32Array(8).fill(42), 'uint32', 8),
        segmentStatus: {count: rig.input(Uint32Array.of(4), 'uint32', 1), overflow: sourceLoss},
        output: {
          segmentRings: segmentRings.view,
          segmentVertices: segmentVertices.view,
          positions: rig.output('float32x2', 20).view,
          ringOffsets: rig.output('uint32', 5).view,
          polygons: {
            kind: 'polygons',
            sourceIds: sourceIds.view,
            positions: rig.output('float32x2', 20).view,
            ringOffsets: rig.output('uint32', 5).view,
            polygonOffsets: rig.output('uint32', 5).view,
            featureOffsets: rig.output('uint32', 5).view
          },
          status: {
            count: count.view,
            overflow: overflow.view,
            openSegmentCount: open.view,
            polygonCount: polygonCount.view,
            ...(publishCandidate ? {candidateOverflow: candidate.view} : {})
          }
        }
      });
      rig.run(producer);
      expect(await readUint32(count.buffer, 2)).toEqual([publishCandidate ? 1 : 0, 55]);
      if (!publishCandidate) {
        expect(await readUint32(segmentRings.buffer, 8)).toEqual(new Array(8).fill(0xffffffff));
        expect(await readUint32(segmentVertices.buffer, 8)).toEqual(new Array(8).fill(0xffffffff));
        expect(await readUint32(sourceIds.buffer, 4)).toEqual(new Array(4).fill(0xffffffff));
      }
      expect(await readUint32(polygonCount.buffer, 1)).toEqual([publishCandidate ? 1 : 0]);
      expect(await readUint32(overflow.buffer, 1)).toEqual([0]);
      expect(await readUint32(open.buffer, 1)).toEqual([0]);
      if (publishCandidate) expect(await readUint32(candidate.buffer, 1)).toEqual([1]);
    } finally {
      rig.destroy();
    }
  }
});

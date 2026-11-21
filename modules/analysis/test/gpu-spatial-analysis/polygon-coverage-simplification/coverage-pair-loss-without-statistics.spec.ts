// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUCoverageSimplification} from '../../../src/gpu-spatial-analysis/polygon-coverage-simplification/gpu-coverage-simplification';
import {getGPULineSimplificationParameterValues} from '../../../src/gpu-spatial-analysis/line-simplification/index';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {readFloat32, readUint32} from '../../utils/gpu-contributor-test-utils';
import {WeightsRig} from '../spatial-weights/spatial-weights-harness';

it('coverage pair loss restores original adjacent polygons without optional topology statistics', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // Two adjacent rectangles, with one exactly collinear, removable bottom-edge vertex each.
  // The shared vertical edge and its endpoint contacts produce more than one topology pair.
  const originalPositions = [0, 0, 1, 0, 2, 0, 2, 2, 0, 2, 2, 0, 3, 0, 4, 0, 4, 2, 2, 2];
  for (const topologyPairCapacity of [256, 1]) {
    const rig = new WeightsRig(device);
    try {
      const positions = rig.output('float32x2', 10);
      const offsets = rig.output('uint32', 3);
      const overflow = rig.output('uint32', 1);
      const required = rig.output('uint32', 1);
      rig.run(
        new GPUCoverageSimplification({
          positions: rig.input(Float32Array.from(originalPositions), 'float32x2', 10),
          ringOffsets: rig.input(Uint32Array.of(0, 5, 10), 'uint32', 3),
          polygonOffsets: rig.input(Uint32Array.of(0, 1, 2), 'uint32', 3),
          featureOffsets: rig.input(Uint32Array.of(0, 1, 2), 'uint32', 3),
          parameters: rig.input(
            getGPULineSimplificationParameterValues({tolerance: 0.1}),
            'float32',
            4
          ),
          maximumIterations: 32,
          simplifyBoundary: true,
          topologyIterationCount: 1,
          topologyPairCapacity,
          output: {
            positions: positions.view,
            ringOffsets: offsets.view,
            status: {overflow: overflow.view, requiredCount: required.view}
          }
        })
      );
      const expectedCount = topologyPairCapacity === 1 ? 10 : 8;
      const actualPositions = await readFloat32(positions.buffer, expectedCount * 2);
      expect(actualPositions.length).toBe(expectedCount * 2);
      actualPositions.forEach(value => expect(Number.isFinite(value)).toBe(true));
      expect(await readUint32(overflow.buffer, 1)).toEqual([0]);
      expect(await readUint32(required.buffer, 1)).toEqual([expectedCount]);
      expect(await readUint32(offsets.buffer, 3)).toEqual(
        topologyPairCapacity === 1 ? [0, 5, 10] : [0, 4, 8]
      );
      // Independent geometry: deleting (1,0) and (3,0) leaves the same two rectangles.
      expect(actualPositions).toEqual(
        topologyPairCapacity === 1
          ? originalPositions
          : [0, 0, 2, 0, 2, 2, 0, 2, 2, 0, 4, 0, 4, 2, 2, 2]
      );
    } finally {
      rig.destroy();
    }
  }
});

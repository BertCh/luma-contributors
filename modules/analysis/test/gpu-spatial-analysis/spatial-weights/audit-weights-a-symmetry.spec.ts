// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {getChunkedSpatialWeightsSymmetryNodes} from '../../../src/gpu-spatial-analysis/spatial-weights/chunked-spatial-weights-symmetry';
import {requireTestDevice} from '../../utils/require-test-device';
import {readUint32} from '../../utils/gpu-contributor-test-utils';
import {WeightsRig} from './spatial-weights-harness';

it('audits chunked symmetry against literal support-union counts with ties, a self slot and an island', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // audit-weights-a-symmetry.oracle.py enumerates the union of ordered support positions.
  // Values and tolerances are binary-exact f32; the inclusive threshold needs no tolerance.
  const offsets = [0, 1, 2, 3, 5, 6, 7, 8, 8];
  const neighbors = [1, 0, 0, 4, 6, 3, 5, 3];
  const values = [2, 2.5, 0, 3, 1, 1, 5, 1];
  for (const chunkRows of [[8], [1, 2, 3, 2]]) {
    for (const [tolerance, expected] of [
      [0, 6],
      [0.5, 4]
    ]) {
      const rig = new WeightsRig(device);
      let rowStart = 0;
      const chunks = chunkRows.map(rowCount => {
        const begin = offsets[rowStart];
        const end = offsets[rowStart + rowCount];
        const chunk = {
          rowStart,
          rowCount,
          weights: rig.uploadWeights({
            offsets: offsets.slice(rowStart, rowStart + rowCount + 1).map(value => value - begin),
            neighbors: neighbors.slice(begin, end),
            weights: values.slice(begin, end),
            distances: []
          })
        };
        rowStart += rowCount;
        return chunk;
      });
      const asymmetricSlots = rig.output('uint32', 1);
      const toleranceView = rig.input(Float32Array.of(tolerance), 'float32', 1);
      try {
        rig.run({
          getCommandNodes: graph =>
            getChunkedSpatialWeightsSymmetryNodes(
              graph,
              'audit-symmetry',
              'AuditSymmetry',
              {rowCount: 8, chunks},
              asymmetricSlots.view,
              toleranceView
            )
        });
        expect(await readUint32(asymmetricSlots.buffer, 1)).toEqual([expected]);
      } finally {
        rig.destroy();
      }
    }
  }
});

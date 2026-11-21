// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUSpatialWeightsTranspose} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-spatial-weights-transpose';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {readFloat32, readUint32} from '../../utils/gpu-contributor-test-utils';
import {WeightsRig} from './spatial-weights-harness';
import {GPUChunkedSpatialWeightsTranspose} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-chunked-spatial-weights-transpose';

it.each([
  {
    columns: 1,
    neighbors: [1, 0],
    symmetric: true,
    offsets: [0, 1],
    retained: [1],
    mass: [0.5],
    transform: 'unknown'
  },
  {
    columns: 2,
    neighbors: [2, 0],
    symmetric: 'unknown',
    offsets: [0, 1, 1],
    retained: [1],
    mass: [0.5],
    transform: 'unknown'
  },
  {
    columns: 2,
    neighbors: [1, 0],
    symmetric: 'unknown',
    offsets: [0, 1, 2],
    retained: [1, 0],
    mass: [0.5, 0.5],
    transform: 'unknown'
  },
  {
    columns: 2,
    neighbors: [1, 0],
    symmetric: true,
    offsets: [0, 1, 2],
    retained: [1, 0],
    mass: [0.5, 0.5],
    transform: 'double'
  }
] as const)('W5-06: transpose normalization with $columns columns, neighbors $neighbors, symmetric $symmetric', async fixture => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const chunked of [false, true]) {
    const rig = new WeightsRig(device);
    try {
      // Each original edge has mass 1/2. Only neighbors within the target domain survive.
      // The third fixture keeps all mass, but its metadata is conservative because the public
      // handle declares no source column bound. The fourth is the proven full-domain control:
      // declared symmetric and square, so W' = W and double normalization is carried over.
      const weights = rig.uploadWeights({
        offsets: [0, 1, 2],
        neighbors: [...fixture.neighbors],
        weights: [0.5, 0.5],
        distances: []
      });
      const information = {
        rowCount: 2,
        transform: 'double' as const,
        symmetric: fixture.symmetric
      };
      const output = rig.weightsOutput(fixture.columns, 2);
      const transpose = chunked
        ? new GPUChunkedSpatialWeightsTranspose({
            weights: {rowCount: 2, chunks: [{rowStart: 0, rowCount: 2, weights}], information},
            columnCount: fixture.columns,
            output: output.spatialWeights
          })
        : new GPUSpatialWeightsTranspose({
            weights: {...weights, information},
            columnCount: fixture.columns,
            output: output.spatialWeights
          });
      rig.run(transpose);
      expect(await output.read()).toEqual({
        offsets: [...fixture.offsets],
        neighbors: [...fixture.retained],
        weights: [...fixture.mass],
        distances: []
      });
      expect(transpose.outputWeights.information?.transform).toBe(fixture.transform);
    } finally {
      rig.destroy();
    }
  }
});
import {
  TRANSPOSE_AUDIT_INPUT as INPUT,
  TRANSPOSE_AUDIT_EXPECTED as EXPECTED
} from './weights-transpose-audit-fixture';

it('rectangular transpose drops input cardinalities and preserves input loss for packed and chunked sources', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const chunked of [false, true]) {
    const rig = new WeightsRig(device);
    try {
      // W=[[0,3],[2,0]]; retain column 0 only, so W.T=[[2]] at source row 1.
      const weights = rig.uploadWeights({
        offsets: [0, 1, 2],
        neighbors: [1, 0],
        weights: [3, 2],
        distances: []
      });
      const status = {
        count: rig.input(Uint32Array.of(2), 'uint32', 1),
        requiredCount: rig.input(Uint32Array.of(2), 'uint32', 1),
        overflow: rig.input(Uint32Array.of(7), 'uint32', 1),
        candidateOverflow: rig.input(Uint32Array.of(0), 'uint32', 1)
      };
      const output = rig.weightsOutput(1, 2);
      const transpose = chunked
        ? new GPUChunkedSpatialWeightsTranspose({
            weights: {rowCount: 2, chunks: [{rowStart: 0, rowCount: 2, weights}], status},
            columnCount: 1,
            output: output.spatialWeights
          })
        : new GPUSpatialWeightsTranspose({
            weights: {...weights, status},
            columnCount: 1,
            output: output.spatialWeights
          });
      rig.run(transpose);
      expect(await output.read()).toEqual({
        offsets: [0, 1],
        neighbors: [1],
        weights: [2],
        distances: []
      });
      expect(Boolean(transpose.outputWeights.status?.count)).toBe(false);
      expect(Boolean(transpose.outputWeights.status?.requiredCount)).toBe(false);
      expect(transpose.outputWeights.status?.overflow).toBe(status.overflow);
      expect(transpose.outputWeights.status?.candidateOverflow).toBe(status.candidateOverflow);
      expect(await readUint32(rig.bufferOf(status.overflow), 1)).toEqual([7]);
      expect(await readUint32(rig.bufferOf(status.candidateOverflow), 1)).toEqual([0]);
    } finally {
      rig.destroy();
    }
  }
});

it('transpose matches every SciPy sparse output, preserves stored zero slots and clears tails', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // (cW).T = c(W.T); absolute symmetry tolerance scales by the same c.
  for (const [columns, scale] of [
    [8, 1],
    [8, 2],
    [5, 1],
    [5, 2]
  ]) {
    for (const tolerance of columns === 8 ? [0, 1] : [0]) {
      const rig = new WeightsRig(device);
      try {
        const weights = rig.uploadWeights(
          {...INPUT, weights: INPUT.weights.map(value => value * scale)},
          2
        );
        const offsets = rig.output('uint32', columns + 1);
        const neighbors = rig.output('uint32', 15);
        const values = rig.output('float32', 15);
        const distances = rig.output('float32', 15);
        const asymmetricSlots = columns === 8 ? rig.output('uint32', 1) : undefined;
        rig.run(
          new GPUSpatialWeightsTranspose({
            weights,
            columnCount: columns,
            symmetryTolerance: tolerance * scale,
            output: {
              offsets: offsets.view,
              neighbors: neighbors.view,
              weights: values.view,
              distances: distances.view,
              asymmetricSlots: asymmetricSlots?.view
            }
          })
        );
        const used = columns === 8 ? 12 : 10;
        expect(await readUint32(offsets.buffer, columns + 1)).toEqual(
          EXPECTED.offsets.slice(0, columns + 1)
        );
        expect(await readUint32(neighbors.buffer, 15)).toEqual([
          ...EXPECTED.neighbors.slice(0, used),
          ...Array(15 - used).fill(0)
        ]);
        expect(await readFloat32(values.buffer, 15)).toEqual([
          ...EXPECTED.weights.slice(0, used).map(value => value * scale),
          ...Array(15 - used).fill(0)
        ]);
        expect(await readFloat32(distances.buffer, 15)).toEqual([
          ...EXPECTED.distances.slice(0, used),
          ...Array(15 - used).fill(0)
        ]);
        if (asymmetricSlots)
          expect(await readUint32(asymmetricSlots.buffer, 1)).toEqual([tolerance === 0 ? 6 : 4]);
      } finally {
        rig.destroy();
      }
    }
  }
});

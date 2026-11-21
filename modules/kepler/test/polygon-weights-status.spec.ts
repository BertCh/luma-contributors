// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {requireTestDevice} from '../../analysis/src/testing/require-test-device';
import {CatalogGpuSession} from '../src/analysis/catalog-gpu-session';
import {addKeplerPolygonWeights} from '../src/analysis/polygon-weights';

test.each([
  [8, 32],
  [1, 32],
  [8, 1]
])('polygon weights keep loss and invalid-row status separate at capacity %s and pair capacity %s', async (neighborCapacity, pairCapacity) => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const session = new CatalogGpuSession(device, `polygon-status-${neighborCapacity}`);
  try {
    const overflow = session.output('overflow', 'uint32', 1);
    const requiredCount = session.output('required', 'uint32', 1);
    const candidateOverflow = session.output('candidate-overflow', 'uint32', 1);
    const invalidCount = session.output('invalid', 'uint32', 1);
    const count = session.output('count', 'uint32', 1);
    addKeplerPolygonWeights(session.graph, {
      source: {datasetId: 'two-touching-triangles-and-empty-row', revision: 1},
      settings: {
        kind: 'contiguity',
        geometryColumn: 'geometry',
        criterion: 'queen',
        equality: 'exact-source-xy',
        neighborCapacity,
        pairCapacity
      },
      polygons: {
        kind: 'polygons',
        // Private equality tokens: the triangles share vertex ID 3; row 2 has no polygon.
        positions: session.input(
          'tokens',
          Float32Array.of(1, 0, 2, 0, 3, 0, 3, 0, 4, 0, 5, 0),
          'float32x2'
        ),
        featureOffsets: session.input('features', Uint32Array.of(0, 1, 2, 2), 'uint32'),
        polygonOffsets: session.input('polygons', Uint32Array.of(0, 1, 2), 'uint32'),
        ringOffsets: session.input('rings', Uint32Array.of(0, 3, 6), 'uint32')
      },
      geometryMask: session.input('geometry-mask', Uint32Array.of(1, 1, 0), 'uint32'),
      status: {
        count: count.view,
        overflow: overflow.view,
        requiredCount: requiredCount.view,
        candidateOverflow: candidateOverflow.view,
        invalidCount: invalidCount.view
      }
    });
    session.run();
    // One undirected queen adjacency means two directed CSR entries. One empty row is invalid.
    if (pairCapacity !== 1) {
      expect(await session.readScalar(requiredCount)).toBe(2);
      expect(await session.readScalar(count)).toBe(neighborCapacity === 1 ? 1 : 2);
    }
    expect(await session.readScalar(invalidCount)).toBe(1);
    expect(await session.readScalar(candidateOverflow)).toBe(pairCapacity === 1 ? 1 : 0);
    expect(await session.readScalar(overflow)).toBe(neighborCapacity === 1 ? 1 : 0);
  } finally {
    await session.dispose();
  }
});

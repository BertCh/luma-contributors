// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {Buffer} from '@luma.gl/core';
import {importGraphBuffer} from '../../src/analysis-core/kernels/gpu-contributor-utils';
import type {GPUPointGeometryPort} from '../../src/analysis-core/contracts/index';
import {GPUPointInPolygonJoin} from '../../src/gpu-spatial-analysis/spatial-join/gpu-point-in-polygon-join';
import {GPUBufferSelection} from '../../src/gpu-spatial-analysis/spatial-join/gpu-buffer-selection';
import {createNullWebGPUDevice} from '../utils/gpu-contributor-test-utils';

for (const kind of ['point-in-polygon', 'buffer-selection'] as const) {
  it(`final1 ${kind} refuses multipart point featureOffsets before flattening`, () => {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    const positions = createTransientView(graph, 'queries', 'float32x2', 2);
    const offsetBuffer = device.createBuffer({data: Uint32Array.of(0, 2), usage: Buffer.STORAGE});
    const offsets = importGraphBuffer(graph, 'point-features', offsetBuffer, 'uint32', 2);
    const polygons = {
      kind: 'polygons' as const,
      positions: createTransientView(graph, 'polygon-positions', 'float32x2', 4),
      featureOffsets: createTransientView(graph, 'features', 'uint32', 2),
      polygonOffsets: createTransientView(graph, 'polygons', 'uint32', 2),
      ringOffsets: createTransientView(graph, 'rings', 'uint32', 2)
    };
    const create = (multipart: boolean) => {
      // Logical offsets [0,2] describe one feature with two point parts. The guard uses
      // port presence, before GPU contents can be inspected or the query reconstructed.
      // Bypass the static single-part exclusion only to exercise the runtime refusal.
      const points = {
        kind: 'points' as const,
        positions,
        ...(multipart ? {featureOffsets: offsets} : {})
      } as GPUPointGeometryPort;
      return kind === 'point-in-polygon'
        ? new GPUPointInPolygonJoin({
            points: {...points},
            polygons,
            candidateCapacity: 4,
            output: {
              pointFeatureIds: createTransientView(graph, `assignments-${multipart}`, 'uint32', 2),
              status: {overflow: createTransientView(graph, `overflow-${multipart}`, 'uint32', 1)}
            }
          })
        : new GPUBufferSelection({
            points: {...points},
            features: {kind: 'points', positions: polygons.positions},
            candidateCapacity: 4,
            distance: createTransientView(graph, `distance-${multipart}`, 'float32', 1),
            output: {outputMask: createTransientView(graph, `mask-${multipart}`, 'uint32', 2)}
          });
    };
    try {
      expect(() => create(true)).toThrow(/featureOffsets multipart points are unsupported/);
      expect(create(false).getCommandNodes(graph).length).toBeGreaterThan(0);
    } finally {
      offsetBuffer.destroy();
      device.destroy();
    }
  });
}

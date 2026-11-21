// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {Buffer} from '@luma.gl/core';
import {importGraphBuffer} from '../../src/analysis-core/kernels/gpu-contributor-utils';
import {GPUPointInPolygonJoin} from '../../src/gpu-spatial-analysis/spatial-join/gpu-point-in-polygon-join';
import {GPUBufferSelection} from '../../src/gpu-spatial-analysis/spatial-join/gpu-buffer-selection';
import {createNullWebGPUDevice} from '../utils/gpu-contributor-test-utils';

for (const kind of ['point-in-polygon', 'buffer-selection'] as const) {
  it(`final1 ${kind} preserves multipart featureOffsets and validates feature-aligned outputs`, () => {
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
    const create = (multipart: boolean, featureAligned = true) => {
      // Logical offsets [0,2] describe one feature with two point parts.
      const outputCount = multipart && featureAligned ? 1 : 2;
      const suffix = `${multipart}-${featureAligned}`;
      const points = {
        kind: 'points' as const,
        positions,
        ...(multipart ? {featureOffsets: offsets} : {})
      };
      return kind === 'point-in-polygon'
        ? new GPUPointInPolygonJoin({
            points: {...points},
            polygons,
            candidateCapacity: 4,
            output: {
              pointFeatureIds: createTransientView(
                graph,
                `assignments-${suffix}`,
                'uint32',
                outputCount
              ),
              status: {overflow: createTransientView(graph, `overflow-${suffix}`, 'uint32', 1)}
            }
          })
        : new GPUBufferSelection({
            points: {...points},
            features: {kind: 'points', positions: polygons.positions},
            candidateCapacity: 4,
            radius: createTransientView(graph, `distance-${suffix}`, 'float32', 1),
            output: {
              outputMask: createTransientView(graph, `mask-${suffix}`, 'uint32', outputCount)
            }
          });
    };
    try {
      expect(() => create(true, false)).toThrow(/one packed row per multipart feature/);
      const grouped = create(true);
      expect(grouped.props.points.featureOffsets).toBe(offsets);
      expect(grouped.getCommandNodes(graph).length).toBeGreaterThan(0);
      expect(create(false).props.points.featureOffsets).toBeUndefined();
    } finally {
      offsetBuffer.destroy();
      device.destroy();
    }
  });
}

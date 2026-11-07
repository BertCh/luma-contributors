// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUBoundsFilter selects features whose bounding boxes meet a per-frame box. Feature bounds come
// from GPUGeometryMeasures in the same graph; polygons are filled by the selection mask.

import {
  getGPUBoundsFilterParameterValues,
  GPUBoundsFilter,
  GPUGeometryMeasures
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {makeRandom} from '../../../demos/kit';
import {defineOperationRun} from '../../define-run';
import {createPolygonSet, uploadPolygonSet} from './shared/polygon-set';
import {addBoxSegments} from './shared/shape-segments';
import {parameters} from './bounds-filter.parameters';

const MODES = ['intersects', 'within', 'contains'] as const;

export default defineOperationRun({
  parameters,
  caption: 'Polygons whose bounding box meets the box are selected.',
  createInputs: ({seed, world}) => ({
    set: createPolygonSet(makeRandom(seed), world, {rows: 4, columns: 7, vertexCount: 10})
  }),
  build(builder, inputs, values) {
    const polygons = uploadPolygonSet(builder, inputs.set);
    const count = inputs.set.count;
    const bounds = builder.allocate('bounds', 'float32x4', count);
    builder.graph.add(
      new GPUGeometryMeasures({
        geometryType: 'polygons',
        positions: polygons.positions.view,
        featureOffsets: polygons.featureOffsets.view,
        polygonOffsets: polygons.polygonOffsets.view,
        ringOffsets: polygons.ringOffsets.view,
        output: {bounds: bounds.view}
      })
    );
    const box = builder.allocate('box', 'float32', 4);
    const boxRow = builder.allocate('box-row', 'float32x4', 1);
    const mask = builder.allocate('mask', 'uint32', count);
    builder.graph.add(
      new GPUBoundsFilter({
        bounds: bounds.view,
        parameters: box.view,
        mode: MODES[values.mode],
        output: {
          ids: builder.allocate('ids', 'uint32', count).view,
          outputMask: mask.view,
          status: {
            count: builder.allocate('count', 'uint32', 1).view,
            overflow: builder.allocate('overflow', 'uint32', 1).view
          }
        }
      })
    );
    return {
      outputs: {...polygons, mask, boxSegments: addBoxSegments(builder, 'box', boxRow)},
      update: current => {
        const half = current.size * 0.5;
        const settings = {
          minX: current.centerX - half * 1.6,
          minY: 0.5 - half,
          maxX: current.centerX + half * 1.6,
          maxY: 0.5 + half
        };
        box.write(getGPUBoundsFilterParameterValues(settings));
        boxRow.write(Float32Array.of(settings.minX, settings.minY, settings.maxX, settings.maxY));
      }
    };
  },
  rebuildOn: ['mode'],
  draw: ({inputs, outputs}) => [
    {
      layer: 'polygons',
      positions: outputs.fillVertices,
      indices: outputs.fillIndices,
      owners: outputs.fillOwners,
      values: outputs.mask,
      triangleCount: inputs.set.fillTriangleCount,
      ramp: 'sequential',
      range: [0, 1]
    },
    {layer: 'lines', context: true, segments: outputs.outline, count: inputs.set.outlineCount},
    {layer: 'lines', segments: outputs.boxSegments, color: [1, 0.6, 0.2, 1], radius: 2}
  ]
});

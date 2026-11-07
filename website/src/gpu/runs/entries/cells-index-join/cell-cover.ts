// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUCellCover fills two polygons with cells (polyfill). Cells that are provably inside the polygon
// (core) are drawn in one color, border cells in another; the outline is the input polygon.

import {GPUCellCover} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {addCellFill, getCellFamily} from './support/cells';
import {createStarRing, packPolygonRings} from './support/polygons';
import {parameters} from './cell-cover.parameters';

const OUTPUT_CAPACITY = 2048;

export default defineOperationRun({
  parameters,
  caption: 'Cells covering two polygons; core cells (inside) and border cells differ in color.',
  createInputs: ({random, world}) =>
    packPolygonRings([
      createStarRing(random, [world.width * 0.3, 0.5], 0.3, 11, 0.45),
      createStarRing(random, [world.width * 0.7, 0.5], 0.32, 13, 0.35)
    ]),
  build(builder, polygons, values) {
    const family = getCellFamily(values.family);
    const resolution = values.resolution + (family === 'h3' ? 0 : 6);
    const positions = builder.upload('positions', 'float32x2', polygons.positions);
    const outline = builder.upload('outline', 'float32x4', polygons.outline);
    const cells = builder.allocate('cells', 'uint32x2', OUTPUT_CAPACITY);
    const featureIds = builder.allocate('feature-ids', 'uint32', OUTPUT_CAPACITY);
    const core = builder.allocate('core', 'uint32', OUTPUT_CAPACITY);
    const status = {
      count: builder.allocate('count', 'uint32', 1),
      overflow: builder.allocate('overflow', 'uint32', 1),
      candidateOverflow: builder.allocate('candidate-overflow', 'uint32', 1)
    };
    builder.graph.add(
      new GPUCellCover({
        family,
        resolution,
        candidateCapacity: 1 << 16,
        positions: positions.view,
        featureOffsets: builder.upload('feature-offsets', 'uint32', polygons.featureOffsets).view,
        polygonOffsets: builder.upload('polygon-offsets', 'uint32', polygons.polygonOffsets).view,
        ringOffsets: builder.upload('ring-offsets', 'uint32', polygons.ringOffsets).view,
        output: {
          cells: cells.view,
          featureIds: featureIds.view,
          core: core.view,
          status: {
            count: status.count.view,
            overflow: status.overflow.view,
            candidateOverflow: status.candidateOverflow.view
          }
        }
      })
    );
    return {
      update: current => {
        const scale = (array: Float32Array, stride: number, ownerOf: (index: number) => number) =>
          Float32Array.from(array, (value, index) => {
            const center = polygons.centers[ownerOf(Math.floor(index / stride)) * 2 + (index % 2)];
            return center + (value - center) * current.size;
          });
        const ringOf = (vertex: number) => (vertex < polygons.ringOffsets[1] ? 0 : 1);
        positions.write(scale(polygons.positions, 2, ringOf));
        // Outline segments carry two vertices each; both belong to the same ring.
        outline.write(scale(polygons.outline, 4, ringOf));
      },
      outputs: {
        core,
        outline,
        fill: addCellFill(builder, family, cells)
      }
    };
  },
  rebuildOn: ['family', 'resolution'],
  draw: ({inputs, outputs}) => [
    {
      layer: 'polygons',
      positions: outputs.fill.boundaries,
      indices: outputs.fill.indices,
      owners: outputs.fill.owners,
      values: outputs.core,
      triangleCount: outputs.fill.triangleCount,
      alpha: 0.8
    },
    {
      layer: 'lines',
      context: true,
      segments: outputs.outline,
      count: inputs.outlineCount,
      radius: 1.8
    }
  ]
});

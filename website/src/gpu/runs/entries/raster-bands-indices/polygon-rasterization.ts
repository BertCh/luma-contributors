// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUPolygonRasterization: blob polygons (every other one has a hole) are burned into a grid of
// zone ids; the boundary cells are flagged. The grid width is a compile-time prop (the run
// rebuilds); the hole size rewrites the polygon vertices.

import {
  getGPUPolygonRasterizationParameterValues,
  GPUPolygonRasterization
} from '@luma-contributors/analysis/raster-analysis';
import {defineOperationRun} from '../../define-run';
import {parameters} from './polygon-rasterization.parameters';
import {makeWorldGrid} from './support/bands';
import {createBlobFeatures} from './support/blobs';

export default defineOperationRun({
  parameters,
  caption: 'Polygons in; a raster of zone ids out. Cells without a polygon stay empty.',
  createInputs: ({random, world}) => ({
    world,
    features: createBlobFeatures(random, world, {holes: true})
  }),
  build(builder, {world, features}, values) {
    const {grid, columns, rows, cellCount} = makeWorldGrid(world, values.gridWidth);
    const positions = builder.upload('positions', 'float32x2', features.positions);
    const featureOffsets = builder.upload('feature-offsets', 'uint32', features.featureOffsets);
    const polygonOffsets = builder.upload('polygon-offsets', 'uint32', features.polygonOffsets);
    const ringOffsets = builder.upload('ring-offsets', 'uint32', features.ringOffsets);
    const extent = builder.upload(
      'extent',
      'float32',
      getGPUPolygonRasterizationParameterValues(grid)
    );
    const zones = builder.allocate('zones', 'uint32', cellCount);
    const boundary = builder.allocate('boundary', 'uint32', cellCount);
    const overflow = builder.allocate('overflow', 'uint32', 1);
    const outline = builder.upload('outline', 'float32x4', features.outline);
    builder.graph.add(
      new GPUPolygonRasterization({
        grid,
        parameters: extent.view,
        positions: positions.view,
        featureOffsets: featureOffsets.view,
        polygonOffsets: polygonOffsets.view,
        ringOffsets: ringOffsets.view,
        crossingCapacity: 4 * cellCount,
        output: {zones: zones.view, boundary: boundary.view, status: {overflow: overflow.view}}
      })
    );
    return {
      outputs: {zones, boundary, outline, columns, rows},
      update: current => {
        const moved = features.getPositions(current.holeScale);
        positions.write(moved);
        outline.write(features.getOutline(moved));
      }
    };
  },
  rebuildOn: ['gridWidth'],
  draw: ({inputs, outputs, theme}) => [
    {layer: 'raster', values: outputs.zones, columns: outputs.columns, rows: outputs.rows},
    {layer: 'lines', context: true, segments: outputs.outline, color: [...theme.ink, 0.9]}
  ]
});

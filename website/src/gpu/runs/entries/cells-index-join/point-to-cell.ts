// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUPointToCell: every point gets the 64-bit key of the cell it falls in. The keys are decoded
// with GPUCellGeometry only to draw them: each outline is the cell of at least one point.

import {GPUPointToCell} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {createClusteredPoints} from '../../inputs';
import {addCellGeometry, createCellWireframe} from './support/cells';
import {parameters} from './point-to-cell.parameters';

export default defineOperationRun({
  parameters,
  caption: 'Each point falls into exactly one cell; the outlines are the cells that were hit.',
  createInputs: ({random, world}) => createClusteredPoints(random, world, 260, 4, 0.15),
  build(builder, points, values) {
    const family = values.family === 0 ? 'h3' : 'quadbin';
    const resolution = family === 'h3' ? values.resolution : values.resolution + 5;
    const positions = builder.upload('positions', 'float32x2', points.positions);
    const cells = builder.allocate('cells', 'uint32x2', points.count);
    builder.graph.add(
      new GPUPointToCell({
        family,
        resolution,
        positions: positions.view,
        output: {cells: cells.view}
      })
    );
    const geometry = addCellGeometry(builder, family, cells);
    const wireframe = createCellWireframe(points.count, geometry.stride, geometry.vertexCount);
    return {
      outputs: {
        positions,
        boundaries: geometry.boundaries,
        wireframe: builder.upload('wireframe', 'uint32', wireframe),
        wireframeCount: wireframe.length / 3
      }
    };
  },
  draw: ({outputs}) => [
    {
      layer: 'triangles',
      positions: outputs.boundaries,
      triangles: outputs.wireframe,
      count: outputs.wireframeCount,
      radius: 1.4
    },
    {layer: 'dots', context: true, positions: outputs.positions, radius: 2.2}
  ]
});

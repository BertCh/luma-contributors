// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUCellGeometry decodes cell keys to centers, boundary polygons and vertex counts. The keys come
// from GPUPointToCell on a few clusters of points; the fill is colored by each cell's vertex count.

import {GPUCellGeometry, GPUPointToCell} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {createClusteredPoints} from '../../inputs';
import {
  createCellFanMesh,
  createCellWireframe,
  getCellFamily,
  getCellResolution,
  H3_BOUNDARY_STRIDE
} from './support/cells';
import {parameters} from './cell-geometry.parameters';

export default defineOperationRun({
  parameters,
  caption: 'Cell keys decoded to boundary polygons (fill) and centers (dots).',
  createInputs: ({random, world}) => createClusteredPoints(random, world, 200, 4, 0.2),
  build(builder, points, values) {
    const family = getCellFamily(values.family);
    const positions = builder.upload('positions', 'float32x2', points.positions);
    const cells = builder.allocate('cells', 'uint32x2', points.count);
    builder.graph.add(
      new GPUPointToCell({
        family,
        resolution: getCellResolution(family, values.resolution),
        positions: positions.view,
        output: {cells: cells.view}
      })
    );
    const stride = family === 'h3' ? H3_BOUNDARY_STRIDE : 4;
    const vertexCount = family === 'h3' ? 6 : 4;
    const centers = builder.allocate('centers', 'float32x2', points.count);
    const boundaries = builder.allocate('boundaries', 'float32x2', points.count * stride);
    const vertexCounts = builder.allocate('vertex-counts', 'uint32', points.count);
    builder.graph.add(
      new GPUCellGeometry({
        family,
        cells: cells.view,
        maximumVertexCount: stride,
        output: {
          centers: centers.view,
          boundaries: boundaries.view,
          vertexCounts: vertexCounts.view
        }
      })
    );
    const mesh = createCellFanMesh(points.count, stride, vertexCount);
    const wireframe = createCellWireframe(points.count, stride, vertexCount);
    return {
      outputs: {
        centers,
        boundaries,
        vertexCounts,
        positions,
        indices: builder.upload('fill-indices', 'uint32', mesh.indices),
        owners: builder.upload('fill-owners', 'uint32', mesh.owners),
        triangleCount: mesh.triangleCount,
        wireframe: builder.upload('wireframe', 'uint32', wireframe),
        wireframeCount: wireframe.length / 3
      }
    };
  },
  draw: ({outputs}) => [
    {layer: 'dots', context: true, positions: outputs.positions, radius: 1.6},
    {
      layer: 'polygons',
      positions: outputs.boundaries,
      indices: outputs.indices,
      owners: outputs.owners,
      values: outputs.vertexCounts,
      triangleCount: outputs.triangleCount,
      range: [0, 8]
    },
    {
      layer: 'triangles',
      positions: outputs.boundaries,
      triangles: outputs.wireframe,
      count: outputs.wireframeCount,
      radius: 1.2
    },
    {layer: 'dots', positions: outputs.centers, radius: 3.2, color: [1, 1, 1, 1]}
  ]
});

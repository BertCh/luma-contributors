// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUCellGeometry decodes cell keys to centers, boundary polygons and vertex counts. The keys come
// from GPUPointToCell on Honshu epicenters; the pointer is one more point, so the cell under the
// cursor is keyed and decoded live with its corners and center marked.

import {GPUCellGeometry, GPUPointToCell} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {getHonshuEvents, HONSHU_WINDOW} from '../../../data/point-observations';
import type {RunGlyph} from '../../interaction';
import {
  addPointerCell,
  createCellFanMesh,
  createCellWireframe,
  drawPointerCell,
  getCellFamily,
  getCellResolution,
  H3_BOUNDARY_STRIDE
} from './support/cells';
import {formatLongitudeLatitude, getWindowCenter, readPointerCell} from './support/pointer';
import {parameters} from './cell-geometry.parameters';

export default defineOperationRun({
  parameters,
  geographicWindow: HONSHU_WINDOW,
  caption: 'Cell keys decoded to boundary polygons (fill) and centers (dots).',
  createInputs: ({random}) => ({
    ...getHonshuEvents(random, 200),
    pointer: [...getWindowCenter(HONSHU_WINDOW)] as [number, number]
  }),
  build(builder, points, values) {
    const family = getCellFamily(values.family);
    const resolution = getCellResolution(family, values.resolution);
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
    const stride = family === 'h3' ? H3_BOUNDARY_STRIDE : 4;
    const vertexCount = family === 'h3' ? 6 : 4;
    const centers = builder.allocate('centers', 'float32x2', points.count);
    const boundaries = builder.allocate('boundaries', 'float32x2', points.count * stride);
    builder.graph.add(
      new GPUCellGeometry({
        family,
        cells: cells.view,
        maximumVertexCount: stride,
        output: {centers: centers.view, boundaries: boundaries.view}
      })
    );
    const mesh = createCellFanMesh(points.count, stride, vertexCount);
    const wireframe = createCellWireframe(points.count, stride, vertexCount);
    const pointer = builder.allocate('pointer', 'float32x2', 1);
    return {
      outputs: {
        centers,
        boundaries,
        positions,
        pointerCell: addPointerCell(builder, family, resolution, pointer),
        indices: builder.upload('fill-indices', 'uint32', mesh.indices),
        owners: builder.upload('fill-owners', 'uint32', mesh.owners),
        triangleCount: mesh.triangleCount,
        wireframe: builder.upload('wireframe', 'uint32', wireframe),
        wireframeCount: wireframe.length / 3
      },
      update: () => pointer.write(Float32Array.from(points.pointer))
    };
  },
  rebuildOn: ['family', 'resolution'],
  interaction: {
    kind: 'place',
    hint: 'move a point',
    rest: [0.5, 0.5],
    resolve(pointer, {inputs, outputs, peek}) {
      inputs.pointer = [pointer.x, pointer.y];
      const cell = readPointerCell(peek, outputs.pointerCell);
      const glyphs: RunGlyph[] = [];
      let center = '…';
      if (cell.outline && cell.boundaries) {
        glyphs.push(cell.outline);
        for (let corner = 0; corner < cell.vertexCount; corner++) {
          glyphs.push({
            shape: 'dot',
            at: [cell.boundaries[corner * 2], cell.boundaries[corner * 2 + 1]]
          });
        }
      }
      if (cell.centers) {
        glyphs.push({shape: 'crosshair', at: [cell.centers[0], cell.centers[1]], size: 7});
        center = formatLongitudeLatitude(cell.centers[0], cell.centers[1]);
      }
      return {
        rerun: true,
        glyphs,
        readout: `key ${cell.key} · ${cell.vertexCount} vertices · center ${center}`
      };
    }
  },
  draw: ({outputs}) => [
    {layer: 'dots', context: true, positions: outputs.positions, radius: 1.6},
    {
      layer: 'polygons',
      positions: outputs.boundaries,
      indices: outputs.indices,
      owners: outputs.owners,
      values: outputs.owners,
      triangleCount: outputs.triangleCount,
      color: [0.4, 0.7, 1, 0.16]
    },
    {
      layer: 'triangles',
      positions: outputs.boundaries,
      triangles: outputs.wireframe,
      count: outputs.wireframeCount,
      radius: 1.2
    },
    drawPointerCell(outputs.pointerCell),
    {layer: 'dots', positions: outputs.centers, radius: 3, color: [1, 1, 1, 0.9]}
  ]
});

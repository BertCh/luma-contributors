// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUPointToCell: every point gets the 64-bit key of the cell it falls in. The keys are decoded
// with GPUCellGeometry only to draw them: each outline is the cell of at least one point. The
// pointer is one more point in the same operation, so the cell under the cursor is keyed live.

import {GPUPointToCell} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {getHonshuEvents, HONSHU_WINDOW} from '../../../data/point-observations';
import {
  addCellFill,
  addPointerCell,
  createCellWireframe,
  drawPointerCell,
  getCellFamily,
  getCellResolution
} from './support/cells';
import {findKeyRows, formatCellKey, getBoundaryGlyph, getWindowCenter} from './support/pointer';
import {parameters} from './point-to-cell.parameters';

export default defineOperationRun({
  parameters,
  geographicWindow: HONSHU_WINDOW,
  caption: 'Each point falls into exactly one cell; the outlines are the cells that were hit.',
  createInputs: ({random}) => ({
    ...getHonshuEvents(random, 260),
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
    const fill = addCellFill(builder, family, cells);
    const wireframe = createCellWireframe(points.count, fill.stride, fill.vertexCount);
    const pointer = builder.allocate('pointer', 'float32x2', 1);
    return {
      outputs: {
        positions,
        cells,
        fill,
        pointerCell: addPointerCell(builder, family, resolution, pointer),
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
    resolve(pointer, {inputs, outputs, values, peek}) {
      inputs.pointer = [pointer.x, pointer.y];
      const keys = peek(outputs.pointerCell.cells);
      const dataKeys = peek(outputs.cells);
      const outline = getBoundaryGlyph(
        peek(outputs.pointerCell.boundaries),
        0,
        outputs.pointerCell.stride,
        outputs.pointerCell.vertexCount
      );
      const shared = keys && dataKeys ? findKeyRows(dataKeys, inputs.count, keys).length : 0;
      return {
        rerun: true,
        glyphs: [{shape: 'crosshair', at: inputs.pointer, size: 7}, ...(outline ? [outline] : [])],
        readout: `key ${formatCellKey(keys)} · res ${getCellResolution(values.family === 0 ? 'h3' : 'quadbin', values.resolution)} · ${shared} points`
      };
    }
  },
  draw: ({outputs}) => [
    {
      layer: 'polygons',
      positions: outputs.fill.boundaries,
      indices: outputs.fill.indices,
      owners: outputs.fill.owners,
      values: outputs.fill.owners,
      triangleCount: outputs.fill.triangleCount,
      color: [0.4, 0.7, 1, 0.16]
    },
    {
      layer: 'triangles',
      positions: outputs.fill.boundaries,
      triangles: outputs.wireframe,
      count: outputs.wireframeCount,
      radius: 1.4
    },
    drawPointerCell(outputs.pointerCell),
    {layer: 'dots', context: true, positions: outputs.positions, radius: 2.6}
  ]
});

// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUCellRollup folds a fine cell table into a coarser one without touching the points again. The
// fine cells (outlines) are aggregated by GPUCellAggregation; the filled cells are the rollup.

import {
  GPUCellAggregation,
  GPUCellRollup,
  GPUCellTopology
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import type {RunGlyph} from '../../interaction';
import {getHonshuEvents, HONSHU_WINDOW} from '../../../data/point-observations';
import {
  addCellFill,
  addPointerCell,
  addPointKeys,
  allocateCellTable,
  createCellWireframe,
  drawPointerCell,
  getCellFamily,
  getCellResolution,
  getCellTableViews,
  H3_BOUNDARY_STRIDE
} from './support/cells';
import {findKeyRows, getBoundaryGlyph, getWindowCenter, readPointerCell} from './support/pointer';
import {parameters} from './cell-rollup.parameters';

const FINE_CAPACITY = 3072;
const COARSE_CAPACITY = 1024;

export default defineOperationRun({
  parameters,
  geographicWindow: HONSHU_WINDOW,
  caption: 'Fine cell table rolled up into parent cells: sums add up exactly.',
  createInputs: ({random}) => ({
    ...getHonshuEvents(random, 900),
    pointer: [...getWindowCenter(HONSHU_WINDOW)] as [number, number]
  }),
  build(builder, points, values) {
    const family = getCellFamily(values.family);
    const resolution = getCellResolution(family, values.resolution);
    const sourceResolution = resolution + 2;
    const positions = builder.upload('positions', 'float32x2', points.positions);
    const pointValues = builder.upload('values', 'float32', points.values);
    const keys = addPointKeys(builder, family, sourceResolution, positions);
    const fine = allocateCellTable(builder, 'fine', FINE_CAPACITY);
    const coarse = allocateCellTable(builder, 'coarse', COARSE_CAPACITY);
    builder.graph.add(
      new GPUCellAggregation({
        family,
        resolution: sourceResolution,
        cells: keys.view,
        values: pointValues.view,
        sumScale: 65536,
        output: getCellTableViews(fine)
      })
    );
    builder.graph.add(
      new GPUCellRollup({
        family,
        sourceResolution,
        resolution,
        source: getCellTableViews(fine),
        sumScale: 65536,
        output: getCellTableViews(coarse)
      })
    );
    const fineFill = addCellFill(builder, family, fine.cells, 'fine-fill');
    // Each fine row's parent key: which fine cells belong to the cell under the pointer.
    const parents = builder.allocate('fine-parents', 'uint32x2', FINE_CAPACITY);
    builder.graph.add(
      new GPUCellTopology({
        id: 'fine-parents-topology',
        family,
        operation: {type: 'parent', resolution},
        cells: fine.cells.view,
        output: {cells: parents.view}
      })
    );
    const pointer = builder.allocate('pointer', 'float32x2', 1);
    const stride = family === 'h3' ? H3_BOUNDARY_STRIDE : 4;
    const wireframe = createCellWireframe(FINE_CAPACITY, stride, family === 'h3' ? 6 : 4);
    return {
      outputs: {
        coarse,
        fine,
        fineFill,
        parents,
        pointerCell: addPointerCell(builder, family, resolution, pointer),
        fineBoundaries: fineFill.boundaries,
        fineWireframe: builder.upload('fine-wireframe', 'uint32', wireframe),
        fineWireframeCount: wireframe.length / 3,
        coarseFill: addCellFill(builder, family, coarse.cells, 'coarse-fill')
      },
      update: () => pointer.write(Float32Array.from(points.pointer))
    };
  },
  rebuildOn: ['family', 'resolution'],
  // The pointer picks a coarse cell; its fine cells are the rows whose parent key is that cell.
  interaction: {
    kind: 'probe',
    hint: 'hover a parent cell',
    rest: [0.5, 0.5],
    resolve(pointer, {inputs, outputs, peek}) {
      inputs.pointer = [pointer.x, pointer.y];
      const cell = readPointerCell(peek, outputs.pointerCell);
      const parents = peek(outputs.parents);
      const fineKeys = peek(outputs.fine.cells);
      const fineCount = peek(outputs.fine.count);
      const fineSums = peek(outputs.fine.sumValues);
      const coarseKeys = peek(outputs.coarse.cells);
      const coarseCount = peek(outputs.coarse.count);
      const coarseSums = peek(outputs.coarse.sumValues);
      const fineBoundaries = peek(outputs.fineBoundaries);
      const glyphs: RunGlyph[] = cell.outline ? [cell.outline] : [];
      if (
        !cell.keys ||
        !parents ||
        !fineKeys ||
        !fineCount ||
        !fineSums ||
        !coarseKeys ||
        !coarseCount ||
        !coarseSums
      ) {
        return {rerun: true, glyphs, readout: 'parent …'};
      }
      const children = findKeyRows(parents, fineCount[0], cell.keys);
      const [coarseRow] = findKeyRows(coarseKeys, coarseCount[0], cell.keys);
      let childSum = 0;
      for (const child of children) {
        childSum += fineSums[child];
        const outline = getBoundaryGlyph(
          fineBoundaries,
          child,
          outputs.fineFill.stride,
          outputs.fineFill.vertexCount
        );
        if (outline) glyphs.push(outline);
      }
      const parentSum = coarseRow === undefined ? 0 : coarseSums[coarseRow];
      return {
        rerun: true,
        glyphs,
        readout: `parent sum ${parentSum.toFixed(1)} = Σ ${children.length} children ${childSum.toFixed(1)}`
      };
    }
  },
  draw: ({outputs}) => [
    {
      layer: 'polygons',
      positions: outputs.coarseFill.boundaries,
      indices: outputs.coarseFill.indices,
      owners: outputs.coarseFill.owners,
      values: outputs.coarse.sumValues,
      triangleCount: outputs.coarseFill.triangleCount,
      range: [0, 120]
    },
    drawPointerCell(outputs.pointerCell, [1, 1, 1, 0.22]),
    {
      layer: 'triangles',
      context: true,
      positions: outputs.fineBoundaries,
      triangles: outputs.fineWireframe,
      count: outputs.fineWireframeCount
    }
  ]
});

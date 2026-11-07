// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUCellRollup folds a fine cell table into a coarser one without touching the points again. The
// fine cells (outlines) are aggregated by GPUCellAggregation; the filled cells are the rollup.

import {GPUCellAggregation, GPUCellRollup} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {createClusteredPoints} from '../../inputs';
import {
  addCellFill,
  addPointKeys,
  allocateCellTable,
  createCellWireframe,
  getCellFamily,
  getCellResolution,
  getCellTableViews,
  H3_BOUNDARY_STRIDE
} from './support/cells';
import {parameters} from './cell-rollup.parameters';

const FINE_CAPACITY = 3072;
const COARSE_CAPACITY = 512;

export default defineOperationRun({
  parameters,
  caption: 'Fine cell table rolled up into parent cells: sums add up exactly.',
  createInputs: ({random, world}) => createClusteredPoints(random, world, 900, 4, 0.12),
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
    const stride = family === 'h3' ? H3_BOUNDARY_STRIDE : 4;
    const wireframe = createCellWireframe(FINE_CAPACITY, stride, family === 'h3' ? 6 : 4);
    return {
      outputs: {
        coarse,
        fineBoundaries: fineFill.boundaries,
        fineWireframe: builder.upload('fine-wireframe', 'uint32', wireframe),
        fineWireframeCount: wireframe.length / 3,
        coarseFill: addCellFill(builder, family, coarse.cells, 'coarse-fill')
      },
      update: () => undefined
    };
  },
  rebuildOn: ['family', 'resolution'],
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
    {
      layer: 'triangles',
      context: true,
      positions: outputs.fineBoundaries,
      triangles: outputs.fineWireframe,
      count: outputs.fineWireframeCount
    }
  ]
});

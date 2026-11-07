// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUCellAggregation bins points into a sorted cell table (count, sum, minimum and maximum per
// cell). The table's cell keys are decoded with GPUCellGeometry only to draw them.

import {GPUCellAggregation} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {createClusteredPoints} from '../../inputs';
import {
  addCellFill,
  addPointKeys,
  allocateCellTable,
  getCellFamily,
  getCellResolution,
  getCellTableViews
} from './support/cells';
import {parameters} from './cell-aggregation.parameters';

export default defineOperationRun({
  parameters,
  caption: 'Points binned into cells: one table row per occupied cell.',
  createInputs: ({random, world}) => createClusteredPoints(random, world, 900, 4, 0.12),
  build(builder, points, values) {
    const family = getCellFamily(values.family);
    const positions = builder.upload('positions', 'float32x2', points.positions);
    const pointValues = builder.upload('values', 'float32', points.values);
    const table = allocateCellTable(builder, 'table', 1024);
    const resolution = getCellResolution(family, values.resolution);
    const keys = addPointKeys(builder, family, resolution, positions);
    builder.graph.add(
      new GPUCellAggregation({
        family,
        resolution,
        cells: keys.view,
        values: pointValues.view,
        sumScale: 65536,
        output: getCellTableViews(table)
      })
    );
    return {
      outputs: {table, positions, fill: addCellFill(builder, family, table.cells)},
      update: () => undefined
    };
  },
  rebuildOn: ['family', 'resolution'],
  draw: ({outputs, values}) => [
    {layer: 'dots', context: true, positions: outputs.positions, radius: 1.4},
    {
      layer: 'polygons',
      positions: outputs.fill.boundaries,
      indices: outputs.fill.indices,
      owners: outputs.fill.owners,
      values: [outputs.table.sumValues, outputs.table.minimums, outputs.table.maximums][
        values.measure
      ],
      triangleCount: outputs.fill.triangleCount,
      ramp: 'sequential',
      range: values.measure === 0 ? [0, 30] : 'auto'
    }
  ]
});

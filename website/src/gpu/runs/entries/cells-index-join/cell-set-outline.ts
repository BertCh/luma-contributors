// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUCellSetOutline emits only the boundary edges of a set of cells: edges whose neighbor is not in
// the set. The set is a GPUCellAggregation table (sorted, distinct keys); the bright lines are the
// operation's output, drawn exactly `status.count` times.

import {
  GPUCellAggregation,
  GPUCellSetOutline
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {createClusteredPoints} from '../../inputs';
import {
  addCellFill,
  addPointKeys,
  allocateCellTable,
  getCellFamily,
  getCellResolution,
  getCountTableViews
} from './support/cells';
import {parameters} from './cell-set-outline.parameters';

const TABLE_CAPACITY = 512;
const EDGES_PER_CELL = {h3: 10, quadbin: 4};

export default defineOperationRun({
  parameters,
  caption: 'Only the edges on the border of the cell set survive.',
  createInputs: ({random, world}) => createClusteredPoints(random, world, 700, 3, 0.04),
  build(builder, points, values) {
    const family = getCellFamily(values.family);
    const resolution = getCellResolution(family, values.resolution);
    const positions = builder.upload('positions', 'float32x2', points.positions);
    const keys = addPointKeys(builder, family, resolution, positions);
    const table = allocateCellTable(builder, 'table', TABLE_CAPACITY);
    builder.graph.add(
      new GPUCellAggregation({
        family,
        resolution,
        cells: keys.view,
        output: getCountTableViews(table)
      })
    );
    const capacity = TABLE_CAPACITY * EDGES_PER_CELL[family];
    const outline = {
      rows: builder.allocate('outline-rows', 'uint32', capacity),
      cells: builder.allocate('outline-cells', 'uint32x2', capacity),
      edgeIndices: builder.allocate('outline-edges', 'uint32', capacity),
      endpoints: builder.allocate('outline-endpoints', 'float32x4', capacity),
      count: builder.allocate('outline-count', 'uint32', 1),
      overflow: builder.allocate('outline-overflow', 'uint32', 1)
    };
    builder.graph.add(
      new GPUCellSetOutline({
        family,
        cells: table.cells.view,
        count: table.count.view,
        output: {
          rows: outline.rows.view,
          cells: outline.cells.view,
          edgeIndices: outline.edgeIndices.view,
          endpoints: outline.endpoints.view,
          status: {count: outline.count.view, overflow: outline.overflow.view}
        }
      })
    );
    return {
      outputs: {outline, fill: addCellFill(builder, family, table.cells)},
      update: current =>
        positions.write(
          Float32Array.from(points.positions, (value, index) => {
            const anchor = index % 2 === 0 ? 0.9 : 0.5;
            return anchor + (value - anchor) * current.spread;
          })
        )
    };
  },
  rebuildOn: ['family', 'resolution'],
  draw: ({outputs}) => [
    {
      layer: 'polygons',
      context: true,
      positions: outputs.fill.boundaries,
      indices: outputs.fill.indices,
      owners: outputs.fill.owners,
      values: outputs.fill.owners,
      triangleCount: outputs.fill.triangleCount,
      color: [0.4, 0.5, 0.7, 0.35]
    },
    {
      layer: 'lines',
      segments: outputs.outline.endpoints,
      count: outputs.outline.count,
      color: [1, 0.75, 0.2, 1],
      radius: 2
    }
  ]
});

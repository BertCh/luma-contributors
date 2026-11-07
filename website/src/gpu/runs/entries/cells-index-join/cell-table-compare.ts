// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUCellTableCompare outer-joins two cell tables (before and after) into one union table with the
// per-cell difference. The "after" points are the "before" points moved east by `shift`; cells that
// lost points are blue, cells that gained points are orange.

import {
  GPUCellAggregation,
  GPUCellTableCompare
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {createClusteredPoints} from '../../inputs';
import {addCellFill, addPointKeys, allocateCellTable, getCellTableViews} from './support/cells';
import {parameters} from './cell-table-compare.parameters';

const TABLE_CAPACITY = 1024;
const UNION_CAPACITY = 2048;

export default defineOperationRun({
  parameters,
  caption: 'Union of two cell tables; color is after minus before.',
  createInputs: ({random, world}) => createClusteredPoints(random, world, 700, 4, 0.1),
  build(builder, points, values) {
    const resolution = values.resolution;
    const beforePositions = builder.upload('before-positions', 'float32x2', points.positions);
    const afterPositions = builder.allocate('after-positions', 'float32x2', points.count);
    const pointValues = builder.upload('values', 'float32', points.values);
    const tables = [
      {name: 'before', positions: beforePositions},
      {name: 'after', positions: afterPositions}
    ].map(({name, positions}) => {
      const keys = addPointKeys(builder, 'h3', resolution, positions, `${name}-keys`);
      const table = allocateCellTable(builder, name, TABLE_CAPACITY);
      builder.graph.add(
        new GPUCellAggregation({
          id: `${name}-aggregation`,
          family: 'h3',
          resolution,
          cells: keys.view,
          values: pointValues.view,
          sumScale: 65536,
          output: getCellTableViews(table)
        })
      );
      return table;
    });
    const union = {
      cells: builder.allocate('union-cells', 'uint32x2', UNION_CAPACITY),
      delta: builder.allocate('delta', 'float32', UNION_CAPACITY),
      count: builder.allocate('union-count', 'uint32', 1),
      overflow: builder.allocate('union-overflow', 'uint32', 1)
    };
    builder.graph.add(
      new GPUCellTableCompare({
        before: getCellTableViews(tables[0]),
        after: getCellTableViews(tables[1]),
        measure: values.measure === 0 ? 'count' : 'sum',
        sumScale: 65536,
        output: {
          cells: union.cells.view,
          delta: union.delta.view,
          status: {count: union.count.view, overflow: union.overflow.view}
        }
      })
    );
    return {
      outputs: {union, fill: addCellFill(builder, 'h3', union.cells, 'union-fill')},
      update: current => {
        const moved = Float32Array.from(points.positions, (value, index) =>
          index % 2 === 0 ? value + current.shift : value
        );
        afterPositions.write(moved);
      }
    };
  },
  rebuildOn: ['measure', 'resolution'],
  draw: ({outputs}) => [
    {
      layer: 'polygons',
      positions: outputs.fill.boundaries,
      indices: outputs.fill.indices,
      owners: outputs.fill.owners,
      values: outputs.union.delta,
      triangleCount: outputs.fill.triangleCount,
      ramp: 'diverging'
    }
  ]
});

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
import {bindAxes} from '../../interaction';
import {getHonshuEvents, HONSHU_WINDOW} from '../../../data/point-observations';
import {addCellFill, addPointKeys, allocateCellTable, getCellTableViews} from './support/cells';
import {getWindowCenter} from './support/pointer';
import {parameters} from './cell-table-compare.parameters';

const TABLE_CAPACITY = 1024;
const UNION_CAPACITY = 2048;

const SHIFT_PAD = bindAxes(parameters, {
  kind: 'pad',
  hint: 'shift the after points',
  x: 'shift',
  y: 'shiftNorth'
});

export default defineOperationRun({
  parameters,
  geographicWindow: HONSHU_WINDOW,
  caption: 'Union of two cell tables; color is after minus before.',
  createInputs: ({random}) => getHonshuEvents(random, 700),
  build(builder, points, values) {
    const resolution = values['resolution'];
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
      overflow: builder.allocate('union-overflow', 'uint32', 1),
      candidateOverflow: builder.allocate('union-candidate-overflow', 'uint32', 1)
    };
    builder.graph.add(
      new GPUCellTableCompare({
        before: getCellTableViews(tables[0]),
        after: getCellTableViews(tables[1]),
        measure: values['measure'] === 0 ? 'count' : 'sum',
        sumScale: 65536,
        output: {
          cells: union.cells.view,
          delta: union.delta.view,
          status: {
            count: union.count.view,
            overflow: union.overflow.view,
            candidateOverflow: union.candidateOverflow.view
          }
        }
      })
    );
    return {
      outputs: {union, fill: addCellFill(builder, 'h3', union.cells, 'union-fill')},
      update: current => {
        const moved = Float32Array.from(
          points.positions,
          (value, index) => value + (index % 2 === 0 ? current['shift'] : current['shiftNorth'])
        );
        afterPositions.write(moved);
      }
    };
  },
  rebuildOn: ['measure', 'resolution'],
  // The pointer sets how far the "after" points moved (east and north), and so which cells
  // gained or lost points.
  interaction: {
    ...SHIFT_PAD,
    resolve(pointer, state) {
      const base = SHIFT_PAD.resolve(pointer, state);
      const values = {...state.values, ...base.values};
      const center = getWindowCenter(HONSHU_WINDOW);
      const delta = state.peek(state.outputs.union.delta);
      const count = state.peek(state.outputs.union.count);
      const overflow = state.peek(state.outputs.union.overflow)?.[0];
      const candidateOverflow = state.peek(state.outputs.union.candidateOverflow)?.[0];
      let gained = 0;
      let lost = 0;
      if (delta && count) {
        for (let row = 0; row < count[0]; row++) {
          if (delta[row] > 0) gained++;
          else if (delta[row] < 0) lost++;
        }
      }
      return {
        ...base,
        glyphs: [
          {shape: 'dot', at: center},
          {
            shape: 'arrow',
            from: center,
            to: [center[0] + values['shift'], center[1] + values['shiftNorth']]
          }
        ],
        readout: `shift ${values['shift'].toFixed(2)}°E ${values['shiftNorth'].toFixed(2)}°N · +${gained} / −${lost} cells${overflow ? ' · union capacity reached' : ''}${candidateOverflow ? ' · input loss' : ''}`
      };
    }
  },
  draw: ({outputs, values}) => [
    {
      layer: 'polygons',
      positions: outputs.fill.boundaries,
      indices: outputs.fill.indices,
      owners: outputs.fill.owners,
      values: outputs.union.delta,
      triangleCount: outputs.fill.triangleCount,
      ramp: 'diverging',
      // Fixed, so a bigger shift looks stronger instead of being stretched to the same colors.
      range: values['measure'] === 0 ? [-8, 8] : [-3, 3]
    }
  ]
});

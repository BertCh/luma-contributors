// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUCostDistancePath follows the back-links of a GPUCostDistance run from the target cell to its
// source: the cheapest route, which has to pass one of the gaps in the wall. The path it wrote is
// drawn as a line over the cost bands.

import {GPUCostDistancePath} from '@luma-contributors/analysis/raster-analysis';
import {defineOperationRun} from '../../define-run';
import {parameters} from './cost-distance-path.parameters';
import {clamp} from '../../interaction';
import {addCostDistance, createCostInputs} from './shared/cost';

function getTargetCell(
  inputs: {columns: number; rows: number},
  targetColumn: number,
  targetRow: number
) {
  const column = Math.min(inputs.columns - 1, Math.floor((inputs.columns * targetColumn) / 100));
  const row = Math.min(inputs.rows - 1, Math.floor((inputs.rows * targetRow) / 100));
  return {column, row, index: row * inputs.columns + column};
}

export default defineOperationRun({
  parameters,
  caption: 'The route follows the cheapest valley and detours through a gap in the wall.',
  createInputs: context => ({
    ...createCostInputs(context),
    // The last target that was not inside the wall, in the parameters' percent.
    lastTarget: [parameters[0].default, parameters[1].default] as [number, number]
  }),
  build(builder, inputs) {
    const cost = addCostDistance(builder, inputs);
    const target = builder.allocate('target', 'uint32', 1);
    const pathIds = builder.allocate('path-ids', 'uint32', inputs.cellCount);
    const pathCount = builder.allocate('path-count', 'uint32', 1);
    builder.graph.add(
      new GPUCostDistancePath({
        grid: inputs.grid,
        backLinks: cost.backLinks.view,
        target: target.view,
        output: {
          path: {
            ids: pathIds.view,
            status: {
              count: pathCount.view,
              overflow: builder.allocate('overflow', 'uint32', 1).view
            }
          }
        }
      })
    );
    const cellCenters = builder.upload('cell-centers', 'float32x2', inputs.cellCenters);
    return {
      outputs: {costs: cost.costs, sources: cost.sources, cellCenters, pathIds, pathCount},
      update: values => {
        cost.writeSettings({...values, sourceCount: 1});
        target.write(
          Uint32Array.of(getTargetCell(inputs, values.targetColumn, values.targetRow).index)
        );
      }
    };
  },
  // The pointer is the target: the route is traced back from the cell under it to the source. The
  // wall cells have no cost, so the target stays on the last reachable cell.
  interaction: {
    kind: 'place',
    hint: 'move the target',
    rest: [parameters[0].default / 100, parameters[1].default / 100],
    resolve(pointer, {inputs, outputs, peek}) {
      let targetColumn = clamp(Math.round(pointer.unitX * 100), 0, 99);
      let targetRow = clamp(Math.round(pointer.unitY * 100), 0, 99);
      if (inputs.isWallCell[getTargetCell(inputs, targetColumn, targetRow).index]) {
        [targetColumn, targetRow] = inputs.lastTarget;
      } else {
        inputs.lastTarget = [targetColumn, targetRow];
      }
      const cell = getTargetCell(inputs, targetColumn, targetRow);
      const costs = peek(outputs.costs);
      const cost = costs ? costs[cell.index] : Number.NaN;
      return {
        values: {targetColumn, targetRow},
        glyphs: [
          {
            shape: 'ring',
            at: [inputs.cellCenters[cell.index * 2], inputs.cellCenters[cell.index * 2 + 1]],
            radius: inputs.cellSize * 2.5
          }
        ],
        readout: `target [${cell.column}, ${cell.row}]${Number.isFinite(cost) ? ` · cost ${cost.toFixed(1)}` : ''}`
      };
    }
  },
  draw: ({inputs, outputs}) => [
    {
      layer: 'raster',
      context: true,
      values: outputs.costs,
      columns: inputs.columns,
      rows: inputs.rows,
      ramp: 'sequential',
      bands: 14
    },
    {
      layer: 'dots',
      context: true,
      positions: outputs.sources,
      count: 1,
      radius: 5,
      color: [1, 1, 1, 1]
    },
    {
      layer: 'paths',
      nodes: outputs.cellCenters,
      ids: outputs.pathIds,
      count: outputs.pathCount,
      radius: 5,
      color: [1, 0.95, 0.45, 1]
    }
  ]
});

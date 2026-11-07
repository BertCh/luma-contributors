// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUCostDistancePath follows the back-links of a GPUCostDistance run from the target cell to its
// source: the cheapest route, which has to pass one of the gaps in the wall. The path it wrote is
// drawn as a line over the cost bands.

import {GPUCostDistancePath} from '@luma-contributors/analysis/raster-analysis';
import {defineOperationRun} from '../../define-run';
import {parameters} from './cost-distance-path.parameters';
import {addCostDistance, createCostInputs} from './shared/cost';

export default defineOperationRun({
  parameters,
  caption: 'The route follows the cheapest valley and detours through a gap in the wall.',
  createInputs: context => createCostInputs(context),
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
        const column = Math.min(
          inputs.columns - 1,
          Math.floor((inputs.columns * values.targetColumn) / 100)
        );
        const row = Math.min(inputs.rows - 1, Math.floor((inputs.rows * values.targetRow) / 100));
        target.write(Uint32Array.of(row * inputs.columns + column));
      }
    };
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

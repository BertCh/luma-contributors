// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUCostDistance over a friction raster with a wall (two gaps) across it: least accumulated cost
// from the sources, drawn as bands. Cost fronts squeeze through the gaps and bend around high
// friction; cells beyond costLimit stay empty.

import {defineOperationRun} from '../../define-run';
import {parameters} from './cost-distance.parameters';
import {addCostDistance, createCostInputs} from './shared/cost';

export default defineOperationRun({
  parameters,
  caption: 'Cost rings bend around high-friction terrain and squeeze through the gaps in the wall.',
  createInputs: context => createCostInputs(context),
  build(builder, inputs) {
    const cost = addCostDistance(builder, inputs);
    return {
      outputs: {friction: cost.friction, costs: cost.costs, sources: cost.sources},
      update: values => cost.writeSettings(values)
    };
  },
  draw: ({inputs, outputs, values}) => [
    {
      layer: 'raster',
      context: true,
      values: outputs.friction,
      columns: inputs.columns,
      rows: inputs.rows
    },
    {
      layer: 'raster',
      values: outputs.costs,
      columns: inputs.columns,
      rows: inputs.rows,
      ramp: 'sequential',
      range: [0, values.costLimit],
      bands: 14,
      alpha: 0.9
    },
    {
      layer: 'dots',
      context: true,
      positions: outputs.sources,
      count: Math.round(values.sourceCount),
      radius: 5,
      color: [1, 1, 1, 1]
    }
  ]
});

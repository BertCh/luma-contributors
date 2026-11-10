// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUCostDistance over a friction raster with a wall (two gaps) across it: least accumulated cost
// from the sources, drawn as bands. Cost fronts squeeze through the gaps and bend around high
// friction; cells beyond costLimit stay empty.

import {defineOperationRun} from '../../define-run';
import {parameters} from './cost-distance.parameters';
import {addCostDistance, createCostInputs, getCostCell} from './shared/cost';

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
  // The pointer is the first source: the cost fronts are recomputed from the cell under it (the
  // wall is not a valid source, so the source stays on its side).
  interaction: {
    kind: 'place',
    hint: 'move the source',
    rest: ({inputs, toUnit}) => {
      const cell = inputs.defaultSources[0];
      return toUnit([inputs.cellCenters[cell * 2], inputs.cellCenters[cell * 2 + 1]]);
    },
    resolve(pointer, {inputs, values}) {
      const cell = getCostCell(inputs, pointer.x, pointer.y);
      let rerun = false;
      if (!inputs.isWallCell[cell.index] && inputs.sources[0] !== cell.index) {
        inputs.sources[0] = cell.index;
        rerun = true;
      }
      const source = inputs.sources[0];
      const at = [inputs.cellCenters[source * 2], inputs.cellCenters[source * 2 + 1]] as const;
      return {
        rerun,
        glyphs: [{shape: 'crosshair', at}],
        readout: `source [${source % inputs.columns}, ${Math.floor(source / inputs.columns)}] · costLimit ${values.costLimit}`
      };
    }
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

// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// addExcessRiskRecipe: empirical Bayes excess risk per tract. Fills are the shrunken excess risk;
// small tracts are pulled toward the regional rate.

import {addExcessRiskRecipe} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {createTracts} from '../../inputs';
import {createRateDraws, getRateCounts, getTractFill, uploadTractMesh} from './lib/tracts';
import {parameters} from './excess-risk-recipe.parameters';

export default defineOperationRun({
  parameters,
  caption: 'Excess risk per tract after empirical Bayes shrinkage.',
  createInputs: ({random, world}) => {
    const tracts = createTracts(random, world);
    return {tracts, draws: createRateDraws(random, tracts.count)};
  },
  build(builder, {tracts, draws}) {
    const events = builder.allocate('events', 'uint32', tracts.count);
    const populations = builder.allocate('populations', 'float32', tracts.count);
    const excessRisk = builder.allocate('excess-risk', 'float32', tracts.count);
    addExcessRiskRecipe(builder.graph, {
      events: events.view,
      populations: populations.view,
      output: excessRisk.view
    });
    return {
      outputs: {excessRisk, mesh: uploadTractMesh(builder, tracts)},
      update: current => {
        const counts = getRateCounts(tracts, draws, current.contrast, current.smallShare);
        events.write(counts.events);
        populations.write(counts.populations);
      }
    };
  },
  draw: ({inputs, outputs}) => [
    getTractFill(outputs.mesh, inputs.tracts, outputs.excessRisk, {ramp: 'sequential'}),
    {
      layer: 'lines',
      context: true,
      segments: outputs.mesh.outline,
      count: inputs.tracts.outlineCount
    }
  ]
});

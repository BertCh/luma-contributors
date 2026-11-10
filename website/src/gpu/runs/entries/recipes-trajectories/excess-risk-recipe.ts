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
    const rawRatio = builder.allocate('raw-ratio', 'float32', tracts.count);
    addExcessRiskRecipe(builder.graph, {
      events: events.view,
      populations: populations.view,
      output: excessRisk.view
    });
    return {
      outputs: {excessRisk, rawRatio, mesh: uploadTractMesh(builder, tracts)},
      update: current => {
        const counts = getRateCounts(tracts, draws, current.contrast, current.smallShare);
        events.write(counts.events);
        populations.write(counts.populations);
        // Events over the expected count at the regional rate: the unshrunk ratio.
        let totalEvents = 0;
        let totalPopulation = 0;
        for (let tract = 0; tract < tracts.count; tract++) {
          totalEvents += counts.events[tract];
          totalPopulation += counts.populations[tract];
        }
        const regionalRate = totalEvents / totalPopulation;
        rawRatio.write(
          Float32Array.from(
            counts.events,
            (count, tract) => count / (counts.populations[tract] * regionalRate)
          )
        );
      }
    };
  },
  // A curtain between the raw events / expected ratio (left) and the shrunken excess risk (right).
  interaction: {
    kind: 'swipe',
    hint: 'swipe raw and shrunk',
    rest: [0.5, 0.5],
    resolve: pointer => ({
      curtain: pointer.x,
      glyphs: [{shape: 'curtain', x: pointer.x}],
      readout: 'raw ratio | empirical Bayes excess risk'
    })
  },
  draw: ({inputs, outputs, interaction, world}) => {
    const curtain = interaction.curtain ?? world.width / 2;
    return [
      getTractFill(outputs.mesh, inputs.tracts, outputs.rawRatio, {
        ramp: 'diverging',
        range: [0.4, 1.6],
        clip: {side: 'left', x: curtain}
      }),
      getTractFill(outputs.mesh, inputs.tracts, outputs.excessRisk, {
        ramp: 'diverging',
        range: [0.4, 1.6],
        clip: {side: 'right', x: curtain}
      }),
      {
        layer: 'lines',
        context: true,
        segments: outputs.mesh.outline,
        count: inputs.tracts.outlineCount
      }
    ];
  }
});

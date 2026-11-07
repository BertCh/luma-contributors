// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// addSpatialRateRecipe: spatial empirical Bayes rates, shrunken toward the neighborhood rate.
// Tracts without neighbors (islands) are undefined (blank), or keep their own rate with
// islandPolicy 'own-rate'.

import {addSpatialRateRecipe} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {createTracts, uploadSpatialWeights} from '../../inputs';
import {
  createRateDraws,
  createShuffledOrder,
  getRateCounts,
  getTractFill,
  isolateTracts,
  uploadTractMesh
} from './lib/tracts';
import {parameters} from './spatial-rate-recipe.parameters';

export default defineOperationRun({
  parameters,
  caption: 'Spatially smoothed rates; islands are blank unless they keep their own rate.',
  createInputs: ({random, world}) => {
    const tracts = createTracts(random, world);
    return {
      tracts,
      draws: createRateDraws(random, tracts.count),
      order: createShuffledOrder(random, tracts.count)
    };
  },
  build(builder, {tracts, draws, order}, values) {
    const events = builder.allocate('events', 'uint32', tracts.count);
    const populations = builder.allocate('populations', 'float32', tracts.count);
    const spatialRates = builder.allocate('spatial-rates', 'float32', tracts.count);
    addSpatialRateRecipe(builder.graph, {
      events: events.view,
      populations: populations.view,
      weights: uploadSpatialWeights(
        builder,
        isolateTracts(tracts.contiguity, order, values.islandCount)
      ),
      islandPolicy: values.islandPolicy === 1 ? 'own-rate' : 'undefined',
      output: spatialRates.view
    });
    return {
      outputs: {spatialRates, mesh: uploadTractMesh(builder, tracts)},
      update: current => {
        const counts = getRateCounts(tracts, draws, 1, current.smallShare);
        events.write(counts.events);
        populations.write(counts.populations);
      }
    };
  },
  rebuildOn: ['islandPolicy', 'islandCount'],
  draw: ({inputs, outputs}) => [
    getTractFill(outputs.mesh, inputs.tracts, outputs.spatialRates, {
      ramp: 'sequential',
      range: [0.01, 0.07]
    }),
    {
      layer: 'lines',
      context: true,
      segments: outputs.mesh.outline,
      count: inputs.tracts.outlineCount
    }
  ]
});

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
import {
  findZoneAt,
  getNeighborhoodGlyphs,
  getTractGeometry,
  getZoneOutlineSegments
} from './lib/zone-lookup';
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
    const islands = order.slice(0, values.islandCount);
    const islandSegments = getZoneOutlineSegments(getTractGeometry(tracts), islands);
    // A buffer cannot be empty: no islands draw one degenerate segment.
    const islandOutline = builder.upload(
      'island-outline',
      'float32x4',
      islandSegments.length ? islandSegments : new Float32Array(4)
    );
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
      outputs: {
        spatialRates,
        islandOutline,
        mesh: uploadTractMesh(builder, tracts)
      },
      update: current => {
        const counts = getRateCounts(tracts, draws, 1, current.smallShare);
        events.write(counts.events);
        populations.write(counts.populations);
      }
    };
  },
  rebuildOn: ['islandPolicy', 'islandCount'],
  // The pointer picks a tract: the neighbors it was shrunk toward. An island has none.
  interaction: {
    kind: 'probe',
    hint: 'hover a tract',
    rest: ({toUnit, inputs}) => toUnit([inputs.tracts.centers[40], inputs.tracts.centers[41]]),
    resolve(pointer, {inputs, outputs, values, peek}) {
      const geometry = getTractGeometry(inputs.tracts);
      const tract = findZoneAt(geometry, pointer.x, pointer.y);
      const weights = isolateTracts(inputs.tracts.contiguity, inputs.order, values.islandCount);
      const around = weights.neighbors.subarray(weights.offsets[tract], weights.offsets[tract + 1]);
      const raw = getRateCounts(inputs.tracts, inputs.draws, 1, values.smallShare).rates[tract];
      const smoothed = peek(outputs.spatialRates);
      return {
        glyphs: getNeighborhoodGlyphs(geometry, tract, around),
        readout:
          around.length === 0
            ? `tract ${tract} · island · raw ${raw.toFixed(3)}`
            : `raw ${raw.toFixed(3)} -> ${smoothed ? smoothed[tract].toFixed(3) : '...'} · ${around.length} neighbors`
      };
    }
  },
  draw: ({inputs, outputs}) => [
    getTractFill(outputs.mesh, inputs.tracts, outputs.spatialRates, {
      ramp: 'sequential',
      range: [0.02, 0.06]
    }),
    {
      layer: 'lines',
      segments: outputs.islandOutline,
      count: outputs.islandOutline.length,
      radius: 2.5
    },
    {
      layer: 'lines',
      context: true,
      segments: outputs.mesh.outline,
      count: inputs.tracts.outlineCount
    }
  ]
});

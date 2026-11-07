// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// addPopulationSupportAlignmentRecipe: source populations (listed in a shuffled order) joined to
// the target units by exact id. The map shows the validity the recipe certifies per target unit:
// one color for exactly matched units, others for absent and repeated ids.

import {addPopulationSupportAlignmentRecipe} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {createShuffledOrder} from './lib/tracts';
import {createQuadZones, getQuadFill, uploadQuadZones} from './lib/quads';
import {addReadout} from './lib/readout';
import {parameters} from './population-support-alignment-recipe.parameters';

export default defineOperationRun({
  parameters,
  caption: 'Exact-id join of populations to target units: matched, absent and repeated ids.',
  createInputs: ({random, world}) => {
    const zones = createQuadZones(random, world, 8, 5);
    return {
      zones,
      sourceOrder: createShuffledOrder(random, zones.count),
      targetOrder: createShuffledOrder(random, zones.count)
    };
  },
  build(builder, {zones, sourceOrder, targetOrder}) {
    const populations = builder.upload(
      'populations',
      'float32x2',
      Float32Array.from(
        {length: zones.count * 2},
        (_, index) => 100 + sourceOrder[Math.floor(index / 2)] * 7 + (index % 2) * 50
      )
    );
    const sourceIds = builder.upload('source-ids', 'uint32', Uint32Array.from(sourceOrder));
    const targetIds = builder.allocate('target-ids', 'uint32', zones.count);
    const recipe = addPopulationSupportAlignmentRecipe(builder.graph, {
      populations: populations.view,
      sourceIds: sourceIds.view,
      targetIds: targetIds.view
    });
    return {
      outputs: {
        validity: addReadout(builder, 'validity', recipe.outputs.validity),
        mesh: uploadQuadZones(builder, zones)
      },
      update: current => {
        const ids = Uint32Array.from({length: zones.count}, (_, unit) => unit);
        for (let slot = 0; slot < current.absent; slot++) ids[targetOrder[slot]] = 1000 + slot;
        // A repeated id copies its neighbor in the order.
        for (let slot = 0; slot < current.duplicated; slot++) {
          const unit = targetOrder[zones.count - 1 - slot];
          ids[unit] = ids[(unit + 1) % zones.count];
        }
        targetIds.write(ids);
      }
    };
  },
  draw: ({inputs, outputs}) => [
    getQuadFill(outputs.mesh, inputs.zones, outputs.validity),
    {
      layer: 'lines',
      context: true,
      segments: outputs.mesh.outline,
      count: inputs.zones.outlineCount
    }
  ]
});

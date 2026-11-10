// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// addPopulationSupportAlignmentRecipe: source populations (listed in a shuffled order) joined to
// the target units by exact id. The map shows the validity the recipe certifies per target unit:
// one color for exactly matched units, others for absent and repeated ids.

import {addPopulationSupportAlignmentRecipe} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {createShuffledOrder} from './lib/tracts';
import {createCensusTracts, getCensusPopulation, getCensusTractCode} from './lib/census-tracts';
import {getQuadFill, uploadQuadZones} from './lib/quads';
import {findZoneAt, getZoneRing} from './lib/zone-lookup';
import {addReadout} from './lib/readout';
import {parameters} from './population-support-alignment-recipe.parameters';

const VALIDITY = ['matched', 'absent from source', 'repeated id'];
/** Matched units stay quiet; the two failures are warm. */
const VALIDITY_COLORS = [
  [0.45, 0.55, 0.68],
  [0.86, 0.2, 0.2],
  [0.98, 0.65, 0.15]
] as const;

export default defineOperationRun({
  parameters,
  caption: 'Exact-id join of populations to target units: matched, absent and repeated ids.',
  createInputs: ({random, world}) => {
    const zones = createCensusTracts(world);
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
        (_, index) => getCensusPopulation(sourceOrder[Math.floor(index / 2)])[index % 2]
      )
    );
    const sourceIds = builder.upload(
      'source-ids',
      'uint32',
      Uint32Array.from(sourceOrder, getCensusTractCode)
    );
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
        const ids = Uint32Array.from({length: zones.count}, (_, unit) => getCensusTractCode(unit));
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
  // The pointer picks a target unit: its id, and whether the join matched, missed or repeated it.
  interaction: {
    kind: 'probe',
    hint: 'hover a unit',
    rest: ({toUnit, inputs}) => toUnit([inputs.zones.centers[0], inputs.zones.centers[1]]),
    resolve(pointer, {inputs, outputs, peek}) {
      const unit = findZoneAt(inputs.zones, pointer.x, pointer.y);
      const validity = peek(outputs.validity);
      return {
        glyphs: [{shape: 'polyline', points: getZoneRing(inputs.zones, unit), closed: true}],
        readout: `tract ${getCensusTractCode(unit)} · ${validity ? VALIDITY[validity[unit]] : '...'}`
      };
    }
  },
  draw: ({inputs, outputs}) => [
    getQuadFill(outputs.mesh, inputs.zones, outputs.validity, {categories: VALIDITY_COLORS}),
    {
      layer: 'lines',
      context: true,
      segments: outputs.mesh.outline,
      count: inputs.zones.outlineCount
    }
  ]
});

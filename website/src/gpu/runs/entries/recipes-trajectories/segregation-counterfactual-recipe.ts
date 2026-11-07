// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// addSegregationCounterfactualRecipe: builds the composition-only and spatial-only counterfactuals
// between two periods by quantile mapping, then decomposes the change. The bars are dissimilarity
// in baseline, composition-only, spatial-only and final; units are filled by the final focal share.

import {addSegregationCounterfactualRecipe} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {getPopulationScenarios} from './lib/populations';
import {createQuadZones, getQuadFill, uploadQuadZones} from './lib/quads';
import {addReadout} from './lib/readout';
import {parameters} from './segregation-counterfactual-recipe.parameters';

const METHODS = ['composition', 'dual-composition', 'share'] as const;

export default defineOperationRun({
  parameters,
  caption: 'Dissimilarity of the baseline, two counterfactuals and the final period.',
  createInputs: ({random, world}) => {
    const zones = createQuadZones(random, world, 8, 5);
    return {zones, draws: Float32Array.from({length: zones.count}, random)};
  },
  build(builder, {zones, draws}, values) {
    const baseline = builder.allocate('baseline', 'float32x2', zones.count);
    const final = builder.allocate('final', 'float32x2', zones.count);
    const share = builder.allocate('final-share', 'float32', zones.count);
    const recipe = addSegregationCounterfactualRecipe(builder.graph, {
      baseline: baseline.view,
      final: final.view,
      supportId: 'zones',
      method: METHODS[values.method]
    });
    return {
      outputs: {
        indices: addReadout(builder, 'indices', recipe.outputs.indices),
        share,
        mesh: uploadQuadZones(builder, zones)
      },
      update: current => {
        const scenarios = getPopulationScenarios(zones, draws, current.sorting, current.growth);
        baseline.write(scenarios.scenarios[0]);
        final.write(scenarios.scenarios[3]);
        share.write(scenarios.finalShare);
      }
    };
  },
  rebuildOn: ['method'],
  draw: ({inputs, outputs}) => [
    getQuadFill(outputs.mesh, inputs.zones, outputs.share, {
      ramp: 'sequential',
      range: [0, 1],
      alpha: 0.55
    }),
    {
      layer: 'lines',
      context: true,
      segments: outputs.mesh.outline,
      count: inputs.zones.outlineCount
    },
    {
      layer: 'chart',
      kind: 'bars',
      values: outputs.indices,
      ramp: 'sequential',
      range: [0, 0.5],
      rectangle: [0.06, 0.06, 0.5, 0.42]
    }
  ]
});

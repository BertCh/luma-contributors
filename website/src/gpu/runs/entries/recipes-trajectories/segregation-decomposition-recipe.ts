// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// addSegregationDecompositionRecipe: dissimilarity in four scenarios (baseline, composition-only,
// spatial-only, final) as bars, and the null distribution of the change under paired label
// exchange as a line. Units are filled by their final focal share.

import {addSegregationDecompositionRecipe} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {getPopulationScenarios} from './lib/populations';
import {createQuadZones, getQuadFill, uploadQuadZones} from './lib/quads';
import {addReadout} from './lib/readout';
import {parameters} from './segregation-decomposition-recipe.parameters';

export default defineOperationRun({
  parameters,
  caption: 'Four-scenario dissimilarity and the randomization null distribution.',
  createInputs: ({random, world}) => {
    const zones = createQuadZones(random, world, 8, 5);
    return {zones, draws: Float32Array.from({length: zones.count}, random)};
  },
  build(builder, {zones, draws}, values) {
    const populations = [0, 1, 2, 3].map(scenario =>
      builder.allocate(`population-${scenario}`, 'float32x2', zones.count)
    ) as [
      ReturnType<typeof builder.allocate<'float32x2'>>,
      ReturnType<typeof builder.allocate<'float32x2'>>,
      ReturnType<typeof builder.allocate<'float32x2'>>,
      ReturnType<typeof builder.allocate<'float32x2'>>
    ];
    const share = builder.allocate('final-share', 'float32', zones.count);
    const recipe = addSegregationDecompositionRecipe(builder.graph, {
      populations: [
        populations[0].view,
        populations[1].view,
        populations[2].view,
        populations[3].view
      ],
      supportId: 'zones',
      counterfactualMethod: 'supplied',
      randomization: {replicates: values.replicates, seed: 7}
    });
    return {
      outputs: {
        indices: addReadout(builder, 'indices', recipe.outputs.indices),
        distribution: addReadout(builder, 'distribution', recipe.randomization!.distribution),
        share,
        mesh: uploadQuadZones(builder, zones)
      },
      update: current => {
        const scenarios = getPopulationScenarios(zones, draws, current.sorting, current.growth);
        for (const [index, buffer] of populations.entries())
          buffer.write(scenarios.scenarios[index]);
        share.write(scenarios.finalShare);
      }
    };
  },
  rebuildOn: ['replicates'],
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
      rectangle: [0.04, 0.06, 0.34, 0.36]
    },
    {layer: 'chart', kind: 'line', values: outputs.distribution, rectangle: [0.4, 0.06, 0.96, 0.36]}
  ]
});

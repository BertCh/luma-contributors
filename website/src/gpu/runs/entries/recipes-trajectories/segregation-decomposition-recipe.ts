// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// addSegregationDecompositionRecipe: dissimilarity in four scenarios (baseline, composition-only,
// spatial-only, final) as bars, and the null distribution of the change under paired label
// exchange as a histogram with the observed change marked. Units are filled by their final focal share.

import {addSegregationDecompositionRecipe} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {getChartPosition, getChartWorldX} from '../../pick';
import {getPopulationScenarios} from './lib/populations';
import {createCensusTracts} from './lib/census-tracts';
import {getQuadFill, uploadQuadZones} from './lib/quads';
import {addReadout} from './lib/readout';
import {parameters} from './segregation-decomposition-recipe.parameters';

const BIN_COUNT = 16;
const HISTOGRAM_RECTANGLE = [0.4, 0.06, 0.96, 0.36] as const;

export default defineOperationRun({
  parameters,
  caption: 'Four-scenario dissimilarity and the randomization null distribution.',
  createInputs: ({world}) => {
    const zones = createCensusTracts(world);
    return {zones};
  },
  build(builder, {zones}, values) {
    const histogram = builder.allocate('histogram', 'float32', BIN_COUNT);
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
        histogram,
        share,
        mesh: uploadQuadZones(builder, zones)
      },
      update: current => {
        const scenarios = getPopulationScenarios(zones, current.sorting, current.growth);
        for (const [index, buffer] of populations.entries())
          buffer.write(scenarios.scenarios[index]);
        share.write(scenarios.finalShare);
      }
    };
  },
  rebuildOn: ['replicates'],
  // The pointer's x on the histogram marks a change value: the share of replicates beyond it is
  // the pseudo p-value of that change; the observed change stays fixed.
  interaction: {
    kind: 'probe',
    hint: 'mark a change',
    rest: [0.8, 0.5],
    resolve(pointer, {outputs, world, peek}) {
      const distribution = peek(outputs.distribution);
      const indices = peek(outputs.indices);
      if (!distribution || !indices) return {};
      const observed = indices[3] - indices[0];
      let lower = Math.min(observed, ...distribution);
      let upper = Math.max(observed, ...distribution);
      const margin = (upper - lower) * 0.08 || 0.01;
      lower -= margin;
      upper += margin;
      const counts = new Float32Array(BIN_COUNT);
      for (const value of distribution) {
        counts[
          Math.min(BIN_COUNT - 1, Math.floor(((value - lower) / (upper - lower)) * BIN_COUNT))
        ]++;
      }
      outputs.histogram.write(counts);
      const {u} = getChartPosition(HISTOGRAM_RECTANGLE, world, pointer.x, pointer.y);
      const marked = lower + u * (upper - lower);
      const beyond = distribution.filter(value => value >= marked).length / distribution.length;
      const top = HISTOGRAM_RECTANGLE[3] * world.height;
      const bottom = HISTOGRAM_RECTANGLE[1] * world.height;
      const observedX = getChartWorldX(
        HISTOGRAM_RECTANGLE,
        world,
        (observed - lower) / (upper - lower)
      );
      return {
        glyphs: [
          {
            shape: 'line',
            from: [getChartWorldX(HISTOGRAM_RECTANGLE, world, u), bottom],
            to: [getChartWorldX(HISTOGRAM_RECTANGLE, world, u), top],
            dashed: true
          },
          {shape: 'line', from: [observedX, bottom], to: [observedX, top]}
        ],
        readout: `observed dD ${observed.toFixed(2)} · marked ${marked.toFixed(2)}: ${(beyond * 100).toFixed(0)}% of ${distribution.length} beyond`
      };
    }
  },
  draw: ({inputs, outputs, values}) => [
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
    {
      layer: 'chart',
      kind: 'bars',
      values: outputs.histogram,
      ramp: 'sequential',
      range: [0, values.replicates / 3],
      rectangle: HISTOGRAM_RECTANGLE
    }
  ]
});

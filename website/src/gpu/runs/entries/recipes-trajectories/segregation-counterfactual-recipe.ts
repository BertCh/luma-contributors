// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// addSegregationCounterfactualRecipe: builds the composition-only and spatial-only counterfactuals
// between two periods by quantile mapping, then decomposes the change. The bars are dissimilarity
// in baseline, composition-only, spatial-only and final; units are filled by the focal share of the scenario under the pointer.

import {addSegregationCounterfactualRecipe} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {clamp} from '../../interaction';
import {createChartAxes, createLegend, getChartRectangle, mergeText} from '../../text';
import {getPopulationScenarios} from './lib/populations';
import {createCensusTracts} from './lib/census-tracts';
import {getQuadFill, uploadQuadZones} from './lib/quads';
import {addReadout} from './lib/readout';
import {parameters} from './segregation-counterfactual-recipe.parameters';

const SCENARIOS = ['baseline', 'composition-only', 'spatial-only', 'final'];
const BAR_NAMES = ['baseline', 'composition', 'spatial', 'final'];
const BAR_RECTANGLE = [0.06, 0.06, 0.5, 0.42] as const;
const BAR_RANGE = 0.5;

/** World y of a dissimilarity value on the bar chart (the bars span 0 to BAR_RANGE). */
function getBarY(value: number): number {
  return (
    BAR_RECTANGLE[1] +
    (Math.min(value, BAR_RANGE) / BAR_RANGE) * (BAR_RECTANGLE[3] - BAR_RECTANGLE[1])
  );
}

const METHODS = ['composition', 'dual-composition', 'share'] as const;

export default defineOperationRun({
  parameters,
  caption: 'Dissimilarity of the baseline, two counterfactuals and the final period.',
  createInputs: ({world}) => {
    const zones = createCensusTracts(world);
    return {zones, view: {scenario: 3}, shares: [] as Float32Array[]};
  },
  build(builder, inputs, values) {
    const {zones} = inputs;
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
        const scenarios = getPopulationScenarios(zones, current.sorting, current.growth);
        baseline.write(scenarios.scenarios[0]);
        final.write(scenarios.scenarios[3]);
        // Focal share of each scenario, for the map the bars select.
        inputs.shares = scenarios.scenarios.map(population =>
          Float32Array.from(
            {length: zones.count},
            (_, unit) => population[unit * 2] / (population[unit * 2] + population[unit * 2 + 1])
          )
        );
        share.write(inputs.shares[inputs.view.scenario]);
      }
    };
  },
  rebuildOn: ['method'],
  // The pointer's x picks a bar: the map is filled with that scenario's focal share.
  interaction: {
    kind: 'probe',
    hint: 'hover a scenario',
    rest: [0.875, 0.5],
    resolve(pointer, {inputs, outputs, world, peek}) {
      const scenario = clamp(Math.floor(pointer.unitX * 4), 0, 3);
      const changed = inputs.view.scenario !== scenario;
      inputs.view.scenario = scenario;
      const indices = peek(outputs.indices);
      const barWidth = (BAR_RECTANGLE[2] - BAR_RECTANGLE[0]) / 4;
      const barLeft = (BAR_RECTANGLE[0] + scenario * barWidth) * world.width;
      const barRight = barLeft + barWidth * world.width;
      return {
        rerun: changed,
        glyphs: [
          {
            shape: 'box',
            minimum: [barLeft, BAR_RECTANGLE[1] * world.height],
            maximum: [barRight, BAR_RECTANGLE[3] * world.height]
          },
          ...(indices
            ? [
                {
                  shape: 'line' as const,
                  from: [
                    BAR_RECTANGLE[0] * world.width,
                    getBarY(indices[0]) * world.height
                  ] as const,
                  to: [BAR_RECTANGLE[2] * world.width, getBarY(indices[0]) * world.height] as const,
                  dashed: true
                }
              ]
            : [])
        ],
        readout: indices
          ? `${SCENARIOS[scenario]} · D ${indices[scenario].toFixed(2)} (baseline ${indices[0].toFixed(2)})`
          : SCENARIOS[scenario]
      };
    }
  },
  async afterRun({outputs}) {
    const bytes = await outputs.indices.buffer.readAsync();
    return new Float32Array(bytes.slice().buffer);
  },
  draw: ({inputs, outputs, world, extra}) => {
    const rectangle = getChartRectangle(world, BAR_RECTANGLE);
    const axes = createChartAxes({
      rectangle,
      x: {categories: BAR_NAMES},
      y: {range: [0, BAR_RANGE], ticks: [0, 0.25, 0.5], title: 'dissimilarity D'}
    });
    return [
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
      {layer: 'text', ...axes.grid},
      {
        layer: 'chart',
        kind: 'bars',
        values: outputs.indices,
        ramp: 'sequential',
        range: [0, 0.5],
        rectangle: BAR_RECTANGLE
      },
      {
        layer: 'text',
        ...mergeText(
          axes,
          // The value sits at the foot of each bar, clear of the dashed baseline rule.
          {
            labels: Array.from(extra ?? [], (value, bar) => ({
              text: value.toFixed(2),
              at: [
                rectangle[0] + ((bar + 0.5) / BAR_NAMES.length) * (rectangle[2] - rectangle[0]),
                rectangle[1]
              ] as const,
              anchor: 'bottom' as const,
              offset: [0, 5] as const,
              role: 'value' as const,
              priority: 5
            }))
          },
          createLegend({
            corner: 'bottom-right',
            entries: [
              {
                label: `map: focal share, ${SCENARIOS[inputs.view.scenario]}`,
                swatch: [0.8, 0.45, 0.3]
              }
            ]
          })
        )
      }
    ];
  }
});

// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUMultipleTestingAdjustment: each p-value (x) against its adjusted value (y). The diagonal is
// "no adjustment". Bonferroni and Holm lift every value; the Benjamini procedures lift the
// large p-values most and keep the strong signals (bottom left) significant. All five methods run
// in one graph so the pointer can switch between them: x picks the method and y sets the
// significance level (the horizontal line); tests whose adjusted p-value falls under it are orange.

import {
  GPUMultipleTestingAdjustment,
  GPULocalPermutationTest,
  getGPUPermutationParameterValues
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun, type RunLayer} from '../../define-run';
import {createTracts, uploadSpatialWeights} from '../../inputs';
import {createChartAxes, getChartRectangle, mergeText} from '../../text';
import {findParameter, getFractionOf, getParameterAt} from '../../interaction';
import {whenShown} from './shared/point-field';
import {parameters} from './multiple-testing-adjustment.parameters';

const METHODS = [
  'none',
  'bonferroni',
  'holm',
  'benjamini-hochberg',
  'benjamini-yekutieli'
] as const;
const METHOD_PARAMETER = findParameter(parameters, 'method');
const ALPHA_PARAMETER = findParameter(parameters, 'alpha');
// 999 shuffles give p-values down to 0.001: with 199 (smallest p 0.005) no strong cluster of
// tracts survives a correction over 120 tests, and every method would read 0 significant.
const PERMUTATION_COUNT = 999;
const PLOT = [0.1, 0.1, 0.9, 0.92] as const;

export default defineOperationRun({
  parameters,
  caption: 'Raw and corrected local Moran permutation p-values for Chicago tract poverty shares.',
  createInputs: ({random, world}) => ({
    random,
    world,
    // The diagonal of the plot: adjusted equals raw.
    diagonal: Float32Array.of(
      PLOT[0] * world.width,
      PLOT[1] * world.height,
      PLOT[2] * world.width,
      PLOT[3] * world.height
    )
  }),
  build(builder, inputs, values) {
    const tracts = createTracts(inputs.random, inputs.world, Math.sqrt(values.testCount));
    const testCount = tracts.count;
    const pValues = builder.allocate('p-values', 'float32', testCount);
    builder.graph.add(
      new GPULocalPermutationTest({
        statistic: 'localMoran',
        alternative: 'two-sided',
        weights: uploadSpatialWeights(builder, tracts.contiguity),
        values: builder.upload('poverty-shares', 'float32', tracts.povertyShare).view,
        parameters: builder.upload(
          'permutation-parameters',
          'uint32',
          getGPUPermutationParameterValues(
            {
              seed: 11,
              permutationCount: PERMUTATION_COUNT
            },
            {maximumPermutations: PERMUTATION_COUNT}
          )
        ).view,
        maximumPermutations: PERMUTATION_COUNT,
        output: {
          exceedances: builder.allocate('exceedances', 'uint32', testCount).view,
          pseudoPValues: pValues.view,
          significant: builder.allocate('significant', 'uint32', testCount).view,
          status: {overflow: builder.allocate('permutation-overflow', 'uint32', 1).view}
        }
      })
    );
    // Every method is built, so switching method is a per-frame choice of which output to draw.
    const adjustedPValues = METHODS.map(method => {
      const adjusted = builder.allocate(`adjusted-${method}`, 'float32', testCount);
      builder.graph.add(
        new GPUMultipleTestingAdjustment({
          id: `adjust-${method}`,
          pValues: pValues.view,
          method,
          output: {
            adjustedPValues: adjusted.view,
            testedCount: builder.allocate(`tested-${method}`, 'uint32', 1).view
          }
        })
      );
      return adjusted;
    });
    return {
      outputs: {
        pValues,
        adjustedPValues,
        // Render-only: plot positions of the tests below the significance level.
        significant: builder.allocate('significant-points', 'float32x2', testCount),
        diagonal: builder.upload('diagonal', 'float32x4', inputs.diagonal)
      },
      update: () => {}
    };
  },
  rebuildOn: ['testCount'],
  interaction: {
    kind: 'level',
    hint: 'method and level',
    rest: [
      getFractionOf(METHOD_PARAMETER, METHOD_PARAMETER.default),
      1 - getFractionOf(ALPHA_PARAMETER, ALPHA_PARAMETER.default)
    ],
    resolve(pointer, {outputs, world, peek}) {
      const method = getParameterAt(METHOD_PARAMETER, pointer.unitX);
      const alpha = getParameterAt(ALPHA_PARAMETER, 1 - pointer.unitY);
      const values = {method, alpha};
      const raw = peek(outputs.pValues);
      const adjusted = peek(outputs.adjustedPValues[values.method]);
      const lineY = (PLOT[1] + values.alpha * (PLOT[3] - PLOT[1])) * world.height;
      const glyphs = [
        {
          shape: 'line',
          from: [PLOT[0] * world.width, lineY],
          to: [PLOT[2] * world.width, lineY],
          dashed: true
        }
      ] as const;
      if (!raw || !adjusted) return {values, glyphs};
      const plotted = new Float32Array(raw.length * 2);
      let rawCount = 0;
      let adjustedCount = 0;
      for (let test = 0; test < raw.length; test++) {
        if (raw[test] <= values.alpha) rawCount++;
        if (!(adjusted[test] <= values.alpha)) continue;
        plotted[adjustedCount * 2] = (PLOT[0] + raw[test] * (PLOT[2] - PLOT[0])) * world.width;
        plotted[adjustedCount * 2 + 1] =
          (PLOT[1] + Math.min(1, adjusted[test]) * (PLOT[3] - PLOT[1])) * world.height;
        adjustedCount++;
      }
      outputs.significant.write(plotted);
      return {
        values,
        glyphs,
        focus: adjustedCount,
        readout: `${METHODS[values.method]} · alpha ${values.alpha.toFixed(2)} · ${rawCount} raw -> ${adjustedCount} significant`
      };
    }
  },
  draw: ({outputs, values, interaction, theme, world}): readonly RunLayer[] => {
    const rectangle = getChartRectangle(world, PLOT);
    const axes = createChartAxes({
      rectangle,
      x: {range: [0, 1], ticks: [0, 0.5, 1], title: 'raw p'},
      y: {range: [0, 1], ticks: [0, 0.5, 1], title: 'adjusted p', grid: false}
    });
    // The permutation p-value cannot fall under 1/(shuffles + 1); Bonferroni multiplies by the tests.
    const floor = 1 / (PERMUTATION_COUNT + 1);
    const isCapped = values.method === 1 || values.method === 2;
    const smallestAdjusted = Math.min(1, floor * outputs.pValues.length);
    const shownCount = typeof interaction.focus === 'number' ? interaction.focus : 0;
    return [
      {layer: 'lines', context: true, segments: outputs.diagonal, radius: 1},
      {
        layer: 'chart',
        kind: 'scatter',
        values: outputs.adjustedPValues[values.method],
        xValues: outputs.pValues,
        xRange: [0, 1],
        range: [0, 1],
        color: [0.4, 0.68, 1, 0.9],
        radius: 7.5,
        rectangle: PLOT
      },
      ...whenShown(shownCount, {
        layer: 'dots',
        positions: outputs.significant,
        count: shownCount,
        color: [...theme.categories[1], 1],
        radius: 9
      }),
      {
        layer: 'text',
        ...mergeText(axes, {
          labels: [
            {
              text: `alpha ${values.alpha.toFixed(2)}`,
              at: [
                rectangle[0] + 0.5 * (rectangle[2] - rectangle[0]),
                rectangle[1] + values.alpha * (rectangle[3] - rectangle[1])
              ],
              anchor: 'bottom',
              offset: [0, 2],
              role: 'label',
              priority: 4
            },
            {
              text: 'significant after adjustment',
              at: [rectangle[2], rectangle[1]],
              anchor: 'bottom-right',
              offset: [0, 22],
              role: 'label',
              priority: 5,
              swatch: 1
            },
            ...(isCapped
              ? [
                  {
                    text: `smallest possible adjusted p: ${smallestAdjusted.toFixed(2)} (${PERMUTATION_COUNT} shuffles)`,
                    at: [rectangle[2], rectangle[1]] as const,
                    anchor: 'bottom-right' as const,
                    offset: [0, 38] as const,
                    role: 'label' as const,
                    priority: 5
                  }
                ]
              : [])
          ]
        })
      }
    ];
  }
});

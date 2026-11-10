// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUCompositeScore over four indicators per tract. Tracts are filled by the composite score; the
// dots show the first indicator alone.

import {
  GPUCompositeScore,
  GPU_COMPOSITE_SCORE_PARAMETER_LENGTH,
  getGPUCompositeScoreParameterValues
} from '@luma-contributors/analysis/dataframe-analysis';
import {defineOperationRun, type RunBuffer} from '../../define-run';
import {clamp, findParameter, formatNumber, snapToParameter} from '../../interaction';
import {createTracts} from '../../inputs';
import {uploadTractGeometry} from './_lib/helpers';
import {parameters} from './composite-score.parameters';

const SCALERS = ['min-max', 'z-score', 'rank'] as const;
const INDICATORS = ['poverty', 'unemployment', 'age 65+', 'population'] as const;
const AGGREGATIONS = ['weighted-sum', 'weighted-geometric-mean', 'principal-component'] as const;

type Inputs = ReturnType<typeof createInputs>;
type Outputs = ReturnType<typeof uploadTractGeometry> & {
  score: RunBuffer;
  centers: RunBuffer;
  indicator: RunBuffer;
};

/** Fixed color range per scaler and aggregation, so a weight change moves colors instead of rescaling. */
function getScoreLook(scaler: number, aggregation: number) {
  const isSigned =
    SCALERS[scaler] === 'z-score' || AGGREGATIONS[aggregation] === 'principal-component';
  return isSigned
    ? ({ramp: 'diverging', range: [-2, 2]} as const)
    : ({ramp: 'sequential', range: [0, 0.7]} as const);
}

function createInputs({
  random,
  world
}: {
  random: () => number;
  world: {width: number; height: number};
}) {
  const tracts = createTracts(random, world, 14, 0.1);
  const indicators = new Float32Array(tracts.count * 4);
  for (let row = 0; row < tracts.count; row++) {
    indicators[row * 4] = tracts.povertyShare[row];
    indicators[row * 4 + 1] = tracts.unemploymentShare[row];
    indicators[row * 4 + 2] = tracts.age65[row] / tracts.population[row];
    indicators[row * 4 + 3] = tracts.population[row];
  }
  return {tracts, indicators, world};
}

export default defineOperationRun<Inputs, Outputs>({
  parameters,
  caption:
    'Composite of poverty, unemployment, older population share and population size; the dots show the weighted indicator.',
  createInputs,
  build(builder, inputs) {
    const {tracts} = inputs;
    const indicators = builder.upload('indicators', 'float32', inputs.indicators);
    const parameterBuffer = builder.allocate(
      'parameters',
      'float32',
      GPU_COMPOSITE_SCORE_PARAMETER_LENGTH
    );
    const score = builder.allocate('score', 'float32', tracts.count);
    builder.graph.add(
      new GPUCompositeScore({
        indicators: indicators.view,
        indicatorCount: 4,
        enableRank: true,
        enablePrincipalComponent: true,
        parameters: parameterBuffer.view,
        output: {score: score.view}
      })
    );
    const indicator = builder.allocate('indicator', 'float32', tracts.count);
    return {
      outputs: {
        ...uploadTractGeometry(builder, tracts),
        score,
        centers: builder.upload('centers', 'float32x2', tracts.centers),
        indicator
      },
      update: current => {
        const weights = [1, 1, 1, 1];
        weights[current.indicator] = current.weight;
        parameterBuffer.write(
          getGPUCompositeScoreParameterValues({
            scaler: SCALERS[current.scaler],
            aggregation: AGGREGATIONS[current.aggregation],
            weights,
            directions: [1, 1, 1, 1]
          })
        );
        // The emphasized indicator, min-max scaled, as the dots over each tract.
        const column = Float32Array.from(
          {length: tracts.count},
          (_, row) => inputs.indicators[row * 4 + current.indicator]
        );
        const minimum = column.reduce((lowest, value) => Math.min(lowest, value), Infinity);
        const maximum = column.reduce((highest, value) => Math.max(highest, value), -Infinity);
        indicator.write(column.map(value => (value - minimum) / (maximum - minimum || 1)));
      }
    };
  },
  rebuildOn: [],
  // The pointer aims at one of four indicators around the map center (north poverty, east
  // unemployment, south age 65+, west population); its distance is that indicator's weight.
  interaction: {
    kind: 'aim',
    hint: 'aim at an indicator',
    rest: ({world}) => {
      const reach = world.height * 0.45;
      const weight = findParameter(parameters, 'weight');
      const fraction = (weight.default - weight.min) / (weight.max - weight.min);
      return [0.5 + (reach * fraction) / world.width, 0.5];
    },
    resolve(pointer, {world, values}) {
      const reach = world.height * 0.45;
      const anchor = [world.width / 2, world.height / 2] as const;
      const deltaX = pointer.x - anchor[0];
      const deltaY = pointer.y - anchor[1];
      const compass = (90 - (Math.atan2(deltaY, deltaX) * 180) / Math.PI + 360) % 360;
      const indicatorParameter = findParameter(parameters, 'indicator');
      const weightParameter = findParameter(parameters, 'weight');
      const patch: Record<string, number> = {
        weight: snapToParameter(
          weightParameter,
          weightParameter.min +
            clamp(Math.hypot(deltaX, deltaY) / reach, 0, 1) *
              (weightParameter.max - weightParameter.min)
        )
      };
      if (Math.hypot(deltaX, deltaY) > reach * 0.06) {
        patch.indicator = snapToParameter(
          indicatorParameter,
          Math.floor(((compass + 45) % 360) / 90)
        );
      }
      const merged = {...values, ...patch};
      const arm = reach * 0.3;
      return {
        values: patch,
        glyphs: [
          {
            shape: 'line',
            from: [anchor[0], anchor[1] + arm],
            to: [anchor[0], anchor[1] + reach * 0.9],
            dashed: true
          },
          {
            shape: 'line',
            from: [anchor[0] + arm, anchor[1]],
            to: [anchor[0] + reach * 0.9, anchor[1]],
            dashed: true
          },
          {
            shape: 'line',
            from: [anchor[0], anchor[1] - arm],
            to: [anchor[0], anchor[1] - reach * 0.9],
            dashed: true
          },
          {
            shape: 'line',
            from: [anchor[0] - arm, anchor[1]],
            to: [anchor[0] - reach * 0.9, anchor[1]],
            dashed: true
          },
          {shape: 'arrow', from: anchor, to: [pointer.x, pointer.y]},
          {shape: 'dot', at: anchor}
        ],
        readout: `weights[${INDICATORS[merged.indicator]}] ${formatNumber(merged.weight)} · others 1`
      };
    }
  },
  draw: ({inputs, outputs, values}) => [
    {
      layer: 'polygons',
      positions: outputs.vertices,
      indices: outputs.indices,
      owners: outputs.owners,
      values: outputs.score,
      triangleCount: inputs.tracts.triangleCount,
      ...getScoreLook(values.scaler, values.aggregation)
    },
    {layer: 'lines', context: true, segments: outputs.outline, count: inputs.tracts.outlineCount},
    {
      layer: 'dots',
      context: true,
      positions: outputs.centers,
      values: outputs.indicator,
      range: [0, 1],
      radius: 3
    }
  ]
});

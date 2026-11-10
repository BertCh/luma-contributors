// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPURipleyEnvelope next to the observed GPURipley. The band between the two thin lines is the
// range of L(r) - r that random (CSR) patterns of the same size reach; the bright curve is the
// observed pattern. Where it leaves the band the pattern is clustered or regular at that radius.

import {
  GPURipley,
  GPURipleyEnvelope,
  GPU_RIPLEY_PARAMETER_LENGTH,
  getGPURipleyParameterValues
} from '@luma-contributors/analysis/dataframe-analysis';
import {defineOperationRun} from '../../define-run';
import {
  DISTANCE_PLOT,
  RING_HIGHLIGHT_LIMIT,
  createRuleSegments,
  drawRingProbe,
  getAxisFraction,
  getAxisPointer,
  getBinBox,
  getBinIndex,
  getHighlightCount,
  getChartWorldY,
  pickSpreadRows
} from './_lib/distance-probe';
import {createPatternSource, getPatternPositions} from './_lib/helpers';
import {parameters} from './ripley-envelope.parameters';

const POINT_COUNT = 400;
const RADIUS_COUNT = 24;
const PLOT = DISTANCE_PLOT;
const RANGE = [-0.03, 0.1] as const;
const MAXIMUM_DISTANCE = 0.2;
const ANCHOR_ROWS = pickSpreadRows(POINT_COUNT, 6);
const REST_FRACTION = 0.3;

export default defineOperationRun({
  parameters,
  caption:
    'Observed L(r) - r against the band (grey) that random patterns reach; the white rule is zero.',
  createInputs: ({random, world}) => ({
    source: createPatternSource(random, POINT_COUNT),
    windowWidth: world.width * 0.5,
    world
  }),
  build(builder, inputs, values) {
    const positions = builder.allocate('positions', 'float32x2', POINT_COUNT);
    const parameterBuffer = builder.allocate('parameters', 'float32', GPU_RIPLEY_PARAMETER_LENGTH);
    const k = builder.allocate('k', 'float32', RADIUS_COUNT);
    const observed = builder.allocate('observed', 'float32', RADIUS_COUNT);
    builder.graph.add(
      new GPURipley({
        positions: positions.view,
        parameters: parameterBuffer.view,
        gridSize: [32, 32],
        radiusCount: RADIUS_COUNT,
        output: {k: k.view, lMinusR: observed.view}
      })
    );
    const lower = builder.allocate('lower', 'float32', RADIUS_COUNT);
    const upper = builder.allocate('upper', 'float32', RADIUS_COUNT);
    builder.graph.add(
      new GPURipleyEnvelope({
        positions: positions.view,
        parameters: parameterBuffer.view,
        gridSize: [32, 32],
        radiusCount: RADIUS_COUNT,
        simulationCount: values.simulationCount,
        rank: values.rank,
        statistic: 'l-minus-r',
        observedK: k.view,
        output: {lower: lower.view, upper: upper.view}
      })
    );
    const highlight = builder.allocate('highlight', 'float32x2', RING_HIGHLIGHT_LIMIT);
    const zero = builder.upload(
      'zero',
      'float32x4',
      createRuleSegments(PLOT, inputs.world, RANGE, [0])
    );
    return {
      outputs: {positions, observed, lower, upper, highlight, zero},
      update: current => {
        positions.write(
          getPatternPositions(inputs.source, inputs.windowWidth, 1, current.clustering)
        );
        parameterBuffer.write(
          getGPURipleyParameterValues({
            bounds: [0, 0, inputs.windowWidth, 1],
            maximumDistance: MAXIMUM_DISTANCE,
            edgeCorrection: 'isotropic'
          })
        );
      }
    };
  },
  rebuildOn: ['simulationCount', 'rank'],
  // The pointer is a radius on the chart's distance axis; the same radius is drawn around a few
  // events, and the readout says whether the observed curve is outside the random band there.
  interaction: {
    kind: 'reach',
    hint: 'hover a radius',
    rest: [getAxisPointer(REST_FRACTION), 0.5],
    resolve(pointer, {inputs, outputs, values, peek}) {
      const bin = getBinIndex(getAxisFraction(pointer.unitX), RADIUS_COUNT);
      const radius = (MAXIMUM_DISTANCE * (bin + 1)) / RADIUS_COUNT;
      const positions = getPatternPositions(
        inputs.source,
        inputs.windowWidth,
        1,
        values.clustering
      );
      const {glyphs, found} = drawRingProbe(positions, ANCHOR_ROWS, radius, outputs.highlight);
      const observed = peek(outputs.observed)?.[bin];
      const lower = peek(outputs.lower)?.[bin];
      const upper = peek(outputs.upper)?.[bin];
      const isReady = observed !== undefined && lower !== undefined && upper !== undefined;
      const verdict = !isReady
        ? ''
        : observed > upper
          ? ' · clustered'
          : observed < lower
            ? ' · regular'
            : ' · random';
      return {
        spotlight: null,
        focus: found,
        glyphs: [
          getBinBox(PLOT, inputs.world, bin, RADIUS_COUNT),
          ...glyphs,
          ...(observed === undefined
            ? []
            : [
                {
                  shape: 'dot' as const,
                  at: [
                    (PLOT[0] + ((PLOT[2] - PLOT[0]) * (bin + 0.5)) / RADIUS_COUNT) *
                      inputs.world.width,
                    getChartWorldY(PLOT, inputs.world, RANGE, observed)
                  ] as const
                }
              ])
        ],
        readout: isReady
          ? `r ${radius.toFixed(2)} · L-r ${observed.toFixed(3)} in [${lower.toFixed(3)}, ${upper.toFixed(3)}]${verdict}`
          : `r ${radius.toFixed(2)}`
      };
    }
  },
  draw: ({outputs, interaction, theme}) => [
    {
      layer: 'dots',
      context: true,
      positions: outputs.positions,
      color: [0.75, 0.8, 0.9, 0.8],
      radius: 2.2
    },
    {
      layer: 'dots',
      positions: outputs.highlight,
      count: getHighlightCount(interaction.focus),
      radius: 3.4,
      color: [1, 0.75, 0.2, 1]
    },
    {
      layer: 'chart',
      kind: 'bars',
      values: outputs.upper,
      range: RANGE,
      rectangle: PLOT,
      color: [0.55, 0.6, 0.7, 0.45]
    },
    {
      layer: 'chart',
      kind: 'bars',
      values: outputs.lower,
      range: RANGE,
      rectangle: PLOT,
      color: [theme.background[0], theme.background[1], theme.background[2], 1]
    },
    {layer: 'lines', context: true, segments: outputs.zero, count: 1, color: [1, 1, 1, 0.7]},
    {
      layer: 'chart',
      kind: 'line',
      values: outputs.observed,
      range: RANGE,
      rectangle: PLOT,
      radius: 2.6,
      color: [0.3, 0.75, 1, 1]
    }
  ]
});

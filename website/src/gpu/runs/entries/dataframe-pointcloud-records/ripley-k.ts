// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPURipley on a point pattern in the left panel. The curve is L(r) - r written by the GPU: it
// stays near zero for a random pattern and rises above it where points cluster at that radius.

import {
  GPURipley,
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
  pickSpreadRows
} from './_lib/distance-probe';
import {createPatternSource, getPatternPositions} from './_lib/helpers';
import {parameters} from './ripley-k.parameters';

const POINT_COUNT = 500;
const PLOT = DISTANCE_PLOT;
const RANGE = [-0.03, 0.12] as const;
const ANCHOR_ROWS = pickSpreadRows(POINT_COUNT, 6);
const REST_FRACTION = 0.3;
const EDGE_CORRECTIONS = ['none', 'border', 'isotropic'] as const;

export default defineOperationRun({
  parameters,
  caption:
    'L(r) - r from Ripley’s K against the zero line of a random pattern; the window is the left panel.',
  createInputs: ({random, world}) => ({
    source: createPatternSource(random, POINT_COUNT),
    windowWidth: world.width * 0.5,
    world
  }),
  build(builder, inputs, values) {
    const positions = builder.allocate('positions', 'float32x2', POINT_COUNT);
    const parameterBuffer = builder.allocate('parameters', 'float32', GPU_RIPLEY_PARAMETER_LENGTH);
    const k = builder.allocate('k', 'float32', values.radiusCount);
    const lMinusR = builder.allocate('l-minus-r', 'float32', values.radiusCount);
    builder.graph.add(
      new GPURipley({
        positions: positions.view,
        parameters: parameterBuffer.view,
        gridSize: [32, 32],
        radiusCount: values.radiusCount,
        output: {k: k.view, lMinusR: lMinusR.view}
      })
    );
    const highlight = builder.allocate('highlight', 'float32x2', RING_HIGHLIGHT_LIMIT);
    const zero = builder.upload(
      'zero',
      'float32x4',
      createRuleSegments(PLOT, inputs.world, RANGE, [0])
    );
    const centers = builder.allocate('centers', 'float32', values.radiusCount);
    return {
      outputs: {positions, k, lMinusR, highlight, zero, centers},
      update: current => {
        centers.write(
          Float32Array.from(
            {length: values.radiusCount},
            (_, bin) => ((bin + 0.5) / values.radiusCount) * current.maximumDistance
          )
        );
        positions.write(
          getPatternPositions(inputs.source, inputs.windowWidth, 1, current.clustering)
        );
        parameterBuffer.write(
          getGPURipleyParameterValues({
            bounds: [0, 0, inputs.windowWidth, 1],
            maximumDistance: current.maximumDistance,
            edgeCorrection: EDGE_CORRECTIONS[current.edgeCorrection]
          })
        );
      }
    };
  },
  rebuildOn: ['radiusCount'],
  // The pointer is a radius on the chart's distance axis; the same radius is drawn around a few
  // events, and the events inside those rings are brightened.
  interaction: {
    kind: 'reach',
    hint: 'hover a radius',
    rest: [getAxisPointer(REST_FRACTION), 0.5],
    resolve(pointer, {inputs, outputs, values, peek}) {
      const radiusCount = values.radiusCount;
      const bin = getBinIndex(getAxisFraction(pointer.unitX), radiusCount);
      const radius = (values.maximumDistance * (bin + 1)) / radiusCount;
      const positions = getPatternPositions(
        inputs.source,
        inputs.windowWidth,
        1,
        values.clustering
      );
      const {glyphs, found} = drawRingProbe(positions, ANCHOR_ROWS, radius, outputs.highlight);
      const ripley = peek(outputs.k)?.[bin];
      const difference = peek(outputs.lMinusR)?.[bin];
      return {
        spotlight: null,
        focus: found,
        glyphs: [getBinBox(PLOT, inputs.world, bin, radiusCount), ...glyphs],
        readout: `r ${radius.toFixed(2)} · K ${ripley === undefined ? '…' : ripley.toFixed(3)} · L-r ${
          difference === undefined ? '…' : difference.toFixed(3)
        }`
      };
    }
  },
  draw: ({outputs, interaction, values}) => [
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
    {layer: 'lines', context: true, segments: outputs.zero, count: 1, color: [1, 1, 1, 0.7]},
    {
      layer: 'chart',
      kind: 'line',
      values: outputs.lMinusR,
      range: RANGE,
      rectangle: PLOT,
      radius: 2.6,
      color: [0.3, 0.75, 1, 1]
    },
    {
      layer: 'chart',
      kind: 'scatter',
      values: outputs.lMinusR,
      xValues: outputs.centers,
      xRange: [0, values.maximumDistance],
      range: RANGE,
      rectangle: PLOT,
      radius: 3.4,
      color: [0.3, 0.75, 1, 1]
    }
  ]
});

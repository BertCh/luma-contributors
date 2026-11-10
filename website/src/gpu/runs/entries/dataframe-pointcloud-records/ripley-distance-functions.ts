// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPURipleyDistanceFunctions: G(r), the share of events whose nearest neighbor is within r (cyan),
// and F(r), the share of empty locations within r of an event (orange). Clustered patterns have
// G above F; the J curve (white) falls below 1.

import {
  GPURipleyDistanceFunctions,
  GPU_RIPLEY_DISTANCE_FUNCTIONS_PARAMETER_LENGTH,
  getGPURipleyDistanceFunctionsParameterValues
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
import {parameters} from './ripley-distance-functions.parameters';

const POINT_COUNT = 500;
const PLOT = DISTANCE_PLOT;
const ANCHOR_ROWS = pickSpreadRows(POINT_COUNT, 6);
const REST_FRACTION = 0.3;
const CORRECTIONS = ['none', 'border', 'kaplan-meier', 'hanisch'] as const;

export default defineOperationRun({
  parameters,
  caption:
    'G (cyan), F (orange) and J (white) against distance; the faint rules are the random-pattern G = F curve and J = 1.',
  createInputs: ({random, world}) => ({
    source: createPatternSource(random, POINT_COUNT),
    windowWidth: world.width * 0.5,
    world,
    nearest: {share: -1, distances: new Float32Array(POINT_COUNT)}
  }),
  build(builder, inputs, values) {
    const positions = builder.allocate('positions', 'float32x2', POINT_COUNT);
    const parameterBuffer = builder.allocate(
      'parameters',
      'float32',
      GPU_RIPLEY_DISTANCE_FUNCTIONS_PARAMETER_LENGTH
    );
    const g = builder.allocate('g', 'float32', values.radiusCount);
    const f = builder.allocate('f', 'float32', values.radiusCount);
    const j = builder.allocate('j', 'float32', values.radiusCount);
    builder.graph.add(
      new GPURipleyDistanceFunctions({
        positions: positions.view,
        parameters: parameterBuffer.view,
        gridSize: [32, 32],
        referenceGrid: [32, 32],
        radiusCount: values.radiusCount,
        output: {g: g.view, f: f.view, j: j.view}
      })
    );
    const highlight = builder.allocate('highlight', 'float32x2', RING_HIGHLIGHT_LIMIT);
    const references = builder.allocate('references', 'float32', values.radiusCount);
    const jReference = builder.upload(
      'j-reference',
      'float32x4',
      createRuleSegments(PLOT, inputs.world, [0, 2], [1])
    );
    return {
      outputs: {positions, g, f, j, highlight, references, jReference},
      update: current => {
        // The G and F curves of a random pattern of the same intensity: 1 - exp(-lambda pi r^2).
        const intensity = POINT_COUNT / inputs.windowWidth;
        references.write(
          Float32Array.from({length: values.radiusCount}, (_, bin) => {
            const radius = (current.maximumDistance * (bin + 1)) / values.radiusCount;
            return 1 - Math.exp(-intensity * Math.PI * radius * radius);
          })
        );
        positions.write(
          getPatternPositions(inputs.source, inputs.windowWidth, 1, current.clustering)
        );
        parameterBuffer.write(
          getGPURipleyDistanceFunctionsParameterValues({
            bounds: [0, 0, inputs.windowWidth, 1],
            maximumDistance: current.maximumDistance,
            edgeCorrection: CORRECTIONS[current.edgeCorrection]
          })
        );
      }
    };
  },
  rebuildOn: ['radiusCount'],
  // The pointer is a radius on the chart's distance axis: rings of that radius are drawn around a
  // few events, and the events whose nearest neighbor is inside it (the G curve) are brightened.
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
      if (inputs.nearest.share !== values.clustering) {
        inputs.nearest.share = values.clustering;
        for (let first = 0; first < POINT_COUNT; first++) {
          let best = Infinity;
          for (let second = 0; second < POINT_COUNT; second++) {
            if (second === first) continue;
            best = Math.min(
              best,
              Math.hypot(
                positions[first * 2] - positions[second * 2],
                positions[first * 2 + 1] - positions[second * 2 + 1]
              )
            );
          }
          inputs.nearest.distances[first] = best;
        }
      }
      const ringGlyphs = drawRingProbe(positions, ANCHOR_ROWS, radius, outputs.highlight).glyphs;
      const brightened = new Float32Array(RING_HIGHLIGHT_LIMIT * 2).fill(-10);
      let found = 0;
      for (let row = 0; row < POINT_COUNT && found < RING_HIGHLIGHT_LIMIT; row++) {
        if (inputs.nearest.distances[row] > radius) continue;
        brightened.set([positions[row * 2], positions[row * 2 + 1]], found * 2);
        found++;
      }
      outputs.highlight.write(brightened);
      const text = (buffer: typeof outputs.g) => {
        const value = peek(buffer)?.[bin];
        return value === undefined ? '…' : value.toFixed(2);
      };
      return {
        spotlight: null,
        focus: found,
        glyphs: [getBinBox(PLOT, inputs.world, bin, radiusCount), ...ringGlyphs],
        readout: `r ${radius.toFixed(2)} · G ${text(outputs.g)} · F ${text(outputs.f)} · J ${text(outputs.j)}`
      };
    }
  },
  draw: ({outputs, interaction}) => [
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
      kind: 'line',
      context: true,
      values: outputs.references,
      range: [0, 1],
      rectangle: PLOT,
      radius: 1.4,
      color: [1, 1, 1, 0.5]
    },
    {layer: 'lines', context: true, segments: outputs.jReference, count: 1, color: [1, 1, 1, 0.4]},
    {
      layer: 'chart',
      kind: 'line',
      values: outputs.g,
      range: [0, 1],
      rectangle: PLOT,
      radius: 2.6,
      color: [0.3, 0.75, 1, 1]
    },
    {
      layer: 'chart',
      kind: 'line',
      values: outputs.f,
      range: [0, 1],
      rectangle: PLOT,
      radius: 2.6,
      color: [1, 0.6, 0.2, 1]
    },
    {
      layer: 'chart',
      kind: 'line',
      values: outputs.j,
      range: [0, 2],
      rectangle: PLOT,
      radius: 1.4,
      color: [1, 1, 1, 0.8]
    }
  ]
});

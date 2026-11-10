// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUSpatialCorrelogram of a spatial field. Bars are Moran's I of the points within each distance
// band (positive means similar values sit together); the line is its z-score. Both fall as the
// bands reach farther.

import {
  GPUSpatialCorrelogram,
  GPU_SPATIAL_CORRELOGRAM_PARAMETER_LENGTH,
  getGPUSpatialCorrelogramParameterValues
} from '@luma-contributors/analysis/dataframe-analysis';
import {defineOperationRun} from '../../define-run';
import {
  DISTANCE_PLOT,
  collectInsideRings,
  createRuleSegments,
  getAxisFraction,
  getAxisPointer,
  getBinBox,
  getBinIndex
} from './_lib/distance-probe';
import {createFieldSample, getFieldValues} from './_lib/helpers';
import {parameters} from './spatial-correlogram.parameters';

const POINT_COUNT = 600;
const PLOT = DISTANCE_PLOT;
const MORAN_RANGE = [-0.1, 0.4] as const;
const Z_RANGE = [-2, 12] as const;
const HIGHLIGHT_LIMIT = 300;
const REST_FRACTION = 0.3;

export default defineOperationRun({
  parameters,
  caption: 'Moran’s I (bars) and z-score (line) of log flash energy by distance band.',
  createInputs: ({random, world}) => ({
    ...createFieldSample(random, world.width * 0.5, POINT_COUNT),
    world,
    anchor: {row: 0}
  }),
  build(builder, sample, values) {
    const positions = builder.upload('positions', 'float32x2', sample.positions);
    const observed = builder.allocate('values', 'float32', POINT_COUNT);
    const parameterBuffer = builder.allocate(
      'parameters',
      'float32',
      GPU_SPATIAL_CORRELOGRAM_PARAMETER_LENGTH
    );
    const moransI = builder.allocate('morans-i', 'float32', values.bandCount);
    const zScores = builder.allocate('z-scores', 'float32', values.bandCount);
    builder.graph.add(
      new GPUSpatialCorrelogram({
        positions: positions.view,
        values: observed.view,
        parameters: parameterBuffer.view,
        gridSize: [32, 32],
        bandCount: values.bandCount,
        bandMode: values.bandMode === 0 ? 'cumulative' : 'annulus',
        output: {moransI: moransI.view, zScores: zScores.view}
      })
    );
    const highlight = builder.allocate('highlight', 'float32x2', HIGHLIGHT_LIMIT);
    const zeroRule = builder.upload(
      'zero-rule',
      'float32x4',
      createRuleSegments(PLOT, sample.world, MORAN_RANGE, [0])
    );
    const significanceRule = builder.upload(
      'significance-rule',
      'float32x4',
      createRuleSegments(PLOT, sample.world, Z_RANGE, [1.96])
    );
    return {
      outputs: {positions, observed, moransI, zScores, highlight, zeroRule, significanceRule},
      update: current => {
        observed.write(getFieldValues(sample, current.noise));
        parameterBuffer.write(
          getGPUSpatialCorrelogramParameterValues({
            bounds: [0, 0, sample.windowWidth, 1],
            maximumDistance: current.maximumDistance
          })
        );
      }
    };
  },
  rebuildOn: ['bandCount', 'bandMode'],
  // The pointer is a distance band on the chart; its ring is drawn around the flash whose height is
  // nearest the pointer's, and the flashes inside the band are brightened.
  interaction: {
    kind: 'probe',
    hint: 'hover a distance band',
    rest: [getAxisPointer(REST_FRACTION), 0.5],
    resolve(pointer, {inputs, outputs, values, peek}) {
      const bandCount = values.bandCount;
      const band = getBinIndex(getAxisFraction(pointer.unitX), bandCount);
      const bandWidth = values.maximumDistance / bandCount;
      const outer = (band + 1) * bandWidth;
      const inner = values.bandMode === 1 ? band * bandWidth : 0;
      let best = inputs.anchor.row;
      let bestDistance = Math.abs(inputs.positions[best * 2 + 1] - pointer.y) * 1.15;
      for (let row = 0; row < inputs.count; row++) {
        const distance = Math.abs(inputs.positions[row * 2 + 1] - pointer.y);
        if (distance < bestDistance) {
          best = row;
          bestDistance = distance;
        }
      }
      inputs.anchor.row = best;
      const anchor = [inputs.positions[best * 2], inputs.positions[best * 2 + 1]] as const;
      const highlight = new Float32Array(HIGHLIGHT_LIMIT * 2).fill(-10);
      const found = collectInsideRings(inputs.positions, [best], outer, highlight, inner);
      outputs.highlight.write(highlight);
      const moran = peek(outputs.moransI)?.[band];
      const score = peek(outputs.zScores)?.[band];
      return {
        spotlight: null,
        focus: found,
        glyphs: [
          getBinBox(PLOT, inputs.world, band, bandCount),
          {shape: 'ring', at: anchor, radius: outer},
          ...(inner > 0 ? [{shape: 'ring' as const, at: anchor, radius: inner}] : [])
        ],
        readout: `band ${inner.toFixed(2)}-${outer.toFixed(2)} · I ${
          moran === undefined ? '…' : moran.toFixed(2)
        } · z ${score === undefined ? '…' : score.toFixed(1)}`
      };
    }
  },
  draw: ({outputs, interaction}) => [
    {
      layer: 'dots',
      context: true,
      positions: outputs.positions,
      values: outputs.observed,
      range: [0, 10],
      radius: 2.8
    },
    {
      layer: 'dots',
      positions: outputs.highlight,
      count: Math.max(1, typeof interaction.focus === 'number' ? interaction.focus : 0),
      radius: 3.6,
      color: [1, 0.75, 0.2, 1]
    },
    {
      layer: 'lines',
      context: true,
      segments: outputs.zeroRule,
      count: 1,
      color: [0.3, 0.75, 1, 0.7]
    },
    {
      layer: 'lines',
      context: true,
      segments: outputs.significanceRule,
      count: 1,
      color: [1, 0.6, 0.2, 0.7]
    },
    {
      layer: 'chart',
      kind: 'bars',
      values: outputs.moransI,
      range: MORAN_RANGE,
      rectangle: PLOT,
      color: [0.3, 0.75, 1, 0.8]
    },
    {
      layer: 'chart',
      kind: 'line',
      values: outputs.zScores,
      range: Z_RANGE,
      rectangle: PLOT,
      radius: 2,
      color: [1, 0.6, 0.2, 1]
    }
  ]
});

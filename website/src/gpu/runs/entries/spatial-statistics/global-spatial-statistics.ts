// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUGlobalSpatialStatistics: Moran's I, Geary's C and Getis-Ord G with their moments and
// p-values in one summary buffer. The bars are the signed significance (sign of z times -log10 p, under the chosen
// alternative) of each statistic against its expectation (the line through the middle is no effect); hovering a point shows its distance-band
// neighbors, the pairs the statistics are summed over.

import {
  GPU_GLOBAL_SPATIAL_STATISTIC_FIELD,
  GPU_GLOBAL_SPATIAL_STATISTICS_LAYOUT,
  GPUGlobalSpatialStatistics
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {uploadSpatialWeights} from '../../inputs';
import {
  createBandGraph,
  createNeighborGlyphs,
  createPointField,
  createProbeInputs,
  getNeighborRows,
  mixNoise,
  pickProbeRow
} from './shared/point-field';
import {parameters} from './global-spatial-statistics.parameters';

const ALTERNATIVES = ['two-sided', 'greater', 'lesser'] as const;
const STATISTIC_BLOCKS = ['moran', 'geary', 'getisOrdG'] as const;
const Z_LIMIT = 8;
const CHART = [0.3, 0.04, 0.7, 0.24] as const;

export default defineOperationRun({
  parameters,
  caption:
    'Global Moran, Geary and Getis-Ord G: signed significance of each against its expectation.',
  createInputs: ({random, world}) => createProbeInputs(createPointField(random, world, 700)),
  build(builder, points, values) {
    const graph = createBandGraph(points.positions, values.radius);
    const valueBuffer = builder.allocate('values', 'float32', points.count);
    const summary = builder.allocate(
      'summary',
      'float32',
      GPU_GLOBAL_SPATIAL_STATISTICS_LAYOUT.length
    );
    const zScores = builder.allocate('z-scores', 'float32', STATISTIC_BLOCKS.length);
    builder.graph.add(
      new GPUGlobalSpatialStatistics({
        weights: uploadSpatialWeights(builder, graph.weights),
        values: valueBuffer.view,
        statistics: ['moran', 'geary', 'getisOrdG'],
        alternative: ALTERNATIVES[values.alternative],
        output: {summary: summary.view}
      })
    );
    return {
      outputs: {
        valueBuffer,
        summary,
        zScores,
        graph,
        positions: builder.upload('positions', 'float32x2', points.positions)
      },
      update: current => valueBuffer.write(mixNoise(points.field, points.noise, current.noise))
    };
  },
  rebuildOn: ['alternative', 'radius'],
  interaction: {
    kind: 'probe',
    hint: 'hover a point',
    rest: [0.5, 0.4],
    resolve(pointer, {inputs, outputs, values, world, peek}) {
      const summary = peek(outputs.summary);
      const getField = (block: number) => (summary ? summary[block] : NaN);
      const moranZ = getField(
        GPU_GLOBAL_SPATIAL_STATISTICS_LAYOUT.moran + GPU_GLOBAL_SPATIAL_STATISTIC_FIELD.zNormality
      );
      if (summary) {
        outputs.zScores.write(
          // Signed significance: sign(z) * -log10(p), so the `alternative` tail changes the bars.
          Float32Array.from(STATISTIC_BLOCKS, block => {
            const base = GPU_GLOBAL_SPATIAL_STATISTICS_LAYOUT[block];
            const z = getField(base + GPU_GLOBAL_SPATIAL_STATISTIC_FIELD.zNormality);
            const p = getField(base + GPU_GLOBAL_SPATIAL_STATISTIC_FIELD.pNormality);
            return Math.sign(z) * Math.min(Z_LIMIT, -Math.log10(Math.max(p, 1e-12)));
          })
        );
      }
      const moran = getField(
        GPU_GLOBAL_SPATIAL_STATISTICS_LAYOUT.moran + GPU_GLOBAL_SPATIAL_STATISTIC_FIELD.statistic
      );
      const zero = (CHART[1] + (CHART[3] - CHART[1]) / 2) * world.height;
      const glyphs = [
        {
          shape: 'line',
          from: [CHART[0] * world.width, zero],
          to: [CHART[2] * world.width, zero]
        }
      ] as const;
      const summaryText = summary
        ? `Moran I ${moran.toFixed(2)} (z ${moranZ.toFixed(1)})`
        : 'Moran I …';
      const row = pickProbeRow(inputs, pointer);
      if (row < 0) return {glyphs, readout: summaryText};
      const neighbors = getNeighborRows(outputs.graph.weights, row);
      return {
        glyphs: [
          ...glyphs,
          ...createNeighborGlyphs(inputs.positions, outputs.graph.weights, row),
          {
            shape: 'ring',
            at: [inputs.positions[row * 2], inputs.positions[row * 2 + 1]],
            radius: values.radius,
            dashed: true
          }
        ],
        readout: `radius ${values.radius} · ${neighbors.length} neighbors · ${summaryText}`
      };
    }
  },
  draw: ({outputs}) => [
    {
      layer: 'dots',
      positions: outputs.positions,
      values: outputs.valueBuffer,
      radius: 4,
      range: [0, 1]
    },
    {
      layer: 'chart',
      kind: 'bars',
      values: outputs.zScores,
      ramp: 'diverging',
      range: [-Z_LIMIT, Z_LIMIT],
      rectangle: CHART
    }
  ]
});

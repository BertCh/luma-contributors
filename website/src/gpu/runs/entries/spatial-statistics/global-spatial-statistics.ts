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
import {drawSignedBars} from '../regression-weights/shared/regression';
import {createChartAxes, getChartRectangle, mergeText} from '../../text';
import {
  createBandGraph,
  createNeighborGlyphs,
  createPointField,
  createProbeInputs,
  getNeighborRows,
  mixNoise
} from './shared/point-field';
import {parameters} from './global-spatial-statistics.parameters';

const ALTERNATIVES = ['two-sided', 'greater', 'lesser'] as const;
const STATISTIC_BLOCKS = ['moran', 'geary', 'getisOrdG'] as const;
const Z_LIMIT = 8;
const CHART = [0.3, 0.1, 0.7, 0.28] as const;
// The band radii the pointer can settle on. The pointer's distance from the anchor, over
// REACH_SPAN, runs from the smallest radius to the largest.
const RADII = [0.04, 0.055, 0.07, 0.09, 0.12] as const;
const DEFAULT_RADIUS = 0.07;
const REACH_SPAN = 0.6;
const REST_REACH =
  ((DEFAULT_RADIUS - RADII[0]) / (RADII[RADII.length - 1] - RADII[0])) * REACH_SPAN;

/** The point nearest the middle of the map: the ring around it shows what a radius gathers. */
function findAnchor(positions: Float32Array, worldWidth: number): number {
  let anchor = 0;
  let nearest = Infinity;
  for (let row = 0; row < positions.length / 2; row++) {
    const distance = Math.hypot(positions[row * 2] - worldWidth / 2, positions[row * 2 + 1] - 0.62);
    if (distance < nearest) {
      nearest = distance;
      anchor = row;
    }
  }
  return anchor;
}

export default defineOperationRun({
  parameters,
  caption:
    'Global Moran, Geary and Getis-Ord G: signed significance of each against its expectation.',
  createInputs: ({random, world}) => {
    const points = createPointField(random, world, 700);
    return {...createProbeInputs(points), anchor: findAnchor(points.positions, world.width)};
  },
  buildCache: 5,
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
  // The pointer's distance from the anchor is the band radius of the weights, settled on one of five values:
  // the points inside it are the neighbors of the anchor, and the three bars are recomputed.
  interaction: {
    kind: 'reach',
    hint: 'stretch the radius',
    rest: ({inputs, world}) => [
      (inputs.positions[inputs.anchor * 2] + REST_REACH) / world.width,
      1 - inputs.positions[inputs.anchor * 2 + 1] / world.height
    ],
    resolve(pointer, {inputs, outputs, values, world, peek}) {
      const summary = peek(outputs.summary);
      const getField = (block: number) => (summary ? summary[block] : NaN);
      const anchor = [
        inputs.positions[inputs.anchor * 2],
        inputs.positions[inputs.anchor * 2 + 1]
      ] as const;
      const reach = Math.hypot(pointer.x - anchor[0], pointer.y - anchor[1]);
      const requested =
        RADII[0] + Math.min(reach / REACH_SPAN, 1) * (RADII[RADII.length - 1] - RADII[0]);
      const radius = RADII.reduce((nearest, candidate) =>
        Math.abs(candidate - requested) < Math.abs(nearest - requested) ? candidate : nearest
      );
      const moranZ = getField(
        GPU_GLOBAL_SPATIAL_STATISTICS_LAYOUT.moran + GPU_GLOBAL_SPATIAL_STATISTIC_FIELD.zNormality
      );
      const moran = getField(
        GPU_GLOBAL_SPATIAL_STATISTICS_LAYOUT.moran + GPU_GLOBAL_SPATIAL_STATISTIC_FIELD.statistic
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
      const zero = (CHART[1] + (CHART[3] - CHART[1]) / 2) * world.height;
      const neighbors = getNeighborRows(outputs.graph.weights, inputs.anchor);
      return {
        settle: {radius},
        glyphs: [
          {
            shape: 'line',
            from: [CHART[0] * world.width, zero],
            to: [CHART[2] * world.width, zero]
          },
          ...createNeighborGlyphs(inputs.positions, outputs.graph.weights, inputs.anchor),
          {shape: 'ring', at: anchor, radius},
          {shape: 'line', from: anchor, to: [pointer.x, pointer.y], dashed: true}
        ],
        readout: `radius ${radius} · ${neighbors.length} neighbors${
          summary ? ` · Moran I ${moran.toFixed(2)} (z ${moranZ.toFixed(1)})` : ''
        }${values.radius !== radius ? ' …' : ''}`
      };
    }
  },
  draw: ({outputs, world}) => {
    const axes = createChartAxes({
      rectangle: getChartRectangle(world, CHART),
      x: {categories: ['Moran I', 'Geary C', 'Getis-Ord G']},
      y: {range: [-Z_LIMIT, Z_LIMIT], ticks: [-Z_LIMIT, 0, Z_LIMIT], title: 'signed significance'}
    });
    return [
      {
        layer: 'dots',
        positions: outputs.positions,
        values: outputs.valueBuffer,
        radius: 4,
        range: [0, 1]
      },
      ...drawSignedBars(outputs.zScores, CHART, Z_LIMIT),
      {layer: 'text', ...mergeText(axes)}
    ];
  }
});

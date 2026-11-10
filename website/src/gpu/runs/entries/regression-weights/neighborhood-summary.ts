// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUNeighborhoodSummary writes eight statistics of every point's neighborhood. Left: the input
// values. Right: the chosen column of the output table, extracted from the buffer the operation wrote.

import {GPUNeighborhoodSummary} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {createKnnWeights, createUniformPoints} from '../../inputs';
import type {RunGlyph} from '../../interaction';
import {chooseSticky, createPointIndex} from '../../pick';
import {parameters} from './neighborhood-summary.parameters';
import {
  addTableColumn,
  getRowPositions,
  peekRow,
  stretchPoints,
  uploadWeightsBuffers
} from './shared/weights';

const STATISTICS = [
  'count',
  'weightSum',
  'sum',
  'mean',
  'min',
  'max',
  'standardDeviation',
  'median'
] as const;

/** Fixed ranges per statistic, so the output is read on the input's scale. */
const STATISTIC_RANGES: ((neighborCount: number) => [number, number])[] = [
  neighborCount => [0, neighborCount + 1],
  neighborCount => [0, neighborCount + 1],
  neighborCount => [0, neighborCount + 1],
  () => [0, 1],
  () => [0, 1],
  () => [0, 1],
  () => [0, 0.5],
  () => [0, 1]
];

export default defineOperationRun({
  parameters,
  caption: 'Input values (left) and one statistic of each neighborhood (right).',
  createInputs: ({random, world}) => {
    const mapWorld = {width: world.width / 2, height: world.height};
    const points = stretchPoints(createUniformPoints(random, mapWorld, 260, 0.35), mapWorld);
    // Rank-normalize the values so the whole ramp is used and neighbors differ visibly.
    const order = Array.from(points.values.keys()).sort(
      (first, second) => points.values[first] - points.values[second]
    );
    const values = new Float32Array(points.count);
    order.forEach((row, rank) => {
      values[row] = rank / (points.count - 1);
    });
    return {world, ...points, values, index: createPointIndex(points.positions), focus: {row: -1}};
  },
  build(builder, points, values) {
    const {world} = points;
    const rightPositions = Float32Array.from(points.positions, (value, index) =>
      index % 2 === 0 ? value + world.width / 2 : value
    );
    const positions = builder.upload('positions', 'float32x2', points.positions);
    const shifted = builder.upload('shifted-positions', 'float32x2', rightPositions);
    const sourceValues = builder.upload('values', 'float32', points.values);
    const weights = uploadWeightsBuffers(
      builder,
      'neighbors',
      createKnnWeights(points.positions, values.k)
    );
    const table = builder.allocate('table', 'float32', points.count * STATISTICS.length);
    const overflow = builder.allocate('overflow', 'uint32', 1);
    builder.graph.add(
      new GPUNeighborhoodSummary({
        weights: weights.input,
        values: sourceValues.view,
        statistics: STATISTICS,
        includeFocal: values.includeFocal === 1,
        output: {table: table.view, status: {overflow: overflow.view}}
      })
    );
    const columns = STATISTICS.map((_, column) =>
      addTableColumn(builder, 'summary', table, STATISTICS.length, column, points.count)
    );
    return {outputs: {positions, shifted, sourceValues, columns, weights}};
  },
  rebuildOn: ['includeFocal', 'k'],
  // The pointer picks a point on either map: its k neighbors on the left and its summary value on the right.
  interaction: {
    kind: 'probe',
    hint: 'hover a point',
    rest: [0.25, 0.5],
    resolve(pointer, {inputs, outputs, values, peek}) {
      const half = inputs.world.width / 2;
      const x = pointer.x > half ? pointer.x - half : pointer.x;
      const row = chooseSticky(inputs.index, x, pointer.y, inputs.focus.row);
      inputs.focus.row = row;
      const neighbors = peekRow(peek, outputs.weights, row);
      const column = peek(outputs.columns[values.statistic]);
      if (row < 0 || !neighbors || !column) return {};
      const from = [inputs.positions[row * 2], inputs.positions[row * 2 + 1]] as const;
      const shifted = [from[0] + half, from[1]] as const;
      const glyphs: RunGlyph[] = [
        {shape: 'ring', at: from, radius: 0.02},
        {shape: 'ring', at: shifted, radius: 0.02},
        ...getRowPositions(inputs.positions, neighbors).flatMap(to => [
          {shape: 'line' as const, from, to},
          {shape: 'dot' as const, at: to}
        ])
      ];
      return {
        focus: row,
        glyphs,
        readout: `${STATISTICS[values.statistic]} of ${neighbors.length} neighbors ${column[row].toFixed(2)} (own ${inputs.values[row].toFixed(2)})`
      };
    }
  },
  draw: ({outputs, values}) => [
    {
      layer: 'dots',
      positions: outputs.positions,
      values: outputs.sourceValues,
      ramp: 'sequential',
      range: [0, 1],
      radius: 4.5
    },
    {
      layer: 'dots',
      positions: outputs.shifted,
      values: outputs.columns[values.statistic],
      ramp: 'sequential',
      range: STATISTIC_RANGES[values.statistic](values.k),
      radius: 4.5
    }
  ]
});

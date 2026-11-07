// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUNeighborhoodSummary writes eight statistics of every point's neighborhood. Left: the input
// values. Right: the chosen column of the output table, extracted from the buffer the operation wrote.

import {GPUNeighborhoodSummary} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {createKnnWeights, createUniformPoints} from '../../inputs';
import {parameters} from './neighborhood-summary.parameters';
import {addTableColumn, uploadWeightsBuffers} from './shared/weights';

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

export default defineOperationRun({
  parameters,
  caption: 'Input values (left) and one statistic of each neighborhood (right).',
  createInputs: ({random, world}) => ({
    world,
    ...createUniformPoints(random, {width: world.width / 2, height: world.height}, 260, 0.35)
  }),
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
    return {outputs: {positions, shifted, sourceValues, columns}};
  },
  rebuildOn: ['includeFocal', 'k'],
  draw: ({outputs, values}) => [
    {
      layer: 'dots',
      positions: outputs.positions,
      values: outputs.sourceValues,
      range: [0, 1],
      radius: 4
    },
    {
      layer: 'dots',
      positions: outputs.shifted,
      values: outputs.columns[values.statistic],
      radius: 4
    }
  ]
});

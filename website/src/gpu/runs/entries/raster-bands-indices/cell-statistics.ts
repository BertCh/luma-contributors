// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPURasterCellStatistics over a five-layer stack (green, red, NIR, SWIR1, SWIR2) where clouds
// blank different cells in each layer. Statistics use each cell's valid layers; cells with fewer
// than `minimumValidCount` valid layers become nodata. All outputs are written; a slider picks
// which one is drawn.

import {
  getGPURasterCellStatisticsParameterValues,
  GPURasterCellStatistics
} from '@luma-contributors/analysis/raster-analysis';
import {defineOperationRun} from '../../define-run';
import {createImageryBands, createSmoothField} from '../../inputs';
import {parameters} from './cell-statistics.parameters';
import {formatNumber} from '../../interaction';
import {drawRaster} from './support/bands';
import {getCellAt, getCellBox} from './support/gestures';

const STATISTICS = ['mean', 'maximum', 'range', 'standardDeviation', 'variety', 'counts'] as const;
const LAYER_COUNT = 5;
const RANGES = {
  mean: [0, 0.25],
  maximum: [0, 0.5],
  range: [0, 0.3],
  standardDeviation: [0, 0.15],
  variety: [0, 5],
  counts: [0, 5]
} as const;

export default defineOperationRun({
  parameters,
  caption: 'A stack of five layers in; per-cell statistics of the valid layers out.',
  createInputs: ({random, world}) => {
    const imagery = createImageryBands(world);
    const layers = [
      imagery.green,
      imagery.red,
      imagery.nearInfrared,
      imagery.shortwaveInfrared,
      imagery.shortwaveInfrared2
    ];
    const clouds = layers.map(() => {
      const field = createSmoothField(random, 3, 4);
      const cloud = new Float32Array(imagery.values.length);
      for (let index = 0; index < cloud.length; index++) {
        const row = Math.floor(index / imagery.columns);
        const column = index % imagery.columns;
        cloud[index] = field((column + 0.5) / imagery.columns, (row + 0.5) / imagery.rows);
      }
      return cloud;
    });
    return {...imagery, layers, clouds};
  },
  build(builder, inputs) {
    const cellCount = inputs.values.length;
    const stack = builder.allocate('stack', 'float32', cellCount * LAYER_COUNT);
    const parameterBuffer = builder.allocate('parameters', 'float32', 4);
    const outputs = {
      mean: builder.allocate('mean', 'float32', cellCount),
      maximum: builder.allocate('maximum', 'float32', cellCount),
      range: builder.allocate('range', 'float32', cellCount),
      standardDeviation: builder.allocate('standard-deviation', 'float32', cellCount),
      variety: builder.allocate('variety', 'uint32', cellCount),
      counts: builder.allocate('counts', 'uint32', cellCount)
    };
    builder.graph.add(
      new GPURasterCellStatistics({
        stack: stack.view,
        layerCount: LAYER_COUNT,
        cellCount,
        parameters: parameterBuffer.view,
        output: {
          mean: outputs.mean.view,
          maximum: outputs.maximum.view,
          range: outputs.range.view,
          standardDeviation: outputs.standardDeviation.view,
          variety: outputs.variety.view,
          counts: outputs.counts.view
        }
      })
    );
    const first = builder.upload('first-layer', 'float32', inputs.layers[0]);
    const inset = builder.allocate('inset', 'float32', LAYER_COUNT);
    return {
      outputs: {...outputs, first, inset},
      update: current => {
        const data = new Float32Array(cellCount * LAYER_COUNT);
        // Cloud mask: the cells where the smooth field is highest are blank in that layer.
        const threshold = 1 - current.cloudCover;
        for (let layer = 0; layer < LAYER_COUNT; layer++) {
          for (let cell = 0; cell < cellCount; cell++) {
            data[layer * cellCount + cell] =
              inputs.clouds[layer][cell] > threshold ? Number.NaN : inputs.layers[layer][cell];
          }
        }
        stack.write(data);
        parameterBuffer.write(
          getGPURasterCellStatisticsParameterValues({minimumValidCount: current.minimumValidCount})
        );
      }
    };
  },
  // The pointer probes one cell: its five layer values (clouded layers are gaps) in an inset bar
  // chart, with the cell's valid count and mean in the readout.
  interaction: {
    kind: 'probe',
    hint: 'probe a cell',
    rest: [0.5, 0.5],
    resolve: (pointer, {inputs, outputs, values, world}) => {
      const {column, row, index} = getCellAt(inputs, world, pointer);
      const threshold = 1 - values.cloudCover;
      const layerValues = inputs.layers.map((layer, layerIndex) =>
        inputs.clouds[layerIndex][index] > threshold ? Number.NaN : layer[index]
      );
      const valid = layerValues.filter(value => Number.isFinite(value));
      outputs.inset.write(
        Float32Array.from(layerValues, value => (Number.isFinite(value) ? value : 0))
      );
      const mean = valid.reduce((sum, value) => sum + value, 0) / (valid.length || 1);
      const kept = valid.length >= values.minimumValidCount;
      return {
        glyphs: [getCellBox(inputs, world, column, row)],
        readout: `${valid.length} of 5 valid | ${kept ? `mean ${formatNumber(mean)}` : 'nodata'}`
      };
    }
  },
  draw: ({inputs, outputs, values}) => {
    const statistic = STATISTICS[Math.round(values.statistic)];
    return [
      drawRaster(inputs, outputs.first, {context: true}),
      drawRaster(inputs, outputs[statistic], {
        ramp: 'sequential',
        range: RANGES[statistic],
        smoothing: statistic !== 'variety' && statistic !== 'counts'
      }),
      {
        layer: 'chart',
        kind: 'bars',
        values: outputs.inset,
        range: [0, 0.6],
        rectangle: [0.03, 0.04, 0.3, 0.3]
      }
    ];
  }
});

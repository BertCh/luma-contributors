// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUAreaOfApplicability over Sentinel-2 bands. The pointer is where the fieldwork was done: the
// training pixels are those in a disc around it (the operation's training mask); every pixel of the scene gets a dissimilarity index, its standardized
// distance to the nearest training pixel relative to the training pixels' own spread. Pixels whose
// index exceeds the threshold (Q3 + 1.5 IQR of the training indices) are outside the area of
// applicability and drawn in the warm overlay.

import {
  GPUAreaOfApplicability,
  GPU_AREA_OF_APPLICABILITY_SUMMARY_LENGTH
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {createImageryBands} from '../../inputs';
import {formatNumber, type RunGlyph} from '../../interaction';
import {parameters} from './area-of-applicability.parameters';

/** Every other pixel in both directions is a candidate training pixel (the training pass is quadratic). */
const TRAINING_STRIDE = 2;
const INDEX_RANGE: readonly [number, number] = [0, 3];
const OUTSIDE_VALUE = 1;
const INSIDE_VALUE = 0xffffffff;

function createInputs({world}: {world: {width: number; height: number}}) {
  const imagery = createImageryBands(world);
  const trainingPixels: number[] = [];
  for (let row = 0; row < imagery.rows; row += TRAINING_STRIDE) {
    for (let column = 0; column < imagery.columns; column += TRAINING_STRIDE) {
      trainingPixels.push(row * imagery.columns + column);
    }
  }
  return {
    ...imagery,
    world,
    trainingPixels: Uint32Array.from(trainingPixels),
    /** The training disc, pointer-owned: center in world units. */
    box: {x: world.width * 0.3, y: 0.45}
  };
}

type Inputs = ReturnType<typeof createInputs>;

/** Bands in the order the predictorCount slider adds them. */
function getBands(imagery: Inputs) {
  return [
    imagery.red,
    imagery.nearInfrared,
    imagery.shortwaveInfrared,
    imagery.green,
    imagery.blue,
    imagery.shortwaveInfrared2
  ];
}

function getDiscRadius(values: Readonly<Record<string, number>>): number {
  return (values.discRadius / 100) * 0.5;
}

/** World position of a pixel center; row 0 is at the top. */
function getPixelCenter(inputs: Inputs, pixel: number): [number, number] {
  const column = pixel % inputs.columns;
  const row = Math.floor(pixel / inputs.columns);
  return [(column + 0.5) * inputs.cellSize, inputs.world.height - (row + 0.5) * inputs.cellSize];
}

function getMask(inputs: Inputs, values: Readonly<Record<string, number>>): Uint32Array {
  const half = getDiscRadius(values);
  return Uint32Array.from(inputs.trainingPixels, pixel => {
    const [x, y] = getPixelCenter(inputs, pixel);
    return Math.hypot(x - inputs.box.x, y - inputs.box.y) <= half ? 1 : 0;
  });
}

export default defineOperationRun({
  parameters,
  caption:
    'Dissimilarity index and area of applicability of Sentinel-2 bands for training pixels in a disc around the pointer; no cross-validation folds, so the threshold is tight.',
  createInputs,
  build(builder, inputs, values) {
    const predictorCount = values.predictorCount;
    const bands = getBands(inputs).slice(0, predictorCount);
    const pixelCount = inputs.columns * inputs.rows;
    const query = new Float32Array(pixelCount * predictorCount);
    const training = new Float32Array(inputs.trainingPixels.length * predictorCount);
    for (let pixel = 0; pixel < pixelCount; pixel++) {
      bands.forEach((band, predictor) => (query[pixel * predictorCount + predictor] = band[pixel]));
    }
    inputs.trainingPixels.forEach((pixel, row) => {
      training.set(
        query.subarray(pixel * predictorCount, (pixel + 1) * predictorCount),
        row * predictorCount
      );
    });
    const mask = builder.allocate('mask', 'uint32', inputs.trainingPixels.length);
    const weights = builder.allocate('weights', 'float32', predictorCount);
    const queryDissimilarity = builder.allocate('query-dissimilarity', 'float32', pixelCount);
    const applicable = builder.allocate('applicable', 'uint32', pixelCount);
    const summary = builder.allocate(
      'summary',
      'float32',
      GPU_AREA_OF_APPLICABILITY_SUMMARY_LENGTH
    );
    builder.graph.add(
      new GPUAreaOfApplicability({
        training: builder.upload('training', 'float32', training).view,
        query: builder.upload('query', 'float32', query).view,
        predictorCount,
        weights: weights.view,
        mask: mask.view,
        output: {
          trainDissimilarity: builder.allocate(
            'train-dissimilarity',
            'float32',
            inputs.trainingPixels.length
          ).view,
          queryDissimilarity: queryDissimilarity.view,
          applicable: applicable.view,
          summary: summary.view,
          status: builder.allocate('status', 'uint32', 1).view
        }
      })
    );
    return {
      outputs: {
        queryDissimilarity,
        applicable,
        summary,
        red: builder.upload('red', 'float32', inputs.red)
      },
      update: (current: Record<string, number>) => {
        mask.write(getMask(inputs, current));
        weights.write(
          Float32Array.from({length: predictorCount}, (_, predictor) =>
            predictor === 0 ? current.weights : 1
          )
        );
      }
    };
  },
  rebuildOn: ['predictorCount'],
  buildCache: 5,
  // The applicable flags are read back so the cells outside the area can be drawn as their own layer.
  async afterRun({inputs, outputs, upload}) {
    const bytes = await outputs.applicable.buffer.readAsync();
    const flags = new Uint32Array(bytes.buffer, bytes.byteOffset, inputs.columns * inputs.rows);
    return {
      outside: upload(
        'outside',
        'uint32',
        Uint32Array.from(flags, flag => (flag === 1 ? INSIDE_VALUE : OUTSIDE_VALUE))
      )
    };
  },
  interaction: {
    kind: 'place',
    hint: 'move the fieldwork',
    rest: ({inputs, toUnit}) => toUnit([inputs.box.x, inputs.box.y]),
    resolve(pointer, {inputs, outputs, values, peek}) {
      const half = getDiscRadius(values);
      inputs.box = {x: pointer.x, y: pointer.y};
      const glyphs: RunGlyph[] = [{shape: 'ring', at: [pointer.x, pointer.y], radius: half}];
      const summary = peek(outputs.summary);
      const flags = peek(outputs.applicable);
      let readout = `training radius ${values.discRadius}%`;
      if (summary && flags) {
        let inside = 0;
        for (const flag of flags) inside += flag === 1 ? 1 : 0;
        readout = `${formatNumber(summary[2], 0)} samples · threshold ${formatNumber(summary[1], 2)} · ${formatNumber((inside / flags.length) * 100, 0)}% inside`;
      }
      return {rerun: true, glyphs, readout, spotlight: null};
    }
  },
  draw: ({inputs, outputs, extra}) => [
    {
      layer: 'raster',
      context: true,
      ramp: 'grey',
      values: outputs.red,
      columns: inputs.columns,
      rows: inputs.rows
    },
    {
      layer: 'raster',
      values: outputs.queryDissimilarity,
      columns: inputs.columns,
      rows: inputs.rows,
      ramp: 'sequential',
      range: INDEX_RANGE,
      alpha: 0.8
    },
    ...(extra
      ? [
          {
            layer: 'raster' as const,
            values: extra.outside,
            columns: inputs.columns,
            rows: inputs.rows,
            alpha: 0.6,
            categories: [
              [0.2, 0.2, 0.2],
              [0.95, 0.6, 0.1]
            ] as const
          }
        ]
      : [])
  ]
});

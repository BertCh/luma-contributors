// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPURasterFocalPairs: a moving-window regression of a response band on a predictor band. The
// response follows the predictor with a slope that changes sign across the map, so the local
// slope surface recovers that hidden relationship; correlation, R-squared and intercept are
// written too. Window radius and minimum pair count are construction props (the run rebuilds);
// noise only rewrites the response band. The pointer is the window: its (2 * radius + 1) cells
// square is outlined and an inset scatter shows its predictor and response pairs with the fitted
// line.

import {GPURasterFocalPairs} from '@luma-contributors/analysis/raster-analysis';
import {defineOperationRun} from '../../define-run';
import {createElevationRaster, createSmoothField} from '../../inputs';
import {formatNumber} from '../../interaction';
import {parameters} from './focal-correlation-regression.parameters';
import {getCellAt, getCellBox} from './support/gestures';
import {drawRaster, makeGridOfRaster, uploadBand} from './support/bands';

const METRICS = ['slope', 'correlation', 'rSquared', 'intercept'] as const;
const SCATTER = [0.03, 0.05, 0.3, 0.36] as const;
const SCATTER_CAPACITY = 289;

export default defineOperationRun({
  parameters,
  rebuildOn: ['radius', 'minimumCount'],
  caption: 'Predictor and response bands in; the local regression of one on the other out.',
  createInputs: ({random, world}) => {
    const predictor = createElevationRaster(world, 56);
    const relation = createSmoothField(random, 2, 2);
    const noise = Float32Array.from(predictor.values, () => random() - 0.5);
    const slope = Float32Array.from(predictor.values, (_, index) => {
      const row = Math.floor(index / predictor.columns);
      const column = index % predictor.columns;
      return 2 * relation((column + 0.5) / predictor.columns, (row + 0.5) / predictor.rows) - 1;
    });
    // The inset's axes follow the window's own extent, kept here for `draw`.
    return {
      ...predictor,
      noise,
      slope,
      scatterX: [0, 1] as [number, number],
      scatterY: [-1, 1] as [number, number]
    };
  },
  build(builder, inputs, values) {
    const cellCount = inputs.values.length;
    const predictor = uploadBand(builder, inputs, 'predictor', inputs.values);
    const response = uploadBand(builder, inputs, 'response', inputs.values);
    const output = {
      pairedCount: builder.allocate('paired-count', 'uint32', cellCount),
      status: builder.allocate('status', 'uint32', cellCount),
      correlation: builder.allocate('correlation', 'float32', cellCount),
      slope: builder.allocate('slope', 'float32', cellCount),
      intercept: builder.allocate('intercept', 'float32', cellCount),
      rSquared: builder.allocate('r-squared', 'float32', cellCount)
    };
    builder.graph.add(
      new GPURasterFocalPairs({
        grid: makeGridOfRaster(inputs),
        predictor: predictor.band,
        response: response.band,
        radius: values.radius,
        minimumCount: values.minimumCount,
        output: {
          pairedCount: output.pairedCount.view,
          status: output.status.view,
          correlation: output.correlation.view,
          slope: output.slope.view,
          intercept: output.intercept.view,
          rSquared: output.rSquared.view
        }
      })
    );
    return {
      outputs: {
        ...output,
        predictor: predictor.buffer,
        windowPredictor: builder.allocate('window-predictor', 'float32', SCATTER_CAPACITY),
        windowResponse: builder.allocate('window-response', 'float32', SCATTER_CAPACITY)
      },
      update: current =>
        response.buffer.write(
          Float32Array.from(
            inputs.values,
            (value, index) => value * inputs.slope[index] + inputs.noise[index] * current.noise
          )
        )
    };
  },
  interaction: {
    kind: 'window',
    hint: 'move the regression window',
    rest: [0.5, 0.5],
    resolve: (pointer, {inputs, outputs, values, world, peek}) => {
      const {column, row, index} = getCellAt(inputs, world, pointer);
      const xs = new Float32Array(SCATTER_CAPACITY).fill(Number.NaN);
      const ys = new Float32Array(SCATTER_CAPACITY).fill(Number.NaN);
      let pairs = 0;
      for (
        let rowIndex = Math.max(0, row - values.radius);
        rowIndex <= Math.min(inputs.rows - 1, row + values.radius);
        rowIndex++
      ) {
        for (
          let columnIndex = Math.max(0, column - values.radius);
          columnIndex <= Math.min(inputs.columns - 1, column + values.radius);
          columnIndex++
        ) {
          const cell = rowIndex * inputs.columns + columnIndex;
          xs[pairs] = inputs.values[cell];
          ys[pairs] = inputs.values[cell] * inputs.slope[cell] + inputs.noise[cell] * values.noise;
          pairs++;
        }
      }
      const finite = (array: Float32Array) => array.filter(Number.isFinite);
      const spread = (array: Float32Array): [number, number] => {
        const valid = finite(array);
        const low = Math.min(...valid);
        const high = Math.max(...valid);
        const pad = Math.max((high - low) * 0.15, 1e-3);
        return [low - pad, high + pad];
      };
      inputs.scatterX = spread(xs);
      inputs.scatterY = spread(ys);
      outputs.windowPredictor.write(xs);
      outputs.windowResponse.write(ys);
      const slope = peek(outputs.slope);
      const intercept = peek(outputs.intercept);
      const correlation = peek(outputs.correlation);
      const [lowX, highX] = inputs.scatterX;
      const [lowY, highY] = inputs.scatterY;
      const toWorld = (x: number, y: number): [number, number] => [
        world.width * (SCATTER[0] + ((x - lowX) / (highX - lowX)) * (SCATTER[2] - SCATTER[0])),
        world.height * (SCATTER[1] + ((y - lowY) / (highY - lowY)) * (SCATTER[3] - SCATTER[1]))
      ];
      const fit = (x: number) => (intercept && slope ? intercept[index] + slope[index] * x : 0);
      return {
        glyphs: [
          getCellBox(inputs, world, column, row, values.radius),
          {shape: 'box', minimum: toWorld(lowX, lowY), maximum: toWorld(highX, highY)},
          ...(slope && intercept
            ? [
                {
                  shape: 'line' as const,
                  from: toWorld(lowX, fit(lowX)),
                  to: toWorld(highX, fit(highX))
                }
              ]
            : [])
        ],
        readout: `slope ${slope ? formatNumber(slope[index]) : '-'} | r ${correlation ? formatNumber(correlation[index]) : '-'} | ${pairs} pairs`
      };
    }
  },
  draw: ({inputs, outputs, values}) => {
    const metric = METRICS[Math.round(values.metric)];
    return [
      drawRaster(inputs, outputs.predictor, {context: true}),
      drawRaster(inputs, outputs[metric], {
        ramp: metric === 'slope' || metric === 'correlation' ? 'diverging' : 'sequential',
        range: metric === 'slope' ? [-1.5, 1.5] : metric === 'correlation' ? [-1, 1] : 'auto',
        smoothing: true
      }),
      {
        layer: 'chart',
        kind: 'scatter',
        values: outputs.windowResponse,
        xValues: outputs.windowPredictor,
        xRange: inputs.scatterX,
        range: inputs.scatterY,
        color: [1, 1, 1, 0.9],
        radius: 3,
        rectangle: SCATTER
      }
    ];
  }
});

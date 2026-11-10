// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUKriging: ordinary kriging of scattered samples with a spherical, exponential or gaussian
// variogram. `output` switches between the prediction (`values`) and the kriging variance, which
// is lowest at the samples.

import {
  getGPUKrigingParameterValues,
  GPUKriging
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {findParameter, snapToParameter} from '../../interaction';
import {parameters} from './kriging.parameters';
import {createRunGrid} from './support/grid';
import {createStretchedSamples, drawSampleDots, findNearestSamples} from './support/interpolation';

const MODELS = ['spherical', 'exponential', 'gaussian'] as const;

export default defineOperationRun({
  parameters,
  caption: 'Kriged surface from scattered samples (dots).',
  createInputs: ({random, world}) => {
    const samples = createStretchedSamples(random, world, 70);
    // The sample nearest the middle anchors the range ring.
    const [anchor] = findNearestSamples(samples.positions, world.width / 2, 0.5, 1);
    return {world, grid: createRunGrid(world, 64), samples, anchor};
  },
  build(builder, {grid, samples}) {
    const positions = builder.upload('positions', 'float32x2', samples.positions);
    const sampleValues = builder.upload('values', 'float32', samples.values);
    const parameterBuffer = builder.allocate('parameters', 'float32', 12);
    const prediction = builder.allocate('prediction', 'float32', grid.columns * grid.rows);
    const variance = builder.allocate('variance', 'float32', grid.columns * grid.rows);
    builder.graph.add(
      new GPUKriging({
        positions: positions.view,
        values: sampleValues.view,
        parameters: parameterBuffer.view,
        grid: grid.grid,
        indexGridSize: [12, 8],
        indexBounds: grid.window,
        maximumNeighborCount: 16,
        output: {values: prediction.view, variance: variance.view}
      })
    );
    return {
      outputs: {prediction, variance, positions, sampleValues},
      update: values =>
        parameterBuffer.write(
          getGPUKrigingParameterValues(
            {
              searchRadius: Infinity,
              neighborCount: 16,
              variogram: {
                model: MODELS[values.model],
                nugget: values.nugget,
                sill: 1,
                range: values.range
              }
            },
            grid.grid
          )
        )
    };
  },
  // The distance from the pointer to the anchor sample is the variogram range: the ring is the
  // distance beyond which two samples no longer inform each other.
  interaction: {
    kind: 'reach',
    hint: 'stretch the range',
    rest: ({inputs, toUnit}) => {
      const [x, y] = [
        inputs.samples.positions[inputs.anchor * 2],
        inputs.samples.positions[inputs.anchor * 2 + 1]
      ];
      const direction = x < inputs.world.width / 2 ? 1 : -1;
      return toUnit([x + direction * 0.3, y]);
    },
    resolve(pointer, {inputs, values}) {
      const x = inputs.samples.positions[inputs.anchor * 2];
      const y = inputs.samples.positions[inputs.anchor * 2 + 1];
      const range = snapToParameter(
        findParameter(parameters, 'range'),
        Math.hypot(pointer.x - x, pointer.y - y)
      );
      return {
        values: {range},
        glyphs: [
          {shape: 'ring', at: [x, y], radius: range},
          {shape: 'dot', at: [x, y]}
        ],
        readout: `variogram.range ${range.toFixed(2)} · ${MODELS[values.model]}`
      };
    }
  },
  draw: ({inputs, outputs, values}) => [
    {
      layer: 'raster',
      values: values.output === 1 ? outputs.variance : outputs.prediction,
      columns: inputs.grid.columns,
      rows: inputs.grid.rows,
      range: values.output === 1 ? 'auto' : [0, 1],
      smoothing: true
    },
    ...drawSampleDots(outputs.positions, outputs.sampleValues)
  ]
});

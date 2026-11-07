// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPURasterTexture computes seven gray-level co-occurrence features (contrast, dissimilarity,
// homogeneity, ASM, energy, correlation, entropy) in a window around every cell. The scene is
// smooth on the left and rough on the right. The seven features of a cell are stored side by side,
// so GPURasterSampling reads the wide feature raster at the right column to pull out the feature
// the slider selects as a plain per-cell raster. Levels, radius and offset are construction props;
// roughness rewrites the scene.

import {
  getGPURasterSamplingParameterValues,
  GPURasterSampling,
  GPURasterTexture,
  makeGPURasterMetadata
} from '@luma-contributors/analysis/raster-analysis';
import {defineOperationRun} from '../../define-run';
import {parameters} from './glcm-texture.parameters';

const FEATURE_COUNT = 7;
const FEATURES = [
  'contrast',
  'dissimilarity',
  'homogeneity',
  'ASM',
  'energy',
  'correlation',
  'entropy'
];
const OFFSETS = [
  [1, 0],
  [0, 1],
  [1, 1],
  [-1, 1]
] as const;
const COLUMNS = 48;
const ROWS = 27;

export default defineOperationRun({
  parameters,
  rebuildOn: ['levels', 'radius', 'direction'],
  caption: 'A smooth-to-rough scene in; seven texture features per cell out.',
  createInputs: ({random}) => ({
    noise: Float32Array.from({length: COLUMNS * ROWS}, () => random())
  }),
  build(builder, inputs, values) {
    const cellCount = COLUMNS * ROWS;
    const scene = builder.allocate('scene', 'float32', cellCount);
    const features = builder.allocate('features', 'float32', cellCount * FEATURE_COUNT);
    const pairCount = builder.allocate('pair-count', 'uint32', cellCount);
    builder.graph.add(
      new GPURasterTexture({
        grid: makeGPURasterMetadata({width: COLUMNS, height: ROWS}),
        values: scene.view,
        levels: values.levels,
        range: [0, 1],
        radius: values.radius,
        offset: OFFSETS[Math.round(values.direction)],
        symmetric: true,
        features: features.view,
        pairCount: pairCount.view
      })
    );
    // The feature raster is COLUMNS * 7 wide; feature m of cell (column, row) is pixel
    // (column * 7 + m, row), so one sampling pass per feature gathers it into a per-cell raster.
    const wideGrid = makeGPURasterMetadata({
      width: COLUMNS * FEATURE_COUNT,
      height: ROWS,
      origin: [0, ROWS]
    });
    const samplingParameters = builder.upload(
      'sampling-parameters',
      'float32',
      getGPURasterSamplingParameterValues({method: 'nearest'}, wideGrid)
    );
    const gathered = FEATURES.map((name, feature) => {
      const positions = new Float32Array(cellCount * 2);
      for (let cell = 0; cell < cellCount; cell++) {
        positions[cell * 2] = (cell % COLUMNS) * FEATURE_COUNT + feature + 0.5;
        positions[cell * 2 + 1] = ROWS - Math.floor(cell / COLUMNS) - 0.5;
      }
      const output = builder.allocate(name, 'float32', cellCount);
      builder.graph.add(
        new GPURasterSampling({
          id: `gather-${name}`,
          grid: wideGrid,
          values: features.view,
          positions: builder.upload(`positions-${name}`, 'float32x2', positions).view,
          parameters: samplingParameters.view,
          output: {values: output.view}
        })
      );
      return output;
    });
    return {
      outputs: {scene, gathered},
      update: current =>
        scene.write(
          Float32Array.from(inputs.noise, (noise, index) => {
            // Texture grows from smooth (left) to rough (right); a gentle ramp underneath.
            const column = (index % COLUMNS) / (COLUMNS - 1);
            return Math.min(1, 0.3 + 0.4 * column + current.roughness * column * (noise - 0.5));
          })
        )
    };
  },
  draw: ({outputs, values}) => [
    {
      layer: 'raster',
      values: outputs.gathered[Math.round(values.feature)],
      columns: COLUMNS,
      rows: ROWS,
      ramp: Math.round(values.feature) === 5 ? 'diverging' : 'sequential',
      smoothing: false
    }
  ]
});

// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPURasterTexture computes seven gray-level co-occurrence features (contrast, dissimilarity,
// homogeneity, ASM, energy, correlation, entropy) in a window around every cell. The scene is
// smooth on the left and rough on the right. The seven features of a cell are stored side by side,
// so GPURasterSampling reads the wide feature raster at the right column to pull out the feature
// the slider selects as a plain per-cell raster. Levels, radius and offset are construction props;
// roughness rewrites the scene: the added noise grows from nothing at the left edge to full at the
// right. The pointer is the window: its (2 * radius + 1) cells square is outlined and the cell's
// features are read out; the left strip shows the input.

import {
  getGPURasterSamplingParameterValues,
  GPURasterSampling,
  GPURasterTexture,
  makeGPURasterMetadata
} from '@luma-contributors/analysis/raster-analysis';
import {defineOperationRun} from '../../define-run';
import {sampleReflectance} from '../../../data/reflectance';
import {formatNumber} from '../../interaction';
import {parameters} from './glcm-texture.parameters';
import {getCellAt, getCellBox} from './support/gestures';

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
const STRIP = 0.16;

export default defineOperationRun({
  parameters,
  rebuildOn: ['levels', 'radius', 'direction'],
  caption: 'Feather River near-infrared reflectance in; seven texture features per cell out.',
  createInputs: ({random}) => ({
    nir: sampleReflectance(COLUMNS, ROWS).nearInfrared,
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
      outputs: {scene, features, gathered},
      update: current =>
        scene.write(
          Float32Array.from(inputs.noise, (noise, index) => {
            const rough = (index % COLUMNS) / (COLUMNS - 1);
            return Math.max(
              0,
              Math.min(1, 2 * inputs.nir[index] + current.roughness * rough * (noise - 0.5))
            );
          })
        )
    };
  },
  // The shown feature's 2nd to 98th percentile fits the ramp, so a few extreme cells cannot flatten
  // the rest. Sequential features start below the minimum: the smooth side stays visible.
  async afterRun({outputs, values}) {
    const feature = Math.round(values.feature);
    const bytes = await outputs.gathered[feature].buffer.readAsync();
    const sorted = Float32Array.from(
      new Float32Array(bytes.buffer, bytes.byteOffset, COLUMNS * ROWS)
    )
      .filter(Number.isFinite)
      .sort();
    const low = sorted[Math.floor(sorted.length * 0.02)] ?? 0;
    const high = sorted[Math.floor(sorted.length * 0.98)] ?? 1;
    const span = Math.max(high - low, 1e-6);
    return {range: [low - span * 0.4, high] as const};
  },
  interaction: {
    kind: 'window',
    hint: 'move the texture window',
    rest: [0.5, 0.5],
    resolve: (pointer, {outputs, values, world, peek}) => {
      const grid = {columns: COLUMNS, rows: ROWS};
      const {column, row, index} = getCellAt(grid, world, pointer);
      const features = peek(outputs.features);
      const at = (feature: number) =>
        features ? formatNumber(features[index * FEATURE_COUNT + feature]) : '-';
      return {
        glyphs: [getCellBox(grid, world, column, row, values.radius)],
        readout: `contrast ${at(0)} | homogeneity ${at(2)} | entropy ${at(6)}`
      };
    }
  },
  draw: ({outputs, values, extra, world}) => [
    {
      layer: 'raster',
      values: outputs.scene,
      columns: COLUMNS,
      rows: ROWS,
      ramp: 'sequential',
      range: [0, 1],
      clip: {side: 'left', x: world.width * STRIP}
    },
    {
      layer: 'raster',
      values: outputs.gathered[Math.round(values.feature)],
      columns: COLUMNS,
      rows: ROWS,
      ramp: Math.round(values.feature) === 5 ? 'diverging' : 'sequential',
      range: Math.round(values.feature) === 5 ? 'auto' : (extra?.range ?? 'auto'),
      smoothing: false,
      clip: {side: 'right', x: world.width * STRIP}
    }
  ]
});

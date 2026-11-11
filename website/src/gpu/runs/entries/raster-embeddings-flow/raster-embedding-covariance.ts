// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPURasterEmbeddingCovariance: the mean vector and the 6 x 6 sample covariance of the usable
// pixels. Only pixels inside the box under the pointer are usable (`validity`), so moving the box
// over water, forest or burn scar shows how the bands co-vary there. The matrix is drawn in a side
// panel as a grid of dots (warm positive, cool negative), row 0 at the top; the readout names the
// strongest off-diagonal pair. Vectors are six standardized Sentinel-2 bands.

import {GPURasterEmbeddingCovariance} from '@luma-contributors/analysis/raster-analysis';
import {defineOperationRun} from '../../define-run';
import {createInsetBacking, getInsetBackingLayer} from '../space-time-models/_lib/inset';
import {
  BAND_NAMES,
  createEmbeddingRaster,
  DIMENSIONS,
  getPixelAt,
  getPixelBox
} from './_lib/embedding';
import {getPaddedRectangle, type InsetFractions} from './_lib/insets';
import {parameters} from './raster-embedding-covariance.parameters';

const REST = [0.32, 0.55] as const;
/** The S4 side panel; the window stays left of it. */
const PANEL: InsetFractions = [0.66, 0.16, 0.96, 0.84];
const MAP_LIMIT = 0.6;

export default defineOperationRun({
  parameters,
  caption:
    'Covariance of the vectors inside the box; the panel is the 6 x 6 matrix, bands blue to SWIR2 down and across. Six Sentinel-2 bands stand in for a learned embedding.',
  createInputs: ({aspect, random}) => {
    const raster = createEmbeddingRaster(aspect);
    const noise = Float32Array.from({length: raster.vectors.length}, () => random() * 2 - 1);
    const worldWidth = raster.columns / raster.rows;
    const panelWidth = (PANEL[2] - PANEL[0]) * worldWidth;
    const panelHeight = PANEL[3] - PANEL[1];
    const cell = Math.min(panelWidth, panelHeight) / DIMENSIONS;
    const left = PANEL[0] * worldWidth + (panelWidth - cell * DIMENSIONS) / 2;
    const top = PANEL[3] - (panelHeight - cell * DIMENSIONS) / 2;
    const matrixPositions = Float32Array.from({length: DIMENSIONS * DIMENSIONS * 2}, (_, item) => {
      const entry = item >> 1;
      return item & 1
        ? top - (Math.floor(entry / DIMENSIONS) + 0.5) * cell
        : left + ((entry % DIMENSIONS) + 0.5) * cell;
    });
    const center =
      Math.floor(REST[1] * raster.rows) * raster.columns + Math.floor(REST[0] * raster.columns);
    return {...raster, noise, matrixPositions, center, worldWidth};
  },
  build(builder, raster) {
    const vectors = builder.allocate('vectors', 'float32', raster.vectors.length);
    const validity = builder.allocate('validity', 'uint32', raster.pixelCount);
    const covariance = builder.allocate('covariance', 'float32', DIMENSIONS * DIMENSIONS);
    const usableCount = builder.allocate('usable-count', 'uint32', 1);
    builder.graph.add(
      new GPURasterEmbeddingCovariance({
        pixelCount: raster.pixelCount,
        dimensions: DIMENSIONS,
        vectors: vectors.view,
        mask: validity.view,
        output: {
          mean: builder.allocate('mean', 'float32', DIMENSIONS).view,
          covariance: covariance.view,
          usableCount: usableCount.view
        }
      })
    );
    return {
      outputs: {
        covariance,
        usableCount,
        nearInfrared: builder.upload('near-infrared', 'float32', raster.nearInfrared),
        matrixPositions: builder.upload('matrix-positions', 'float32x2', raster.matrixPositions),
        backing: createInsetBacking(builder, 'panel', getPaddedRectangle(PANEL, raster.worldWidth))
      },
      update: values => {
        vectors.write(
          raster.vectors.map((value, element) => value + values.noise * raster.noise[element])
        );
        const radius = Math.round(values.windowRadius);
        const column = raster.center % raster.columns;
        const row = Math.floor(raster.center / raster.columns);
        validity.write(
          Uint32Array.from({length: raster.pixelCount}, (_, pixel) =>
            Math.abs((pixel % raster.columns) - column) <= radius &&
            Math.abs(Math.floor(pixel / raster.columns) - row) <= radius
              ? 1
              : 0
          )
        );
      }
    };
  },
  interaction: {
    kind: 'window',
    hint: 'move the window',
    rest: REST,
    resolve(pointer, {inputs, outputs, values, world, peek}) {
      const radius = Math.round(values.windowRadius);
      const limit = MAP_LIMIT * world.width - (radius + 0.5) / inputs.rows;
      const {index} = getPixelAt(inputs, world, Math.min(pointer.x, limit), pointer.y);
      const moved = index !== inputs.center;
      inputs.center = index;
      const covariance = peek(outputs.covariance);
      const count = peek(outputs.usableCount)?.[0] ?? 0;
      let strongest = {row: 0, column: 1, value: Number.NaN};
      if (covariance) {
        for (let row = 0; row < DIMENSIONS; row++) {
          for (let column = row + 1; column < DIMENSIONS; column++) {
            const value = covariance[row * DIMENSIONS + column];
            if (!(Math.abs(value) <= Math.abs(strongest.value))) {
              strongest = {row, column, value};
            }
          }
        }
      }
      return {
        rerun: moved,
        spotlight: null,
        glyphs: [getPixelBox(inputs, world, index, radius)],
        readout: `strongest cov(${BAND_NAMES[strongest.row]}, ${BAND_NAMES[strongest.column]}) ${Number.isFinite(strongest.value) ? strongest.value.toFixed(3) : '-'} · usableCount ${count}`
      };
    }
  },
  draw: ({inputs, outputs, theme}) => [
    {
      layer: 'raster',
      context: true,
      values: outputs.nearInfrared,
      columns: inputs.columns,
      rows: inputs.rows,
      ramp: 'grey',
      smoothing: true
    },
    getInsetBackingLayer(outputs.backing, theme),
    {
      layer: 'dots',
      positions: outputs.matrixPositions,
      values: outputs.covariance,
      ramp: 'diverging',
      range: [-0.15, 0.15],
      radius: 16
    }
  ]
});

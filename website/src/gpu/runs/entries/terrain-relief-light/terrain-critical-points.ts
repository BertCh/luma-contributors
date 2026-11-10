// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUTerrainCriticalPoints on a Grand Canyon DEM with measurement noise: every cell is classified as
// regular, peak, pit, saddle or boundary from its neighbour sign changes, and a class histogram is
// reduced on the GPU. Connectivity is a construction prop; the noise slider rewrites the input.

import {GPUTerrainCriticalPoints} from '@luma-contributors/analysis/gpu-terrain';
import {defineOperationRun} from '../../define-run';
import {createMaskState, drawClassDimmer, writeClassMask} from './support/class-highlight';
import {
  addTerrainNoise,
  createTerrainInput,
  drawElevationContext,
  drawGrid,
  getCellOfWorldPosition,
  getCellWorldPosition,
  getWorldPerCell,
  uploadTerrain
} from './support/terrain';
import {parameters} from './terrain-critical-points.parameters';

// Index is the class code: regular, peak, pit, saddle, boundary, noData. Peaks are warm, pits
// cool, saddles gold; regular and unclassified cells stay clear.
const CLASS_COLORS = [
  [0, 0, 0, 0],
  [0.9, 0.2, 0.15, 1],
  [0.15, 0.4, 0.95, 1],
  [0.95, 0.75, 0.1, 1],
  [0, 0, 0, 0],
  [0, 0, 0, 0]
] as const;
const CLASS_NAMES = ['regular', 'peak', 'pit', 'saddle', 'boundary', 'no data'];

export default defineOperationRun({
  parameters,
  rebuildOn: ['connectivity'],
  caption: 'Noisy elevation in; peaks, pits and saddles per cell, and their counts, out.',
  createInputs: ({random, world}) => {
    const terrain = createTerrainInput(world, 48);
    return {
      ...terrain,
      noise: Float32Array.from(terrain.elevation, () => random()),
      maskState: createMaskState()
    };
  },
  build(builder, terrain, values) {
    const {grid, band, elevation, cellCount} = uploadTerrain(builder, terrain);
    const classes = builder.allocate('classes', 'uint32', cellCount);
    const counts = builder.allocate('counts', 'uint32', 6);
    const signChanges = builder.allocate('sign-changes', 'uint32', cellCount);
    const highlight = builder.allocate('highlight', 'float32', cellCount);
    builder.graph.add(
      new GPUTerrainCriticalPoints({
        grid,
        elevation: band,
        connectivity: values.connectivity === 1 ? 6 : 8,
        output: {classes: classes.view, counts: counts.view, signChanges: signChanges.view}
      })
    );
    return {
      outputs: {elevation, classes, counts, signChanges, highlight},
      update: current => elevation.write(addTerrainNoise(terrain, terrain.noise, current.noise))
    };
  },
  // The pointer is a cell: the nearest peak, pit or saddle within a few cells is picked, its 3x3
  // neighborhood is boxed, and every cell of that class stays lit.
  interaction: {
    kind: 'probe',
    hint: 'hover a peak or pit',
    rest: [0.5, 0.5],
    resolve(pointer, {inputs, outputs, world, peek}) {
      const [column, row] = getCellOfWorldPosition(inputs, world, pointer.x, pointer.y);
      const classes = peek(outputs.classes);
      const counts = peek(outputs.counts);
      const worldPerCell = getWorldPerCell(inputs, world);
      let picked = -1;
      if (classes) {
        let nearest = Infinity;
        for (let offsetRow = -4; offsetRow <= 4; offsetRow++) {
          for (let offsetColumn = -4; offsetColumn <= 4; offsetColumn++) {
            const candidateColumn = column + offsetColumn;
            const candidateRow = row + offsetRow;
            if (
              candidateColumn < 0 ||
              candidateRow < 0 ||
              candidateColumn >= inputs.columns ||
              candidateRow >= inputs.rows
            ) {
              continue;
            }
            const code = classes[candidateRow * inputs.columns + candidateColumn];
            const distance = Math.hypot(offsetColumn, offsetRow);
            if (code >= 1 && code <= 3 && distance < nearest) {
              nearest = distance;
              picked = candidateRow * inputs.columns + candidateColumn;
            }
          }
        }
      }
      if (!classes) return {spotlight: null, readout: 'peaks, pits, saddles'};
      if (picked < 0) {
        writeClassMask(outputs.highlight, classes, -1, inputs.maskState);
        return {
          spotlight: null,
          glyphs: [{shape: 'dot', at: [pointer.x, pointer.y]}],
          readout: counts
            ? `regular · ${counts[1]} peaks ${counts[2]} pits ${counts[3]} saddles`
            : 'regular'
        };
      }
      const code = classes[picked];
      writeClassMask(outputs.highlight, classes, code, inputs.maskState);
      const center = getCellWorldPosition(
        inputs,
        world,
        picked % inputs.columns,
        Math.floor(picked / inputs.columns)
      );
      const signChanges = peek(outputs.signChanges);
      const half = 1.5 * worldPerCell;
      return {
        spotlight: null,
        glyphs: [
          {
            shape: 'box',
            minimum: [center[0] - half, center[1] - half],
            maximum: [center[0] + half, center[1] + half]
          }
        ],
        readout: `${CLASS_NAMES[code]}${signChanges ? ` · ${signChanges[picked]} sign changes` : ''}${
          counts ? ` · ${counts[code]} cells` : ''
        }`
      };
    }
  },
  draw: ({inputs, outputs}) => [
    drawElevationContext(inputs, outputs.elevation),
    drawGrid(inputs, outputs.classes, {ramp: 'categorical', categories: CLASS_COLORS}),
    drawClassDimmer(inputs, outputs.highlight),
    {
      layer: 'chart',
      kind: 'bars',
      values: outputs.counts,
      rectangle: [0.62, 0.06, 0.97, 0.3],
      logarithmic: true,
      range: 'auto'
    }
  ]
});

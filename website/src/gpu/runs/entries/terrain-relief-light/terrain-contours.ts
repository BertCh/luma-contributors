// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUTerrainContours on a Grand Canyon DEM: one marching-squares polyline set per level. Every level
// is a GPU float read from one buffer, so the sliders move the contours without recompiling. A
// small kernel maps the pixel-space vertices the operation wrote into the drawing world.

import {GPUTerrainContours} from '@luma-contributors/analysis/gpu-terrain';
import {defineOperationRun} from '../../define-run';
import {findParameter, snapToParameter} from '../../interaction';
import {addPixelToWorld} from './support/pixel-to-world';
import {
  createTerrainInput,
  drawElevationContext,
  getCellOfWorldPosition,
  getCellWorldPosition,
  uploadTerrain
} from './support/terrain';
import {parameters} from './terrain-contours.parameters';

const LEVEL_COUNT = 10;
const SEGMENT_CAPACITY = 2400;

/** The first level that puts one of the levels exactly through `height`. */
function getFirstLevelThrough(height: number, interval: number): number {
  const first = findParameter(parameters, 'firstLevel');
  const offset = (((height - first.min) % interval) + interval) % interval;
  return snapToParameter(first, first.min + offset);
}

export default defineOperationRun({
  parameters,
  caption: 'Elevation in; one contour line set per level out, written straight to GPU buffers.',
  createInputs: ({world}) => ({...createTerrainInput(world), world}),
  build(builder, terrain) {
    const {grid, band, elevation} = uploadTerrain(builder, terrain);
    const levelBuffer = builder.allocate('levels', 'float32', LEVEL_COUNT);
    const overflow = builder.allocate('overflow', 'uint32', 1);
    const levels = Array.from({length: LEVEL_COUNT}, (_, index) => {
      const pixels = builder.allocate(`pixels-${index}`, 'float32x4', SEGMENT_CAPACITY);
      const count = builder.allocate(`count-${index}`, 'uint32', 1);
      const requiredCount = builder.allocate(`required-${index}`, 'uint32', 1);
      return {
        pixels,
        count,
        props: {
          level: builder.graph.createDataView(levelBuffer.view.buffer, {
            format: 'float32',
            length: 1,
            byteOffset: index * 4
          }),
          vertices: builder.graph.createDataView(pixels.view.buffer, {
            format: 'float32x2',
            length: SEGMENT_CAPACITY * 2
          }),
          status: {count: count.view, requiredCount: requiredCount.view}
        }
      };
    });
    builder.graph.add(
      new GPUTerrainContours({
        grid,
        elevation: band,
        levels: levels.map(level => level.props),
        output: {status: {overflow: overflow.view}}
      })
    );
    // Mapped after the operation: the kernel reads the pixel vertices it wrote.
    const drawn = levels.map((level, index) => ({
      vertices: addPixelToWorld(builder, `vertices-${index}`, level.pixels, terrain, terrain.world),
      count: level.count
    }));
    return {
      outputs: {elevation, levels: drawn},
      update: current =>
        levelBuffer.write(
          Float32Array.from(
            {length: LEVEL_COUNT},
            (_, index) => current.firstLevel + index * current.interval
          )
        )
    };
  },
  // The pointer is a height: the levels shift so that one contour passes through the cell under
  // the cursor, drawn bold. Moving over the canyon walls walks the contour set up and down.
  interaction: {
    kind: 'level',
    hint: 'trace a contour',
    rest: ({inputs, world}) => {
      // The cell nearest the center whose height puts a level at the default firstLevel.
      const first = findParameter(parameters, 'firstLevel');
      const interval = findParameter(parameters, 'interval').default;
      let best = [inputs.columns >> 1, inputs.rows >> 1];
      let bestDistance = Infinity;
      for (let row = 0; row < inputs.rows; row++) {
        for (let column = 0; column < inputs.columns; column++) {
          const height = inputs.elevation[row * inputs.columns + column];
          if (getFirstLevelThrough(height, interval) !== first.default) continue;
          const distance = Math.hypot(column - inputs.columns / 2, (row - inputs.rows / 2) * 1.5);
          if (distance < bestDistance) {
            bestDistance = distance;
            best = [column, row];
          }
        }
      }
      const [x, y] = getCellWorldPosition(inputs, world, best[0], best[1]);
      return [x / world.width, 1 - y / world.height];
    },
    resolve(pointer, {inputs, values, world}) {
      const [column, row] = getCellOfWorldPosition(inputs, world, pointer.x, pointer.y);
      const height = inputs.elevation[row * inputs.columns + column];
      const firstLevel = getFirstLevelThrough(height, values.interval);
      const focus = Math.round((height - firstLevel) / values.interval);
      return {
        values: {firstLevel},
        focus,
        glyphs: [{shape: 'dot', at: getCellWorldPosition(inputs, world, column, row)}],
        readout: `level ${Math.round(firstLevel + focus * values.interval)} m · interval ${values.interval} m`
      };
    }
  },
  draw: ({inputs, outputs, interaction}) => [
    drawElevationContext(inputs, outputs.elevation),
    ...outputs.levels.map((level, index) => ({
      layer: 'lines' as const,
      segments: level.vertices,
      count: level.count,
      radius: index === interaction.focus ? 3 : 1.4,
      color: [1 - index * 0.06, 0.8 - index * 0.05, 0.35 + index * 0.05, 1] as const
    }))
  ]
});

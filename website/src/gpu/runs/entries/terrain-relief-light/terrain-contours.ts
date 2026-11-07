// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUTerrainContours on a synthetic DEM: one marching-squares polyline set per level. Every level
// is a GPU float read from one buffer, so the sliders move the contours without recompiling. A
// small kernel maps the pixel-space vertices the operation wrote into the drawing world.

import {GPUTerrainContours} from '@luma-contributors/analysis/gpu-terrain';
import {defineOperationRun} from '../../define-run';
import {addPixelToWorld} from './support/pixel-to-world';
import {createTerrainInput, drawElevationContext, uploadTerrain} from './support/terrain';
import {parameters} from './terrain-contours.parameters';

const LEVEL_COUNT = 10;
const SEGMENT_CAPACITY = 2400;

export default defineOperationRun({
  parameters,
  caption: 'Elevation in; one contour line set per level out, written straight to GPU buffers.',
  createInputs: ({random, world}) => ({...createTerrainInput(random, world), world}),
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
  draw: ({inputs, outputs}) => [
    drawElevationContext(inputs, outputs.elevation),
    ...outputs.levels.map((level, index) => ({
      layer: 'lines' as const,
      segments: level.vertices,
      count: level.count,
      radius: index % 5 === 4 ? 1.6 : 0.9,
      color: [1 - index * 0.06, 0.78 - index * 0.04, 0.4 + index * 0.04, 1] as const
    }))
  ]
});

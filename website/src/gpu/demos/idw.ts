// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Demo: GPUInverseDistanceWeighting. Scattered samples become a surface; the distance power sweeps
// from smooth to spiky, and one sample follows the pointer so you can pull the surface around.

import {
  getGPUInverseDistanceWeightingParameterValues,
  GPUInverseDistanceWeighting
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {makeGPURasterMetadata} from '@luma-contributors/analysis/raster-analysis';
import type {GpuScene} from '../types';
import {
  assertWebGPU,
  createEmptyStorageBuffer,
  createStorageBuffer,
  DotLayer,
  drawLayers,
  getDemoTheme,
  getWorldPosition,
  GraphRunner,
  importBufferView,
  makeRandom,
  RasterLayer,
  type DemoView
} from './kit';

const ROWS = 90;
const SAMPLE_COUNT = 64;

const scene: GpuScene = async context => {
  const {device} = context;
  assertWebGPU(device);
  const theme = getDemoTheme(context.theme);
  const columns = Math.max(32, Math.min(200, Math.round((ROWS * context.width) / context.height)));
  const cellSize = 1 / ROWS;
  const worldWidth = columns * cellSize;
  const view: DemoView = {worldWidth, worldHeight: 1, width: context.width, height: context.height};
  const random = makeRandom(context.seed);
  const window = [0, 0, worldWidth, 1] as const;
  const grid = makeGPURasterMetadata({width: columns, height: ROWS, cellSize, origin: [0, 1]});

  // Sample 0 is the pointer; the rest sit on a jittered lattice so every cell has neighbors.
  const homes = new Float32Array(SAMPLE_COUNT * 2);
  for (let index = 1; index < SAMPLE_COUNT; index++) {
    const column = (index - 1) % 9;
    const row = Math.floor((index - 1) / 9);
    homes[index * 2] = ((column + 0.5 + (random() - 0.5) * 0.8) / 9) * worldWidth;
    homes[index * 2 + 1] = (row + 0.5 + (random() - 0.5) * 0.8) / 7;
  }
  const positions = Float32Array.from(homes);
  const values = new Float32Array(SAMPLE_COUNT);
  for (let index = 1; index < SAMPLE_COUNT; index++) {
    const x = homes[index * 2] / worldWidth;
    const y = homes[index * 2 + 1];
    values[index] = 0.5 + 0.35 * Math.sin(x * 6.5) * Math.cos(y * 5) + 0.25 * (random() - 0.5);
  }
  values[0] = 1.15;

  const positionBuffer = createStorageBuffer(device, positions);
  const valueBuffer = createStorageBuffer(device, values);
  const parameterBuffer = createEmptyStorageBuffer(device, 9);
  const surface = createEmptyStorageBuffer(device, columns * ROWS);

  const runner = await GraphRunner.create(device, 'demo-idw', graph => {
    graph.add(
      new GPUInverseDistanceWeighting({
        positions: importBufferView(graph, 'positions', positionBuffer, 'float32x2', SAMPLE_COUNT),
        values: importBufferView(graph, 'values', valueBuffer, 'float32', SAMPLE_COUNT),
        parameters: importBufferView(graph, 'parameters', parameterBuffer, 'float32', 9),
        grid,
        indexGridSize: [10, 8],
        indexBounds: window,
        output: {values: importBufferView(graph, 'surface', surface, 'float32', columns * ROWS)}
      })
    );
  });

  const raster = new RasterLayer(device, surface, columns, ROWS);
  const dots = new DotLayer(device, positionBuffer, valueBuffer, SAMPLE_COUNT);
  let pointer: [number, number] | null = null;

  return {
    render(timeMilliseconds) {
      const seconds = context.reducedMotion ? 3 : timeMilliseconds / 1000;
      const power = 1 + 3 * (0.5 + 0.5 * Math.sin(seconds * 0.7));
      for (let index = 1; index < SAMPLE_COUNT; index++) {
        positions[index * 2] = homes[index * 2] + Math.sin(seconds * 0.5 + index) * 0.012;
        positions[index * 2 + 1] =
          homes[index * 2 + 1] + Math.cos(seconds * 0.45 + index * 1.7) * 0.012;
      }
      const [x, y] = pointer ?? [
        worldWidth * (0.5 + 0.38 * Math.cos(seconds * 0.35)),
        0.5 + 0.34 * Math.sin(seconds * 0.5)
      ];
      positions[0] = x;
      positions[1] = y;
      positionBuffer.write(positions);
      parameterBuffer.write(
        getGPUInverseDistanceWeightingParameterValues(
          {searchRadius: 0.6, power, minimumNeighborCount: 1},
          grid
        )
      );
      runner.run();
      raster.update(view, theme, {range: [0, 1.15], smoothing: true});
      dots.update(view, theme, {mode: -1, radius: 3.4, solid: [...theme.ink, 0.95], hillshade: 1});
      drawLayers(device, theme, [raster, dots]);
    },
    resize(width, height) {
      view.width = width;
      view.height = height;
    },
    setPointer(position) {
      pointer = position ? getWorldPosition(view, position) : null;
    },
    destroy() {
      raster.destroy();
      dots.destroy();
      runner.destroy();
      for (const buffer of [positionBuffer, valueBuffer, parameterBuffer, surface]) {
        buffer.destroy();
      }
    }
  };
};

export default scene;

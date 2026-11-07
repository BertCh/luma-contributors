// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Demo: GPUTerrainDerivatives and GPUTerrainFlow. A seeded synthetic landscape, a sun that circles
// it (Horn hillshade, recomputed on the GPU), and a stream network extracted by depression
// filling, D8 routing and flow accumulation as the stream threshold breathes.

import {
  getGPUTerrainDerivativesParameterValues,
  getGPUTerrainFlowParameterValues,
  GPUTerrainDerivatives,
  GPUTerrainFlow
} from '@luma-contributors/analysis/gpu-terrain';
import {makeGPURasterMetadata} from '@luma-contributors/analysis/raster-analysis';
import type {GpuScene} from '../types';
import {
  assertWebGPU,
  createEmptyStorageBuffer,
  createStorageBuffer,
  drawLayers,
  getDemoTheme,
  GraphRunner,
  importBufferView,
  makeRandom,
  RasterLayer,
  type Color,
  type DemoView
} from './kit';

const ROWS = 96;
const CELL_SIZE = 60;
const LAND_RAMP: Color[] = ['#17323f', '#2c6556', '#6f9b55', '#c4b868', '#a98457', '#f1eee6'].map(
  value => {
    const number = Number.parseInt(value.slice(1), 16);
    return [
      ((number >> 16) & 255) / 255,
      ((number >> 8) & 255) / 255,
      (number & 255) / 255
    ] as Color;
  }
);

const scene: GpuScene = async context => {
  const {device} = context;
  assertWebGPU(device);
  const theme = getDemoTheme(context.theme);
  const columns = Math.max(48, Math.min(256, Math.round((ROWS * context.width) / context.height)));
  const worldWidth = columns / ROWS;
  const view: DemoView = {worldWidth, worldHeight: 1, width: context.width, height: context.height};
  const random = makeRandom(context.seed);
  const grid = makeGPURasterMetadata({
    width: columns,
    height: ROWS,
    cellSize: CELL_SIZE,
    origin: [0, ROWS * CELL_SIZE]
  });
  const cellCount = columns * ROWS;

  // Fractal landscape: octaves of random plane waves plus a few ridge-lines.
  const waves = Array.from({length: 22}, (_, index) => {
    const octave = Math.floor(index / 4);
    const angle = random() * Math.PI * 2;
    const frequency = (2.2 + octave * 2.1) * (0.8 + 0.4 * random());
    return {
      kx: Math.cos(angle) * frequency,
      ky: Math.sin(angle) * frequency,
      phase: random() * Math.PI * 2,
      amplitude: 260 / (1 + octave * 1.9)
    };
  });
  const elevation = new Float32Array(cellCount);
  for (let row = 0; row < ROWS; row++) {
    for (let column = 0; column < columns; column++) {
      const x = column / ROWS;
      const y = row / ROWS;
      let height = 0;
      for (const wave of waves) {
        height += wave.amplitude * Math.sin(wave.kx * x * 6 + wave.ky * y * 6 + wave.phase);
      }
      const edge = Math.min(1, 4 * Math.min(x, worldWidth - x, y, 1 - y) + 0.3);
      elevation[row * columns + column] = 420 + height * (0.55 + 0.45 * edge);
    }
  }
  let low = Infinity;
  let high = -Infinity;
  for (const value of elevation) {
    low = Math.min(low, value);
    high = Math.max(high, value);
  }

  const elevationBuffer = createStorageBuffer(device, elevation);
  const derivativeParameters = createEmptyStorageBuffer(device, 8);
  const flowParameters = createEmptyStorageBuffer(device, 8);
  const hillshade = createEmptyStorageBuffer(device, cellCount);
  const streams = createEmptyStorageBuffer(device, cellCount);

  const runner = await GraphRunner.create(device, 'demo-terrain-hydrology', graph => {
    const band = {
      id: 'elevation',
      format: 'float32',
      storage: {
        kind: 'buffer',
        values: importBufferView(graph, 'elevation', elevationBuffer, 'float32', cellCount)
      }
    } as const;
    graph.add(
      new GPUTerrainDerivatives({
        grid,
        elevation: band,
        parameters: importBufferView(
          graph,
          'derivative-parameters',
          derivativeParameters,
          'float32',
          8
        ),
        output: {hillshade: importBufferView(graph, 'hillshade', hillshade, 'float32', cellCount)}
      })
    );
    graph.add(
      new GPUTerrainFlow({
        grid,
        elevation: band,
        parameters: importBufferView(graph, 'flow-parameters', flowParameters, 'float32', 8),
        fillDepressions: true,
        output: {streams: importBufferView(graph, 'streams', streams, 'uint32', cellCount)}
      })
    );
  });

  const land = new RasterLayer(device, elevationBuffer, columns, ROWS);
  const shade = new RasterLayer(device, hillshade, columns, ROWS);
  const rivers = new RasterLayer(device, streams, columns, ROWS, 'uint32');
  const water = theme.categories[0];
  const riverColors = [
    [water[0], water[1], water[2], 0],
    [0.35, 0.8, 1, 1]
  ] as const;
  let lastKey = '';

  return {
    render(timeMilliseconds) {
      const seconds = context.reducedMotion ? 5 : timeMilliseconds / 1000;
      const azimuthDegrees = (seconds * 22) % 360;
      const threshold = 55 + 45 * (0.5 + 0.5 * Math.sin(seconds * 0.5));
      const key = `${azimuthDegrees.toFixed(1)}:${threshold.toFixed(0)}`;
      if (key !== lastKey) {
        lastKey = key;
        derivativeParameters.write(
          getGPUTerrainDerivativesParameterValues(
            {azimuthDegrees, altitudeDegrees: 38, zFactor: 1},
            grid
          )
        );
        flowParameters.write(
          getGPUTerrainFlowParameterValues(
            {streamThreshold: Math.round(threshold), fillEpsilon: 0.02},
            grid
          )
        );
        runner.run();
      }
      land.update(view, theme, {range: [low, high], ramp: LAND_RAMP, smoothing: true});
      // Hillshade 0..1 inverted into black with alpha: dark where the slope faces away from the sun.
      shade.update(view, theme, {
        range: [0.95, 0.3],
        ramp: [[0, 0, 0]],
        fade: true,
        alpha: 0.55,
        smoothing: true
      });
      rivers.update(view, theme, {mode: 2, categories: riverColors});
      drawLayers(device, theme, [land, shade, rivers]);
    },
    resize(width, height) {
      view.width = width;
      view.height = height;
    },
    destroy() {
      land.destroy();
      shade.destroy();
      rivers.destroy();
      runner.destroy();
      for (const buffer of [
        elevationBuffer,
        derivativeParameters,
        flowParameters,
        hillshade,
        streams
      ]) {
        buffer.destroy();
      }
    }
  };
};

export default scene;

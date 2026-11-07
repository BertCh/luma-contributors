// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Demo: GPUKernelDensity. A seeded point cloud, a bandwidth that sweeps back and forth, and the
// density surface drawn straight from the operation's output buffer.

import {
  GPUKernelDensity,
  getGPUKernelDensityParameterValues
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {makeGPURasterMetadata} from '@luma-contributors/analysis/raster-analysis';
import {Buffer} from '@luma.gl/core';
import type {GpuScene} from '../types';
import {
  assertWebGPU,
  createEmptyStorageBuffer,
  createStorageBuffer,
  DotLayer,
  drawLayers,
  drawGaussianCluster,
  getDemoTheme,
  GraphRunner,
  importBufferView,
  makeRandom,
  RasterLayer,
  type DemoView
} from './kit';

const ROWS = 72;
const CLUSTER_CENTERS = [
  [0.28, 0.35, 0.07, 150],
  [0.62, 0.62, 0.05, 130],
  [0.78, 0.28, 0.09, 110],
  [0.45, 0.78, 0.06, 70]
] as const;

const scene: GpuScene = async context => {
  const {device} = context;
  assertWebGPU(device);
  const theme = getDemoTheme(context.theme);
  const columns = Math.max(32, Math.min(160, Math.round((ROWS * context.width) / context.height)));
  const cellSize = 1 / ROWS;
  const worldWidth = columns * cellSize;
  const view: DemoView = {worldWidth, worldHeight: 1, width: context.width, height: context.height};
  const random = makeRandom(context.seed);

  const positions: number[] = [];
  for (const [centerX, centerY, spread, count] of CLUSTER_CENTERS) {
    for (let index = 0; index < count; index++) {
      const [x, y] = drawGaussianCluster(random, centerX * worldWidth, centerY, spread);
      positions.push(x, y);
    }
  }
  const pointCount = positions.length / 2;

  const grid = makeGPURasterMetadata({width: columns, height: ROWS, cellSize, origin: [0, 1]});
  const window = [0, 0, worldWidth, 1] as const;
  const positionBuffer = createStorageBuffer(device, Float32Array.from(positions));
  const densityBuffer = createEmptyStorageBuffer(device, columns * ROWS);
  const parameterBuffer = createEmptyStorageBuffer(device, 9);
  const pointValues = createEmptyStorageBuffer(device, pointCount);

  const runner = await GraphRunner.create(device, 'demo-kernel-density', graph => {
    graph.add(
      new GPUKernelDensity({
        id: 'demo-kernel-density',
        positions: importBufferView(graph, 'positions', positionBuffer, 'float32x2', pointCount),
        parameters: importBufferView(graph, 'parameters', parameterBuffer, 'float32', 9),
        grid,
        indexGridSize: [24, 24],
        indexBounds: window,
        kernel: 'gaussian',
        edgeCorrection: 'none',
        method: 'exact',
        output: {
          values: importBufferView(graph, 'density', densityBuffer, 'float32', columns * ROWS)
        }
      })
    );
  });

  const raster = new RasterLayer(device, densityBuffer, columns, ROWS);
  const dots = new DotLayer(device, positionBuffer, pointValues, pointCount);
  let peak = 1;
  let lastBandwidth = 0;

  return {
    render(timeMilliseconds) {
      const phase = context.reducedMotion ? 0.35 : 0.5 + 0.5 * Math.sin(timeMilliseconds / 2600);
      const bandwidth = 0.025 + 0.085 * phase;
      if (Math.abs(bandwidth - lastBandwidth) > 1e-4 || lastBandwidth === 0) {
        lastBandwidth = bandwidth;
        parameterBuffer.write(getGPUKernelDensityParameterValues({bandwidth, window}, grid));
        runner.run();
        // Peak density falls as the bandwidth grows; keep a fixed color scale relative to it.
        peak = Math.max(
          ...CLUSTER_CENTERS.map(
            ([, , spread, count]) =>
              count / (2 * Math.PI * (spread * spread + bandwidth * bandwidth))
          )
        );
      }
      raster.update(view, theme, {range: [0, peak * 1.15], smoothing: true, alpha: 0.95});
      dots.update(view, theme, {mode: -1, radius: 2.2, alpha: 0.75, solid: [...theme.ink, 0.55]});
      drawLayers(device, theme, [raster, dots]);
    },
    resize(width, height) {
      view.width = width;
      view.height = height;
    },
    destroy() {
      raster.destroy();
      dots.destroy();
      runner.destroy();
      for (const buffer of [
        positionBuffer,
        densityBuffer,
        parameterBuffer,
        pointValues
      ] as Buffer[]) {
        buffer.destroy();
      }
    }
  };
};

export default scene;

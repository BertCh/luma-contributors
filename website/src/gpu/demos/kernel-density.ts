// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Demo: GPUKernelDensity. A seeded point cloud, a bandwidth slider, and the density surface drawn
// straight from the operation's output buffer. A small cluster of points follows the pointer.

import {
  GPUKernelDensity,
  getGPUKernelDensityParameterValues
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import type {GPUKernelDensityKernel} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {makeGPURasterMetadata} from '@luma-contributors/analysis/raster-analysis';
import type {GpuScene} from '../types';
import {KERNEL_DENSITY_PARAMETERS, readParameterValues} from './parameters';
import {
  assertWebGPU,
  createEmptyStorageBuffer,
  createStorageBuffer,
  DotLayer,
  drawGaussianCluster,
  DrawScheduler,
  drawLayers,
  getDemoTheme,
  getWorldPosition,
  GraphRunner,
  importBufferView,
  makeRandom,
  RasterLayer,
  type DemoView
} from './kit';

export const parameters = KERNEL_DENSITY_PARAMETERS;

const ROWS = 72;
const HOTSPOT_COUNT = 60;
const HOTSPOT_SPREAD = 0.03;
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
  const values = readParameterValues(parameters, context.parameters);
  const columns = Math.max(32, Math.min(160, Math.round((ROWS * context.width) / context.height)));
  const cellSize = 1 / ROWS;
  const worldWidth = columns * cellSize;
  const view: DemoView = {worldWidth, worldHeight: 1, width: context.width, height: context.height};
  const random = makeRandom(context.seed);

  let baseCount = 0;
  for (const [, , , count] of CLUSTER_CENTERS) baseCount += count;
  const pointCount = baseCount + HOTSPOT_COUNT;
  const positions = new Float32Array(pointCount * 2);
  let cursor = 0;
  for (const [centerX, centerY, spread, count] of CLUSTER_CENTERS) {
    for (let index = 0; index < count; index++) {
      const [x, y] = drawGaussianCluster(random, centerX * worldWidth, centerY, spread);
      positions[cursor++] = x;
      positions[cursor++] = y;
    }
  }
  const hotspotOffsets = new Float32Array(HOTSPOT_COUNT * 2);
  for (let index = 0; index < HOTSPOT_COUNT; index++) {
    const [x, y] = drawGaussianCluster(random, 0, 0, HOTSPOT_SPREAD);
    hotspotOffsets[index * 2] = x;
    hotspotOffsets[index * 2 + 1] = y;
  }
  const hotspotCenter: [number, number] = [0.12 * worldWidth, 0.84];
  const writeHotspot = () => {
    for (let index = 0; index < HOTSPOT_COUNT; index++) {
      positions[(baseCount + index) * 2] = hotspotCenter[0] + hotspotOffsets[index * 2];
      positions[(baseCount + index) * 2 + 1] = hotspotCenter[1] + hotspotOffsets[index * 2 + 1];
    }
  };
  writeHotspot();

  const grid = makeGPURasterMetadata({width: columns, height: ROWS, cellSize, origin: [0, 1]});
  const window = [0, 0, worldWidth, 1] as const;
  const positionBuffer = createStorageBuffer(device, positions);
  const densityBuffer = createEmptyStorageBuffer(device, columns * ROWS);
  const parameterBuffer = createEmptyStorageBuffer(device, 9);
  const pointValues = createEmptyStorageBuffer(device, pointCount);

  // The kernel is compiled into the shader, so each choice is its own compiled graph.
  const kernels = KERNEL_DENSITY_PARAMETERS.find(parameter => parameter.id === 'kernel')
    ?.options as readonly GPUKernelDensityKernel[];
  const runners = await Promise.all(
    kernels.map(kernel =>
      GraphRunner.create(device, `demo-kernel-density-${kernel}`, graph => {
        graph.add(
          new GPUKernelDensity({
            id: `demo-kernel-density-${kernel}`,
            positions: importBufferView(
              graph,
              'positions',
              positionBuffer,
              'float32x2',
              pointCount
            ),
            parameters: importBufferView(graph, 'parameters', parameterBuffer, 'float32', 9),
            grid,
            indexGridSize: [24, 24],
            indexBounds: window,
            kernel,
            edgeCorrection: 'none',
            method: 'exact',
            output: {
              values: importBufferView(graph, 'density', densityBuffer, 'float32', columns * ROWS)
            }
          })
        );
      })
    )
  );

  const raster = new RasterLayer(device, densityBuffer, columns, ROWS);
  const dots = new DotLayer(device, positionBuffer, pointValues, pointCount);
  let isDirty = true;

  const draw = () => {
    if (isDirty) {
      isDirty = false;
      const bandwidth = values.bandwidth / 100;
      parameterBuffer.write(getGPUKernelDensityParameterValues({bandwidth, window}, grid));
      positionBuffer.write(positions);
      runners[Math.round(values.kernel)].run();
    }
    // Peak density falls as the bandwidth grows; the color scale follows it.
    // Compact kernels have less variance than a gaussian of the same bandwidth: h^2 / 4 and h^2 / 6.
    const kernelVariance = [1, 1 / 6, 1 / 4][Math.round(values.kernel)];
    const bandwidth = (values.bandwidth / 100) * Math.sqrt(kernelVariance);
    const peak = Math.max(
      ...CLUSTER_CENTERS.map(
        ([, , spread, count]) => count / (2 * Math.PI * (spread * spread + bandwidth * bandwidth))
      ),
      HOTSPOT_COUNT / (2 * Math.PI * (HOTSPOT_SPREAD ** 2 + bandwidth * bandwidth))
    );
    raster.update(view, theme, {range: [0, peak * 1.15], smoothing: true, alpha: 0.95});
    dots.update(view, theme, {mode: -1, radius: 2.2, alpha: 0.75, solid: [...theme.ink, 0.55]});
    drawLayers(device, theme, [raster, dots]);
  };
  const scheduler = new DrawScheduler(draw);

  return {
    render() {
      scheduler.flush();
    },
    resize(width, height) {
      view.width = width;
      view.height = height;
      scheduler.request();
    },
    setPointer(position) {
      if (!position) return;
      const [x, y] = getWorldPosition(view, position);
      hotspotCenter[0] = x;
      hotspotCenter[1] = y;
      writeHotspot();
      isDirty = true;
      scheduler.request();
    },
    setParameters(update) {
      Object.assign(values, update);
      isDirty = true;
      scheduler.request();
    },
    destroy() {
      scheduler.cancel();
      raster.destroy();
      dots.destroy();
      for (const runner of runners) runner.destroy();
      for (const buffer of [positionBuffer, densityBuffer, parameterBuffer, pointValues]) {
        buffer.destroy();
      }
    }
  };
};

export default scene;

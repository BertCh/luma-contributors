// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Demo: GPUKernelDensity of sampled Japanese earthquake epicenters. The pointer's horizontal
// position sets the kernel bandwidth, and a ring at the pointer shows the kernel footprint.

import {
  GPUKernelDensity,
  getGPUKernelDensityParameterValues
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import type {GPUKernelDensityKernel} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {makeGPURasterMetadata} from '@luma-contributors/analysis/raster-analysis';
import {getEarthquakes} from '../data/point-observations';
import {getFractionOf, getParameterAt} from '../runs/interaction';
import {createPointerDriver} from '../runs/pointer-driver';
import type {GpuScene} from '../types';
import {createJapanCoast} from './japan-coast';
import {KERNEL_DENSITY_PARAMETERS, readParameterValues} from './parameters';
import {
  assertWebGPU,
  createEmptyStorageBuffer,
  createStorageBuffer,
  DotLayer,
  DrawScheduler,
  drawLayers,
  getDemoTheme,
  GraphRunner,
  importBufferView,
  makeRandom,
  RasterLayer,
  type DemoView
} from './kit';

export const parameters = KERNEL_DENSITY_PARAMETERS;

const ROWS = 72;
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

  const {positions, count: pointCount} = getEarthquakes(
    random,
    {width: worldWidth, height: 1},
    800
  );
  let pointerBandwidth: number | null = null;

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
  const coast = createJapanCoast(device, worldWidth);
  const dots = new DotLayer(device, positionBuffer, pointValues, pointCount);
  let isDirty = true;

  const draw = () => {
    if (isDirty) {
      isDirty = false;
      const bandwidth = (pointerBandwidth ?? values.bandwidth) / 100;
      parameterBuffer.write(getGPUKernelDensityParameterValues({bandwidth, window}, grid));
      runners[Math.round(values.kernel)].run();
    }
    const bandwidth = (pointerBandwidth ?? values.bandwidth) / 100;
    // Display scale follows the effective kernel area; narrow bandwidths expose individual swarms.
    const kernelScale = [1, 1.6, 1.4][Math.round(values.kernel)];
    const peak = (kernelScale * 320) / (0.025 + bandwidth);
    raster.update(view, theme, {range: [0, peak], smoothing: true, alpha: 0.95});
    dots.update(view, theme, {mode: -1, radius: 2.2, alpha: 0.75, solid: [...theme.ink, 0.55]});
    coast.layer.update(view, theme, {radius: 0.7, solid: [...theme.muted, 0.8]});
    drawLayers(device, theme, [raster, coast.layer, dots, ...driver.getLayers()]);
  };
  const scheduler = new DrawScheduler(draw);

  const bandwidthParameter = KERNEL_DENSITY_PARAMETERS[0];
  const driver = createPointerDriver({
    device,
    view,
    theme,
    reducedMotion: context.reducedMotion,
    kind: 'reach',
    hint: 'size the kernel',
    rest: [getFractionOf(bandwidthParameter, bandwidthParameter.default, 'logarithmic'), 0.5],
    reportReadout: context.reportReadout,
    requestDraw: () => scheduler.request(),
    resolve(pointer) {
      const bandwidth = getParameterAt(bandwidthParameter, pointer.unitX, 'logarithmic');
      if (bandwidth !== pointerBandwidth) {
        pointerBandwidth = bandwidth;
        isDirty = true;
        scheduler.request();
        context.reportParameters?.({bandwidth});
      }
      const at = [pointer.x, pointer.y] as const;
      return {
        glyphs: [
          {shape: 'ring', at, radius: bandwidth / 100},
          {shape: 'crosshair', at}
        ],
        readout: `bandwidth ${bandwidth.toFixed(1)}%`
      };
    }
  });

  return {
    gesture: driver.gesture,
    render(time) {
      driver.advance(time);
      scheduler.flush();
    },
    resize(width, height) {
      view.width = width;
      view.height = height;
      driver.refresh();
      scheduler.request();
    },
    setPointer: driver.setPointer,
    resetPointer: driver.resetPointer,
    setParameters(update) {
      Object.assign(values, update);
      if (update.bandwidth !== undefined && update.bandwidth !== pointerBandwidth) {
        pointerBandwidth = null;
        driver.release();
      }
      isDirty = true;
      scheduler.request();
    },
    destroy() {
      scheduler.cancel();
      driver.destroy();
      raster.destroy();
      coast.destroy();
      dots.destroy();
      for (const runner of runners) runner.destroy();
      for (const buffer of [positionBuffer, densityBuffer, parameterBuffer, pointValues]) {
        buffer.destroy();
      }
    }
  };
};

export default scene;

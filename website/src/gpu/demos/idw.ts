// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Demo: GPUInverseDistanceWeighting. Scattered samples become a surface; power and search
// radius are sliders. The pointer is a query location: the gauges that carry its estimate are
// linked to it, so the reader sees what "inverse distance weighting" weighs.

import {
  getGPUInverseDistanceWeightingParameterValues,
  GPUInverseDistanceWeighting
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {makeGPURasterMetadata} from '@luma-contributors/analysis/raster-analysis';
import {getRainStations} from '../data/point-observations';
import type {RunGlyph} from '../runs/interaction';
import {createPointerDriver} from '../runs/pointer-driver';
import type {GpuScene} from '../types';
import {IDW_PARAMETERS, readParameterValues} from './parameters';
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

export const parameters = IDW_PARAMETERS;

const ROWS = 90;
const SAMPLE_COUNT = 64;
/** Gauges that carry at least this share of the estimate are linked to the query point. */
const LINKED_SHARE = 0.04;

const scene: GpuScene = async context => {
  const {device} = context;
  assertWebGPU(device);
  const theme = getDemoTheme(context.theme);
  const parameterValues = readParameterValues(parameters, context.parameters);
  const columns = Math.max(32, Math.min(200, Math.round((ROWS * context.width) / context.height)));
  const cellSize = 1 / ROWS;
  const worldWidth = columns * cellSize;
  const view: DemoView = {worldWidth, worldHeight: 1, width: context.width, height: context.height};
  const random = makeRandom(context.seed);
  const window = [0, 0, worldWidth, 1] as const;
  const grid = makeGPURasterMetadata({width: columns, height: ROWS, cellSize, origin: [0, 1]});

  const {positions, rain: values} = getRainStations(
    random,
    {width: worldWidth, height: 1},
    SAMPLE_COUNT,
    true
  );
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
  let isDirty = true;
  const draw = () => {
    if (isDirty) {
      isDirty = false;
      parameterBuffer.write(
        getGPUInverseDistanceWeightingParameterValues(
          {
            searchRadius: parameterValues.searchRadius / 100,
            power: parameterValues.power,
            minimumNeighborCount: 1
          },
          grid
        )
      );
      runner.run();
    }
    raster.update(view, theme, {range: [0, 180], smoothing: true});
    dots.update(view, theme, {mode: -1, radius: 3.4, solid: [...theme.ink, 0.95], hillshade: 1});
    drawLayers(device, theme, [raster, dots, ...driver.getLayers()]);
  };
  const scheduler = new DrawScheduler(draw);

  // The same weighting the operation applies to every cell, for the one location under the pointer.
  const driver = createPointerDriver({
    device,
    view,
    theme,
    reducedMotion: context.reducedMotion,
    kind: 'place',
    hint: 'move the query point',
    rest: [0.5, 0.5],
    reportReadout: context.reportReadout,
    requestDraw: () => scheduler.request(),
    resolve(pointer) {
      const searchRadius = parameterValues.searchRadius / 100;
      const weights: {index: number; weight: number}[] = [];
      let weightSum = 0;
      let weightedRain = 0;
      for (let index = 0; index < SAMPLE_COUNT; index++) {
        const distance = Math.hypot(
          positions[index * 2] - pointer.x,
          positions[index * 2 + 1] - pointer.y
        );
        if (distance > searchRadius) continue;
        const weight = 1 / Math.pow(Math.max(distance, 1e-4), parameterValues.power);
        weights.push({index, weight});
        weightSum += weight;
        weightedRain += weight * values[index];
      }
      const query = [pointer.x, pointer.y] as const;
      const glyphs: RunGlyph[] = [
        {shape: 'ring', at: query, radius: searchRadius, dashed: true},
        {shape: 'crosshair', at: query}
      ];
      let linkedCount = 0;
      for (const {index, weight} of weights) {
        if (weight / weightSum < LINKED_SHARE) continue;
        glyphs.push({
          shape: 'line',
          from: query,
          to: [positions[index * 2], positions[index * 2 + 1]]
        });
        linkedCount++;
      }
      return {
        glyphs,
        readout:
          weights.length > 0
            ? `${(weightedRain / weightSum).toFixed(0)} mm · ${linkedCount} of ${weights.length} gauges carry it`
            : 'no gauge within the search radius'
      };
    }
  });

  return {
    gesture: driver.gesture,
    render(time) {
      driver.advance(time);
      scheduler.flush();
    },
    setParameters(update) {
      Object.assign(parameterValues, update);
      isDirty = true;
      // The links and the estimate depend on power and search radius.
      driver.refresh();
      scheduler.request();
    },
    resize(width, height) {
      view.width = width;
      view.height = height;
      isDirty = true;
      driver.refresh();
      scheduler.request();
    },
    setPointer: driver.setPointer,
    resetPointer: driver.resetPointer,
    destroy() {
      scheduler.cancel();
      driver.destroy();
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

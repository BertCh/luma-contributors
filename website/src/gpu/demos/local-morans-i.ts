// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Demo: GPUNeighborSearch feeding GPULocalMoran. A drifting spatial field with noise, neighbors
// within a radius, and the HH, LL, LH and HL clusters the GPU finds above a significance level.

import {
  getGPUNeighborSearchParameterValues,
  getGPUSpatialAutocorrelationParameterValues,
  GPULocalMoran,
  GPUNeighborSearch,
  GPU_NEIGHBOR_SEARCH_PARAMETER_LENGTH,
  GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import type {GpuScene} from '../types';
import {LOCAL_MORANS_I_PARAMETERS, readParameterValues} from './parameters';
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
  type DemoView
} from './kit';

export const parameters = LOCAL_MORANS_I_PARAMETERS;

const POINT_COUNT = 1100;
const SLOTS_PER_POINT = 48;

const scene: GpuScene = async context => {
  const {device} = context;
  assertWebGPU(device);
  const theme = getDemoTheme(context.theme);
  const values = readParameterValues(parameters, context.parameters);
  const worldWidth = context.width / context.height;
  const view: DemoView = {worldWidth, worldHeight: 1, width: context.width, height: context.height};
  const random = makeRandom(context.seed);
  const bounds = [0, 0, worldWidth, 1] as const;

  const positions = new Float32Array(POINT_COUNT * 2);
  const noise = new Float32Array(POINT_COUNT);
  for (let index = 0; index < POINT_COUNT; index++) {
    positions[index * 2] = random() * worldWidth;
    positions[index * 2 + 1] = random();
    noise[index] = random() - 0.5;
  }
  const fieldValues = new Float32Array(POINT_COUNT);
  const writeValues = (phase: number) => {
    for (let index = 0; index < POINT_COUNT; index++) {
      const x = positions[index * 2] / worldWidth;
      const y = positions[index * 2 + 1];
      fieldValues[index] =
        Math.sin(x * 7 + phase) * Math.cos(y * 5 - phase * 0.7) +
        0.35 * Math.sin(x * 19 - y * 13) +
        noise[index] * values.noise;
    }
    valueBuffer.write(fieldValues);
  };

  const slotCount = POINT_COUNT * SLOTS_PER_POINT;
  const positionBuffer = createStorageBuffer(device, positions);
  const valueBuffer = createEmptyStorageBuffer(device, POINT_COUNT);
  const neighborParameters = createEmptyStorageBuffer(device, GPU_NEIGHBOR_SEARCH_PARAMETER_LENGTH);
  const moranParameters = createEmptyStorageBuffer(
    device,
    GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH
  );
  const offsets = createEmptyStorageBuffer(device, POINT_COUNT + 1);
  const neighbors = createEmptyStorageBuffer(device, slotCount);
  const weights = createEmptyStorageBuffer(device, slotCount);
  const overflow = createEmptyStorageBuffer(device, 1);
  const zScores = createEmptyStorageBuffer(device, POINT_COUNT);
  const quadrants = createEmptyStorageBuffer(device, POINT_COUNT);
  const globalStatistics = createEmptyStorageBuffer(device, 5);

  const runner = await GraphRunner.create(device, 'demo-local-morans-i', graph => {
    const slots = {
      offsets: importBufferView(graph, 'offsets', offsets, 'uint32', POINT_COUNT + 1),
      neighbors: importBufferView(graph, 'neighbors', neighbors, 'uint32', slotCount),
      weights: importBufferView(graph, 'weights', weights, 'float32', slotCount)
    };
    graph.add(
      new GPUNeighborSearch({
        mode: 'radius',
        gridSize: [32, 32],
        positions: importBufferView(graph, 'positions', positionBuffer, 'float32x2', POINT_COUNT),
        parameters: importBufferView(
          graph,
          'neighbor-parameters',
          neighborParameters,
          'float32',
          GPU_NEIGHBOR_SEARCH_PARAMETER_LENGTH
        ),
        output: {
          ...slots,
          status: {overflow: importBufferView(graph, 'overflow', overflow, 'uint32', 1)}
        }
      })
    );
    graph.add(
      new GPULocalMoran({
        weights: slots,
        values: importBufferView(graph, 'values', valueBuffer, 'float32', POINT_COUNT),
        parameters: importBufferView(
          graph,
          'moran-parameters',
          moranParameters,
          'float32',
          GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH
        ),
        minimumNeighbors: 3,
        output: {
          zScores: importBufferView(graph, 'z-scores', zScores, 'float32', POINT_COUNT),
          quadrants: importBufferView(graph, 'quadrants', quadrants, 'uint32', POINT_COUNT),
          globalStatistics: importBufferView(
            graph,
            'global-statistics',
            globalStatistics,
            'float32',
            5
          )
        }
      })
    );
  });

  // Quadrant colors: not significant, HH, LH, LL, HL.
  const quadrantColors: (readonly [number, number, number, number])[] =
    context.theme === 'dark'
      ? [
          [...theme.muted, 0.45],
          [1, 0.36, 0.34, 1],
          [0.55, 0.76, 1, 1],
          [0.25, 0.5, 1, 1],
          [1, 0.72, 0.3, 1]
        ]
      : [
          [...theme.muted, 0.5],
          [0.82, 0.16, 0.2, 1],
          [0.45, 0.65, 0.95, 1],
          [0.12, 0.33, 0.8, 1],
          [0.9, 0.5, 0.05, 1]
        ];

  const field = new DotLayer(device, positionBuffer, valueBuffer, POINT_COUNT);
  const clusters = new DotLayer(device, positionBuffer, quadrants, POINT_COUNT, 'uint32');
  let isDirty = true;
  let seconds = context.reducedMotion ? 2 : 0;

  const draw = () => {
    if (isDirty) {
      isDirty = false;
      writeValues(seconds * 0.45);
      neighborParameters.write(
        getGPUNeighborSearchParameterValues({
          bounds,
          radius: values.radius / 100,
          weightKind: 'binary'
        })
      );
      moranParameters.write(
        getGPUSpatialAutocorrelationParameterValues({significanceLevel: values.significance})
      );
      runner.run();
    }
    field.update(view, theme, {
      mode: 1,
      range: [-1.6, 1.6],
      radius: 11,
      alpha: 0.2,
      ramp: theme.diverging
    });
    clusters.update(view, theme, {
      mode: 2,
      categories: quadrantColors,
      radius: 4,
      solid: [...theme.muted, 0.4]
    });
    drawLayers(device, theme, [field, clusters]);
  };
  const scheduler = new DrawScheduler(draw);

  return {
    render(timeMilliseconds) {
      if (!context.reducedMotion) {
        // The field drifts slowly; recompute when it has moved a visible step.
        const nextSeconds = timeMilliseconds / 1000;
        if (Math.abs(nextSeconds - seconds) > 0.03) {
          seconds = nextSeconds;
          isDirty = true;
        }
      }
      scheduler.flush();
    },
    setParameters(update) {
      Object.assign(values, update);
      isDirty = true;
      scheduler.request();
    },
    resize(width, height) {
      view.width = width;
      view.height = height;
      scheduler.request();
    },
    destroy() {
      scheduler.cancel();
      field.destroy();
      clusters.destroy();
      runner.destroy();
      for (const buffer of [
        positionBuffer,
        valueBuffer,
        neighborParameters,
        moranParameters,
        offsets,
        neighbors,
        weights,
        overflow,
        zScores,
        quadrants,
        globalStatistics
      ]) {
        buffer.destroy();
      }
    }
  };
};

export default scene;

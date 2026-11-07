// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Demo: GPUNeighborSearch feeding GPUHotSpotAnalysis (Getis-Ord Gi*). Noisy points carry a few hidden
// hot and cold patches; the GPU's z-scores light them up. A hot patch follows the pointer, and
// sliders set the neighbor radius and the strength of the hidden pattern.

import {
  getGPUNeighborSearchParameterValues,
  getGPUSpatialAutocorrelationParameterValues,
  GPUHotSpotAnalysis,
  GPUNeighborSearch,
  GPU_NEIGHBOR_SEARCH_PARAMETER_LENGTH,
  GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import type {GpuScene} from '../types';
import {HOT_SPOTS_PARAMETERS, readParameterValues} from './parameters';
import {
  assertWebGPU,
  createEmptyStorageBuffer,
  createStorageBuffer,
  DotLayer,
  DrawScheduler,
  drawLayers,
  getDemoTheme,
  getWorldPosition,
  GraphRunner,
  importBufferView,
  makeRandom,
  type DemoView
} from './kit';

export const parameters = HOT_SPOTS_PARAMETERS;

const POINT_COUNT = 1400;
const SLOTS_PER_POINT = 48;

const scene: GpuScene = async context => {
  const {device} = context;
  assertWebGPU(device);
  const theme = getDemoTheme(context.theme);
  const settings = readParameterValues(parameters, context.parameters);
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
  // Hidden patches: center x (fraction of width), center y, radius, signed amplitude.
  const patches = [
    [0.22, 0.65, 0.14, 1],
    [0.7, 0.3, 0.12, 1],
    [0.5, 0.75, 0.1, -1],
    [0.86, 0.78, 0.1, -1],
    [0.34, 0.22, 0.09, -1]
  ] as const;
  let pointer: [number, number] | null = null;
  const fieldValues = new Float32Array(POINT_COUNT);
  const writeValues = () => {
    for (let index = 0; index < POINT_COUNT; index++) {
      const x = positions[index * 2];
      const y = positions[index * 2 + 1];
      let value = noise[index] * 1.6;
      for (const [centerX, centerY, radius, sign] of patches) {
        const distance = Math.hypot(x - centerX * worldWidth, y - centerY) / radius;
        value += sign * settings.strength * 2.2 * Math.exp(-distance * distance);
      }
      if (pointer) {
        const distance = Math.hypot(x - pointer[0], y - pointer[1]) / 0.1;
        value += 2.6 * Math.exp(-distance * distance);
      }
      fieldValues[index] = value;
    }
    valueBuffer.write(fieldValues);
  };

  const slotCount = POINT_COUNT * SLOTS_PER_POINT;
  const positionBuffer = createStorageBuffer(device, positions);
  const valueBuffer = createEmptyStorageBuffer(device, POINT_COUNT);
  const neighborParameters = createEmptyStorageBuffer(device, GPU_NEIGHBOR_SEARCH_PARAMETER_LENGTH);
  const statisticParameters = createEmptyStorageBuffer(
    device,
    GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH
  );
  const offsets = createEmptyStorageBuffer(device, POINT_COUNT + 1);
  const neighbors = createEmptyStorageBuffer(device, slotCount);
  const weights = createEmptyStorageBuffer(device, slotCount);
  const overflow = createEmptyStorageBuffer(device, 1);
  const zScores = createEmptyStorageBuffer(device, POINT_COUNT);

  const runner = await GraphRunner.create(device, 'demo-hot-spots', graph => {
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
      new GPUHotSpotAnalysis({
        weights: slots,
        values: importBufferView(graph, 'values', valueBuffer, 'float32', POINT_COUNT),
        parameters: importBufferView(
          graph,
          'statistic-parameters',
          statisticParameters,
          'float32',
          GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH
        ),
        output: {zScores: importBufferView(graph, 'z-scores', zScores, 'float32', POINT_COUNT)}
      })
    );
  });

  const halo = new DotLayer(device, positionBuffer, valueBuffer, POINT_COUNT);
  const hotSpots = new DotLayer(device, positionBuffer, zScores, POINT_COUNT);
  let isDirty = true;

  const draw = () => {
    if (isDirty) {
      isDirty = false;
      writeValues();
      neighborParameters.write(
        getGPUNeighborSearchParameterValues({
          bounds,
          radius: settings.radius / 100,
          weightKind: 'binary'
        })
      );
      statisticParameters.write(getGPUSpatialAutocorrelationParameterValues({}));
      runner.run();
    }
    halo.update(view, theme, {
      mode: 1,
      range: [-3, 3],
      radius: 10,
      alpha: 0.12,
      ramp: theme.diverging
    });
    hotSpots.update(view, theme, {
      mode: 1,
      range: [-4, 4],
      radius: 4,
      ramp: theme.diverging,
      solid: [...theme.muted, 0.4]
    });
    drawLayers(device, theme, [halo, hotSpots]);
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
      pointer = getWorldPosition(view, position);
      isDirty = true;
      scheduler.request();
    },
    setParameters(update) {
      Object.assign(settings, update);
      isDirty = true;
      scheduler.request();
    },
    destroy() {
      scheduler.cancel();
      halo.destroy();
      hotSpots.destroy();
      runner.destroy();
      for (const buffer of [
        positionBuffer,
        valueBuffer,
        neighborParameters,
        statisticParameters,
        offsets,
        neighbors,
        weights,
        overflow,
        zScores
      ]) {
        buffer.destroy();
      }
    }
  };
};

export default scene;

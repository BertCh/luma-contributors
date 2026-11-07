// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Demo: GPUKMeans. Six compiled graphs with 1, 2, 3, 4, 6 and 12 Lloyd iterations replay the
// algorithm step by step: centers walk to the middle of their clusters, then a new seed restarts it.

import {
  GPUKMeans,
  getGPUKMeansParameterValues,
  GPU_KMEANS_PARAMETER_LENGTH
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import type {GpuScene} from '../types';
import {
  assertWebGPU,
  createEmptyStorageBuffer,
  createStorageBuffer,
  DotLayer,
  drawGaussianCluster,
  drawLayers,
  getDemoTheme,
  GraphRunner,
  importBufferView,
  makeRandom,
  type DemoView
} from './kit';

const CLUSTER_COUNT = 5;
const ITERATION_STEPS = [1, 2, 3, 4, 6, 12] as const;
const STEP_MILLISECONDS = 520;
const HOLD_MILLISECONDS = 1800;

const scene: GpuScene = async context => {
  const {device} = context;
  assertWebGPU(device);
  const theme = getDemoTheme(context.theme);
  const worldWidth = context.width / context.height;
  const view: DemoView = {worldWidth, worldHeight: 1, width: context.width, height: context.height};
  const random = makeRandom(context.seed);

  const positions: number[] = [];
  const blobs = [
    [0.17, 0.32, 0.09, 170],
    [0.33, 0.72, 0.08, 150],
    [0.52, 0.42, 0.1, 190],
    [0.74, 0.7, 0.08, 150],
    [0.86, 0.28, 0.09, 160]
  ] as const;
  for (const [centerX, centerY, spread, count] of blobs) {
    for (let index = 0; index < count; index++) {
      positions.push(...drawGaussianCluster(random, centerX * worldWidth, centerY, spread));
    }
  }
  const pointCount = positions.length / 2;

  const positionBuffer = createStorageBuffer(device, Float32Array.from(positions));
  const parameterBuffer = createEmptyStorageBuffer(device, GPU_KMEANS_PARAMETER_LENGTH);
  const labels = createEmptyStorageBuffer(device, pointCount);
  const centers = createEmptyStorageBuffer(device, CLUSTER_COUNT * 2);
  const centerValues = createEmptyStorageBuffer(device, CLUSTER_COUNT);

  const runners: GraphRunner[] = [];
  for (const maximumIterations of ITERATION_STEPS) {
    runners.push(
      await GraphRunner.create(device, `demo-k-means-${maximumIterations}`, graph => {
        graph.add(
          new GPUKMeans({
            id: `demo-k-means-${maximumIterations}`,
            positions: importBufferView(
              graph,
              'positions',
              positionBuffer,
              'float32x2',
              pointCount
            ),
            k: CLUSTER_COUNT,
            maximumIterations,
            parameters: importBufferView(
              graph,
              'parameters',
              parameterBuffer,
              'uint32',
              GPU_KMEANS_PARAMETER_LENGTH
            ),
            output: {
              labels: importBufferView(graph, 'labels', labels, 'uint32', pointCount),
              centers: importBufferView(graph, 'centers', centers, 'float32x2', CLUSTER_COUNT)
            }
          })
        );
      })
    );
  }

  const dots = new DotLayer(device, positionBuffer, labels, pointCount, 'uint32');
  const centerRings = new DotLayer(device, centers, centerValues, CLUSTER_COUNT);
  const cycleMilliseconds = ITERATION_STEPS.length * STEP_MILLISECONDS + HOLD_MILLISECONDS;
  let lastKey = '';

  return {
    render(timeMilliseconds) {
      const elapsed = context.reducedMotion ? cycleMilliseconds - 1 : timeMilliseconds;
      const cycle = Math.floor(elapsed / cycleMilliseconds);
      const within = elapsed - cycle * cycleMilliseconds;
      const step = Math.min(ITERATION_STEPS.length - 1, Math.floor(within / STEP_MILLISECONDS));
      const key = `${cycle}:${step}`;
      if (key !== lastKey) {
        lastKey = key;
        parameterBuffer.write(getGPUKMeansParameterValues({seed: 11 + cycle * 7919}));
        runners[step].run();
      }
      dots.update(view, theme, {mode: 2, radius: 3.2, alpha: 0.8});
      centerRings.update(view, theme, {
        mode: -1,
        radius: 9,
        solid: [...theme.ink, 1],
        hillshade: 1
      });
      drawLayers(device, theme, [dots, centerRings]);
    },
    resize(width, height) {
      view.width = width;
      view.height = height;
    },
    destroy() {
      dots.destroy();
      centerRings.destroy();
      for (const runner of runners) {
        runner.destroy();
      }
      for (const buffer of [positionBuffer, parameterBuffer, labels, centers, centerValues]) {
        buffer.destroy();
      }
    }
  };
};

export default scene;

// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Demo: GPUSpatialClustering (DBSCAN). Blobs, two moons and scattered noise; the epsilon and
// minimum-points sliders relabel every point on the GPU, so clusters merge, split and shed noise.

import {
  getGPUSpatialClusteringParameterValues,
  GPUSpatialClustering,
  GPU_SPATIAL_CLUSTERING_PARAMETER_LENGTH
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import type {GpuScene} from '../types';
import {DBSCAN_PARAMETERS, readParameterValues} from './parameters';
import {
  assertWebGPU,
  createEmptyStorageBuffer,
  createStorageBuffer,
  DotLayer,
  drawGaussianCluster,
  DrawScheduler,
  drawLayers,
  getDemoTheme,
  GraphRunner,
  importBufferView,
  makeRandom,
  type DemoView
} from './kit';

export const parameters = DBSCAN_PARAMETERS;

const scene: GpuScene = async context => {
  const {device} = context;
  assertWebGPU(device);
  const theme = getDemoTheme(context.theme);
  const values = readParameterValues(parameters, context.parameters);
  const worldWidth = context.width / context.height;
  const view: DemoView = {worldWidth, worldHeight: 1, width: context.width, height: context.height};
  const random = makeRandom(context.seed);
  const bounds = [0, 0, worldWidth, 1] as const;

  const positions: number[] = [];
  const addBlob = (centerX: number, centerY: number, spread: number, count: number) => {
    for (let index = 0; index < count; index++) {
      positions.push(...drawGaussianCluster(random, centerX * worldWidth, centerY, spread));
    }
  };
  const addMoon = (
    centerX: number,
    centerY: number,
    radius: number,
    flip: number,
    count: number
  ) => {
    for (let index = 0; index < count; index++) {
      const angle = Math.PI * random();
      const [x, y] = drawGaussianCluster(
        random,
        centerX * worldWidth + Math.cos(angle) * radius,
        centerY + flip * Math.sin(angle) * radius,
        0.008
      );
      positions.push(x, y);
    }
  };
  addBlob(0.17, 0.3, 0.035, 150);
  addBlob(0.2, 0.78, 0.03, 120);
  addMoon(0.55, 0.45, 0.2, 1, 220);
  addMoon(0.67, 0.55, 0.2, -1, 220);
  addBlob(0.9, 0.3, 0.04, 110);
  for (let index = 0; index < 70; index++) {
    positions.push(random() * worldWidth, random());
  }
  const pointCount = positions.length / 2;

  const positionBuffer = createStorageBuffer(device, Float32Array.from(positions));
  const parameterBuffer = createEmptyStorageBuffer(device, GPU_SPATIAL_CLUSTERING_PARAMETER_LENGTH);
  const labels = createEmptyStorageBuffer(device, pointCount);
  const clusterCount = createEmptyStorageBuffer(device, 1);

  const runner = await GraphRunner.create(device, 'demo-dbscan', graph => {
    graph.add(
      new GPUSpatialClustering({
        positions: importBufferView(graph, 'positions', positionBuffer, 'float32x2', pointCount),
        parameters: importBufferView(
          graph,
          'parameters',
          parameterBuffer,
          'float32',
          GPU_SPATIAL_CLUSTERING_PARAMETER_LENGTH
        ),
        gridSize: [64, 36],
        output: {
          labels: importBufferView(graph, 'labels', labels, 'uint32', pointCount),
          clusterCount: importBufferView(graph, 'cluster-count', clusterCount, 'uint32', 1)
        }
      })
    );
  });

  const dots = new DotLayer(device, positionBuffer, labels, pointCount, 'uint32');
  let isDirty = true;

  const draw = () => {
    if (isDirty) {
      isDirty = false;
      parameterBuffer.write(
        getGPUSpatialClusteringParameterValues({
          bounds,
          epsilon: values.epsilon / 100,
          minimumPoints: Math.round(values.minimumPoints)
        })
      );
      runner.run();
    }
    dots.update(view, theme, {
      mode: 2,
      radius: 4.2,
      solid: [...theme.muted, 0.55],
      hillshade: 0.4
    });
    drawLayers(device, theme, [dots]);
  };
  const scheduler = new DrawScheduler(draw);

  return {
    render() {
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
      dots.destroy();
      runner.destroy();
      for (const buffer of [positionBuffer, parameterBuffer, labels, clusterCount]) {
        buffer.destroy();
      }
    }
  };
};

export default scene;

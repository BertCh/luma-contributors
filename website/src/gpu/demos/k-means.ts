// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Demo: GPUSpatialKMeans. Graphs with 1 to 12 Lloyd iterations replay the algorithm step by step: centers
// walk to the middle of their clusters, then a new seed restarts it. Moving either slider stops the
// replay and shows that iteration count and seed.

import {
  GPUSpatialKMeans,
  getGPUSpatialKMeansParameterValues,
  GPU_SPATIAL_KMEANS_PARAMETER_LENGTH
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {getEarthquakes} from '../data/point-observations';
import {createPointIndex} from '../runs/pick';
import type {RunGlyph} from '../runs/interaction';
import {createPointerDriver} from '../runs/pointer-driver';
import type {GpuScene} from '../types';
import {createJapanCoast} from './japan-coast';
import {K_MEANS_PARAMETERS, readParameterValues} from './parameters';
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

export const parameters = K_MEANS_PARAMETERS;

const CLUSTER_COUNT = 5;
const ITERATION_STEPS = [1, 2, 3, 4, 6, 12] as const;
const STEP_MILLISECONDS = 520;
const HOLD_MILLISECONDS = 1800;

const scene: GpuScene = async context => {
  const {device} = context;
  assertWebGPU(device);
  const theme = getDemoTheme(context.theme);
  const values = readParameterValues(parameters, context.parameters);
  const worldWidth = context.width / context.height;
  const view: DemoView = {worldWidth, worldHeight: 1, width: context.width, height: context.height};
  const random = makeRandom(context.seed);

  const {positions, count: pointCount} = getEarthquakes(
    random,
    {width: worldWidth, height: 1},
    820
  );

  const positionBuffer = createStorageBuffer(device, positions);
  const parameterBuffer = createEmptyStorageBuffer(device, GPU_SPATIAL_KMEANS_PARAMETER_LENGTH);
  const labels = createEmptyStorageBuffer(device, pointCount);
  const centers = createEmptyStorageBuffer(device, CLUSTER_COUNT * 2);
  const centerValues = createEmptyStorageBuffer(device, CLUSTER_COUNT);

  const createRunner = (maximumIterations: number) =>
    GraphRunner.create(device, `demo-k-means-${maximumIterations}`, graph => {
      graph.add(
        new GPUSpatialKMeans({
          id: `demo-k-means-${maximumIterations}`,
          positions: importBufferView(graph, 'positions', positionBuffer, 'float32x2', pointCount),
          clusterCount: CLUSTER_COUNT,
          maximumIterations,
          parameters: importBufferView(
            graph,
            'parameters',
            parameterBuffer,
            'uint32',
            GPU_SPATIAL_KMEANS_PARAMETER_LENGTH
          ),
          output: {
            labels: importBufferView(graph, 'labels', labels, 'uint32', pointCount),
            centers: importBufferView(graph, 'centers', centers, 'float32x2', CLUSTER_COUNT)
          }
        })
      );
    });
  // One compiled graph per iteration count, built on first use.
  const runners = new Map<number, GraphRunner>();
  const pending = new Set<number>();
  for (const maximumIterations of ITERATION_STEPS) {
    runners.set(maximumIterations, await createRunner(maximumIterations));
  }
  let isDestroyed = false;

  const coast = createJapanCoast(device, worldWidth);
  const dots = new DotLayer(device, positionBuffer, labels, pointCount, 'uint32');
  const centerRings = new DotLayer(device, centers, centerValues, CLUSTER_COUNT);
  const litPositions = createEmptyStorageBuffer(device, pointCount * 2);
  const litValues = createEmptyStorageBuffer(device, pointCount);
  const litDots = new DotLayer(device, litPositions, litValues, pointCount);
  litDots.setInstanceCount(0);
  const pointIndex = createPointIndex(positions, pointCount);
  // CPU copies of the latest run, for the probe. The run counter drops reads that arrive late.
  let clusterLabels: Uint32Array | null = null;
  let clusterCenters: Float32Array | null = null;
  let runCount = 0;
  const readClusters = async () => {
    const thisRun = ++runCount;
    const [labelBytes, centerBytes] = await Promise.all([labels.readAsync(), centers.readAsync()]);
    if (isDestroyed || thisRun !== runCount) return;
    clusterLabels = new Uint32Array(labelBytes.buffer, labelBytes.byteOffset, pointCount).slice();
    clusterCenters = new Float32Array(
      centerBytes.buffer,
      centerBytes.byteOffset,
      CLUSTER_COUNT * 2
    ).slice();
    driver.refresh();
    scheduler.request();
  };
  const cycleMilliseconds = ITERATION_STEPS.length * STEP_MILLISECONDS + HOLD_MILLISECONDS;
  let lastKey = '';
  let isManual = false;

  const draw = (timeMilliseconds: number) => {
    let iterations = Math.round(values.iterations);
    let seed = 11 + Math.round(values.seed) * 7919;
    if (!isManual) {
      const elapsed = context.reducedMotion ? cycleMilliseconds - 1 : timeMilliseconds;
      const cycle = Math.floor(elapsed / cycleMilliseconds);
      const within = elapsed - cycle * cycleMilliseconds;
      const step = Math.min(ITERATION_STEPS.length - 1, Math.floor(within / STEP_MILLISECONDS));
      iterations = ITERATION_STEPS[step];
      seed = 11 + cycle * 7919;
    }
    const runner = runners.get(iterations);
    const key = `${iterations}:${seed}:${runner ? 1 : 0}`;
    if (key !== lastKey) {
      if (runner) {
        lastKey = key;
        parameterBuffer.write(getGPUSpatialKMeansParameterValues({seed}));
        runner.run();
        void readClusters();
      } else if (!pending.has(iterations)) {
        pending.add(iterations);
        void createRunner(iterations).then(created => {
          if (isDestroyed) {
            created.destroy();
            return;
          }
          runners.set(iterations, created);
          scheduler.request();
        });
      }
    }
    dots.update(view, theme, {mode: 2, radius: 3.2, alpha: 0.8});
    centerRings.update(view, theme, {
      mode: -1,
      radius: 9,
      solid: [...theme.ink, 1],
      hillshade: 1
    });
    coast.layer.update(view, theme, {radius: 0.7, solid: [...theme.muted, 0.8]});
    litDots.update(view, theme, {mode: -1, radius: 4.4, solid: [...theme.ink, 1], hillshade: 1});
    drawLayers(device, theme, [coast.layer, dots, litDots, centerRings, ...driver.getLayers()]);
  };
  let lastTime = 0;
  const scheduler = new DrawScheduler(() => draw(lastTime));

  // The seeding is internal to GPUSpatialKMeans (a seed parameter, no initial-centers input), so the
  // pointer probes the nearest epicenter instead of carrying a center.
  const lit = new Float32Array(pointCount * 2);
  const driver = createPointerDriver({
    device,
    view,
    theme,
    reducedMotion: context.reducedMotion,
    kind: 'probe',
    hint: 'hover an epicenter',
    rest: [0.5, 0.5],
    reportReadout: context.reportReadout,
    requestDraw: () => scheduler.request(),
    resolve(pointer, pointerContext) {
      if (!clusterLabels || !clusterCenters) {
        litDots.setInstanceCount(0);
        return {};
      }
      const nearest = pointIndex.nearest(pointer.x, pointer.y);
      if (nearest < 0) return {};
      const cluster = clusterLabels[nearest];
      let litCount = 0;
      for (let index = 0; index < pointCount; index++) {
        if (clusterLabels[index] !== cluster) continue;
        lit[litCount * 2] = positions[index * 2];
        lit[litCount * 2 + 1] = positions[index * 2 + 1];
        litCount++;
      }
      litPositions.write(lit);
      litDots.setInstanceCount(litCount);
      scheduler.request();
      const center: [number, number] = [
        clusterCenters[cluster * 2],
        clusterCenters[cluster * 2 + 1]
      ];
      const epicenter: [number, number] = [positions[nearest * 2], positions[nearest * 2 + 1]];
      const glyphs: RunGlyph[] = [
        {shape: 'line', from: epicenter, to: center},
        {shape: 'ring', at: epicenter, radius: pointerContext.pixelSize * 7},
        {shape: 'crosshair', at: center}
      ];
      return {
        glyphs,
        readout: `cluster ${cluster + 1} of ${CLUSTER_COUNT} · ${litCount} quakes · ${Math.hypot(
          epicenter[0] - center[0],
          epicenter[1] - center[1]
        ).toFixed(2)} from its center`
      };
    }
  });

  return {
    gesture: driver.gesture,
    render(timeMilliseconds) {
      lastTime = timeMilliseconds;
      driver.advance(timeMilliseconds);
      scheduler.flush();
    },
    setParameters(update) {
      isManual = true;
      Object.assign(values, update);
      scheduler.request();
    },
    resize(width, height) {
      view.width = width;
      view.height = height;
      driver.refresh();
      scheduler.request();
    },
    setPointer: driver.setPointer,
    resetPointer: driver.resetPointer,
    destroy() {
      isDestroyed = true;
      scheduler.cancel();
      driver.destroy();
      litDots.destroy();
      coast.destroy();
      dots.destroy();
      centerRings.destroy();
      for (const runner of runners.values()) {
        runner.destroy();
      }
      for (const buffer of [
        positionBuffer,
        parameterBuffer,
        labels,
        centers,
        centerValues,
        litPositions,
        litValues
      ]) {
        buffer.destroy();
      }
    }
  };
};

export default scene;

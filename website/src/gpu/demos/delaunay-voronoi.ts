// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Demo: GPUDelaunayTessellation and GPUVoronoiDiagram. Drifting sites are re-triangulated on the
// GPU every frame (sites and drift are sliders), and the Voronoi cells are derived from the triangles in the same command graph.

import {
  getDelaunayMaximumTriangleCount,
  getVoronoiMaximumSegmentCount,
  GPUDelaunayTessellation,
  GPUVoronoiDiagram
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import type {GpuScene} from '../types';
import {DELAUNAY_VORONOI_PARAMETERS, readParameterValues} from './parameters';
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
  SegmentListLayer,
  TriangleEdgeLayer,
  type DemoView
} from './kit';

export const parameters = DELAUNAY_VORONOI_PARAMETERS;

const SITE_COUNT = 96;

const scene: GpuScene = async context => {
  const {device} = context;
  assertWebGPU(device);
  const theme = getDemoTheme(context.theme);
  const parameterValues = readParameterValues(parameters, context.parameters);
  const worldWidth = context.width / context.height;
  const view: DemoView = {worldWidth, worldHeight: 1, width: context.width, height: context.height};
  const random = makeRandom(context.seed);

  // Site 0 follows the pointer; the others wander around jittered lattice homes.
  const homes = new Float32Array(SITE_COUNT * 2);
  const phases = new Float32Array(SITE_COUNT * 2);
  for (let index = 1; index < SITE_COUNT; index++) {
    const column = (index - 1) % 12;
    const row = Math.floor((index - 1) / 12);
    homes[index * 2] = ((column + 0.5 + (random() - 0.5) * 0.9) / 12) * worldWidth;
    homes[index * 2 + 1] = (row + 0.5 + (random() - 0.5) * 0.9) / 8;
    phases[index * 2] = random() * 6.28;
    phases[index * 2 + 1] = random() * 6.28;
  }
  // Shuffle the sites so any prefix of them covers the whole canvas (the Sites slider).
  for (let index = SITE_COUNT - 1; index > 1; index--) {
    const other = 1 + Math.floor(random() * index);
    for (const array of [homes, phases]) {
      for (let axis = 0; axis < 2; axis++) {
        const held = array[index * 2 + axis];
        array[index * 2 + axis] = array[other * 2 + axis];
        array[other * 2 + axis] = held;
      }
    }
  }
  const positions = Float32Array.from(homes);

  const triangleCapacity = getDelaunayMaximumTriangleCount(SITE_COUNT);
  const segmentCapacity = getVoronoiMaximumSegmentCount(triangleCapacity);
  const positionBuffer = createStorageBuffer(device, positions);
  const triangles = createEmptyStorageBuffer(device, triangleCapacity * 3);
  const triangleCount = createEmptyStorageBuffer(device, 1);
  const triangleRequired = createEmptyStorageBuffer(device, 1);
  const triangleOverflow = createEmptyStorageBuffer(device, 1);
  const clipBounds = createStorageBuffer(device, Float32Array.of(0, 0, worldWidth, 1));
  const segments = createEmptyStorageBuffer(device, segmentCapacity * 4);
  const segmentSites = createEmptyStorageBuffer(device, segmentCapacity * 2);
  const segmentCount = createEmptyStorageBuffer(device, 1);
  const segmentRequired = createEmptyStorageBuffer(device, 1);
  const segmentOverflow = createEmptyStorageBuffer(device, 1);
  const siteValues = createEmptyStorageBuffer(device, SITE_COUNT);

  const createRunner = (siteCount: number) =>
    GraphRunner.create(device, `demo-delaunay-voronoi-${siteCount}`, graph => {
      const capacity = getDelaunayMaximumTriangleCount(siteCount);
      const segmentLimit = getVoronoiMaximumSegmentCount(capacity);
      const sites = importBufferView(graph, 'positions', positionBuffer, 'float32x2', siteCount);
      const count = importBufferView(graph, 'triangle-count', triangleCount, 'uint32', 1);
      const triangleView = importBufferView(graph, 'triangles', triangles, 'uint32x3', capacity);
      graph.add(
        new GPUDelaunayTessellation({
          positions: sites,
          output: {
            triangles: triangleView,
            status: {
              count,
              requiredCount: importBufferView(
                graph,
                'triangle-required',
                triangleRequired,
                'uint32',
                1
              ),
              overflow: importBufferView(graph, 'triangle-overflow', triangleOverflow, 'uint32', 1)
            }
          }
        })
      );
      graph.add(
        new GPUVoronoiDiagram({
          positions: sites,
          triangles: triangleView,
          triangleCount: count,
          clipBounds: importBufferView(graph, 'clip-bounds', clipBounds, 'float32x4', 1),
          output: {
            segments: importBufferView(graph, 'segments', segments, 'float32x4', segmentLimit),
            siteIds: importBufferView(
              graph,
              'segment-sites',
              segmentSites,
              'uint32x2',
              segmentLimit
            ),
            status: {
              count: importBufferView(graph, 'segment-count', segmentCount, 'uint32', 1),
              requiredCount: importBufferView(
                graph,
                'segment-required',
                segmentRequired,
                'uint32',
                1
              ),
              overflow: importBufferView(graph, 'segment-overflow', segmentOverflow, 'uint32', 1)
            }
          }
        })
      );
    });
  // One compiled graph per site count, built on first use.
  const runners = new Map<number, GraphRunner>([[SITE_COUNT, await createRunner(SITE_COUNT)]]);
  const pending = new Set<number>();
  let activeSiteCount = SITE_COUNT;
  let isDestroyed = false;

  const delaunayEdges = new TriangleEdgeLayer(
    device,
    positionBuffer,
    triangles,
    triangleCount,
    triangleCapacity
  );
  const voronoiEdges = new SegmentListLayer(device, segments, segmentCount, segmentCapacity);
  const sites = new DotLayer(device, positionBuffer, siteValues, SITE_COUNT);
  let pointer: [number, number] | null = null;
  let timeSeconds = context.reducedMotion ? 4 : 0;

  const draw = () => {
    const seconds = timeSeconds;
    const wander = parameterValues.wander / 100;
    const siteCount = Math.round(parameterValues.sites);
    let runner = runners.get(siteCount);
    if (!runner) {
      if (!pending.has(siteCount)) {
        pending.add(siteCount);
        void createRunner(siteCount).then(created => {
          if (isDestroyed) {
            created.destroy();
            return;
          }
          runners.set(siteCount, created);
          scheduler.request();
        });
      }
      runner = runners.get(activeSiteCount);
    } else {
      activeSiteCount = siteCount;
    }
    for (let index = 1; index < SITE_COUNT; index++) {
      positions[index * 2] =
        homes[index * 2] + Math.sin(seconds * 0.6 + phases[index * 2]) * wander;
      positions[index * 2 + 1] =
        homes[index * 2 + 1] + Math.cos(seconds * 0.5 + phases[index * 2 + 1]) * wander;
    }
    const [x, y] = pointer ?? [
      worldWidth * (0.5 + 0.35 * Math.cos(seconds * 0.4)),
      0.5 + 0.32 * Math.sin(seconds * 0.55)
    ];
    positions[0] = x;
    positions[1] = y;
    positionBuffer.write(positions);
    runner?.run();
    sites.setInstanceCount(activeSiteCount);
    const accent = theme.categories[0];
    const warm = theme.categories[1];
    delaunayEdges.update(view, theme, {
      radius: 0.7,
      solid: [accent[0], accent[1], accent[2], 0.45]
    });
    voronoiEdges.update(view, theme, {radius: 1.1, solid: [warm[0], warm[1], warm[2], 0.95]});
    sites.update(view, theme, {mode: -1, radius: 3, solid: [...theme.ink, 1]});
    drawLayers(device, theme, [delaunayEdges, voronoiEdges, sites]);
  };
  const scheduler = new DrawScheduler(draw);

  return {
    render(timeMilliseconds) {
      if (!context.reducedMotion) timeSeconds = timeMilliseconds / 1000;
      scheduler.flush();
    },
    setParameters(update) {
      Object.assign(parameterValues, update);
      scheduler.request();
    },
    resize(width, height) {
      view.width = width;
      view.height = height;
      scheduler.request();
    },
    setPointer(position) {
      pointer = position ? getWorldPosition(view, position) : null;
      scheduler.request();
    },
    destroy() {
      isDestroyed = true;
      scheduler.cancel();
      delaunayEdges.destroy();
      voronoiEdges.destroy();
      sites.destroy();
      for (const runner of runners.values()) {
        runner.destroy();
      }
      for (const buffer of [
        positionBuffer,
        triangles,
        triangleCount,
        triangleRequired,
        triangleOverflow,
        clipBounds,
        segments,
        segmentSites,
        segmentCount,
        segmentRequired,
        segmentOverflow,
        siteValues
      ]) {
        buffer.destroy();
      }
    }
  };
};

export default scene;

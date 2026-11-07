// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Demo: GPUDelaunayTessellation and GPUVoronoiDiagram. Drifting sites are re-triangulated on the
// GPU every frame, and the Voronoi cells are derived from the triangles in the same command graph.

import {
  getDelaunayMaximumTriangleCount,
  getVoronoiMaximumSegmentCount,
  GPUDelaunayTessellation,
  GPUVoronoiDiagram
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import type {GpuScene} from '../types';
import {
  assertWebGPU,
  createEmptyStorageBuffer,
  createStorageBuffer,
  DotLayer,
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

const SITE_COUNT = 96;

const scene: GpuScene = async context => {
  const {device} = context;
  assertWebGPU(device);
  const theme = getDemoTheme(context.theme);
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

  const runner = await GraphRunner.create(device, 'demo-delaunay-voronoi', graph => {
    const sites = importBufferView(graph, 'positions', positionBuffer, 'float32x2', SITE_COUNT);
    const count = importBufferView(graph, 'triangle-count', triangleCount, 'uint32', 1);
    const triangleView = importBufferView(
      graph,
      'triangles',
      triangles,
      'uint32x3',
      triangleCapacity
    );
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
          segments: importBufferView(graph, 'segments', segments, 'float32x4', segmentCapacity),
          siteIds: importBufferView(
            graph,
            'segment-sites',
            segmentSites,
            'uint32x2',
            segmentCapacity
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

  return {
    render(timeMilliseconds) {
      const seconds = context.reducedMotion ? 4 : timeMilliseconds / 1000;
      for (let index = 1; index < SITE_COUNT; index++) {
        positions[index * 2] =
          homes[index * 2] + Math.sin(seconds * 0.6 + phases[index * 2]) * 0.035;
        positions[index * 2 + 1] =
          homes[index * 2 + 1] + Math.cos(seconds * 0.5 + phases[index * 2 + 1]) * 0.035;
      }
      const [x, y] = pointer ?? [
        worldWidth * (0.5 + 0.35 * Math.cos(seconds * 0.4)),
        0.5 + 0.32 * Math.sin(seconds * 0.55)
      ];
      positions[0] = x;
      positions[1] = y;
      positionBuffer.write(positions);
      runner.run();
      const accent = theme.categories[0];
      const warm = theme.categories[1];
      delaunayEdges.update(view, theme, {
        radius: 0.7,
        solid: [accent[0], accent[1], accent[2], 0.45]
      });
      voronoiEdges.update(view, theme, {radius: 1.1, solid: [warm[0], warm[1], warm[2], 0.95]});
      sites.update(view, theme, {mode: -1, radius: 3, solid: [...theme.ink, 1]});
      drawLayers(device, theme, [delaunayEdges, voronoiEdges, sites]);
    },
    resize(width, height) {
      view.width = width;
      view.height = height;
    },
    setPointer(position) {
      pointer = position ? getWorldPosition(view, position) : null;
    },
    destroy() {
      delaunayEdges.destroy();
      voronoiEdges.destroy();
      sites.destroy();
      runner.destroy();
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

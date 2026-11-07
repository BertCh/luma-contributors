// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Demo: GPUNetworkReachability. Least travel cost over a jittered street grid from a moving
// origin, with a cost limit that grows so the isochrone wavefront sweeps outward. Roads are
// colored by the GPU's cost buffer, banded into time rings.

import {GPUNetworkReachability} from '@luma-contributors/analysis/gpu-network';
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
  SegmentLayer,
  type DemoView
} from './kit';

const ROWS = 20;

const scene: GpuScene = async context => {
  const {device} = context;
  assertWebGPU(device);
  const theme = getDemoTheme(context.theme);
  const worldWidth = context.width / context.height;
  const columns = Math.max(16, Math.round(ROWS * worldWidth));
  const view: DemoView = {worldWidth, worldHeight: 1, width: context.width, height: context.height};
  const random = makeRandom(context.seed);

  // Jittered lattice of intersections, with some blocks knocked out and a few diagonals added.
  const nodeCount = columns * ROWS;
  const nodes = new Float32Array(nodeCount * 2);
  for (let row = 0; row < ROWS; row++) {
    for (let column = 0; column < columns; column++) {
      const index = row * columns + column;
      nodes[index * 2] = ((column + 0.5 + (random() - 0.5) * 0.5) / columns) * worldWidth;
      nodes[index * 2 + 1] = (row + 0.5 + (random() - 0.5) * 0.5) / ROWS;
    }
  }
  const edgePairs: number[] = [];
  const addEdge = (from: number, to: number) => edgePairs.push(from, to);
  for (let row = 0; row < ROWS; row++) {
    for (let column = 0; column < columns; column++) {
      const index = row * columns + column;
      if (column + 1 < columns && random() > 0.1) addEdge(index, index + 1);
      if (row + 1 < ROWS && random() > 0.1) addEdge(index, index + columns);
      if (column + 1 < columns && row + 1 < ROWS && random() > 0.93)
        addEdge(index, index + columns + 1);
    }
  }
  const edgeCount = edgePairs.length / 2;

  // Undirected roads become two directed CSR slots; cost is length at a per-road speed.
  const degree = new Uint32Array(nodeCount);
  for (let edge = 0; edge < edgeCount; edge++) {
    degree[edgePairs[edge * 2]]++;
    degree[edgePairs[edge * 2 + 1]]++;
  }
  const offsets = new Uint32Array(nodeCount + 1);
  for (let node = 0; node < nodeCount; node++) offsets[node + 1] = offsets[node] + degree[node];
  const slotCount = offsets[nodeCount];
  const neighbors = new Uint32Array(slotCount);
  const weights = new Float32Array(slotCount);
  const cursor = Uint32Array.from(offsets.subarray(0, nodeCount));
  for (let edge = 0; edge < edgeCount; edge++) {
    const from = edgePairs[edge * 2];
    const to = edgePairs[edge * 2 + 1];
    const length = Math.hypot(
      nodes[from * 2] - nodes[to * 2],
      nodes[from * 2 + 1] - nodes[to * 2 + 1]
    );
    const cost = length * (1 + random() * 0.9);
    neighbors[cursor[from]] = to;
    weights[cursor[from]++] = cost;
    neighbors[cursor[to]] = from;
    weights[cursor[to]++] = cost;
  }

  const nodeBuffer = createStorageBuffer(device, nodes);
  const edgeBuffer = createStorageBuffer(device, Uint32Array.from(edgePairs));
  const offsetBuffer = createStorageBuffer(device, offsets);
  const neighborBuffer = createStorageBuffer(device, neighbors);
  const weightBuffer = createStorageBuffer(device, weights);
  const startNodeBuffer = createStorageBuffer(device, Uint32Array.of(0));
  const costLimitBuffer = createStorageBuffer(device, Float32Array.of(0));
  const costBuffer = createEmptyStorageBuffer(device, nodeCount);
  const originBuffer = createStorageBuffer(device, Float32Array.of(0, 0));
  const originValue = createEmptyStorageBuffer(device, 1);

  const runner = await GraphRunner.create(device, 'demo-network-isochrones', graph => {
    graph.add(
      new GPUNetworkReachability({
        topology: {
          offsets: importBufferView(graph, 'offsets', offsetBuffer, 'uint32', nodeCount + 1),
          neighbors: importBufferView(graph, 'neighbors', neighborBuffer, 'uint32', slotCount),
          weights: importBufferView(graph, 'weights', weightBuffer, 'float32', slotCount)
        },
        startNodes: importBufferView(graph, 'start-nodes', startNodeBuffer, 'uint32', 1),
        costLimit: importBufferView(graph, 'cost-limit', costLimitBuffer, 'float32', 1),
        maximumIterations: 24,
        localIterations: 16,
        output: {costs: importBufferView(graph, 'costs', costBuffer, 'float32', nodeCount)}
      })
    );
  });

  const roads = new SegmentLayer(device, nodeBuffer, edgeBuffer, costBuffer, edgeCount);
  const origin = new DotLayer(device, originBuffer, originValue, 1);
  const maximumCost = 1.25;
  let startNode = -1;
  let pointer: [number, number] | null = null;

  const findNearestNode = (x: number, y: number): number => {
    let best = 0;
    let bestDistance = Infinity;
    for (let node = 0; node < nodeCount; node++) {
      const distance = (nodes[node * 2] - x) ** 2 + (nodes[node * 2 + 1] - y) ** 2;
      if (distance < bestDistance) {
        bestDistance = distance;
        best = node;
      }
    }
    return best;
  };

  return {
    render(timeMilliseconds) {
      const seconds = context.reducedMotion ? 6 : timeMilliseconds / 1000;
      const [x, y] = pointer ?? [
        worldWidth * (0.5 + 0.32 * Math.cos(seconds * 0.17)),
        0.5 + 0.3 * Math.sin(seconds * 0.23)
      ];
      const nextStart = findNearestNode(x, y);
      if (nextStart !== startNode) {
        startNode = nextStart;
        startNodeBuffer.write(Uint32Array.of(startNode));
        originBuffer.write(Float32Array.of(nodes[startNode * 2], nodes[startNode * 2 + 1]));
      }
      // The wavefront grows for three seconds, holds, then restarts.
      const cycle = (seconds % 4.5) / 3;
      const limit = context.reducedMotion ? maximumCost : maximumCost * Math.min(1, cycle);
      costLimitBuffer.write(Float32Array.of(Math.max(limit, 0.001)));
      runner.run();
      const muted = theme.muted;
      roads.update(view, theme, {
        range: [-0.25, maximumCost],
        radius: 1.5,
        bands: 10,
        solid: [muted[0], muted[1], muted[2], 0.35]
      });
      origin.update(view, theme, {mode: -1, radius: 7, solid: [...theme.ink, 1], hillshade: 1});
      drawLayers(device, theme, [roads, origin]);
    },
    resize(width, height) {
      view.width = width;
      view.height = height;
    },
    setPointer(position) {
      pointer = position ? getWorldPosition(view, position) : null;
    },
    destroy() {
      roads.destroy();
      origin.destroy();
      runner.destroy();
      for (const buffer of [
        nodeBuffer,
        edgeBuffer,
        offsetBuffer,
        neighborBuffer,
        weightBuffer,
        startNodeBuffer,
        costLimitBuffer,
        costBuffer,
        originBuffer,
        originValue
      ]) {
        buffer.destroy();
      }
    }
  };
};

export default scene;

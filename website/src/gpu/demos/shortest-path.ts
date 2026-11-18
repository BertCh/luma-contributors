// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Demo: GPUNetworkReachability with predecessors, then GPUNetworkPathExtraction. The GPU finds the
// least-cost route from a fixed origin to the node under the pointer by walking predecessors.
// Congestion restretches road costs, so the route detours; the origin is a slider.

import {
  GPUNetworkPathExtraction,
  GPUNetworkReachability
} from '@luma-contributors/analysis/gpu-network';
import {createTransientView} from '@luma.gl/graph/gpu-core';
import type {RunGlyph} from '../runs/interaction';
import {chooseSticky, createPointIndex} from '../runs/pick';
import {createPointerDriver} from '../runs/pointer-driver';
import type {GpuScene} from '../types';
import {readParameterValues, SHORTEST_PATH_PARAMETERS} from './parameters';
import {createStreetNetwork} from './street-network';
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
  PathLayer,
  SegmentLayer,
  type DemoView
} from './kit';

export const parameters = SHORTEST_PATH_PARAMETERS;

const MAXIMUM_NODES = 640;
const MAXIMUM_PATH_LENGTH = 256;

const scene: GpuScene = async context => {
  const {device} = context;
  assertWebGPU(device);
  const theme = getDemoTheme(context.theme);
  const values = readParameterValues(parameters, context.parameters);
  const worldWidth = context.width / context.height;
  const view: DemoView = {worldWidth, worldHeight: 1, width: context.width, height: context.height};
  const network = createStreetNetwork(worldWidth, MAXIMUM_NODES);
  const {nodes, nodeCount, edgeCount, slotCount} = network;
  const weights = new Float32Array(slotCount);
  network.fillWeights(weights, values.congestion);

  const nodeBuffer = createStorageBuffer(device, nodes);
  const edgeBuffer = createStorageBuffer(device, network.edgePairs);
  const offsetBuffer = createStorageBuffer(device, network.offsets);
  const neighborBuffer = createStorageBuffer(device, network.neighbors);
  const weightBuffer = createStorageBuffer(device, weights);
  const startNodeBuffer = createStorageBuffer(device, Uint32Array.of(0));
  const targetBuffer = createStorageBuffer(device, Uint32Array.of(0));
  const costBuffer = createEmptyStorageBuffer(device, nodeCount);
  const predecessorBuffer = createEmptyStorageBuffer(device, nodeCount);
  const convergedBuffer = createEmptyStorageBuffer(device, 1);
  const pathIds = createEmptyStorageBuffer(device, MAXIMUM_PATH_LENGTH);
  const pathCount = createEmptyStorageBuffer(device, 1);
  const pathOverflow = createEmptyStorageBuffer(device, 1);
  const endpointBuffer = createStorageBuffer(device, new Float32Array(4));
  const endpointValues = createEmptyStorageBuffer(device, 2);

  const runner = await GraphRunner.create(device, 'demo-shortest-path', graph => {
    const costs = importBufferView(graph, 'costs', costBuffer, 'float32', nodeCount);
    const predecessors = importBufferView(
      graph,
      'predecessors',
      predecessorBuffer,
      'uint32',
      nodeCount
    );
    const converged = importBufferView(graph, 'converged', convergedBuffer, 'uint32', 1);
    graph.add(
      new GPUNetworkReachability({
        topology: {
          offsets: importBufferView(graph, 'offsets', offsetBuffer, 'uint32', nodeCount + 1),
          neighbors: importBufferView(graph, 'neighbors', neighborBuffer, 'uint32', slotCount),
          weights: importBufferView(graph, 'weights', weightBuffer, 'float32', slotCount)
        },
        startNodes: importBufferView(graph, 'start-nodes', startNodeBuffer, 'uint32', 1),
        maximumIterations: 24,
        localIterations: 16,
        output: {costs, predecessors, status: {converged}}
      })
    );
    graph.add(
      new GPUNetworkPathExtraction({
        predecessors,
        costs,
        targets: importBufferView(graph, 'targets', targetBuffer, 'uint32', 1),
        predecessorsComplete: converged,
        maximumPathLength: MAXIMUM_PATH_LENGTH,
        output: {
          ids: createTransientView(graph, 'path-identities', 'uint32', MAXIMUM_PATH_LENGTH),
          nodeRows: importBufferView(graph, 'path-rows', pathIds, 'uint32', MAXIMUM_PATH_LENGTH),
          status: {
            count: importBufferView(graph, 'path-count', pathCount, 'uint32', 1),
            overflow: importBufferView(graph, 'path-overflow', pathOverflow, 'uint32', 1)
          }
        }
      })
    );
  });

  const roads = new SegmentLayer(device, nodeBuffer, edgeBuffer, costBuffer, edgeCount);
  const route = new PathLayer(device, nodeBuffer, pathIds, pathCount, MAXIMUM_PATH_LENGTH);
  const endpoints = new DotLayer(device, endpointBuffer, endpointValues, 2);
  let startNode = 0;
  let targetNode = network.findNearestNode(worldWidth * 0.85, 0.7);
  let isDirty = true;
  const accent = theme.categories[1];

  const draw = () => {
    if (isDirty) {
      isDirty = false;
      startNode = network.findNearestNode(worldWidth * (values.originPosition / 100), 0.5);
      network.fillWeights(weights, values.congestion);
      weightBuffer.write(weights);
      startNodeBuffer.write(Uint32Array.of(startNode));
      targetBuffer.write(Uint32Array.of(targetNode));
      endpointBuffer.write(
        Float32Array.of(
          nodes[startNode * 2],
          nodes[startNode * 2 + 1],
          nodes[targetNode * 2],
          nodes[targetNode * 2 + 1]
        )
      );
      runner.run();
    }
    const muted = theme.muted;
    roads.update(view, theme, {
      range: [0, 3],
      radius: 1.2,
      alpha: 0.5,
      solid: [muted[0], muted[1], muted[2], 0.3]
    });
    route.update(view, theme, {radius: 3.2, solid: [accent[0], accent[1], accent[2], 1]});
    endpoints.update(view, theme, {mode: -1, radius: 7, solid: [...theme.ink, 1], hillshade: 1});
    drawLayers(device, theme, [roads, route, endpoints, ...driver.getLayers()]);
  };
  const scheduler = new DrawScheduler(draw);

  const nodeIndex = createPointIndex(nodes, nodeCount);
  const driver = createPointerDriver({
    device,
    view,
    theme,
    reducedMotion: context.reducedMotion,
    kind: 'place',
    hint: 'move the destination',
    rest: pointerContext =>
      pointerContext.toUnit([nodes[targetNode * 2], nodes[targetNode * 2 + 1]]),
    reportReadout: context.reportReadout,
    requestDraw: () => scheduler.request(),
    resolve(pointer) {
      const nextTarget = chooseSticky(nodeIndex, pointer.x, pointer.y, targetNode);
      if (nextTarget !== targetNode) {
        targetNode = nextTarget;
        isDirty = true;
        scheduler.request();
      }
      const snapped: [number, number] = [nodes[targetNode * 2], nodes[targetNode * 2 + 1]];
      const glyphs: RunGlyph[] = [
        {shape: 'line', from: [pointer.x, pointer.y], to: snapped, dashed: true},
        {shape: 'crosshair', at: snapped, size: 12},
        {shape: 'dot', at: snapped}
      ];
      return {glyphs, readout: `destination node ${targetNode}`};
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
      isDirty = true;
      scheduler.request();
    },
    destroy() {
      scheduler.cancel();
      driver.destroy();
      roads.destroy();
      route.destroy();
      endpoints.destroy();
      runner.destroy();
      for (const buffer of [
        nodeBuffer,
        edgeBuffer,
        offsetBuffer,
        neighborBuffer,
        weightBuffer,
        startNodeBuffer,
        targetBuffer,
        costBuffer,
        predecessorBuffer,
        convergedBuffer,
        pathIds,
        pathCount,
        pathOverflow,
        endpointBuffer,
        endpointValues
      ]) {
        buffer.destroy();
      }
    }
  };
};

export default scene;

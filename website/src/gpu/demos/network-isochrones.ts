// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Demo: GPUNetworkReachability. Least travel cost over a jittered street grid. The pointer picks
// the origin, sliders set the travel budget (cost limit) and congestion (road costs). Roads are
// colored by the GPU's cost buffer, banded into time rings.

import {GPUNetworkReachability} from '@luma-contributors/analysis/gpu-network';
import {createStreetNetwork} from './street-network';
import type {GpuScene} from '../types';
import {NETWORK_ISOCHRONES_PARAMETERS, readParameterValues} from './parameters';
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
  SegmentLayer,
  type DemoView
} from './kit';

export const parameters = NETWORK_ISOCHRONES_PARAMETERS;

const ROWS = 20;

const scene: GpuScene = async context => {
  const {device} = context;
  assertWebGPU(device);
  const theme = getDemoTheme(context.theme);
  const parameterValues = readParameterValues(parameters, context.parameters);
  const worldWidth = context.width / context.height;
  const view: DemoView = {worldWidth, worldHeight: 1, width: context.width, height: context.height};
  const random = makeRandom(context.seed);

  const network = createStreetNetwork(random, worldWidth, ROWS);
  const {nodes, nodeCount, edgeCount, slotCount, findNearestNode} = network;
  const weights = new Float32Array(slotCount);
  const fillWeights = () => network.fillWeights(weights, parameterValues.congestion);
  fillWeights();

  const nodeBuffer = createStorageBuffer(device, nodes);
  const edgeBuffer = createStorageBuffer(device, network.edgePairs);
  const offsetBuffer = createStorageBuffer(device, network.offsets);
  const neighborBuffer = createStorageBuffer(device, network.neighbors);
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
  let isDirty = true;

  startNode = findNearestNode(worldWidth * 0.5, 0.5);

  const draw = () => {
    if (isDirty) {
      isDirty = false;
      fillWeights();
      weightBuffer.write(weights);
      startNodeBuffer.write(Uint32Array.of(startNode));
      originBuffer.write(Float32Array.of(nodes[startNode * 2], nodes[startNode * 2 + 1]));
      costLimitBuffer.write(Float32Array.of(Math.max(parameterValues.costLimit, 0.001)));
      runner.run();
    }
    const muted = theme.muted;
    roads.update(view, theme, {
      range: [-0.25, maximumCost],
      radius: 1.5,
      bands: 10,
      solid: [muted[0], muted[1], muted[2], 0.35]
    });
    origin.update(view, theme, {mode: -1, radius: 7, solid: [...theme.ink, 1], hillshade: 1});
    drawLayers(device, theme, [roads, origin]);
  };
  const scheduler = new DrawScheduler(draw);

  return {
    render() {
      scheduler.flush();
    },
    setParameters(update) {
      Object.assign(parameterValues, update);
      isDirty = true;
      scheduler.request();
    },
    resize(width, height) {
      view.width = width;
      view.height = height;
      scheduler.request();
    },
    setPointer(position) {
      if (!position) return;
      const [x, y] = getWorldPosition(view, position);
      const nextStart = findNearestNode(x, y);
      if (nextStart === startNode) return;
      startNode = nextStart;
      isDirty = true;
      scheduler.request();
    },
    destroy() {
      scheduler.cancel();
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

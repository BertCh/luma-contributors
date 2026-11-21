// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Layer} from '@deck.gl/core';
import {Buffer} from '@luma.gl/core';
import {
  GPUNetworkCostMatrix,
  GPUNetworkInhomogeneousK,
  GPUNetworkKernelDensity
} from '@luma-contributors/analysis/gpu-network';
import {GPUCommandGraph, type CompiledGPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {createGraphBufferImporter} from '../../engine/graph-buffers';
import {addKernelPass} from '../../engine/mode-kernels';
import {GraphSwap} from '../../engine/graph-swap';
import {formatCount, SpatialAnalysisResources} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import {SpatialAnalysisPointLayer, SpatialAnalysisSegmentLayer} from '../../engine/layers';
import type {SceneContext, SceneInstance} from '../scene';
import {createSeededRandom} from '../../engine/projection';
import {buildRoadGraph, formatLength, NO_EDGE} from './b10-road-graph';
import type {LoadedDataset} from '../../data/catalog';

export type RetailNetworkIntensityOptions = {
  category: 'retail' | 'restaurant_cafe' | 'grocery' | 'combined';
  maximumSnapDistance: number;
  sampleSeed: number;
  bandwidth: '300' | '600' | '1000' | '1500' | '2000' | '3000';
  maximumRadius: number;
  correction: 'ang' | 'none';
  normalization: 'network-length' | 'inverse-intensity' | 'inverse-intensity-squared';
  showPlaces: boolean;
};

const STUDY_BOUNDS: readonly [number, number, number, number] = [-87.67, 41.85, -87.61, 41.91];
const EVENT_COUNT = 96;
const BANDWIDTHS = [300, 600, 1000, 1500, 2000, 3000] as const;
const RADIUS_COUNT = 24;
const NODE_GRID_CELL = 200;
const CATEGORY_ROWS: Record<RetailNetworkIntensityOptions['category'], readonly number[]> = {
  restaurant_cafe: [0],
  grocery: [2],
  retail: [7],
  combined: [0, 2, 7]
};
const CATEGORY_NAMES: Record<RetailNetworkIntensityOptions['category'], string> = {
  restaurant_cafe: 'restaurants and cafés',
  grocery: 'grocery places',
  retail: 'retail places',
  combined: 'shops, cafés and groceries'
};
const DENSITY_BANDS = [0.2, 0.5, 1, 2, 4, 8];
const DENSITY_COLORS = [
  [133, 195, 219, 230],
  [98, 179, 192, 235],
  [119, 191, 141, 240],
  [220, 195, 94, 245],
  [236, 133, 67, 250],
  [189, 61, 83, 255],
  [137, 30, 85, 255]
] as const;

type LocalEdge = {
  source: number;
  target: number;
  length: number;
  sourceOriginal: number;
  edge: number;
};
type LocalNetwork = {
  nodeCount: number;
  edgeCount: number;
  nodePositions: Float32Array;
  offsets: Uint32Array;
  neighbors: Uint32Array;
  weights: Float32Array;
  edgeNodes: Uint32Array;
  edgeLengths: Float32Array;
  quadrature: Float32Array;
  segments: Float32Array;
  segmentNodeIndices: Uint32Array;
  networkLength: number;
  bounds: readonly [number, number, number, number];
  nodeOriginalIds: Uint32Array;
};
type PlaceCandidate = {row: number; node: number; distance: number; x: number; y: number};
type KBuild = {
  compiled: CompiledGPUCommandGraph<void>;
  reader: SummaryReader | null;
  k: Buffer;
  numerator: Buffer;
  normalization: Buffer;
  status: Buffer;
  bandwidth: number;
  correction: RetailNetworkIntensityOptions['correction'];
  normalizationMode: RetailNetworkIntensityOptions['normalization'];
  sampleGeneration: number;
};

function keyForNode(x: number, y: number): string {
  return `${Math.floor(x / NODE_GRID_CELL)},${Math.floor(y / NODE_GRID_CELL)}`;
}

function createLocalNetwork(
  roads: ReturnType<typeof buildRoadGraph>,
  roadDataset: LoadedDataset
): LocalNetwork {
  const rawNodes = roadDataset.column<Float32Array>('nodes');
  const graph = roads;
  const inside = new Uint8Array(graph.nodeCount);
  for (let node = 0; node < graph.nodeCount; node++) {
    const longitude = rawNodes[node * 2];
    const latitude = rawNodes[node * 2 + 1];
    inside[node] = Number(
      longitude >= STUDY_BOUNDS[0] &&
        longitude <= STUDY_BOUNDS[2] &&
        latitude >= STUDY_BOUNDS[1] &&
        latitude <= STUDY_BOUNDS[3]
    );
  }
  const originalEdges: LocalEdge[] = [];
  const adjacency: number[][] = Array.from({length: graph.nodeCount}, () => []);
  for (let edge = 0; edge < graph.edgeCount; edge++) {
    const reverse = graph.edgeReverse[edge];
    if (reverse !== NO_EDGE && edge > reverse) continue;
    const sourceOriginal = graph.edgeSource[edge];
    const targetOriginal = graph.edgeTarget[edge];
    if (!inside[sourceOriginal] || !inside[targetOriginal]) continue;
    const length = graph.edgeLength[edge];
    if (!Number.isFinite(length) || length <= 0) continue;
    const edgeIndex = originalEdges.length;
    originalEdges.push({
      source: sourceOriginal,
      target: targetOriginal,
      length,
      sourceOriginal,
      edge
    });
    adjacency[sourceOriginal].push(edgeIndex);
    adjacency[targetOriginal].push(edgeIndex);
  }
  const componentIds = new Int32Array(graph.nodeCount).fill(-1);
  const componentSizes: number[] = [];
  for (let start = 0; start < graph.nodeCount; start++) {
    if (!inside[start] || componentIds[start] >= 0 || adjacency[start].length === 0) continue;
    const component = componentSizes.length;
    const queue = [start];
    componentIds[start] = component;
    for (let head = 0; head < queue.length; head++) {
      const node = queue[head];
      for (const edgeIndex of adjacency[node]) {
        const edge = originalEdges[edgeIndex];
        const next = edge.source === node ? edge.target : edge.source;
        if (componentIds[next] >= 0) continue;
        componentIds[next] = component;
        queue.push(next);
      }
    }
    componentSizes.push(queue.length);
  }
  let largestComponent = -1;
  for (let component = 0; component < componentSizes.length; component++) {
    if (largestComponent < 0 || componentSizes[component] > componentSizes[largestComponent])
      largestComponent = component;
  }
  const originalToLocal = new Int32Array(graph.nodeCount).fill(-1);
  const nodeOriginalIds: number[] = [];
  for (let node = 0; node < graph.nodeCount; node++) {
    if (componentIds[node] !== largestComponent) continue;
    originalToLocal[node] = nodeOriginalIds.length;
    nodeOriginalIds.push(node);
  }
  const nodeCount = nodeOriginalIds.length;
  const edges: LocalEdge[] = [];
  for (const edge of originalEdges) {
    const source = originalToLocal[edge.source];
    const target = originalToLocal[edge.target];
    if (source < 0 || target < 0 || source === target) continue;
    edges.push({...edge, source, target});
  }
  const rows: Array<Array<{target: number; length: number}>> = Array.from(
    {length: nodeCount},
    () => []
  );
  const quadrature = new Float32Array(nodeCount);
  const edgeNodes = new Uint32Array(edges.length * 2);
  const edgeLengths = new Float32Array(edges.length);
  let networkLength = 0;
  for (let edge = 0; edge < edges.length; edge++) {
    const item = edges[edge];
    rows[item.source].push({target: item.target, length: item.length});
    rows[item.target].push({target: item.source, length: item.length});
    quadrature[item.source] += item.length / 2;
    quadrature[item.target] += item.length / 2;
    edgeNodes[edge * 2] = item.source;
    edgeNodes[edge * 2 + 1] = item.target;
    edgeLengths[edge] = item.length;
    networkLength += item.length;
  }
  const offsets = new Uint32Array(nodeCount + 1);
  for (let node = 0; node < nodeCount; node++) {
    rows[node].sort((first, second) => first.target - second.target);
    offsets[node + 1] = offsets[node] + rows[node].length;
  }
  const neighbors = new Uint32Array(offsets[nodeCount]);
  const weights = new Float32Array(offsets[nodeCount]);
  for (let node = 0; node < nodeCount; node++) {
    let index = offsets[node];
    for (const edge of rows[node]) {
      neighbors[index] = edge.target;
      weights[index] = edge.length;
      index++;
    }
  }
  const nodePositions = new Float32Array(nodeCount * 2);
  let minimumX = Infinity,
    minimumY = Infinity,
    maximumX = -Infinity,
    maximumY = -Infinity;
  for (let node = 0; node < nodeCount; node++) {
    const original = nodeOriginalIds[node];
    const x = graph.nodePositions[original * 2],
      y = graph.nodePositions[original * 2 + 1];
    nodePositions[node * 2] = x;
    nodePositions[node * 2 + 1] = y;
    minimumX = Math.min(minimumX, x);
    maximumX = Math.max(maximumX, x);
    minimumY = Math.min(minimumY, y);
    maximumY = Math.max(maximumY, y);
  }

  const localEdgeByOriginal = new Map<number, LocalEdge>();
  for (const edge of edges) localEdgeByOriginal.set(edge.edge, edge);
  const displayRows: number[] = [],
    displayNodeIndices: number[] = [];
  let activeEdge = -1,
    activeNodes: LocalEdge | undefined,
    totalGeometryLength = 0,
    halfGeometryLength = 0,
    cumulative = 0;
  const emitGeometry = (x0: number, y0: number, x1: number, y1: number, nodeIndex: number) => {
    if (Math.hypot(x1 - x0, y1 - y0) < 0.01) return;
    displayRows.push(x0, y0, x1, y1);
    displayNodeIndices.push(nodeIndex);
  };
  const finishEdge = () => {
    activeEdge = -1;
    activeNodes = undefined;
    totalGeometryLength = 0;
    halfGeometryLength = 0;
    cumulative = 0;
  };
  for (let segment = 0; segment < graph.segmentCount; segment++) {
    const originalEdge = graph.segmentEdge[segment];
    const localEdge = localEdgeByOriginal.get(originalEdge);
    if (!localEdge) {
      if (activeEdge >= 0) finishEdge();
      continue;
    }
    if (activeEdge !== originalEdge) {
      if (activeEdge >= 0) finishEdge();
      activeEdge = originalEdge;
      activeNodes = localEdge;
      let end = segment;
      while (end < graph.segmentCount && graph.segmentEdge[end] === originalEdge) {
        const offset = end * 4;
        totalGeometryLength += Math.hypot(
          graph.segments[offset + 2] - graph.segments[offset],
          graph.segments[offset + 3] - graph.segments[offset + 1]
        );
        end++;
      }
      halfGeometryLength = totalGeometryLength / 2;
    }
    if (!activeNodes) continue;
    const offset = segment * 4;
    const x0 = graph.segments[offset],
      y0 = graph.segments[offset + 1],
      x1 = graph.segments[offset + 2],
      y1 = graph.segments[offset + 3];
    const length = Math.hypot(x1 - x0, y1 - y0);
    const sourceNode = localEdge.source;
    const targetNode = localEdge.target;
    if (cumulative + length <= halfGeometryLength || length === 0) {
      emitGeometry(x0, y0, x1, y1, sourceNode);
    } else if (cumulative >= halfGeometryLength) {
      emitGeometry(x0, y0, x1, y1, targetNode);
    } else {
      const fraction = (halfGeometryLength - cumulative) / length;
      const middleX = x0 + (x1 - x0) * fraction,
        middleY = y0 + (y1 - y0) * fraction;
      emitGeometry(x0, y0, middleX, middleY, sourceNode);
      emitGeometry(middleX, middleY, x1, y1, targetNode);
    }
    cumulative += length;
  }
  if (activeEdge >= 0) finishEdge();
  return {
    nodeCount,
    edgeCount: edges.length,
    nodePositions,
    offsets,
    neighbors,
    weights,
    edgeNodes,
    edgeLengths,
    quadrature,
    segments: Float32Array.from(displayRows),
    segmentNodeIndices: Uint32Array.from(displayNodeIndices),
    networkLength,
    bounds: [minimumX, minimumY, maximumX, maximumY] as const,
    nodeOriginalIds: Uint32Array.from(nodeOriginalIds)
  };
}

function findNearestNode(
  x: number,
  y: number,
  grid: Map<string, number[]>,
  network: LocalNetwork,
  maximumDistance: number
): {node: number; distance: number} {
  const radius = Math.ceil(maximumDistance / NODE_GRID_CELL);
  const column = Math.floor(x / NODE_GRID_CELL),
    row = Math.floor(y / NODE_GRID_CELL);
  let node = -1,
    distance = maximumDistance;
  for (let dx = -radius; dx <= radius; dx++)
    for (let dy = -radius; dy <= radius; dy++) {
      for (const candidate of grid.get(`${column + dx},${row + dy}`) ?? []) {
        const distanceX = x - network.nodePositions[candidate * 2];
        const distanceY = y - network.nodePositions[candidate * 2 + 1];
        const candidateDistance = Math.hypot(distanceX, distanceY);
        if (candidateDistance < distance) {
          distance = candidateDistance;
          node = candidate;
        }
      }
    }
  return {node, distance};
}

/** GPU network intensity and intensity-reweighted pair statistics over a bounded real street subgraph. */
export async function createRetailNetworkIntensity(
  ctx: SceneContext<RetailNetworkIntensityOptions>
): Promise<SceneInstance<RetailNetworkIntensityOptions>> {
  const roadDataset = ctx.datasets.get('chicago-roads');
  const placeDataset = ctx.datasets.get('chicago-places');
  const origin = roadDataset.defaultOrigin;
  const roadNodePositions = roadDataset.column<Float32Array>('nodes');
  const completeRoads = buildRoadGraph(roadDataset, origin);
  const network = createLocalNetwork(completeRoads, roadDataset);
  const queryCount = network.nodeCount + EVENT_COUNT;
  const placePositions = placeDataset.projectColumn('position', origin);
  const placeCategories = placeDataset.column<Uint8Array>('category');
  const placeCount = placeCategories.length;
  const grid = new Map<string, number[]>();
  for (let node = 0; node < network.nodeCount; node++) {
    const key = keyForNode(network.nodePositions[node * 2], network.nodePositions[node * 2 + 1]);
    const values = grid.get(key) ?? [];
    values.push(node);
    grid.set(key, values);
  }
  const eventCandidates = new Map<number, PlaceCandidate[]>();
  for (const category of [0, 2, 7]) eventCandidates.set(category, []);
  for (let row = 0; row < placeCount; row++) {
    const category = placeCategories[row];
    const list = eventCandidates.get(category);
    if (!list) continue;
    const x = placePositions[row * 2],
      y = placePositions[row * 2 + 1];
    if (
      x < network.bounds[0] ||
      x > network.bounds[2] ||
      y < network.bounds[1] ||
      y > network.bounds[3]
    )
      continue;
    const nearest = findNearestNode(x, y, grid, network, 200);
    if (nearest.node >= 0) list.push({row, node: nearest.node, distance: nearest.distance, x, y});
  }
  const resources = new SpatialAnalysisResources(ctx.device, 'retail-network-intensity');
  const graph = new GPUCommandGraph<void>(ctx.device, {id: 'retail-network-intensity'});
  const view = createGraphBufferImporter(graph);
  const topologyBuffers = {
    offsets: resources.createBuffer('local-network-offsets', network.offsets),
    neighbors: resources.createBuffer('local-network-neighbors', network.neighbors),
    weights: resources.createBuffer('local-network-weights', network.weights),
    nodePositions: resources.createBuffer('local-network-node-positions', network.nodePositions),
    edgeNodes: resources.createBuffer('local-network-unique-edge-nodes', network.edgeNodes),
    edgeLengths: resources.createBuffer('local-network-unique-edge-lengths', network.edgeLengths),
    quadrature: resources.createBuffer('edge-length-trapezoid-weights', network.quadrature),
    segments: resources.createBuffer('density-segments', network.segments, Buffer.VERTEX),
    segmentNodeIndices: resources.createBuffer(
      'density-segment-node-indices',
      network.segmentNodeIndices
    ),
    eventNodes: resources.createBuffer('sampled-event-network-nodes', new Uint32Array(EVENT_COUNT)),
    originalPlaces: resources.createBuffer(
      'sampled-original-place-positions',
      new Float32Array(EVENT_COUNT * 2)
    ),
    snappedPlaces: resources.createBuffer(
      'sampled-snapped-network-positions',
      new Float32Array(EVENT_COUNT * 2),
      Buffer.VERTEX
    ),
    radii: resources.createBuffer('inhomogeneous-k-radii', new Float32Array(RADIUS_COUNT))
  };
  const queryQuadrature = new Float32Array(queryCount);
  queryQuadrature.set(network.quadrature);
  const queryQuadratureBuffer = resources.createBuffer(
    'node-and-event-query-quadrature',
    queryQuadrature
  );
  const excludedEventIds = new Uint32Array(queryCount).fill(0xffffffff);
  for (let event = 0; event < EVENT_COUNT; event++)
    excludedEventIds[network.nodeCount + event] = event;
  const excludedEventIdsBuffer = resources.createBuffer(
    'leave-one-out-event-ids',
    excludedEventIds
  );
  const offsetsView = view(topologyBuffers.offsets, 'uint32', network.nodeCount + 1);
  const neighborsView = view(
    topologyBuffers.neighbors,
    'uint32',
    network.offsets[network.nodeCount]
  );
  const weightsView = view(topologyBuffers.weights, 'float32', network.offsets[network.nodeCount]);
  const eventNodesView = view(topologyBuffers.eventNodes, 'uint32', EVENT_COUNT);
  const eventCosts = resources.createBuffer(
    'event-to-all-node-costs',
    EVENT_COUNT * network.nodeCount * 4
  );
  const converged = resources.createBuffer('complete-distance-convergence', 4);
  const costsComplete = view(converged, 'uint32', 1);
  const costMatrix = new GPUNetworkCostMatrix({
    id: 'retail-intensity-complete-event-costs',
    topology: {
      offsets: offsetsView,
      neighbors: neighborsView,
      weights: weightsView,
      nodePositions: view(topologyBuffers.nodePositions, 'float32x2', network.nodeCount)
    },
    startNodes: eventNodesView,
    laneCount: EVENT_COUNT,
    maximumIterations: Math.ceil(network.nodeCount / 12) + 8,
    localIterations: 16,
    output: {
      costs: view(eventCosts, 'float32', EVENT_COUNT * network.nodeCount),
      status: {converged: costsComplete}
    }
  });
  graph.add(costMatrix);
  const queryDistances = resources.createBuffer(
    'node-and-event-to-event-query-major-distances',
    EVENT_COUNT * queryCount * 4
  );
  addKernelPass(graph, {
    id: 'transpose-complete-network-costs-for-kde',
    bindings: [
      {
        name: 'eventCosts',
        view: view(eventCosts, 'float32', EVENT_COUNT * network.nodeCount),
        type: 'f32',
        access: 'read'
      },
      {name: 'eventNodes', view: eventNodesView, type: 'u32', access: 'read'},
      {
        name: 'queryCosts',
        view: view(queryDistances, 'float32', EVENT_COUNT * queryCount),
        type: 'f32',
        access: 'read_write'
      }
    ],
    invocationCount: EVENT_COUNT * queryCount,
    declarations: `const EVENT_COUNT: u32 = ${EVENT_COUNT}u; const NODE_COUNT: u32 = ${network.nodeCount}u; const QUERY_COUNT: u32 = ${queryCount}u;`,
    body: 'let query = index / EVENT_COUNT; let event = index % EVENT_COUNT; let node = select(eventNodes[eventNodesOffset + event], query, query < NODE_COUNT); let sourceEvent = select(query - NODE_COUNT, event, query < NODE_COUNT); queryCosts[queryCostsOffset + index] = eventCosts[eventCostsOffset + sourceEvent * NODE_COUNT + node];'
  });
  const baseVariants = new Map<
    number,
    {density: Buffer; scaled: Buffer; mass: Buffer; densityValidity: Buffer; eventIntensity: Buffer}
  >();
  for (const bandwidth of BANDWIDTHS) {
    const density = resources.createBuffer(
      `density-${bandwidth}m-node-and-event-queries`,
      queryCount * 4
    );
    const scaled = resources.createBuffer(`density-${bandwidth}m-per-km`, network.nodeCount * 4);
    const mass = resources.createBuffer(`density-${bandwidth}m-event-kernel-mass`, EVENT_COUNT * 4);
    const densityValidity = resources.createBuffer(
      `density-${bandwidth}m-query-validity`,
      queryCount * 4
    );
    const eventIntensity = resources.createBuffer(
      `density-${bandwidth}m-event-intensities`,
      EVENT_COUNT * 4
    );
    const densityProducer = new GPUNetworkKernelDensity({
      id: `network-kernel-density-${bandwidth}m`,
      distances: view(queryDistances, 'float32', EVENT_COUNT * queryCount),
      costsComplete,
      eventCount: EVENT_COUNT,
      quadratureLengths: view(queryQuadratureBuffer, 'float32', queryCount),
      bandwidth,
      excludedEventIds: view(excludedEventIdsBuffer, 'uint32', queryCount),
      output: {
        intensity: view(density, 'float32', queryCount),
        kernelMass: view(mass, 'float32', EVENT_COUNT),
        validity: view(densityValidity, 'uint32', queryCount)
      }
    });
    graph.add(densityProducer);
    addKernelPass(graph, {
      id: `scale-density-${bandwidth}m-to-per-kilometre`,
      bindings: [
        {name: 'density', view: view(density, 'float32', queryCount), type: 'f32', access: 'read'},
        {
          name: 'display',
          view: view(scaled, 'float32', network.nodeCount),
          type: 'f32',
          access: 'read_write'
        }
      ],
      invocationCount: network.nodeCount,
      body: 'display[displayOffset + index] = density[densityOffset + index] * 1000.0;'
    });
    addKernelPass(graph, {
      id: `gather-${bandwidth}m-event-intensity`,
      bindings: [
        {name: 'density', view: view(density, 'float32', queryCount), type: 'f32', access: 'read'},
        {
          name: 'eventIntensity',
          view: view(eventIntensity, 'float32', EVENT_COUNT),
          type: 'f32',
          access: 'read_write'
        }
      ],
      invocationCount: EVENT_COUNT,
      body: `eventIntensity[eventIntensityOffset + index] = density[densityOffset + ${network.nodeCount}u + index];`
    });
    baseVariants.set(bandwidth, {density, scaled, mass, densityValidity, eventIntensity});
  }
  const compiledBase = resources.track(await graph.compileAsync());
  const radii = new Float32Array(RADIUS_COUNT);
  const writeRadii = () => {
    for (let index = 0; index < RADIUS_COUNT; index++)
      radii[index] = ((index + 1) * ctx.options.maximumRadius) / RADIUS_COUNT;
    topologyBuffers.radii.write(radii);
  };
  writeRadii();

  const graphSwap = resources.track(new GraphSwap());
  let activeK: KBuild | null = null;
  let nextBuildId = 1;
  let activeDirty = true;
  let baseDirty = true;
  let destroyed = false;
  let ready = false;
  let activeDensityValues = new Float32Array(network.nodeCount);
  let activeRadii = radii.slice();
  let sampleGeneration = 0;
  let analysisGeneration = 0;
  let pendingReadGeneration = 0;
  let statusText = 'calculating complete network costs';
  const releaseBuild = (build: KBuild) => {
    build.reader?.stop();
    resources.release(build.compiled);
    for (const buffer of [build.k, build.numerator, build.normalization, build.status])
      resources.release(buffer);
  };
  const refreshSummary = (build: KBuild, bytes: ArrayBuffer) => {
    if (
      destroyed ||
      activeK !== build ||
      sampleGeneration !== build.sampleGeneration ||
      pendingReadGeneration !== analysisGeneration ||
      build.bandwidth !== Number(ctx.options.bandwidth) ||
      build.correction !== ctx.options.correction ||
      build.normalizationMode !== ctx.options.normalization
    )
      return;
    const words = new Uint32Array(bytes),
      floats = new Float32Array(bytes);
    let offset = 0;
    const readWords = (length: number) => {
      const values = words.slice(offset, offset + length);
      offset += length;
      return values;
    };
    const readFloats = (length: number) => {
      const values = floats.slice(offset, offset + length);
      offset += length;
      return values;
    };
    const costConvergence = readWords(1)[0];
    const allDensityValues = readFloats(queryCount);
    const densityValues = allDensityValues.subarray(0, network.nodeCount);
    const kernelMass = readFloats(EVENT_COUNT);
    const densityValidity = readWords(queryCount);
    const eventIntensity = readFloats(EVENT_COUNT);
    const kValues = readFloats(RADIUS_COUNT);
    const normalizationValues = readFloats(3);
    const kStatus = readWords(RADIUS_COUNT);
    const nodeDensityValidity = densityValidity.subarray(0, network.nodeCount);
    const eventDensityValidity = densityValidity.subarray(network.nodeCount);
    const densityValid =
      costConvergence !== 0 &&
      nodeDensityValidity.every(word => word !== 0) &&
      kernelMass.every(value => Number.isFinite(value) && value > 0) &&
      densityValues.every(Number.isFinite);
    const positiveEventIntensity = eventIntensity.filter(
      (value, index) => eventDensityValidity[index] !== 0 && Number.isFinite(value) && value > 0
    );
    const eventIntensityValid = positiveEventIntensity.length === EVENT_COUNT;
    const kValid =
      densityValid &&
      eventIntensityValid &&
      kStatus.every(flag => flag === 0) &&
      kValues.every(Number.isFinite);
    ready = densityValid;
    activeDensityValues = densityValues;
    activeRadii = radii.slice();
    statusText = !costConvergence
      ? 'complete network cost matrix did not converge'
      : !densityValid
        ? `network density withheld · ${nodeDensityValidity.filter(word => word === 0).length} invalid nodes`
        : !kValid
          ? `${positiveEventIntensity.length}/${EVENT_COUNT} leave-one-out intensities positive · K withheld`
          : 'complete distances · positive leave-one-out intensity · all K radii valid';
    const sortedIntensity = Array.from(positiveEventIntensity, value => value * 1000).sort(
      (first, second) => first - second
    );
    const integratedIntensity = densityValues.reduce(
      (total, value, index) => total + value * network.quadrature[index],
      0
    );
    ctx.setReadout(
      'intensitySummary',
      densityValid
        ? `${sortedIntensity.length}/${EVENT_COUNT} positive at sampled sites · ${sortedIntensity.length ? `median ${sortedIntensity[Math.floor(sortedIntensity.length / 2)].toFixed(2)} places/km` : 'none positive'}`
        : 'withheld until complete node density is available'
    );
    ctx.setReadout(
      'kernelIntegral',
      densityValid
        ? `${integratedIntensity.toFixed(1)} mapped events over ${EVENT_COUNT} sampled events`
        : 'withheld'
    );
    ctx.setReadout(
      'normalizationSummary',
      `network ${(normalizationValues[0] / 1000).toFixed(1)} km · inverse-intensity sum ${normalizationValues[1].toPrecision(3)} m · denominator ${normalizationValues[2].toPrecision(3)} m`
    );
    ctx.setReadout('solverStatus', statusText);
    const hasIdentityReference =
      build.correction === 'ang' && build.normalizationMode === 'network-length';
    ctx.setReadout(
      'kEndpoint',
      kValid
        ? `${(kValues.at(-1)! / 1000).toPrecision(4)} km at ${(ctx.options.maximumRadius / 1000).toFixed(2)} km${hasIdentityReference ? ` · identity ${(ctx.options.maximumRadius / 1000).toFixed(3)} km` : ''}`
        : 'K curve withheld'
    );
    ctx.setChart(
      'inhomogeneousK',
      kValid
        ? {
            kind: 'line',
            xLabel: 'network distance (km)',
            yLabel: `${build.correction === 'ang' ? 'Ang-corrected' : 'Uncorrected'} inhomogeneous K (km)`,
            series: [
              {
                label: 'estimated inhomogeneous K',
                x: Array.from(activeRadii, value => value / 1000),
                y: Array.from(kValues, value => value / 1000),
                color: 0,
                width: 2.4
              },
              ...(hasIdentityReference
                ? [
                    {
                      label: 'complete-random identity',
                      x: Array.from(activeRadii, value => value / 1000),
                      y: Array.from(activeRadii, value => value / 1000),
                      color: 1,
                      dashed: true
                    }
                  ]
                : [])
            ],
            description: hasIdentityReference
              ? 'Observed intensity-reweighted network K compared with the geometrically corrected complete-random identity. Descriptive; no simulation envelope.'
              : 'Intensity-reweighted network K under the selected correction/normalization. Descriptive; no simulation envelope.'
          }
        : null
    );
    ctx.requestLayers();
  };
  async function buildKGraph(
    bandwidth: number,
    correction: RetailNetworkIntensityOptions['correction'],
    normalizationMode: RetailNetworkIntensityOptions['normalization']
  ): Promise<KBuild> {
    const serial = nextBuildId++;
    const currentSampleGeneration = sampleGeneration;
    const kGraph = new GPUCommandGraph<void>(ctx.device, {
      id: `retail-network-inhomogeneous-k-${serial}`
    });
    const viewK = createGraphBufferImporter(kGraph);
    const k = resources.createBuffer(`retail-k-${serial}`, RADIUS_COUNT * 4);
    const numerator = resources.createBuffer(`retail-k-numerator-${serial}`, RADIUS_COUNT * 4);
    const normalization = resources.createBuffer(`retail-k-normalization-${serial}`, 3 * 4);
    const status = resources.createBuffer(`retail-k-status-${serial}`, RADIUS_COUNT * 4);
    const inhomogeneousK = new GPUNetworkInhomogeneousK({
      id: `retail-network-inhomogeneous-k-${serial}`,
      eventNodeIds: viewK(topologyBuffers.eventNodes, 'uint32', EVENT_COUNT),
      intensity: viewK(baseVariants.get(bandwidth)!.eventIntensity, 'float32', EVENT_COUNT),
      nodeDistances: viewK(eventCosts, 'float32', EVENT_COUNT * network.nodeCount),
      nodeCount: network.nodeCount,
      edgeNodes: viewK(topologyBuffers.edgeNodes, 'uint32x2', network.edgeCount),
      edgeLengths: viewK(topologyBuffers.edgeLengths, 'float32', network.edgeCount),
      radii: viewK(topologyBuffers.radii, 'float32', RADIUS_COUNT),
      correction,
      normalization: normalizationMode,
      maximumPairCount: EVENT_COUNT * EVENT_COUNT,
      output: {
        k: viewK(k, 'float32', RADIUS_COUNT),
        numerator: viewK(numerator, 'float32', RADIUS_COUNT),
        normalization: viewK(normalization, 'float32', 3),
        status: viewK(status, 'uint32', RADIUS_COUNT)
      }
    });
    kGraph.add(inhomogeneousK);
    const compiled = resources.track(await kGraph.compileAsync());
    const sources = [
      converged,
      baseVariants.get(bandwidth)!.density,
      baseVariants.get(bandwidth)!.mass,
      baseVariants.get(bandwidth)!.densityValidity,
      baseVariants.get(bandwidth)!.eventIntensity,
      k,
      normalization,
      status
    ].map(buffer => ({buffer, size: buffer.byteLength}));
    const build: KBuild = {
      compiled,
      reader: null,
      k,
      numerator,
      normalization,
      status,
      bandwidth,
      correction,
      normalizationMode,
      sampleGeneration: currentSampleGeneration
    };
    build.reader = new SummaryReader(resources, `retail-k-summary-${serial}`, sources, bytes =>
      refreshSummary(build, bytes)
    );
    return build;
  }
  const clearResult = () => {
    analysisGeneration++;
    ready = false;
    ctx.setReadout('solverStatus', 'updating network analysis');
    ctx.setReadout('intensitySummary', 'updating network intensity');
    ctx.setReadout('kernelIntegral', 'updating');
    ctx.setReadout('normalizationSummary', 'updating');
    ctx.setReadout('kEndpoint', 'updating');
    ctx.setChart('inhomogeneousK', null);
    ctx.requestLayers();
  };
  const rebuildK = async () => {
    const bandwidth = Number(ctx.options.bandwidth);
    const correction = ctx.options.correction;
    const normalizationMode = ctx.options.normalization;
    clearResult();
    const requestedAnalysisGeneration = analysisGeneration;
    try {
      await graphSwap.request({
        build: () => buildKGraph(bandwidth, correction, normalizationMode),
        commit: next => {
          const previous = activeK;
          next.sampleGeneration = sampleGeneration;
          activeK = next;
          activeDirty = true;
          if (previous) releaseBuild(previous);
          ctx.requestLayers();
        },
        discard: releaseBuild
      });
    } catch {
      if (requestedAnalysisGeneration === analysisGeneration && !destroyed) {
        ctx.setReadout('solverStatus', 'K graph rebuild failed; requested analysis withheld');
      }
    }
  };
  let selectedEventCount = 0;
  let medianSnapDistance = 0;
  let maximumSnapDistance = 0;
  function writeSample(): void {
    sampleGeneration++;
    const categoryCodes = CATEGORY_ROWS[ctx.options.category];
    const candidates = categoryCodes
      .flatMap(code => eventCandidates.get(code) ?? [])
      .filter(event => event.distance <= ctx.options.maximumSnapDistance);
    const random = createSeededRandom(ctx.options.sampleSeed * 7919 + 13);
    const eligible = candidates.filter(event => event.distance <= ctx.options.maximumSnapDistance);
    const shuffled = eligible.slice();
    for (let index = shuffled.length - 1; index > 0; index--) {
      const swap = Math.floor(random() * (index + 1));
      [shuffled[index], shuffled[swap]] = [shuffled[swap], shuffled[index]];
    }
    const chosen = shuffled.slice(0, EVENT_COUNT);
    selectedEventCount = chosen.length;
    const eventNodes = new Uint32Array(EVENT_COUNT).fill(0xffffffff);
    const original = new Float32Array(EVENT_COUNT * 2).fill(NaN);
    const snapped = new Float32Array(EVENT_COUNT * 2).fill(NaN);
    for (let index = 0; index < EVENT_COUNT; index++) {
      const event = chosen[index];
      if (!event) continue;
      eventNodes[index] = event.node;
      original[index * 2] = event.x;
      original[index * 2 + 1] = event.y;
      snapped[index * 2] = network.nodePositions[event.node * 2];
      snapped[index * 2 + 1] = network.nodePositions[event.node * 2 + 1];
    }
    topologyBuffers.eventNodes.write(eventNodes);
    topologyBuffers.originalPlaces.write(original);
    topologyBuffers.snappedPlaces.write(snapped);
    const snapDistances = chosen
      .map(event => event.distance)
      .sort((first, second) => first - second);
    medianSnapDistance = snapDistances[Math.floor(snapDistances.length / 2)] ?? 0;
    maximumSnapDistance = snapDistances.at(-1) ?? 0;
    ctx.setReadout(
      'eventSample',
      `${formatCount(selectedEventCount)} of ${formatCount(eligible.length)} eligible ${CATEGORY_NAMES[ctx.options.category]} · sample ${ctx.options.sampleSeed}`
    );
    ctx.setReadout(
      'snapSummary',
      `${ctx.options.maximumSnapDistance} m limit · median ${medianSnapDistance.toFixed(1)} m · max ${maximumSnapDistance.toFixed(1)} m`
    );
    baseDirty = true;
    clearResult();
    if (activeK) {
      activeK.sampleGeneration = sampleGeneration;
      activeDirty = true;
      activeK.reader?.markStale();
    }
  }
  const initialK = await buildKGraph(
    Number(ctx.options.bandwidth),
    ctx.options.correction,
    ctx.options.normalization
  );
  activeK = initialK;
  writeSample();
  ctx.setReadout(
    'networkSummary',
    `${formatCount(network.nodeCount)} nodes · ${formatCount(network.edgeCount)} undirected edges · ${formatLength(network.networkLength)} · largest component in study box`
  );
  ctx.setReadout(
    'quadratureSummary',
    `${network.edgeCount} unique edges · half-edge length at each endpoint · total weight ${formatLength(network.quadrature.reduce((sum, value) => sum + value, 0))}`
  );
  ctx.setReadout('studyBox', 'Downtown Chicago local graph · [-87.67, 41.85, -87.61, 41.91]');
  return {
    getCompiledGraphs: () => [compiledBase, ...(activeK ? [activeK.compiled] : [])],
    encode(encoder) {
      if (baseDirty) {
        compiledBase.encode(encoder, {parameters: undefined});
        baseDirty = false;
      }
      if (activeK) {
        if (activeDirty) {
          activeK.compiled.encode(encoder, {parameters: undefined});
          activeDirty = false;
          activeK.reader?.markStale();
        }
        if (activeK.reader && !activeK.reader.isPending) pendingReadGeneration = analysisGeneration;
        activeK.reader?.flush(encoder);
      }
    },
    setOption(id) {
      if (id === 'category' || id === 'maximumSnapDistance' || id === 'sampleSeed') writeSample();
      if (id === 'maximumRadius') {
        writeRadii();
        activeRadii = radii.slice();
        activeDirty = true;
        activeK?.reader?.markStale();
        clearResult();
      }
      if (id === 'bandwidth' || id === 'correction' || id === 'normalization') void rebuildK();
      if (id === 'showPlaces') ctx.requestLayers();
    },
    getLayers() {
      const layers: Layer[] = [
        new SpatialAnalysisSegmentLayer({
          id: 'retail-study-street-context',
          coordinateOrigin: [origin[0], origin[1], 0],
          segments: topologyBuffers.segments,
          instanceCount: topologyBuffers.segments.byteLength / 16,
          widthPixels: 0.8,
          color: [97, 111, 128, 90]
        }),
        ...(ready
          ? [
              new SpatialAnalysisSegmentLayer({
                id: 'retail-network-density-km',
                coordinateOrigin: [origin[0], origin[1], 0],
                segments: topologyBuffers.segments,
                instanceCount: topologyBuffers.segments.byteLength / 16,
                values: baseVariants.get(Number(ctx.options.bandwidth))!.scaled,
                valueFormat: 'float32',
                valueIndices: topologyBuffers.segmentNodeIndices,
                classBreaks: DENSITY_BANDS,
                classColors: DENSITY_COLORS,
                valueDivisor: 1,
                widthPixels: 3.2,
                dimOpacity: 0
              })
            ]
          : []),
        ...(ctx.options.showPlaces
          ? [
              new SpatialAnalysisPointLayer({
                id: 'retail-original-mapped-places',
                coordinateOrigin: [origin[0], origin[1], 0],
                positions: topologyBuffers.originalPlaces,
                instanceCount: selectedEventCount,
                radiusPixels: 3.4,
                shape: 'circle',
                color: [255, 225, 125, 220],
                outlineColor: [16, 24, 35, 220],
                outlineWidthPixels: 0.7
              }),
              new SpatialAnalysisPointLayer({
                id: 'retail-snapped-network-events',
                coordinateOrigin: [origin[0], origin[1], 0],
                positions: topologyBuffers.snappedPlaces,
                instanceCount: selectedEventCount,
                radiusPixels: 2.4,
                shape: 'ring',
                color: [41, 219, 205, 240]
              })
            ]
          : [])
      ];
      return layers;
    },
    getTooltip(event) {
      const viewport = ctx.getViewport();
      if (!viewport || !ready) return null;
      let nearest = -1,
        best = 16 * 16;
      for (let node = 0; node < network.nodeCount; node++) {
        const originalNode = network.nodeOriginalIds[node];
        const point = viewport.project([
          roadNodePositions[originalNode * 2],
          roadNodePositions[originalNode * 2 + 1]
        ]);
        const distance = (point[0] - event.pixel[0]) ** 2 + (point[1] - event.pixel[1]) ** 2;
        if (distance < best) {
          best = distance;
          nearest = node;
        }
      }
      if (nearest < 0) return null;
      return {
        title: `Street intersection ${network.nodeOriginalIds[nearest]}`,
        subtitle: `${CATEGORY_NAMES[ctx.options.category]} · ${ctx.options.bandwidth} m bandwidth`,
        rows: [
          {
            label: 'Estimated network intensity',
            value: `${(activeDensityValues[nearest] * 1000).toFixed(2)} mapped places/km`
          },
          {
            label: 'Network quadrature weight',
            value: `${network.quadrature[nearest].toFixed(1)} m`
          },
          {label: 'K correction', value: ctx.options.correction},
          {label: 'Normalization', value: ctx.options.normalization}
        ]
      };
    },
    destroy() {
      destroyed = true;
      graphSwap.destroy();
      if (activeK) releaseBuild(activeK);
      resources.destroy();
    }
  };
}

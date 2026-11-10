// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Helpers shared by the facility and accessibility runs: a seeded service scene (facilities and
// demand locations snapped to intersections), and the real network operations that feed them, a
// facility-by-node cost matrix and its decay weights.

import {GPUNetworkCostMatrix, GPUNetworkCostWeights} from '@luma-contributors/analysis/gpu-network';
import type {RunBuffer, RunBuilder} from '../../../define-run';
import type {NetworkInput, World} from '../../../inputs';
import type {DemoTheme} from '../../../../demos/kit';
import type {DotsRunLayer} from '../../../define-run';
import {createRoads, gatherPositions, pickNodes} from './roads';
import {uploadNetworkTopology} from '../../../inputs';
import {createPointIndex, type PointIndex} from '../../../pick';

export type AccessScene = {
  network: NetworkInput;
  facilityNodes: Uint32Array;
  demandNodes: Uint32Array;
  facilityPositions: Float32Array;
  demandPositions: Float32Array;
  /** Facility supply (beds, jobs, floor area). */
  supply: Float32Array;
  /** Demand per location (population). */
  population: Float32Array;
};

/** Scenario facilities and demand locations on the San Francisco street network. */
export function createAccessScene(
  random: () => number,
  world: World,
  facilityCount = 8,
  demandCount = 160
): AccessScene {
  const network = createRoads(random, world, 14);
  const facilityNodes = pickNodes(network, world, facilityCount, 2);
  const demandNodes = Uint32Array.from({length: demandCount}, (_, index) =>
    network.findNearestNode(random() * world.width, random())
  );
  return {
    network,
    facilityNodes,
    demandNodes,
    facilityPositions: gatherPositions(network.nodes, facilityNodes),
    demandPositions: gatherPositions(network.nodes, demandNodes),
    supply: Float32Array.from({length: facilityCount}, () => 0.5 + random() * 2),
    population: Float32Array.from({length: demandCount}, () => 1 + random() * 9)
  };
}

/** Topology plus a one-row `costLimit` buffer. */
export function uploadSearch(builder: RunBuilder, network: NetworkInput) {
  const {topology, weights} = uploadNetworkTopology(builder, network);
  return {topology, weights, costLimit: builder.allocate('cost-limit', 'float32', 1)};
}

/** Node lists for the matrix: fixed contents (an array) or a buffer rewritten each frame. */
type NodeList = Uint32Array | RunBuffer<'uint32'>;

/** Adds GPUNetworkCostMatrix: one row of costs per start node, to every node or each target. */
export function addCostMatrix(
  builder: RunBuilder,
  search: ReturnType<typeof uploadSearch>,
  startNodes: NodeList,
  nodeCount: number,
  targetNodes?: NodeList
): RunBuffer<'float32'> {
  const toBuffer = (id: string, nodes: NodeList) =>
    nodes instanceof Uint32Array ? builder.upload(id, 'uint32', nodes) : nodes;
  const starts = toBuffer('start-nodes', startNodes);
  const targets = targetNodes ? toBuffer('target-nodes', targetNodes) : undefined;
  const matrix = builder.allocate(
    'cost-matrix',
    'float32',
    starts.length * (targets?.length ?? nodeCount)
  );
  builder.graph.add(
    new GPUNetworkCostMatrix({
      topology: search.topology,
      startNodes: starts.view,
      targetNodes: targets?.view,
      costLimit: search.costLimit.view,
      maximumIterations: 48,
      output: {costs: matrix.view}
    })
  );
  return matrix;
}

/** Adds GPUNetworkCostWeights over a matrix; `demandNodes` map weights neighbors to locations. */
export function addCostWeights(
  builder: RunBuilder,
  matrix: RunBuffer<'float32'>,
  rowCount: number,
  demandNodes: Uint32Array | undefined,
  orientation: 'rows' | 'columns',
  parameters: RunBuffer<'float32'>
) {
  const locationCount = demandNodes?.length ?? matrix.length / rowCount;
  const weightsRowCount = orientation === 'rows' ? rowCount : locationCount;
  const capacity = rowCount * locationCount;
  const offsets = builder.allocate(`${orientation}-offsets`, 'uint32', weightsRowCount + 1);
  const neighbors = builder.allocate(`${orientation}-neighbors`, 'uint32', capacity);
  const weights = builder.allocate(`${orientation}-weights`, 'float32', capacity);
  const distances = builder.allocate(`${orientation}-distances`, 'float32', capacity);
  const operation = new GPUNetworkCostWeights({
    id: `cost-weights-${orientation}`,
    costs: matrix.view,
    rowCount,
    columns: demandNodes
      ? builder.upload(`${orientation}-columns`, 'uint32', demandNodes).view
      : undefined,
    orientation,
    parameters: parameters.view,
    output: {
      offsets: offsets.view,
      neighbors: neighbors.view,
      weights: weights.view,
      distances: distances.view
    }
  });
  builder.graph.add(operation);
  return {
    operation,
    offsets,
    neighbors,
    weights,
    distances,
    spatialWeights: operation.outputWeights
  };
}

/** Decay options shared by the sliders: index into this list. */
export const DECAY_NAMES = ['none', 'exponential', 'power', 'gaussian'] as const;

// ---------------------------------------------------------------------------------------------
// Dragging a facility

/** What the facility drag moves: the intersections and positions the cost matrix reads. */
export type FacilityScene = {
  network: NetworkInput;
  facilityNodes: Uint32Array;
  facilityPositions: Float32Array;
};

/** Default facility sites and the lookup that moves one of them to the pointer's intersection. */
export type FacilityDrag = {
  defaultNodes: Uint32Array;
  defaultPositions: Float32Array;
  nodeIndex: PointIndex;
  /** Facility moved by the last pointer position, or -1. */
  active: number;
  node: number;
};

export function createFacilityDrag(scene: FacilityScene): FacilityDrag {
  return {
    defaultNodes: Uint32Array.from(scene.facilityNodes),
    defaultPositions: Float32Array.from(scene.facilityPositions),
    nodeIndex: createPointIndex(scene.network.nodes, scene.network.nodeCount),
    active: -1,
    node: -1
  };
}

/**
 * Moves the facility whose default site is nearest the pointer to the intersection nearest the
 * pointer; every other facility stays at its default site, so the result depends only on the pointer.
 * Returns whether the scene changed, which is when the operation has to run again.
 */
export function moveFacility(
  scene: FacilityScene,
  drag: FacilityDrag,
  x: number,
  y: number,
  count = scene.facilityNodes.length
): {facility: number; node: number; changed: boolean} {
  const distanceTo = (facility: number) =>
    Math.hypot(
      drag.defaultPositions[facility * 2] - x,
      drag.defaultPositions[facility * 2 + 1] - y
    );
  let facility = 0;
  for (let candidate = 1; candidate < count; candidate++) {
    if (distanceTo(candidate) < distanceTo(facility)) facility = candidate;
  }
  // Keep the previous facility unless another is clearly nearer, so the choice does not flicker.
  if (
    drag.active >= 0 &&
    drag.active < count &&
    distanceTo(facility) > distanceTo(drag.active) * 0.85
  ) {
    facility = drag.active;
  }
  const node = drag.nodeIndex.nearest(x, y);
  if (facility === drag.active && node === drag.node) return {facility, node, changed: false};
  scene.facilityNodes.set(drag.defaultNodes);
  scene.facilityPositions.set(drag.defaultPositions);
  scene.facilityNodes[facility] = node;
  scene.facilityPositions[facility * 2] = scene.network.nodes[node * 2];
  scene.facilityPositions[facility * 2 + 1] = scene.network.nodes[node * 2 + 1];
  drag.active = facility;
  drag.node = node;
  return {facility, node, changed: true};
}

/** Row i holds i: the categorical id of facility i, shared by the facility and demand colors. */
export function createFacilityIds(count: number): Uint32Array {
  return Uint32Array.from({length: count}, (_, index) => index);
}

/**
 * Facility markers: an ink halo under a dot in the facility's own categorical color (`ids`), or in
 * solid ink-white when no ids are given.
 */
export function getFacilityMarkers(
  positions: RunBuffer,
  theme: DemoTheme,
  options: {ids?: RunBuffer; count?: number; radius?: number} = {}
): DotsRunLayer[] {
  const radius = options.radius ?? 9;
  return [
    {
      layer: 'dots',
      positions,
      count: options.count,
      radius: radius + 3,
      color: [...theme.ink, 1]
    },
    options.ids
      ? {layer: 'dots', positions, values: options.ids, count: options.count, radius}
      : {layer: 'dots', positions, count: options.count, radius, color: [1, 1, 1, 1]}
  ];
}

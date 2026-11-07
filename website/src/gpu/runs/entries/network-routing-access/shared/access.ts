// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Helpers shared by the facility and accessibility runs: a seeded service scene (facilities and
// demand locations snapped to intersections), and the real network operations that feed them, a
// facility-by-node cost matrix and its decay weights.

import {GPUNetworkCostMatrix, GPUNetworkCostWeights} from '@luma-contributors/analysis/gpu-network';
import type {RunBuffer, RunBuilder} from '../../../define-run';
import type {NetworkInput, World} from '../../../inputs';
import {createRoads, gatherPositions, pickNodes} from './roads';
import {uploadNetworkTopology} from '../../../inputs';

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

/** Facilities and demand locations on a street grid. */
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

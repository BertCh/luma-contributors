// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Layer} from '@deck.gl/core';
import type {Buffer} from '@luma.gl/core';
import {
  GPUAlternativeRoutes,
  GPUNetworkCostTable,
  GPUNetworkPointRouting,
  GPUNetworkTimeRouting,
  prepareGPUNetworkRestrictionStates
} from '@luma-contributors/analysis/gpu-network';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {createGraphBufferImporter} from '../../engine/graph-buffers';
import {SpatialAnalysisPointLayer, SpatialAnalysisSegmentLayer} from '../../engine/layers';
import {addKernelPass} from '../../engine/mode-kernels';
import {SpatialAnalysisResources} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import type {LoadedDataset} from '../../data/catalog';
import type {SceneContext, SceneInstance} from '../scene';

export type ChicagoRouteWorkbenchOptions = {
  routePair: string;
  originFraction: number;
  destinationFraction: number;
  departureMinute: number;
  timeModel: 'affine-duration' | 'piecewise-arrival';
  affineSlope: number;
  firstSampleMinute: '0' | '15' | '60';
  showAlternatives: boolean;
  showCostTable: boolean;
  costLimitMinutes: number;
  turnRestriction: 'none' | 'hypothetical-ban';
};

type LocalWalkGraph = {
  origin: readonly [number, number];
  nodes: Float32Array;
  sources: Uint32Array;
  targets: Uint32Array;
  lengths: Float32Array;
  travelMinutes: Float32Array;
  offsets: Uint32Array;
  edgeVertices: Float32Array;
  segmentEdges: Uint32Array;
  turns: Uint32Array;
  pairs: Record<string, readonly [number, number]>;
  excludedDirectedEdges: number;
};

const CHICAGO_ROUTE_WORKBENCH_NODE_LIMIT = 64;
const CHICAGO_ROUTE_WORKBENCH_EDGE_LIMIT = 128;
const CHICAGO_ROUTE_WORKBENCH_ROUTE_COUNT = 3;
const HYPOTHETICAL_FORBIDDEN_TURN = [66, 38] as const;
function buildChicagoRouteWorkbenchGraph(dataset: LoadedDataset): LocalWalkGraph {
  const allSources = dataset.column<Uint32Array>('edgeSource');
  const allTargets = dataset.column<Uint32Array>('edgeTarget');
  const allLengths = dataset.column<Float32Array>('edgeLength');
  const allTravelSeconds = dataset.column<Float32Array>('edgeTravelTime');
  const allOffsets = dataset.column<Uint32Array>('nodeEdgeOffsets');
  const pathOffsets = dataset.column<Uint32Array>('edgePathOffsets');
  const pathVertices = dataset.column<Float32Array>('edgeVertices');
  const origin = dataset.defaultOrigin;
  const projectedNodes = dataset.projectColumn('nodes', origin);
  const centerLng = -87.6298;
  const centerLat = 41.8827;
  const projection = dataset.getProjection(origin);
  const [centerX, centerY] = projection.project(centerLng, centerLat);
  let startNode = 0;
  let nearestDistance = Infinity;
  for (let node = 0; node < projectedNodes.length / 2; node++) {
    const distance =
      (projectedNodes[node * 2] - centerX) ** 2 + (projectedNodes[node * 2 + 1] - centerY) ** 2;
    if (distance < nearestDistance) {
      nearestDistance = distance;
      startNode = node;
    }
  }

  // Deterministic breadth-first growth retains actual directed links and their parallel identities.
  const selectedNodes: number[] = [startNode];
  const selected = new Set(selectedNodes);
  const queue = [startNode];
  let selectedEdgeCount = 0;
  let queueIndex = 0;
  while (queueIndex < queue.length && selectedNodes.length < CHICAGO_ROUTE_WORKBENCH_NODE_LIMIT) {
    const source = queue[queueIndex++];
    for (let edge = allOffsets[source]; edge < allOffsets[source + 1]; edge++) {
      const target = allTargets[edge];
      if (selected.has(target)) continue;
      let additionalEdges = 0;
      for (let candidate = allOffsets[target]; candidate < allOffsets[target + 1]; candidate++) {
        if (selected.has(allTargets[candidate]) || allTargets[candidate] === source)
          additionalEdges++;
      }
      for (let candidate = allOffsets[source]; candidate < allOffsets[source + 1]; candidate++) {
        if (allTargets[candidate] === target) additionalEdges++;
      }
      if (selectedEdgeCount + additionalEdges > CHICAGO_ROUTE_WORKBENCH_EDGE_LIMIT) continue;
      selectedEdgeCount += additionalEdges;
      selected.add(target);
      selectedNodes.push(target);
      queue.push(target);
    }
  }
  const countInducedEdges = () => {
    let count = 0;
    for (const source of selectedNodes) {
      for (let edge = allOffsets[source]; edge < allOffsets[source + 1]; edge++) {
        if (selected.has(allTargets[edge])) count++;
      }
    }
    return count;
  };
  while (countInducedEdges() > CHICAGO_ROUTE_WORKBENCH_EDGE_LIMIT && selectedNodes.length > 1) {
    selected.delete(selectedNodes.pop()!);
  }
  const localEdges: number[] = [];
  const nodeMap = new Map(selectedNodes.map((node, index) => [node, index]));
  for (const source of selectedNodes) {
    for (let edge = allOffsets[source]; edge < allOffsets[source + 1]; edge++) {
      if (nodeMap.has(allTargets[edge])) localEdges.push(edge);
    }
  }
  const localSources = Uint32Array.from(localEdges, edge => nodeMap.get(allSources[edge])!);
  const localTargets = Uint32Array.from(localEdges, edge => nodeMap.get(allTargets[edge])!);
  const localLengths = Float32Array.from(localEdges, edge => allLengths[edge]);
  const travelMinutes = Float32Array.from(localEdges, edge => allTravelSeconds[edge] / 60);
  const nodes = new Float32Array(selectedNodes.length * 2);
  selectedNodes.forEach((node, index) =>
    nodes.set(projectedNodes.subarray(node * 2, node * 2 + 2), index * 2)
  );
  const offsets = new Uint32Array(selectedNodes.length + 1);
  for (const source of localSources) offsets[source + 1]++;
  for (let node = 0; node < selectedNodes.length; node++) offsets[node + 1] += offsets[node];

  const localEdgeMap = new Map(localEdges.map((edge, index) => [edge, index]));
  const turns: number[] = [];
  for (let incoming = 0; incoming < localEdges.length; incoming++) {
    const globalIncoming = localEdges[incoming];
    const node = allTargets[globalIncoming];
    for (
      let globalOutgoing = allOffsets[node];
      globalOutgoing < allOffsets[node + 1];
      globalOutgoing++
    ) {
      const outgoing = localEdgeMap.get(globalOutgoing);
      if (outgoing !== undefined) turns.push(incoming, outgoing);
    }
  }
  const edgeVertices: number[] = [];
  const segmentEdges: number[] = [];
  for (let edge = 0; edge < localEdges.length; edge++) {
    const globalEdge = localEdges[edge];
    for (let vertex = pathOffsets[globalEdge]; vertex < pathOffsets[globalEdge + 1] - 1; vertex++) {
      const first = projection.project(pathVertices[vertex * 2], pathVertices[vertex * 2 + 1]);
      const second = projection.project(
        pathVertices[(vertex + 1) * 2],
        pathVertices[(vertex + 1) * 2 + 1]
      );
      edgeVertices.push(first[0], first[1], second[0], second[1]);
      segmentEdges.push(edge);
    }
  }

  const nearestEdge = (positionX: number, positionY: number): number => {
    const point = [positionX, positionY];
    let best = 0;
    let distance = Infinity;
    for (let edge = 0; edge < localEdges.length; edge++) {
      for (let segment = 0; segment < segmentEdges.length; segment++) {
        if (segmentEdges[segment] !== edge) continue;
        const offset = segment * 4;
        const deltaX = edgeVertices[offset + 2] - edgeVertices[offset];
        const deltaY = edgeVertices[offset + 3] - edgeVertices[offset + 1];
        const fraction = Math.max(
          0,
          Math.min(
            1,
            ((point[0] - edgeVertices[offset]) * deltaX +
              (point[1] - edgeVertices[offset + 1]) * deltaY) /
              (deltaX * deltaX + deltaY * deltaY)
          )
        );
        const current = Math.hypot(
          point[0] - edgeVertices[offset] - fraction * deltaX,
          point[1] - edgeVertices[offset + 1] - fraction * deltaY
        );
        if (current < distance) {
          best = edge;
          distance = current;
        }
      }
    }
    return best;
  };
  const east = Math.max(...Array.from(nodes).filter((_, index) => index % 2 === 0));
  const west = Math.min(...Array.from(nodes).filter((_, index) => index % 2 === 0));
  const north = Math.max(...Array.from(nodes).filter((_, index) => index % 2 === 1));
  const south = Math.min(...Array.from(nodes).filter((_, index) => index % 2 === 1));
  const eastEdge = nearestEdge(east, (north + south) / 2);
  const westEdge = nearestEdge(west, (north + south) / 2);
  const northEdge = nearestEdge((east + west) / 2, north);
  const pairs = {
    'east-north': [eastEdge, northEdge] as const,
    'east-west': [eastEdge, westEdge] as const,
    'north-west': [northEdge, westEdge] as const
  };
  return {
    origin,
    nodes,
    sources: localSources,
    targets: localTargets,
    lengths: localLengths,
    travelMinutes,
    offsets,
    edgeVertices: Float32Array.from(edgeVertices),
    segmentEdges: Uint32Array.from(segmentEdges),
    turns: Uint32Array.from(turns),
    pairs,
    excludedDirectedEdges: allSources.length - localEdges.length
  };
}

function formatMinutes(value: number): string {
  return Number.isFinite(value) ? `${value.toFixed(1)} min` : 'unavailable';
}

export async function createChicagoRouteWorkbench(
  ctx: SceneContext<ChicagoRouteWorkbenchOptions>
): Promise<SceneInstance<ChicagoRouteWorkbenchOptions>> {
  const device = ctx.device;
  const network = buildChicagoRouteWorkbenchGraph(ctx.datasets.get('chicago-walk'));
  const nodeCount = network.nodes.length / 2;
  const edgeCount = network.sources.length;
  const segmentCount = network.segmentEdges.length;
  const restrictionStateGraph = prepareGPUNetworkRestrictionStates({
    nodeCount,
    edges: Array.from(
      {length: edgeCount},
      (_, edge) => [network.sources[edge], network.targets[edge]] as const
    ),
    restrictions: [HYPOTHETICAL_FORBIDDEN_TURN]
  });
  const {stateCount: restrictionStateCount, transitionCount: restrictionTransitionCount} =
    restrictionStateGraph;
  const resources = new SpatialAnalysisResources(device, 'chicago-route-workbench');
  const edgeSegmentsBuffer = resources.createBuffer(
    'chicago-route-workbench-edge-segments',
    network.edgeVertices
  );
  const segmentEdgesBuffer = resources.createBuffer(
    'chicago-route-workbench-segment-edges',
    network.segmentEdges
  );
  const originMarkerBuffer = resources.createBuffer('chicago-route-workbench-origin-marker', 8);
  const destinationMarkerBuffer = resources.createBuffer(
    'chicago-route-workbench-destination-marker',
    8
  );
  const edgeTargetBuffer = resources.createBuffer(
    'chicago-route-workbench-edge-target',
    network.targets
  );
  const baseTravelBuffer = resources.createBuffer(
    'chicago-route-workbench-base-travel-minutes',
    network.travelMinutes
  );
  const pointRoutingCostsBuffer = resources.createBuffer(
    'chicago-route-workbench-point-costs',
    network.travelMinutes
  );
  const offsetsBuffer = resources.createBuffer('chicago-route-workbench-offsets', network.offsets);
  const turnsBuffer = resources.createBuffer('chicago-route-workbench-turns', network.turns);
  const pointQueriesBuffer = resources.createBuffer(
    'chicago-route-workbench-point-query',
    new Uint32Array(2)
  );
  const pointFractionsBuffer = resources.createBuffer(
    'chicago-route-workbench-point-fractions',
    new Float32Array(2)
  );
  const pointCostBuffer = resources.createBuffer('chicago-route-workbench-point-cost', 4);
  const pointStatusBuffer = resources.createBuffer('chicago-route-workbench-point-status', 4);
  const pointPredecessorsBuffer = resources.createBuffer(
    'chicago-route-workbench-point-predecessors',
    edgeCount * 4
  );
  const timeStateVerticesBuffer = resources.createBuffer(
    'chicago-route-workbench-time-vertices',
    Uint32Array.from({length: nodeCount}, (_, index) => index)
  );
  const transitions = new Uint32Array(edgeCount * 2);
  transitions.set(network.sources);
  transitions.set(network.targets, edgeCount);
  // Interleave source,target pairs for the packed uint32x2 format.
  for (let edge = 0; edge < edgeCount; edge++) {
    transitions[edge * 2] = network.sources[edge];
    transitions[edge * 2 + 1] = network.targets[edge];
  }
  const transitionsBuffer = resources.createBuffer(
    'chicago-route-workbench-time-transitions',
    transitions
  );
  const affineCosts = new Float32Array(edgeCount * 2);
  const piecewiseCosts = new Float32Array(edgeCount * 6);
  for (let edge = 0; edge < edgeCount; edge++) {
    const duration = network.travelMinutes[edge];
    affineCosts[edge * 2] = duration;
    piecewiseCosts.set(
      [0, duration * 1.12, 360, 360 + duration * 1.04, 720, 720 + duration * 1.18],
      edge * 6
    );
  }
  const affineCostsBuffer = resources.createBuffer(
    'chicago-route-workbench-affine-costs',
    affineCosts
  );
  const piecewiseCostsBuffer = resources.createBuffer(
    'chicago-route-workbench-piecewise-costs',
    piecewiseCosts
  );
  const timeEndpointsBuffer = resources.createBuffer(
    'chicago-route-workbench-time-endpoints',
    new Uint32Array(2)
  );
  const departureBuffer = resources.createBuffer('chicago-route-workbench-departure', 4);
  const timeOutput = (prefix: string, stateCount = nodeCount) => ({
    arrivals: resources.createBuffer(`${prefix}-arrivals`, stateCount * 4),
    predecessors: resources.createBuffer(`${prefix}-predecessors`, stateCount * 4),
    summary: resources.createBuffer(`${prefix}-summary`, 16)
  });
  const affineOutput = timeOutput('chicago-route-workbench-affine');
  const piecewiseOutput = timeOutput('chicago-route-workbench-piecewise');
  const restrictionStateVerticesBuffer = resources.createBuffer(
    'chicago-route-workbench-restriction-state-vertices',
    restrictionStateGraph.stateVertices
  );
  const restrictionTransitionsBuffer = resources.createBuffer(
    'chicago-route-workbench-restriction-transitions',
    restrictionStateGraph.transitions
  );
  const restrictionOriginalEdgesBuffer = resources.createBuffer(
    'chicago-route-workbench-restriction-original-edges',
    restrictionStateGraph.originalEdges
  );
  const restrictionCosts = new Float32Array(restrictionTransitionCount * 2);
  for (let transition = 0; transition < restrictionTransitionCount; transition++) {
    restrictionCosts[transition * 2] =
      network.travelMinutes[restrictionStateGraph.originalEdges[transition]];
  }
  const restrictionCostsBuffer = resources.createBuffer(
    'chicago-route-workbench-restriction-affine-costs',
    restrictionCosts
  );
  const restrictionEndpointsBuffer = resources.createBuffer(
    'chicago-route-workbench-restriction-endpoints',
    new Uint32Array(2)
  );
  const restrictionOutput = timeOutput(
    'chicago-route-workbench-restriction',
    restrictionStateCount
  );
  const restrictionRouteFlagsBuffer = resources.createBuffer(
    'chicago-route-workbench-restriction-route-flags',
    edgeCount * 4
  );
  const prohibitedTurnFlags = new Uint32Array(edgeCount);
  for (const edge of HYPOTHETICAL_FORBIDDEN_TURN) prohibitedTurnFlags[edge] = 1;
  const prohibitedTurnFlagsBuffer = resources.createBuffer(
    'chicago-route-workbench-hypothetical-prohibited-turn-flags',
    prohibitedTurnFlags
  );

  const routeOutputCount = CHICAGO_ROUTE_WORKBENCH_ROUTE_COUNT;
  const alternativeSourcesBuffer = resources.createBuffer('chicago-route-workbench-alt-sources', 4);
  const alternativeTargetsBuffer = resources.createBuffer('chicago-route-workbench-alt-targets', 4);
  const alternativeOutput = {
    vertices: resources.createBuffer(
      'chicago-route-workbench-alt-vertices',
      routeOutputCount * nodeCount * 4
    ),
    edges: resources.createBuffer(
      'chicago-route-workbench-alt-edges',
      routeOutputCount * nodeCount * 4
    ),
    lengths: resources.createBuffer('chicago-route-workbench-alt-lengths', routeOutputCount * 4),
    costs: resources.createBuffer('chicago-route-workbench-alt-costs', routeOutputCount * 4),
    counts: resources.createBuffer('chicago-route-workbench-alt-counts', 4),
    complete: resources.createBuffer('chicago-route-workbench-alt-complete', 4)
  };
  const alternativeRouteFlags = Array.from({length: routeOutputCount}, (_, index) =>
    resources.createBuffer(`chicago-route-workbench-alt-flags-${index}`, edgeCount * 4)
  );
  const sharedRouteFlags = resources.createBuffer(
    'chicago-route-workbench-shared-flags',
    edgeCount * 4
  );

  const costLimitBuffer = resources.createBuffer('chicago-route-workbench-cost-limit', 4);
  const costTableEntryCapacity = nodeCount * nodeCount;
  const costTableOutput = {
    offsets: resources.createBuffer('chicago-route-workbench-cost-offsets', (nodeCount + 1) * 4),
    targets: resources.createBuffer(
      'chicago-route-workbench-cost-targets',
      costTableEntryCapacity * 4
    ),
    costs: resources.createBuffer('chicago-route-workbench-cost-costs', costTableEntryCapacity * 4),
    overflow: resources.createBuffer('chicago-route-workbench-cost-overflow', 4),
    required: resources.createBuffer('chicago-route-workbench-cost-required', 4),
    converged: resources.createBuffer('chicago-route-workbench-cost-converged', 4)
  };
  const reusedPairCostBuffer = resources.createBuffer(
    'chicago-route-workbench-reused-pair-cost',
    4
  );
  const reusedPairStatusBuffer = resources.createBuffer(
    'chicago-route-workbench-reused-pair-status',
    4
  );

  let queryGeneration = 0;
  let submittedGeneration = -1;
  let graphDirty = true;
  let lastComputeReaderGeneration = -1;
  let routeResultGeneration = -1;
  let computeResultGeneration = -1;
  let destroyed = false;
  let routesValid = false;
  let routeCosts = new Float32Array(routeOutputCount);
  let routeLengths = new Uint32Array(routeOutputCount);
  let routeEdges = new Uint32Array(routeOutputCount * nodeCount);
  let routeCount = 0;
  let routeComplete = false;
  let pointRouteValue = Infinity;
  let pointRouteStatus = 1;
  let selectedArrivalValue = Infinity;
  let selectedArrivalValid = false;
  let selectedArrivalConverged = false;
  let restrictionSummary = new Uint32Array(4);
  let restrictionArrival = Infinity;
  let restrictionValid = false;
  let restrictionConverged = false;
  let costReuseDescription = 'Cost-table result pending';
  let affineSummary = new Uint32Array(4);
  let piecewiseSummary = new Uint32Array(4);
  let tableSummary = new Uint32Array(3);
  const graph = new GPUCommandGraph<void>(device, {id: 'chicago-route-workbench'});
  const view = createGraphBufferImporter(graph);
  const topology = {
    offsets: view(offsetsBuffer, 'uint32', nodeCount + 1),
    neighbors: view(edgeTargetBuffer, 'uint32', edgeCount),
    weights: view(baseTravelBuffer, 'float32', edgeCount)
  };
  graph.add(
    new GPUNetworkPointRouting({
      id: 'chicago-route-workbench-point-routing',
      turns: view(turnsBuffer, 'uint32x2', network.turns.length / 2),
      costs: view(pointRoutingCostsBuffer, 'float32', edgeCount),
      queries: view(pointQueriesBuffer, 'uint32x2', 1),
      fractions: view(pointFractionsBuffer, 'float32x2', 1),
      output: {
        costs: view(pointCostBuffer, 'float32', 1),
        status: view(pointStatusBuffer, 'uint32', 1),
        predecessorEdges: view(pointPredecessorsBuffer, 'uint32', edgeCount)
      }
    })
  );
  const addTimeRouting = (
    id: string,
    costModel: 'affine-duration' | 'piecewise-arrival',
    costsBuffer: Buffer,
    output: ReturnType<typeof timeOutput>
  ) => {
    graph.add(
      new GPUNetworkTimeRouting({
        id,
        stateVertices: view(timeStateVerticesBuffer, 'uint32', nodeCount),
        transitions: view(transitionsBuffer, 'uint32x2', edgeCount),
        costs: view(
          costsBuffer,
          'float32x2',
          edgeCount * (costModel === 'piecewise-arrival' ? 3 : 1)
        ),
        endpoints: view(timeEndpointsBuffer, 'uint32', 2),
        departure: view(departureBuffer, 'float32', 1),
        costModel,
        ...(costModel === 'piecewise-arrival'
          ? {samplesPerTransition: 3, waitingPolicy: 'wait-until-first-sample' as const}
          : {}),
        output: {
          arrivals: view(output.arrivals, 'float32', nodeCount),
          predecessorTransitions: view(output.predecessors, 'uint32', nodeCount),
          summary: view(output.summary, 'uint32', 4)
        }
      })
    );
  };
  addTimeRouting(
    'chicago-route-workbench-affine-time',
    'affine-duration',
    affineCostsBuffer,
    affineOutput
  );
  addTimeRouting(
    'chicago-route-workbench-piecewise-time',
    'piecewise-arrival',
    piecewiseCostsBuffer,
    piecewiseOutput
  );
  const restrictionStateView = view(
    restrictionStateVerticesBuffer,
    restrictionStateGraph.formats.stateVertices,
    restrictionStateCount
  );
  const restrictionTransitionView = view(
    restrictionTransitionsBuffer,
    restrictionStateGraph.formats.transitions,
    restrictionTransitionCount
  );
  const restrictionTransitionWordsView = view(
    restrictionTransitionsBuffer,
    'uint32',
    restrictionTransitionCount * 2
  );
  const restrictionOriginalEdgeView = view(
    restrictionOriginalEdgesBuffer,
    restrictionStateGraph.formats.originalEdges,
    restrictionTransitionCount
  );
  const restrictionCostView = view(restrictionCostsBuffer, 'float32x2', restrictionTransitionCount);
  const restrictionEndpointView = view(restrictionEndpointsBuffer, 'uint32', 2);
  const restrictionDepartureView = view(departureBuffer, 'float32', 1);
  const restrictionPredecessorView = view(
    restrictionOutput.predecessors,
    'uint32',
    restrictionStateCount
  );
  const restrictionSummaryView = view(restrictionOutput.summary, 'uint32', 4);
  const restrictionRouteFlagsView = view(restrictionRouteFlagsBuffer, 'uint32', edgeCount);
  graph.add(
    new GPUNetworkTimeRouting({
      id: 'chicago-route-workbench-hypothetical-turn-restriction',
      stateVertices: restrictionStateView,
      transitions: restrictionTransitionView,
      costs: restrictionCostView,
      endpoints: restrictionEndpointView,
      departure: restrictionDepartureView,
      costModel: 'affine-duration',
      output: {
        arrivals: view(restrictionOutput.arrivals, 'float32', restrictionStateCount),
        predecessorTransitions: restrictionPredecessorView,
        summary: restrictionSummaryView
      }
    })
  );
  addKernelPass(graph, {
    id: 'chicago-route-workbench-publish-restriction-path',
    invocationCount: edgeCount,
    bindings: [
      {name: 'predecessors', view: restrictionPredecessorView, type: 'u32', access: 'read'},
      {name: 'transitions', view: restrictionTransitionWordsView, type: 'u32', access: 'read'},
      {name: 'originalEdges', view: restrictionOriginalEdgeView, type: 'u32', access: 'read'},
      {name: 'endpoints', view: restrictionEndpointView, type: 'u32', access: 'read'},
      {name: 'summary', view: restrictionSummaryView, type: 'u32', access: 'read'},
      {name: 'flags', view: restrictionRouteFlagsView, type: 'u32', access: 'read_write'}
    ],
    declarations: `const STATE_COUNT: u32 = ${restrictionStateCount}u; const TRANSITION_COUNT: u32 = ${restrictionTransitionCount}u;`,
    body: `flags[flagsOffset + index] = 0u;
if (summary[summaryOffset] == 0u || summary[summaryOffset + 1u] == 0u) { return; }
let startState = endpoints[endpointsOffset]; var currentState = summary[summaryOffset + 2u]; var found = false;
for (var step = 0u; step < STATE_COUNT; step++) {
  if (currentState == startState || currentState >= STATE_COUNT) { break; }
  let transition = predecessors[predecessorsOffset + currentState];
  if (transition >= TRANSITION_COUNT) { break; }
  if (originalEdges[originalEdgesOffset + transition] == index) { found = true; }
  currentState = transitions[transitionsOffset + 2u * transition];
}
flags[flagsOffset + index] = select(0u, 1u, found);`
  });
  graph.add(
    new GPUAlternativeRoutes({
      id: 'chicago-route-workbench-alternatives',
      topology,
      sources: view(alternativeSourcesBuffer, 'uint32', 1),
      targets: view(alternativeTargetsBuffer, 'uint32', 1),
      routeCount: routeOutputCount,
      maximumExpansions: 100000,
      maximumFrontierSize: 1024,
      output: {
        vertices: view(alternativeOutput.vertices, 'uint32', routeOutputCount * nodeCount),
        edges: view(alternativeOutput.edges, 'uint32', routeOutputCount * nodeCount),
        lengths: view(alternativeOutput.lengths, 'uint32', routeOutputCount),
        costs: view(alternativeOutput.costs, 'float32', routeOutputCount),
        counts: view(alternativeOutput.counts, 'uint32', 1),
        complete: view(alternativeOutput.complete, 'uint32', 1)
      }
    })
  );
  graph.add(
    new GPUNetworkCostTable({
      id: 'chicago-route-workbench-cost-table',
      topology,
      costLimit: view(costLimitBuffer, 'float32', 1),
      laneCount: Math.min(16, nodeCount),
      maximumIterations: nodeCount,
      localIterations: 8,
      output: {
        offsets: view(costTableOutput.offsets, 'uint32', nodeCount + 1),
        targets: view(costTableOutput.targets, 'uint32', costTableEntryCapacity),
        costs: view(costTableOutput.costs, 'float32', costTableEntryCapacity),
        status: {
          overflow: view(costTableOutput.overflow, 'uint32', 1),
          requiredCount: view(costTableOutput.required, 'uint32', 1),
          converged: view(costTableOutput.converged, 'uint32', 1)
        }
      }
    })
  );
  addKernelPass(graph, {
    id: 'chicago-route-workbench-read-reused-pair',
    bindings: [
      {
        name: 'offsets',
        view: view(costTableOutput.offsets, 'uint32', nodeCount + 1),
        type: 'u32',
        access: 'read'
      },
      {
        name: 'targets',
        view: view(costTableOutput.targets, 'uint32', costTableEntryCapacity),
        type: 'u32',
        access: 'read'
      },
      {
        name: 'costs',
        view: view(costTableOutput.costs, 'float32', costTableEntryCapacity),
        type: 'f32',
        access: 'read'
      },
      {
        name: 'endpoints',
        view: view(timeEndpointsBuffer, 'uint32', 2),
        type: 'u32',
        access: 'read'
      },
      {
        name: 'overflow',
        view: view(costTableOutput.overflow, 'uint32', 1),
        type: 'u32',
        access: 'read'
      },
      {
        name: 'converged',
        view: view(costTableOutput.converged, 'uint32', 1),
        type: 'u32',
        access: 'read'
      },
      {
        name: 'pairCost',
        view: view(reusedPairCostBuffer, 'float32', 1),
        type: 'f32',
        access: 'read_write'
      },
      {
        name: 'pairStatus',
        view: view(reusedPairStatusBuffer, 'uint32', 1),
        type: 'u32',
        access: 'read_write'
      }
    ],
    invocationCount: 1,
    declarations: `const NODE_COUNT: u32 = ${nodeCount}u; const ENTRY_CAPACITY: u32 = ${costTableEntryCapacity}u;`,
    body: `let source = endpoints[endpointsOffset]; let destination = endpoints[endpointsOffset + 1u];
  pairCost[pairCostOffset] = 0.0;
  pairStatus[pairStatusOffset] = 1u;
  if (overflow[overflowOffset] != 0u) { pairStatus[pairStatusOffset] = 3u; return; }
  if (converged[convergedOffset] == 0u) { pairStatus[pairStatusOffset] = 4u; return; }
  if (source >= NODE_COUNT || destination >= NODE_COUNT) { pairStatus[pairStatusOffset] = 2u; return; }
  for (var slot = offsets[offsetsOffset + source]; slot < min(offsets[offsetsOffset + source + 1u], ENTRY_CAPACITY); slot++) {
    if (targets[targetsOffset + slot] == destination) { pairCost[pairCostOffset] = costs[costsOffset + slot]; pairStatus[pairStatusOffset] = 0u; return; }
  }`
  });
  // Publish route masks only after every producer has written its outputs.
  for (let route = 0; route < routeOutputCount; route++) {
    addKernelPass(graph, {
      id: `chicago-route-workbench-clear-route-${route}`,
      bindings: [
        {
          name: 'flags',
          view: view(alternativeRouteFlags[route], 'uint32', edgeCount),
          type: 'u32',
          access: 'read_write'
        }
      ],
      invocationCount: edgeCount,
      body: 'flags[flagsOffset + index] = 0u;'
    });
    addKernelPass(graph, {
      id: `chicago-route-workbench-publish-route-${route}`,
      bindings: [
        {
          name: 'edges',
          view: view(alternativeOutput.edges, 'uint32', routeOutputCount * nodeCount),
          type: 'u32',
          access: 'read'
        },
        {
          name: 'lengths',
          view: view(alternativeOutput.lengths, 'uint32', routeOutputCount),
          type: 'u32',
          access: 'read'
        },
        {
          name: 'flags',
          view: view(alternativeRouteFlags[route], 'uint32', edgeCount),
          type: 'u32',
          access: 'read_write'
        }
      ],
      invocationCount: nodeCount,
      declarations: `const EDGE_COUNT: u32 = ${edgeCount}u; const ROUTE_OFFSET: u32 = ${route * nodeCount}u;`,
      body: `if (index + 1u < lengths[lengthsOffset + ${route}u]) { let edge = edges[edgesOffset + ROUTE_OFFSET + index]; if (edge < EDGE_COUNT) { flags[flagsOffset + edge] = 1u; } }`
    });
  }
  addKernelPass(graph, {
    id: 'chicago-route-workbench-publish-shared-route',
    bindings: [
      {
        name: 'first',
        view: view(alternativeRouteFlags[0], 'uint32', edgeCount),
        type: 'u32',
        access: 'read'
      },
      {
        name: 'second',
        view: view(alternativeRouteFlags[1], 'uint32', edgeCount),
        type: 'u32',
        access: 'read'
      },
      {
        name: 'third',
        view: view(alternativeRouteFlags[2], 'uint32', edgeCount),
        type: 'u32',
        access: 'read'
      },
      {
        name: 'sharedMask',
        view: view(sharedRouteFlags, 'uint32', edgeCount),
        type: 'u32',
        access: 'read_write'
      }
    ],
    invocationCount: edgeCount,
    body: 'let overlapsFirst = first[firstOffset + index] != 0u && (second[secondOffset + index] != 0u || third[thirdOffset + index] != 0u); sharedMask[sharedMaskOffset + index] = select(0u, 1u, overlapsFirst);'
  });
  const compiled = resources.track(await graph.compileAsync());

  const routeBytes = [
    routeOutputCount * 4,
    routeOutputCount * 4,
    routeOutputCount * nodeCount * 4,
    4,
    4
  ];
  let lastReaderGeneration = -1;
  const routeReader = new SummaryReader(
    resources,
    'chicago-route-workbench-routes',
    [
      {buffer: alternativeOutput.costs, size: routeBytes[0]},
      {buffer: alternativeOutput.lengths, size: routeBytes[1]},
      {buffer: alternativeOutput.edges, size: routeBytes[2]},
      {buffer: alternativeOutput.counts, size: 4},
      {buffer: alternativeOutput.complete, size: 4}
    ],
    bytes => {
      if (destroyed || lastReaderGeneration !== queryGeneration) return;
      const costsEnd = routeBytes[0];
      const lengthsEnd = costsEnd + routeBytes[1];
      const edgesEnd = lengthsEnd + routeBytes[2];
      routeCosts = new Float32Array(bytes.slice(0, costsEnd));
      routeLengths = new Uint32Array(bytes.slice(costsEnd, lengthsEnd));
      routeEdges = new Uint32Array(bytes.slice(lengthsEnd, edgesEnd));
      routeCount = new Uint32Array(bytes.slice(edgesEnd, edgesEnd + 4))[0];
      routeComplete = new Uint32Array(bytes.slice(edgesEnd + 4))[0] === 1;
      routeResultGeneration = queryGeneration;
      routesValid =
        routeResultGeneration === queryGeneration && computeResultGeneration === queryGeneration;
      const validRoutes = Math.min(routeCount, CHICAGO_ROUTE_WORKBENCH_ROUTE_COUNT);
      let sharedEdges = 0;
      if (validRoutes > 1) {
        const first = new Set(Array.from(routeEdges.subarray(0, Math.max(0, routeLengths[0] - 1))));
        const other = new Set<number>();
        for (let route = 1; route < validRoutes; route++) {
          const offset = route * nodeCount;
          for (const edge of routeEdges.subarray(
            offset,
            offset + Math.max(0, routeLengths[route] - 1)
          ))
            if (first.has(edge)) other.add(edge);
        }
        sharedEdges = other.size;
      }
      ctx.setReadout(
        'alternativeCosts',
        validRoutes
          ? Array.from(routeCosts.subarray(0, validRoutes), formatMinutes).join(' · ')
          : 'No route found'
      );
      ctx.setReadout('routeOverlap', `${sharedEdges} shared directed links with route 1`);
      ctx.setReadout(
        'alternativeStatus',
        routeComplete
          ? `${validRoutes} routes certified (top-k or all routes exhausted)`
          : 'Search bounded: shown routes are incumbents; more may exist'
      );
      ctx.requestLayers();
    }
  );
  const computeReader = new SummaryReader(
    resources,
    'chicago-route-workbench-compute',
    [
      {buffer: pointCostBuffer, size: 4},
      {buffer: pointStatusBuffer, size: 4},
      {buffer: affineOutput.summary, size: 16},
      {buffer: piecewiseOutput.summary, size: 16},
      {buffer: costTableOutput.overflow, size: 4},
      {buffer: costTableOutput.required, size: 4},
      {buffer: costTableOutput.converged, size: 4},
      {buffer: reusedPairCostBuffer, size: 4},
      {buffer: reusedPairStatusBuffer, size: 4},
      {buffer: restrictionOutput.summary, size: 16}
    ],
    bytes => {
      if (destroyed || lastComputeReaderGeneration !== queryGeneration) return;
      const pointCost = new Float32Array(bytes.slice(0, 4))[0];
      const pointStatus = new Uint32Array(bytes.slice(4, 8))[0];
      pointRouteValue = pointCost;
      pointRouteStatus = pointStatus;
      affineSummary = new Uint32Array(bytes.slice(8, 24));
      piecewiseSummary = new Uint32Array(bytes.slice(24, 40));
      tableSummary = new Uint32Array([
        new Uint32Array(bytes.slice(40, 44))[0],
        new Uint32Array(bytes.slice(44, 48))[0],
        new Uint32Array(bytes.slice(48, 52))[0]
      ]);
      const reusedPairCost = new Float32Array(bytes.slice(52, 56))[0];
      const reusedPairStatus = new Uint32Array(bytes.slice(56, 60))[0];
      restrictionSummary = new Uint32Array(bytes.slice(60, 76));
      restrictionArrival = new Float32Array(Uint32Array.of(restrictionSummary[3]).buffer)[0];
      restrictionValid = restrictionSummary[0] === 1;
      restrictionConverged = restrictionSummary[1] === 1;
      ctx.setReadout(
        'pointRoute',
        pointStatus === 0
          ? Number.isFinite(pointCost)
            ? formatMinutes(pointCost)
            : 'No feasible directed route (search complete)'
          : pointStatus === 2
            ? Number.isFinite(pointCost)
              ? `${formatMinutes(pointCost)} · incomplete incumbent`
              : 'Search incomplete; no route found yet'
            : 'Invalid endpoint or cost data'
      );
      const affineArrival = Number.isFinite(
        new Float32Array(Uint32Array.of(affineSummary[3]).buffer)[0]
      )
        ? new Float32Array(Uint32Array.of(affineSummary[3]).buffer)[0]
        : Infinity;
      const piecewiseArrival = Number.isFinite(
        new Float32Array(Uint32Array.of(piecewiseSummary[3]).buffer)[0]
      )
        ? new Float32Array(Uint32Array.of(piecewiseSummary[3]).buffer)[0]
        : Infinity;
      const modelArrival =
        ctx.options.timeModel === 'affine-duration' ? affineArrival : piecewiseArrival;
      const modelConverged =
        ctx.options.timeModel === 'affine-duration'
          ? affineSummary[1] === 1
          : piecewiseSummary[1] === 1;
      const clock = Number.isFinite(modelArrival)
        ? `${String(Math.floor((8 * 60 + modelArrival) / 60) % 24).padStart(2, '0')}:${String(Math.floor((8 * 60 + modelArrival) % 60)).padStart(2, '0')}`
        : 'unavailable';
      const modelValid =
        ctx.options.timeModel === 'affine-duration'
          ? affineSummary[0] === 1
          : piecewiseSummary[0] === 1;
      selectedArrivalValue = modelArrival;
      selectedArrivalValid = modelValid;
      selectedArrivalConverged = modelConverged;
      computeResultGeneration = queryGeneration;
      routesValid =
        routeResultGeneration === queryGeneration && computeResultGeneration === queryGeneration;
      ctx.setReadout(
        'arrivalTime',
        Number.isFinite(modelArrival)
          ? `${clock} local${modelConverged ? ' (modeled)' : ' · search incomplete'}`
          : !modelValid
            ? 'Invalid time-cost input'
            : modelConverged
              ? 'No feasible route in this extract'
              : 'No route found before relaxation limit'
      );
      ctx.requestLayers();
      ctx.setReadout(
        'affineArrival',
        affineSummary[0] && affineSummary[1]
          ? `${(new Float32Array(Uint32Array.of(affineSummary[3]).buffer)[0]).toFixed(1)} min after 08:00`
          : 'unavailable / incomplete'
      );
      ctx.setReadout(
        'piecewiseArrival',
        piecewiseSummary[0] && piecewiseSummary[1]
          ? `${(new Float32Array(Uint32Array.of(piecewiseSummary[3]).buffer)[0]).toFixed(1)} min after 08:00`
          : 'unavailable / incomplete'
      );
      const baselineAffineArrival = new Float32Array(Uint32Array.of(affineSummary[3]).buffer)[0];
      const restrictionDescription = !restrictionValid
        ? 'Invalid restricted-route input'
        : !restrictionConverged
          ? 'Expanded-state search incomplete'
          : !Number.isFinite(restrictionArrival)
            ? 'No route under the hypothetical turn rule'
            : ctx.options.turnRestriction === 'hypothetical-ban'
              ? `Hypothetical rule active · expanded-state route converged`
              : 'Rule off · GPU counterfactual converged';
      const restrictionComparison =
        restrictionValid &&
        restrictionConverged &&
        Number.isFinite(baselineAffineArrival) &&
        Number.isFinite(restrictionArrival)
          ? `Baseline ${formatMinutes(baselineAffineArrival)} · rule ${formatMinutes(restrictionArrival)} · +${formatMinutes(restrictionArrival - baselineAffineArrival)}`
          : 'Comparison unavailable until both routes converge';
      ctx.setReadout('restrictionStatus', restrictionDescription);
      ctx.setReadout('restrictionComparison', restrictionComparison);
      costReuseDescription = !ctx.options.showCostTable
        ? 'Reuse paused at zero-minute bound'
        : reusedPairStatus === 0
          ? `Reused pair lookup: ${formatMinutes(reusedPairCost)} · ${tableSummary[1]} entries cached`
          : reusedPairStatus === 1
            ? 'No pair within the configured cost limit'
            : reusedPairStatus === 3
              ? 'Table capacity exceeded: lookup is inconclusive'
              : reusedPairStatus === 4
                ? 'Table search incomplete: lookup is inconclusive'
                : 'Invalid route endpoints';
      ctx.setReadout('costReuse', costReuseDescription);
    }
  );

  const updateParameters = () => {
    queryGeneration++;
    graphDirty = true;
    routesValid = false;
    const activeRoutePair =
      ctx.options.turnRestriction === 'hypothetical-ban' ? 'east-north' : ctx.options.routePair;
    const pair = network.pairs[activeRoutePair] ?? network.pairs['east-north'];
    const sourceEdge = pair[0] % edgeCount;
    const targetEdge = pair[1] % edgeCount;
    const pointOnEdge = (edge: number, fraction: number): Float32Array => {
      let totalLength = 0;
      for (let segment = 0; segment < segmentCount; segment++) {
        if (network.segmentEdges[segment] !== edge) continue;
        totalLength += Math.hypot(
          network.edgeVertices[segment * 4 + 2] - network.edgeVertices[segment * 4],
          network.edgeVertices[segment * 4 + 3] - network.edgeVertices[segment * 4 + 1]
        );
      }
      let remaining = totalLength * fraction;
      for (let segment = 0; segment < segmentCount; segment++) {
        if (network.segmentEdges[segment] !== edge) continue;
        const offset = segment * 4;
        const startX = network.edgeVertices[offset];
        const startY = network.edgeVertices[offset + 1];
        const deltaX = network.edgeVertices[offset + 2] - startX;
        const deltaY = network.edgeVertices[offset + 3] - startY;
        const length = Math.hypot(deltaX, deltaY);
        if (remaining <= length || segment === segmentCount - 1) {
          const localFraction = length > 0 ? remaining / length : 0;
          return Float32Array.of(startX + deltaX * localFraction, startY + deltaY * localFraction);
        }
        remaining -= length;
      }
      return Float32Array.of(
        network.nodes[network.sources[edge] * 2],
        network.nodes[network.sources[edge] * 2 + 1]
      );
    };
    originMarkerBuffer.write(pointOnEdge(sourceEdge, ctx.options.originFraction));
    destinationMarkerBuffer.write(pointOnEdge(targetEdge, ctx.options.destinationFraction));
    const pointCosts = new Float32Array(edgeCount);
    for (let edge = 0; edge < edgeCount; edge++) {
      const slope = ctx.options.affineSlope;
      affineCosts[edge * 2] = network.travelMinutes[edge];
      affineCosts[edge * 2 + 1] = slope;
      pointCosts[edge] = network.travelMinutes[edge];
    }
    pointRoutingCostsBuffer.write(pointCosts);
    affineCostsBuffer.write(affineCosts);
    for (let transition = 0; transition < restrictionTransitionCount; transition++) {
      restrictionCosts[transition * 2 + 1] = ctx.options.affineSlope;
    }
    restrictionCostsBuffer.write(restrictionCosts);
    pointQueriesBuffer.write(Uint32Array.of(sourceEdge, targetEdge));
    pointFractionsBuffer.write(
      Float32Array.of(ctx.options.originFraction, ctx.options.destinationFraction)
    );
    alternativeSourcesBuffer.write(Uint32Array.of(network.sources[sourceEdge]));
    alternativeTargetsBuffer.write(Uint32Array.of(network.targets[targetEdge]));
    timeEndpointsBuffer.write(
      Uint32Array.of(network.sources[sourceEdge], network.targets[targetEdge])
    );
    restrictionEndpointsBuffer.write(
      Uint32Array.of(
        restrictionStateGraph.initialStates[network.sources[sourceEdge]],
        network.targets[targetEdge]
      )
    );
    const firstSampleMinute = Number(ctx.options.firstSampleMinute);
    for (let edge = 0; edge < edgeCount; edge++) {
      const duration = network.travelMinutes[edge];
      // Shift both sample departure and arrival clocks together, preserving FIFO durations.
      piecewiseCosts.set(
        [
          firstSampleMinute,
          firstSampleMinute + duration * 1.12,
          firstSampleMinute + 360,
          firstSampleMinute + 360 + duration * 1.04,
          firstSampleMinute + 720,
          firstSampleMinute + 720 + duration * 1.18
        ],
        edge * 6
      );
    }
    piecewiseCostsBuffer.write(piecewiseCosts);
    ctx.setReadout(
      'sampleSupport',
      `${firstSampleMinute} min after 08:00 · initial wait ${Math.max(0, firstSampleMinute - ctx.options.departureMinute)} min`
    );
    departureBuffer.write(Float32Array.of(ctx.options.departureMinute));
    costLimitBuffer.write(
      Float32Array.of(ctx.options.showCostTable ? ctx.options.costLimitMinutes : 0)
    );
    ctx.setReadout('pointRoute', 'pending current edge-position query');
    ctx.setReadout('arrivalTime', 'pending current departure');
    ctx.setReadout('alternativeCosts', 'pending current route search');
    ctx.setReadout('routeOverlap', 'pending current alternatives');
    ctx.setReadout('alternativeStatus', 'pending current alternatives');
    ctx.setReadout('costReuse', 'pending current table status');
    ctx.setReadout('affineArrival', 'pending current departure');
    ctx.setReadout('piecewiseArrival', 'pending current departure');
    ctx.setReadout('restrictionStatus', 'pending expanded-state search');
    ctx.setReadout('restrictionComparison', 'pending current route pair');
    ctx.requestLayers();
  };
  updateParameters();
  ctx.setReadout(
    'networkExtent',
    `${nodeCount} vertices · ${edgeCount} directed links · ${network.excludedDirectedEdges} links outside extract`
  );
  ctx.setReadout('costReuse', 'preparing reusable bounded cost table');
  ctx.setReadout('pointRoute', 'pending current edge-position query');

  return {
    getCompiledGraphs: () => [compiled],
    setOption() {
      updateParameters();
    },
    onThemeChange() {
      ctx.requestLayers();
    },
    encode(commandEncoder) {
      if (graphDirty) {
        compiled.encode(commandEncoder, {parameters: undefined});
        submittedGeneration = queryGeneration;
        graphDirty = false;
        routeReader.markStale();
        computeReader.markStale();
      }
      if (!routeReader.isPending) lastReaderGeneration = submittedGeneration;
      if (!computeReader.isPending) lastComputeReaderGeneration = submittedGeneration;
      routeReader.flush(commandEncoder);
      computeReader.flush(commandEncoder);
    },
    getLayers(): Layer[] {
      const coordinateOrigin: [number, number, number] = [network.origin[0], network.origin[1], 0];
      const layers: Layer[] = [
        new SpatialAnalysisSegmentLayer({
          id: 'chicago-route-workbench-streets',
          coordinateOrigin,
          segments: edgeSegmentsBuffer,
          instanceCount: segmentCount,
          widthPixels: 1.15,
          color: [126, 135, 143, 125]
        })
      ];
      if (routesValid && ctx.options.showAlternatives) {
        const routeColors = [
          [220, 91, 54, 235],
          [44, 121, 133, 220],
          [127, 91, 158, 220]
        ] as const;
        for (let route = CHICAGO_ROUTE_WORKBENCH_ROUTE_COUNT - 1; route >= 0; route--) {
          layers.push(
            new SpatialAnalysisSegmentLayer({
              id: `chicago-route-workbench-route-${route + 1}`,
              coordinateOrigin,
              segments: edgeSegmentsBuffer,
              instanceCount: segmentCount,
              widthPixels: route === 0 ? 5 : 3.3,
              values: alternativeRouteFlags[route],
              valueFormat: 'uint32',
              valueIndices: segmentEdgesBuffer,
              colormap: 'mask',
              color: routeColors[route],
              noDataColor: [0, 0, 0, 0]
            })
          );
        }
        layers.push(
          new SpatialAnalysisSegmentLayer({
            id: 'chicago-route-workbench-shared-links',
            coordinateOrigin,
            segments: edgeSegmentsBuffer,
            instanceCount: segmentCount,
            widthPixels: 1.2,
            values: sharedRouteFlags,
            valueFormat: 'uint32',
            valueIndices: segmentEdgesBuffer,
            colormap: 'mask',
            color: [255, 255, 255, 230],
            noDataColor: [0, 0, 0, 0]
          })
        );
      }
      if (
        routesValid &&
        ctx.options.turnRestriction === 'hypothetical-ban' &&
        restrictionValid &&
        restrictionConverged &&
        Number.isFinite(restrictionArrival)
      ) {
        layers.push(
          new SpatialAnalysisSegmentLayer({
            id: 'chicago-route-workbench-restricted-route',
            coordinateOrigin,
            segments: edgeSegmentsBuffer,
            instanceCount: segmentCount,
            widthPixels: 5,
            values: restrictionRouteFlagsBuffer,
            valueFormat: 'uint32',
            valueIndices: segmentEdgesBuffer,
            colormap: 'mask',
            color: [50, 157, 108, 245],
            noDataColor: [0, 0, 0, 0]
          }),
          new SpatialAnalysisSegmentLayer({
            id: 'chicago-route-workbench-prohibited-sequence',
            coordinateOrigin,
            segments: edgeSegmentsBuffer,
            instanceCount: segmentCount,
            widthPixels: 6.2,
            values: prohibitedTurnFlagsBuffer,
            valueFormat: 'uint32',
            valueIndices: segmentEdgesBuffer,
            colormap: 'mask',
            color: [206, 59, 63, 235],
            noDataColor: [0, 0, 0, 0]
          })
        );
      }
      layers.push(
        new SpatialAnalysisPointLayer({
          id: 'chicago-route-workbench-origin-position',
          coordinateOrigin,
          positions: originMarkerBuffer,
          instanceCount: 1,
          radiusPixels: 8,
          color: [255, 255, 255, 255],
          outlineColor: [207, 82, 49, 255],
          outlineWidthPixels: 2,
          shape: 'ring'
        })
      );
      layers.push(
        new SpatialAnalysisPointLayer({
          id: 'chicago-route-workbench-destination-position',
          coordinateOrigin,
          positions: destinationMarkerBuffer,
          instanceCount: 1,
          radiusPixels: 8,
          color: [255, 255, 255, 255],
          outlineColor: [39, 115, 132, 255],
          outlineWidthPixels: 2,
          shape: 'ring'
        })
      );
      return layers;
    },
    getTooltip(event) {
      if (!event.coordinate) return null;
      if (!routesValid) return 'Route results pending for current options';
      const pointDescription =
        pointRouteStatus === 1
          ? 'Point-on-link query invalid'
          : pointRouteStatus === 2
            ? Number.isFinite(pointRouteValue)
              ? `Point-on-link incumbent: ${formatMinutes(pointRouteValue)} (incomplete)`
              : 'Point-on-link search incomplete'
            : Number.isFinite(pointRouteValue)
              ? `Point-on-link cost: ${formatMinutes(pointRouteValue)}`
              : 'No feasible directed point-on-link route';
      const timeDescription = !selectedArrivalValid
        ? 'Selected time model invalid'
        : !Number.isFinite(selectedArrivalValue)
          ? selectedArrivalConverged
            ? 'No feasible time-dependent route'
            : 'Time search incomplete; no route found yet'
          : `Selected arrival: ${String(Math.floor((8 * 60 + selectedArrivalValue) / 60) % 24).padStart(2, '0')}:${String(Math.floor((8 * 60 + selectedArrivalValue) % 60)).padStart(2, '0')} local`;
      if (!routeCount)
        return `${pointDescription}\n${timeDescription}\n${costReuseDescription}\n${routeComplete ? 'No feasible alternative in the clipped graph' : 'No route returned; bounded search is inconclusive'}`;
      return `${pointDescription}\n${timeDescription}\nComputed alternatives: ${Math.min(routeCount, 3)}\nRoute 1 cost: ${formatMinutes(routeCosts[0])}\n${costReuseDescription}\nShared directed links: see white overlay\n${routeComplete ? 'Top-k certified or all routes exhausted within this graph' : 'Search stopped at capacity; more routes may exist'}`;
    },
    destroy() {
      destroyed = true;
      routeReader.stop();
      computeReader.stop();
      resources.destroy();
    }
  };
}

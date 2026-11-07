// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Layer} from '@deck.gl/core';
import {Buffer} from '@luma.gl/core';
import {GPUCommandGraph, type CompiledGPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {
  GPUNetworkCentrality,
  GPUNetworkCoarsening,
  GPUNetworkLouvain,
  GPUNetworkStructure
} from '@luma-contributors/analysis/gpu-network';
import {createGraphBufferImporter} from '../../engine/graph-buffers';
import {createAnalysisFrameProjection} from '../../engine/analysis-frame';
import {SpatialAnalysisPointLayer, SpatialAnalysisSegmentLayer} from '../../engine/layers';
import {SpatialAnalysisResources} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import {GraphSwap} from '../../engine/graph-swap';
import type {LoadedDataset} from '../../data/catalog';
import type {MapAnnotation} from '../../cartography/types';
import type {SceneContext, SceneInstance} from '../scene';
import {loadAirportTable} from './b11-geography';

export type AviationNetworkOptions = {
  region: 'west' | 'northeast' | 'south';
  measure: 'traffic' | 'distanceBroker' | 'hopBroker' | 'communities' | 'components';
  gamma: number;
  communityModel: 'directed' | 'symmetric';
};

const REGION_BOUNDS = {
  west: [-125, 25, -105, 50],
  northeast: [-90, 35, -65, 50],
  south: [-105, 24, -75, 38]
} as const;

type BuiltGraph = {
  nodeCount: number;
  edgeCount: number;
  names: string[];
  iata: string[];
  positions: Float32Array;
  lonLat: Float32Array;
  airportFlights: Float32Array;
  sources: Uint32Array;
  targets: Uint32Array;
  distance: Float32Array;
  flights: Float32Array;
  offsets: Uint32Array;
  reverseOffsets: Uint32Array;
  reverseNeighbors: Uint32Array;
  reverseWeightsIndex: Uint32Array;
  denseWeights: Float32Array;
  symmetricOffsets: Uint32Array;
  symmetricTargets: Uint32Array;
  symmetricCoarseningWeights: Float32Array;
  symmetricDenseWeights: Float32Array;
  symmetricEdgeCount: number;
  routeSegments: Float32Array;
};

function buildRegion(
  dataset: LoadedDataset,
  airports: Awaited<ReturnType<typeof loadAirportTable>>,
  region: AviationNetworkOptions['region']
): BuiltGraph {
  const locations = dataset.column<Float32Array>('locations');
  const origins = dataset.column<Uint32Array>('origin');
  const destinations = dataset.column<Uint32Array>('destination');
  const counts = dataset.column<Uint32Array>('count');
  const distances = dataset.column<Float32Array>('distanceKm');
  const bounds = REGION_BOUNDS[region];
  const flightsByAirport = new Float64Array(locations.length / 2);
  for (let row = 0; row < origins.length; row++) {
    flightsByAirport[origins[row]] += counts[row];
    flightsByAirport[destinations[row]] += counts[row];
  }
  const selected = Array.from({length: flightsByAirport.length}, (_, airport) => airport)
    .filter(airport => {
      const longitude = locations[airport * 2],
        latitude = locations[airport * 2 + 1];
      return (
        longitude >= bounds[0] &&
        longitude <= bounds[2] &&
        latitude >= bounds[1] &&
        latitude <= bounds[3]
      );
    })
    .sort((first, second) => flightsByAirport[second] - flightsByAirport[first])
    .slice(0, 24);
  const localIndex = new Int32Array(flightsByAirport.length).fill(-1);
  selected.forEach((airport, index) => {
    localIndex[airport] = index;
  });
  const rows: {source: number; target: number; count: number; distance: number}[] = [];
  for (let row = 0; row < origins.length; row++) {
    const source = localIndex[origins[row]],
      target = localIndex[destinations[row]];
    if (source >= 0 && target >= 0)
      rows.push({source, target, count: counts[row], distance: distances[row]});
  }
  rows.sort((first, second) => first.source - second.source || first.target - second.target);
  const nodeCount = selected.length,
    edgeCount = rows.length;
  const positions = new Float32Array(nodeCount * 2),
    lonLat = new Float32Array(nodeCount * 2);
  const names = new Array<string>(nodeCount),
    iata = new Array<string>(nodeCount),
    airportFlights = new Float32Array(nodeCount);
  selected.forEach((airport, node) => {
    lonLat.set(locations.subarray(airport * 2, airport * 2 + 2), node * 2);
    airportFlights[node] = flightsByAirport[airport];
    names[node] = airports[airport]?.name ?? `Airport ${airport}`;
    iata[node] = airports[airport]?.iata ?? '---';
  });
  const sources = Uint32Array.from(rows, row => row.source),
    targets = Uint32Array.from(rows, row => row.target);
  const distance = Float32Array.from(rows, row => Math.max(1, row.distance));
  const flights = Float32Array.from(rows, row => row.count);
  const offsets = new Uint32Array(nodeCount + 1);
  for (const source of sources) offsets[source + 1]++;
  for (let node = 0; node < nodeCount; node++) offsets[node + 1] += offsets[node];
  const reverseOrder = Array.from({length: edgeCount}, (_, edge) => edge).sort(
    (first, second) => targets[first] - targets[second] || sources[first] - sources[second]
  );
  const reverseOffsets = new Uint32Array(nodeCount + 1);
  for (const edge of reverseOrder) reverseOffsets[targets[edge] + 1]++;
  for (let node = 0; node < nodeCount; node++) reverseOffsets[node + 1] += reverseOffsets[node];
  const reverseNeighbors = new Uint32Array(edgeCount),
    reverseWeightsIndex = Uint32Array.from(reverseOrder);
  reverseOrder.forEach((edge, index) => {
    reverseNeighbors[index] = sources[edge];
  });
  const denseWeights = new Float32Array(nodeCount * nodeCount);
  rows.forEach(row => {
    denseWeights[row.source * nodeCount + row.target] += row.count;
  });
  // Match total weight while removing direction: each symmetric pair gets the mean of Wuv and Wvu.
  const symmetricDenseWeights = new Float32Array(nodeCount * nodeCount);
  for (let source = 0; source < nodeCount; source++) {
    for (let target = 0; target < nodeCount; target++) {
      const forward = denseWeights[source * nodeCount + target];
      const reverse = denseWeights[target * nodeCount + source];
      symmetricDenseWeights[source * nodeCount + target] = (forward + reverse) * 0.5;
    }
  }
  const symmetricRows: {source: number; target: number}[] = [];
  const symmetricCoarseningRows: {source: number; target: number; weight: number}[] = [];
  for (let source = 0; source < nodeCount; source++)
    for (let target = 0; target < nodeCount; target++) {
      const weight = symmetricDenseWeights[source * nodeCount + target];
      if (weight > 0) {
        symmetricRows.push({source, target});
        const forward = denseWeights[source * nodeCount + target];
        const reverse = denseWeights[target * nodeCount + source];
        symmetricCoarseningRows.push({
          source,
          target,
          weight: source === target ? forward : forward + reverse
        });
      }
    }
  const symmetricOffsets = new Uint32Array(nodeCount + 1);
  for (const row of symmetricRows) symmetricOffsets[row.source + 1]++;
  for (let node = 0; node < nodeCount; node++) symmetricOffsets[node + 1] += symmetricOffsets[node];
  return {
    nodeCount,
    edgeCount,
    names,
    iata,
    positions,
    lonLat,
    airportFlights,
    sources,
    targets,
    distance,
    flights,
    offsets,
    reverseOffsets,
    reverseNeighbors,
    reverseWeightsIndex,
    denseWeights,
    symmetricOffsets,
    symmetricTargets: Uint32Array.from(symmetricRows, row => row.target),
    symmetricCoarseningWeights: Float32Array.from(symmetricCoarseningRows, row => row.weight),
    symmetricDenseWeights,
    symmetricEdgeCount: symmetricRows.length,
    routeSegments: new Float32Array(edgeCount * 4)
  };
}

async function buildAviationNetworkRegion(
  ctx: SceneContext<AviationNetworkOptions>,
  region: AviationNetworkOptions['region']
): Promise<{
  region: AviationNetworkOptions['region'];
  instance: SceneInstance<AviationNetworkOptions>;
  activate: () => void;
  deactivate: () => void;
}> {
  const dataset = ctx.datasets.get('us-airline-flows');
  const airports = await loadAirportTable('us-airline-flows', ctx.signal);
  const resources = new SpatialAnalysisResources(ctx.device, 'aviation-network-lab');
  const graph = new GPUCommandGraph<void>(ctx.device, {id: 'aviation-network-lab'});
  const view = createGraphBufferImporter(graph);
  const regionData = buildRegion(dataset, airports, region);
  let active = false;
  const stagedReadouts = new Map<string, string | number | null>();
  type ChartValue = Parameters<SceneContext<AviationNetworkOptions>['setChart']>[1];
  const stagedCharts = new Map<string, ChartValue>();
  const stagedAnnotations = new Map<string, readonly MapAnnotation[] | null>();
  const setReadout = (id: string, value: string | number | null) => {
    if (active) ctx.setReadout(id, value);
    else stagedReadouts.set(id, value);
  };
  const setChart = (id: string, chart: ChartValue) => {
    if (active) ctx.setChart(id, chart);
    else stagedCharts.set(id, chart);
  };
  const setAnnotations = (key: string, annotations: readonly MapAnnotation[] | null) => {
    if (active) ctx.setAnnotations(key, annotations);
    else stagedAnnotations.set(key, annotations);
  };
  const requestLayers = () => {
    if (active) ctx.requestLayers();
  };
  const {nodeCount, edgeCount} = regionData;
  const projection = createAnalysisFrameProjection({
    kind: 'albers-equal-area',
    origin: [-96, 37.5],
    standardParallels: [29.5, 45.5]
  });
  for (let node = 0; node < nodeCount; node++)
    regionData.positions.set(
      projection.project(regionData.lonLat[node * 2], regionData.lonLat[node * 2 + 1]),
      node * 2
    );
  for (let edge = 0; edge < edgeCount; edge++) {
    regionData.routeSegments.set(
      regionData.positions.subarray(regionData.sources[edge] * 2, regionData.sources[edge] * 2 + 2),
      edge * 4
    );
    regionData.routeSegments.set(
      regionData.positions.subarray(regionData.targets[edge] * 2, regionData.targets[edge] * 2 + 2),
      edge * 4 + 2
    );
  }
  const nodePositions = resources.createBuffer(
    'airport-positions',
    regionData.positions,
    Buffer.VERTEX
  );
  const routeSegments = resources.createBuffer(
    'directed-flight-legs',
    regionData.routeSegments,
    Buffer.VERTEX
  );
  const offsetsBuffer = resources.createBuffer('outgoing-offsets', regionData.offsets);
  const neighborsBuffer = resources.createBuffer('outgoing-neighbors', regionData.targets);
  const reverseOffsetsBuffer = resources.createBuffer(
    'incoming-offsets',
    regionData.reverseOffsets
  );
  const reverseNeighborsBuffer = resources.createBuffer(
    'incoming-neighbors',
    regionData.reverseNeighbors
  );
  const distanceBuffer = resources.createBuffer('route-distance-km', regionData.distance);
  const reverseDistanceBuffer = resources.createBuffer(
    'reverse-route-distance-km',
    Float32Array.from(regionData.reverseWeightsIndex, edge => regionData.distance[edge])
  );
  const hopWeightsBuffer = resources.createBuffer(
    'unit-hop-costs',
    new Float32Array(edgeCount).fill(1)
  );
  const reverseHopWeightsBuffer = resources.createBuffer(
    'reverse-unit-hop-costs',
    new Float32Array(edgeCount).fill(1)
  );
  const flightBuffer = resources.createBuffer('scheduled-flight-count', regionData.flights);
  const pivotBuffer = resources.createBuffer(
    'all-airport-pivots',
    Uint32Array.from({length: nodeCount}, (_, node) => node)
  );
  const denseWeightsBuffer = resources.createBuffer(
    'dense-flight-adjacency',
    regionData.denseWeights
  );
  const symmetricWeightsBuffer = resources.createBuffer(
    'symmetrized-flight-adjacency',
    regionData.symmetricDenseWeights
  );
  const symmetricOffsetsBuffer = resources.createBuffer(
    'symmetrized-flight-offsets',
    regionData.symmetricOffsets
  );
  const symmetricNeighborsBuffer = resources.createBuffer(
    'symmetrized-flight-neighbors',
    regionData.symmetricTargets
  );
  const symmetricCoarseningWeightsBuffer = resources.createBuffer(
    'symmetric-coarsening-input-edge-weights',
    regionData.symmetricCoarseningWeights
  );
  const louvainParametersBuffer = resources.createParameterBuffer(
    'community-parameters',
    'float32',
    2
  );
  const distanceCentralityBuffer = resources.createBuffer('distance-betweenness', nodeCount * 4);
  const hopCentralityBuffer = resources.createBuffer('hop-betweenness', nodeCount * 4);
  const communitiesBuffer = resources.createBuffer('communities', nodeCount * 4);
  const objectiveBuffer = resources.createBuffer('community-objective', (1 + 8 * 16) * 4);
  const communitySummaryBuffer = resources.createBuffer('community-summary', 16);
  const symmetricCommunitiesBuffer = resources.createBuffer('symmetric-communities', nodeCount * 4);
  const symmetricObjectiveBuffer = resources.createBuffer(
    'symmetric-community-objective',
    (1 + 8 * 16) * 4
  );
  const symmetricCommunitySummaryBuffer = resources.createBuffer('symmetric-community-summary', 16);
  const louvainParametersView = louvainParametersBuffer.importToGraph(graph);
  const coarseningBuffers = (prefix: string) => ({
    groupNodeCount: resources.createBuffer(`${prefix}-group-node-count`, nodeCount * 4),
    groupIntraEdgeCount: resources.createBuffer(`${prefix}-group-intra-count`, nodeCount * 4),
    groupIntraWeight: resources.createBuffer(`${prefix}-group-intra-weight`, nodeCount * 4),
    edges: resources.createBuffer(`${prefix}-edges`, nodeCount * nodeCount * 4),
    edgeTargetGroups: resources.createBuffer(`${prefix}-edge-targets`, nodeCount * nodeCount * 4),
    edgeCounts: resources.createBuffer(`${prefix}-edge-counts`, nodeCount * nodeCount * 4),
    edgeWeights: resources.createBuffer(`${prefix}-edge-weights`, nodeCount * nodeCount * 4),
    edgeCount: resources.createBuffer(`${prefix}-edge-count`, 4),
    summary: resources.createBuffer(`${prefix}-summary`, 8 * 4)
  });
  const directedCoarseningBuffers = coarseningBuffers('directed-coarsening');
  const symmetricCoarseningBuffers = coarseningBuffers('symmetric-coarsening');
  const directedCoarseningOverflow = resources.createBuffer('directed-coarsening-overflow', 4);
  const symmetricCoarseningOverflow = resources.createBuffer('symmetric-coarsening-overflow', 4);
  const componentsBuffer = resources.createBuffer('strong-components', nodeCount * 4);
  const distanceStatusBuffer = resources.createBuffer('distance-status', 4);
  const hopStatusBuffer = resources.createBuffer('hop-status', 4);
  const topology = {
    offsets: view(offsetsBuffer, 'uint32', nodeCount + 1),
    neighbors: view(neighborsBuffer, 'uint32', edgeCount)
  };
  const reverseTopology = {
    offsets: view(reverseOffsetsBuffer, 'uint32', nodeCount + 1),
    neighbors: view(reverseNeighborsBuffer, 'uint32', edgeCount)
  };
  graph.add(
    new GPUNetworkCentrality({
      id: 'distance-centrality',
      topology: {...topology, weights: view(distanceBuffer, 'float32', edgeCount)},
      reverseTopology: {
        ...reverseTopology,
        weights: view(reverseDistanceBuffer, 'float32', edgeCount)
      },
      pivotNodes: view(pivotBuffer, 'uint32', nodeCount),
      laneCount: nodeCount,
      maximumIterations: nodeCount,
      localIterations: 8,
      output: {
        betweenness: view(distanceCentralityBuffer, 'float32', nodeCount),
        status: {converged: view(distanceStatusBuffer, 'uint32', 1)}
      }
    })
  );
  graph.add(
    new GPUNetworkLouvain({
      id: 'symmetrized-scheduled-flight-communities',
      nodeCount,
      directed: false,
      seed: 17,
      weights: view(symmetricWeightsBuffer, 'float32', nodeCount * nodeCount),
      parameters: louvainParametersView,
      maximumLevels: 8,
      maximumIterations: 16,
      output: {
        communities: view(symmetricCommunitiesBuffer, 'uint32', nodeCount),
        objectiveTrace: view(symmetricObjectiveBuffer, 'float32', 1 + 8 * 16),
        summary: view(symmetricCommunitySummaryBuffer, 'uint32', 4)
      }
    })
  );
  graph.add(
    new GPUNetworkLouvain({
      id: 'scheduled-flight-communities',
      nodeCount,
      directed: true,
      seed: 17,
      weights: view(denseWeightsBuffer, 'float32', nodeCount * nodeCount),
      parameters: louvainParametersView,
      maximumLevels: 8,
      maximumIterations: 16,
      output: {
        communities: view(communitiesBuffer, 'uint32', nodeCount),
        objectiveTrace: view(objectiveBuffer, 'float32', 1 + 8 * 16),
        summary: view(communitySummaryBuffer, 'uint32', 4)
      }
    })
  );
  const communityLabels = view(communitiesBuffer, 'uint32', nodeCount);
  const symmetricCommunityLabels = view(symmetricCommunitiesBuffer, 'uint32', nodeCount);
  graph.add(
    new GPUNetworkCoarsening({
      id: 'directed-community-coarsening',
      topology: {...topology, weights: view(flightBuffer, 'float32', edgeCount)},
      directed: true,
      labels: communityLabels,
      groupCapacity: nodeCount,
      output: {
        groupNodeCount: view(directedCoarseningBuffers.groupNodeCount, 'uint32', nodeCount),
        groupIntraEdgeCount: view(
          directedCoarseningBuffers.groupIntraEdgeCount,
          'uint32',
          nodeCount
        ),
        groupIntraWeight: view(directedCoarseningBuffers.groupIntraWeight, 'float32', nodeCount),
        edges: {
          ids: view(directedCoarseningBuffers.edges, 'uint32', nodeCount * nodeCount),
          status: {
            count: view(directedCoarseningBuffers.edgeCount, 'uint32', 1),
            overflow: view(directedCoarseningOverflow, 'uint32', 1)
          }
        },
        edgeTargetGroups: view(
          directedCoarseningBuffers.edgeTargetGroups,
          'uint32',
          nodeCount * nodeCount
        ),
        edgeCounts: view(directedCoarseningBuffers.edgeCounts, 'uint32', nodeCount * nodeCount),
        edgeWeights: view(directedCoarseningBuffers.edgeWeights, 'float32', nodeCount * nodeCount),
        summary: view(directedCoarseningBuffers.summary, 'uint32', 8)
      }
    })
  );
  const symmetricTopology = {
    offsets: view(symmetricOffsetsBuffer, 'uint32', nodeCount + 1),
    neighbors: view(symmetricNeighborsBuffer, 'uint32', regionData.symmetricEdgeCount),
    weights: view(symmetricCoarseningWeightsBuffer, 'float32', regionData.symmetricEdgeCount)
  };
  graph.add(
    new GPUNetworkCoarsening({
      id: 'symmetric-community-coarsening',
      topology: symmetricTopology,
      directed: false,
      labels: symmetricCommunityLabels,
      groupCapacity: nodeCount,
      output: {
        groupNodeCount: view(symmetricCoarseningBuffers.groupNodeCount, 'uint32', nodeCount),
        groupIntraEdgeCount: view(
          symmetricCoarseningBuffers.groupIntraEdgeCount,
          'uint32',
          nodeCount
        ),
        groupIntraWeight: view(symmetricCoarseningBuffers.groupIntraWeight, 'float32', nodeCount),
        edges: {
          ids: view(symmetricCoarseningBuffers.edges, 'uint32', nodeCount * nodeCount),
          status: {
            count: view(symmetricCoarseningBuffers.edgeCount, 'uint32', 1),
            overflow: view(symmetricCoarseningOverflow, 'uint32', 1)
          }
        },
        edgeTargetGroups: view(
          symmetricCoarseningBuffers.edgeTargetGroups,
          'uint32',
          nodeCount * nodeCount
        ),
        edgeCounts: view(symmetricCoarseningBuffers.edgeCounts, 'uint32', nodeCount * nodeCount),
        edgeWeights: view(symmetricCoarseningBuffers.edgeWeights, 'float32', nodeCount * nodeCount),
        summary: view(symmetricCoarseningBuffers.summary, 'uint32', 8)
      }
    })
  );
  graph.add(
    new GPUNetworkCentrality({
      id: 'hop-centrality',
      topology: {...topology, weights: view(hopWeightsBuffer, 'float32', edgeCount)},
      reverseTopology: {
        ...reverseTopology,
        weights: view(reverseHopWeightsBuffer, 'float32', edgeCount)
      },
      pivotNodes: view(pivotBuffer, 'uint32', nodeCount),
      laneCount: nodeCount,
      maximumIterations: nodeCount,
      localIterations: 8,
      output: {
        betweenness: view(hopCentralityBuffer, 'float32', nodeCount),
        status: {converged: view(hopStatusBuffer, 'uint32', 1)}
      }
    })
  );
  graph.add(
    new GPUNetworkStructure({
      id: 'directed-airport-components',
      topology,
      output: {strongComponents: view(componentsBuffer, 'uint32', nodeCount)}
    })
  );
  const compiled: CompiledGPUCommandGraph<void> = resources.track(await graph.compileAsync());
  let generation = 1,
    pendingGeneration = 0,
    dirty = true,
    destroyed = false;
  let lastAnalysis: {
    distance: Float32Array;
    hops: Float32Array;
    communities: Uint32Array;
    components: Uint32Array;
    communitySummary: Uint32Array;
    symmetricCommunities: Uint32Array;
    symmetricCommunitySummary: Uint32Array;
    coarsening: {
      directed: {
        summary: Uint32Array;
        intraWeights: Float32Array;
        edgeWeights: Float32Array;
        overflow: number;
      };
      symmetric: {
        summary: Uint32Array;
        intraWeights: Float32Array;
        edgeWeights: Float32Array;
        overflow: number;
      };
    };
    converged: boolean;
  } | null = null;
  const airportVolumeBuffer = resources.createBuffer(
    'airport-flight-volume',
    regionData.airportFlights
  );
  setReadout(
    'networkShape',
    `${nodeCount} airports · ${edgeCount} directed observed airport pairs`
  );
  setReadout(
    'scheduledFlights',
    `${Math.round(regionData.flights.reduce((sum, value) => sum + value, 0)).toLocaleString()} scheduled flight legs`
  );
  setReadout('solver', 'Waiting for GPU graph analysis');
  setReadout('topBrokers', '—');
  setReadout('components', '—');
  setReadout('communities', 'Waiting for GPU partition');
  setReadout('communityMembers', 'Waiting for GPU partition');
  setReadout('communityCoarsening', 'Waiting for GPU partition');
  setChart('communityCoarseningChart', null);
  setChart('centralityChart', null);
  const reader = new SummaryReader(
    resources,
    'centrality-and-community-summary',
    [
      {buffer: distanceCentralityBuffer, size: nodeCount * 4},
      {buffer: hopCentralityBuffer, size: nodeCount * 4},
      {buffer: communitiesBuffer, size: nodeCount * 4},
      {buffer: componentsBuffer, size: nodeCount * 4},
      {buffer: communitySummaryBuffer, size: 16},
      {buffer: symmetricCommunitiesBuffer, size: nodeCount * 4},
      {buffer: symmetricCommunitySummaryBuffer, size: 16},
      {buffer: directedCoarseningBuffers.summary, size: 32},
      {buffer: directedCoarseningBuffers.groupIntraWeight, size: nodeCount * 4},
      {buffer: directedCoarseningBuffers.edgeWeights, size: nodeCount * nodeCount * 4},
      {buffer: directedCoarseningOverflow, size: 4},
      {buffer: symmetricCoarseningBuffers.summary, size: 32},
      {buffer: symmetricCoarseningBuffers.groupIntraWeight, size: nodeCount * 4},
      {buffer: symmetricCoarseningBuffers.edgeWeights, size: nodeCount * nodeCount * 4},
      {buffer: symmetricCoarseningOverflow, size: 4},
      {buffer: distanceStatusBuffer, size: 4},
      {buffer: hopStatusBuffer, size: 4}
    ],
    bytes => {
      if (destroyed || pendingGeneration !== generation) return;
      const floats = new Float32Array(bytes),
        words = new Uint32Array(bytes);
      const distanceCentrality = floats.slice(0, nodeCount),
        hopCentrality = floats.slice(nodeCount, nodeCount * 2);
      const communities = words.slice(nodeCount * 2, nodeCount * 3),
        components = words.slice(nodeCount * 3, nodeCount * 4);
      let offset = nodeCount * 4;
      const communitySummary = words.slice(offset, offset + 4);
      offset += 4;
      const symmetricCommunities = words.slice(offset, offset + nodeCount);
      offset += nodeCount;
      const symmetricCommunitySummary = words.slice(offset, offset + 4);
      offset += 4;
      const directedCoarseningSummary = words.slice(offset, offset + 8);
      offset += 8;
      const directedIntraWeights = floats.slice(offset, offset + nodeCount);
      offset += nodeCount;
      const directedEdgeWeights = floats.slice(offset, offset + nodeCount * nodeCount);
      offset += nodeCount * nodeCount;
      const directedOverflow = words[offset++];
      const symmetricCoarseningSummary = words.slice(offset, offset + 8);
      offset += 8;
      const symmetricIntraWeights = floats.slice(offset, offset + nodeCount);
      offset += nodeCount;
      const symmetricEdgeWeights = floats.slice(offset, offset + nodeCount * nodeCount);
      offset += nodeCount * nodeCount;
      const symmetricOverflow = words[offset++];
      const statusOffset = offset;
      const converged = words[statusOffset] !== 0 && words[statusOffset + 1] !== 0;
      const directedSuperedgeCount = directedCoarseningSummary[7];
      const symmetricSuperedgeCount = symmetricCoarseningSummary[7];
      lastAnalysis = {
        distance: distanceCentrality,
        hops: hopCentrality,
        communities,
        components,
        communitySummary,
        symmetricCommunities,
        symmetricCommunitySummary,
        coarsening: {
          directed: {
            summary: directedCoarseningSummary,
            intraWeights: directedIntraWeights,
            edgeWeights: directedEdgeWeights.subarray(0, directedSuperedgeCount),
            overflow: directedOverflow
          },
          symmetric: {
            summary: symmetricCoarseningSummary,
            intraWeights: symmetricIntraWeights,
            edgeWeights: symmetricEdgeWeights.subarray(0, symmetricSuperedgeCount),
            overflow: symmetricOverflow
          }
        },
        converged
      };
      publishSummary();
    }
  );
  function publishSummary(): void {
    if (!lastAnalysis) return;
    const {distance, hops, components, communitySummary, converged} = lastAnalysis;
    const selectedCommunitySummary =
      ctx.options.communityModel === 'directed'
        ? communitySummary
        : lastAnalysis.symmetricCommunitySummary;
    const selectedCommunities =
      ctx.options.communityModel === 'directed'
        ? lastAnalysis.communities
        : lastAnalysis.symmetricCommunities;
    const selectedCoarsening =
      ctx.options.communityModel === 'directed'
        ? lastAnalysis.coarsening.directed
        : lastAnalysis.coarsening.symmetric;
    const selectedCentrality = ctx.options.measure === 'hopBroker' ? hops : distance;
    const sorted = Array.from({length: nodeCount}, (_, node) => node).sort(
      (first, second) => selectedCentrality[second] - selectedCentrality[first]
    );
    const top = sorted
      .slice(0, 5)
      .map(node => `${regionData.iata[node]} ${selectedCentrality[node].toFixed(2)}`)
      .join(' · ');
    setReadout('topBrokers', converged ? top : 'withheld until centrality converges');
    setReadout('components', `${new Set(components).size} directed strongly connected components`);
    setReadout(
      'communities',
      `${ctx.options.communityModel === 'directed' ? 'Directed' : 'Symmetric'} Louvain · ${selectedCommunitySummary[2]} levels · ${selectedCommunitySummary[3]} sweeps · ${selectedCommunitySummary[1] ? 'converged' : 'iteration limit'}${selectedCommunitySummary[0] ? '' : ' · invalid partition'}`
    );
    const membersByCommunity = new Map<number, string[]>();
    for (let node = 0; node < nodeCount; node++) {
      const community = selectedCommunities[node];
      const members = membersByCommunity.get(community) ?? [];
      members.push(regionData.iata[node]);
      membersByCommunity.set(community, members);
    }
    setReadout(
      'communityMembers',
      selectedCommunitySummary[0] === 1
        ? [...membersByCommunity.entries()]
            .sort(([first], [second]) => first - second)
            .map(([community, members]) => `${community}: ${members.join(',')}`)
            .join(' · ')
        : 'membership withheld; invalid partition'
    );
    const intraWeight = selectedCoarsening.intraWeights.reduce((sum, value) => sum + value, 0);
    const interWeight = selectedCoarsening.edgeWeights.reduce((sum, value) => sum + value, 0);
    const inputWeight = regionData.flights.reduce((sum, value) => sum + value, 0);
    const coarseningSummary = selectedCoarsening.summary;
    const coarseningComplete =
      selectedCommunitySummary[0] === 1 &&
      selectedCoarsening.overflow === 0 &&
      coarseningSummary[1] === 0 &&
      coarseningSummary[3] === 0;
    const weightError = Math.abs(intraWeight + interWeight - inputWeight);
    setReadout(
      'communityCoarsening',
      coarseningComplete
        ? `${coarseningSummary[6]} groups · ${coarseningSummary[5]} inter-group pairs / ${coarseningSummary[4]} within-group pairs · weight ${Math.round(intraWeight).toLocaleString()} within + ${Math.round(interWeight).toLocaleString()} between = ${Math.round(intraWeight + interWeight).toLocaleString()} / ${Math.round(inputWeight).toLocaleString()} ${weightError < 1 ? 'conserved' : `difference ${weightError.toFixed(1)}`}`
        : selectedCommunitySummary[0] !== 1
          ? 'Coarsening withheld; selected community partition is invalid'
          : 'Coarsening incomplete; overflow or dropped links'
    );
    setChart(
      'communityCoarseningChart',
      coarseningComplete
        ? {
            kind: 'bars',
            height: 112,
            values: [intraWeight, interWeight],
            labels: ['within groups', 'between groups'],
            highlight: [0],
            yLabel: 'scheduled-count weight',
            description: `${ctx.options.communityModel} community-link aggregation; total weight is checked against the original regional graph.`
          }
        : null
    );
    setReadout(
      'solver',
      converged
        ? 'Both centrality searches converged'
        : 'Centrality search did not converge; rankings withheld'
    );
    setChart(
      'centralityChart',
      converged
        ? {
            kind: 'bars',
            height: 130,
            values: sorted.slice(0, 8).map(node => selectedCentrality[node]),
            labels: sorted.slice(0, 8).map(node => regionData.iata[node]),
            highlight: [0],
            yLabel: 'broker score',
            description:
              ctx.options.measure === 'hopBroker'
                ? 'Directed betweenness with each observed flight leg assigned unit cost.'
                : 'Directed betweenness on great-circle route distance.'
          }
        : null
    );
    if (
      converged &&
      (ctx.options.measure === 'distanceBroker' || ctx.options.measure === 'hopBroker')
    ) {
      setAnnotations(
        'aviation-leaders',
        sorted.slice(0, 5).map(
          (node, index): MapAnnotation => ({
            kind: 'point',
            coordinate: [regionData.lonLat[node * 2], regionData.lonLat[node * 2 + 1]],
            text: `${regionData.iata[node]} ${selectedCentrality[node].toFixed(1)}`,
            detail: regionData.names[node],
            marker: 'none',
            rank: index < 3 ? 'subject' : 'context',
            priority: 10 - index,
            id: `aviation-leader:${region}:${ctx.options.measure}:${regionData.iata[node]}`
          })
        )
      );
    } else setAnnotations('aviation-leaders', null);
    requestLayers();
  }
  function update(): void {
    louvainParametersBuffer.write(Float32Array.of(ctx.options.gamma, 0.0001));
    generation++;
    dirty = true;
    lastAnalysis = null;
    setReadout('solver', 'Waiting for GPU graph analysis');
    setReadout('topBrokers', '—');
    setReadout('components', '—');
    setReadout('communities', '—');
    setReadout('communityMembers', '—');
    setReadout('communityCoarsening', '—');
    setChart('communityCoarseningChart', null);
    setChart('centralityChart', null);
  }
  louvainParametersBuffer.write(Float32Array.of(ctx.options.gamma, 0.0001));
  const instance: SceneInstance<AviationNetworkOptions> = {
    getCompiledGraphs: () => [compiled],
    encode(commandEncoder) {
      if (dirty) {
        compiled.encode(commandEncoder, {parameters: undefined});
        dirty = false;
        reader.markStale();
      }
      if (!reader.isPending && generation !== pendingGeneration) pendingGeneration = generation;
      reader.flush(commandEncoder);
    },
    setOption(id) {
      if (id === 'gamma') update();
      else if (id === 'measure' || id === 'communityModel') {
        if (lastAnalysis) publishSummary();
        else requestLayers();
      } else requestLayers();
    },
    getLayers() {
      const dark = ctx.ground() === 'dark';
      const outputReady =
        Boolean(lastAnalysis) &&
        (ctx.options.measure === 'components' ||
          (lastAnalysis!.converged && ctx.options.measure !== 'communities') ||
          (ctx.options.measure === 'communities' &&
            (ctx.options.communityModel === 'directed'
              ? lastAnalysis!.communitySummary[0] === 1
              : lastAnalysis!.symmetricCommunitySummary[0] === 1)));
      const values =
        !outputReady || ctx.options.measure === 'traffic'
          ? airportVolumeBuffer
          : ctx.options.measure === 'communities'
            ? ctx.options.communityModel === 'directed'
              ? communitiesBuffer
              : symmetricCommunitiesBuffer
            : ctx.options.measure === 'components'
              ? componentsBuffer
              : ctx.options.measure === 'distanceBroker'
                ? distanceCentralityBuffer
                : hopCentralityBuffer;
      const format =
        outputReady &&
        (ctx.options.measure === 'communities' || ctx.options.measure === 'components')
          ? ('uint32' as const)
          : ('float32' as const);
      const maximumVolume = Math.max(...regionData.airportFlights);
      const mapVolumeMaximum = region === 'west' ? 35041 : region === 'northeast' ? 45584 : 59834;
      const mapDistanceMaximum = region === 'west' ? 73 : region === 'northeast' ? 65 : 70;
      const mapHopMaximum = region === 'west' ? 34 : region === 'northeast' ? 14 : 35;
      return [
        new SpatialAnalysisSegmentLayer({
          id: 'aviation-scheduled-flow-arcs',
          analysisFrame: projection.frame,
          segments: routeSegments,
          instanceCount: edgeCount,
          widthPixels: 0.8,
          weights: flightBuffer,
          color: dark ? [112, 164, 209, 48] : [55, 106, 150, 42]
        }),
        new SpatialAnalysisPointLayer({
          id: 'aviation-airport-metric',
          analysisFrame: projection.frame,
          positions: nodePositions,
          instanceCount: nodeCount,
          values,
          valueFormat: format,
          colormap: format === 'uint32' ? 'category' : 'magma',
          valueRange:
            format === 'float32'
              ? [
                  0,
                  ctx.options.measure === 'traffic' || !outputReady
                    ? mapVolumeMaximum
                    : ctx.options.measure === 'distanceBroker'
                      ? mapDistanceMaximum
                      : mapHopMaximum
                ]
              : undefined,
          palette: [
            [52, 180, 214, 255],
            [248, 173, 62, 255],
            [227, 77, 74, 255],
            [148, 100, 211, 255],
            [90, 190, 128, 255],
            [229, 108, 180, 255],
            [185, 191, 75, 255],
            [96, 132, 217, 255]
          ],
          radiusPixels: 9,
          sizeValues: airportVolumeBuffer,
          sizeMaximumValue: maximumVolume,
          noDataColor: dark ? [103, 113, 123, 160] : [165, 172, 178, 180],
          outlineColor: dark ? [12, 16, 23, 255] : [255, 255, 255, 255],
          outlineWidthPixels: 1
        })
      ] as Layer[];
    },
    getTooltip(event) {
      const viewport = ctx.getViewport();
      if (!viewport) return null;
      let nearest = -1,
        closest = 15 * 15;
      for (let node = 0; node < nodeCount; node++) {
        const pixel = viewport.project([
          regionData.lonLat[node * 2],
          regionData.lonLat[node * 2 + 1]
        ]);
        const distance = (pixel[0] - event.pixel[0]) ** 2 + (pixel[1] - event.pixel[1]) ** 2;
        if (distance < closest) {
          closest = distance;
          nearest = node;
        }
      }
      if (nearest < 0) return null;
      const analysis = lastAnalysis;
      const score = !analysis
        ? 'waiting for GPU output'
        : ctx.options.measure === 'communities'
          ? (ctx.options.communityModel === 'directed'
              ? analysis.communitySummary
              : analysis.symmetricCommunitySummary)[0]
            ? `${ctx.options.communityModel} community ${(ctx.options.communityModel === 'directed' ? analysis.communities : analysis.symmetricCommunities)[nearest]}`
            : 'partition invalid'
          : ctx.options.measure === 'components'
            ? `component ${analysis.components[nearest]}`
            : ctx.options.measure === 'distanceBroker'
              ? analysis.converged
                ? analysis.distance[nearest].toFixed(2)
                : 'withheld'
              : ctx.options.measure === 'hopBroker'
                ? analysis.converged
                  ? analysis.hops[nearest].toFixed(2)
                  : 'withheld'
                : '—';
      return {
        title: `${regionData.iata[nearest]} · ${regionData.names[nearest]}`,
        subtitle: 'Regional BTS airport network',
        rows: [
          {
            label: 'Direct outgoing flight arcs',
            value: `${regionData.offsets[nearest + 1] - regionData.offsets[nearest]}`
          },
          {
            label: 'Full-source scheduled legs touching airport',
            value: Math.round(regionData.airportFlights[nearest]).toLocaleString()
          },
          {
            label:
              ctx.options.measure === 'communities'
                ? `GPU ${ctx.options.communityModel} Louvain community`
                : ctx.options.measure === 'components'
                  ? 'GPU strong component'
                  : ctx.options.measure === 'traffic'
                    ? 'Map color value'
                    : 'GPU betweenness score',
            value: score
          },
          {label: 'Selected region', value: region}
        ],
        note: 'Airport volume uses all source records for this airport; route count and centrality use only the selected regional induced graph. Scheduled legs are not passengers or feasible itineraries.'
      };
    },
    onThemeChange() {
      requestLayers();
    },
    destroy() {
      destroyed = true;
      reader.stop();
      resources.destroy();
    }
  };
  return {
    region,
    instance,
    activate() {
      active = true;
      ctx.setAnnotations('aviation-leaders', null);
      for (const [id, value] of stagedReadouts) ctx.setReadout(id, value);
      for (const [id, chart] of stagedCharts) ctx.setChart(id, chart);
      for (const [key, annotations] of stagedAnnotations) ctx.setAnnotations(key, annotations);
      stagedReadouts.clear();
      stagedCharts.clear();
      stagedAnnotations.clear();
    },
    deactivate() {
      active = false;
    }
  };
}

export async function createAviationNetwork(
  ctx: SceneContext<AviationNetworkOptions>
): Promise<SceneInstance<AviationNetworkOptions>> {
  const regionViews = {
    west: {longitude: -114, latitude: 38, zoom: 3.6},
    northeast: {longitude: -77, latitude: 41, zoom: 4},
    south: {longitude: -89, latitude: 31, zoom: 4}
  } as const;
  let activeRegion = await buildAviationNetworkRegion(ctx, ctx.options.region);
  activeRegion.activate();
  let requestedRegion = ctx.options.region;
  let destroyed = false;
  const graphSwap = new GraphSwap();
  function requestRegion(region: AviationNetworkOptions['region']): void {
    requestedRegion = region;
    ctx.setStatus(`Building the ${region} regional graph`);
    void graphSwap
      .request({
        build: () => buildAviationNetworkRegion(ctx, region),
        commit(next) {
          const previous = activeRegion;
          previous.deactivate();
          activeRegion = next;
          activeRegion.activate();
          previous.instance.destroy();
          ctx.setStatus('');
          ctx.flyTo(regionViews[next.region], {transitionMs: 1000});
          ctx.requestLayers();
        },
        discard(next) {
          next.deactivate();
          next.instance.destroy();
        }
      })
      .catch(() => {
        if (destroyed || requestedRegion !== region) return;
        requestedRegion = activeRegion.region;
        ctx.setOptions({region: activeRegion.region});
        ctx.setStatus('');
        ctx.setReadout('solver', 'Regional graph rebuild failed; previous region remains active');
      });
  }
  return {
    getCompiledGraphs: () => activeRegion.instance.getCompiledGraphs(),
    encode: (commandEncoder, frame) => activeRegion.instance.encode(commandEncoder, frame),
    setOption(id, value, state) {
      if (id === 'region' && ctx.options.region !== requestedRegion)
        requestRegion(ctx.options.region);
      else activeRegion.instance.setOption?.(id, value, state);
    },
    getLayers: () => activeRegion.instance.getLayers(),
    getTooltip: event => activeRegion.instance.getTooltip?.(event) ?? null,
    onThemeChange: theme => activeRegion.instance.onThemeChange?.(theme),
    destroy() {
      destroyed = true;
      graphSwap.destroy();
      activeRegion.deactivate();
      activeRegion.instance.destroy();
    }
  };
}

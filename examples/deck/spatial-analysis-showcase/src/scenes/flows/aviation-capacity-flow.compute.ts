// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Layer} from '@deck.gl/core';
import {Buffer} from '@luma.gl/core';
import {GPUCommandGraph, type CompiledGPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {
  GPUNetworkFlow,
  prepareGPUNetworkFlowDemands
} from '@luma-contributors/analysis/gpu-network';
import {createGraphBufferImporter} from '../../engine/graph-buffers';
import {createAnalysisFrameProjection} from '../../engine/analysis-frame';
import {addKernelPass} from '../../engine/mode-kernels';
import {SpatialAnalysisPointLayer, SpatialAnalysisSegmentLayer} from '../../engine/layers';
import {SpatialAnalysisResources} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import type {LoadedDataset} from '../../data/catalog';
import type {SceneContext, SceneInstance} from '../scene';
import {loadAirportTable} from './b11-geography';

export type AviationCapacityFlowOptions = {
  source: string;
  sink: string;
  terminalMode: 'singlePair' | 'balanced4';
  objective: 'maximum' | 'fixedBudget';
  flowBudget: number;
  capacityShare: '100' | '75' | '50' | '25';
};

const WESTERN_BOUNDS = [-125, 25, -105, 50] as const;
const MAXIMUM_FLOW_SENTINEL = 0xffffffff;
const SOURCE_SUPER_NODE = 24;
const SINK_SUPER_NODE = 25;
const SOLVER_NODE_COUNT = 26;
const SUPER_TERMINAL_BUDGET = 1_000_000;

type AirportRecord = Awaited<ReturnType<typeof loadAirportTable>>[number];
type RegionalNetwork = {
  nodeCount: number;
  edgeCount: number;
  airportIds: number[];
  airportNames: string[];
  airportIata: string[];
  airportFlights: Float32Array;
  positions: Float32Array;
  sources: Uint32Array;
  targets: Uint32Array;
  scheduledFlights: Uint32Array;
  delayCosts: Uint32Array;
  segments: Float32Array;
  projectedEndpointByIata: Map<string, [number, number]>;
};

function buildWesternNetwork(dataset: LoadedDataset, airports: AirportRecord[]): RegionalNetwork {
  const locations = dataset.column<Float32Array>('locations');
  const origins = dataset.column<Uint32Array>('origin');
  const destinations = dataset.column<Uint32Array>('destination');
  const counts = dataset.column<Uint32Array>('count');
  const delays = dataset.column<Float32Array>('meanDepDelay');
  const flightsByAirport = new Float64Array(locations.length / 2);
  for (let edge = 0; edge < origins.length; edge++) {
    flightsByAirport[origins[edge]] += counts[edge];
    flightsByAirport[destinations[edge]] += counts[edge];
  }
  const airportIds = Array.from({length: flightsByAirport.length}, (_, airport) => airport)
    .filter(airport => {
      const longitude = locations[airport * 2];
      const latitude = locations[airport * 2 + 1];
      return (
        longitude >= WESTERN_BOUNDS[0] &&
        longitude <= WESTERN_BOUNDS[2] &&
        latitude >= WESTERN_BOUNDS[1] &&
        latitude <= WESTERN_BOUNDS[3]
      );
    })
    .sort((first, second) => flightsByAirport[second] - flightsByAirport[first] || first - second)
    .slice(0, 24);
  const localIndex = new Int32Array(flightsByAirport.length).fill(-1);
  airportIds.forEach((airport, node) => (localIndex[airport] = node));
  const retainedEdges: {source: number; target: number; count: number; delay: number}[] = [];
  for (let edge = 0; edge < origins.length; edge++) {
    const source = localIndex[origins[edge]];
    const target = localIndex[destinations[edge]];
    if (source < 0 || target < 0) continue;
    retainedEdges.push({
      source,
      target,
      count: counts[edge],
      delay: Number.isFinite(delays[edge]) ? Math.max(0, delays[edge]) : 0
    });
  }
  retainedEdges.sort(
    (first, second) => first.source - second.source || first.target - second.target
  );
  const nodeCount = airportIds.length;
  const edgeCount = retainedEdges.length;
  const projection = createAnalysisFrameProjection({
    kind: 'albers-equal-area',
    origin: [-114, 38],
    standardParallels: [29.5, 45.5]
  });
  const positions = new Float32Array(nodeCount * 2);
  const airportNames = new Array<string>(nodeCount);
  const airportIata = new Array<string>(nodeCount);
  const airportFlights = new Float32Array(nodeCount);
  const projectedEndpointByIata = new Map<string, [number, number]>();
  airportIds.forEach((airport, node) => {
    const longitude = locations[airport * 2];
    const latitude = locations[airport * 2 + 1];
    positions.set(projection.project(longitude, latitude), node * 2);
    airportNames[node] = airports[airport]?.name ?? `Airport ${airport}`;
    airportIata[node] = airports[airport]?.iata ?? '---';
    airportFlights[node] = flightsByAirport[airport];
    projectedEndpointByIata.set(airportIata[node], [positions[node * 2], positions[node * 2 + 1]]);
  });
  const sources = Uint32Array.from(retainedEdges, edge => edge.source);
  const targets = Uint32Array.from(retainedEdges, edge => edge.target);
  const scheduledFlights = Uint32Array.from(retainedEdges, edge => edge.count);
  const delayCosts = Uint32Array.from(retainedEdges, edge => Math.round(edge.delay));
  const segments = new Float32Array(edgeCount * 4);
  for (let edge = 0; edge < edgeCount; edge++) {
    segments.set(positions.subarray(sources[edge] * 2, sources[edge] * 2 + 2), edge * 4);
    segments.set(positions.subarray(targets[edge] * 2, targets[edge] * 2 + 2), edge * 4 + 2);
  }
  return {
    nodeCount,
    edgeCount,
    airportIds,
    airportNames,
    airportIata,
    airportFlights,
    positions,
    sources,
    targets,
    scheduledFlights,
    delayCosts,
    segments,
    projectedEndpointByIata
  };
}

export async function createAviationCapacityFlow(
  ctx: SceneContext<AviationCapacityFlowOptions>
): Promise<SceneInstance<AviationCapacityFlowOptions>> {
  const dataset = ctx.datasets.get('us-airline-flows');
  const airports = await loadAirportTable('us-airline-flows', ctx.signal);
  const locations = dataset.column<Float32Array>('locations');
  const network = buildWesternNetwork(dataset, airports);
  const {nodeCount, edgeCount} = network;
  const solverEdgeCount = edgeCount + 4;
  const initialSourcePosition = network.projectedEndpointByIata.get(ctx.options.source) ?? [0, 0];
  const initialSinkPosition = network.projectedEndpointByIata.get(ctx.options.sink) ?? [0, 0];
  const northwestHub = network.projectedEndpointByIata.get('SEA') ?? [0, 0];
  const southwestHub = network.projectedEndpointByIata.get('LAX') ?? [0, 0];
  const terminalPositions = new Float32Array([
    ...initialSourcePosition,
    ...initialSinkPosition,
    ...northwestHub,
    ...southwestHub
  ]);
  const endpointRoles = Uint32Array.of(1, 2, 1, 2);
  const resources = new SpatialAnalysisResources(ctx.device, 'aviation-capacity-flow');
  const graph = new GPUCommandGraph<void>(ctx.device, {id: 'aviation-capacity-flow'});
  const view = createGraphBufferImporter(graph);
  const edgeBuffer = resources.createBuffer('directed-regional-links', solverEdgeCount * 8);
  const edges = new Uint32Array(solverEdgeCount * 2);
  for (let edge = 0; edge < edgeCount; edge++) {
    edges[edge * 2] = network.sources[edge];
    edges[edge * 2 + 1] = network.targets[edge];
  }
  edgeBuffer.write(edges);
  const capacityParameters = resources.createParameterBuffer(
    'scenario-link-budgets',
    'uint32',
    solverEdgeCount
  );
  const delayCostParameters = resources.createParameterBuffer(
    'delay-score-costs',
    'uint32',
    solverEdgeCount
  );
  const flowParameters = resources.createParameterBuffer('flow-request', 'uint32', 3);
  const flowBuffer = resources.createBuffer('edge-flow', solverEdgeCount * 4);
  const cutBuffer = resources.createBuffer('minimum-cut-source-side', SOLVER_NODE_COUNT * 4);
  const flowSummaryBuffer = resources.createBuffer('flow-summary', 8 * 4);
  const exactCostBuffer = resources.createBuffer('exact-delay-score', 8);
  const flowShareBuffer = resources.createBuffer('flow-capacity-share', edgeCount * 4);
  const nodePositionsBuffer = resources.createBuffer(
    'regional-airport-positions',
    network.positions,
    Buffer.VERTEX
  );
  const endpointPositionsBuffer = resources.createBuffer(
    'selected-endpoint-positions',
    terminalPositions,
    Buffer.VERTEX
  );
  const endpointRolesBuffer = resources.createBuffer('selected-endpoint-roles', endpointRoles);
  const networkSegmentsBuffer = resources.createBuffer(
    'regional-directed-links',
    network.segments,
    Buffer.VERTEX
  );
  const airportFlightsBuffer = resources.createBuffer(
    'airport-scheduled-volume',
    network.airportFlights
  );
  const edgeView = view(edgeBuffer, 'uint32x2', solverEdgeCount);
  const capacityView = capacityParameters.importToGraph(graph);
  const delayCostView = delayCostParameters.importToGraph(graph);
  const flowParameterView = flowParameters.importToGraph(graph);
  const flowView = view(flowBuffer, 'uint32', solverEdgeCount);
  const cutView = view(cutBuffer, 'uint32', SOLVER_NODE_COUNT);
  const summaryView = view(flowSummaryBuffer, 'uint32', 8);
  const exactCostView = view(exactCostBuffer, 'uint32x2', 1);
  const flowShareView = view(flowShareBuffer, 'float32', edgeCount);
  graph.add(
    new GPUNetworkFlow({
      id: 'regional-scheduled-leg-capacity-flow',
      nodeCount: SOLVER_NODE_COUNT,
      edges: edgeView,
      capacities: capacityView,
      costs: delayCostView,
      parameters: flowParameterView,
      maximumAugmentations: 1024,
      output: {flow: flowView, cut: cutView, summary: summaryView, exactCost: exactCostView}
    })
  );
  addKernelPass(graph, {
    id: 'publish-flow-utilization',
    invocationCount: edgeCount,
    bindings: [
      {name: 'flow', view: flowView, type: 'u32', access: 'read'},
      {name: 'capacity', view: capacityView, type: 'u32', access: 'read'},
      {name: 'summary', view: summaryView, type: 'u32', access: 'read'},
      {name: 'share', view: flowShareView, type: 'f32', access: 'read_write'}
    ],
    body: `if (summary[summaryOffset] == 0u) { share[shareOffset + index] = 0.0; return; }
let capacityValue = capacity[capacityOffset + index]; share[shareOffset + index] = select(0.0, f32(flow[flowOffset + index]) / f32(capacityValue), capacityValue > 0u);`
  });
  const compiled: CompiledGPUCommandGraph<void> = resources.track(await graph.compileAsync());
  let generation = 1;
  let pendingGeneration = 0;
  let dirty = true;
  let destroyed = false;
  let lastResult: {
    flow: Uint32Array;
    cut: Uint32Array;
    summary: Uint32Array;
    exactCost: Uint32Array;
    delayScore: bigint;
  } | null = null;
  const reader = new SummaryReader(
    resources,
    'regional-flow-summary',
    [
      {buffer: flowBuffer, size: solverEdgeCount * 4},
      {buffer: cutBuffer, size: SOLVER_NODE_COUNT * 4},
      {buffer: flowSummaryBuffer, size: 8 * 4},
      {buffer: exactCostBuffer, size: 8}
    ],
    bytes => {
      if (destroyed || pendingGeneration !== generation) return;
      const words = new Uint32Array(bytes);
      const flow = words.slice(0, solverEdgeCount);
      const cut = words.slice(solverEdgeCount, solverEdgeCount + SOLVER_NODE_COUNT);
      const summaryOffset = solverEdgeCount + SOLVER_NODE_COUNT;
      const flowSummary = words.slice(summaryOffset, summaryOffset + 8);
      const exactCost = words.slice(summaryOffset + 8, summaryOffset + 10);
      lastResult = {
        flow,
        cut,
        summary: flowSummary,
        exactCost,
        delayScore: (BigInt(exactCost[1]) << 32n) | BigInt(exactCost[0])
      };
      publishSummary();
    }
  );
  const iataIndex = new Map(network.airportIata.map((iata, node) => [iata, node]));
  function formatInteger(value: bigint): string {
    return value.toLocaleString('en-US');
  }
  function publishSummary(): void {
    const result = lastResult;
    if (!result) return;
    const singlePair = ctx.options.terminalMode === 'singlePair';
    if (singlePair && ctx.options.source === ctx.options.sink) {
      ctx.setReadout('flowStatus', 'Choose different source and sink airports');
      ctx.setReadout('flowValue', '—');
      ctx.setReadout('flowCut', '—');
      ctx.setReadout('delayScore', '—');
      ctx.setReadout('augmentations', '—');
      ctx.setChart('flowLinks', null);
      ctx.requestLayers();
      return;
    }
    const [valid, complete, value, unserved, augmentations, , cutCapacity, cutCertified] =
      result.summary;
    if (!valid) {
      ctx.setReadout('flowStatus', 'Input invalid; GPU result withheld');
      ctx.setReadout('flowValue', '—');
      ctx.setReadout('flowCut', '—');
      ctx.setChart('flowLinks', null);
      ctx.requestLayers();
      return;
    }
    if (!complete) {
      ctx.setReadout('flowStatus', 'Augmentation limit reached; authoritative result withheld');
      ctx.setReadout('flowValue', '—');
      ctx.setReadout('flowCut', '—');
      ctx.setReadout('delayScore', '—');
      ctx.setReadout('augmentations', `${augmentations} augmentations · incomplete`);
      ctx.setChart('flowLinks', null);
      ctx.requestLayers();
      return;
    }
    const requested =
      singlePair && ctx.options.objective === 'maximum'
        ? 'maximum throughput'
        : `${ctx.options.flowBudget} flow units`;
    const budgetMet =
      (singlePair && ctx.options.objective === 'maximum') || value >= ctx.options.flowBudget;
    ctx.setReadout(
      'flowStatus',
      budgetMet
        ? `Complete · ${requested}${cutCertified ? ' · minimum cut certified' : ''}`
        : `Budget shortfall · ${unserved.toLocaleString()} flow units unserved${cutCertified ? ' · minimum cut certified' : ''}`
    );
    ctx.setReadout('flowValue', `${value.toLocaleString()} monthly flow units`);
    ctx.setReadout(
      'flowCut',
      cutCertified
        ? `Augmented-graph cut ${cutCapacity.toLocaleString()} = flow ${value.toLocaleString()}`
        : 'Requested amount met; a maximum-flow cut was not sought'
    );
    ctx.setReadout(
      'delayScore',
      `${formatInteger(result.delayScore)} flow-unit × delay-minute score`
    );
    ctx.setReadout('augmentations', `${augmentations} augmenting paths`);
    const topEdges = Array.from({length: edgeCount}, (_, edge) => edge)
      .filter(edge => result.flow[edge] > 0)
      .sort((first, second) => result.flow[second] - result.flow[first])
      .slice(0, 8);
    ctx.setChart(
      'flowLinks',
      topEdges.length
        ? {
            kind: 'bars',
            height: 145,
            values: topEdges.map(edge => result.flow[edge]),
            labels: topEdges.map(
              edge =>
                `${network.airportIata[network.sources[edge]]}→${network.airportIata[network.targets[edge]]}`
            ),
            highlight: [0],
            yLabel: 'allocated flow units',
            description:
              'Largest allocated values on directed regional airport links; a unit is an abstract monthly flight-leg flow, not a passenger.'
          }
        : null
    );
    const balanced = !singlePair;
    const terminalIatas = balanced
      ? ['SEA', 'PDX', 'LAX', 'SAN']
      : [ctx.options.source, ctx.options.sink];
    ctx.setAnnotations(
      'aviation-flow-endpoints',
      terminalIatas.map((iata, index) => {
        const airport = network.airportIds[iataIndex.get(iata) ?? 0];
        const role = balanced ? (index < 2 ? 'Supply' : 'Demand') : index === 0 ? 'Source' : 'Sink';
        return {
          kind: 'point',
          coordinate: [locations[airport * 2], locations[airport * 2 + 1]],
          text: `${role} ${iata}`,
          detail: balanced ? 'Hypothetical balanced terminal' : 'Regional flow terminal',
          marker: 'none',
          rank: 'subject',
          priority: 20 - index,
          id: `aviation-flow-terminal-${iata}`
        };
      })
    );
    ctx.requestLayers();
  }
  function updateParameters(): void {
    const capacityFactor = Number(ctx.options.capacityShare) / 100;
    const capacities = Uint32Array.from(network.scheduledFlights, capacity =>
      Math.floor(capacity * capacityFactor)
    );
    const effectiveCosts =
      ctx.options.terminalMode === 'balanced4' || ctx.options.objective === 'fixedBudget'
        ? network.delayCosts
        : new Uint32Array(edgeCount);
    const solverEdges = new Uint32Array(solverEdgeCount * 2);
    solverEdges.set(edges.subarray(0, edgeCount * 2));
    const solverCapacities = new Uint32Array(solverEdgeCount);
    solverCapacities.set(capacities);
    const solverCosts = new Uint32Array(solverEdgeCount);
    solverCosts.set(effectiveCosts);
    const source = iataIndex.get(ctx.options.source) ?? 0;
    const sink = iataIndex.get(ctx.options.sink) ?? 1;
    let flowRequest: Uint32Array;
    let terminalIatas: string[];
    if (ctx.options.terminalMode === 'balanced4') {
      const demands = new Int32Array(nodeCount);
      const halfBudget = Math.floor(ctx.options.flowBudget / 2);
      demands[iataIndex.get('SEA') ?? 0] -= halfBudget;
      demands[iataIndex.get('PDX') ?? 0] -= halfBudget;
      demands[iataIndex.get('LAX') ?? 0] += halfBudget;
      demands[iataIndex.get('SAN') ?? 0] += halfBudget;
      const prepared = prepareGPUNetworkFlowDemands({
        nodeCount,
        edges: Array.from(
          {length: edgeCount},
          (_, edge) => [network.sources[edge], network.targets[edge]] as const
        ),
        capacities,
        demands,
        costs: effectiveCosts
      });
      solverEdges.set(prepared.edges);
      solverCapacities.set(prepared.capacities);
      solverCosts.set(prepared.costs!);
      flowRequest = prepared.parameters;
      terminalIatas = ['SEA', 'PDX', 'LAX', 'SAN'];
    } else {
      solverEdges.set([SOURCE_SUPER_NODE, source, sink, SINK_SUPER_NODE], edgeCount * 2);
      solverCapacities[edgeCount] = SUPER_TERMINAL_BUDGET;
      solverCapacities[edgeCount + 1] = SUPER_TERMINAL_BUDGET;
      flowRequest = Uint32Array.of(
        SOURCE_SUPER_NODE,
        SINK_SUPER_NODE,
        ctx.options.objective === 'maximum' ? MAXIMUM_FLOW_SENTINEL : ctx.options.flowBudget
      );
      terminalIatas = [ctx.options.source, ctx.options.sink];
    }
    edgeBuffer.write(solverEdges);
    capacityParameters.write(solverCapacities);
    delayCostParameters.write(solverCosts);
    const endpointCoordinates = terminalIatas.flatMap(iata => {
      const index = iataIndex.get(iata) ?? 0;
      return [network.positions[index * 2], network.positions[index * 2 + 1]];
    });
    while (endpointCoordinates.length < 8) endpointCoordinates.push(0, 0);
    endpointPositionsBuffer.write(Float32Array.from(endpointCoordinates));
    endpointRolesBuffer.write(
      ctx.options.terminalMode === 'balanced4'
        ? Uint32Array.of(1, 1, 2, 2)
        : Uint32Array.of(1, 2, 1, 2)
    );
    flowParameters.write(flowRequest);
    generation++;
    dirty = true;
    lastResult = null;
    ctx.setReadout('flowStatus', 'Waiting for GPU flow solution');
    if (ctx.options.terminalMode === 'singlePair' && source === sink)
      ctx.setReadout('flowStatus', 'Choose different source and sink airports');
    ctx.setReadout('flowValue', '—');
    ctx.setReadout('flowCut', '—');
    ctx.setReadout('delayScore', '—');
    ctx.setReadout('augmentations', '—');
    ctx.setChart('flowLinks', null);
    ctx.setAnnotations('aviation-flow-endpoints', null);
    ctx.requestLayers();
  }
  updateParameters();
  ctx.setReadout(
    'networkShape',
    `${nodeCount} western airports · ${edgeCount} observed directed links`
  );
  ctx.setReadout(
    'regionalCapacity',
    `${network.scheduledFlights.reduce((sum, count) => sum + count, 0).toLocaleString()} July scheduled-count values across retained links`
  );
  const projection = createAnalysisFrameProjection({
    kind: 'albers-equal-area',
    origin: [-114, 38],
    standardParallels: [29.5, 45.5]
  });
  return {
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
    setOption() {
      updateParameters();
    },
    getLayers() {
      const dark = ctx.ground() === 'dark';
      const outputReady = Boolean(
        lastResult &&
          lastResult.summary[0] === 1 &&
          lastResult.summary[1] === 1 &&
          !(ctx.options.terminalMode === 'singlePair' && ctx.options.source === ctx.options.sink)
      );
      const cutReady = Boolean(outputReady && lastResult?.summary[7] === 1);
      return [
        new SpatialAnalysisSegmentLayer({
          id: 'aviation-flow-background-links',
          analysisFrame: projection.frame,
          segments: networkSegmentsBuffer,
          instanceCount: edgeCount,
          widthPixels: 1.1,
          color: dark ? [114, 148, 176, 42] : [70, 92, 112, 45]
        }),
        ...(outputReady
          ? [
              new SpatialAnalysisSegmentLayer({
                id: 'aviation-flow-allocated-links',
                analysisFrame: projection.frame,
                segments: networkSegmentsBuffer,
                instanceCount: edgeCount,
                weights: flowShareBuffer,
                widthPixels: 3.5,
                color: [252, 184, 66, 230],
                outlineColor: [34, 40, 46, 210],
                outlineWidthPixels: 0.6
              })
            ]
          : []),
        ...(cutReady
          ? [
              new SpatialAnalysisPointLayer({
                id: 'aviation-flow-residual-cut',
                analysisFrame: projection.frame,
                positions: nodePositionsBuffer,
                instanceCount: nodeCount,
                values: cutBuffer,
                valueFormat: 'uint32',
                colormap: 'category',
                palette: [
                  [100, 111, 126, 215],
                  [39, 205, 185, 255]
                ],
                sizeValues: airportFlightsBuffer,
                sizeMaximumValue: Math.max(...network.airportFlights),
                radiusPixels: 8.5,
                outlineColor: dark ? [7, 13, 20, 255] : [255, 255, 255, 255],
                outlineWidthPixels: 1.1
              })
            ]
          : [
              new SpatialAnalysisPointLayer({
                id: 'aviation-flow-airport-volume',
                analysisFrame: projection.frame,
                positions: nodePositionsBuffer,
                instanceCount: nodeCount,
                values: airportFlightsBuffer,
                valueFormat: 'float32',
                colormap: 'magma',
                valueRange: [0, Math.max(...network.airportFlights)],
                radiusPixels: 8.5,
                outlineColor: dark ? [7, 13, 20, 255] : [255, 255, 255, 255],
                outlineWidthPixels: 1.1
              })
            ]),
        new SpatialAnalysisPointLayer({
          id: 'aviation-flow-endpoints',
          analysisFrame: projection.frame,
          positions: endpointPositionsBuffer,
          instanceCount: ctx.options.terminalMode === 'balanced4' ? 4 : 2,
          values: endpointRolesBuffer,
          valueFormat: 'uint32',
          colormap: 'category',
          palette: [
            [246, 104, 64, 255],
            [66, 220, 237, 255]
          ],
          radiusPixels: 13,
          outlineColor: [16, 22, 30, 255],
          outlineWidthPixels: 1.6
        })
      ] as Layer[];
    },
    getTooltip(event) {
      const viewport = ctx.getViewport();
      if (!viewport) return null;
      let nearest = -1;
      let minimumDistance = 18 * 18;
      for (let node = 0; node < nodeCount; node++) {
        const pixel = viewport.project([
          locations[network.airportIds[node] * 2],
          locations[network.airportIds[node] * 2 + 1]
        ]);
        const distance = (pixel[0] - event.pixel[0]) ** 2 + (pixel[1] - event.pixel[1]) ** 2;
        if (distance < minimumDistance) {
          nearest = node;
          minimumDistance = distance;
        }
      }
      if (nearest < 0) return null;
      const completeResult = Boolean(
        lastResult &&
          lastResult.summary[0] === 1 &&
          lastResult.summary[1] === 1 &&
          !(ctx.options.terminalMode === 'singlePair' && ctx.options.source === ctx.options.sink)
      );
      const cutSide =
        completeResult && lastResult?.summary[7] === 1
          ? lastResult.cut[nearest] === 1
            ? 'residual source side'
            : 'other side of residual cut'
          : completeResult
            ? 'no maximum-flow cut sought'
            : 'waiting for a complete GPU result';
      const availableResult = completeResult ? lastResult : null;
      const outgoingFlow = availableResult
        ? Array.from({length: edgeCount}, (_, edge) => edge)
            .filter(edge => network.sources[edge] === nearest)
            .reduce((sum, edge) => sum + availableResult.flow[edge], 0)
        : 0;
      return {
        title: `${network.airportIata[nearest]} · ${network.airportNames[nearest]}`,
        subtitle: 'Western US regional BTS network · July 2023',
        rows: [
          {
            label: 'Role',
            value:
              ctx.options.terminalMode === 'balanced4'
                ? ['SEA', 'PDX'].includes(network.airportIata[nearest])
                  ? 'hypothetical supply terminal'
                  : ['LAX', 'SAN'].includes(network.airportIata[nearest])
                    ? 'hypothetical demand terminal'
                    : 'intermediate'
                : ctx.options.source === network.airportIata[nearest]
                  ? 'source'
                  : ctx.options.sink === network.airportIata[nearest]
                    ? 'sink'
                    : 'intermediate'
          },
          {
            label: 'Source records touching airport',
            value: Math.round(network.airportFlights[nearest]).toLocaleString()
          },
          {
            label: 'Outgoing allocated flow units',
            value: availableResult ? outgoingFlow.toLocaleString() : 'waiting'
          },
          {label: 'Residual minimum-cut side', value: cutSide}
        ],
        note: 'Link budgets are illustrative scheduled-count values, including cancelled flights; they are not realized capacity. Flow units are abstract and do not represent passengers, compatible tickets, or schedule-feasible multi-leg itineraries.'
      };
    },
    onThemeChange() {
      ctx.requestLayers();
    },
    destroy() {
      destroyed = true;
      reader.stop();
      resources.destroy();
      ctx.setAnnotations('aviation-flow-endpoints', null);
    }
  };
}

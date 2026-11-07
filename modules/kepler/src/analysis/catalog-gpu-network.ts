// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {KeplerAllocationError} from '../runtime/dataset-registry';
import {
  GPUNetworkIsochrones,
  GPUNetworkNoding,
  GPUNetworkReachability,
  GPU_NETWORK_REACHABILITY_NONE,
  getGPUNetworkIsochronesParameterValues
} from '@luma-contributors/graph/gpu-network';
import {
  addCatalogDiagnostic,
  appendCatalogRow,
  getFeatureIdentity,
  getLineParts,
  getPointPositions,
  makeCatalogGeometryText,
  makeCatalogTable,
  type CatalogCoordinateSystem,
  type CatalogDiagnostic,
  type CatalogFeature,
  type CatalogFeatureCollection,
  type CatalogOutputTable,
  type CatalogPosition
} from './catalog-geojson';
import {getGreatCircleDistanceMeters} from './catalog-geodesy';
import type {
  CatalogGpuContext,
  CatalogRunOutput,
  CatalogParameterValue
} from './catalog-binding-types';
import {
  buildNetworkGraph,
  makeOriginLegs,
  runNetworkBuild,
  runNetworkServiceArea,
  runNetworkShortestPaths,
  searchNetwork,
  type CatalogNetworkParameters,
  type CatalogRoutingParameters,
  type NetworkGraph,
  type NetworkSearch,
  type SnappedPoint
} from './catalog-network';
import {CatalogGpuUnsupportedError, withCatalogGpuSession} from './catalog-gpu-session';
import {snapCatalogPoints} from './catalog-gpu-snapping';
import {createCatalogFrame, packCatalogLines, type CatalogFrame} from './catalog-gpu-pack';

/** CPU-side directed CSR for the reachability contributor. */
export type CatalogDirectedCsr = {
  offsets: Uint32Array;
  neighbors: Uint32Array;
  weights: Float32Array;
};

export type CatalogArc = {from: number; to: number; cost: number};

/** Builds offsets, neighbors and weights from directed arcs, keeping arc order within a row. */
export function buildCatalogDirectedCsr(
  nodeCount: number,
  arcs: readonly CatalogArc[]
): CatalogDirectedCsr {
  const offsets = new Uint32Array(nodeCount + 1);
  for (const arc of arcs) offsets[arc.from + 1]++;
  for (let node = 0; node < nodeCount; node++) offsets[node + 1] += offsets[node];
  const cursor = offsets.slice(0, nodeCount);
  const neighbors = new Uint32Array(arcs.length);
  const weights = new Float32Array(arcs.length);
  for (const arc of arcs) {
    const slot = cursor[arc.from]++;
    neighbors[slot] = arc.to;
    weights[slot] = arc.cost;
  }
  return {offsets, neighbors, weights};
}

export function getNetworkGraphArcs(graph: NetworkGraph): CatalogArc[] {
  const arcs: CatalogArc[] = [];
  graph.outgoing.forEach((row, from) => {
    for (const arc of row) arcs.push({from, to: arc.to, cost: arc.cost});
  });
  return arcs;
}

/**
 * Turns reachability output into the search structure the CPU assembly consumes. Parent edges are
 * recovered from the predecessor as the cheapest arc between the two nodes; a node with a finite
 * cost and no predecessor is a seed.
 */
export function makeNetworkSearchFromCosts(
  graph: NetworkGraph,
  costs: ArrayLike<number>,
  predecessors: ArrayLike<number> | undefined,
  seeds: readonly {node: number; cost: number}[]
): NetworkSearch {
  const count = graph.nodes.length;
  const search: NetworkSearch = {
    distance: new Float64Array(count).fill(Infinity),
    parentNode: new Int32Array(count).fill(-1),
    parentEdge: new Int32Array(count).fill(-1),
    parentForward: new Uint8Array(count),
    seed: new Int32Array(count).fill(-1)
  };
  for (let node = 0; node < count; node++) {
    const cost = costs[node];
    search.distance[node] = Number.isFinite(cost) && cost < 3e38 ? cost : Infinity;
  }
  if (!predecessors) return search;
  for (let node = 0; node < count; node++) {
    if (!Number.isFinite(search.distance[node])) continue;
    const predecessor = predecessors[node];
    if (predecessor === GPU_NETWORK_REACHABILITY_NONE || predecessor >= count) {
      let best = -1;
      seeds.forEach((seed, seedIndex) => {
        if (seed.node === node && (best < 0 || seed.cost < seeds[best].cost)) best = seedIndex;
      });
      search.seed[node] = best;
      continue;
    }
    let chosen: {edge: number; forward: boolean; cost: number} | null = null;
    for (const arc of graph.outgoing[predecessor])
      if (arc.to === node && (!chosen || arc.cost < chosen.cost))
        chosen = {edge: arc.edge, forward: arc.forward, cost: arc.cost};
    if (chosen) {
      search.parentNode[node] = predecessor;
      search.parentEdge[node] = chosen.edge;
      search.parentForward[node] = chosen.forward ? 1 : 0;
    }
  }
  return search;
}

const MAXIMUM_REACHABILITY_ROUNDS = 1024;

type ReachabilityResult = {
  /** Costs per lane, `laneCount * nodeCount` rows. */
  costs: Float32Array;
  predecessors?: Uint32Array;
};

/**
 * Runs GPUNetworkReachability over a directed CSR from seed legs. With `laneCount > 1` each seed
 * group is an independent search sharing the CSR. Rounds double until the fixpoint is reached.
 */
export async function runGpuReachability(
  context: CatalogGpuContext,
  csr: CatalogDirectedCsr,
  nodeCount: number,
  seedGroups: readonly (readonly {node: number; cost: number}[])[],
  withPredecessors: boolean
): Promise<ReachabilityResult> {
  const laneCount = seedGroups.length;
  const startNodes: number[] = [];
  const startCosts: number[] = [];
  seedGroups.forEach((group, lane) => {
    for (const seed of group) {
      startNodes.push(lane * nodeCount + seed.node);
      startCosts.push(seed.cost);
    }
  });
  if (!startNodes.length) return {costs: new Float32Array(laneCount * nodeCount).fill(Infinity)};
  for (let rounds = 64; rounds <= MAXIMUM_REACHABILITY_ROUNDS; rounds *= 2) {
    const result = await withCatalogGpuSession(
      context.device,
      'catalog-reachability',
      async session => {
        const costs = session.output('costs', 'float32', laneCount * nodeCount);
        const converged = session.output('converged', 'uint32', 1);
        const predecessors = withPredecessors
          ? session.output('predecessors', 'uint32', nodeCount)
          : undefined;
        session.graph.add(
          new GPUNetworkReachability({
            topology: {
              offsets: session.input('offsets', csr.offsets, 'uint32'),
              neighbors: session.input('neighbors', csr.neighbors, 'uint32'),
              weights: session.input('weights', csr.weights, 'float32')
            },
            startNodes: session.input('start-nodes', Uint32Array.from(startNodes), 'uint32'),
            startCosts: session.input('start-costs', Float32Array.from(startCosts), 'float32'),
            maximumIterations: rounds,
            laneCount,
            output: {
              costs: costs.view,
              ...(predecessors ? {predecessors: predecessors.view} : {}),
              status: {converged: converged.view}
            }
          })
        );
        session.run();
        return {
          converged: (await session.readScalar(converged)) !== 0,
          costs: await session.readFloat32(costs),
          predecessors: predecessors ? await session.readUint32(predecessors) : undefined
        };
      }
    );
    if (result.converged) return {costs: result.costs, predecessors: result.predecessors};
  }
  throw new CatalogGpuUnsupportedError(
    'GPUNetworkReachability did not converge within 1024 rounds'
  );
}

// ---------------------------------------------------------------------------------------------
// Crossing noding
// ---------------------------------------------------------------------------------------------

export type CatalogNodedNetwork = {
  /** One feature per noded piece, in the source coordinate system. */
  pieces: CatalogFeatureCollection;
  /** Source feature index of every piece. */
  pieceSourceIndices: number[];
  /** Source part index of every piece. */
  piecePartIndices: number[];
  /** GPU node ids at the ends of every piece and the piece length in frame units. */
  fromNodes: Uint32Array;
  toNodes: Uint32Array;
  lengths: Float32Array;
  nodes: CatalogPosition[];
  frame: CatalogFrame;
};

/**
 * GPUNetworkNoding cuts lines at crossings, touches and overlaps and snaps piece ends within the
 * tolerance. Positions are projected into a local frame in meters (or used as is when planar).
 */
export async function nodeCatalogLines(
  context: CatalogGpuContext,
  lines: CatalogFeatureCollection,
  parameters: {coordinateSystem: CatalogCoordinateSystem; snapTolerance: number}
): Promise<CatalogNodedNetwork> {
  const frame = createCatalogFrame(parameters.coordinateSystem, [lines]);
  const packed = packCatalogLines(lines.features, {project: frame.project});
  const lineCount = packed.lineSourceIndices.length;
  if (!lineCount) throw new Error('Input required');
  let intersectionCapacity = Math.max(256, 2 * packed.vertexCount);
  for (let attempt = 0; attempt < 4; attempt++) {
    const pieceCapacity = lineCount + 4 * intersectionCapacity;
    const vertexCapacity = packed.vertexCount + 8 * intersectionCapacity + 2 * lineCount;
    const nodeCapacity = 2 * pieceCapacity;
    const result = await withCatalogGpuSession(context.device, 'catalog-noding', async session => {
      const tolerance = session.input(
        'tolerance',
        Float32Array.of(parameters.snapTolerance),
        'float32'
      );
      const pieceSources = session.output('piece-sources', 'uint32', pieceCapacity);
      const pieceOffsets = session.output('piece-offsets', 'uint32', pieceCapacity + 1);
      const piecePositions = session.output('piece-positions', 'float32x2', vertexCapacity);
      const pieceCount = session.output('piece-count', 'uint32', 1);
      const pieceOverflow = session.output('piece-overflow', 'uint32', 1);
      const nodePositions = session.output('node-positions', 'float32x2', nodeCapacity);
      const nodeCount = session.output('node-count', 'uint32', 1);
      const nodeRequired = session.output('node-required', 'uint32', 1);
      const fromNodes = session.output('from-nodes', 'uint32', pieceCapacity);
      const toNodes = session.output('to-nodes', 'uint32', pieceCapacity);
      const lengths = session.output('lengths', 'float32', pieceCapacity);
      const csrOffsets = session.output('csr-offsets', 'uint32', nodeCapacity + 1);
      const csrNeighbors = session.output('csr-neighbors', 'uint32', 2 * pieceCapacity);
      const csrWeights = session.output('csr-weights', 'float32', 2 * pieceCapacity);
      const overflow = session.output('overflow', 'uint32', 1);
      session.graph.add(
        new GPUNetworkNoding({
          lines: {
            kind: 'lines',
            positions: session.input('positions', packed.positions, 'float32x2'),
            lineOffsets: session.input('line-offsets', packed.lineOffsets, 'uint32')
          },
          intersectionCapacity,
          tolerance,
          output: {
            pieces: {
              geometry: {
                kind: 'lines',
                positions: piecePositions.view,
                lineOffsets: pieceOffsets.view
              },
              sourceIds: pieceSources.view,
              status: {count: pieceCount.view, overflow: pieceOverflow.view}
            },
            nodes: {
              positions: nodePositions.view,
              status: {count: nodeCount.view, requiredCount: nodeRequired.view}
            },
            edges: {fromNodes: fromNodes.view, toNodes: toNodes.view, lengths: lengths.view},
            topology: {
              offsets: csrOffsets.view,
              neighbors: csrNeighbors.view,
              weights: csrWeights.view
            },
            status: {overflow: overflow.view}
          }
        })
      );
      session.run();
      const overflowed =
        (await session.readScalar(overflow)) !== 0 ||
        (await session.readScalar(pieceOverflow)) !== 0;
      if (overflowed) return null;
      const count = await session.readScalar(pieceCount);
      const nodes = await session.readScalar(nodeCount);
      const offsets = await session.readUint32(pieceOffsets);
      const positions = await session.readFloat32(piecePositions);
      const sources = await session.readUint32(pieceSources);
      const nodeXY = await session.readFloat32(nodePositions);
      return {
        count,
        nodes,
        offsets,
        positions,
        sources,
        nodeXY,
        from: (await session.readUint32(fromNodes)).slice(0, count),
        to: (await session.readUint32(toNodes)).slice(0, count),
        lengths: (await session.readFloat32(lengths)).slice(0, count)
      };
    });
    if (!result) {
      intersectionCapacity *= 4;
      continue;
    }
    const pieces: CatalogFeature[] = [];
    const pieceSourceIndices: number[] = [];
    const piecePartIndices: number[] = [];
    for (let piece = 0; piece < result.count; piece++) {
      const coordinates: CatalogPosition[] = [];
      for (let vertex = result.offsets[piece]; vertex < result.offsets[piece + 1]; vertex++)
        coordinates.push(
          frame.unproject(result.positions[2 * vertex], result.positions[2 * vertex + 1])
        );
      const line = result.sources[piece];
      const sourceIndex = packed.lineSourceIndices[line];
      const source = lines.features[sourceIndex];
      pieceSourceIndices.push(sourceIndex);
      piecePartIndices.push(packed.linePartIndices[line]);
      pieces.push({
        id: source.id ?? null,
        geometry: {type: 'LineString', coordinates},
        properties: {...(source.properties ?? {})}
      });
    }
    return {
      pieces: {type: 'FeatureCollection', features: pieces},
      pieceSourceIndices,
      piecePartIndices,
      fromNodes: result.from,
      toNodes: result.to,
      lengths: result.lengths,
      nodes: Array.from({length: result.nodes}, (_, node) =>
        frame.unproject(result.nodeXY[2 * node], result.nodeXY[2 * node + 1])
      ),
      frame
    };
  }
  throw new CatalogGpuUnsupportedError('GPUNetworkNoding overflowed its capacities');
}

/**
 * Prorates property costs over noded pieces: the CPU graph prorates by length share inside one
 * part, but a noded piece is its own part, so each piece receives its share of the source value.
 */
export function prorateNodedProperties(
  noded: CatalogNodedNetwork,
  source: CatalogFeatureCollection,
  parameters: Pick<
    CatalogNetworkParameters,
    'costMode' | 'forwardCostProperty' | 'reverseCostProperty'
  >,
  measure: (coordinates: readonly CatalogPosition[]) => number
): void {
  if (parameters.costMode !== 'property') return;
  const sourceLengths = new Map<string, number>();
  const partLengths = (sourceIndex: number, partIndex: number) =>
    measure(getLineParts(source.features[sourceIndex].geometry)[partIndex] ?? []);
  noded.pieces.features.forEach((piece, index) => {
    const key = `${noded.pieceSourceIndices[index]}:${noded.piecePartIndices[index]}`;
    if (!sourceLengths.has(key))
      sourceLengths.set(
        key,
        partLengths(noded.pieceSourceIndices[index], noded.piecePartIndices[index])
      );
    const whole = sourceLengths.get(key) as number;
    const coordinates = (piece.geometry as {coordinates: CatalogPosition[]}).coordinates;
    const share = whole > 0 ? measure(coordinates) / whole : 1;
    const properties = piece.properties as Record<string, unknown>;
    for (const name of [parameters.forwardCostProperty, parameters.reverseCostProperty])
      if (name && name in properties) properties[name] = Number(properties[name]) * share;
  });
}

/** Rewrites piece-based source indices of CPU tables back to the original feature indices. */
export function remapNodedSourceIndices(
  tables: readonly CatalogOutputTable[],
  pieceSourceIndices: readonly number[]
): void {
  for (const table of tables) {
    const indices = table.columns['source_index'] as (number | null)[] | undefined;
    if (indices)
      indices.forEach((value, row) => {
        if (value !== null) indices[row] = pieceSourceIndices[value];
      });
    const lists = table.columns['source_indices'] as (string | null)[] | undefined;
    if (lists)
      lists.forEach((value, row) => {
        if (value)
          lists[row] = value
            .split(',')
            .map(item => pieceSourceIndices[Number(item)])
            .join(',');
      });
  }
}

function getMeasureLine(coordinateSystem: CatalogCoordinateSystem) {
  return (coordinates: readonly CatalogPosition[]): number => {
    let length = 0;
    for (let index = 1; index < coordinates.length; index++) {
      const [ax, ay] = coordinates[index - 1];
      const [bx, by] = coordinates[index];
      if (coordinateSystem === 'planar') length += Math.hypot(bx - ax, by - ay);
      else {
        length += getGreatCircleDistanceMeters(ax, ay, bx, by);
      }
    }
    return length;
  };
}

/** Standalone crossing noding: noded edges and nodes as tables. */
export async function runGpuNetworkNoding(
  context: CatalogGpuContext,
  lines: CatalogFeatureCollection,
  parameters: {coordinateSystem: CatalogCoordinateSystem; snapTolerance: number}
): Promise<CatalogRunOutput> {
  const noded = await nodeCatalogLines(context, lines, parameters);
  const edges = makeCatalogTable(
    'noded-edges',
    'Noded edges',
    {
      piece_id: 'string',
      source_index: 'int32',
      source_id: 'string',
      part_index: 'int32',
      from_node: 'int32',
      to_node: 'int32',
      length: 'float64',
      geometry: 'string'
    },
    'geometry'
  );
  noded.pieces.features.forEach((piece, index) =>
    appendCatalogRow(edges, {
      piece_id: String(index),
      source_index: noded.pieceSourceIndices[index],
      source_id: getFeatureIdentity(piece),
      part_index: noded.piecePartIndices[index],
      from_node: noded.fromNodes[index],
      to_node: noded.toNodes[index],
      length: noded.lengths[index],
      geometry: JSON.stringify(piece.geometry)
    })
  );
  const degrees = new Uint32Array(noded.nodes.length);
  for (let index = 0; index < noded.fromNodes.length; index++) {
    degrees[noded.fromNodes[index]]++;
    degrees[noded.toNodes[index]]++;
  }
  const nodes = makeCatalogTable(
    'noded-nodes',
    'Noded nodes',
    {node_id: 'int32', degree: 'int32', geometry: 'string'},
    'geometry'
  );
  noded.nodes.forEach((position, node) =>
    appendCatalogRow(nodes, {
      node_id: node,
      degree: degrees[node],
      geometry: makeCatalogGeometryText('Point', position)
    })
  );
  const inputLineCount = lines.features.reduce(
    (count, feature) => count + getLineParts(feature.geometry).length,
    0
  );
  const diagnostics: CatalogDiagnostic[] = [];
  addCatalogDiagnostic(
    diagnostics,
    'noded-lengths',
    'info',
    noded.frame.kind === 'identity'
      ? 'Lengths are in coordinate units.'
      : 'Lengths are meters in a local azimuthal-equidistant frame around the input center.'
  );
  addCatalogDiagnostic(
    diagnostics,
    'bridges-not-modeled',
    'info',
    'Lines that cross without a shared vertex are connected; bridges and tunnels are not modeled.'
  );
  return {
    tables: [edges, nodes],
    diagnostics,
    summary: {
      inputLineCount,
      edgeCount: edges.rowCount,
      nodeCount: nodes.rowCount,
      splitCount: edges.rowCount - inputLineCount
    },
    dispatchedContributors: ['GPUNetworkNoding']
  };
}

// ---------------------------------------------------------------------------------------------
// Graph-based operations
// ---------------------------------------------------------------------------------------------

export type CatalogGpuNetworkParameters = CatalogNetworkParameters & {
  /** `noded` nodes crossings with GPUNetworkNoding before the graph is built. */
  crossings: 'not-noded' | 'noded';
};

export function readCatalogCrossings(
  values: Readonly<Record<string, CatalogParameterValue>>
): 'not-noded' | 'noded' {
  return values['crossings'] === 'noded' ? 'noded' : 'not-noded';
}

type PreparedNetwork = {
  lines: CatalogFeatureCollection;
  graph: NetworkGraph;
  noded?: CatalogNodedNetwork;
};

async function prepareNetwork(
  context: CatalogGpuContext,
  lines: CatalogFeatureCollection,
  parameters: CatalogGpuNetworkParameters,
  maximumEdges: number
): Promise<PreparedNetwork> {
  if (parameters.crossings !== 'noded') {
    return {lines, graph: buildNetworkGraph(lines, parameters, maximumEdges)};
  }
  const noded = await nodeCatalogLines(context, lines, parameters);
  prorateNodedProperties(noded, lines, parameters, getMeasureLine(parameters.coordinateSystem));
  return {
    lines: noded.pieces,
    noded,
    graph: buildNetworkGraph(
      noded.pieces,
      {...parameters, splitAtSharedVertices: true},
      maximumEdges
    )
  };
}

function finishNetworkRun(
  output: {
    tables: CatalogOutputTable[];
    diagnostics: CatalogDiagnostic[];
    summary: Record<string, string | number>;
  },
  prepared: PreparedNetwork,
  contributors: readonly string[]
): CatalogRunOutput {
  if (prepared.noded) {
    remapNodedSourceIndices(output.tables, prepared.noded.pieceSourceIndices);
    addCatalogDiagnostic(
      output.diagnostics,
      'crossings-noded',
      'info',
      'Lines were cut at crossings by GPUNetworkNoding; edge ids use noded piece numbers and source_index refers to the input feature.'
    );
    const index = output.diagnostics.findIndex(
      diagnostic => diagnostic.code === 'crossings-not-noded'
    );
    if (index >= 0) output.diagnostics.splice(index, 1);
  }
  return {...output, dispatchedContributors: contributors};
}

function getContributors(prepared: PreparedNetwork, ...names: string[]): string[] {
  return prepared.noded ? ['GPUNetworkNoding', ...names] : names;
}

/** Network build: crossing noding on the GPU, then the CPU graph assembly. */
export async function runGpuNetworkBuild(
  context: CatalogGpuContext,
  lines: CatalogFeatureCollection,
  parameters: CatalogGpuNetworkParameters,
  maximumEdges: number
): Promise<CatalogRunOutput> {
  if (parameters.crossings !== 'noded')
    throw new CatalogGpuUnsupportedError(
      'GPUNetworkNoding always nodes crossings; set crossings to noded for the GPU path'
    );
  const prepared = await prepareNetwork(context, lines, parameters, maximumEdges);
  const output = runNetworkBuild(
    prepared.lines,
    {...parameters, splitAtSharedVertices: true},
    maximumEdges
  );
  return finishNetworkRun(output, prepared, getContributors(prepared));
}

/**
 * Snaps the first point of every origin collection in one GPUNetworkSnapping dispatch (CPU snap
 * when no device result is available) and turns each snapped point into search seeds.
 */
async function getSeedsOfOrigins(
  context: CatalogGpuContext,
  graph: NetworkGraph,
  origins: readonly CatalogFeatureCollection[],
  maximumSnapDistance: number
) {
  const positions = origins.map(
    origin => getPointPositions(origin.features[0]?.geometry ?? null)[0] ?? null
  );
  const present = positions.flatMap((position, index) => (position ? [index] : []));
  const result = await snapCatalogPoints(
    context,
    graph,
    present.map(index => positions[index] as CatalogPosition),
    maximumSnapDistance
  );
  const entries = origins.map(() => ({
    snapped: null as SnappedPoint | null,
    seeds: [] as {node: number; cost: number}[]
  }));
  present.forEach((originIndex, row) => {
    const snapped = result.snapped[row];
    entries[originIndex].snapped = snapped;
    const legs = snapped ? makeOriginLegs(graph, snapped) : [];
    entries[originIndex].seeds = legs.map(leg => ({node: leg.node, cost: leg.cost}));
  });
  return {entries, snapContributors: result.engine === 'gpu' ? ['GPUNetworkSnapping'] : []};
}

async function getSeedsOfOrigin(
  context: CatalogGpuContext,
  graph: NetworkGraph,
  origin: CatalogFeatureCollection,
  maximumSnapDistance: number
) {
  const {entries, snapContributors} = await getSeedsOfOrigins(
    context,
    graph,
    [origin],
    maximumSnapDistance
  );
  return {...entries[0], snapContributors};
}

export async function runGpuNetworkServiceArea(
  context: CatalogGpuContext,
  lines: CatalogFeatureCollection,
  origins: CatalogFeatureCollection,
  parameters: CatalogGpuNetworkParameters &
    CatalogRoutingParameters & {budgets: readonly number[]; includeUnreachable: boolean},
  limits: {maximumEdges: number}
): Promise<CatalogRunOutput> {
  const prepared = await prepareNetwork(context, lines, parameters, limits.maximumEdges);
  const {graph} = prepared;
  const {seeds, snapContributors} = await getSeedsOfOrigin(
    context,
    graph,
    origins,
    parameters.maximumSnapDistance
  );
  let search: NetworkSearch | undefined;
  const arcs = getNetworkGraphArcs(graph);
  if (seeds.length && arcs.length) {
    const result = await runGpuReachability(
      context,
      buildCatalogDirectedCsr(graph.nodes.length, arcs),
      graph.nodes.length,
      [seeds],
      true
    );
    search = makeNetworkSearchFromCosts(graph, result.costs, result.predecessors, seeds);
  }
  const output = runNetworkServiceArea(prepared.lines, origins, parameters, limits, {
    graph,
    search: search ? () => search as NetworkSearch : searchNetwork
  });
  addCatalogDiagnostic(
    output.diagnostics,
    'gpu-search-precision',
    'info',
    'Least costs come from float32 GPUNetworkReachability; edge geometry and bands are assembled on the CPU.'
  );
  return finishNetworkRun(
    output,
    prepared,
    getContributors(prepared, ...snapContributors, ...(search ? ['GPUNetworkReachability'] : []))
  );
}

export async function runGpuNetworkShortestPaths(
  context: CatalogGpuContext,
  lines: CatalogFeatureCollection,
  origin: CatalogFeatureCollection,
  destinations: CatalogFeatureCollection,
  parameters: CatalogGpuNetworkParameters & CatalogRoutingParameters,
  limits: {maximumEdges: number; maximumDestinations: number}
): Promise<CatalogRunOutput> {
  const prepared = await prepareNetwork(context, lines, parameters, limits.maximumEdges);
  const {graph} = prepared;
  const originSeeds = await getSeedsOfOrigin(
    context,
    graph,
    origin,
    parameters.maximumSnapDistance
  );
  const {seeds, snapped: snappedOrigin} = originSeeds;
  let snapContributors = originSeeds.snapContributors;
  let search: NetworkSearch | undefined;
  const arcs = getNetworkGraphArcs(graph);
  if (seeds.length && arcs.length) {
    const result = await runGpuReachability(
      context,
      buildCatalogDirectedCsr(graph.nodes.length, arcs),
      graph.nodes.length,
      [seeds],
      true
    );
    search = makeNetworkSearchFromCosts(graph, result.costs, result.predecessors, seeds);
  }
  // Destinations snap in one GPUNetworkSnapping dispatch, like the origin, instead of per point on the CPU.
  const destinationPositions = destinations.features.flatMap(feature =>
    getPointPositions(feature.geometry)
  );
  // Destination capacity exceeded; the CPU run below reports it by throwing.
  const destinationSnap =
    destinationPositions.length && destinationPositions.length <= limits.maximumDestinations
      ? await snapCatalogPoints(
          context,
          graph,
          destinationPositions,
          parameters.maximumSnapDistance
        )
      : undefined;
  const output = runNetworkShortestPaths(prepared.lines, origin, destinations, parameters, limits, {
    graph,
    search: search ? () => search as NetworkSearch : searchNetwork,
    snappedOrigin,
    ...(destinationSnap ? {snappedDestinations: destinationSnap.snapped} : {})
  });
  if (destinationSnap) output.summary['destination_snap_engine'] = destinationSnap.engine;
  if (destinationSnap?.engine === 'gpu')
    snapContributors = snapContributors.includes('GPUNetworkSnapping')
      ? snapContributors
      : [...snapContributors, 'GPUNetworkSnapping'];
  else if (destinationSnap?.fallbackReason)
    addCatalogDiagnostic(
      output.diagnostics,
      'destination-snap-cpu',
      'info',
      `Destinations were snapped on the CPU: ${destinationSnap.fallbackReason}.`
    );
  addCatalogDiagnostic(
    output.diagnostics,
    'gpu-search-precision',
    'info',
    'Least costs and predecessors come from float32 GPUNetworkReachability; equal-cost ties may choose a different path than the CPU.'
  );
  return finishNetworkRun(
    output,
    prepared,
    getContributors(prepared, ...snapContributors, ...(search ? ['GPUNetworkReachability'] : []))
  );
}

export const MAXIMUM_MULTI_ORIGIN_COUNT = 32;

/** Service areas for several origins in one GPU dispatch, one search lane per origin. */
export async function runGpuNetworkMultiOrigin(
  context: CatalogGpuContext,
  lines: CatalogFeatureCollection,
  origins: CatalogFeatureCollection,
  parameters: CatalogGpuNetworkParameters &
    CatalogRoutingParameters & {budgets: readonly number[]; includeUnreachable: boolean},
  limits: {maximumEdges: number}
): Promise<CatalogRunOutput> {
  const prepared = await prepareNetwork(context, lines, parameters, limits.maximumEdges);
  const {graph} = prepared;
  const originFeatures = origins.features.filter(
    feature => getPointPositions(feature.geometry).length
  );
  // Origin capacity exceeded; one search lane is allocated per origin.
  if (originFeatures.length > MAXIMUM_MULTI_ORIGIN_COUNT)
    throw new KeplerAllocationError('capacity', {message: 'Origin capacity exceeded'});
  const {entries: originSeeds, snapContributors} = await getSeedsOfOrigins(
    context,
    graph,
    originFeatures.map(feature => ({type: 'FeatureCollection', features: [feature]})),
    parameters.maximumSnapDistance
  );
  const arcs = getNetworkGraphArcs(graph);
  const nodeCount = graph.nodes.length;
  const lanes =
    arcs.length && originSeeds.some(entry => entry.seeds.length)
      ? await runGpuReachability(
          context,
          buildCatalogDirectedCsr(nodeCount, arcs),
          nodeCount,
          originSeeds.map(entry => entry.seeds),
          false
        )
      : undefined;
  const table = makeCatalogTable(
    'network-multi-origin',
    'Service areas by origin',
    {
      origin_index: 'int32',
      origin_source_id: 'string',
      is_nearest_origin: 'boolean',
      edge_id: 'string',
      source_index: 'int32',
      source_id: 'string',
      status: 'string',
      first_reach_cost: 'float64',
      band: 'int32',
      band_budget: 'float64',
      reached_fraction: 'float64',
      geometry: 'string'
    },
    'geometry'
  );
  const diagnostics: CatalogDiagnostic[] = [];
  const nearest = new Map<string, number>();
  const perOrigin: CatalogOutputTable[] = [];
  let unsnapped = 0;
  originFeatures.forEach((feature, originIndex) => {
    const seeds = originSeeds[originIndex].seeds;
    if (!seeds.length) {
      unsnapped++;
      return;
    }
    const laneCosts = lanes!.costs.subarray(originIndex * nodeCount, (originIndex + 1) * nodeCount);
    const search = makeNetworkSearchFromCosts(graph, laneCosts, undefined, seeds);
    const output = runNetworkServiceArea(
      prepared.lines,
      {type: 'FeatureCollection', features: [feature]},
      {...parameters, includeUnreachable: false},
      limits,
      {graph, search: () => search}
    );
    perOrigin[originIndex] = output.tables[0];
    const costs = output.tables[0].columns['first_reach_cost'];
    output.tables[0].columns['edge_id'].forEach((edgeId, row) => {
      const cost = costs[row] as number | null;
      if (cost !== null && cost < (nearest.get(edgeId as string) ?? Infinity))
        nearest.set(edgeId as string, cost);
    });
  });
  perOrigin.forEach((source, originIndex) => {
    if (!source) return;
    for (let row = 0; row < source.rowCount; row++) {
      const edgeId = source.columns['edge_id'][row] as string;
      appendCatalogRow(table, {
        origin_index: originIndex,
        origin_source_id: getFeatureIdentity(originFeatures[originIndex]),
        is_nearest_origin: source.columns['first_reach_cost'][row] === nearest.get(edgeId),
        edge_id: edgeId,
        source_index: source.columns['source_index'][row],
        source_id: source.columns['source_id'][row],
        status: source.columns['status'][row],
        first_reach_cost: source.columns['first_reach_cost'][row],
        band: source.columns['band'][row],
        band_budget: source.columns['band_budget'][row],
        reached_fraction: source.columns['reached_fraction'][row],
        geometry: source.columns['geometry'][row]
      });
    }
  });
  if (unsnapped)
    addCatalogDiagnostic(
      diagnostics,
      'origin-not-snapped',
      'warning',
      'These origins have no network edge within the maximum snap distance and produce no rows.',
      unsnapped
    );
  addCatalogDiagnostic(
    diagnostics,
    'origins-independent',
    'info',
    'Each origin is a separate search lane; is_nearest_origin marks the origin with the lowest first-reach cost per edge (ties share the flag).'
  );
  const output = {
    tables: [table],
    diagnostics,
    summary: {
      originCount: originFeatures.length,
      rowCount: table.rowCount,
      nodeCount,
      edgeCount: graph.edges.length
    }
  };
  return finishNetworkRun(
    output,
    prepared,
    getContributors(prepared, ...snapContributors, ...(lanes ? ['GPUNetworkReachability'] : []))
  );
}

// ---------------------------------------------------------------------------------------------
// Isochrone polygons
// ---------------------------------------------------------------------------------------------

export type CatalogIsochroneParameters = CatalogGpuNetworkParameters &
  CatalogRoutingParameters & {
    budgets: readonly number[];
    bufferRadius: number;
    rasterSize: number;
  };

/** Band polygons from GPUNetworkIsochrones: edge costs splatted to a raster, then contoured into rings. */
export async function runGpuNetworkIsochrones(
  context: CatalogGpuContext,
  lines: CatalogFeatureCollection,
  origins: CatalogFeatureCollection,
  parameters: CatalogIsochroneParameters,
  limits: {maximumEdges: number}
): Promise<CatalogRunOutput> {
  const prepared = await prepareNetwork(context, lines, parameters, limits.maximumEdges);
  const {graph} = prepared;
  const frame = createCatalogFrame(parameters.coordinateSystem, [lines, origins]);
  const budgets = [...parameters.budgets].sort((first, second) => first - second);
  const diagnostics: CatalogDiagnostic[] = [];
  // Facilities: every origin becomes an extra node connected to the two ends of its snapped edge.
  const nodePositions = graph.nodes.map(position => frame.project(position));
  const arcs = getNetworkGraphArcs(graph);
  const startNodes: number[] = [];
  let unsnapped = 0;
  const isochroneOrigins = await getSeedsOfOrigins(
    context,
    graph,
    origins.features.map(feature => ({type: 'FeatureCollection', features: [feature]})),
    parameters.maximumSnapDistance
  );
  for (const {snapped} of isochroneOrigins.entries) {
    if (!snapped) {
      unsnapped++;
      continue;
    }
    const edge = graph.edges[snapped.edge];
    const originNode = nodePositions.length;
    nodePositions.push(frame.project(snapped.position));
    if (edge.reverseCost !== null)
      arcs.push({from: originNode, to: edge.fromNode, cost: snapped.fraction * edge.reverseCost});
    if (edge.forwardCost !== null)
      arcs.push({
        from: originNode,
        to: edge.toNode,
        cost: (1 - snapped.fraction) * edge.forwardCost
      });
    startNodes.push(originNode);
  }
  if (unsnapped)
    addCatalogDiagnostic(
      diagnostics,
      'origin-not-snapped',
      'warning',
      'These origins have no network edge within the maximum snap distance.',
      unsnapped
    );
  const table = makeCatalogTable(
    'network-isochrones',
    'Isochrone bands',
    {
      band: 'int32',
      minimum_cost: 'float64',
      maximum_cost: 'float64',
      polygon_id: 'string',
      ring_count: 'int32',
      hole_count: 'int32',
      area: 'float64',
      geometry: 'string'
    },
    'geometry'
  );
  if (!startNodes.length || !arcs.length) {
    addCatalogDiagnostic(
      diagnostics,
      'no-isochrones',
      'error',
      'No origin snapped to a routable network edge, so no isochrone is produced.'
    );
    return {
      tables: [table],
      diagnostics,
      summary: {bandCount: 0, polygonCount: 0},
      dispatchedContributors: getContributors(prepared, ...isochroneOrigins.snapContributors)
    };
  }
  const nodeCount = nodePositions.length;
  const csr = buildCatalogDirectedCsr(nodeCount, arcs);
  const bounds = nodePositions.reduce(
    (box, [x, y]) => [
      Math.min(box[0], x),
      Math.min(box[1], y),
      Math.max(box[2], x),
      Math.max(box[3], y)
    ],
    [Infinity, Infinity, -Infinity, -Infinity]
  );
  const size = parameters.rasterSize;
  const pad =
    parameters.bufferRadius + Math.max(bounds[2] - bounds[0], bounds[3] - bounds[1], 1) / size;
  const extent: [number, number, number, number] = [
    bounds[0] - pad,
    bounds[1] - pad,
    Math.max(bounds[2] + pad, bounds[0] - pad + 1),
    Math.max(bounds[3] + pad, bounds[1] - pad + 1)
  ];
  const bandCount = budgets.length;
  const edgeCapacity = 32 * 2 * size * bandCount + 1024;
  const ringCapacity = 4096;
  const vertexCapacity = edgeCapacity + ringCapacity;
  const result = await withCatalogGpuSession(
    context.device,
    'catalog-isochrones',
    async session => {
      const costs = session.output('costs', 'float32', nodeCount);
      const converged = session.output('converged', 'uint32', 1);
      const values = session.output('values', 'float32', size * size);
      const bandClasses = session.output('band-classes', 'uint32', size * size);
      const ringOffsets = session.output('ring-offsets', 'uint32', ringCapacity + 1);
      const ringPositions = session.output('ring-positions', 'float32x2', vertexCapacity);
      const ringAreas = session.output('ring-areas', 'float32', ringCapacity);
      const ringIsHole = session.output('ring-is-hole', 'uint32', ringCapacity);
      const ringShells = session.output('ring-shells', 'uint32', ringCapacity);
      const ringGroups = session.output('ring-groups', 'uint32', ringCapacity);
      const ringCount = session.output('ring-count', 'uint32', 1);
      const ringOverflow = session.output('ring-overflow', 'uint32', 1);
      const edgeOverflow = session.output('edge-overflow', 'uint32', 1);
      const flat = new Float32Array(2 * nodeCount);
      nodePositions.forEach(([x, y], node) => {
        flat[2 * node] = x;
        flat[2 * node + 1] = y;
      });
      session.graph.add(
        new GPUNetworkIsochrones({
          topology: {
            offsets: session.input('offsets', csr.offsets, 'uint32'),
            neighbors: session.input('neighbors', csr.neighbors, 'uint32'),
            weights: session.input('weights', csr.weights, 'float32'),
            nodePositions: session.input('node-positions', flat, 'float32x2')
          },
          costs: costs.view,
          startNodes: session.input('start-nodes', Uint32Array.from(startNodes), 'uint32'),
          maximumIterations: 256,
          output: {status: {converged: converged.view}},
          costBreaks: session.input('cost-breaks', Float32Array.from(budgets), 'float32'),
          parameters: session.input(
            'parameters',
            getGPUNetworkIsochronesParameterValues({
              breakCount: bandCount,
              extent,
              bufferRadius: parameters.bufferRadius
            }),
            'float32'
          ),
          raster: {
            width: size,
            height: size,
            mode: 'min',
            output: {values: values.view, bandClasses: bandClasses.view},
            rings: {
              edgeCapacity,
              output: {
                ringOffsets: ringOffsets.view,
                positions: ringPositions.view,
                ringAreas: ringAreas.view,
                ringIsHole: ringIsHole.view,
                ringShells: ringShells.view,
                ringGroups: ringGroups.view,
                status: {
                  count: ringCount.view,
                  overflow: ringOverflow.view,
                  edgeOverflow: edgeOverflow.view
                }
              }
            }
          }
        })
      );
      session.run();
      const count = await session.readScalar(ringCount);
      return {
        converged: (await session.readScalar(converged)) !== 0,
        overflow:
          (await session.readScalar(ringOverflow)) !== 0 ||
          (await session.readScalar(edgeOverflow)) !== 0,
        count,
        offsets: await session.readUint32(ringOffsets),
        positions: await session.readFloat32(ringPositions),
        isHole: await session.readUint32(ringIsHole),
        shells: await session.readUint32(ringShells),
        groups: await session.readUint32(ringGroups)
      };
    }
  );
  if (!result.converged)
    addCatalogDiagnostic(
      diagnostics,
      'search-not-converged',
      'warning',
      'The cost search stopped at its round limit; polygons are smaller than the true ones.'
    );
  // Ring capacity or boundary edge capacity exceeded; reduce the raster size or the number of budgets.
  if (result.overflow) throw new CatalogGpuUnsupportedError('Isochrone ring capacity overflowed');
  const ringVertices = (ring: number): CatalogPosition[] => {
    const vertices: CatalogPosition[] = [];
    for (let vertex = result.offsets[ring]; vertex < result.offsets[ring + 1]; vertex++)
      vertices.push(
        frame.unproject(result.positions[2 * vertex], result.positions[2 * vertex + 1])
      );
    return vertices;
  };
  const framedArea = (ring: number): number => {
    let doubled = 0;
    const first = result.offsets[ring];
    const last = result.offsets[ring + 1];
    for (let vertex = first; vertex < last; vertex++) {
      const next = vertex + 1 < last ? vertex + 1 : first;
      doubled +=
        result.positions[2 * vertex] * result.positions[2 * next + 1] -
        result.positions[2 * next] * result.positions[2 * vertex + 1];
    }
    return Math.abs(doubled / 2);
  };
  const polygonsByShell = new Map<number, {band: number; rings: number[]}>();
  for (let ring = 0; ring < result.count; ring++)
    if (!result.isHole[ring]) polygonsByShell.set(ring, {band: result.groups[ring], rings: [ring]});
  for (let ring = 0; ring < result.count; ring++)
    if (result.isHole[ring]) polygonsByShell.get(result.shells[ring])?.rings.push(ring);
  const sorted = [...polygonsByShell.values()].sort((first, second) => first.band - second.band);
  const counters = new Map<number, number>();
  for (const polygon of sorted) {
    const index = counters.get(polygon.band) ?? 0;
    counters.set(polygon.band, index + 1);
    let area = 0;
    polygon.rings.forEach((ring, position) => {
      area += (position === 0 ? 1 : -1) * framedArea(ring);
    });
    appendCatalogRow(table, {
      band: polygon.band,
      minimum_cost: polygon.band === 0 ? 0 : budgets[polygon.band - 1],
      maximum_cost: budgets[polygon.band],
      polygon_id: `${polygon.band}:${index}`,
      ring_count: polygon.rings.length,
      hole_count: polygon.rings.length - 1,
      area,
      geometry: makeCatalogGeometryText(
        'Polygon',
        polygon.rings.map(ring => ringVertices(ring))
      )
    });
  }
  addCatalogDiagnostic(
    diagnostics,
    'isochrone-bands-exclusive',
    'info',
    'Each polygon covers costs from minimum_cost (exclusive of earlier bands) up to maximum_cost; union bands for cumulative areas.'
  );
  addCatalogDiagnostic(
    diagnostics,
    'isochrone-raster-resolution',
    'info',
    `Polygons come from a ${size} by ${size} raster, accurate to about one pixel; streets are straight between nodes and buffered by ${parameters.bufferRadius} ${frame.kind === 'identity' ? 'coordinate units' : 'meters'}.`
  );
  return {
    tables: [table],
    diagnostics,
    summary: {bandCount: counters.size, polygonCount: sorted.length, nodeCount: nodeCount},
    dispatchedContributors: getContributors(
      prepared,
      ...isochroneOrigins.snapContributors,
      'GPUNetworkReachability',
      'GPUNetworkIsochrones'
    )
  };
}

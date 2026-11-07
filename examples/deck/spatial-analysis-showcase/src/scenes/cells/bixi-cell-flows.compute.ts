// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {latLngToCell, cellToLatLng} from 'h3-js';
import {PathLayer, ScatterplotLayer} from '@deck.gl/layers';
import type {Layer} from '@deck.gl/core';
import type {Buffer} from '@luma.gl/core';
import {GPUCommandGraph, type CompiledGPUCommandGraph} from '@luma-contributors/graph/gpu-core';
import {addCellFlowTopologyRecipe} from '@luma-contributors/graph/gpu-spatial-analysis';
import {GraphSwap} from '../../engine/graph-swap';
import {importGraphBuffer} from '../../engine/graph-buffers';
import {SpatialAnalysisResources} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import type {BarChartData} from '../chart-types';
import type {SceneContext, SceneInstance, ScenePointerEvent} from '../scene';
import {readBixiFlows} from '../flows/bixi-data';
import type {BixiCellFlowsOptions} from './bixi-cell-flows.types';

type H3Resolution = 6 | 7 | 8;
const RESOLUTIONS: readonly H3Resolution[] = [6, 7, 8];
const CELL_INK: [number, number, number, number] = [250, 187, 91, 245];
const CELL_CONTEXT: [number, number, number, number] = [107, 162, 183, 175];
const STATION_INK: [number, number, number, number] = [214, 224, 221, 95];
const ENDPOINT_INK: [number, number, number, number] = [250, 187, 91, 145];
const BOUNDARY_INK: [number, number, number, number] = [113, 214, 217, 230];
const BOUNDARY_CONTEXT: [number, number, number, number] = [94, 122, 132, 155];

type CellPairTable = {
  resolution: H3Resolution;
  keys: string[];
  keyWords: Uint32Array;
  centers: [number, number][];
  boroughs: string[];
  stationCells: Uint32Array;
  origins: Uint32Array;
  destinations: Uint32Array;
  values: Float32Array;
  sourcePairCount: number;
  sourceRides: number;
  sourceMonthlyRides: number;
  sameStationRides: number;
};
type FlowBuffers = {
  cellKeys: Buffer;
  outgoing: Buffer;
  incoming: Buffer;
  directedEdges: Buffer;
  neighbors: Buffer;
  edgeFlows: Buffer;
  boundaries: Buffer;
  boundaryCounts: Buffer;
  validity: Buffer;
  invalidCount: Buffer;
};
type BixiCellGraph = {
  resources: SpatialAnalysisResources;
  compiled: CompiledGPUCommandGraph<void>;
  reader: SummaryReader;
  table: CellPairTable;
  buffers: FlowBuffers;
};
type FlowResult = {
  cellKeys: Uint32Array;
  outgoing: Float32Array;
  incoming: Float32Array;
  directedEdges: Uint32Array;
  neighbors: Uint32Array;
  edgeFlows: Float32Array;
  boundaries: Float32Array;
  boundaryCounts: Uint32Array;
  validity: Uint32Array;
  invalidCount: number;
};
type CellMark = {index: number; key: string; coordinates: [number, number]; borough: string};
type PathMark = {
  path: [number, number][];
  value: number;
  destination: string;
  color?: [number, number, number, number];
};

function keyToWords(key: string, words: Uint32Array, row: number): void {
  const value = BigInt(`0x${key}`);
  words[row * 2] = Number(value & 0xffffffffn);
  words[row * 2 + 1] = Number(value >> 32n);
}

function prepareCellPairs(
  flows: ReturnType<typeof readBixiFlows>,
  resolution: H3Resolution
): CellPairTable {
  const stationKeys = new Array<string>(flows.stationCount);
  const boroughFrequency = new Map<string, Map<number, number>>();
  for (let station = 0; station < flows.stationCount; station++) {
    const key = latLngToCell(flows.lngLat[station * 2 + 1], flows.lngLat[station * 2], resolution);
    stationKeys[station] = key;
    let counts = boroughFrequency.get(key);
    if (!counts) {
      counts = new Map();
      boroughFrequency.set(key, counts);
    }
    counts.set(flows.borough[station], (counts.get(flows.borough[station]) ?? 0) + 1);
  }
  const keys = [...boroughFrequency.keys()].sort();
  const cellRows = new Map(keys.map((key, index) => [key, index]));
  const keyWords = new Uint32Array(keys.length * 2);
  const centers: [number, number][] = [];
  const boroughs: string[] = [];
  keys.forEach((key, index) => {
    keyToWords(key, keyWords, index);
    const [latitude, longitude] = cellToLatLng(key);
    centers.push([longitude, latitude]);
    const majorBorough = [...boroughFrequency.get(key)!.entries()].sort(
      (first, second) => second[1] - first[1]
    )[0][0];
    boroughs.push(flows.boroughNames[majorBorough] ?? 'Multiple boroughs');
  });
  const stationCells = Uint32Array.from(stationKeys, key => cellRows.get(key)!);
  const aggregatePairs = new Map<string, number>();
  let sourceRides = 0;
  for (let row = 0; row < flows.pairs.count.length; row++) {
    const origin = cellRows.get(stationKeys[flows.pairs.origin[row]]);
    const destination = cellRows.get(stationKeys[flows.pairs.destination[row]]);
    if (origin === undefined || destination === undefined) continue;
    const pairKey = `${origin},${destination}`;
    aggregatePairs.set(pairKey, (aggregatePairs.get(pairKey) ?? 0) + flows.pairs.count[row]);
    sourceRides += flows.pairs.count[row];
  }
  const pairRows = [...aggregatePairs.entries()]
    .map(([key, value]) => {
      const [origin, destination] = key.split(',').map(Number);
      return {origin, destination, value};
    })
    .sort(
      (first, second) => first.origin - second.origin || first.destination - second.destination
    );
  const sourceMonthlyRides = flows.departures.reduce((sum, value) => sum + value, 0);
  return {
    resolution,
    keys,
    keyWords,
    centers,
    boroughs,
    stationCells,
    origins: Uint32Array.from(pairRows, pair => pair.origin),
    destinations: Uint32Array.from(pairRows, pair => pair.destination),
    values: Float32Array.from(pairRows, pair => pair.value),
    sourcePairCount: flows.pairs.count.length,
    sourceRides,
    sourceMonthlyRides,
    sameStationRides: flows.sameStationRides
  };
}

function makeOutputBuffers(
  resources: SpatialAnalysisResources,
  table: CellPairTable,
  pairCount: number
): FlowBuffers {
  const cellCount = table.keys.length;
  const edgeCount = cellCount * 6;
  return {
    cellKeys: resources.createBuffer('cell-keys', table.keyWords),
    outgoing: resources.createBuffer('outgoing-totals', cellCount * 4),
    incoming: resources.createBuffer('incoming-totals', cellCount * 4),
    directedEdges: resources.createBuffer('directed-edge-keys', edgeCount * 8),
    neighbors: resources.createBuffer('neighbor-cell-keys', edgeCount * 8),
    edgeFlows: resources.createBuffer('edge-flows', edgeCount * 4),
    boundaries: resources.createBuffer('gpu-edge-boundaries', edgeCount * 3 * 8),
    boundaryCounts: resources.createBuffer('boundary-point-counts', edgeCount * 4),
    validity: resources.createBuffer('pair-validity', pairCount * 4),
    invalidCount: resources.createBuffer('invalid-pair-count', 4)
  };
}

function makeSummarySources(buffers: FlowBuffers, cellCount: number, pairCount: number) {
  const edgeCount = cellCount * 6;
  return [
    {buffer: buffers.cellKeys, size: cellCount * 8},
    {buffer: buffers.outgoing, size: cellCount * 4},
    {buffer: buffers.incoming, size: cellCount * 4},
    {buffer: buffers.directedEdges, size: edgeCount * 8},
    {buffer: buffers.neighbors, size: edgeCount * 8},
    {buffer: buffers.edgeFlows, size: edgeCount * 4},
    {buffer: buffers.boundaries, size: edgeCount * 3 * 8},
    {buffer: buffers.boundaryCounts, size: edgeCount * 4},
    {buffer: buffers.validity, size: pairCount * 4},
    {buffer: buffers.invalidCount, size: 4}
  ];
}

function parseResult(bytes: ArrayBuffer, cellCount: number, pairCount: number): FlowResult {
  const view = new DataView(bytes);
  let offset = 0;
  const readFloats = (count: number) => {
    const values = new Float32Array(count);
    for (let index = 0; index < count; index++)
      values[index] = view.getFloat32(offset + index * 4, true);
    offset += count * 4;
    return values;
  };
  const readWords = (count: number) => {
    const values = new Uint32Array(count);
    for (let index = 0; index < count; index++)
      values[index] = view.getUint32(offset + index * 4, true);
    offset += count * 4;
    return values;
  };
  return {
    cellKeys: readWords(cellCount * 2),
    outgoing: readFloats(cellCount),
    incoming: readFloats(cellCount),
    directedEdges: readWords(cellCount * 6 * 2),
    neighbors: readWords(cellCount * 6 * 2),
    edgeFlows: readFloats(cellCount * 6),
    boundaries: readFloats(cellCount * 6 * 3 * 2),
    boundaryCounts: readWords(cellCount * 6),
    validity: readWords(pairCount),
    invalidCount: readWords(1)[0]
  };
}

function cellIdentity(words: Uint32Array, row: number): string {
  const value = (BigInt(words[row * 2 + 1]) << 32n) | BigInt(words[row * 2]);
  return value === 0n ? '' : value.toString(16).padStart(15, '0');
}

function sum(values: ArrayLike<number>): number {
  let total = 0;
  for (let index = 0; index < values.length; index++) total += values[index];
  return total;
}

async function buildGraph(
  ctx: SceneContext<BixiCellFlowsOptions>,
  table: CellPairTable,
  onResult: (graph: BixiCellGraph, result: FlowResult) => void
): Promise<BixiCellGraph> {
  const resources = new SpatialAnalysisResources(ctx.device, `bixi-cell-flow-r${table.resolution}`);
  const cellCount = table.keys.length;
  const pairCount = table.origins.length;
  const buffers = makeOutputBuffers(resources, table, pairCount);
  const graph = new GPUCommandGraph<void>(ctx.device, {id: `bixi-cell-flow-r${table.resolution}`});
  const cells = importGraphBuffer(
    graph,
    `r${table.resolution}-cell-keys`,
    buffers.cellKeys,
    'uint32x2',
    cellCount
  );
  const origins = importGraphBuffer(
    graph,
    `r${table.resolution}-pair-origins`,
    resources.createBuffer('pair-origins', table.origins),
    'uint32',
    pairCount
  );
  const destinations = importGraphBuffer(
    graph,
    `r${table.resolution}-pair-destinations`,
    resources.createBuffer('pair-destinations', table.destinations),
    'uint32',
    pairCount
  );
  const values = importGraphBuffer(
    graph,
    `r${table.resolution}-pair-rides`,
    resources.createBuffer('pair-rides', table.values),
    'float32',
    pairCount
  );
  const output = {
    outgoingTotals: importGraphBuffer(
      graph,
      'outgoing-totals',
      buffers.outgoing,
      'float32',
      cellCount
    ),
    incomingTotals: importGraphBuffer(
      graph,
      'incoming-totals',
      buffers.incoming,
      'float32',
      cellCount
    ),
    directedEdges: importGraphBuffer(
      graph,
      'directed-edge-keys',
      buffers.directedEdges,
      'uint32x2',
      cellCount * 6
    ),
    neighboringCells: importGraphBuffer(
      graph,
      'neighbor-cell-keys',
      buffers.neighbors,
      'uint32x2',
      cellCount * 6
    ),
    edgeFlows: importGraphBuffer(graph, 'edge-flows', buffers.edgeFlows, 'float32', cellCount * 6),
    boundaries: importGraphBuffer(
      graph,
      'gpu-edge-boundaries',
      buffers.boundaries,
      'float32x2',
      cellCount * 6 * 3
    ),
    boundaryCounts: importGraphBuffer(
      graph,
      'boundary-point-counts',
      buffers.boundaryCounts,
      'uint32',
      cellCount * 6
    ),
    validity: importGraphBuffer(graph, 'pair-validity', buffers.validity, 'uint32', pairCount),
    status: {
      invalidCount: importGraphBuffer(
        graph,
        'invalid-pair-count',
        buffers.invalidCount,
        'uint32',
        1
      )
    }
  };
  addCellFlowTopologyRecipe(graph, {
    id: `bixi-cell-flow-r${table.resolution}`,
    cells,
    origins,
    destinations,
    values,
    outputs: output
  });
  const compiled = resources.track(await graph.compileAsync());
  let activeGraph: BixiCellGraph;
  const reader = new SummaryReader(
    resources,
    `bixi-cell-flow-r${table.resolution}`,
    makeSummarySources(buffers, cellCount, pairCount),
    bytes => {
      onResult(activeGraph, parseResult(bytes, cellCount, pairCount));
    }
  );
  activeGraph = {resources, compiled, reader, table, buffers};
  return activeGraph;
}

export async function createBixiCellFlows(
  ctx: SceneContext<BixiCellFlowsOptions>
): Promise<SceneInstance<BixiCellFlowsOptions>> {
  const flows = readBixiFlows(ctx.datasets.get('bixi-flows'));
  const tables = new Map(
    RESOLUTIONS.map(resolution => [String(resolution), prepareCellPairs(flows, resolution)])
  );
  const crossStationRides = sum(flows.departures);
  const sourcePairRides = sum(flows.pairs.count);
  const rareExcludedRides = Math.max(0, crossStationRides - sourcePairRides);
  const resources = new SpatialAnalysisResources(ctx.device, 'bixi-cell-flows');
  const graphSwap = resources.track(new GraphSwap());
  let activeGraph: BixiCellGraph;
  let latestResult: FlowResult | null = null;
  let selectedCellIndex = 0;
  let selectedCellCoordinate: [number, number] | null = null;
  let hasInitialSelection = false;
  let graphDirty = true;

  const publishResult = (graph: BixiCellGraph, result: FlowResult): void => {
    const table = graph.table;
    let validPairRides = 0;
    let sameCellRides = 0;
    for (let row = 0; row < table.values.length; row++) {
      if (result.validity[row] !== 1) continue;
      validPairRides += table.values[row];
      if (table.origins[row] === table.destinations[row]) sameCellRides += table.values[row];
    }
    const outgoingTotal = sum(result.outgoing);
    const incomingTotal = sum(result.incoming);
    const neighborRides = sum(result.edgeFlows);
    const nonNeighborRides = validPairRides - sameCellRides - neighborRides;
    const conservationDifference = outgoingTotal - incomingTotal;
    const neighborShare = validPairRides > 0 ? neighborRides / validPairRides : 0;
    if (!hasInitialSelection) {
      selectedCellIndex = result.outgoing.reduce(
        (maximumIndex, value, index, values) =>
          value > values[maximumIndex] ? index : maximumIndex,
        0
      );
      hasInitialSelection = true;
      selectedCellCoordinate = table.centers[selectedCellIndex] ?? null;
    }
    const selected = Math.min(selectedCellIndex, table.keys.length - 1);
    selectedCellIndex = selected;
    const selectedOutgoing = result.outgoing[selected] ?? 0;
    const selectedIncoming = result.incoming[selected] ?? 0;
    const pairRows: {destination: number; value: number}[] = [];
    for (let row = 0; row < table.values.length; row++) {
      if (result.validity[row] === 1 && table.origins[row] === selected) {
        pairRows.push({destination: table.destinations[row], value: table.values[row]});
      }
    }
    pairRows.sort((first, second) => second.value - first.value);
    const selectedOtherDestinationRides = pairRows
      .slice(18)
      .reduce((total, pair) => total + pair.value, 0);
    const validNeighbors: {
      slot: number;
      cellIndex: number | undefined;
      key: string;
      value: number;
    }[] = [];
    const keyToCell = new Map(table.keys.map((key, index) => [key, index]));
    for (let slot = 0; slot < 6; slot++) {
      const edgeRow = selected * 6 + slot;
      const key = cellIdentity(result.neighbors, edgeRow);
      const cellIndex = keyToCell.get(key);
      if (key) validNeighbors.push({slot, cellIndex, key, value: result.edgeFlows[edgeRow]});
    }
    const selectedNeighborRides = validNeighbors.reduce((total, edge) => total + edge.value, 0);

    ctx.setReadout('stationCount', flows.stationCount);
    ctx.setReadout('cellCount', table.keys.length);
    ctx.setReadout('selectedResolution', table.resolution);
    ctx.setReadout('sourcePairs', flows.pairs.count.length);
    ctx.setReadout('sourcePairRides', sourcePairRides);
    ctx.setReadout('sameStationRides', flows.sameStationRides);
    ctx.setReadout('rareExcludedRides', rareExcludedRides);
    ctx.setReadout('sourceMonthlyRides', crossStationRides);
    ctx.setReadout('resolution', table.resolution);
    ctx.setReadout('gpuOutgoingTotal', outgoingTotal);
    ctx.setReadout('gpuIncomingTotal', incomingTotal);
    ctx.setReadout('validInputMass', validPairRides);
    ctx.setReadout('outgoingMinusValidInput', outgoingTotal - validPairRides);
    ctx.setReadout('incomingMinusValidInput', incomingTotal - validPairRides);
    ctx.setReadout('conservationDifference', conservationDifference);
    ctx.setReadout('sameCellRides', sameCellRides);
    ctx.setReadout('neighborRides', neighborRides);
    ctx.setReadout('nonNeighborRides', nonNeighborRides);
    ctx.setReadout('neighborShare', neighborShare);
    ctx.setReadout('invalidPairs', result.invalidCount);
    const gpuSelectedKey = cellIdentity(result.cellKeys, selected);
    ctx.setReadout(
      'selectedCell',
      gpuSelectedKey
        ? `H3 0x${gpuSelectedKey} · dominant station borough: ${table.boroughs[selected]}`
        : 'No selected cell'
    );
    ctx.setReadout('selectedOutgoing', selectedOutgoing);
    ctx.setReadout('selectedIncoming', selectedIncoming);
    ctx.setReadout('selectedDestinationCount', pairRows.length);
    ctx.setReadout('selectedOtherDestinationRides', selectedOtherDestinationRides);
    ctx.setReadout('selectedNeighborCount', validNeighbors.length);
    ctx.setReadout('selectedNeighborRides', selectedNeighborRides);

    const accounting = [sameCellRides, neighborRides, nonNeighborRides];
    ctx.setChart('accountingChart', {
      kind: 'bars',
      title: 'Retained valid rides by cell relation',
      labels: ['Same cell', 'Neighbor cells', 'Non-neighbor cells'],
      values: accounting,
      colors: [
        [120, 177, 197, 245],
        [113, 214, 217, 245],
        [181, 158, 197, 220]
      ],
      horizontal: true,
      description: `At H3 resolution ${table.resolution}, valid GPU input rows partition into same-cell pairs, ${neighborRides.toFixed(0)} rides on GPU directed neighbor edges, and non-neighbor cell pairs. Same-cell values are derived from the valid prepared pair rows.`
    } satisfies BarChartData);

    const destinationRows = pairRows.slice(0, 18);
    if (destinationRows.length) {
      ctx.setChart('destinationChart', {
        kind: 'bars',
        title: 'Top 18 cell endpoint links from selected origin',
        labels: destinationRows.map(
          pair => `0x${table.keys[pair.destination]} · ${table.boroughs[pair.destination]}`
        ),
        values: destinationRows.map(pair => pair.value),
        color: 2,
        horizontal: true,
        description: `Highest-count 18 of ${pairRows.length} destination-cell pairs from CPU-preaggregated input rows that passed GPU validity. Other destinations contain ${selectedOtherDestinationRides.toFixed(0)} rides. Links are endpoints, not traveled paths.`
      } satisfies BarChartData);
    } else ctx.setChart('destinationChart', null);
    if (validNeighbors.length) {
      ctx.setChart('neighborChart', {
        kind: 'bars',
        title: 'Outgoing immediate-neighbor edge flow',
        labels: validNeighbors.map(
          edge =>
            `0x${edge.key} · ${edge.cellIndex === undefined ? 'no retained stations' : table.boroughs[edge.cellIndex]}`
        ),
        values: validNeighbors.map(edge => edge.value),
        color: 4,
        horizontal: true,
        description:
          'Per-neighbor flow values read from GPUCellFlowTopologyRecipe edgeFlows. The row is one selected origin cell; zero-flow neighbor edges remain in the chart.'
      } satisfies BarChartData);
    } else ctx.setChart('neighborChart', null);
    ctx.requestLayers();
  };

  const onResult = (graph: BixiCellGraph, result: FlowResult): void => {
    if (graph !== activeGraph) return;
    latestResult = result;
    publishResult(graph, result);
  };
  const releaseGraph = (graph: BixiCellGraph): void => {
    graph.reader.release();
    graph.resources.destroy();
  };
  const requestGraph = (resolution: BixiCellFlowsOptions['resolution']): void => {
    ctx.setReadout('invalidPairs', 'Recomputing');
    void graphSwap
      .request({
        build: () => buildGraph(ctx, tables.get(resolution)!, onResult),
        commit: next => {
          const previous = activeGraph;
          activeGraph = next;
          latestResult = null;
          const previousCoordinate = selectedCellCoordinate;
          if (previousCoordinate) {
            let closest = 0;
            let closestDistance = Infinity;
            for (let index = 0; index < next.table.centers.length; index++) {
              const [longitude, latitude] = next.table.centers[index];
              const distance =
                (longitude - previousCoordinate[0]) ** 2 + (latitude - previousCoordinate[1]) ** 2;
              if (distance < closestDistance) {
                closest = index;
                closestDistance = distance;
              }
            }
            selectedCellIndex = closest;
          } else selectedCellIndex = 0;
          selectedCellCoordinate = selectedCellCoordinate
            ? (next.table.centers[selectedCellIndex] ?? null)
            : null;
          graphDirty = true;
          next.reader.markStale();
          for (const id of [
            'cellCount',
            'gpuOutgoingTotal',
            'gpuIncomingTotal',
            'neighborRides',
            'nonNeighborRides'
          ])
            ctx.setReadout(id, 'Recomputing');
          releaseGraph(previous);
          ctx.requestLayers();
        },
        discard: releaseGraph
      })
      .catch(error => ctx.setStatus(`BIXI cell-flow graph failed: ${String(error)}`));
  };

  activeGraph = await buildGraph(ctx, tables.get(ctx.options.resolution)!, onResult);
  selectedCellCoordinate = null;
  activeGraph.reader.markStale();
  ctx.setReadout('stationCount', flows.stationCount);
  ctx.setReadout('sourcePairs', flows.pairs.count.length);
  ctx.setReadout('sourcePairRides', sourcePairRides);
  ctx.setReadout('sameStationRides', flows.sameStationRides);
  ctx.setReadout('rareExcludedRides', rareExcludedRides);
  ctx.setReadout('sourceMonthlyRides', crossStationRides);

  return {
    getCompiledGraphs: () => [activeGraph.compiled],
    setOption(id) {
      if (id === 'resolution') requestGraph(ctx.options.resolution);
      if (id === 'display') ctx.requestLayers();
    },
    encode(commandEncoder) {
      if (graphDirty) {
        activeGraph.compiled.encode(commandEncoder, {parameters: undefined});
        activeGraph.reader.request(commandEncoder);
        graphDirty = false;
      } else activeGraph.reader.flush(commandEncoder);
    },
    getLayers(): Layer[] {
      const graph = activeGraph;
      const result = latestResult;
      const table = graph.table;
      const cells: CellMark[] = table.keys.map((key, index) => ({
        index,
        key,
        coordinates: table.centers[index],
        borough: table.boroughs[index]
      }));
      const cellMaximum = result ? Math.max(1, ...Array.from(result.outgoing)) : 1;
      const stationPoints = Array.from({length: flows.stationCount}, (_, station) => ({
        coordinates: [flows.lngLat[station * 2], flows.lngLat[station * 2 + 1]] as [number, number]
      }));
      const layers: Layer[] = [
        new ScatterplotLayer({
          id: 'bixi-station-source-points',
          data: stationPoints,
          getPosition: (item: {coordinates: [number, number]}) => item.coordinates,
          getRadius: 2,
          radiusUnits: 'pixels',
          radiusMinPixels: 1.5,
          getFillColor: STATION_INK,
          stroked: false,
          pickable: false
        }),
        new ScatterplotLayer({
          id: 'bixi-cell-anchors',
          data: cells,
          getPosition: (item: CellMark) => item.coordinates,
          getRadius: (item: CellMark) =>
            result ? 4 + 11 * Math.sqrt(Math.max(0, result.outgoing[item.index]) / cellMaximum) : 5,
          radiusUnits: 'pixels',
          radiusMinPixels: 4,
          getFillColor: (item: CellMark) =>
            item.index === selectedCellIndex ? CELL_INK : CELL_CONTEXT,
          getLineColor: [20, 30, 37, 255],
          lineWidthMinPixels: 1,
          stroked: true,
          pickable: true
        })
      ];
      if (result && ctx.options.display === 'neighbor-boundaries') {
        const selectedBoundaryPaths: PathMark[] = [];
        for (let slot = 0; slot < 6; slot++) {
          const edgeRow = selectedCellIndex * 6 + slot;
          const pointCount = result.boundaryCounts[edgeRow];
          if (pointCount < 2) continue;
          const path: [number, number][] = [];
          for (let point = 0; point < Math.min(pointCount, 3); point++) {
            const offset = edgeRow * 6 + point * 2;
            path.push([result.boundaries[offset], result.boundaries[offset + 1]]);
          }
          const value = result.edgeFlows[edgeRow];
          selectedBoundaryPaths.push({
            path,
            value,
            destination: cellIdentity(result.neighbors, edgeRow),
            color: value > 0 ? BOUNDARY_INK : BOUNDARY_CONTEXT
          });
        }
        layers.push(
          new PathLayer({
            id: 'bixi-gpu-neighbor-boundaries',
            data: selectedBoundaryPaths,
            getPath: (item: PathMark) => item.path,
            getColor: (item: PathMark) => item.color ?? BOUNDARY_INK,
            getWidth: (item: PathMark) =>
              1.5 +
              4 *
                Math.sqrt(
                  item.value / Math.max(1, ...selectedBoundaryPaths.map(path => path.value))
                ),
            widthUnits: 'pixels',
            widthMinPixels: 2.5,
            capRounded: true,
            jointRounded: true,
            pickable: false
          })
        );
      }
      if (result && ctx.options.display === 'od-endpoints') {
        const endpointPaths: PathMark[] = [];
        let maximumValue = 1;
        for (let row = 0; row < table.values.length; row++) {
          if (result.validity[row] !== 1 || table.origins[row] !== selectedCellIndex) continue;
          const destination = table.destinations[row];
          const value = table.values[row];
          maximumValue = Math.max(maximumValue, value);
          if (destination === selectedCellIndex) continue;
          endpointPaths.push({
            path: [table.centers[selectedCellIndex], table.centers[destination]],
            value,
            destination: table.keys[destination]
          });
        }
        layers.push(
          new PathLayer({
            id: 'bixi-selected-cell-od-endpoints',
            data: endpointPaths,
            getPath: (item: PathMark) => item.path,
            getColor: ENDPOINT_INK,
            getWidth: (item: PathMark) => 1 + 3 * Math.sqrt(item.value / maximumValue),
            widthUnits: 'pixels',
            widthMinPixels: 1.25,
            capRounded: true,
            jointRounded: true,
            pickable: false
          })
        );
      }
      return layers;
    },
    getTooltip(event: ScenePointerEvent) {
      const mark = event.info.object as CellMark | undefined;
      if (!mark || !latestResult) return null;
      return `H3 0x${mark.key} · ${mark.borough}\nOutgoing ${Math.round(latestResult.outgoing[mark.index]).toLocaleString()} rides · incoming ${Math.round(latestResult.incoming[mark.index]).toLocaleString()} rides`;
    },
    onClick(event: ScenePointerEvent) {
      const mark = event.info.object as CellMark | undefined;
      if (!mark) return false;
      selectedCellIndex = mark.index;
      selectedCellCoordinate = mark.coordinates;
      hasInitialSelection = true;
      if (latestResult) publishResult(activeGraph, latestResult);
      else ctx.requestLayers();
      return true;
    },
    onAction(id) {
      if (id !== 'previousOrigin' && id !== 'nextOrigin') return;
      const cellCount = activeGraph.table.keys.length;
      const direction = id === 'nextOrigin' ? 1 : -1;
      selectedCellIndex = (selectedCellIndex + direction + cellCount) % cellCount;
      selectedCellCoordinate = activeGraph.table.centers[selectedCellIndex] ?? null;
      hasInitialSelection = true;
      if (latestResult) publishResult(activeGraph, latestResult);
      else ctx.requestLayers();
    },
    destroy() {
      graphSwap.destroy();
      releaseGraph(activeGraph);
      resources.destroy();
    }
  };
}

// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  validateCapacity,
  validateMaximumCount,
  validateRequirement
} from '@luma-contributors/analysis/analysis-core';
import {addCellFlowTopologyRecipe} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {clipGeographicPolyline} from '@math.gl/geospatial';
import type {
  CatalogOperationBinding,
  CatalogParameterValue,
  CatalogRunOutput
} from '../../analysis/catalog-binding-types';
import {
  validateCatalogTableInput,
  type CatalogDataInput,
  type CatalogTableInput
} from '../../analysis/catalog-data-input';
import {
  appendCatalogRow,
  makeCatalogTable,
  makeCatalogGeometryText,
  type CatalogOutputTable
} from '../../analysis/catalog-geojson';
import {withCatalogGpuSession} from '../../analysis/catalog-gpu-session';
import {joinDggsWords} from '../../analysis/catalog-platform4-dggs-cells';
import type {KeplerCuratedAnalysis} from '../curated-analysis-types';

const BINDING_ID = 'curated-cell-flow-topology';
const MISSING_CELL = 0xffffffff;

/** Bounds the recipe's direct scan to six million edge/pair visits and small readbacks. */
export const CELL_FLOW_TOPOLOGY_LIMITS = {
  pairs: 10000,
  cells: 1024,
  cellPairWork: 1000000
} as const;

export type PreparedCellFlowTopology = {
  input: CatalogTableInput;
  cellKeys: string[];
  cells: Uint32Array;
  origins: Uint32Array;
  destinations: Uint32Array;
  values: Float32Array;
  originProperty: string;
  destinationProperty: string;
  flowProperty: string;
  resolution: number;
};

/** Decode exact hex text only. H3 semantic validity remains the recipe's GPU check. */
function readCellKey(value: unknown, resolution: number): string | null {
  if (typeof value !== 'string' || !/^(?:0x)?[0-9a-f]{15,16}$/i.test(value)) return null;
  const key = BigInt(value.toLowerCase().startsWith('0x') ? value : `0x${value}`);
  if (Number((key >> 52n) & 15n) !== resolution) return null;
  return key.toString(16);
}

function readMappedColumn(
  table: CatalogOutputTable,
  parameters: Readonly<Record<string, CatalogParameterValue>>,
  name: string,
  types: readonly string[]
): string {
  const column = parameters[name];
  validateRequirement(
    BINDING_ID,
    typeof column === 'string' &&
      Object.hasOwn(table.columns, column) &&
      types.includes(table.columnTypes[column]),
    `${name} must name an existing ${types.join(' or ')} column`
  );
  return column;
}

/** One-time OD decoding: distinct cell indices in first-occurrence order; pair order never changes. */
export function prepareCellFlowTopology(
  inputs: Readonly<Record<string, CatalogDataInput>>,
  parameters: Readonly<Record<string, CatalogParameterValue>>
): PreparedCellFlowTopology {
  const input = inputs['flows'];
  validateRequirement(BINDING_ID, input?.type === 'table', 'flows must be a geometry-free table');
  validateCatalogTableInput(BINDING_ID, input, CELL_FLOW_TOPOLOGY_LIMITS.pairs);
  const {table} = input;
  validateCapacity(
    BINDING_ID,
    'flows.rowCount',
    table.rowCount,
    1,
    CELL_FLOW_TOPOLOGY_LIMITS.pairs
  );
  const resolution = parameters['resolution'] ?? 1;
  validateRequirement(
    BINDING_ID,
    typeof resolution === 'number',
    'resolution must be an integer in [0, 15]'
  );
  validateCapacity(BINDING_ID, 'resolution', resolution, 0, 15);
  const originProperty = readMappedColumn(table, parameters, 'originProperty', ['string']);
  const destinationProperty = readMappedColumn(table, parameters, 'destinationProperty', [
    'string'
  ]);
  const flowProperty = readMappedColumn(table, parameters, 'flowProperty', ['float64', 'int32']);
  const cellKeys: string[] = [];
  const cellIndices = new Map<string, number>();
  const readCellIndex = (value: unknown): number => {
    const key = readCellKey(value, resolution);
    if (key === null) return MISSING_CELL;
    let index = cellIndices.get(key);
    if (index === undefined) {
      index = cellKeys.length;
      cellIndices.set(key, index);
      cellKeys.push(key);
    }
    return index;
  };
  const origins = new Uint32Array(table.rowCount);
  const destinations = new Uint32Array(table.rowCount);
  const values = new Float32Array(table.rowCount);
  for (let row = 0; row < table.rowCount; row++) {
    origins[row] = readCellIndex(table.columns[originProperty][row]);
    destinations[row] = readCellIndex(table.columns[destinationProperty][row]);
    const value = table.columns[flowProperty][row];
    // Nulls and nonnumeric values become NaN, never a fabricated zero flow.
    values[row] = typeof value === 'number' && value >= 0 ? value : Number.NaN;
  }
  validateMaximumCount(BINDING_ID, 'cellCount', cellKeys.length, CELL_FLOW_TOPOLOGY_LIMITS.cells);
  validateMaximumCount(
    BINDING_ID,
    'cellCount * pairCount',
    Math.max(cellKeys.length, 1) * table.rowCount,
    CELL_FLOW_TOPOLOGY_LIMITS.cellPairWork
  );
  // A zero sentinel gives the nonempty recipe a safe cell buffer when every key is malformed.
  // It is an internal placeholder only and is never published as a cell.
  const cells = new Uint32Array(Math.max(cellKeys.length, 1) * 2);
  cellKeys.forEach((key, index) => {
    const integer = BigInt(`0x${key}`);
    cells[index * 2] = Number(integer & 0xffffffffn);
    cells[index * 2 + 1] = Number(integer >> 32n);
  });
  return {
    input,
    cellKeys,
    cells,
    origins,
    destinations,
    values,
    originProperty,
    destinationProperty,
    flowProperty,
    resolution
  };
}

export type CellFlowTopologyReadback = {
  outgoingTotals: Float32Array;
  incomingTotals: Float32Array;
  directedEdges: Uint32Array;
  neighboringCells: Uint32Array;
  edgeFlows: Float32Array;
  boundaries: Float32Array;
  boundaryCounts: Uint32Array;
  validity: Uint32Array;
  invalidCount: number;
};

/** Plain tables for native Sources publication; no geometry is invented for the original OD rows. */
export function publishCellFlowTopology(
  prepared: PreparedCellFlowTopology,
  result: CellFlowTopologyReadback
): CatalogRunOutput {
  const cellCount = prepared.cells.length / 2;
  const pairCount = prepared.values.length;
  for (const [name, length] of [
    ['outgoingTotals', cellCount],
    ['incomingTotals', cellCount],
    ['directedEdges', cellCount * 12],
    ['neighboringCells', cellCount * 12],
    ['edgeFlows', cellCount * 6],
    ['boundaries', cellCount * 36],
    ['boundaryCounts', cellCount * 6],
    ['validity', pairCount]
  ] as const)
    validateRequirement(
      BINDING_ID,
      result[name].length === length,
      `${name}.length must equal ${length}`
    );
  const pairs = makeCatalogTable('pairs', 'Original OD rows and validity', {
    source_index: 'int32',
    source_id: 'string',
    origin: 'string',
    destination: 'string',
    origin_index: 'int32',
    destination_index: 'int32',
    flow: 'float64',
    gpu_flow: 'float64',
    validity: 'boolean',
    status: 'string'
  });
  const cells = makeCatalogTable('cells', 'Cell flow totals', {
    cell_index: 'int32',
    cell_id: 'string',
    outgoing: 'float64',
    incoming: 'float64',
    validity: 'boolean'
  });
  const edges = makeCatalogTable(
    'edges',
    'Neighboring cell flows',
    {
      cell_index: 'int32',
      direction: 'int32',
      origin: 'string',
      destination: 'string',
      destination_index: 'int32',
      directed_edge: 'string',
      flow: 'float64',
      boundary_count: 'int32',
      geometry: 'string'
    },
    'geometry'
  );
  edges.coordinateSystem = 'EPSG:4326';
  const indices = new Map(prepared.cellKeys.map((key, index) => [key, index]));
  let rejectedCount = 0;
  for (let row = 0; row < pairCount; row++) {
    const validity = result.validity[row] !== 0;
    if (!validity) rejectedCount++;
    const identity = prepared.input.identityColumn
      ? prepared.input.table.columns[prepared.input.identityColumn][row]
      : null;
    appendCatalogRow(pairs, {
      source_index: row,
      source_id: identity === null ? null : String(identity),
      origin: prepared.input.table.columns[prepared.originProperty][row],
      destination: prepared.input.table.columns[prepared.destinationProperty][row],
      origin_index: prepared.origins[row] === MISSING_CELL ? null : prepared.origins[row],
      destination_index:
        prepared.destinations[row] === MISSING_CELL ? null : prepared.destinations[row],
      flow: prepared.input.table.columns[prepared.flowProperty][row],
      gpu_flow: Number.isFinite(prepared.values[row]) ? prepared.values[row] : null,
      validity,
      status: validity ? 'valid' : 'rejected'
    });
  }
  let nonfiniteCount = 0;
  let outgoingTotal = 0;
  let incomingTotal = 0;
  let neighboringFlowTotal = 0;
  let invalidCellCount = 0;
  let invalidBoundaryCount = 0;
  let antimeridianBoundaryCount = 0;
  for (let cell = 0; cell < prepared.cellKeys.length; cell++) {
    const validity = result.boundaryCounts
      .subarray(cell * 6, cell * 6 + 6)
      .some(count => count > 0);
    if (!validity) invalidCellCount++;
    outgoingTotal += result.outgoingTotals[cell];
    incomingTotal += result.incomingTotals[cell];
    if (
      !Number.isFinite(result.outgoingTotals[cell]) ||
      !Number.isFinite(result.incomingTotals[cell])
    )
      nonfiniteCount++;
    appendCatalogRow(cells, {
      cell_index: cell,
      cell_id: prepared.cellKeys[cell],
      outgoing: result.outgoingTotals[cell],
      incoming: result.incomingTotals[cell],
      validity
    });
    for (let direction = 1; direction <= 6; direction++) {
      const edge = cell * 6 + direction - 1;
      const count = result.boundaryCounts[edge];
      // Pentagon deleted directions and invalid H3 cells have no boundary.
      if (!count) continue;
      if (count < 2 || count > 3) {
        invalidBoundaryCount++;
        continue;
      }
      const destination = joinDggsWords(
        result.neighboringCells[edge * 2],
        result.neighboringCells[edge * 2 + 1]
      ).toString(16);
      const coordinates = Array.from({length: count}, (_, vertex) => [
        result.boundaries[edge * 6 + vertex * 2],
        result.boundaries[edge * 6 + vertex * 2 + 1]
      ]);
      const finiteCoordinates = coordinates.flat().every(value => Number.isFinite(value));
      if (!Number.isFinite(result.edgeFlows[edge]) || !finiteCoordinates) nonfiniteCount++;
      // Split the short geographic edge at +/-180 while retaining one row and flow per H3 edge.
      // Invalid readback bypasses the splitter and is rejected by the aggregate check below.
      const parts = finiteCoordinates
        ? clipGeographicPolyline(coordinates.flat(), {
            edgeModel: 'linear',
            linearLongitude: 'shortest',
            latitudeRange: [-90, 90]
          }).map(part =>
            Array.from({length: part.positions.length / 2}, (_, vertex) => [
              part.positions[vertex * 2],
              part.positions[vertex * 2 + 1]
            ])
          )
        : [coordinates];
      if (parts.length > 1) antimeridianBoundaryCount++;
      neighboringFlowTotal += result.edgeFlows[edge];
      appendCatalogRow(edges, {
        cell_index: cell,
        direction,
        origin: prepared.cellKeys[cell],
        destination,
        destination_index: indices.get(destination) ?? null,
        directed_edge: joinDggsWords(
          result.directedEdges[edge * 2],
          result.directedEdges[edge * 2 + 1]
        ).toString(16),
        flow: result.edgeFlows[edge],
        boundary_count: count,
        geometry:
          parts.length > 1
            ? makeCatalogGeometryText('MultiLineString', parts)
            : makeCatalogGeometryText('LineString', parts[0] ?? coordinates)
      });
    }
  }
  validateRequirement(
    BINDING_ID,
    invalidBoundaryCount === 0,
    'boundaryCounts must be 0 or in [2, 3]'
  );
  validateRequirement(
    BINDING_ID,
    rejectedCount === result.invalidCount,
    'status.invalidCount must equal the rejected pair count'
  );
  validateRequirement(
    BINDING_ID,
    nonfiniteCount === 0,
    'outgoingTotals, incomingTotals, edgeFlows and boundaries must be finite; reduce flow magnitudes'
  );
  return {
    tables: [pairs, cells, edges],
    dispatchedContributors: ['GPUH3CellIdentities'],
    diagnostics: rejectedCount
      ? [
          {
            code: 'rejected-flow',
            severity: 'warning',
            count: rejectedCount,
            message:
              'Rejected OD rows contribute to neither totals nor edges. Check exact H3 hex keys, resolution, and nonnegative finite float32 flows.'
          }
        ]
      : [],
    summary: {
      pairCount,
      cellCount: prepared.cellKeys.length,
      invalidCellCount,
      antimeridianBoundaryCount,
      invalidCount: result.invalidCount,
      validPairCount: pairCount - result.invalidCount,
      outgoingTotal,
      incomingTotal,
      neighboringFlowTotal,
      resolution: prepared.resolution,
      flowUnit: 'input flow units',
      boundaryCoordinates: 'EPSG:4326 degrees',
      recipe: 'addCellFlowTopologyRecipe',
      precision: 'float32; exact H3 keys'
    }
  };
}

export const CURATED_CELL_FLOW_TOPOLOGY_BINDING: CatalogOperationBinding = {
  id: BINDING_ID,
  title: 'Cell flow topology',
  domain: 'analysis',
  summary: 'Summarize OD flows and show the boundaries of neighboring H3 cells.',
  capabilityIds: ['dggs-indexing-and-topology'],
  contributorNames: ['GPUH3CellIdentities'],
  gpuContributorNames: ['GPUH3CellIdentities'],
  engine: 'gpu',
  inputs: [
    {
      name: 'flows',
      label: 'OD table',
      dataKind: 'table',
      accepts: [],
      maximumFeatures: CELL_FLOW_TOPOLOGY_LIMITS.pairs,
      maximumVertices: 0
    }
  ],
  parameters: [
    ...['origin', 'destination', 'flow'].map(role => ({
      name: `${role}Property`,
      label: `${role} column`,
      kind: 'string' as const,
      maximumLength: 256,
      defaultValue: ''
    })),
    {
      name: 'resolution',
      label: 'H3 resolution',
      kind: 'integer',
      minimum: 0,
      maximum: 15,
      defaultValue: 1
    }
  ],
  outputs: [
    {
      tableId: 'pairs',
      label: 'OD rows',
      geometry: 'none',
      identityColumns: ['source_index', 'source_id'],
      maximumRows: CELL_FLOW_TOPOLOGY_LIMITS.pairs
    },
    {
      tableId: 'cells',
      label: 'Cell totals',
      geometry: 'none',
      identityColumns: ['cell_index', 'cell_id'],
      maximumRows: CELL_FLOW_TOPOLOGY_LIMITS.cells
    },
    {
      tableId: 'edges',
      label: 'Neighbor flows',
      geometry: 'line',
      identityColumns: ['cell_index', 'direction', 'directed_edge'],
      maximumRows: CELL_FLOW_TOPOLOGY_LIMITS.cells * 6
    }
  ],
  cleanup:
    'All GPU resources are released after readback, including failures; the caller owns the device and plain output arrays.',
  declaredBehavior: [
    'Exact H3 hex strings are decoded as little-endian uint32 pairs, never JavaScript numeric keys. Case and optional leading zero/0x normalize to the same cell.',
    'Cell indices follow first occurrence. Original OD row order and source identity are retained, including duplicates and rejected rows.',
    'Missing, malformed, invalid or wrong-resolution cells and negative/nonfinite float32 flows are rejected; validity and invalidCount come from the recipe.',
    'All valid pairs, including self and nonneighbor flows, enter cell totals. Repeated neighboring pairs sum; no routes are inferred.',
    'Every existing direction has one shared-boundary line, including zero flow and neighbors outside the OD cell list. Pentagon missing directions are omitted.',
    'Antimeridian boundaries are split into MultiLineString parts at +/-180 in a single directed-edge row. Output geometry declares EPSG:4326 for downstream profiling.',
    'Pair, cell and cell-pair work caps throw before GPU allocation. Nonfinite GPU aggregates prevent publication.'
  ],
  limitation:
    'Requires WebGPU. Direct scan is bounded; preaggregate larger OD tables. Flow sums and geographic boundaries have the existing recipe float32 precision limits. Opposite directed flows share a boundary and remain separate rows.',
  async runDataGpu({device}, inputs, parameters) {
    const prepared = prepareCellFlowTopology(inputs, parameters);
    return withCatalogGpuSession(device, BINDING_ID, async session => {
      const cellCount = prepared.cells.length / 2;
      const pairCount = prepared.values.length;
      const outgoingTotals = session.output('outgoing', 'float32', cellCount);
      const incomingTotals = session.output('incoming', 'float32', cellCount);
      const directedEdges = session.output('directed-edges', 'uint32x2', cellCount * 6);
      const neighboringCells = session.output('neighbors', 'uint32x2', cellCount * 6);
      const edgeFlows = session.output('edge-flows', 'float32', cellCount * 6);
      const boundaries = session.output('boundaries', 'float32x2', cellCount * 18);
      const boundaryCounts = session.output('boundary-counts', 'uint32', cellCount * 6);
      const validity = session.output('validity', 'uint32', pairCount);
      const invalidCount = session.output('invalid-count', 'uint32', 1);
      addCellFlowTopologyRecipe(session.graph, {
        cells: session.input('cells', prepared.cells, 'uint32x2'),
        origins: session.input('origins', prepared.origins, 'uint32'),
        destinations: session.input('destinations', prepared.destinations, 'uint32'),
        values: session.input('values', prepared.values, 'float32'),
        output: {
          outgoingTotals: outgoingTotals.view,
          incomingTotals: incomingTotals.view,
          directedEdges: directedEdges.view,
          neighboringCells: neighboringCells.view,
          edgeFlows: edgeFlows.view,
          boundaries: boundaries.view,
          boundaryCounts: boundaryCounts.view,
          validity: validity.view,
          status: {invalidCount: invalidCount.view}
        }
      });
      session.run();
      const readback = await Promise.all([
        session.readFloat32(outgoingTotals),
        session.readFloat32(incomingTotals),
        session.readUint32(directedEdges),
        session.readUint32(neighboringCells),
        session.readFloat32(edgeFlows),
        session.readFloat32(boundaries),
        session.readUint32(boundaryCounts),
        session.readUint32(validity),
        session.readScalar(invalidCount)
      ]);
      return publishCellFlowTopology(prepared, {
        outgoingTotals: readback[0],
        incomingTotals: readback[1],
        directedEdges: readback[2],
        neighboringCells: readback[3],
        edgeFlows: readback[4],
        boundaries: readback[5],
        boundaryCounts: readback[6],
        validity: readback[7],
        invalidCount: readback[8]
      });
    });
  }
};

/** Ready for the curated registry owner to adopt; plain OD tables profile as attributes. */
export const CURATED_CELL_FLOW_TOPOLOGY_ANALYSIS: KeplerCuratedAnalysis = {
  id: 'cell-flow-topology',
  question: 'how-it-moves',
  title: 'Which neighbouring cells exchange flow?',
  name: 'Cell flow topology',
  wave: 2,
  method:
    'Check exact H3 OD keys, retain all original rows with validity, compute all valid cell totals and aggregate neighboring flows along their shared boundaries. Self and distant pairs affect totals without inferring routes.',
  slots: [
    {
      name: 'flows',
      label: 'Flows between cells',
      kind: 'attributes',
      fields: [
        {role: 'origin', label: 'Origin H3 cell', accepts: ['text', 'category'], required: true},
        {
          role: 'destination',
          label: 'Destination H3 cell',
          accepts: ['text', 'category'],
          required: true
        },
        {
          role: 'flow',
          label: 'Flow',
          accepts: ['number', 'integer'],
          measurement: ['count', 'amount'],
          required: true
        }
      ]
    }
  ],
  execution: {
    status: 'available',
    bindingId: BINDING_ID,
    inputs: {flows: 'flows'},
    fieldParameters: {
      'flows.origin': 'originProperty',
      'flows.destination': 'destinationProperty',
      'flows.flow': 'flowProperty'
    },
    parameters: {}
  },
  exposedParameters: ['resolution'],
  sampleId: 'cell-flow-topology'
};

// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Layer} from '@deck.gl/core';
import {Buffer, type CommandEncoder} from '@luma.gl/core';
import {
  GPUBoundedFacilityAllocation,
  GPULocationAllocation,
  type GPULocationAllocationSparseCosts,
  type GPULocationAllocationOperation
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {GPUCommandGraph, type GraphDataView} from '@luma.gl/graph/gpu-core';
import type {GPUVectorFormat} from '@luma.gl/graph/gpu-data';
import {createGraphBufferImporter} from '../../engine/graph-buffers';
import {SpatialAnalysisPointLayer, SpatialAnalysisSegmentLayer} from '../../engine/layers';
import {SpatialAnalysisResources} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import type {SceneContext, SceneInstance} from '../scene';
import type {MapAnnotation} from '../../cartography/types';
import {readBixiFlows} from '../flows/bixi-data';
import {
  BIXI_ASSIGNMENT_PALETTE,
  findBixiStationAtPixel,
  BIXI_MODEL_ORIGIN,
  makeBixiModelData,
  type BixiModelData
} from '../flows/bixi-model-data';

const LOCATION_OPERATIONS: readonly GPULocationAllocationOperation[] = [
  'p-median',
  'maximum-coverage',
  'set-covering'
];
const BOUNDED_OPERATIONS = ['p-center', 'capacitated-p-median'] as const;
const SELECT_COUNT = 3;
const SPARSE_COST_CUTOFF_KM = 3;
const CAPACITY_MULTIPLIER = 0.4;
const STATION_ORDERS = ['station-id', 'activity-descending'] as const;

export type BixiServiceLocationsOptions = {
  operation: (typeof LOCATION_OPERATIONS)[number] | (typeof BOUNDED_OPERATIONS)[number];
  demandOrder: (typeof STATION_ORDERS)[number];
  costSupport: 'dense' | 'sparse-3km';
  coverageRadiusKm: number;
  display: 'assignments' | 'activity';
};

type AllocationBuffers = {
  selected: Buffer;
  assignments: Buffer;
  objective: Buffer;
  status: Buffer;
  assignedCosts?: Buffer;
};

type AllocationSnapshot = {
  objective: number;
  status: Uint32Array;
  selectedCount: number;
  selected: Uint32Array;
  assignments: Uint32Array;
};

/** Small hypothetical siting experiment using the same observed BIXI station activity. */
export async function createBixiServiceLocations(
  context: SceneContext<BixiServiceLocationsOptions>
): Promise<SceneInstance<BixiServiceLocationsOptions>> {
  const dataset = context.datasets.get('bixi-flows');
  const data = makeBixiModelData(readBixiFlows(dataset));
  const resources = new SpatialAnalysisResources(context.device, 'bixi-service-locations');
  const demandOrders = createDemandOrders(data);
  const facilityCount = data.candidateIds.length;
  const demandCount = data.stationIds.length;
  const costs = createCosts(data, facilityCount, demandCount);
  const capacityValue = data.activity.reduce((sum, value) => sum + value, 0) * CAPACITY_MULTIPLIER;
  const capacities = resources.createBuffer(
    'candidate-capacities',
    new Float32Array(facilityCount).fill(capacityValue)
  );
  const coverageDistance = resources.createBuffer(
    'coverage-radius-km',
    new Float32Array([context.options.coverageRadiusKm])
  );
  const outputs = new Map<string, AllocationBuffers>();
  const orderBuffers = new Map<string, {positions: Buffer; activity: Buffer}>();
  const graph = new GPUCommandGraph<void>(context.device, {id: 'bixi-service-locations'});
  const importBuffer = createGraphBufferImporter(graph);
  const capacityView = importBuffer(capacities, 'float32', facilityCount);
  const coverageView = importBuffer(coverageDistance, 'float32', 1);

  for (const [orderName, order] of Object.entries(demandOrders)) {
    const orderPositions = reorderPairs(data.positions, order);
    const orderActivity = Float32Array.from(order, station => data.activity[station]);
    const positionBuffer = resources.createBuffer(
      `${orderName}-demand-positions`,
      orderPositions,
      Buffer.VERTEX
    );
    const demandBuffer = resources.createBuffer(
      `${orderName}-demand-activity`,
      orderActivity,
      Buffer.VERTEX
    );
    orderBuffers.set(orderName, {positions: positionBuffer, activity: demandBuffer});
    const costValues = reorderCosts(costs, order, facilityCount);
    const costBuffer = resources.createBuffer(`${orderName}-costs-km`, costValues);
    const costView = importBuffer(costBuffer, 'float32', demandCount * facilityCount);
    const sparseValues = createSparseCosts(costValues, demandCount, facilityCount);
    const sparseBuffers = {
      offsets: resources.createBuffer(`${orderName}-sparse-offsets`, sparseValues.offsets),
      facilities: resources.createBuffer(`${orderName}-sparse-facilities`, sparseValues.facilities),
      costs: resources.createBuffer(`${orderName}-sparse-costs-km`, sparseValues.costs)
    };
    const sparseCostViews: GPULocationAllocationSparseCosts = {
      offsets: importBuffer(sparseBuffers.offsets, 'uint32', sparseValues.offsets.length),
      facilities: importBuffer(sparseBuffers.facilities, 'uint32', sparseValues.facilities.length),
      costs: importBuffer(sparseBuffers.costs, 'float32', sparseValues.costs.length)
    };
    const demandView = importBuffer(demandBuffer, 'float32', demandCount);
    const prefix = orderName;
    for (const operation of LOCATION_OPERATIONS) {
      for (const support of ['dense', 'sparse-3km'] as const) {
        const output = makeOutput(
          resources,
          `${prefix}-${operation}-${support}`,
          facilityCount,
          demandCount,
          true
        );
        outputs.set(`${operation}:${orderName}:${support}`, output);
        const outputViews = makeOutputViews(importBuffer, output, facilityCount, demandCount, true);
        const props = {
          id: `bixi-${operation}-${orderName}-${support}`,
          ...(support === 'dense' ? {costs: costView} : {sparseCosts: sparseCostViews}),
          demandWeights: demandView,
          facilityCount,
          operation,
          maximumFacilityCount: operation === 'set-covering' ? facilityCount : SELECT_COUNT,
          maximumSwapIterations: operation === 'set-covering' ? 0 : 16,
          ...(operation !== 'p-median' ? {coverageDistance: coverageView} : {}),
          output: outputViews
        };
        graph.add(new GPULocationAllocation(props));
      }
    }
    for (const operation of BOUNDED_OPERATIONS) {
      const output = makeOutput(
        resources,
        `${prefix}-${operation}`,
        facilityCount,
        demandCount,
        false
      );
      outputs.set(`${operation}:${orderName}:dense`, output);
      const outputViews = makeOutputViews(importBuffer, output, facilityCount, demandCount, false);
      graph.add(
        new GPUBoundedFacilityAllocation({
          id: `bixi-${operation}-${orderName}`,
          operation,
          facilityCount,
          selectCount: SELECT_COUNT,
          costs: costView,
          demand: demandView,
          capacities: capacityView,
          output: outputViews
        })
      );
    }
  }
  const compiled = resources.track(await graph.compileAsync());
  const candidateLayerBuffer = resources.createBuffer(
    'candidate-points',
    data.candidatePositions,
    Buffer.VERTEX
  );
  const spokeBuffersByOrder = new Map<string, Buffer[]>();
  for (const [orderName, order] of Object.entries(demandOrders)) {
    spokeBuffersByOrder.set(
      orderName,
      Array.from(data.candidateIds, (_, candidate) =>
        resources.createBuffer(
          `${orderName}-candidate-spokes-${candidate}`,
          createSpokes(data, candidate, order),
          Buffer.VERTEX
        )
      )
    );
  }
  const outputsByKey = Array.from(outputs.entries());
  const readerSources = outputsByKey.flatMap(([, output]) => [
    {buffer: output.objective, size: 4},
    {buffer: output.status, size: output.status.byteLength},
    {buffer: output.selected, size: facilityCount * 4},
    {buffer: output.assignments, size: demandCount * 4}
  ]);
  let dirty = true;
  let destroyed = false;
  let configurationRevision = 0;
  let readbackRevision = 0;
  let lastStatuses = new Map<string, AllocationSnapshot>();
  const reader = new SummaryReader(resources, 'bixi-service-locations', readerSources, bytes => {
    if (destroyed) return;
    if (readbackRevision !== configurationRevision) return;
    const floats = new Float32Array(bytes);
    const words = new Uint32Array(bytes);
    let byteOffset = 0;
    const nextStatuses = new Map<string, AllocationSnapshot>();
    for (const [key, output] of outputsByKey) {
      const objective = floats[byteOffset / 4];
      const statusLength = output.status.byteLength / 4;
      const status = words.slice(byteOffset / 4 + 1, byteOffset / 4 + 1 + statusLength);
      const selectedOffset = byteOffset / 4 + 1 + statusLength;
      const selected = words.slice(selectedOffset, selectedOffset + facilityCount);
      const selectedCount = selected.reduce((count, value) => count + Number(value !== 0), 0);
      const assignments = words.slice(
        selectedOffset + facilityCount,
        selectedOffset + facilityCount + demandCount
      );
      nextStatuses.set(key, {objective, status, selectedCount, selected, assignments});
      byteOffset += 4 + output.status.byteLength + (facilityCount + demandCount) * 4;
    }
    lastStatuses = nextStatuses;
    updateReadouts();
    context.requestLayers();
  });
  const selectedOrder = () =>
    context.options.operation === 'capacitated-p-median'
      ? context.options.demandOrder
      : 'activity-descending';
  const selectedSupport = () =>
    context.options.operation === 'p-center' || context.options.operation === 'capacitated-p-median'
      ? 'dense'
      : context.options.costSupport;
  const outputKey = () => `${context.options.operation}:${selectedOrder()}:${selectedSupport()}`;
  const updateReadouts = () => {
    const status = lastStatuses.get(outputKey());
    if (!status) {
      for (const readout of [
        'objectiveValue',
        'solverStatus',
        'selectedSites',
        'uncoveredDemand',
        'swapStatus',
        'capacityStatus',
        'selectedStationNames'
      ]) {
        context.setReadout(readout, 'pending');
      }
      context.setAnnotations('bixi-support-sites', null);
      return;
    }
    context.setReadout(
      'objectiveValue',
      isInvalidStatus(context.options.operation, status.status)
        ? 'withheld'
        : context.options.operation === 'set-covering'
          ? status.selectedCount
          : Number.isFinite(status.objective)
            ? status.objective.toFixed(2)
            : 'unavailable'
    );
    context.setReadout('solverStatus', formatStatus(context.options.operation, status.status));
    const invalid = isInvalidStatus(context.options.operation, status.status);
    context.setReadout('selectedSites', invalid ? 'withheld' : status.selectedCount);
    context.setReadout(
      'uncoveredDemand',
      invalid
        ? 'withheld'
        : context.options.operation === 'p-center' ||
            context.options.operation === 'capacitated-p-median'
          ? (status.status[0] ?? 0)
          : (status.status[1] ?? 0)
    );
    const candidateName = (candidate: number) =>
      data.stationNames[data.stationIds.indexOf(data.candidateIds[candidate])];
    context.setReadout(
      'selectedStationNames',
      invalid
        ? 'withheld'
        : Array.from(status.selected)
            .flatMap((selected, candidate) =>
              selected
                ? [`${String.fromCharCode(65 + candidate)}: ${candidateName(candidate)}`]
                : []
            )
            .join(' · ')
    );
    context.setAnnotations(
      'bixi-support-sites',
      invalid
        ? null
        : Array.from(data.candidateIds, (stationId, candidate): MapAnnotation => {
            const station = data.stationIds.indexOf(stationId);
            return {
              kind: 'point',
              id: `bixi-site-${candidate}`,
              coordinate: [
                data.geographicPositions[station * 2],
                data.geographicPositions[station * 2 + 1]
              ],
              text: String.fromCharCode(65 + candidate),
              marker: 'none',
              priority: status.selected[candidate] ? 100 : 0,
              rank: status.selected[candidate] ? 'subject' : 'context'
            };
          })
    );
    context.setReadout('swapStatus', swapStatus(status.status));
    context.setReadout(
      'costSupport',
      selectedSupport() === 'dense'
        ? 'Full demand-major dense 16 × 8 matrix'
        : 'Sparse CSR support truncated at 3 km'
    );
    context.setReadout('coverageRadius', `${context.options.coverageRadiusKm.toFixed(1)} km`);
    context.setReadout('objectiveUnits', objectiveUnits(context.options.operation));
    context.setReadout(
      'capacityStatus',
      context.options.operation === 'capacitated-p-median'
        ? status.status[1] !== 1
          ? 'invalid inputs'
          : status.status[0] === 0
            ? 'feasible; order-sensitive greedy assignment'
            : `partial; ${status.status[0]} unassigned rows`
        : 'not a capacity scenario'
    );
  };
  const assignmentBuffer = () => outputs.get(outputKey())!.assignments;
  const selectedBuffer = () => outputs.get(outputKey())!.selected;
  const getLayers = (): Layer[] => {
    const order = selectedOrder();
    const demandBuffers = orderBuffers.get(order)!;
    const assignments = assignmentBuffer();
    const activity = context.options.display === 'activity' ? demandBuffers.activity : assignments;
    const status = lastStatuses.get(outputKey());
    const outputIsValid = status
      ? !isInvalidStatus(context.options.operation, status.status)
      : false;
    const candidateValues = selectedBuffer();
    const coordinateOrigin: [number, number, number] = [...BIXI_MODEL_ORIGIN, 0];
    const spokeLayers =
      context.options.display === 'assignments' && outputIsValid
        ? Array.from(
            data.candidateIds,
            (_, candidate) =>
              new SpatialAnalysisSegmentLayer({
                id: `bixi-assignment-spokes-${candidate}`,
                coordinateOrigin,
                segments: spokeBuffersByOrder.get(order)![candidate],
                instanceCount: demandCount,
                values: assignments,
                valueFormat: 'uint32',
                highlightClasses: [candidate],
                highlightActive: true,
                dimOpacity: 0,
                colormap: 'category',
                palette: BIXI_ASSIGNMENT_PALETTE,
                noDataValue: 0xffffffff,
                noDataColor: [0, 0, 0, 0],
                widthPixels: 1.2
              })
          )
        : [];
    return [
      ...spokeLayers,
      new SpatialAnalysisPointLayer({
        id: 'bixi-demand-stations',
        coordinateOrigin,
        positions: demandBuffers.positions,
        instanceCount: demandCount,
        values: context.options.display === 'activity' || outputIsValid ? activity : undefined,
        valueFormat:
          context.options.display === 'activity' ? 'float32' : outputIsValid ? 'uint32' : 'none',
        colormap: context.options.display === 'activity' ? 'cividis' : 'category',
        ...(context.options.display === 'assignments' ? {palette: BIXI_ASSIGNMENT_PALETTE} : {}),
        valueRange: [0, Math.max(...data.activity)],
        radiusPixels: 5,
        noDataValue: 0xffffffff,
        noDataColor: [183, 55, 48, 255]
      }),
      new SpatialAnalysisPointLayer({
        id: 'bixi-hypothetical-sites',
        coordinateOrigin,
        positions: candidateLayerBuffer,
        instanceCount: facilityCount,
        ...(outputIsValid
          ? {values: candidateValues, valueFormat: 'uint32' as const}
          : {valueFormat: 'none' as const}),
        colormap: 'mask',
        color: [28, 33, 39, 255],
        noDataColor: [0, 0, 0, 0],
        radiusPixels: 9,
        outlineColor: [248, 242, 222, 255],
        outlineWidthPixels: 2
      })
    ];
  };
  return {
    getCompiledGraphs: () => [compiled],
    encode: (commandEncoder: CommandEncoder) => {
      if (dirty) {
        compiled.encode(commandEncoder, {parameters: undefined});
        if (!reader.isPending) readbackRevision = configurationRevision;
        reader.request(commandEncoder);
        dirty = false;
      } else {
        if (!reader.isPending) readbackRevision = configurationRevision;
        reader.flush(commandEncoder);
      }
    },
    getLayers,
    getTooltip(event) {
      const viewport = context.getViewport();
      if (!viewport) return null;
      const station = findBixiStationAtPixel(data, viewport, event.pixel);
      if (station < 0) return null;
      const snapshot = lastStatuses.get(outputKey());
      const demandRow = demandOrders[selectedOrder()].indexOf(station);
      const assignment = snapshot?.assignments[demandRow];
      const assignedName =
        !snapshot || isInvalidStatus(context.options.operation, snapshot.status)
          ? 'waiting / invalid'
          : assignment === undefined || assignment >= facilityCount
            ? 'unassigned'
            : `${String.fromCharCode(65 + assignment)} · ${data.stationNames[data.stationIds.indexOf(data.candidateIds[assignment])]}`;
      return {
        title: data.stationNames[station],
        subtitle: 'Observed BIXI station · hypothetical support assignment',
        rows: [
          {label: 'August departures + arrivals', value: data.activity[station].toLocaleString()},
          {label: 'GPU assigned candidate', value: assignedName}
        ],
        note: 'Activity is a ride-count proxy. Candidate support sites and straight-line service costs are hypothetical.'
      };
    },
    setOption: optionId => {
      if (optionId === 'coverageRadiusKm') {
        coverageDistance.write(new Float32Array([context.options.coverageRadiusKm]));
        configurationRevision++;
        lastStatuses.clear();
        dirty = true;
      }
      context.requestLayers();
      updateReadouts();
    },
    destroy: () => {
      destroyed = true;
      reader.release();
      resources.destroy();
    }
  };
}

function makeOutput(
  resources: SpatialAnalysisResources,
  name: string,
  facilityCount: number,
  demandCount: number,
  withSwap: boolean
): AllocationBuffers {
  return {
    selected: resources.createBuffer(
      `${name}-selected`,
      new Uint32Array(facilityCount),
      Buffer.VERTEX
    ),
    assignments: resources.createBuffer(
      `${name}-assignments`,
      new Uint32Array(demandCount).fill(0xffffffff),
      Buffer.VERTEX
    ),
    objective: resources.createBuffer(`${name}-objective`, new Float32Array(1)),
    status: resources.createBuffer(`${name}-status`, new Uint32Array(withSwap ? 7 : 3)),
    ...(withSwap
      ? {
          assignedCosts: resources.createBuffer(
            `${name}-assigned-costs`,
            new Float32Array(demandCount)
          )
        }
      : {})
  };
}

function createSparseCosts(
  costValues: Float32Array,
  demandCount: number,
  facilityCount: number
): {offsets: Uint32Array; facilities: Uint32Array; costs: Float32Array} {
  const offsets = new Uint32Array(demandCount + 1);
  const facilities: number[] = [];
  const costs: number[] = [];
  for (let demand = 0; demand < demandCount; demand++) {
    for (let facility = 0; facility < facilityCount; facility++) {
      const cost = costValues[demand * facilityCount + facility];
      if (cost <= SPARSE_COST_CUTOFF_KM) {
        facilities.push(facility);
        costs.push(cost);
      }
    }
    offsets[demand + 1] = facilities.length;
  }
  return {offsets, facilities: Uint32Array.from(facilities), costs: Float32Array.from(costs)};
}

function makeOutputViews(
  importBuffer: <Format extends GPUVectorFormat>(
    buffer: Buffer,
    format: Format,
    length?: number
  ) => GraphDataView<Format>,
  output: AllocationBuffers,
  facilityCount: number,
  demandCount: number,
  withSwap: boolean
) {
  const view = <Format extends GPUVectorFormat>(buffer: Buffer, format: Format, length: number) =>
    importBuffer(buffer, format, length);
  return {
    selectedFacilities: view(output.selected, 'uint32', facilityCount),
    assignments: view(output.assignments, 'uint32', demandCount),
    ...(withSwap && output.assignedCosts
      ? {assignedCosts: view(output.assignedCosts, 'float32', demandCount)}
      : {}),
    objective: view(output.objective, 'float32', 1),
    status: view(output.status, 'uint32', withSwap ? 7 : 3)
  };
}

function createCosts(
  data: BixiModelData,
  facilityCount: number,
  demandCount: number
): Float32Array {
  const costs = new Float32Array(demandCount * facilityCount);
  for (let demand = 0; demand < demandCount; demand++) {
    for (let facility = 0; facility < facilityCount; facility++) {
      const deltaX = data.positions[demand * 2] - data.candidatePositions[facility * 2];
      const deltaY = data.positions[demand * 2 + 1] - data.candidatePositions[facility * 2 + 1];
      costs[demand * facilityCount + facility] = Math.hypot(deltaX, deltaY) / 1000;
    }
  }
  return costs;
}

function createDemandOrders(data: BixiModelData): Record<string, number[]> {
  return {
    'activity-descending': Array.from({length: data.stationIds.length}, (_, index) => index),
    'station-id': Array.from({length: data.stationIds.length}, (_, index) => index).sort(
      (left, right) => data.stationIds[left] - data.stationIds[right]
    )
  };
}

function reorderPairs(source: Float32Array, order: readonly number[]): Float32Array {
  const values = new Float32Array(order.length * 2);
  order.forEach((station, index) =>
    values.set(source.subarray(station * 2, station * 2 + 2), index * 2)
  );
  return values;
}

function reorderCosts(
  source: Float32Array,
  order: readonly number[],
  facilityCount: number
): Float32Array {
  const values = new Float32Array(source.length);
  order.forEach((station, index) =>
    values.set(
      source.subarray(station * facilityCount, (station + 1) * facilityCount),
      index * facilityCount
    )
  );
  return values;
}

function createSpokes(
  data: BixiModelData,
  facility: number,
  demandOrder: readonly number[]
): Float32Array {
  const segments = new Float32Array(data.stationIds.length * 4);
  for (let demand = 0; demand < data.stationIds.length; demand++) {
    const station = demandOrder[demand];
    segments.set(
      [
        data.positions[station * 2],
        data.positions[station * 2 + 1],
        data.candidatePositions[facility * 2],
        data.candidatePositions[facility * 2 + 1]
      ],
      demand * 4
    );
  }
  return segments;
}

function formatStatus(
  operation: BixiServiceLocationsOptions['operation'],
  status: Uint32Array
): string {
  if (operation === 'p-center' || operation === 'capacitated-p-median') {
    if (status[1] !== 1) return 'invalid inputs';
    if (status[0] !== 0) return `partial result; ${status[0]} unassigned rows`;
    return operation === 'p-center'
      ? status[2] === 1
        ? 'exact p-center over 56 subsets'
        : 'feasible result; exactness certificate unavailable'
      : 'feasible greedy assignment; order-sensitive';
  }
  if (status[3] !== 0) return `invalid inputs (${status[3]} invalid values)`;
  if (operation === 'set-covering') {
    return status[2] === 1
      ? 'coverage complete'
      : `partial coverage; ${status[1]} uncovered stations`;
  }
  if (status[2] !== 1) return `partial result; ${status[1]} uncovered stations`;
  return `${status[4]} swaps; ${status[5] === 1 ? 'local optimum' : 'swap limit reached'}`;
}

function isInvalidStatus(
  operation: BixiServiceLocationsOptions['operation'],
  status: Uint32Array
): boolean {
  return operation === 'p-center' || operation === 'capacitated-p-median'
    ? status[1] !== 1
    : status[3] !== 0;
}

function objectiveUnits(operation: BixiServiceLocationsOptions['operation']): string {
  if (operation === 'p-center') return 'maximum straight-line distance (km)';
  if (operation === 'maximum-coverage') return 'covered activity proxy (rides)';
  if (operation === 'set-covering')
    return 'selected sites (GPU mask count; scalar objective withheld)';
  return 'activity proxy × distance (ride-km)';
}

function swapStatus(status: Uint32Array): string {
  if (status.length < 7) return 'not applicable to bounded operations';
  if (status[6] === 1) return 'iteration limit reached; local optimum unverified';
  return status[5] === 1
    ? `local optimum after ${status[4]} swaps`
    : 'swap phase did not certify a local optimum';
}

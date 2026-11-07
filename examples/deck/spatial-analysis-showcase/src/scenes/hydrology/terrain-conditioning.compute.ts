// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, CommandEncoder} from '@luma.gl/core';
import type {Layer} from '@deck.gl/core';
import {
  getGPUTerrainFlowParameterValues,
  GPU_TERRAIN_FLOW_PARAMETER_LENGTH,
  GPUTerrainFlow,
  GPUTerrainBreach,
  GPUTerrainDepressionHierarchy,
  GPUTerrainFlowFractions,
  GPUTerrainBranchingFlow,
  GPUTerrainCatchmentSummary
} from '@luma-contributors/analysis/gpu-terrain';
import {
  GPUCommandGraph,
  type CompiledGPUCommandGraph,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {importGraphBuffer} from '../../engine/graph-buffers';
import {SpatialAnalysisPointLayer, SpatialAnalysisRasterLayer} from '../../engine/layers';
import {addKernelPass} from '../../engine/mode-kernels';
import {SpatialAnalysisResources} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import type {SceneContext, SceneInstance} from '../scene';
import {createCanyonGrid} from './b15-common';

export type TerrainConditioningOptions = {
  layer:
    | 'source'
    | 'filled'
    | 'breach'
    | 'fractions'
    | 'transport'
    | 'distance'
    | 'minimum-distance'
    | 'maximum-distance'
    | 'fractional-catchment'
    | 'hierarchy'
    | 'profile';
  routing: 'd-infinity' | 'mfd-freeman' | 'mfd-quinn';
  stage: number;
  opacity: number;
};

const ROUTINGS = ['d-infinity', 'mfd-freeman', 'mfd-quinn'] as const;
const STAGE_COUNT = 9;
const HIERARCHY_CAPACITY = 4096;
const HISTOGRAM_BINS = 16;
const MAXIMUM_BREACH_STEPS = 8;
const MAXIMUM_BREACH_CANDIDATES = 32;
const BREACH_MAXIMUM_CUT_DEPTH = 25;
const BREACH_ELEVATION_DECREMENT = 0.25;

function selectPitCandidates(elevation: Float32Array, width: number, height: number): Uint32Array {
  const candidates: number[] = [];
  for (let row = 1; row < height - 1; row++) {
    for (let column = 1; column < width - 1; column++) {
      const index = row * width + column;
      const value = elevation[index];
      if (!Number.isFinite(value)) continue;
      let minimum = true;
      for (let offsetY = -1; offsetY <= 1 && minimum; offsetY++) {
        for (let offsetX = -1; offsetX <= 1; offsetX++) {
          if (!offsetX && !offsetY) continue;
          if (elevation[(row + offsetY) * width + column + offsetX] < value) {
            minimum = false;
            break;
          }
        }
      }
      if (minimum) candidates.push(index);
    }
  }
  candidates.sort((first, second) => elevation[first] - elevation[second] || first - second);
  return Uint32Array.from(candidates.slice(0, MAXIMUM_BREACH_CANDIDATES));
}

export async function createTerrainConditioning(
  ctx: SceneContext<TerrainConditioningOptions>
): Promise<SceneInstance<TerrainConditioningOptions>> {
  const {device} = ctx;
  const grid = createCanyonGrid(ctx.datasets.get('grand-canyon-dem'), 32);
  if (grid.width !== 64 || grid.height !== 64)
    throw new Error(`Expected a 64 × 64 reduced DEM, received ${grid.width} × ${grid.height}`);
  const {width, height, cellCount} = grid;
  const resources = new SpatialAnalysisResources(device, 'terrain-conditioning');
  const elevation = resources.createBuffer('elevation', grid.elevation);
  const parameterValues = getGPUTerrainFlowParameterValues(
    {
      fillEpsilon: 0.01,
      streamThreshold: grid.groundCellSize[0] * grid.groundCellSize[1] * 12,
      flowExponent: 1.1
    },
    grid.rasterGrid
  );
  const flowParameters = resources.createParameterBuffer(
    'flow-settings',
    'float32',
    GPU_TERRAIN_FLOW_PARAMETER_LENGTH,
    parameterValues
  );
  const buffer = (id: string, format: 'float32' | 'uint32', length = cellCount) =>
    resources.createBuffer(id, Math.max(4, length * 4));
  const flowFilled = buffer('flow-filled', 'float32');
  const directions = buffer('flow-directions', 'uint32');
  const accumulation = buffer('flow-accumulation', 'float32');
  const streams = buffer('flow-streams', 'uint32');
  const flowFillStatus = buffer('flow-fill-status', 'uint32', 1);
  const flowFlatStatus = buffer('flow-flat-status', 'uint32', 1);
  const flowAccumulationStatus = buffer('flow-accumulation-status', 'uint32', 1);
  const breachElevation = buffer('breached-elevation', 'float32');
  const breachDepth = buffer('breach-cut-depth', 'float32');
  const breachMask = buffer('breach-mask', 'uint32');
  const pitCellsCpu = selectPitCandidates(grid.elevation, width, height);
  const pitCells = resources.createBuffer(
    'pit-candidates',
    pitCellsCpu.length ? pitCellsCpu : Uint32Array.of(0)
  );
  const pathCells = buffer(
    'breach-path-cells',
    'uint32',
    pitCellsCpu.length * (MAXIMUM_BREACH_STEPS + 1)
  );
  const pathLengths = buffer('breach-path-lengths', 'uint32', pitCellsCpu.length);
  const pitStatus = buffer('breach-pit-status', 'uint32', pitCellsCpu.length);
  const breachUnresolved = buffer('breach-unresolved', 'uint32', 1);
  const hierarchyOutletsCpu = new Uint32Array(cellCount);
  for (let row = 0; row < height; row++)
    for (let column = 0; column < width; column++)
      hierarchyOutletsCpu[row * width + column] = Number(
        row === 0 || column === 0 || row === height - 1 || column === width - 1
      );
  const hierarchyOutlets = resources.createBuffer(
    'hierarchy-boundary-outlets',
    hierarchyOutletsCpu
  );
  let minimumElevation = Infinity;
  let maximumElevation = -Infinity;
  let maximumInteriorCell = -1;
  let lowestBoundaryCell = -1;
  for (let cell = 0; cell < cellCount; cell++) {
    const value = grid.elevation[cell];
    if (!Number.isFinite(value)) continue;
    minimumElevation = Math.min(minimumElevation, value);
    maximumElevation = Math.max(maximumElevation, value);
    const row = Math.floor(cell / width);
    const column = cell % width;
    if (
      row > 0 &&
      column > 0 &&
      row < height - 1 &&
      column < width - 1 &&
      (maximumInteriorCell < 0 || value > grid.elevation[maximumInteriorCell])
    )
      maximumInteriorCell = cell;
    if (
      (row === 0 || column === 0 || row === height - 1 || column === width - 1) &&
      (lowestBoundaryCell < 0 || value < grid.elevation[lowestBoundaryCell])
    )
      lowestBoundaryCell = cell;
  }
  const formatGridCell = (cell: number): string =>
    `row ${Math.floor(cell / width) + 1}, column ${(cell % width) + 1}`;
  const stagesCpu = Float32Array.from(
    {length: STAGE_COUNT},
    (_, index) => minimumElevation + index * 200
  );
  const stages = resources.createBuffer('hierarchy-stages', stagesCpu);
  const hierarchyNodes = buffer('hierarchy-nodes', 'uint32', HIERARCHY_CAPACITY * 4);
  const hierarchyLevels = buffer('hierarchy-levels', 'float32', HIERARCHY_CAPACITY * 2);
  const hierarchyOwnership = buffer('hierarchy-cell-owners', 'uint32');
  const invalidHierarchy = resources.createBuffer(
    'invalid-hierarchy',
    Uint32Array.from({length: cellCount}, () => 0xffffffff)
  );
  const hierarchyCurves = buffer(
    'hierarchy-stage-curves',
    'float32',
    HIERARCHY_CAPACITY * STAGE_COUNT * 2
  );
  const hierarchyStatus = buffer('hierarchy-status', 'uint32', 3);
  const fractionOutputs = new Map<
    string,
    {
      fractions: Buffer;
      edgeLengths: Buffer;
      maximumShare: Buffer;
      status: Buffer;
      unresolved: Buffer;
    }
  >();
  const branchOutputs = new Map<string, {values: Buffer; status: Buffer; unresolved: Buffer}>();
  const branchingUnresolvedBuffers: Buffer[] = [];
  const basinIdsCpu = Uint32Array.from(grid.elevation, value =>
    Number.isFinite(value) ? 0 : 0xffffffff
  );
  const basinIds = resources.createBuffer('window-basin-ids', basinIdsCpu);
  const metricGrid = {
    width,
    height,
    affine: [
      grid.groundCellSize[0],
      0,
      grid.bounds[0],
      0,
      -grid.groundCellSize[1],
      grid.bounds[3]
    ] as const,
    pixelInterpretation: 'area' as const
  };
  const catchmentStatistics = buffer('catchment-statistics', 'float32', 6);
  const catchmentHypsometry = buffer('catchment-hypsometry', 'float32', HISTOGRAM_BINS);
  const catchmentProfiles = buffer('catchment-profiles', 'float32', cellCount * 3);
  const profileStatus = buffer('catchment-profile-status', 'uint32');

  const graph = new GPUCommandGraph<void>(device, {id: 'terrain-conditioning'});
  const flowParameterView = flowParameters.importToGraph(graph);
  const views = new Map<string, GraphDataView<'float32' | 'uint32'>>();
  const view = <Format extends 'float32' | 'uint32'>(
    id: string,
    source: Buffer,
    format: Format,
    length = cellCount
  ): GraphDataView<Format> => {
    const existing = views.get(id);
    if (existing) return existing as GraphDataView<Format>;
    const created = importGraphBuffer(graph, id, source, format, length);
    views.set(id, created);
    return created;
  };
  const elevationView = view('source-elevation', elevation, 'float32');
  const filledView = view('filled-elevation', flowFilled, 'float32');
  const directionsView = view('flow-directions', directions, 'uint32');
  const streamsView = view('flow-streams', streams, 'uint32');
  graph.add(
    new GPUTerrainFlow({
      id: 'canonical-flow',
      grid: grid.rasterGrid,
      elevation: {
        id: 'elevation',
        format: 'float32',
        storage: {kind: 'buffer', values: elevationView}
      },
      parameters: flowParameterView,
      cellSizeMode: grid.cellSizeMode,
      fillDepressions: true,
      resolveFlats: true,
      flowRouting: 'd8',
      accumulationUnits: 'area',
      output: {
        filledElevation: filledView,
        flowDirections: directionsView,
        accumulation: view('accumulation', accumulation, 'float32'),
        streams: streamsView,
        status: {
          fillConverged: view('flow-status-fill', flowFillStatus, 'uint32', 1),
          flatsConverged: view('flow-status-flats', flowFlatStatus, 'uint32', 1),
          accumulationConverged: view(
            'flow-status-accumulation',
            flowAccumulationStatus,
            'uint32',
            1
          )
        }
      }
    })
  );
  graph.add(
    new GPUTerrainBreach({
      id: 'bounded-breach',
      grid: grid.rasterGrid,
      elevation: elevationView,
      pitCells: view('pit-cell-ids', pitCells, 'uint32', pitCellsCpu.length),
      maximumSteps: MAXIMUM_BREACH_STEPS,
      maximumCutDepth: BREACH_MAXIMUM_CUT_DEPTH,
      elevationDecrement: BREACH_ELEVATION_DECREMENT,
      output: {
        elevation: view('breached-elevation', breachElevation, 'float32'),
        cutDepth: view('breach-cut-depth', breachDepth, 'float32'),
        modificationMask: view('breach-mask', breachMask, 'uint32'),
        pathCells: view(
          'breach-path-cells',
          pathCells,
          'uint32',
          pitCellsCpu.length * (MAXIMUM_BREACH_STEPS + 1)
        ),
        pathLengths: view('breach-path-lengths', pathLengths, 'uint32', pitCellsCpu.length),
        pitStatus: view('breach-pit-status', pitStatus, 'uint32', pitCellsCpu.length),
        unresolvedCount: view('breach-unresolved', breachUnresolved, 'uint32', 1)
      }
    })
  );
  graph.add(
    new GPUTerrainDepressionHierarchy({
      id: 'depression-hierarchy',
      grid: metricGrid,
      elevation: elevationView,
      outlets: view('hierarchy-outlets', hierarchyOutlets, 'uint32'),
      stages: view('hierarchy-stages', stages, 'float32', STAGE_COUNT),
      nodeCapacity: HIERARCHY_CAPACITY,
      output: {
        nodes: view('hierarchy-nodes', hierarchyNodes, 'uint32', HIERARCHY_CAPACITY * 4),
        levels: view('hierarchy-levels', hierarchyLevels, 'float32', HIERARCHY_CAPACITY * 2),
        cellNodes: view('hierarchy-ownership', hierarchyOwnership, 'uint32'),
        curves: view(
          'hierarchy-curves',
          hierarchyCurves,
          'float32',
          HIERARCHY_CAPACITY * STAGE_COUNT * 2
        ),
        status: view('hierarchy-status', hierarchyStatus, 'uint32', 3)
      }
    })
  );

  for (const routing of ROUTINGS) {
    const fractions = buffer(`${routing}-fractions`, 'float32', cellCount * 8);
    const edgeLengths = buffer(`${routing}-edge-lengths`, 'float32', cellCount * 8);
    const maximumShare = buffer(`${routing}-maximum-share`, 'float32');
    const branchingValues = buffer(`${routing}-expected-distance`, 'float32');
    const branchingStatus = buffer(`${routing}-branching-status`, 'uint32');
    const branchingUnresolved = buffer(`${routing}-branching-unresolved`, 'uint32', 1);
    fractionOutputs.set(routing, {
      fractions,
      edgeLengths,
      maximumShare,
      status: branchingStatus,
      unresolved: branchingUnresolved
    });
    branchOutputs.set(routing, {
      values: branchingValues,
      status: branchingStatus,
      unresolved: branchingUnresolved
    });
    branchingUnresolvedBuffers.push(branchingUnresolved);
    const fractionsView = view(`${routing}-fractions`, fractions, 'float32', cellCount * 8);
    const edgeLengthsView = view(`${routing}-edge-lengths`, edgeLengths, 'float32', cellCount * 8);
    graph.add(
      new GPUTerrainFlowFractions({
        id: `${routing}-flow-fractions`,
        grid: grid.rasterGrid,
        elevation: filledView,
        flowDirections: directionsView,
        parameters: flowParameterView,
        routing,
        cellSizeMode: grid.cellSizeMode,
        output: {fractions: fractionsView, edgeLengths: edgeLengthsView}
      })
    );
    addKernelPass(graph, {
      id: `${routing}-maximum-outgoing-share`,
      invocationCount: cellCount,
      bindings: [
        {name: 'fractions', view: fractionsView, type: 'f32', access: 'read'},
        {
          name: 'maximumShare',
          view: view(`${routing}-maximum-share`, maximumShare, 'float32'),
          type: 'f32',
          access: 'read_write'
        }
      ],
      body: `var maximum = 0.0; var valid = true; var nanBits = 0x7fc00000u;
for (var direction = 0u; direction < 8u; direction++) { let fraction = fractions[fractionsOffset + index * 8u + direction]; if (fraction != fraction) { valid = false; } maximum = max(maximum, fraction); }
maximumShare[maximumShareOffset + index] = select(maximum, bitcast<f32>(nanBits), !valid);`
    });
    graph.add(
      new GPUTerrainBranchingFlow({
        id: `${routing}-expected-distance`,
        grid: grid.rasterGrid,
        fractions: fractionsView,
        values: view(
          `${routing}-zero-source`,
          resources.createBuffer(`${routing}-zero-source`, new Float32Array(cellCount)),
          'float32'
        ),
        edgeLengths: edgeLengthsView,
        mode: 'expected-distance',
        maximumIterations: 128,
        output: {
          values: view(`${routing}-distance`, branchingValues, 'float32'),
          status: view(`${routing}-branch-status`, branchingStatus, 'uint32'),
          unresolvedCount: view(`${routing}-branch-unresolved`, branchingUnresolved, 'uint32', 1)
        }
      })
    );
  }

  // The single dense label describes the finite analysis window. It is deliberately not presented as an outlet catchment.
  const dInfinityOutputs = fractionOutputs.get('d-infinity')!;
  const flowStatusProfiles = profileStatus;
  const transportValues = resources.createBuffer('transport-output', cellCount * 4);
  const transportStatuses = resources.createBuffer('transport-status', cellCount * 4);
  const minimumDistanceValues = resources.createBuffer('minimum-distance-output', cellCount * 4);
  const minimumDistanceStatuses = resources.createBuffer('minimum-distance-status', cellCount * 4);
  const maximumDistanceValues = resources.createBuffer('maximum-distance-output', cellCount * 4);
  const maximumDistanceStatuses = resources.createBuffer('maximum-distance-status', cellCount * 4);
  const fractionalCatchmentValues = resources.createBuffer(
    'fractional-catchment-output',
    cellCount * 4
  );
  const fractionalCatchmentStatuses = resources.createBuffer(
    'fractional-catchment-status',
    cellCount * 4
  );
  const transportUnresolved = resources.createBuffer('transport-unresolved', 4);
  const minimumDistanceUnresolved = resources.createBuffer('minimum-distance-unresolved', 4);
  const maximumDistanceUnresolved = resources.createBuffer('maximum-distance-unresolved', 4);
  const fractionalCatchmentUnresolved = resources.createBuffer(
    'fractional-catchment-unresolved',
    4
  );
  branchingUnresolvedBuffers.push(
    transportUnresolved,
    minimumDistanceUnresolved,
    maximumDistanceUnresolved,
    fractionalCatchmentUnresolved
  );
  const transportInjection = new Float32Array(cellCount);
  if (maximumInteriorCell >= 0) transportInjection[maximumInteriorCell] = 1;
  const transportSource = resources.createBuffer('window-transport-injection', transportInjection);
  const getCellCenter = (cell: number): Float32Array => {
    const row = Math.floor(cell / width);
    const column = cell % width;
    return Float32Array.of(
      grid.bounds[0] + (column + 0.5) * grid.displayCellSize[0],
      grid.bounds[3] - (row + 0.5) * grid.displayCellSize[1]
    );
  };
  const transportMarker = resources.createBuffer(
    'transport-source-marker',
    getCellCenter(maximumInteriorCell)
  );
  const outletMarker = resources.createBuffer(
    'fractional-outlet-marker',
    getCellCenter(lowestBoundaryCell)
  );
  graph.add(
    new GPUTerrainBranchingFlow({
      id: 'transport-flow',
      grid: grid.rasterGrid,
      fractions: view('d-infinity-fractions', dInfinityOutputs.fractions, 'float32', cellCount * 8),
      values: view('transport-injection', transportSource, 'float32'),
      edgeLengths: view(
        'd-infinity-edge-lengths',
        dInfinityOutputs.edgeLengths,
        'float32',
        cellCount * 8
      ),
      mode: 'transport',
      maximumIterations: 128,
      output: {
        values: view('transport-output', transportValues, 'float32'),
        status: view('transport-status', transportStatuses, 'uint32'),
        unresolvedCount: view('transport-unresolved', transportUnresolved, 'uint32', 1)
      }
    })
  );
  graph.add(
    new GPUTerrainBranchingFlow({
      id: 'minimum-distance-flow',
      grid: grid.rasterGrid,
      fractions: view('d-infinity-fractions', dInfinityOutputs.fractions, 'float32', cellCount * 8),
      values: view(
        'distance-seeds',
        resources.createBuffer('distance-seeds', new Float32Array(cellCount)),
        'float32'
      ),
      edgeLengths: view(
        'd-infinity-edge-lengths',
        dInfinityOutputs.edgeLengths,
        'float32',
        cellCount * 8
      ),
      mode: 'minimum-distance',
      maximumIterations: 128,
      output: {
        values: view('minimum-distance-output', minimumDistanceValues, 'float32'),
        status: view('minimum-distance-status', minimumDistanceStatuses, 'uint32'),
        unresolvedCount: view('minimum-distance-unresolved', minimumDistanceUnresolved, 'uint32', 1)
      }
    })
  );
  graph.add(
    new GPUTerrainBranchingFlow({
      id: 'maximum-distance-flow',
      grid: grid.rasterGrid,
      fractions: view('d-infinity-fractions', dInfinityOutputs.fractions, 'float32', cellCount * 8),
      values: view(
        'maximum-distance-seeds',
        resources.createBuffer('maximum-distance-seeds', new Float32Array(cellCount)),
        'float32'
      ),
      edgeLengths: view(
        'd-infinity-edge-lengths',
        dInfinityOutputs.edgeLengths,
        'float32',
        cellCount * 8
      ),
      mode: 'maximum-distance',
      maximumIterations: 128,
      output: {
        values: view('maximum-distance-output', maximumDistanceValues, 'float32'),
        status: view('maximum-distance-status', maximumDistanceStatuses, 'uint32'),
        unresolvedCount: view('maximum-distance-unresolved', maximumDistanceUnresolved, 'uint32', 1)
      }
    })
  );
  graph.add(
    new GPUTerrainBranchingFlow({
      id: 'fractional-catchment-flow',
      grid: grid.rasterGrid,
      fractions: view('d-infinity-fractions', dInfinityOutputs.fractions, 'float32', cellCount * 8),
      values: view(
        'fractional-outlet-membership',
        resources.createBuffer(
          'fractional-outlet-membership',
          Float32Array.from({length: cellCount}, (_, index) =>
            index === lowestBoundaryCell ? 1 : 0
          )
        ),
        'float32'
      ),
      edgeLengths: view(
        'd-infinity-edge-lengths',
        dInfinityOutputs.edgeLengths,
        'float32',
        cellCount * 8
      ),
      mode: 'fractional-catchment',
      maximumIterations: 128,
      output: {
        values: view('fractional-catchment-output', fractionalCatchmentValues, 'float32'),
        status: view('fractional-catchment-status', fractionalCatchmentStatuses, 'uint32'),
        unresolvedCount: view(
          'fractional-catchment-unresolved',
          fractionalCatchmentUnresolved,
          'uint32',
          1
        )
      }
    })
  );
  graph.add(
    new GPUTerrainCatchmentSummary({
      id: 'window-catchment-summary',
      grid: metricGrid,
      elevation: filledView,
      basinIds: view('dense-window-basin-ids', basinIds, 'uint32'),
      basinCount: 1,
      streams: streamsView,
      flowDirections: directionsView,
      histogramBins: HISTOGRAM_BINS,
      output: {
        statistics: view('catchment-statistics', catchmentStatistics, 'float32', 6),
        hypsometry: view('catchment-hypsometry', catchmentHypsometry, 'float32', HISTOGRAM_BINS),
        profiles: view('catchment-profiles', catchmentProfiles, 'float32', cellCount * 3),
        profileStatus: view('profile-status', flowStatusProfiles, 'uint32')
      }
    })
  );

  const compiled: CompiledGPUCommandGraph<void> = resources.track(await graph.compileAsync());
  const summaryBuffers = [
    pitStatus,
    breachUnresolved,
    hierarchyStatus,
    flowFillStatus,
    flowFlatStatus,
    flowAccumulationStatus,
    ...branchingUnresolvedBuffers,
    catchmentStatistics,
    catchmentHypsometry
  ];
  summaryBuffers.push(hierarchyNodes, hierarchyCurves);
  const summarySizes = [
    pitStatus.byteLength,
    4,
    12,
    4,
    4,
    4,
    ...branchingUnresolvedBuffers.map(() => 4),
    24,
    HISTOGRAM_BINS * 4,
    HIERARCHY_CAPACITY * 16,
    HIERARCHY_CAPACITY * STAGE_COUNT * 8
  ];
  let summaryText = 'Waiting for GPU summary…';
  let hierarchyNodeCount = 0;
  let hierarchyStageValues: Array<{area: number; volume: number}> = [];
  let hierarchyOverflowed = false;
  let branchingFailures: number[] = [];
  let hypsometryChart = {
    kind: 'line' as const,
    title: 'GPU catchment hypsometry',
    xLabel: 'Elevation (m)',
    yLabel: 'Cumulative area (km²)',
    series: [
      {
        label: 'Cumulative area',
        x: new Float32Array(HISTOGRAM_BINS),
        y: new Float32Array(HISTOGRAM_BINS),
        area: true
      }
    ],
    description: 'Cumulative valid area by elevation bin from the GPU catchment summary.'
  };
  let accepted = false;
  const updateHierarchyReadout = () => {
    const selectedStage = Math.max(0, Math.min(STAGE_COUNT - 1, Math.round(ctx.options.stage)));
    const selected = hierarchyStageValues[selectedStage];
    ctx.setReadout(
      'hierarchyStatus',
      hierarchyOverflowed
        ? 'capacity overflow: hierarchy output invalid'
        : selected
          ? `${hierarchyNodeCount} hierarchy nodes · stage ${stagesCpu[selectedStage].toFixed(3)} m · root area ${(selected.area / 1e6).toFixed(3)} km² · volume ${(selected.volume / 1e6).toFixed(3)} million m³`
          : `Hierarchy unavailable · stage ${stagesCpu[selectedStage].toFixed(3)} m`
    );
    if (hierarchyOverflowed) {
      ctx.setChart('hierarchyChart', null);
    } else {
      ctx.setChart('hierarchyChart', {
        kind: 'line',
        title: 'Sublevel-set area and volume',
        xLabel: 'Absolute elevation (m)',
        yLabel: 'Area (km²)',
        y2Label: 'Geometric volume (million m³)',
        y2Domain: [0, Math.max(1, ...hierarchyStageValues.map(value => value.volume / 1e6))],
        series: [
          {
            label: 'Area',
            x: stagesCpu,
            y: Float32Array.from(hierarchyStageValues, value => value.area / 1e6),
            color: 0
          },
          {
            label: 'Volume',
            x: stagesCpu,
            y: Float32Array.from(hierarchyStageValues, value => value.volume / 1e6),
            color: 1,
            axis: 'y2'
          }
        ],
        markers: [{x: stagesCpu[selectedStage], label: 'selected query'}],
        description:
          'GPU hierarchy root curves across absolute elevation queries. Geometric sublevel-set volume is not connected inundation.'
      });
    }
  };
  const updateProductReadout = () => {
    const routingIndex = ROUTINGS.indexOf(ctx.options.routing);
    const product = ctx.options.layer;
    if (product === 'distance') {
      ctx.setReadout(
        'productStatus',
        `Expected route distance · ${ctx.options.routing} · ${branchingFailures[routingIndex] ?? 'pending'} unresolved cells`
      );
    } else if (product === 'transport') {
      ctx.setReadout(
        'productStatus',
        `D-infinity transport from high interior ${formatGridCell(maximumInteriorCell)} · ${branchingFailures[3] ?? 'pending'} unresolved cells`
      );
    } else if (product === 'minimum-distance') {
      ctx.setReadout(
        'productStatus',
        `D-infinity minimum route distance · ${branchingFailures[4] ?? 'pending'} unresolved cells`
      );
    } else if (product === 'maximum-distance') {
      ctx.setReadout(
        'productStatus',
        `D-infinity maximum route distance · ${branchingFailures[5] ?? 'pending'} unresolved cells`
      );
    } else if (product === 'fractional-catchment') {
      ctx.setReadout(
        'productStatus',
        `D-infinity membership from lowest boundary ${formatGridCell(lowestBoundaryCell)} · ${branchingFailures[6] ?? 'pending'} unresolved cells`
      );
    } else if (product === 'fractions') {
      ctx.setReadout(
        'productStatus',
        `${ctx.options.routing} fractions on canonical filled surface`
      );
    } else {
      ctx.setReadout('productStatus', `${product} surface displayed`);
    }
  };
  const reader = new SummaryReader(
    resources,
    'terrain-conditioning',
    summaryBuffers.map((buffer, index) => ({buffer, size: summarySizes[index]})),
    bytes => {
      const words = new Uint32Array(bytes);
      let offset = 0;
      const pitCount = pitCellsCpu.length;
      let breached = 0;
      for (let index = 0; index < pitCount; index++)
        breached += Number(words[offset + index] === 1);
      offset += Math.max(1, pitCount);
      const breachUnresolvedCount = words[offset++];
      const storedNodes = words[offset++];
      const requiredNodes = words[offset++];
      const hierarchyOverflow = words[offset++];
      const fillOk = words[offset++] === 1;
      const flatsOk = words[offset++] === 1;
      const accumulationOk = words[offset++] === 1;
      branchingFailures = [];
      for (let index = 0; index < branchingUnresolvedBuffers.length; index++)
        branchingFailures.push(words[offset++]);
      const statisticsOffset = offset;
      const stats = new Float32Array(bytes, statisticsOffset * 4, 6);
      const histogram = new Float32Array(bytes, (statisticsOffset + 6) * 4, HISTOGRAM_BINS);
      const cumulativeAreas = Float32Array.from(histogram, area =>
        Number.isFinite(area) ? area / 1e6 : 0
      );
      const histogramElevations = Float32Array.from(
        {length: HISTOGRAM_BINS},
        (_, bin) => stats[2] + ((stats[3] - stats[2]) * (bin + 1)) / HISTOGRAM_BINS
      );
      hypsometryChart = {
        kind: 'line',
        title: 'GPU catchment hypsometry',
        xLabel: 'Elevation (m)',
        yLabel: 'Cumulative area (km²)',
        series: [
          {label: 'Cumulative area', x: histogramElevations, y: cumulativeAreas, area: true}
        ],
        description: `Cumulative valid area across ${HISTOGRAM_BINS} elevation bins, computed by GPUTerrainCatchmentSummary.`
      };
      const hierarchyNodesOffset = statisticsOffset + 6 + HISTOGRAM_BINS;
      const nodeValues = new Uint32Array(bytes, hierarchyNodesOffset * 4, HIERARCHY_CAPACITY * 4);
      const curvesOffset = hierarchyNodesOffset + HIERARCHY_CAPACITY * 4;
      const curveValues = new Float32Array(
        bytes,
        curvesOffset * 4,
        HIERARCHY_CAPACITY * STAGE_COUNT * 2
      );
      const roots: number[] = [];
      hierarchyNodeCount = hierarchyOverflow ? 0 : storedNodes;
      if (!hierarchyOverflow)
        for (let node = 0; node < storedNodes; node++) {
          if (nodeValues[node * 4] === 0xffffffff && nodeValues[node * 4 + 3] === 1)
            roots.push(node);
        }
      hierarchyStageValues = Array.from({length: STAGE_COUNT}, (_, stage) => {
        let area = 0;
        let volume = 0;
        for (const root of roots) {
          const curveIndex = (root * STAGE_COUNT + stage) * 2;
          area += curveValues[curveIndex];
          volume += curveValues[curveIndex + 1];
        }
        return {area, volume};
      });
      summaryText = `window area ${(stats[0] / 1e6).toFixed(3)} km² · final hypsometry bin ${cumulativeAreas[HISTOGRAM_BINS - 1].toFixed(3)} km² · elevation ${stats[2].toFixed(3)}–${stats[3].toFixed(3)} m · stream length ${(stats[4] / 1000).toFixed(3)} km · density ${(stats[5] * 1000).toFixed(3)} km/km²`;
      hierarchyOverflowed = Boolean(hierarchyOverflow);
      ctx.setReadout(
        'breachStatus',
        `${breached}/${pitCount} candidates breached · ${breachUnresolvedCount} unresolved · bound ${MAXIMUM_BREACH_STEPS} steps / ${BREACH_MAXIMUM_CUT_DEPTH} m cut`
      );
      if (hierarchyOverflow)
        ctx.setReadout(
          'hierarchyStatus',
          `capacity overflow: required ${requiredNodes}, stored ${storedNodes}; hierarchy invalid`
        );
      else updateHierarchyReadout();
      ctx.setReadout(
        'flowStatus',
        `fill ${fillOk ? 'converged' : 'unresolved'} · flats ${flatsOk ? 'converged' : 'unresolved'} · accumulation ${accumulationOk ? 'converged' : 'unresolved'} · unresolved cells [D∞ expected, Freeman expected, Quinn expected, transport, minimum distance, maximum distance, fractional catchment]: ${branchingFailures.join(', ')}`
      );
      ctx.setReadout('catchmentSummary', summaryText);
      ctx.setChart('hypsometryChart', hypsometryChart);
      updateProductReadout();
      accepted = true;
      ctx.requestLayers();
    }
  );
  ctx.setReadout(
    'grid',
    `${width} × ${height} cells · ${Math.round(grid.groundCellSize[0])} m nominal cell · centre-row planar approximation`
  );
  ctx.setReadout(
    'pitCandidates',
    `${pitCellsCpu.length} interior minima selected from the 64 × 64 block-average (cap ${MAXIMUM_BREACH_CANDIDATES})`
  );
  ctx.setReadout('breachStatus', 'Waiting for GPU summary…');
  ctx.setReadout('hierarchyStatus', 'Waiting for GPU summary…');
  ctx.setReadout('flowStatus', 'Waiting for GPU summary…');
  ctx.setReadout('catchmentSummary', 'Waiting for GPU summary…');
  ctx.setReadout('productStatus', 'Waiting for GPU summary…');
  ctx.setChart('hierarchyChart', null);
  return {
    getCompiledGraphs: () => [compiled],
    encode(commandEncoder: CommandEncoder, frame) {
      if (!frame.frameIndex || frame.frameIndex < 2)
        compiled.encode(commandEncoder, {parameters: undefined});
      if (!accepted && !reader.isPending) reader.request(commandEncoder);
      reader.flush(commandEncoder);
    },
    getLayers() {
      const options = ctx.options;
      const fraction = fractionOutputs.get(options.routing)!;
      const branch = branchOutputs.get(options.routing)!;
      const layerValues: Record<TerrainConditioningOptions['layer'], Buffer> = {
        source: elevation,
        filled: flowFilled,
        breach: breachDepth,
        fractions: fraction.maximumShare,
        transport: transportValues,
        distance: branch.values,
        'minimum-distance': minimumDistanceValues,
        'maximum-distance': maximumDistanceValues,
        'fractional-catchment': fractionalCatchmentValues,
        hierarchy: hierarchyOverflowed ? invalidHierarchy : hierarchyOwnership,
        profile: profileStatus
      };
      const formats: Record<TerrainConditioningOptions['layer'], 'float32' | 'uint32'> = {
        source: 'float32',
        filled: 'float32',
        breach: 'float32',
        fractions: 'float32',
        transport: 'float32',
        distance: 'float32',
        'minimum-distance': 'float32',
        'maximum-distance': 'float32',
        'fractional-catchment': 'float32',
        hierarchy: 'uint32',
        profile: 'uint32'
      };
      const ranges: Record<TerrainConditioningOptions['layer'], readonly [number, number]> = {
        source: [700, 2700],
        filled: [700, 2700],
        breach: [0, BREACH_MAXIMUM_CUT_DEPTH],
        fractions: [0, 1],
        transport: [0, 1],
        distance: [0, 50000],
        'minimum-distance': [0, 50000],
        'maximum-distance': [0, 50000],
        'fractional-catchment': [0, 1],
        hierarchy: [0, HIERARCHY_CAPACITY],
        profile: [0, 2]
      };
      const ramps: Record<TerrainConditioningOptions['layer'], 'isolum' | 'cividis' | 'category'> =
        {
          source: 'isolum',
          filled: 'isolum',
          breach: 'cividis',
          fractions: 'cividis',
          transport: 'cividis',
          distance: 'cividis',
          'minimum-distance': 'cividis',
          'maximum-distance': 'cividis',
          'fractional-catchment': 'cividis',
          hierarchy: 'category',
          profile: 'category'
        };
      const layers: Layer[] = [];
      if (options.layer === 'profile') {
        layers.push(
          new SpatialAnalysisRasterLayer({
            id: 'terrain-conditioning-profile-elevation-context',
            coordinateOrigin: [grid.origin[0], grid.origin[1], 0],
            gridSize: [width, height],
            bounds: grid.bounds,
            rowOrigin: 'north',
            values: elevation,
            valueFormat: 'float32',
            colormap: 'grayscale',
            valueRange: [700, 2700],
            opacity: 0.45,
            noDataColor: [0, 0, 0, 0]
          })
        );
      }
      layers.push(
        new SpatialAnalysisRasterLayer({
          id: `terrain-conditioning-${options.layer}`,
          coordinateOrigin: [grid.origin[0], grid.origin[1], 0],
          gridSize: [width, height],
          bounds: grid.bounds,
          rowOrigin: 'north',
          values: layerValues[options.layer],
          valueFormat: formats[options.layer],
          colormap: ramps[options.layer],
          palette: [
            [65, 105, 225, 255],
            [245, 195, 66, 255],
            [201, 64, 64, 255],
            [104, 170, 120, 255]
          ],
          valueRange: ranges[options.layer],
          opacity: options.opacity,
          noDataColor: [0, 0, 0, 0],
          ...(options.layer === 'hierarchy'
            ? {noDataValue: 0xffffffff}
            : options.layer === 'profile'
              ? {noDataValue: 0}
              : {})
        })
      );
      if (options.layer === 'transport' && maximumInteriorCell >= 0) {
        layers.push(
          new SpatialAnalysisPointLayer({
            id: 'terrain-transport-source-marker',
            coordinateOrigin: [grid.origin[0], grid.origin[1], 0],
            positions: transportMarker,
            instanceCount: 1,
            radiusPixels: 8,
            shape: 'star',
            color: [255, 206, 54, 255],
            outlineColor: [20, 25, 32, 255],
            outlineWidthPixels: 1.5
          })
        );
      }
      if (options.layer === 'fractional-catchment' && lowestBoundaryCell >= 0) {
        layers.push(
          new SpatialAnalysisPointLayer({
            id: 'terrain-fractional-outlet-marker',
            coordinateOrigin: [grid.origin[0], grid.origin[1], 0],
            positions: outletMarker,
            instanceCount: 1,
            radiusPixels: 8,
            shape: 'square',
            color: [248, 75, 70, 255],
            outlineColor: [20, 25, 32, 255],
            outlineWidthPixels: 1.5
          })
        );
      }
      return layers;
    },
    setOption(id) {
      if (id === 'stage' && accepted) updateHierarchyReadout();
      if (accepted) updateProductReadout();
      ctx.requestLayers();
    },
    getTooltip(event) {
      if (!event.coordinate) return null;
      return accepted
        ? `Terrain products use a ${width} × ${height} block-averaged grid. ${summaryText}`
        : 'GPU terrain summaries are running…';
    },
    destroy() {
      reader.stop();
      resources.destroy();
    }
  };
}

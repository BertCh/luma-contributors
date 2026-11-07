// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Layer} from '@deck.gl/core';
import type {Buffer} from '@luma.gl/core';
import {
  getGPUTerrainDerivativesParameterValues,
  getGPUTerrainFlowParameterValues,
  getGPUTerrainHydrologicIndicesParameterValues,
  GPU_TERRAIN_DERIVATIVES_PARAMETER_LENGTH,
  GPU_TERRAIN_FLOW_PARAMETER_LENGTH,
  GPU_TERRAIN_HYDROLOGIC_INDICES_PARAMETER_LENGTH,
  GPUTerrainDerivatives,
  GPUTerrainFlow,
  GPUTerrainFlowDistance,
  GPUTerrainHeightAboveDrainage,
  GPUTerrainHydrologicIndices,
  GPUTerrainStreamOrder,
  GPUTerrainStreamNetwork,
  GPUTerrainWatersheds,
  type GPUTerrainFlowRouting
} from '@luma-contributors/analysis/gpu-terrain';
import {
  GPUCommandGraph,
  GPUHistogram,
  type CompiledGPUCommandGraph,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {importGraphBuffer} from '../../engine/graph-buffers';
import {GraphSwap} from '../../engine/graph-swap';
import {
  SpatialAnalysisPointLayer,
  SpatialAnalysisRasterLayer,
  SpatialAnalysisSegmentLayer
} from '../../engine/layers';
import {addKernelPass} from '../../engine/mode-kernels';
import {SpatialAnalysisResources} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import type {SceneContext, SceneInstance} from '../scene';
import {
  addFillDepthPass,
  addLogDisplayPass,
  addStreamSegmentPass,
  createCanyonGrid,
  formatInteger,
  formatKilometers,
  getCellAt,
  getCellCenter,
  HIDDEN_VALUE,
  NAN_DECLARATIONS
} from './b15-common';
import {addBasinClassPass} from '../../../../spatial-analysis-shared/terrain-passes';
import {FILL_EPSILONS} from '../../../../spatial-analysis-shared/layer-constants';
import {BASIN_PALETTE} from '../../../../spatial-analysis-shared/layer-constants';
import {ORDER_PALETTE} from '../../../../spatial-analysis-shared/layer-constants';
import {ORDER_WIDTHS} from '../../../../spatial-analysis-shared/layer-constants';
import {UNRESOLVED_COLOR} from '../../../../spatial-analysis-shared/layer-constants';
export {ORDER_PALETTE} from '../../../../spatial-analysis-shared/layer-constants';
export {BASIN_PALETTE} from '../../../../spatial-analysis-shared/layer-constants';
export {FILL_EPSILONS} from '../../../../spatial-analysis-shared/layer-constants';

/** Option state of the watersheds scene. */
export type WatershedsOptions = {
  product:
    | 'accumulation'
    | 'fill'
    | 'watersheds'
    | 'stream-order'
    | 'flow-distance'
    | 'network'
    | 'hand'
    | 'flood'
    | 'indices';
  routing: GPUTerrainFlowRouting;
  resolveFlats: boolean;
  flowExponent: number;
  streamArea: number;
  fillEpsilon: number;
  showStreams: boolean;
  basinMode: 'pour-points' | 'outlets';
  handRange: number;
  floodStage: number;
  indexKind: 'wetness' | 'catchment' | 'power';
  minimumSlope: number;
  overlayOpacity: number;
};

/** Grid reduction: 1 keeps the 15 m DEM, 2 averages 2 x 2 blocks to 31 m. */
export const WATERSHEDS_GRID_STRIDE = 1;
/** Compile-time iteration limits of the relaxations. */
const MAXIMUM_FILL_ITERATIONS = 256;
const MAXIMUM_FLAT_ITERATIONS = 256;
const MAXIMUM_ACCUMULATION_ITERATIONS = 128;
const SUN_AZIMUTH_DEGREES = 315;
const SUN_ALTITUDE_DEGREES = 40;
const POUR_POINT_CAPACITY = 8;
const UNUSED_POUR_POINT = 0xffffffff;
/** Cells searched around a click for the cell of highest accumulation. */
const POUR_POINT_SNAP_RADIUS = 5;
const ORDER_BIN_COUNT = 8;

/** Pour points the story starts from: Bright Angel Creek, Garden Creek and the Colorado. */
export const DEFAULT_POUR_POINTS: readonly {
  name: string;
  lngLat: [number, number];
  /** Cells searched for the stream, and the largest drainage area (km²) it may have. */
  snapRadius: number;
  maximumAreaKm2: number;
}[] = [
  {
    name: 'Phantom Ranch (Bright Angel Creek)',
    lngLat: [-112.0953, 36.107],
    snapRadius: 16,
    maximumAreaKm2: 300
  },
  {
    name: 'Indian Garden (Garden Creek)',
    lngLat: [-112.1262, 36.0771],
    snapRadius: 24,
    maximumAreaKm2: 120
  },
  {
    name: 'Colorado River at Bright Angel Creek',
    lngLat: [-112.093, 36.106],
    snapRadius: 16,
    maximumAreaKm2: Infinity
  }
];

/** Fixed color range of the log10 accumulation display, log10 of square meters. */
export const ACCUMULATION_LOG_RANGE: readonly [number, number] = [4, 8.8];
export const WETNESS_RANGE: readonly [number, number] = [2, 18];
export const CATCHMENT_LOG_RANGE: readonly [number, number] = [0.5, 3.5];
export const POWER_LOG_RANGE: readonly [number, number] = [-1, 3];

/**
 * Drainage of the central Grand Canyon on the GPU. One compiled graph chains `GPUTerrainFlow`
 * (fill, flat resolution, routing and accumulation, streams), `GPUTerrainHeightAboveDrainage`,
 * `GPUTerrainWatersheds` (outlet basins and pour-point watersheds), `GPUTerrainStreamOrder` and
 * `GPUTerrainHydrologicIndices`. Routing and flat resolution are compile-time and rebuild the graph;
 * everything else is a parameter-buffer write and the graph re-encodes only when an input changed.
 */
export async function createWatersheds(
  ctx: SceneContext<WatershedsOptions>
): Promise<SceneInstance<WatershedsOptions>> {
  const {device} = ctx;
  const grid = createCanyonGrid(ctx.datasets.get('grand-canyon-dem'), WATERSHEDS_GRID_STRIDE);
  const {width, height, cellCount} = grid;
  const origin: [number, number, number] = [grid.origin[0], grid.origin[1], 0];
  const cellAreaSquareMeters = grid.groundCellSize[0] * grid.groundCellSize[1];
  const resources = new SpatialAnalysisResources(device, 'watersheds');
  let destroyed = false;
  let extensionDirty = true;

  const float32Cells = (name: string) => resources.createBuffer(name, cellCount * 4);
  const elevationBuffer = resources.createBuffer('elevation', grid.elevation);
  const hillshadeBuffer = float32Cells('hillshade');
  const filledBuffer = float32Cells('filled');
  const directionsBuffer = float32Cells('directions');
  const accumulationBuffer = float32Cells('accumulation');
  const streamsBuffer = float32Cells('streams');
  const handBuffer = float32Cells('hand');
  const basinLabelsBuffer = float32Cells('basin-labels');
  const watershedLabelsBuffer = float32Cells('watershed-labels');
  const streamOrderBuffer = float32Cells('stream-order');
  const flowDistanceBuffer = float32Cells('flow-distance');
  const flowDistanceDisplayBuffer = float32Cells('flow-distance-display');
  const flowDistanceValidityBuffer = resources.createBuffer(
    'flow-distance-validity',
    cellCount * 4
  );
  const flowDistanceTerminalBuffer = resources.createBuffer(
    'flow-distance-terminals',
    cellCount * 4
  );
  const flowDistanceCostsBuffer = float32Cells('flow-distance-costs');
  const networkSourcesBuffer = resources.createBuffer('network-sources', cellCount * 4);
  const networkTargetsBuffer = resources.createBuffer('network-targets', cellCount * 4);
  const networkOffsetsBuffer = resources.createBuffer('network-offsets', (cellCount + 1) * 4);
  const networkPositionsBuffer = resources.createBuffer('network-positions', cellCount * 2 * 2 * 4);
  const networkEndpointPositionsBuffer = resources.createBuffer(
    'network-endpoint-positions',
    cellCount * 2 * 4
  );
  const networkLengthsBuffer = float32Cells('network-lengths');
  const networkReachCountBuffer = resources.createBuffer('network-reach-count', 4);
  const networkVertexCountBuffer = resources.createBuffer('network-vertex-count', 4);
  const networkOverflowBuffer = resources.createBuffer('network-overflow', 4);
  const networkUnresolvedBuffer = resources.createBuffer('network-unresolved', 4);
  const catchmentBuffer = float32Cells('catchment');
  const wetnessBuffer = float32Cells('wetness');
  const powerBuffer = float32Cells('power');
  const accumulationDisplayBuffer = float32Cells('accumulation-display');
  const indexDisplayBuffer = float32Cells('index-display');
  const basinClassBuffer = float32Cells('basin-class');
  const fillDepthBuffer = float32Cells('fill-depth');
  const floodDepthBuffer = float32Cells('flood-depth');
  const floodMaskBuffer = float32Cells('flood-mask');
  const segmentsBuffer = resources.createBuffer('stream-segments', cellCount * 16);
  const widthBuffers = ORDER_WIDTHS.map((_, index) => float32Cells(`stream-width-${index}`));
  const flagNames = ['fill', 'flats', 'accumulation', 'hand', 'basins', 'watersheds', 'order'];
  const flagBuffers = flagNames.map(name => resources.createBuffer(`${name}-converged`, 4));
  const iterationNames = ['fill', 'flats', 'accumulation', 'order'];
  const iterationBuffers = iterationNames.map(name =>
    resources.createBuffer(`${name}-iterations`, 4)
  );
  const streamCountsBuffer = resources.createBuffer('stream-counts', 8);
  const floodCountsBuffer = resources.createBuffer('flood-counts', 8);
  const orderHistogramBuffer = resources.createBuffer('order-histogram', ORDER_BIN_COUNT * 4);
  const watershedHistogramBuffer = resources.createBuffer(
    'watershed-histogram',
    POUR_POINT_CAPACITY * 4
  );
  const flowSettings = resources.createParameterBuffer(
    'flow-settings',
    'float32',
    GPU_TERRAIN_FLOW_PARAMETER_LENGTH
  );
  const indicesSettings = resources.createParameterBuffer(
    'indices-settings',
    'float32',
    GPU_TERRAIN_HYDROLOGIC_INDICES_PARAMETER_LENGTH
  );
  const derivativesSettings = resources.createParameterBuffer(
    'derivatives-settings',
    'float32',
    GPU_TERRAIN_DERIVATIVES_PARAMETER_LENGTH
  );
  const displayParameters = resources.createParameterBuffer('display-parameters', 'float32', 2);
  const pourPointValues = new Uint32Array(POUR_POINT_CAPACITY).fill(UNUSED_POUR_POINT);
  const pourPointBuffer = resources.createParameterBuffer(
    'pour-points',
    'uint32',
    POUR_POINT_CAPACITY,
    pourPointValues
  );
  const pourPointPositionsBuffer = resources.createBuffer(
    'pour-point-positions',
    POUR_POINT_CAPACITY * 8
  );
  const pourPointIndexBuffer = resources.createBuffer(
    'pour-point-indices',
    new Uint32Array(Array.from({length: POUR_POINT_CAPACITY}, (_, index) => index))
  );

  // --- Setup graph: hillshade backdrop, computed once -------------------------------------------
  const setupGraph = new GPUCommandGraph<void>(device, {id: 'watersheds-setup'});
  setupGraph.add(
    new GPUTerrainDerivatives({
      id: 'derivatives',
      grid: grid.rasterGrid,
      elevation: {
        // Terrarium mixes sources (3DEP, SRTM, Copernicus...): vertical datum unspecified, so verticalDatum stays undeclared.
        id: 'elevation',
        format: 'float32' as const,
        storage: {
          kind: 'buffer' as const,
          values: importGraphBuffer(setupGraph, 'elevation', elevationBuffer, 'float32', cellCount)
        }
      },
      parameters: derivativesSettings.importToGraph(setupGraph),
      output: {
        hillshade: importGraphBuffer(setupGraph, 'hillshade', hillshadeBuffer, 'float32', cellCount)
      },
      cellSizeMode: grid.cellSizeMode,
      rowDirection: 'south'
    })
  );
  const compiledSetup: CompiledGPUCommandGraph<void> = resources.track(
    await setupGraph.compileAsync()
  );

  /** Builds the drainage graph for one routing and flat-resolution choice (both compile-time). */
  async function compileDrainageGraph(
    flowRouting: GPUTerrainFlowRouting,
    resolveFlats: boolean
  ): Promise<CompiledGPUCommandGraph<void>> {
    const graph = new GPUCommandGraph<void>(device, {id: 'watersheds'});
    const imported = new Map<string, GraphDataView<'float32' | 'uint32' | 'float32x2'>>();
    const view = <Format extends 'float32' | 'uint32' | 'float32x2'>(
      id: string,
      buffer: Buffer,
      format: Format,
      length = cellCount
    ): GraphDataView<Format> => {
      let existing = imported.get(id);
      if (!existing) {
        existing = importGraphBuffer(graph, id, buffer, format, length);
        imported.set(id, existing);
      }
      return existing as GraphDataView<Format>;
    };
    const elevationView = view('elevation', elevationBuffer, 'float32');
    const filled = view('filled', filledBuffer, 'float32');
    const directions = view('directions', directionsBuffer, 'uint32');
    const accumulation = view('accumulation', accumulationBuffer, 'float32');
    const streams = view('streams', streamsBuffer, 'uint32');
    const streamOrder = view('stream-order', streamOrderBuffer, 'uint32');
    const flag = (index: number) =>
      view(`${flagNames[index]}-converged`, flagBuffers[index], 'uint32', 1);
    const iterations = (index: number) =>
      view(`${iterationNames[index]}-iterations`, iterationBuffers[index], 'uint32', 1);
    // The routing surface is the filled elevation; its NaNs mark invalid cells.
    const filledBand = {
      id: 'filled',
      format: 'float32' as const,
      storage: {kind: 'buffer' as const, values: filled}
    };

    graph.add(
      new GPUTerrainFlow({
        id: 'flow',
        grid: grid.rasterGrid,
        elevation: {
          id: 'elevation',
          format: 'float32' as const,
          storage: {kind: 'buffer' as const, values: elevationView}
        },
        parameters: flowSettings.importToGraph(graph),
        cellSizeMode: grid.cellSizeMode,
        fillDepressions: true,
        maximumFillIterations: MAXIMUM_FILL_ITERATIONS,
        resolveFlats,
        maximumFlatIterations: MAXIMUM_FLAT_ITERATIONS,
        flowRouting,
        maximumAccumulationIterations: MAXIMUM_ACCUMULATION_ITERATIONS,
        accumulationUnits: 'area',
        output: {
          filledElevation: filled,
          flowDirections: directions,
          accumulation,
          streams,
          status: {
            fillConverged: flag(0),
            fillIterationCount: iterations(0),
            ...(resolveFlats ? {flatsConverged: flag(1), flatsIterationCount: iterations(1)} : {}),
            accumulationConverged: flag(2),
            accumulationIterationCount: iterations(2)
          }
        }
      })
    );
    graph.add(
      new GPUTerrainHeightAboveDrainage({
        id: 'hand',
        grid: grid.rasterGrid,
        elevation: filledBand,
        flowDirections: directions,
        streams,
        output: {
          heightAboveDrainage: view('hand', handBuffer, 'float32'),
          status: {converged: flag(3)}
        }
      })
    );
    graph.add(
      new GPUTerrainWatersheds({
        id: 'basins',
        grid: grid.rasterGrid,
        flowDirections: directions,
        output: {
          labels: view('basin-labels', basinLabelsBuffer, 'uint32'),
          status: {converged: flag(4)}
        }
      })
    );
    graph.add(
      new GPUTerrainWatersheds({
        id: 'watersheds',
        grid: grid.rasterGrid,
        flowDirections: directions,
        pourPoints: pourPointBuffer.importToGraph(graph),
        output: {
          labels: view('watershed-labels', watershedLabelsBuffer, 'uint32'),
          status: {converged: flag(5)}
        }
      })
    );
    graph.add(
      new GPUTerrainStreamOrder({
        id: 'order',
        grid: grid.rasterGrid,
        flowDirections: directions,
        streams,
        output: {
          streamOrder,
          status: {converged: flag(6), iterationCount: iterations(3)}
        }
      })
    );
    graph.add(
      new GPUTerrainHydrologicIndices({
        id: 'indices',
        grid: grid.rasterGrid,
        elevation: filledBand,
        accumulation,
        parameters: indicesSettings.importToGraph(graph),
        cellSizeMode: grid.cellSizeMode,
        output: {
          specificCatchmentArea: view('catchment', catchmentBuffer, 'float32'),
          wetnessIndex: view('wetness', wetnessBuffer, 'float32'),
          streamPowerIndex: view('power', powerBuffer, 'float32')
        }
      })
    );
    // Display products ----------------------------------------------------------------------
    const displayView = displayParameters.importToGraph(graph);
    addLogDisplayPass(graph, {
      id: 'accumulation-display',
      cellCount,
      input: accumulation,
      output: view('accumulation-display', accumulationDisplayBuffer, 'float32'),
      useLog: true
    });
    addIndexDisplayPass(graph, {
      cellCount,
      wetness: view('wetness', wetnessBuffer, 'float32'),
      catchment: view('catchment', catchmentBuffer, 'float32'),
      power: view('power', powerBuffer, 'float32'),
      kind: displayView,
      output: view('index-display', indexDisplayBuffer, 'float32')
    });
    addFillDepthPass(graph, {
      id: 'fill-depth',
      cellCount,
      elevation: elevationView as GraphDataView<'float32'>,
      filled,
      output: view('fill-depth', fillDepthBuffer, 'float32')
    });
    addFloodPass(graph, {
      cellCount,
      hand: view('hand', handBuffer, 'float32'),
      stage: displayView,
      depth: view('flood-depth', floodDepthBuffer, 'float32'),
      mask: view('flood-mask', floodMaskBuffer, 'uint32')
    });
    addBasinClassPass(graph, {
      cellCount,
      labels: view('basin-labels', basinLabelsBuffer, 'uint32'),
      output: view('basin-class', basinClassBuffer, 'uint32')
    });
    addStreamSegmentPass(graph, {
      cellCount,
      grid,
      directions,
      streamOrder,
      segments: view('stream-segments', segmentsBuffer, 'float32', cellCount * 4),
      widths: widthBuffers.map((buffer, index) => view(`stream-width-${index}`, buffer, 'float32'))
    });
    // Summaries -----------------------------------------------------------------------------
    graph.add(
      new GPUHistogram({
        id: 'stream-count',
        input: streams,
        output: view('stream-counts', streamCountsBuffer, 'uint32', 2),
        domain: [0, 2]
      })
    );
    graph.add(
      new GPUHistogram({
        id: 'flood-count',
        input: view('flood-mask', floodMaskBuffer, 'uint32'),
        output: view('flood-counts', floodCountsBuffer, 'uint32', 2),
        domain: [0, 2]
      })
    );
    graph.add(
      new GPUHistogram({
        id: 'order-histogram',
        input: streamOrder,
        output: view('order-histogram', orderHistogramBuffer, 'uint32', ORDER_BIN_COUNT),
        domain: [0, ORDER_BIN_COUNT]
      })
    );
    graph.add(
      new GPUHistogram({
        id: 'watershed-histogram',
        input: view('watershed-labels', watershedLabelsBuffer, 'uint32'),
        output: view(
          'watershed-histogram',
          watershedHistogramBuffer,
          'uint32',
          POUR_POINT_CAPACITY
        ),
        domain: [0, POUR_POINT_CAPACITY]
      })
    );
    return await graph.compileAsync();
  }

  /** Optional product graphs keep path tracing and network extraction off the common render path. */
  async function compileExtensionGraph(
    product: 'flow-distance' | 'network'
  ): Promise<CompiledGPUCommandGraph<void>> {
    const graph = new GPUCommandGraph<void>(device, {id: `watersheds-${product}`});
    const directions = importGraphBuffer(
      graph,
      'directions',
      directionsBuffer,
      'uint32',
      cellCount
    );
    const streams = importGraphBuffer(graph, 'streams', streamsBuffer, 'uint32', cellCount);
    if (product === 'flow-distance') {
      const costs = importGraphBuffer(
        graph,
        'flow-distance-costs',
        flowDistanceCostsBuffer,
        'float32',
        cellCount
      );
      const distance = importGraphBuffer(
        graph,
        'flow-distance',
        flowDistanceBuffer,
        'float32',
        cellCount
      );
      const validity = importGraphBuffer(
        graph,
        'flow-distance-validity',
        flowDistanceValidityBuffer,
        'uint32',
        cellCount
      );
      const terminals = importGraphBuffer(
        graph,
        'flow-distance-terminals',
        flowDistanceTerminalBuffer,
        'uint32',
        cellCount
      );
      const display = importGraphBuffer(
        graph,
        'flow-distance-display',
        flowDistanceDisplayBuffer,
        'float32',
        cellCount
      );
      addFlowDistanceCostPass(graph, {
        cellCount,
        flowDirections: directions,
        output: costs,
        groundCellSize: grid.groundCellSize[0]
      });
      graph.add(
        new GPUTerrainFlowDistance({
          id: 'flow-distance',
          grid: grid.rasterGrid,
          flowDirections: directions,
          streams,
          transitionCosts: costs,
          output: {distance, validity, terminalCells: terminals}
        })
      );
      addFlowDistanceDisplayPass(graph, {
        cellCount,
        distance,
        validity,
        terminals,
        streams,
        output: display
      });
    } else {
      const sourceCells = importGraphBuffer(
        graph,
        'network-sources',
        networkSourcesBuffer,
        'uint32',
        cellCount
      );
      const reachCount = importGraphBuffer(
        graph,
        'network-reach-count',
        networkReachCountBuffer,
        'uint32',
        1
      );
      graph.add(
        new GPUTerrainStreamNetwork({
          id: 'stream-network',
          grid: grid.rasterGrid,
          flowDirections: directions,
          streams,
          output: {
            sourceCells,
            targetCells: importGraphBuffer(
              graph,
              'network-targets',
              networkTargetsBuffer,
              'uint32',
              cellCount
            ),
            vertexOffsets: importGraphBuffer(
              graph,
              'network-offsets',
              networkOffsetsBuffer,
              'uint32',
              cellCount + 1
            ),
            positions: importGraphBuffer(
              graph,
              'network-positions',
              networkPositionsBuffer,
              'float32x2',
              cellCount * 2
            ),
            lengths: importGraphBuffer(
              graph,
              'network-lengths',
              networkLengthsBuffer,
              'float32',
              cellCount
            ),
            reachCount,
            vertexCount: importGraphBuffer(
              graph,
              'network-vertex-count',
              networkVertexCountBuffer,
              'uint32',
              1
            ),
            status: {
              overflow: importGraphBuffer(
                graph,
                'network-overflow',
                networkOverflowBuffer,
                'uint32',
                1
              )
            },
            unresolvedCount: importGraphBuffer(
              graph,
              'network-unresolved',
              networkUnresolvedBuffer,
              'uint32',
              1
            )
          }
        })
      );
      addNetworkEndpointPositionsPass(graph, {
        capacity: cellCount,
        width,
        displayCellSize: grid.displayCellSize,
        bounds: grid.bounds,
        sourceCells,
        reachCount,
        output: importGraphBuffer(
          graph,
          'network-endpoint-positions',
          networkEndpointPositionsBuffer,
          'float32x2',
          cellCount
        )
      });
    }
    return await graph.compileAsync();
  }

  // --- State --------------------------------------------------------------------------------
  let builtRouting = ctx.options.routing;
  let builtResolveFlats = ctx.options.resolveFlats;
  let compiledDrainage = resources.track(
    await compileDrainageGraph(builtRouting, builtResolveFlats)
  );
  let drainageDirty = true;
  let networkReachCount = 0;
  const extensionGraphs = new Map<'flow-distance' | 'network', CompiledGPUCommandGraph<void>>();
  const extensionSwap = resources.track(new GraphSwap());
  const pourPoints: number[] = [];
  let defaultsSeeded = false;
  let landCellCount = cellCount;

  const getStreamThreshold = () => 10 ** ctx.options.streamArea * 1e6;
  const writeFlowSettings = () => {
    flowSettings.write(
      getGPUTerrainFlowParameterValues(
        {
          fillEpsilon: FILL_EPSILONS[ctx.options.fillEpsilon] ?? 0,
          streamThreshold: getStreamThreshold(),
          flowExponent: ctx.options.flowExponent
        },
        grid.rasterGrid
      )
    );
    drainageDirty = true;
    extensionDirty = true;
  };
  const writeIndicesSettings = () => {
    indicesSettings.write(
      getGPUTerrainHydrologicIndicesParameterValues(
        {minimumSlope: ctx.options.minimumSlope / 100},
        grid.rasterGrid
      )
    );
    drainageDirty = true;
  };
  const writeDisplayParameters = () => {
    const kind = {wetness: 0, catchment: 1, power: 2}[ctx.options.indexKind];
    displayParameters.write(Float32Array.of(kind, ctx.options.floodStage));
    drainageDirty = true;
  };
  const writePourPoints = () => {
    pourPointValues.fill(UNUSED_POUR_POINT);
    const positions = new Float32Array(POUR_POINT_CAPACITY * 2);
    pourPoints.forEach((cell, index) => {
      pourPointValues[index] = cell;
      const [x, y] = getCellCenter(grid, cell);
      positions[2 * index] = x;
      positions[2 * index + 1] = y;
    });
    pourPointBuffer.write(pourPointValues);
    pourPointPositionsBuffer.write(positions);
    drainageDirty = true;
    ctx.setReadout('pourPoints', pourPoints.length);
    ctx.requestLayers();
  };
  derivativesSettings.write(
    getGPUTerrainDerivativesParameterValues(
      {azimuthDegrees: SUN_AZIMUTH_DEGREES, altitudeDegrees: SUN_ALTITUDE_DEGREES},
      grid.rasterGrid
    )
  );
  writeFlowSettings();
  writeIndicesSettings();
  writeDisplayParameters();

  ctx.setReadout('grid', `${width} × ${height} cells`);
  ctx.setReadout(
    'cellSize',
    `${grid.groundCellSize[0].toFixed(1)} m (${formatInteger(cellAreaSquareMeters)} m² per cell)`
  );
  let minimumElevation = Infinity;
  let maximumElevation = -Infinity;
  for (const value of grid.elevation) {
    if (value < minimumElevation) minimumElevation = value;
    if (value > maximumElevation) maximumElevation = value;
  }
  ctx.setReadout(
    'relief',
    `${formatInteger(minimumElevation)} to ${formatInteger(maximumElevation)} m (${formatInteger(maximumElevation - minimumElevation)} m)`
  );
  ctx.setReadout('pourPoints', 0);

  const summary = new SummaryReader(
    resources,
    'watersheds',
    [
      ...flagBuffers.map(buffer => ({buffer, size: 4})),
      {buffer: streamCountsBuffer, size: 8},
      {buffer: orderHistogramBuffer, size: ORDER_BIN_COUNT * 4},
      ...iterationBuffers.map(buffer => ({buffer, size: 4})),
      {buffer: watershedHistogramBuffer, size: POUR_POINT_CAPACITY * 4},
      {buffer: floodCountsBuffer, size: 8},
      {buffer: networkReachCountBuffer, size: 4},
      {buffer: networkVertexCountBuffer, size: 4},
      {buffer: networkOverflowBuffer, size: 4},
      {buffer: networkUnresolvedBuffer, size: 4}
    ],
    bytes => {
      if (destroyed) return;
      const words = new Uint32Array(bytes);
      const iterationBase = 9 + ORDER_BIN_COUNT;
      const watershedBase = iterationBase + 4;
      const floodBase = watershedBase + POUR_POINT_CAPACITY;
      const flag = (index: number, label: string, limit: number, iterationIndex: number) =>
        `${label} ${words[index] ? 'yes' : 'NO'} ${words[iterationBase + iterationIndex]}/${limit}`;
      ctx.setReadout(
        'converged',
        [
          flag(0, 'fill', MAXIMUM_FILL_ITERATIONS, 0),
          builtResolveFlats ? flag(1, 'flats', MAXIMUM_FLAT_ITERATIONS, 1) : 'flats off',
          flag(2, 'accumulation', MAXIMUM_ACCUMULATION_ITERATIONS, 2),
          words[3] && words[4] && words[5] && words[6] ? 'tracing yes' : 'tracing NO',
          `order ${words[iterationBase + 3]} rounds`
        ].join(', ')
      );
      const streamCells = words[8];
      ctx.setReadout(
        'streams',
        `${formatInteger(streamCells)} cells, ${((streamCells / landCellCount) * 100).toFixed(1)} % of the window`
      );
      const orders = Array.from(words.slice(9 + 1, 9 + ORDER_BIN_COUNT));
      const maximumOrder = orders.reduce((last, count, index) => (count > 0 ? index + 1 : last), 0);
      ctx.setReadout('maximumOrder', maximumOrder);
      ctx.setReadout(
        'orders',
        orders
          .slice(0, Math.max(maximumOrder, 1))
          .map((count, index) => `${index + 1}: ${formatInteger(count)}`)
          .join('  ')
      );
      const watersheds = Array.from(words.slice(watershedBase, watershedBase + pourPoints.length));
      ctx.setReadout(
        'watershedAreas',
        watersheds.length === 0
          ? 'no pour points'
          : watersheds
              .map(
                (count, index) => `${index + 1}: ${formatKilometers(count * cellAreaSquareMeters)}`
              )
              .join('  ')
      );
      ctx.setReadout(
        'floodArea',
        `${formatKilometers(words[floodBase + 1] * cellAreaSquareMeters)} within ${ctx.options.floodStage} m of a stream`
      );
      const networkBase = floodBase + 2;
      const networkUnresolvedIndex = networkBase + 3;
      const nextNetworkReachCount = words[networkBase];
      if (nextNetworkReachCount !== networkReachCount) {
        networkReachCount = nextNetworkReachCount;
        ctx.requestLayers();
      }
      ctx.setReadout('networkReaches', formatInteger(words[networkBase]));
      ctx.setReadout('networkVertices', formatInteger(words[networkBase + 1]));
      ctx.setReadout(
        'networkStatus',
        `${words[networkBase + 2] ? 'capacity reached' : 'complete'}, ${formatInteger(words[networkUnresolvedIndex])} unresolved stream cells`
      );
      if (!defaultsSeeded) {
        defaultsSeeded = true;
        void seedDefaultPourPoints();
      }
    }
  );

  /** Highest-accumulation cell within the snap radius, read from a small window of rows. */
  async function snapToStream(
    cell: number,
    radius = POUR_POINT_SNAP_RADIUS,
    maximumArea = Infinity
  ): Promise<number> {
    const column = cell % width;
    const row = Math.floor(cell / width);
    const firstRow = Math.max(0, row - radius);
    const lastRow = Math.min(height - 1, row + radius);
    const rowCount = lastRow - firstRow + 1;
    const bytes = await accumulationBuffer.readAsync(firstRow * width * 4, rowCount * width * 4);
    const window = new Float32Array(bytes.buffer, bytes.byteOffset, rowCount * width);
    let best = cell;
    let bestValue = -Infinity;
    for (let dRow = -radius; dRow <= radius; dRow++) {
      const candidateRow = row + dRow;
      if (candidateRow < firstRow || candidateRow > lastRow) continue;
      for (let dColumn = -radius; dColumn <= radius; dColumn++) {
        const candidateColumn = column + dColumn;
        if (candidateColumn < 0 || candidateColumn >= width) continue;
        const value = window[(candidateRow - firstRow) * width + candidateColumn];
        if (Number.isFinite(value) && value <= maximumArea && value > bestValue) {
          best = candidateRow * width + candidateColumn;
          bestValue = value;
        }
      }
    }
    return best;
  }

  async function addPourPoint(cell: number): Promise<void> {
    const snapped = await snapToStream(cell);
    if (destroyed) return;
    if (pourPoints.length >= POUR_POINT_CAPACITY) pourPoints.shift();
    pourPoints.push(snapped);
    writePourPoints();
  }

  async function seedDefaultPourPoints(): Promise<void> {
    try {
      for (const {lngLat, snapRadius, maximumAreaKm2} of DEFAULT_POUR_POINTS) {
        const cell = getCellAt(grid, lngLat[0], lngLat[1]);
        if (cell >= 0 && !destroyed) {
          const snapped = await snapToStream(cell, snapRadius, maximumAreaKm2 * 1e6);
          if (destroyed) return;
          pourPoints.push(snapped);
        }
      }
      writePourPoints();
    } catch {
      // The device was destroyed while the accumulation window was being read.
    }
  }

  let requestedRouting = builtRouting;
  let requestedResolveFlats = builtResolveFlats;
  const drainageSwap = resources.track(new GraphSwap());

  /** Recompiles the drainage graph; the old one keeps running until the new one is ready. */
  function requestDrainageGraph(): void {
    requestedRouting = ctx.options.routing;
    requestedResolveFlats = ctx.options.resolveFlats;
    void drainageSwap.request({
      build: async () => {
        const routing = ctx.options.routing;
        const resolveFlats = ctx.options.resolveFlats;
        const graph = resources.track(await compileDrainageGraph(routing, resolveFlats));
        return {graph, routing, resolveFlats};
      },
      commit: next => {
        const previous = compiledDrainage;
        compiledDrainage = next.graph;
        builtRouting = next.routing;
        builtResolveFlats = next.resolveFlats;
        resources.release(previous);
        drainageDirty = true;
        extensionDirty = true;
        ctx.requestLayers();
      },
      discard: next => resources.release(next.graph)
    });
  }

  /** Compiles an optional story product only when the reader first selects it. */
  function requestExtensionGraph(product: 'flow-distance' | 'network'): void {
    if (extensionGraphs.has(product)) {
      extensionDirty = true;
      ctx.requestLayers();
      return;
    }
    void extensionSwap.request({
      build: async () => resources.track(await compileExtensionGraph(product)),
      commit: compiled => {
        extensionGraphs.set(product, compiled);
        extensionDirty = true;
        ctx.requestLayers();
      },
      discard: compiled => resources.release(compiled)
    });
  }

  if (ctx.options.product === 'flow-distance' || ctx.options.product === 'network') {
    requestExtensionGraph(ctx.options.product);
  }

  return {
    getCompiledGraphs: () => [compiledSetup, compiledDrainage, ...extensionGraphs.values()],

    setOption(id, _value, state) {
      switch (id) {
        case 'routing':
        case 'resolveFlats':
          if (state.routing !== requestedRouting || state.resolveFlats !== requestedResolveFlats) {
            requestDrainageGraph();
          }
          break;
        case 'product':
          if (state.product === 'flow-distance' || state.product === 'network') {
            requestExtensionGraph(state.product);
          } else {
            extensionDirty = true;
            ctx.requestLayers();
          }
          break;
        case 'flowExponent':
        case 'streamArea':
        case 'fillEpsilon':
          writeFlowSettings();
          ctx.requestLayers();
          break;
        case 'minimumSlope':
          writeIndicesSettings();
          break;
        case 'indexKind':
        case 'floodStage':
          writeDisplayParameters();
          ctx.requestLayers();
          break;
        default:
          ctx.requestLayers();
      }
    },

    onAction(id) {
      if (id === 'clearPourPoints') {
        pourPoints.length = 0;
        writePourPoints();
      } else if (id === 'resetPourPoints') {
        pourPoints.length = 0;
        void seedDefaultPourPoints();
      }
    },

    encode(commandEncoder, frame) {
      let summaryDirty = false;
      if (frame.frameIndex < 2) {
        compiledSetup.encode(commandEncoder, {parameters: undefined});
      }
      if (drainageDirty || frame.frameIndex < 2) {
        compiledDrainage.encode(commandEncoder, {parameters: undefined});
        drainageDirty = false;
        summaryDirty = true;
        extensionDirty = true;
      }
      const product = ctx.options.product;
      const extension =
        product === 'flow-distance' || product === 'network'
          ? extensionGraphs.get(product)
          : undefined;
      if (extension && (extensionDirty || frame.frameIndex < 2)) {
        extension.encode(commandEncoder, {parameters: undefined});
        extensionDirty = false;
        summaryDirty = true;
      }
      if (summaryDirty) summary.request(commandEncoder);
      summary.flush(commandEncoder);
    },

    getLayers() {
      const options = ctx.options;
      const rasterProps = {
        coordinateOrigin: origin,
        gridSize: [width, height] as const,
        bounds: grid.bounds,
        rowOrigin: 'north' as const
      };
      const dark = ctx.theme() === 'dark';
      const alpha = Math.round(options.overlayOpacity * 255);
      const layers: Layer[] = [
        new SpatialAnalysisRasterLayer({
          ...rasterProps,
          id: 'watersheds-hillshade',
          values: hillshadeBuffer,
          valueFormat: 'float32',
          colormap: 'grayscale',
          valueRange: [0, 1],
          color: [255, 255, 255, dark ? 200 : 170]
        })
      ];
      const scalar = (
        id: string,
        values: Buffer,
        colormap: 'bamako' | 'inferno' | 'cividis',
        valueRange: readonly [number, number],
        discardAtOrBelow: number,
        noDataColor?: readonly [number, number, number, number]
      ) =>
        new SpatialAnalysisRasterLayer({
          ...rasterProps,
          id,
          values,
          valueFormat: 'float32',
          colormap,
          valueRange,
          discardAtOrBelow,
          color: [255, 255, 255, alpha],
          ...(noDataColor ? {noDataColor} : {})
        });
      const categorical = (
        id: string,
        values: Buffer,
        palette: readonly (readonly [number, number, number, number])[]
      ) =>
        new SpatialAnalysisRasterLayer({
          ...rasterProps,
          id,
          values,
          valueFormat: 'uint32',
          colormap: 'category',
          palette,
          noDataColor: [0, 0, 0, 0],
          color: [255, 255, 255, alpha]
        });
      switch (options.product) {
        case 'accumulation':
          layers.push(
            scalar(
              'watersheds-accumulation',
              accumulationDisplayBuffer,
              'bamako',
              ACCUMULATION_LOG_RANGE,
              // Hillslope cells below 0.01 km² stay transparent so the stream network reads.
              ACCUMULATION_LOG_RANGE[0]
            )
          );
          break;
        case 'fill':
          layers.push(scalar('watersheds-fill', fillDepthBuffer, 'inferno', [0, 10], 0.0005));
          break;
        case 'hand':
          layers.push(
            scalar(
              'watersheds-hand',
              handBuffer,
              'inferno',
              [0, options.handRange],
              HIDDEN_VALUE / 2,
              UNRESOLVED_COLOR
            )
          );
          break;
        case 'flood':
          layers.push(
            scalar(
              'watersheds-flood',
              floodDepthBuffer,
              'cividis',
              [0, options.floodStage],
              0,
              UNRESOLVED_COLOR
            )
          );
          break;
        case 'watersheds':
          layers.push(
            categorical(
              'watersheds-labels',
              options.basinMode === 'pour-points' ? watershedLabelsBuffer : basinClassBuffer,
              BASIN_PALETTE
            )
          );
          break;
        case 'indices': {
          const range =
            options.indexKind === 'wetness'
              ? WETNESS_RANGE
              : options.indexKind === 'catchment'
                ? CATCHMENT_LOG_RANGE
                : POWER_LOG_RANGE;
          layers.push(
            scalar(
              'watersheds-index',
              indexDisplayBuffer,
              options.indexKind === 'power' ? 'inferno' : 'bamako',
              range,
              HIDDEN_VALUE / 2
            )
          );
          break;
        }
        case 'stream-order':
          ORDER_WIDTHS.forEach((widthPixels, index) => {
            layers.push(
              new SpatialAnalysisSegmentLayer({
                id: `watersheds-order-${index}`,
                coordinateOrigin: origin,
                segments: segmentsBuffer,
                instanceCount: cellCount,
                values: streamOrderBuffer,
                valueFormat: 'uint32',
                colormap: 'category',
                palette: ORDER_PALETTE,
                weights: widthBuffers[index],
                widthPixels
              })
            );
          });
          break;
        case 'flow-distance':
          layers.push(
            scalar(
              'watersheds-flow-distance',
              flowDistanceDisplayBuffer,
              'cividis',
              [0, 5000],
              HIDDEN_VALUE / 2,
              UNRESOLVED_COLOR
            )
          );
          break;
        case 'network':
          ORDER_WIDTHS.forEach((widthPixels, index) => {
            layers.push(
              new SpatialAnalysisSegmentLayer({
                id: `watersheds-network-order-${index}`,
                coordinateOrigin: origin,
                segments: segmentsBuffer,
                instanceCount: cellCount,
                values: streamOrderBuffer,
                valueFormat: 'uint32',
                colormap: 'category',
                palette: ORDER_PALETTE,
                weights: widthBuffers[index],
                widthPixels
              })
            );
          });
          if (networkReachCount > 0) {
            layers.push(
              new SpatialAnalysisPointLayer({
                id: 'watersheds-network-endpoints',
                coordinateOrigin: origin,
                positions: networkEndpointPositionsBuffer,
                instanceCount: networkReachCount,
                radiusPixels: 3.5,
                color: [25, 75, 90, 255]
              })
            );
          }
          break;
      }
      if (
        options.showStreams &&
        options.product !== 'stream-order' &&
        options.product !== 'network' &&
        options.product !== 'accumulation'
      ) {
        layers.push(
          new SpatialAnalysisRasterLayer({
            ...rasterProps,
            id: 'watersheds-streams',
            values: streamsBuffer,
            valueFormat: 'uint32',
            colormap: 'mask',
            color: dark ? [90, 190, 255, 255] : [20, 110, 230, 255],
            // 0xffffffff is GPU_TERRAIN_FLOW_STREAM_UNRESOLVED (accumulation did not converge).
            // It is the layer's default noDataValue, so it draws as "unknown", not as a stream.
            noDataColor: UNRESOLVED_COLOR
          })
        );
      }
      if (options.product === 'watersheds' && options.basinMode === 'pour-points') {
        const markerCount = pourPoints.length;
        if (markerCount > 0) {
          layers.push(
            new SpatialAnalysisPointLayer({
              id: 'watersheds-pour-halo',
              coordinateOrigin: origin,
              positions: pourPointPositionsBuffer,
              instanceCount: markerCount,
              radiusPixels: 9,
              color: [255, 255, 255, 255]
            }),
            new SpatialAnalysisPointLayer({
              id: 'watersheds-pour-points',
              coordinateOrigin: origin,
              positions: pourPointPositionsBuffer,
              instanceCount: markerCount,
              radiusPixels: 6,
              values: pourPointIndexBuffer,
              valueFormat: 'uint32',
              colormap: 'category',
              palette: BASIN_PALETTE.map(([r, g, b]) => [r, g, b, 255] as const)
            })
          );
        }
      }
      return layers;
    },

    onClick(event) {
      if (ctx.options.product !== 'watersheds' || ctx.options.basinMode !== 'pour-points') {
        return false;
      }
      if (!event.coordinate) return false;
      const cell = getCellAt(grid, event.coordinate[0], event.coordinate[1]);
      if (cell < 0) return false;
      void addPourPoint(cell).catch(() => {});
      return true;
    },

    getTooltip(event) {
      if (!event.coordinate) return null;
      const cell = getCellAt(grid, event.coordinate[0], event.coordinate[1]);
      if (cell < 0) return null;
      return `Elevation ${formatInteger(grid.elevation[cell])} m\n${event.coordinate[1].toFixed(4)}°, ${event.coordinate[0].toFixed(4)}°`;
    },

    destroy() {
      destroyed = true;
      summary.stop();
      extensionSwap.destroy();
      resources.destroy();
    }
  };
}

/**
 * `display = wetness | log10(specific catchment area) | log10(stream power)` selected by the first
 * value of `kind` (0, 1, 2), with {@link HIDDEN_VALUE} for NaN and infinite cells.
 */
function addIndexDisplayPass(
  graph: GPUCommandGraph<void>,
  props: {
    cellCount: number;
    wetness: GraphDataView<'float32'>;
    catchment: GraphDataView<'float32'>;
    power: GraphDataView<'float32'>;
    kind: GraphDataView<'float32'>;
    output: GraphDataView<'float32'>;
  }
): void {
  addKernelPass(graph, {
    id: 'index-display',
    bindings: [
      {name: 'wetness', view: props.wetness, type: 'f32', access: 'read'},
      {name: 'catchment', view: props.catchment, type: 'f32', access: 'read'},
      {name: 'power', view: props.power, type: 'f32', access: 'read'},
      {name: 'parameters', view: props.kind, type: 'f32', access: 'read'},
      {name: 'display', view: props.output, type: 'f32', access: 'read_write'}
    ],
    invocationCount: props.cellCount,
    declarations: NAN_DECLARATIONS,
    body: `let kind = u32(parameters[parametersOffset] + 0.5);
  var value = wetness[wetnessOffset + index];
  if (kind == 1u) { value = log2(max(catchment[catchmentOffset + index], 0.000001)) * 0.30102999566; }
  if (kind == 2u) { value = log2(max(power[powerOffset + index], 0.000001)) * 0.30102999566; }
  let invalid = isNaNValue(value) || isInfiniteValue(value);
  display[displayOffset + index] = select(value, ${HIDDEN_VALUE.toFixed(1)}, invalid);`
  });
}

/** Ground-scale D8 edge lengths for the canyon window's center latitude. */
function addFlowDistanceCostPass(
  graph: GPUCommandGraph<void>,
  props: {
    cellCount: number;
    flowDirections: GraphDataView<'uint32'>;
    output: GraphDataView<'float32'>;
    groundCellSize: number;
  }
): void {
  addKernelPass(graph, {
    id: 'flow-distance-costs',
    bindings: [
      {name: 'directions', view: props.flowDirections, type: 'u32', access: 'read'},
      {name: 'costs', view: props.output, type: 'f32', access: 'read_write'}
    ],
    invocationCount: props.cellCount,
    body: `let code = directions[directionsOffset + index];
  let diagonal = code == 2u || code == 8u || code == 32u || code == 128u;
  costs[costsOffset + index] = select(${props.groundCellSize}, ${props.groundCellSize * Math.SQRT2}, diagonal);`
  });
}

/** Keeps paths that end on a selected stream and marks every other terminal as unresolved. */
function addFlowDistanceDisplayPass(
  graph: GPUCommandGraph<void>,
  props: {
    cellCount: number;
    distance: GraphDataView<'float32'>;
    validity: GraphDataView<'uint32'>;
    terminals: GraphDataView<'uint32'>;
    streams: GraphDataView<'uint32'>;
    output: GraphDataView<'float32'>;
  }
): void {
  addKernelPass(graph, {
    id: 'flow-distance-display',
    bindings: [
      {name: 'distance', view: props.distance, type: 'f32', access: 'read'},
      {name: 'validity', view: props.validity, type: 'u32', access: 'read'},
      {name: 'terminals', view: props.terminals, type: 'u32', access: 'read'},
      {name: 'streams', view: props.streams, type: 'u32', access: 'read'},
      {name: 'display', view: props.output, type: 'f32', access: 'read_write'}
    ],
    invocationCount: props.cellCount,
    declarations: 'fn getQuietNaN() -> f32 { var bits = 0x7fc00000u; return bitcast<f32>(bits); }',
    body: `let terminal = terminals[terminalsOffset + index];
  var terminalIsStream = false;
  if (terminal != 0xffffffffu && terminal != 0xfffffffeu) {
    let terminalStream = streams[streamsOffset + terminal];
    terminalIsStream = terminalStream != 0u && terminalStream != 0xffffffffu && terminalStream != 0xfffffffeu;
  }
  let valid = validity[validityOffset + index] != 0u && terminalIsStream;
  display[displayOffset + index] = select(getQuietNaN(), distance[distanceOffset + index], valid);`
  });
}

/** Converts compact reach-start cell IDs to local map coordinates for the network endpoint markers. */
function addNetworkEndpointPositionsPass(
  graph: GPUCommandGraph<void>,
  props: {
    capacity: number;
    width: number;
    displayCellSize: readonly [number, number];
    bounds: readonly [number, number, number, number];
    sourceCells: GraphDataView<'uint32'>;
    reachCount: GraphDataView<'uint32'>;
    output: GraphDataView<'float32x2'>;
  }
): void {
  addKernelPass(graph, {
    id: 'network-endpoint-positions',
    bindings: [
      {name: 'sources', view: props.sourceCells, type: 'u32', access: 'read'},
      {name: 'reachCount', view: props.reachCount, type: 'u32', access: 'read'},
      {name: 'positions', view: props.output, type: 'f32', access: 'read_write'}
    ],
    invocationCount: props.capacity,
    body: `if (index >= reachCount[reachCountOffset]) { return; }
  let cell = sources[sourcesOffset + index];
  let column = f32(cell % ${props.width}u) + 0.5;
  let row = f32(cell / ${props.width}u) + 0.5;
  positions[positionsOffset + index * 2u] = ${props.bounds[0]} + column * ${props.displayCellSize[0]};
  positions[positionsOffset + index * 2u + 1u] = ${props.bounds[3]} - row * ${props.displayCellSize[1]};`
  });
}

/**
 * Flood at a stage: `depth = stage - HAND` where the cell is lower than `stage` above its drainage
 * (else 0, hidden by the layer) and `mask` is 1 there. `stage` is the second value of the view.
 */
function addFloodPass(
  graph: GPUCommandGraph<void>,
  props: {
    cellCount: number;
    hand: GraphDataView<'float32'>;
    stage: GraphDataView<'float32'>;
    depth: GraphDataView<'float32'>;
    mask: GraphDataView<'uint32'>;
  }
): void {
  addKernelPass(graph, {
    id: 'flood-depth',
    bindings: [
      {name: 'hand', view: props.hand, type: 'f32', access: 'read'},
      {name: 'parameters', view: props.stage, type: 'f32', access: 'read'},
      {name: 'depth', view: props.depth, type: 'f32', access: 'read_write'},
      {name: 'mask', view: props.mask, type: 'u32', access: 'read_write'}
    ],
    invocationCount: props.cellCount,
    declarations: NAN_DECLARATIONS,
    body: `let stage = parameters[parametersOffset + 1u];
  let height = hand[handOffset + index];
  let valid = !isNaNValue(height) && !isInfiniteValue(height) && height >= 0.0;
  let flooded = valid && height <= stage;
  depth[depthOffset + index] = select(0.0, max(stage - height, 0.0001), flooded);
  mask[maskOffset + index] = select(0u, 1u, flooded);`
  });
}

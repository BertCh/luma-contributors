// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, CommandEncoder} from '@luma.gl/core';
import type {Layer} from '@deck.gl/core';
import {
  GPUTerrainPeakProminence,
  GPUTerrainBranchingFlow,
  GPUTerrainFlowFractions,
  GPUTerrainStreamNetwork,
  GPUTerrainTiledFlow,
  getGPUTerrainFlowParameterValues,
  GPU_TERRAIN_FLOW_PARAMETER_LENGTH
} from '@luma-contributors/analysis/gpu-terrain';
import {addDrainageNetworkRecipe} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {
  GPUCommandGraph,
  type CompiledGPUCommandGraph,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {createGraphBufferImporter} from '../../engine/graph-buffers';
import {
  SpatialAnalysisPointLayer,
  SpatialAnalysisRasterLayer,
  SpatialAnalysisSegmentLayer
} from '../../engine/layers';
import {addKernelPass} from '../../engine/mode-kernels';
import {GraphSwap} from '../../engine/graph-swap';
import {SpatialAnalysisResources} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import type {SceneContext, SceneInstance} from '../scene';
import {createCanyonGrid} from './b15-common';

export type TerrainBoundariesOptions = {
  product: 'prominence' | 'streams' | 'tiled';
  iterations: '4' | '32' | '128';
  routing: 'd8' | 'branching';
  branchMethod: 'd-infinity' | 'mfd-freeman';
};

const CELL_COUNT = 64 * 64;
const TILE_SIZE = 32;
const MAXIMUM_REACHES = CELL_COUNT;
const MAXIMUM_VERTICES = CELL_COUNT * 2;

type TileBuffers = {
  row: number;
  column: number;
  width: number;
  height: number;
  owned: readonly [number, number, number, number];
  directions: Buffer;
  runoff: Buffer;
  streams: Buffer;
  accumulation: Buffer;
  status: Buffer;
  reachIds: Buffer;
  fractions: Buffer;
};

/** Publish graph-owned storage outputs without requiring hardware-copy usage. */
function addStorageCopyPass<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  source: GraphDataView,
  destination: GraphDataView
): void {
  const wordCount = (source.length * source.rowByteLength) / 4;
  const sourceWords = graph.createDataView(source.buffer, {
    format: 'uint32',
    length: wordCount,
    byteOffset: source.byteOffset
  });
  const destinationWords = graph.createDataView(destination.buffer, {
    format: 'uint32',
    length: wordCount,
    byteOffset: destination.byteOffset
  });
  // Copy stored words exactly, including float32 payloads and missing-value bit patterns.
  addKernelPass(graph, {
    id,
    invocationCount: wordCount,
    bindings: [
      {name: 'sourceWords', view: sourceWords, type: 'u32', access: 'read'},
      {name: 'destinationWords', view: destinationWords, type: 'u32', access: 'read_write'}
    ],
    body: 'destinationWords[destinationWordsOffset + index] = sourceWords[sourceWordsOffset + index];'
  });
}

export async function createTerrainBoundaries(
  ctx: SceneContext<TerrainBoundariesOptions>
): Promise<SceneInstance<TerrainBoundariesOptions>> {
  const {device} = ctx;
  const grid = createCanyonGrid(ctx.datasets.get('grand-canyon-dem'), 32);
  const {width, height, cellCount} = grid;
  const resources = new SpatialAnalysisResources(device, 'terrain-boundaries');
  const elevation = resources.createBuffer('source-elevation', grid.elevation);
  const prominence = resources.createBuffer('peak-prominence', cellCount * 4);
  const prominenceDisplay = resources.createBuffer('peak-prominence-display', cellCount * 4);
  const keySaddle = resources.createBuffer('peak-key-saddle', cellCount * 4);
  const parentPeak = resources.createBuffer('peak-parent', cellCount * 4);
  const streams = resources.createBuffer('d8-streams', cellCount * 4);
  const streamOrder = resources.createBuffer('d8-stream-order', cellCount * 4);
  const streamOrderDisplay = resources.createBuffer('d8-stream-order-display', cellCount * 4);
  const accumulation = resources.createBuffer('d8-accumulation', cellCount * 4);
  const flowDirections = resources.createBuffer('d8-directions', cellCount * 4);
  const flowStatus = resources.createBuffer('d8-status', 12);
  const catchmentReadback = resources.createBuffer('catchment-readback', cellCount * 16);
  const networkPositions = resources.createBuffer('network-positions', MAXIMUM_VERTICES * 8);
  const networkOffsets = resources.createBuffer('network-offsets', (MAXIMUM_REACHES + 1) * 4);
  const networkSources = resources.createBuffer('network-sources', MAXIMUM_REACHES * 4);
  const networkTargets = resources.createBuffer('network-targets', MAXIMUM_REACHES * 4);
  const networkLengths = resources.createBuffer('network-lengths', MAXIMUM_REACHES * 4);
  const networkReachCount = resources.createBuffer('network-reach-count', 4);
  const networkVertexCount = resources.createBuffer('network-vertex-count', 4);
  const networkOverflow = resources.createBuffer('network-overflow', 4);
  const networkUnresolved = resources.createBuffer('network-unresolved', 4);
  const reachSegments = resources.createBuffer('reach-segments', cellCount * 16);
  const tiledAccumulation = resources.createBuffer('tiled-accumulation', cellCount * 4);
  const tiledStatus = resources.createBuffer('tiled-status', cellCount * 4);
  const tiledReachIds = resources.createBuffer('tiled-reach-ids', cellCount * 4);
  const tiledSummary = resources.createBuffer('tiled-summary', 24);
  const branchingFractions = resources.createBuffer('branching-fractions', cellCount * 8 * 4);
  const branchingEdgeLengths = resources.createBuffer('branching-edge-lengths', cellCount * 8 * 4);
  const branchingReference = resources.createBuffer('branching-reference', cellCount * 4);
  const branchingReferenceStatus = resources.createBuffer(
    'branching-reference-status',
    cellCount * 4
  );
  const branchingReferenceUnresolved = resources.createBuffer('branching-reference-unresolved', 4);
  const unitRunoff = resources.createBuffer(
    'branching-unit-runoff',
    new Float32Array(cellCount).fill(1)
  );
  const selectedPeakMarker = resources.createBuffer('selected-peak-marker', new Float32Array(2));
  const selectedCellMarker = resources.createBuffer('selected-cell-marker', new Float32Array(2));
  const elderMarkers = resources.createBuffer('infinite-elder-markers', cellCount * 8);
  let infiniteElderCount = 0;
  const windowOutline = resources.createBuffer(
    'window-outline',
    Float32Array.of(
      grid.bounds[0],
      grid.bounds[1],
      grid.bounds[2],
      grid.bounds[1],
      grid.bounds[2],
      grid.bounds[1],
      grid.bounds[2],
      grid.bounds[3],
      grid.bounds[2],
      grid.bounds[3],
      grid.bounds[0],
      grid.bounds[3],
      grid.bounds[0],
      grid.bounds[3],
      grid.bounds[0],
      grid.bounds[1]
    )
  );
  const tileSeams = resources.createBuffer(
    'tile-seams',
    Float32Array.of(
      grid.bounds[0] + TILE_SIZE * grid.displayCellSize[0],
      grid.bounds[1],
      grid.bounds[0] + TILE_SIZE * grid.displayCellSize[0],
      grid.bounds[3],
      grid.bounds[0],
      grid.bounds[3] - TILE_SIZE * grid.displayCellSize[1],
      grid.bounds[2],
      grid.bounds[3] - TILE_SIZE * grid.displayCellSize[1]
    )
  );
  const parameterValues = getGPUTerrainFlowParameterValues(
    {fillEpsilon: 0.01, streamThreshold: 12},
    grid.rasterGrid
  );
  const parameters = resources.createParameterBuffer(
    'flow-parameters',
    'float32',
    GPU_TERRAIN_FLOW_PARAMETER_LENGTH,
    parameterValues
  );

  const tileBuffers: TileBuffers[] = [];
  for (const tileRow of [0, 1]) {
    for (const tileColumn of [0, 1]) {
      const ownedColumn = tileColumn * TILE_SIZE;
      const ownedRow = tileRow * TILE_SIZE;
      const originColumn = Math.max(0, ownedColumn - 1);
      const originRow = Math.max(0, ownedRow - 1);
      const tileWidth = Math.min(width - originColumn, TILE_SIZE + Number(tileColumn > 0));
      const tileHeight = Math.min(height - originRow, TILE_SIZE + Number(tileRow > 0));
      const owned: readonly [number, number, number, number] = [
        ownedColumn - originColumn,
        ownedRow - originRow,
        TILE_SIZE,
        TILE_SIZE
      ];
      const tileCellCount = tileWidth * tileHeight;
      const tileIndex = tileRow * 2 + tileColumn;
      tileBuffers.push({
        row: originRow,
        column: originColumn,
        width: tileWidth,
        height: tileHeight,
        owned,
        directions: resources.createBuffer(`tile-${tileIndex}-directions`, tileCellCount * 4),
        runoff: resources.createBuffer(`tile-${tileIndex}-runoff`, tileCellCount * 4),
        streams: resources.createBuffer(`tile-${tileIndex}-streams`, tileCellCount * 4),
        accumulation: resources.createBuffer(`tile-${tileIndex}-accumulation`, tileCellCount * 4),
        status: resources.createBuffer(`tile-${tileIndex}-status`, tileCellCount * 4),
        reachIds: resources.createBuffer(`tile-${tileIndex}-reach-ids`, tileCellCount * 4),
        fractions: resources.createBuffer(`tile-${tileIndex}-fractions`, tileCellCount * 8 * 4)
      });
    }
  }

  const graphSwap = resources.track(new GraphSwap());
  type ActiveGraph = {
    compiled: CompiledGPUCommandGraph<void>;
    reader: SummaryReader;
    revision: number;
  };
  let active: ActiveGraph | null = null;
  let graphRevision = 0;
  let acceptedRevision = -1;
  let selectedCell = Math.floor(height / 2) * width + Math.floor(width / 2);
  let selectedFinitePeak = -1;
  let cachedBranchingUnresolved = 0;
  let cachedFinitePeakCount = 0;
  let cachedElderCount = 0;
  let cachedProminence = new Float32Array(cellCount);
  let cachedSaddles = new Float32Array(cellCount);
  let cachedParents = new Uint32Array(cellCount);
  let cachedCellCatchments = new Uint32Array(cellCount);
  let cachedCatchmentIds = new Uint32Array(cellCount);
  let cachedCatchmentCounts = new Uint32Array(cellCount);
  let cachedMeanDistance = new Float32Array(cellCount);
  let cachedFlowStatus = new Uint32Array(3);
  let cachedTiledSummary = new Uint32Array(6);
  let selectedPeakText = 'Waiting for GPU peak tree…';
  let drainageText = 'Waiting for GPU D8 recipe…';
  let tiledText = 'Waiting for four owned tiles…';
  let chart = {
    kind: 'bars' as const,
    title: 'Selected peak and its merge level',
    xLabel: 'GPU output',
    yLabel: 'Elevation (m)',
    values: new Float32Array(3),
    labels: ['peak', 'key saddle', 'saddle + prominence']
  };
  const peakReadback = resources.createBuffer('peak-readback', cellCount * 12);
  const readbackStatuses = resources.createBuffer('summary-readback', 72);

  const formatCell = (cell: number) =>
    `row ${Math.floor(cell / width) + 1}, column ${(cell % width) + 1}`;
  const catchmentSummaryForCell = (cell: number) => {
    const basin = cachedCellCatchments[cell];
    const group = cachedCatchmentIds.indexOf(basin);
    return basin === 0xffffffff || group < 0 || !Number.isFinite(cachedMeanDistance[group])
      ? `${formatCell(cell)} · no resolved catchment summary`
      : `${formatCell(cell)} · basin ${basin} · ${cachedCatchmentCounts[group]} cells · mean downstream distance ${(cachedMeanDistance[group] / 1000).toFixed(2)} km`;
  };
  const catchmentText = () => catchmentSummaryForCell(selectedCell);
  const cellPosition = (cell: number): [number, number] => {
    const row = Math.floor(cell / width);
    const column = cell % width;
    return [
      grid.bounds[0] + (column + 0.5) * grid.displayCellSize[0],
      grid.bounds[3] - (row + 0.5) * grid.displayCellSize[1]
    ];
  };
  selectedCellMarker.write(Float32Array.from(cellPosition(selectedCell)));
  const updatePeak = () => {
    if (selectedFinitePeak < 0) return;
    const heightValue = grid.elevation[selectedFinitePeak];
    const saddle = cachedSaddles[selectedFinitePeak];
    const value = cachedProminence[selectedFinitePeak];
    selectedPeakText = `${formatCell(selectedFinitePeak)} · ${heightValue.toFixed(2)} m peak · ${saddle.toFixed(2)} m key saddle · ${value.toFixed(2)} m finite prominence · ${cachedFinitePeakCount} finite peaks · ${cachedElderCount} infinite component elders · elder link ${cachedParents[selectedFinitePeak] === 0xffffffff ? 'none' : formatCell(cachedParents[selectedFinitePeak])}`;
    const marker = cellPosition(selectedFinitePeak);
    selectedPeakMarker.write(Float32Array.from(marker));
    chart = {
      kind: 'bars',
      title: 'Selected peak and its merge level',
      xLabel: 'GPU output',
      yLabel: 'Elevation (m)',
      values: Float32Array.of(heightValue, saddle, saddle + value),
      labels: ['peak', 'key saddle', 'saddle + prominence']
    };
    ctx.setReadout('selectedPeak', selectedPeakText);
    ctx.setChart('peakChart', chart);
  };
  const makeGraph = async (
    iterations: number,
    routing: TerrainBoundariesOptions['routing'],
    branchMethod: TerrainBoundariesOptions['branchMethod']
  ): Promise<ActiveGraph> => {
    const revision = ++graphRevision;
    const graph = new GPUCommandGraph<void>(device, {id: `terrain-boundaries-${iterations}`});
    const importBuffer = createGraphBufferImporter(graph);
    const input = importBuffer(elevation, 'float32', cellCount);
    const sourceBand = {
      id: 'canyon-elevation',
      format: 'float32' as const,
      storage: {kind: 'buffer' as const, values: input}
    };
    const flowParameterView = parameters.importToGraph(graph);
    const prominenceView = importBuffer(prominence, 'float32', cellCount);
    const saddleView = importBuffer(keySaddle, 'float32', cellCount);
    const parentView = importBuffer(parentPeak, 'uint32', cellCount);
    graph.add(
      new GPUTerrainPeakProminence({
        id: 'finite-window-peak-tree',
        grid: grid.rasterGrid,
        elevation: sourceBand,
        output: {prominence: prominenceView, keySaddle: saddleView, parentPeak: parentView}
      })
    );
    addKernelPass(graph, {
      id: 'prepare-finite-prominence-display',
      invocationCount: cellCount,
      bindings: [
        {name: 'prominence', view: prominenceView, type: 'f32', access: 'read'},
        {
          name: 'displayProminence',
          view: importBuffer(prominenceDisplay, 'float32', cellCount),
          type: 'f32',
          access: 'read_write'
        }
      ],
      body: 'var missingBits:u32=0x7fc00000u;let missing=bitcast<f32>(missingBits);let value=prominence[prominenceOffset+index];let valid=value==value&&abs(value)<3.402823e38;displayProminence[displayProminenceOffset+index]=select(missing,value,valid);'
    });

    const drainage = addDrainageNetworkRecipe(graph, {
      id: 'bounded-d8-drainage',
      flow: {
        grid: grid.rasterGrid,
        elevation: sourceBand,
        parameters: flowParameterView,
        cellSizeMode: grid.cellSizeMode,
        fillDepressions: true,
        resolveFlats: true,
        maximumFillIterations: 256,
        maximumAccumulationIterations: 256,
        accumulationUnits: 'cells'
      },
      maximumSteps: cellCount
    });
    const flowDirsView = drainage.outputs.flowDirections;
    const streamsView = drainage.outputs.streams;
    const accumulationView = drainage.outputs.accumulation;
    const orderView = drainage.outputs.streamOrder;
    const fractionView = importBuffer(branchingFractions, 'float32', cellCount * 8);
    const edgeLengthView = importBuffer(branchingEdgeLengths, 'float32', cellCount * 8);
    graph.add(
      new GPUTerrainFlowFractions({
        id: 'global-branching-fractions',
        grid: grid.rasterGrid,
        elevation: input,
        flowDirections: flowDirsView,
        parameters: flowParameterView,
        routing: branchMethod,
        cellSizeMode: grid.cellSizeMode,
        output: {fractions: fractionView, edgeLengths: edgeLengthView}
      })
    );
    const branchingReferenceView = importBuffer(branchingReference, 'float32', cellCount);
    const branchingReferenceStatusView = importBuffer(
      branchingReferenceStatus,
      'uint32',
      cellCount
    );
    const branchingReferenceUnresolvedView = importBuffer(
      branchingReferenceUnresolved,
      'uint32',
      1
    );
    const unitRunoffView = importBuffer(unitRunoff, 'float32', cellCount);
    graph.add(
      new GPUTerrainBranchingFlow({
        id: 'whole-window-branching-reference',
        grid: grid.rasterGrid,
        fractions: fractionView,
        values: unitRunoffView,
        mode: 'transport',
        maximumIterations: iterations,
        output: {
          values: branchingReferenceView,
          status: branchingReferenceStatusView,
          unresolvedCount: branchingReferenceUnresolvedView
        }
      })
    );
    addStorageCopyPass(
      graph,
      'publish-d8-directions',
      flowDirsView,
      importBuffer(flowDirections, 'uint32', cellCount)
    );
    addStorageCopyPass(
      graph,
      'publish-d8-streams',
      streamsView,
      importBuffer(streams, 'uint32', cellCount)
    );
    addStorageCopyPass(
      graph,
      'publish-d8-accumulation',
      accumulationView,
      importBuffer(accumulation, 'float32', cellCount)
    );
    addStorageCopyPass(
      graph,
      'publish-d8-order',
      orderView,
      importBuffer(streamOrder, 'uint32', cellCount)
    );
    addKernelPass(graph, {
      id: 'prepare-stream-order-display',
      invocationCount: cellCount,
      bindings: [
        {name: 'order', view: orderView, type: 'u32', access: 'read'},
        {
          name: 'displayOrder',
          view: importBuffer(streamOrderDisplay, 'uint32', cellCount),
          type: 'u32',
          access: 'read_write'
        }
      ],
      body: 'let value=order[orderOffset+index];let valid=value>0u&&value<0xfffffffeu;displayOrder[displayOrderOffset+index]=select(0xffffffffu,min(value,4u),valid);'
    });
    const catchmentReadbackView = importBuffer(catchmentReadback, 'uint32', cellCount * 4);
    addStorageCopyPass(
      graph,
      'readback-cell-catchment-labels',
      drainage.outputs.catchments,
      catchmentReadbackView
    );
    addStorageCopyPass(
      graph,
      'readback-catchment-ids',
      drainage.outputs.summary.catchmentIds,
      graph.createDataView(catchmentReadbackView.buffer, {
        format: 'uint32',
        length: cellCount,
        byteOffset: cellCount * 4
      })
    );
    addStorageCopyPass(
      graph,
      'readback-catchment-counts',
      drainage.outputs.summary.cellCounts,
      graph.createDataView(catchmentReadbackView.buffer, {
        format: 'uint32',
        length: cellCount,
        byteOffset: cellCount * 8
      })
    );
    addStorageCopyPass(
      graph,
      'readback-mean-downstream-distance',
      drainage.outputs.summary.meanDownstreamDistance,
      graph.createDataView(catchmentReadbackView.buffer, {
        format: 'float32',
        length: cellCount,
        byteOffset: cellCount * 12
      })
    );
    const flowStatusTarget = importBuffer(flowStatus, 'uint32', 3);
    addStorageCopyPass(
      graph,
      'publish-fill-status',
      drainage.intermediates.flowStatus.fillConverged!,
      graph.createDataView(flowStatusTarget.buffer, {format: 'uint32', length: 1, byteOffset: 0})
    );
    addStorageCopyPass(
      graph,
      'publish-flat-status',
      drainage.intermediates.flowStatus.flatsConverged!,
      graph.createDataView(flowStatusTarget.buffer, {format: 'uint32', length: 1, byteOffset: 4})
    );
    addStorageCopyPass(
      graph,
      'publish-accumulation-status',
      drainage.intermediates.flowStatus.accumulationConverged!,
      graph.createDataView(flowStatusTarget.buffer, {format: 'uint32', length: 1, byteOffset: 8})
    );

    const networkView = <Format extends 'float32' | 'float32x2' | 'uint32'>(
      buffer: Buffer,
      format: Format,
      length: number
    ) => importBuffer(buffer, format, length);
    graph.add(
      new GPUTerrainStreamNetwork({
        id: 'd8-reach-geometry',
        grid: grid.rasterGrid,
        flowDirections: flowDirsView,
        streams: streamsView,
        output: {
          sourceCells: networkView(networkSources, 'uint32', MAXIMUM_REACHES),
          targetCells: networkView(networkTargets, 'uint32', MAXIMUM_REACHES),
          vertexOffsets: networkView(networkOffsets, 'uint32', MAXIMUM_REACHES + 1),
          positions: networkView(networkPositions, 'float32x2', MAXIMUM_VERTICES),
          lengths: networkView(networkLengths, 'float32', MAXIMUM_REACHES),
          reachCount: networkView(networkReachCount, 'uint32', 1),
          vertexCount: networkView(networkVertexCount, 'uint32', 1),
          status: {overflow: networkView(networkOverflow, 'uint32', 1)},
          unresolvedCount: networkView(networkUnresolved, 'uint32', 1)
        }
      })
    );

    const segmentView = importBuffer(reachSegments, 'float32', cellCount * 4);
    addKernelPass(graph, {
      id: 'draw-d8-cell-reaches',
      invocationCount: cellCount,
      bindings: [
        {name: 'directions', view: flowDirsView, type: 'u32', access: 'read'},
        {name: 'streams', view: streamsView, type: 'u32', access: 'read'},
        {name: 'segments', view: segmentView, type: 'f32', access: 'read_write'}
      ],
      declarations: `const GRID_WIDTH:u32=${width}u; const GRID_HEIGHT:u32=${height}u;`,
      body: `let direction=directions[directionsOffset+index]; let stream=streams[streamsOffset+index];
let row=index/GRID_WIDTH; let column=index%GRID_WIDTH;
var delta=vec2<i32>(0,0);
if(direction==1u){delta=vec2<i32>(1,0);}else if(direction==2u){delta=vec2<i32>(1,1);}else if(direction==4u){delta=vec2<i32>(0,1);}else if(direction==8u){delta=vec2<i32>(-1,1);}else if(direction==16u){delta=vec2<i32>(-1,0);}else if(direction==32u){delta=vec2<i32>(-1,-1);}else if(direction==64u){delta=vec2<i32>(0,-1);}else if(direction==128u){delta=vec2<i32>(1,-1);}
let targetColumn=i32(column)+delta.x; let targetRow=i32(row)+delta.y;
let valid=stream==1u && direction>0u && direction<256u && targetColumn>=0 && targetColumn<i32(GRID_WIDTH) && targetRow>=0 && targetRow<i32(GRID_HEIGHT);
let sourceEasting=${grid.bounds[0]}+(${grid.displayCellSize[0]}*(f32(column)+0.5)); let sourceNorthing=${grid.bounds[3]}-(${grid.displayCellSize[1]}*(f32(row)+0.5));
let targetEasting=${grid.bounds[0]}+(${grid.displayCellSize[0]}*(f32(targetColumn)+0.5)); let targetNorthing=${grid.bounds[3]}-(${grid.displayCellSize[1]}*(f32(targetRow)+0.5));
var missingBits:u32=0x7fc00000u;let missing=bitcast<f32>(missingBits);segments[segmentsOffset+index*4u]=select(missing,sourceEasting,valid);segments[segmentsOffset+index*4u+1u]=select(missing,sourceNorthing,valid);segments[segmentsOffset+index*4u+2u]=select(missing,targetEasting,valid);segments[segmentsOffset+index*4u+3u]=select(missing,targetNorthing,valid);`
    });

    const tileViews = tileBuffers.map((tile, tileIndex) => {
      const tileCellCount = tile.width * tile.height;
      const directions = importBuffer(tile.directions, 'uint32', tileCellCount);
      const runoff = importBuffer(tile.runoff, 'float32', tileCellCount);
      const tileStreams = importBuffer(tile.streams, 'uint32', tileCellCount);
      const tileAccum = importBuffer(tile.accumulation, 'float32', tileCellCount);
      const tileStatus = importBuffer(tile.status, 'uint32', tileCellCount);
      const tileReach = importBuffer(tile.reachIds, 'uint32', tileCellCount);
      const tileFractions = importBuffer(tile.fractions, 'float32', tileCellCount * 8);
      addKernelPass(graph, {
        id: `prepare-tile-${tileIndex}`,
        invocationCount: tileCellCount,
        bindings: [
          {name: 'globalDirections', view: flowDirsView, type: 'u32', access: 'read'},
          {name: 'globalStreams', view: streamsView, type: 'u32', access: 'read'},
          {name: 'directions', view: directions, type: 'u32', access: 'read_write'},
          {name: 'runoff', view: runoff, type: 'f32', access: 'read_write'},
          {name: 'streams', view: tileStreams, type: 'u32', access: 'read_write'},
          {name: 'globalFractions', view: fractionView, type: 'f32', access: 'read'},
          {name: 'fractions', view: tileFractions, type: 'f32', access: 'read_write'}
        ],
        declarations: `const GRID_WIDTH:u32=${width}u; const TILE_WIDTH:u32=${tile.width}u; const OWNED_COLUMN:u32=${tile.owned[0]}u; const OWNED_ROW:u32=${tile.owned[1]}u; const OWNED_WIDTH:u32=${tile.owned[2]}u; const OWNED_HEIGHT:u32=${tile.owned[3]}u;`,
        body: `let row=index/TILE_WIDTH; let column=index%TILE_WIDTH; let globalRow=row+${tile.row}u; let globalColumn=column+${tile.column}u; let global=globalRow*GRID_WIDTH+globalColumn;
let owned=column>=OWNED_COLUMN&&column<OWNED_COLUMN+OWNED_WIDTH&&row>=OWNED_ROW&&row<OWNED_ROW+OWNED_HEIGHT;
directions[directionsOffset+index]=globalDirections[globalDirectionsOffset+global]; runoff[runoffOffset+index]=select(0.0,1.0,owned); streams[streamsOffset+index]=globalStreams[globalStreamsOffset+global]; for(var direction=0u;direction<8u;direction++){fractions[fractionsOffset+index*8u+direction]=globalFractions[globalFractionsOffset+global*8u+direction];}`
      });
      return {
        tile,
        directions,
        runoff,
        streams: tileStreams,
        accumulation: tileAccum,
        status: tileStatus,
        reachIds: tileReach,
        fractions: tileFractions
      };
    });
    graph.add(
      new GPUTerrainTiledFlow({
        id: 'four-owned-quadrant-d8',
        grid: grid.rasterGrid,
        routing: routing === 'd8' ? 'd8' : 'branching',
        maximumIterations: iterations,
        tiles: tileViews.map(
          ({
            tile,
            directions,
            fractions,
            runoff,
            streams: tileStreamView,
            accumulation: tileAccum,
            status: tileStatus,
            reachIds: tileReach
          }) => ({
            column: tile.column,
            row: tile.row,
            width: tile.width,
            height: tile.height,
            owned: tile.owned,
            flowDirections: directions,
            ...(routing === 'branching' ? {fractions} : {}),
            runoff,
            streams: tileStreamView,
            output: {accumulation: tileAccum, status: tileStatus, reachIds: tileReach}
          })
        )
      })
    );
    const tiledAccView = importBuffer(tiledAccumulation, 'float32', cellCount);
    const tiledStatusView = importBuffer(tiledStatus, 'uint32', cellCount);
    const tiledReachView = importBuffer(tiledReachIds, 'uint32', cellCount);
    const tiledSummaryView = importBuffer(tiledSummary, 'uint32', 6);
    const selectedReferenceView = routing === 'd8' ? accumulationView : branchingReferenceView;
    const branchingStatusView = importBuffer(branchingReferenceStatus, 'uint32', cellCount);
    const d8AccumulationStatusView = graph.createDataView(flowStatusTarget.buffer, {
      format: 'uint32',
      length: 1,
      byteOffset: 8
    });
    const selectedReferenceStatusView =
      routing === 'd8' ? d8AccumulationStatusView : branchingStatusView;
    tileViews.forEach(
      ({tile, accumulation: tileAccum, status: tileState, reachIds: tileReach}, tileIndex) => {
        const tileCellCount = tile.width * tile.height;
        addKernelPass(graph, {
          id: `assemble-owned-tile-${tileIndex}`,
          invocationCount: tileCellCount,
          bindings: [
            {name: 'tileAccum', view: tileAccum, type: 'f32', access: 'read'},
            {name: 'tileStatus', view: tileState, type: 'u32', access: 'read'},
            {name: 'tileReach', view: tileReach, type: 'u32', access: 'read'},
            {name: 'outputAccum', view: tiledAccView, type: 'f32', access: 'read_write'},
            {name: 'outputStatus', view: tiledStatusView, type: 'u32', access: 'read_write'},
            {name: 'outputReach', view: tiledReachView, type: 'u32', access: 'read_write'}
          ],
          declarations: `const GRID_WIDTH:u32=${width}u; const TILE_WIDTH:u32=${tile.width}u; const OWNED_COLUMN:u32=${tile.owned[0]}u; const OWNED_ROW:u32=${tile.owned[1]}u; const OWNED_WIDTH:u32=${tile.owned[2]}u; const OWNED_HEIGHT:u32=${tile.owned[3]}u;`,
          body: `let row=index/TILE_WIDTH;let column=index%TILE_WIDTH;let owned=column>=OWNED_COLUMN&&column<OWNED_COLUMN+OWNED_WIDTH&&row>=OWNED_ROW&&row<OWNED_ROW+OWNED_HEIGHT;if(owned){let globalRow=row+${tile.row}u;let globalColumn=column+${tile.column}u;let global=globalRow*GRID_WIDTH+globalColumn;outputAccum[outputAccumOffset+global]=tileAccum[tileAccumOffset+index];outputStatus[outputStatusOffset+global]=tileStatus[tileStatusOffset+index];outputReach[outputReachOffset+global]=tileReach[tileReachOffset+index];}`
        });
      }
    );
    addKernelPass(graph, {
      id: 'reduce-tiled-status-and-difference',
      invocationCount: 1,
      bindings: [
        {name: 'reference', view: selectedReferenceView, type: 'f32', access: 'read'},
        {name: 'referenceStatus', view: selectedReferenceStatusView, type: 'u32', access: 'read'},
        {name: 'directions', view: flowDirsView, type: 'u32', access: 'read'},
        {name: 'fractions', view: fractionView, type: 'f32', access: 'read'},
        {name: 'tiledValues', view: tiledAccView, type: 'f32', access: 'read'},
        {name: 'tiledStates', view: tiledStatusView, type: 'u32', access: 'read'},
        {name: 'summary', view: tiledSummaryView, type: 'u32', access: 'read_write'}
      ],
      declarations: `const GRID_WIDTH:u32=${width}u; const CELL_COUNT:u32=${cellCount}u; const CHECK_REFERENCE_STATUS:u32=${routing === 'branching' ? 1 : 0}u;`,
      body: `var resolved=0u;var unresolved=0u;var invalid=0u;var crossings=0u;var compared=0u;var maximumDifference=0.0;
for(var cell=0u;cell<CELL_COUNT;cell++){let state=tiledStates[tiledStatesOffset+cell];if(state==1u){resolved++;let first=reference[referenceOffset+cell];let second=tiledValues[tiledValuesOffset+cell];let referenceIndex=select(cell,0u,CHECK_REFERENCE_STATUS==0u);let referenceReady=referenceStatus[referenceStatusOffset+referenceIndex]==1u;if(referenceReady&&first==first&&second==second){compared++;maximumDifference=max(maximumDifference,abs(first-second));}}else if(state==2u){unresolved++;}else{invalid++;}
let row=cell/GRID_WIDTH;let column=cell%GRID_WIDTH;let code=directions[directionsOffset+cell];if(CHECK_REFERENCE_STATUS==0u){if((column==${TILE_SIZE - 1}u&&(code==1u||code==2u||code==128u))||(column==${TILE_SIZE}u&&(code==8u||code==16u||code==32u))||(row==${TILE_SIZE - 1}u&&(code==2u||code==4u||code==8u))||(row==${TILE_SIZE}u&&(code==32u||code==64u||code==128u))){crossings++;}}else{for(var direction=0u;direction<8u;direction++){let amount=fractions[fractionsOffset+cell*8u+direction];let crossesColumn=(column==${TILE_SIZE - 1}u&&(direction==0u||direction==1u||direction==7u))||(column==${TILE_SIZE}u&&(direction==3u||direction==4u||direction==5u));let crossesRow=(row==${TILE_SIZE - 1}u&&(direction==1u||direction==2u||direction==3u))||(row==${TILE_SIZE}u&&(direction==5u||direction==6u||direction==7u));if(amount>0.0&&(crossesColumn||crossesRow)){crossings++;}}}}
summary[summaryOffset]=resolved;summary[summaryOffset+1u]=unresolved;summary[summaryOffset+2u]=invalid;summary[summaryOffset+3u]=crossings;summary[summaryOffset+4u]=bitcast<u32>(maximumDifference);summary[summaryOffset+5u]=compared;`
    });

    addStorageCopyPass(
      graph,
      'readback-prominence',
      prominenceView,
      importBuffer(peakReadback, 'float32', cellCount * 3)
    );
    addStorageCopyPass(
      graph,
      'readback-key-saddle',
      saddleView,
      graph.createDataView(importBuffer(peakReadback, 'float32', cellCount * 3).buffer, {
        format: 'float32',
        length: cellCount,
        byteOffset: cellCount * 4
      })
    );
    addStorageCopyPass(
      graph,
      'readback-parent-peak',
      parentView,
      graph.createDataView(importBuffer(peakReadback, 'float32', cellCount * 3).buffer, {
        format: 'uint32',
        length: cellCount,
        byteOffset: cellCount * 8
      })
    );
    const stageStatus = (name: string) =>
      drainage.status.stages.find(stage => stage.stage === name)!.status;
    const statusReadbackView = importBuffer(readbackStatuses, 'uint32', 18);
    addStorageCopyPass(
      graph,
      'readback-flow-status',
      graph.createDataView(importBuffer(flowStatus, 'uint32', 3).buffer, {
        format: 'uint32',
        length: 3
      }),
      graph.createDataView(statusReadbackView.buffer, {
        format: 'uint32',
        length: 3
      })
    );
    addStorageCopyPass(
      graph,
      'readback-network-reaches',
      importBuffer(networkReachCount, 'uint32', 1),
      graph.createDataView(statusReadbackView.buffer, {
        format: 'uint32',
        length: 1,
        byteOffset: 12
      })
    );
    addStorageCopyPass(
      graph,
      'readback-network-vertices',
      importBuffer(networkVertexCount, 'uint32', 1),
      graph.createDataView(statusReadbackView.buffer, {
        format: 'uint32',
        length: 1,
        byteOffset: 16
      })
    );
    addStorageCopyPass(
      graph,
      'readback-network-overflow',
      importBuffer(networkOverflow, 'uint32', 1),
      graph.createDataView(statusReadbackView.buffer, {
        format: 'uint32',
        length: 1,
        byteOffset: 20
      })
    );
    addStorageCopyPass(
      graph,
      'readback-network-unresolved',
      importBuffer(networkUnresolved, 'uint32', 1),
      graph.createDataView(statusReadbackView.buffer, {
        format: 'uint32',
        length: 1,
        byteOffset: 24
      })
    );
    for (const [index, name] of [
      'stream-order',
      'catchments',
      'downstream-distance',
      'catchment-summary'
    ].entries()) {
      const stage = stageStatus(name);
      const scalar = stage.converged ?? stage.unresolvedCount ?? stage.overflow;
      if (scalar) {
        addStorageCopyPass(
          graph,
          `readback-${name}-status`,
          scalar,
          graph.createDataView(statusReadbackView.buffer, {
            format: 'uint32',
            length: 1,
            byteOffset: 28 + index * 4
          })
        );
      }
    }
    addStorageCopyPass(
      graph,
      'readback-tiled-summary',
      tiledSummaryView,
      graph.createDataView(statusReadbackView.buffer, {
        format: 'uint32',
        length: 6,
        byteOffset: 44
      })
    );
    addStorageCopyPass(
      graph,
      'readback-branching-reference-unresolved',
      importBuffer(branchingReferenceUnresolved, 'uint32', 1),
      graph.createDataView(statusReadbackView.buffer, {
        format: 'uint32',
        length: 1,
        byteOffset: 68
      })
    );
    const compiled = resources.track(await graph.compileAsync());
    let reader: SummaryReader;
    reader = new SummaryReader(
      resources,
      `terrain-boundaries-${revision}`,
      [
        {buffer: peakReadback, size: cellCount * 12},
        {buffer: catchmentReadback, size: cellCount * 16},
        {buffer: readbackStatuses, size: 72}
      ],
      bytes => {
        if (active?.revision !== revision) return;
        cachedProminence = new Float32Array(bytes, 0, cellCount).slice();
        cachedSaddles = new Float32Array(bytes, cellCount * 4, cellCount).slice();
        cachedParents = new Uint32Array(bytes, cellCount * 8, cellCount).slice();
        const summaryOffset = cellCount * 12;
        cachedCellCatchments = new Uint32Array(bytes, summaryOffset, cellCount).slice();
        cachedCatchmentIds = new Uint32Array(
          bytes,
          summaryOffset + cellCount * 4,
          cellCount
        ).slice();
        cachedCatchmentCounts = new Uint32Array(
          bytes,
          summaryOffset + cellCount * 8,
          cellCount
        ).slice();
        cachedMeanDistance = new Float32Array(
          bytes,
          summaryOffset + cellCount * 12,
          cellCount
        ).slice();
        const words = new Uint32Array(bytes, summaryOffset + cellCount * 16);
        cachedFlowStatus = words.slice(0, 3);
        const networkWordOffset = 3;
        const networkReachCount = words[networkWordOffset];
        const networkVertices = words[networkWordOffset + 1];
        const streamOrderConverged = words[7] === 1;
        const catchmentsConverged = words[8] === 1;
        const uncertainDistanceCount = words[9];
        const catchmentSummaryOverflow = words[10];
        cachedTiledSummary = words.slice(11, 17);
        cachedBranchingUnresolved = words[17];
        let finitePeakCount = 0;
        let elderCount = 0;
        let maximumFinite = -Infinity;
        selectedFinitePeak = -1;
        infiniteElderCount = 0;
        const elderCoordinates: number[] = [];
        for (let cell = 0; cell < cellCount; cell++) {
          const value = cachedProminence[cell];
          if (value === Infinity) {
            elderCount++;
            elderCoordinates.push(...cellPosition(cell));
          }
          if (Number.isFinite(value) && value > 0) {
            finitePeakCount++;
            if (value > maximumFinite) {
              maximumFinite = value;
              selectedFinitePeak = cell;
            }
          }
        }
        infiniteElderCount = elderCount;
        cachedFinitePeakCount = finitePeakCount;
        cachedElderCount = elderCount;
        if (elderCoordinates.length > 0) elderMarkers.write(Float32Array.from(elderCoordinates));
        selectedPeakText =
          selectedFinitePeak >= 0
            ? `largest finite peak ${formatCell(selectedFinitePeak)} · ${maximumFinite.toFixed(2)} m prominence · ${finitePeakCount} finite peaks · ${elderCount} infinite component elders`
            : `${finitePeakCount} finite peaks · ${elderCount} infinite component elder(s) · no positive finite prominence in this window`;
        const networkCapacityOverflow = words[networkWordOffset + 2];
        const unresolvedNetworkWalks = words[networkWordOffset + 3];
        drainageText = `Fill ${cachedFlowStatus[0] ? 'converged' : 'unresolved'} · flats ${cachedFlowStatus[1] ? 'converged' : 'unresolved'} · accumulation ${cachedFlowStatus[2] ? 'converged' : 'unresolved'} · order ${streamOrderConverged ? 'converged' : 'unresolved'} · catchments ${catchmentsConverged ? 'converged' : 'unresolved'} · distance uncertain ${uncertainDistanceCount} · summary overflow ${catchmentSummaryOverflow} · ${networkReachCount} reaches / ${networkVertices} vertices · overflow ${networkCapacityOverflow} · unresolved walks ${unresolvedNetworkWalks}`;
        const maximumDifference = new Float32Array(Uint32Array.of(cachedTiledSummary[4]).buffer)[0];
        tiledText = `${cachedTiledSummary[0]} owned cells resolved · ${cachedTiledSummary[1]} unresolved · ${cachedTiledSummary[2]} invalid · ${cachedTiledSummary[3]} ${routing === 'd8' ? 'D8 edges' : 'branch links'} cross a quadrant seam · ${routing === 'd8' ? 'D8 recipe' : `${branchMethod} whole-window reference; ${cachedBranchingUnresolved} unresolved`} · ${cachedTiledSummary[5]} cells compared · ${cachedTiledSummary[5] ? `maximum accumulation difference ${maximumDifference.toExponential(2)} cells` : 'difference unavailable without shared resolved cells'}`;
        updatePeak();
        ctx.setReadout('selectedPeak', selectedPeakText);
        ctx.setReadout('drainage', drainageText);
        ctx.setReadout('catchment', catchmentText());
        ctx.setReadout('tiledFlow', tiledText);
        acceptedRevision = revision;
        ctx.requestLayers();
      }
    );
    return {compiled, reader, revision};
  };

  active = await makeGraph(
    Number(ctx.options.iterations) as 4 | 32 | 128,
    ctx.options.routing,
    ctx.options.branchMethod
  );
  const swapCompiled = async (
    iterations: number,
    routing: TerrainBoundariesOptions['routing'],
    branchMethod: TerrainBoundariesOptions['branchMethod']
  ) => {
    const expectedRevision = graphRevision + 1;
    acceptedRevision = -1;
    cachedProminence.fill(NaN);
    selectedPeakText = 'Rebuilding tile graph…';
    drainageText = 'Rebuilding drainage and branching graphs…';
    tiledText = 'Rebuilding tile graph…';
    ctx.setReadout('selectedPeak', selectedPeakText);
    ctx.setReadout('drainage', drainageText);
    ctx.setReadout('catchment', 'Waiting for the rebuilt catchment summary…');
    ctx.setReadout('tiledFlow', tiledText);
    ctx.setChart('peakChart', null);
    await graphSwap.request({
      build: () => makeGraph(iterations, routing, branchMethod),
      commit: next => {
        const previous = active;
        active = next;
        if (previous) {
          previous.reader.release();
          resources.release(previous.compiled);
        }
        ctx.requestLayers();
        if (graphRevision !== expectedRevision) return;
      },
      discard: next => {
        next.reader.release();
        resources.release(next.compiled);
      }
    });
  };

  ctx.setReadout(
    'grid',
    `${width} × ${height} block-average cells · ${grid.groundCellSize[0].toFixed(1)} m nominal centre-row spacing · closed finite DEM window`
  );
  ctx.setReadout('selectedPeak', 'Waiting for GPU peak tree…');
  ctx.setReadout('drainage', 'Waiting for GPU D8 recipe…');
  ctx.setReadout('catchment', 'Waiting for GPU catchment summary…');
  ctx.setReadout('tiledFlow', 'Waiting for four owned tiles…');
  ctx.setChart('peakChart', null);
  return {
    getCompiledGraphs: () => (active ? [active.compiled] : []),
    encode(commandEncoder: CommandEncoder, frame) {
      if (!active) return;
      active.compiled.encode(commandEncoder, {parameters: undefined});
      if (acceptedRevision !== active.revision && !active.reader.isPending)
        active.reader.request(commandEncoder);
      active.reader.flush(commandEncoder);
      if (!frame.frameIndex || frame.frameIndex < 2) ctx.requestLayers();
    },
    getLayers() {
      const base = new SpatialAnalysisRasterLayer({
        id: 'terrain-boundaries-source',
        coordinateOrigin: [grid.origin[0], grid.origin[1], 0],
        gridSize: [width, height],
        bounds: grid.bounds,
        rowOrigin: 'north',
        values: elevation,
        valueFormat: 'float32',
        colormap: 'isolum',
        valueRange: [700, 2700],
        opacity: 0.83,
        noDataColor: [0, 0, 0, 0]
      });
      const layers: Layer[] = [base];
      const valid = acceptedRevision === active?.revision;
      const product = ctx.options.product;
      if (product === 'prominence') {
        layers.push(
          new SpatialAnalysisRasterLayer({
            id: 'finite-peak-prominence',
            coordinateOrigin: [grid.origin[0], grid.origin[1], 0],
            gridSize: [width, height],
            bounds: grid.bounds,
            rowOrigin: 'north',
            values: valid ? prominenceDisplay : null,
            valueFormat: 'float32',
            colormap: 'cividis',
            valueRange: [0, 500],
            opacity: 0.86,
            noDataColor: [0, 0, 0, 0],
            discardAtOrBelow: 0
          })
        );
        if (valid && selectedFinitePeak >= 0)
          layers.push(
            new SpatialAnalysisPointLayer({
              id: 'selected-finite-peak',
              coordinateOrigin: [grid.origin[0], grid.origin[1], 0],
              positions: selectedPeakMarker,
              instanceCount: 1,
              color: [255, 245, 210, 255],
              radiusPixels: 7,
              outlineColor: [28, 34, 40, 255],
              outlineWidthPixels: 1.5
            })
          );
        if (valid && infiniteElderCount > 0)
          layers.push(
            new SpatialAnalysisPointLayer({
              id: 'finite-window-component-elders',
              coordinateOrigin: [grid.origin[0], grid.origin[1], 0],
              positions: elderMarkers,
              instanceCount: infiniteElderCount,
              shape: 'diamond',
              radiusPixels: 7,
              color: [235, 106, 72, 255],
              outlineColor: [28, 34, 40, 255],
              outlineWidthPixels: 1.5
            })
          );
      } else if (product === 'streams') {
        layers.push(
          new SpatialAnalysisRasterLayer({
            id: 'd8-stream-order',
            coordinateOrigin: [grid.origin[0], grid.origin[1], 0],
            gridSize: [width, height],
            bounds: grid.bounds,
            rowOrigin: 'north',
            values: valid ? streamOrderDisplay : null,
            valueFormat: 'uint32',
            colormap: 'category',
            palette: [
              [194, 218, 225, 230],
              [48, 130, 166, 255],
              [16, 89, 126, 255],
              [9, 52, 92, 255],
              [232, 157, 75, 255]
            ],
            noDataColor: [0, 0, 0, 0],
            discardAtOrBelow: 0,
            opacity: 1
          })
        );
        if (valid)
          layers.push(
            new SpatialAnalysisSegmentLayer({
              id: 'd8-stream-reach-lines',
              coordinateOrigin: [grid.origin[0], grid.origin[1], 0],
              segments: reachSegments,
              instanceCount: cellCount,
              color: [17, 74, 104, 255],
              widthPixels: 1.1,
              outlineColor: [235, 244, 238, 200],
              outlineWidthPixels: 1.5
            })
          );
      } else {
        layers.push(
          new SpatialAnalysisRasterLayer({
            id: 'four-tile-owned-accumulation',
            coordinateOrigin: [grid.origin[0], grid.origin[1], 0],
            gridSize: [width, height],
            bounds: grid.bounds,
            rowOrigin: 'north',
            values: valid ? tiledAccumulation : null,
            valueFormat: 'float32',
            colormap: 'mako',
            valueRange: [1, 500],
            opacity: 0.85,
            noDataColor: [0, 0, 0, 0]
          })
        );
        layers.push(
          new SpatialAnalysisRasterLayer({
            id: 'owned-quadrant-boundaries',
            coordinateOrigin: [grid.origin[0], grid.origin[1], 0],
            gridSize: [width, height],
            bounds: grid.bounds,
            rowOrigin: 'north',
            values: valid ? tiledStatus : null,
            valueFormat: 'uint32',
            colormap: 'category',
            palette: [
              [0, 0, 0, 0],
              [251, 205, 93, 165],
              [214, 83, 72, 210]
            ],
            noDataColor: [0, 0, 0, 0],
            opacity: 0.9
          })
        );
        layers.push(
          new SpatialAnalysisSegmentLayer({
            id: 'owned-quadrant-seams',
            coordinateOrigin: [grid.origin[0], grid.origin[1], 0],
            segments: tileSeams,
            instanceCount: 2,
            color: [255, 240, 184, 230],
            widthPixels: 1.5,
            dashArray: [4, 3],
            cap: 'butt'
          })
        );
      }
      layers.push(
        new SpatialAnalysisSegmentLayer({
          id: 'analysis-window-outline',
          coordinateOrigin: [grid.origin[0], grid.origin[1], 0],
          segments: windowOutline,
          instanceCount: 4,
          color: [28, 36, 42, 220],
          widthPixels: 1.5,
          dashArray: [5, 3],
          cap: 'butt'
        })
      );
      layers.push(
        new SpatialAnalysisPointLayer({
          id: 'selected-terrain-cell',
          coordinateOrigin: [grid.origin[0], grid.origin[1], 0],
          positions: selectedCellMarker,
          instanceCount: 1,
          shape: 'square',
          radiusPixels: 5,
          color: [255, 245, 200, 255],
          outlineColor: [28, 34, 40, 255],
          outlineWidthPixels: 1.5
        })
      );
      return layers;
    },
    setOption(id, value) {
      if (id === 'iterations' || id === 'routing' || id === 'branchMethod') {
        const iterations = Number(id === 'iterations' ? value : ctx.options.iterations) as
          | 4
          | 32
          | 128;
        const routing =
          id === 'routing' ? (value as TerrainBoundariesOptions['routing']) : ctx.options.routing;
        const branchMethod =
          id === 'branchMethod'
            ? (value as TerrainBoundariesOptions['branchMethod'])
            : ctx.options.branchMethod;
        void swapCompiled(iterations, routing, branchMethod);
      }
      ctx.requestLayers();
    },
    onClick(info) {
      if (typeof info.coordinate?.[0] !== 'number' || typeof info.coordinate?.[1] !== 'number')
        return false;
      const projected = grid.projection.project(info.coordinate[0], info.coordinate[1]);
      const column = Math.floor((projected[0] - grid.bounds[0]) / grid.displayCellSize[0]);
      const row = Math.floor((grid.bounds[3] - projected[1]) / grid.displayCellSize[1]);
      if (column < 0 || column >= width || row < 0 || row >= height) return false;
      selectedCell = row * width + column;
      selectedCellMarker.write(Float32Array.from(cellPosition(selectedCell)));
      if (acceptedRevision === active?.revision) ctx.setReadout('catchment', catchmentText());
      if (
        acceptedRevision === active?.revision &&
        Number.isFinite(cachedProminence[selectedCell]) &&
        cachedProminence[selectedCell] > 0
      ) {
        selectedFinitePeak = selectedCell;
        updatePeak();
      }
      ctx.refreshTooltip();
      return true;
    },
    getTooltip(info) {
      if (typeof info.coordinate?.[0] !== 'number' || typeof info.coordinate?.[1] !== 'number')
        return null;
      const projected = grid.projection.project(info.coordinate[0], info.coordinate[1]);
      const column = Math.floor((projected[0] - grid.bounds[0]) / grid.displayCellSize[0]);
      const row = Math.floor((grid.bounds[3] - projected[1]) / grid.displayCellSize[1]);
      if (column < 0 || column >= width || row < 0 || row >= height) return null;
      const cell = row * width + column;
      if (acceptedRevision !== active?.revision)
        return {title: 'GPU outputs pending', rows: [{label: 'Cell', value: formatCell(cell)}]};
      const prominenceValue = cachedProminence[cell];
      return {
        title: formatCell(cell),
        rows: [
          {label: 'Elevation', value: `${grid.elevation[cell].toFixed(2)} m`},
          {
            label: 'Finite-window prominence',
            value: Number.isFinite(prominenceValue)
              ? `${prominenceValue.toFixed(2)} m`
              : prominenceValue === Infinity
                ? 'infinite component elder'
                : 'not a peak or missing'
          },
          {
            label: 'Key saddle',
            value: Number.isFinite(cachedSaddles[cell])
              ? `${cachedSaddles[cell].toFixed(2)} m`
              : cachedSaddles[cell] === -Infinity
                ? 'component elder (no merge saddle)'
                : 'not applicable'
          },
          {
            label: 'Parent peak',
            value: cachedParents[cell] === 0xffffffff ? 'none' : formatCell(cachedParents[cell])
          },
          {
            label: 'Catchment summary',
            value: catchmentSummaryForCell(cell).replace(`${formatCell(cell)} · `, '')
          }
        ]
      };
    },
    destroy() {
      active?.reader.release();
      resources.destroy();
    }
  };
}

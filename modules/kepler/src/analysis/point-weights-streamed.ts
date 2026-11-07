// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {KeplerAllocationError} from '../runtime/dataset-registry';
import {readBufferUint32} from '../runtime/gpu-readback';
import {Buffer, type Device} from '@luma.gl/core';
import {
  GPUCommandGraph,
  createTransientView,
  type CompiledGPUCommandGraph,
  type GraphDataView
} from '@luma-contributors/graph/gpu-core';
import {GPUParameterBuffer} from '@luma-contributors/graph/gpu-dataframe';
import {
  GPUNeighborSearch,
  type GPUNeighborSearchStructure,
  GPU_NEIGHBOR_SEARCH_PARAMETER_LENGTH,
  getGPUNeighborSearchParameterValues,
  planGPUSpatialWeightsChunks,
  type GPUChunkedSpatialWeights
} from '@luma-contributors/graph/gpu-spatial-analysis';
import {
  parseKeplerPointWeightsSettings,
  type KeplerPointWeightsSettings
} from '../recipes/analysis-recipe';
import {
  KEPLER_ANALYSIS_ROW_TILE_SIZE,
  getKeplerMaximumBindingElements,
  getKeplerNeighborGridSize
} from './analysis-scaling';
import {addCoordinatePacking, addKeplerPointWeights} from './point-weights';
import {
  getKeplerPointWeightBounds,
  getKeplerPointWeightSpatialContext
} from './point-weight-coordinates';

/**
 * Streamed distance-band build. A count pass (the ordinary self join with a one-slot output, whose
 * per-row neighbor counts stay exact) sizes row chunks that each fit one storage binding; every
 * chunk is then built as a cross join of its query rows against all targets into its own exactly
 * sized buffers. The result is a chunked CSR (`GPUChunkedSpatialWeights` after
 * {@link importKeplerStreamedPointWeights}) for consumers that dispatch per chunk, such as
 * `GPULocalPermutationTest`, so the total neighbor count is not bound by one buffer or by the
 * recipe's `neighborCapacity`. A cross join lists the query row itself at distance 0: every row
 * keeps one self slot with weight 1 and `hasSelfWeights: true`; consumers skip self slots.
 * Chunks are never repacked into one buffer.
 */
export type KeplerStreamedPointWeightsProps = {
  device: Device;
  rowCount: number;
  /** Float32 first and second coordinate columns (longitude/latitude degrees, or planar x/y). */
  firstCoordinate: Buffer;
  secondCoordinate: Buffer;
  /** Optional uint32 inclusion mask (nonzero includes the row), for example a selection. */
  coordinateValidityMask?: Buffer;
  settings: Extract<KeplerPointWeightsSettings, {kind: 'distance-band'}>;
  /** Override of the per-chunk slot limit (defaults to the device storage binding limit). */
  maximumSlotsPerChunk?: number;
  /** Search structure (results never depend on it); `'bvh'` supports planar settings only. */
  searchStructure?: GPUNeighborSearchStructure;
};

export type KeplerStreamedPointWeightsChunk = {
  rowStart: number;
  rowCount: number;
  /** Physical slots including one self slot per row. */
  slotCount: number;
  offsets: Buffer;
  neighbors: Buffer;
  weights: Buffer;
};

export type KeplerStreamedPointWeights = {
  rowCount: number;
  settings: Extract<KeplerPointWeightsSettings, {kind: 'distance-band'}>;
  chunks: readonly KeplerStreamedPointWeightsChunk[];
  /** Total slots across chunks, without the self slots. */
  neighborCount: number;
  /** Rows with null, non-finite or out-of-range coordinates. */
  invalidCount: number;
  coordinateMask: Buffer;
  destroy(): void;
};

const SELF_SLOT_COUNT_PER_ROW = 1;

function createStorageBuffer(device: Device, id: string, wordCount: number): Buffer {
  return device.createBuffer({
    id,
    byteLength: Math.max(wordCount, 1) * 4,
    usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
  });
}

function importBuffer<Format extends 'uint32' | 'float32' | 'float32x2'>(
  graph: GPUCommandGraph,
  id: string,
  buffer: Buffer,
  format: Format,
  length: number
): GraphDataView<Format> {
  return graph.createDataView(
    graph.importBuffer({id, byteLength: buffer.byteLength, usage: buffer.usage}, buffer),
    {format, length}
  );
}

async function runGraph(device: Device, graph: GPUCommandGraph): Promise<void> {
  const compiled = graph.compile() as CompiledGPUCommandGraph<undefined>;
  try {
    const commandEncoder = device.createCommandEncoder({id: `${graph.id}-encoder`});
    compiled.encode(commandEncoder, {parameters: undefined});
    device.submit(commandEncoder.finish());
  } finally {
    compiled.destroy();
  }
}

export async function buildKeplerStreamedPointWeights(
  props: KeplerStreamedPointWeightsProps
): Promise<KeplerStreamedPointWeights> {
  const {device, rowCount} = props;
  const settings = parseKeplerPointWeightsSettings(props.settings) as Extract<
    KeplerPointWeightsSettings,
    {kind: 'distance-band'}
  >;
  const owned: Buffer[] = [];
  const track = (buffer: Buffer) => {
    owned.push(buffer);
    return buffer;
  };
  const coordinateMask = track(createStorageBuffer(device, 'streamed-coordinate-mask', rowCount));
  try {
    // Phase 1: exact per-row counts from the ordinary self join with a one-slot output.
    const counts = createStorageBuffer(device, 'streamed-counts', rowCount);
    const countStatus = [0, 1, 2].map(index =>
      createStorageBuffer(device, `streamed-count-status-${index}`, 1)
    );
    const countGraph = new GPUCommandGraph(device, {id: 'streamed-weights-count'});
    const countCommon = {
      id: 'streamed-count',
      source: {datasetId: 'streamed', revision: 1},
      ...(props.searchStructure ? {searchStructure: props.searchStructure} : {}),
      ...(props.coordinateValidityMask
        ? {
            coordinateValidityMask: importBuffer(
              countGraph,
              'validity',
              props.coordinateValidityMask,
              'uint32',
              rowCount
            )
          }
        : {}),
      coordinateMask: importBuffer(countGraph, 'mask', coordinateMask, 'uint32', rowCount),
      output: {
        offsets: importBuffer(
          countGraph,
          'offsets',
          createStorageBuffer(device, 'streamed-count-offsets', rowCount + 1),
          'uint32',
          rowCount + 1
        ),
        neighbors: importBuffer(
          countGraph,
          'neighbors',
          createStorageBuffer(device, 'streamed-count-neighbors', 1),
          'uint32',
          1
        ),
        weights: importBuffer(
          countGraph,
          'weights',
          createStorageBuffer(device, 'streamed-count-weights', 1),
          'float32',
          1
        ),
        neighborCounts: importBuffer(countGraph, 'counts', counts, 'uint32', rowCount),
        status: {
          overflow: importBuffer(countGraph, 'overflow', countStatus[0], 'uint32', 1),
          requiredCount: importBuffer(countGraph, 'required', countStatus[1], 'uint32', 1),
          invalidCount: importBuffer(countGraph, 'invalid', countStatus[2], 'uint32', 1)
        }
      }
    };
    const countFirst = importBuffer(
      countGraph,
      'first',
      props.firstCoordinate,
      'float32',
      rowCount
    );
    const countSecond = importBuffer(
      countGraph,
      'second',
      props.secondCoordinate,
      'float32',
      rowCount
    );
    const countSettings = {...settings, neighborCapacity: 1};
    const countWeights = addKeplerPointWeights(
      countGraph,
      countSettings.coordinateSpace === 'planar'
        ? {...countCommon, settings: countSettings, x: countFirst, y: countSecond}
        : {...countCommon, settings: countSettings, longitude: countFirst, latitude: countSecond}
    );
    await runGraph(device, countGraph);
    countWeights.destroyParameters();
    const rowNeighborCounts = await readBufferUint32(counts, rowCount);
    const invalidCount = (await readBufferUint32(countStatus[2], 1))[0];
    counts.destroy();
    countStatus.forEach(buffer => buffer.destroy());

    // Phase 2: chunk plan; every row also reserves its self slot of the cross join.
    const maximumSlotsPerChunk =
      props.maximumSlotsPerChunk ?? getKeplerMaximumBindingElements(device.limits, 4);
    const rowSlotCounts = Uint32Array.from(
      rowNeighborCounts,
      count => count + SELF_SLOT_COUNT_PER_ROW
    );
    const plan = planGPUSpatialWeightsChunks(rowSlotCounts, maximumSlotsPerChunk);
    let neighborCount = 0;
    for (const count of rowNeighborCounts) neighborCount += count;

    // Phase 3: one cross-join graph per chunk, into exactly sized buffers.
    const chunks: KeplerStreamedPointWeightsChunk[] = [];
    for (const [chunkIndex, chunkPlan] of plan.entries()) {
      // A single row larger than the binding limit cannot be bound; the build refuses it here.
      if (chunkPlan.slotCount > maximumSlotsPerChunk)
        throw new KeplerAllocationError('capacity', {
          message: 'neighbor slots exceed the device storage binding limit'
        });
      const chunk: KeplerStreamedPointWeightsChunk = {
        rowStart: chunkPlan.rowStart,
        rowCount: chunkPlan.rowCount,
        slotCount: chunkPlan.slotCount,
        offsets: track(
          createStorageBuffer(device, `streamed-offsets-${chunkIndex}`, chunkPlan.rowCount + 1)
        ),
        neighbors: track(
          createStorageBuffer(device, `streamed-neighbors-${chunkIndex}`, chunkPlan.slotCount)
        ),
        weights: track(
          createStorageBuffer(device, `streamed-weights-${chunkIndex}`, chunkPlan.slotCount)
        )
      };
      const status = [0, 1, 2].map(index =>
        createStorageBuffer(device, `streamed-status-${chunkIndex}-${index}`, 1)
      );
      const graph = new GPUCommandGraph(device, {id: `streamed-weights-chunk-${chunkIndex}`});
      const first = importBuffer(graph, 'first', props.firstCoordinate, 'float32', rowCount);
      const second = importBuffer(graph, 'second', props.secondCoordinate, 'float32', rowCount);
      const validity = props.coordinateValidityMask
        ? importBuffer(graph, 'validity', props.coordinateValidityMask, 'uint32', rowCount)
        : undefined;
      const maskView = importBuffer(graph, 'mask', coordinateMask, 'uint32', rowCount);
      const positions = createTransientView(graph, 'streamed-positions', 'float32x2', rowCount);
      addCoordinatePacking(
        graph,
        'streamed-chunk',
        {
          firstCoordinate: first,
          secondCoordinate: second,
          coordinateValidityMask: validity,
          bounds: getKeplerPointWeightBounds(settings),
          planar: settings.coordinateSpace === 'planar'
        },
        positions,
        maskView
      );
      const queryPositions = graph.createDataView(positions.buffer, {
        format: 'float32x2',
        length: chunk.rowCount,
        byteOffset: positions.byteOffset + chunk.rowStart * 8
      });
      const queryMask = graph.createDataView(maskView.buffer, {
        format: 'uint32',
        length: chunk.rowCount,
        byteOffset: chunk.rowStart * 4
      });
      const spatialContext = getKeplerPointWeightSpatialContext(settings);
      const parameters = new GPUParameterBuffer(device, {
        id: `streamed-parameters-${chunkIndex}`,
        format: 'float32',
        length: GPU_NEIGHBOR_SEARCH_PARAMETER_LENGTH,
        values: getGPUNeighborSearchParameterValues({
          spatialContext,
          bounds: getKeplerPointWeightBounds(settings),
          radius: settings.radiusMeters,
          weightKind: 'binary',
          rowStandardize: false
        })
      });
      try {
        graph.add(
          new GPUNeighborSearch({
            id: 'streamed-chunk-search',
            spatialContext,
            mode: 'radius',
            positions,
            queryPositions,
            mask: maskView,
            queryMask,
            parameters: parameters.importToGraph(graph),
            gridSize: getKeplerNeighborGridSize(
              rowCount,
              settings.coordinateSpace === 'planar' ? settings.bounds : undefined
            ),
            ...(props.searchStructure ? {searchStructure: props.searchStructure} : {}),
            rowTileSize: KEPLER_ANALYSIS_ROW_TILE_SIZE,
            weightTransform: 'binary',
            rowStandardize: false,
            output: {
              offsets: importBuffer(graph, 'offsets', chunk.offsets, 'uint32', chunk.rowCount + 1),
              neighbors: importBuffer(
                graph,
                'neighbors',
                chunk.neighbors,
                'uint32',
                chunk.slotCount
              ),
              weights: importBuffer(graph, 'weights', chunk.weights, 'float32', chunk.slotCount),
              status: {
                overflow: importBuffer(graph, 'overflow', status[0], 'uint32', 1),
                requiredCount: importBuffer(graph, 'required', status[1], 'uint32', 1),
                invalidCount: importBuffer(graph, 'invalid', status[2], 'uint32', 1)
              }
            }
          })
        );
        await runGraph(device, graph);
        const overflow = (await readBufferUint32(status[0], 1))[0];
        // The exact count pass sized this chunk, so an overflow means the passes disagreed.
        if (overflow)
          throw new KeplerAllocationError('capacity', {
            message: 'streamed chunk overflowed its counted capacity'
          });
      } finally {
        parameters.destroy();
        status.forEach(buffer => buffer.destroy());
      }
      chunks.push(chunk);
    }
    return {
      rowCount,
      settings,
      chunks,
      neighborCount,
      invalidCount,
      coordinateMask,
      destroy: () => owned.forEach(buffer => buffer.destroy())
    };
  } catch (error) {
    owned.forEach(buffer => buffer.destroy());
    throw error;
  }
}

/** Imports the streamed chunks into a consumer graph as chunked weights; buffers stay caller-owned. */
export function importKeplerStreamedPointWeights<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  streamed: KeplerStreamedPointWeights
): GPUChunkedSpatialWeights {
  const spatialContext = getKeplerPointWeightSpatialContext(streamed.settings);
  return {
    rowCount: streamed.rowCount,
    spatialContext,
    information: {
      rowCount: streamed.rowCount,
      transform: 'binary',
      symmetric: 'unknown',
      hasSelfWeights: true,
      provenance: {
        source: 'GPUNeighborSearch',
        criterion: 'distance-band',
        radius: streamed.settings.radiusMeters
      }
    },
    chunks: streamed.chunks.map((chunk, index) => ({
      rowStart: chunk.rowStart,
      rowCount: chunk.rowCount,
      weights: {
        offsets: graph.createDataView(
          graph.importBuffer(
            {
              id: `streamed-offsets-${index}`,
              byteLength: chunk.offsets.byteLength,
              usage: chunk.offsets.usage
            },
            chunk.offsets
          ),
          {format: 'uint32', length: chunk.rowCount + 1}
        ),
        neighbors: graph.createDataView(
          graph.importBuffer(
            {
              id: `streamed-neighbors-${index}`,
              byteLength: chunk.neighbors.byteLength,
              usage: chunk.neighbors.usage
            },
            chunk.neighbors
          ),
          {format: 'uint32', length: chunk.slotCount}
        ),
        weights: graph.createDataView(
          graph.importBuffer(
            {
              id: `streamed-weights-${index}`,
              byteLength: chunk.weights.byteLength,
              usage: chunk.weights.usage
            },
            chunk.weights
          ),
          {format: 'float32', length: chunk.slotCount}
        )
      }
    }))
  };
}

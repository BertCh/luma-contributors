// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {
  GPUCommandGraph,
  type CompiledGPUCommandGraph,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  GPURasterGlobalHistogramMerge,
  GPURasterGlobalInitialize,
  GPURasterGlobalPercentile,
  GPURasterGlobalStatisticsMerge,
  GPURasterGaussianBlur,
  GPURasterCategoricalOverview,
  GPURasterOverview,
  GPURasterTileCache,
  GPURasterTileCoreExtract,
  GPURasterTileHaloAssembler,
  GPURasterTileHaloFill,
  GPURasterTileReader,
  makeRasterOverviewMetadata,
  type GPURasterBufferBand,
  type GPURasterDecodedTile,
  type GPURasterTileRequest,
  type GPURasterTileSource,
  type GPURasterTileSourceMetadata
} from '@luma.gl/graph/gpu-raster';
import {
  GPURasterReprojection,
  GPURasterTileReprojection,
  makeGPURasterMetadata,
  type GPURasterReprojectionSourceTile
} from '@luma-contributors/analysis/raster-analysis';
import {SpatialAnalysisResources} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import {GraphSwap} from '../../engine/graph-swap';
import {addKernelPass} from '../../engine/mode-kernels';
import {SpatialAnalysisSegmentLayer} from '../../engine/layers';
import {fetchBytes} from '../../data/loaders';
import {
  addTiledObservationTopology,
  getTiledObservationTopologyFocusBounds,
  TILED_TOPOLOGY_REGION_CAPACITY,
  TILED_TOPOLOGY_THRESHOLD,
  type TiledObservationTopologyGraph
} from './tiled-observation-topology';
import {TILED_TOPOLOGY_PALETTE} from './tiled-observation-topology-style';
import type {SceneContext, SceneInstance} from '../scene';
import {createGraphViewer, createUtmFrame, UtmRasterLayer} from './b16-common';

const SOURCE_SIZE = 250;
const CROP_SIZE = 128;
const CROP_COLUMN = 60;
const CROP_ROW = 50;
const TILE_SIZE = 64;
const TILE_COUNT = 4;
const CELL_COUNT = CROP_SIZE * CROP_SIZE;
const CELL_SIZE_METERS = 60;
const PROJECTED_BOUNDS = [662800, 4441040, 677800, 4456040] as const;

export type TiledObservationsOptions = {
  date: 'july' | 'september';
  step: 'ownership' | 'seams' | 'population' | 'histogram' | 'overview' | 'resampling';
  mapLayer:
    | 'source'
    | 'blur'
    | 'isolated'
    | 'difference'
    | 'wholecropDifference'
    | 'overview'
    | 'support'
    | 'categorical'
    | 'resampled'
    | 'resampleDifference'
    | 'topology';
  radius: number;
  noDataPolicy: 'ignore-renormalize' | 'propagate';
  edgePolicy: 'nodata' | 'clamp' | 'reflect';
  binCount: '32' | '128';
  percentile: '0' | '0.1' | '0.5' | '0.9' | '1';
  overviewScale: '2' | '4';
  categoricalPolicy: 'nearest' | 'mode';
  resampling: 'nearest' | 'bilinear-strict' | 'bilinear-renormalize';
  omitEast: boolean;
};

type SourceArrays = {values: Float32Array; validity: Uint32Array};
type TileWindow = {column: number; row: number; values: Float32Array; validity: Uint32Array};
type TileGraph = {
  generation: number;
  key: string;
  resources: SpatialAnalysisResources;
  compiled: CompiledGPUCommandGraph<void>;
  output: Buffer;
  validity: Buffer;
  isolated: Buffer;
  difference: Buffer;
  wholecropDifference: Buffer;
  overview: Buffer;
  overviewCount: Buffer;
  overviewSupport: Buffer;
  overviewDirect4: Buffer;
  overviewDirect4Count: Buffer;
  overviewCascaded4: Buffer;
  overviewCascaded4Count: Buffer;
  overviewAgreement: Buffer;
  overviewSumAgreement: Buffer;
  categorical: Buffer;
  categoricalValidity: Buffer;
  categoricalCount: Buffer;
  categoricalDisplay: Buffer;
  resampling: Buffer;
  resamplingValidity: Buffer;
  resamplingReference: Buffer;
  resamplingReferenceValidity: Buffer;
  resamplingDifference: Buffer;
  global: {extent: Buffer; count: Buffer; sum: Buffer; histogram: Buffer; overflow: Buffer};
  percentile: Buffer;
  reader: SummaryReader;
  windows: TileWindow[];
  haloLeases: Awaited<ReturnType<GPURasterTileHaloAssembler['acquire']>>[];
  topology: TiledObservationTopologyGraph;
};

function makeCrop(values: Float32Array, sourceValidity: Uint8Array): SourceArrays {
  const croppedValues = new Float32Array(CELL_COUNT).fill(Number.NaN);
  const validity = new Uint32Array(CELL_COUNT);
  for (let row = 0; row < CROP_SIZE; row++) {
    for (let column = 0; column < CROP_SIZE; column++) {
      const targetIndex = row * CROP_SIZE + column;
      const sourceIndex = (row + CROP_ROW) * SOURCE_SIZE + column + CROP_COLUMN;
      if (sourceValidity[sourceIndex] === 1 && Number.isFinite(values[sourceIndex])) {
        croppedValues[targetIndex] = values[sourceIndex];
        validity[targetIndex] = 1;
      }
    }
  }
  return {values: croppedValues, validity};
}

function makeReaderSource(source: SourceArrays, date: string): GPURasterTileSource {
  const metadata: GPURasterTileSourceMetadata = {
    id: `dixie-${date}-decoded-crop`,
    width: CROP_SIZE,
    height: CROP_SIZE,
    affine: [60, 0, 0, 0, -60, CROP_SIZE * 60],
    pixelInterpretation: 'area',
    bands: [{id: 'nir', format: 'float32'}],
    levels: [
      {
        level: 0,
        width: CROP_SIZE,
        height: CROP_SIZE,
        tileWidth: TILE_SIZE,
        tileHeight: TILE_SIZE,
        downsample: [1, 1]
      }
    ]
  };
  return {
    metadata,
    async readTile(
      request: GPURasterTileRequest,
      signal: AbortSignal
    ): Promise<GPURasterDecodedTile> {
      signal.throwIfAborted();
      const [left, top, right, bottom] = request.pixelBounds ?? [0, 0, CROP_SIZE, CROP_SIZE];
      const width = right - left;
      const height = bottom - top;
      const values = new Float32Array(width * height);
      const validity = new Uint32Array(width * height);
      for (let row = 0; row < height; row++) {
        const sourceOffset = (top + row) * CROP_SIZE + left;
        values.set(source.values.subarray(sourceOffset, sourceOffset + width), row * width);
        validity.set(source.validity.subarray(sourceOffset, sourceOffset + width), row * width);
      }
      return {
        level: 0,
        column: request.column ?? 0,
        row: request.row ?? 0,
        pixelBounds: [left, top, right, bottom],
        levelZeroBounds: [left, top, right, bottom],
        metadata: {
          ...makeGPURasterMetadata({
            width,
            height,
            cellSize: 60,
            origin: [left * 60, (CROP_SIZE - top) * 60]
          }),
          level: 0,
          levelZeroOrigin: [left, top]
        },
        bands: [{id: 'nir', format: 'float32', values, validity}]
      };
    }
  };
}

function makeWindows(source: SourceArrays): TileWindow[] {
  const windows: TileWindow[] = [];
  for (let tileRow = 0; tileRow < 2; tileRow++) {
    for (let tileColumn = 0; tileColumn < 2; tileColumn++) {
      const values = new Float32Array(TILE_SIZE * TILE_SIZE);
      const validity = new Uint32Array(TILE_SIZE * TILE_SIZE);
      for (let row = 0; row < TILE_SIZE; row++) {
        const sourceOffset = (tileRow * TILE_SIZE + row) * CROP_SIZE + tileColumn * TILE_SIZE;
        values.set(source.values.subarray(sourceOffset, sourceOffset + TILE_SIZE), row * TILE_SIZE);
        validity.set(
          source.validity.subarray(sourceOffset, sourceOffset + TILE_SIZE),
          row * TILE_SIZE
        );
      }
      windows.push({column: tileColumn, row: tileRow, values, validity});
    }
  }
  return windows;
}

export async function createTiledObservations(
  ctx: SceneContext<TiledObservationsOptions>
): Promise<SceneInstance<TiledObservationsOptions>> {
  const dataset = ctx.datasets.get('dixie-spectral');
  ctx.setStatus('Loading calibrated reflectance and strict Sentinel validity');
  const [julyRaster, septemberRaster, julyMaskBytes, septemberMaskBytes] = await Promise.all([
    dataset.loadRaster('nir_before', ctx.signal),
    dataset.loadRaster('nir_after', ctx.signal),
    fetchBytes(dataset.fileUrl('validity_before.bin'), ctx.signal),
    fetchBytes(dataset.fileUrl('validity_after.bin'), ctx.signal)
  ]);
  const sourceByDate = {
    july: makeCrop(Float32Array.from(julyRaster.values), new Uint8Array(julyMaskBytes)),
    september: makeCrop(
      Float32Array.from(septemberRaster.values),
      new Uint8Array(septemberMaskBytes)
    )
  };
  const frame = createUtmFrame(dataset.defaultOrigin, {
    boundsProjected: [
      PROJECTED_BOUNDS[0] + CROP_COLUMN * CELL_SIZE_METERS,
      PROJECTED_BOUNDS[1] + (SOURCE_SIZE - CROP_ROW - CROP_SIZE) * CELL_SIZE_METERS,
      PROJECTED_BOUNDS[0] + (CROP_COLUMN + CROP_SIZE) * CELL_SIZE_METERS,
      PROJECTED_BOUNDS[3] - CROP_ROW * CELL_SIZE_METERS
    ],
    cellSizeM: CELL_SIZE_METERS,
    width: CROP_SIZE,
    height: CROP_SIZE
  });
  const [centerLongitude, centerLatitude] = frame.cellToLngLat(CROP_SIZE / 2, CROP_SIZE / 2);
  ctx.flyTo(
    {longitude: centerLongitude, latitude: centerLatitude, zoom: 12.5, pitch: 0},
    {transitionMs: 0}
  );
  const rootResources = new SpatialAnalysisResources(ctx.device, 'tiled-observations');
  const graphSwap = rootResources.track(new GraphSwap());
  let active: TileGraph | null = null;
  let generation = 0;
  let acceptedTopologyGeneration = -1;
  let requestedKey = '';
  let dirty = true;
  let disposed = false;
  const compile = async (options: TiledObservationsOptions): Promise<TileGraph> => {
    const sceneGeneration = ++generation;
    const resources = new SpatialAnalysisResources(
      ctx.device,
      `tiled-observations-${sceneGeneration}`
    );
    const source = sourceByDate[options.date];
    const windows = makeWindows(source);
    const reader = new GPURasterTileReader(makeReaderSource(source, options.date));
    const cache = resources.track(
      new GPURasterTileCache({
        device: ctx.device,
        reader,
        maxTiles: TILE_COUNT,
        maxGraphs: 1,
        maxCpuBytes: CELL_COUNT * 8,
        maxGpuBytes: CELL_COUNT * 8
      })
    );
    const assembler = new GPURasterTileHaloAssembler(cache);
    const haloLeases = [];
    try {
      for (let tileIndex = 0; tileIndex < TILE_COUNT; tileIndex++) {
        const tileColumn = tileIndex % 2;
        const tileRow = Math.floor(tileIndex / 2);
        const minimumColumn = tileColumn * TILE_SIZE;
        const minimumRow = tileRow * TILE_SIZE;
        const halo = Number(options.radius);
        haloLeases.push(
          await assembler.acquire(
            {
              level: 0,
              column: tileColumn,
              row: tileRow,
              pixelBounds: [
                minimumColumn,
                minimumRow,
                minimumColumn + TILE_SIZE,
                minimumRow + TILE_SIZE
              ],
              bandIds: ['nir'],
              stages: [{requiredHalo: halo}]
            },
            ctx.signal
          )
        );
      }
      const output = resources.createBuffer(
        'tiled-output',
        new Float32Array(CELL_COUNT).fill(Number.NaN)
      );
      const validity = resources.createBuffer('tiled-output-validity', CELL_COUNT * 4);
      const isolated = resources.createBuffer(
        'isolated-output',
        new Float32Array(CELL_COUNT).fill(Number.NaN)
      );
      const difference = resources.createBuffer(
        'seam-difference',
        new Float32Array(CELL_COUNT).fill(Number.NaN)
      );
      const wholecropDifference = resources.createBuffer(
        'wholecrop-seam-difference',
        new Float32Array(CELL_COUNT).fill(Number.NaN)
      );
      const wholecropOutput = resources.createBuffer('wholecrop-gaussian-output', CELL_COUNT * 4);
      const wholecropValidity = resources.createBuffer(
        'wholecrop-gaussian-validity',
        CELL_COUNT * 4
      );
      const overviewWidth = CROP_SIZE / Number(options.overviewScale);
      const overview = resources.createBuffer(
        'weighted-overview',
        new Float32Array(overviewWidth * overviewWidth).fill(Number.NaN)
      );
      const overviewCount = resources.createBuffer(
        'weighted-overview-count',
        overviewWidth * overviewWidth * 4
      );
      const overviewSupport = resources.createBuffer(
        'weighted-overview-support-display',
        new Float32Array(overviewWidth * overviewWidth).fill(Number.NaN)
      );
      const global = {
        extent: resources.createBuffer('global-extent', 8),
        count: resources.createBuffer('global-count', 4),
        sum: resources.createBuffer('global-sum', 4),
        histogram: resources.createBuffer('global-histogram', Number(options.binCount) * 4),
        overflow: resources.createBuffer('global-overflow', 4)
      };
      const percentile = resources.createBuffer('global-percentile', 4);
      const percentileValidity = resources.createBuffer('global-percentile-validity', 4);
      const resampling = resources.createBuffer('tile-resampling', CELL_COUNT * 4);
      const resamplingValidity = resources.createBuffer('tile-resampling-validity', CELL_COUNT * 4);
      const resamplingReference = resources.createBuffer(
        'untiled-resampling-reference',
        CELL_COUNT * 4
      );
      const resamplingReferenceValidity = resources.createBuffer(
        'untiled-resampling-reference-validity',
        CELL_COUNT * 4
      );
      const resamplingDifference = resources.createBuffer('resampling-difference', CELL_COUNT * 4);
      const graph = new GPUCommandGraph<void>(ctx.device, {
        id: `tiled-observations-${sceneGeneration}`
      });
      const view = createGraphViewer(graph);
      const floatView = (
        name: string,
        buffer: Buffer,
        length = CELL_COUNT
      ): GraphDataView<'float32'> => view(name, buffer, 'float32', length);
      const wordView = (
        name: string,
        buffer: Buffer,
        length = CELL_COUNT
      ): GraphDataView<'uint32'> => view(name, buffer, 'uint32', length);
      const topology = addTiledObservationTopology(
        graph,
        resources,
        {float: floatView, word: wordView},
        source
      );
      const wholecropSourceValues = resources.createBuffer(
        'wholecrop-source-values',
        source.values
      );
      const wholecropSourceValidity = resources.createBuffer(
        'wholecrop-source-validity',
        source.validity
      );
      const wholecropSourceBand: GPURasterBufferBand<'float32'> = {
        id: 'wholecrop-source',
        format: 'float32',
        storage: {
          kind: 'buffer',
          values: floatView('wholecrop-source-values-view', wholecropSourceValues)
        },
        validity: wordView('wholecrop-source-validity-view', wholecropSourceValidity)
      };
      new GPURasterGaussianBlur({
        id: `wholecrop-gaussian-reference-radius-${options.radius}`,
        width: CROP_SIZE,
        height: CROP_SIZE,
        input: wholecropSourceBand,
        output: floatView('wholecrop-gaussian-output-view', wholecropOutput),
        outputValidity: wordView('wholecrop-gaussian-validity-view', wholecropValidity),
        radius: Number(options.radius),
        sigma: Math.max(0.5, Number(options.radius) / 2),
        borderMode: options.edgePolicy,
        noDataPolicy: options.noDataPolicy
      }).addToGraph(graph);
      const accumulator = {
        extent: floatView('global-extent', global.extent, 2),
        count: wordView('global-count', global.count, 1),
        sum: floatView('global-sum', global.sum, 1),
        histogram: wordView('global-histogram', global.histogram, Number(options.binCount)),
        overflow: wordView('global-overflow', global.overflow, 1)
      };
      const tileOutputViews: GraphDataView<'float32'>[] = [];
      const tileValidityViews: GraphDataView<'uint32'>[] = [];
      const isolatedTileViews: GraphDataView<'float32'>[] = [];
      const globalSourceCoreBands: GPURasterBufferBand<'float32'>[] = [];
      new GPURasterGlobalInitialize({accumulator}).addToGraph(graph);
      for (const [tileIndex, haloLease] of haloLeases.entries()) {
        const plan = haloLease.plan;
        const sources = haloLease.tiles.flatMap((lease, sourceIndex) => {
          const band = lease.bands[0];
          return [
            {
              pixelBounds: plan.requests[sourceIndex].pixelBounds!,
              input: {
                id: band.id,
                format: 'float32' as const,
                scale: band.scale ?? 1,
                offset: band.offset ?? 0,
                noDataValue: band.noDataValue,
                storage: {
                  kind: 'buffer' as const,
                  values: view(
                    `tile-${tileIndex}-source-${sourceIndex}`,
                    band.buffer,
                    'float32',
                    band.buffer.byteLength / 4
                  )
                },
                validity: band.validity
                  ? view(
                      `tile-${tileIndex}-mask-${sourceIndex}`,
                      band.validity,
                      'uint32',
                      band.validity.byteLength / 4
                    )
                  : undefined
              }
            }
          ];
        });
        const assembledBuffer = resources.createBuffer(
          `tile-${tileIndex}-halo`,
          plan.width * plan.height * 4
        );
        const assembledValidityBuffer = resources.createBuffer(
          `tile-${tileIndex}-halo-validity`,
          plan.width * plan.height * 4
        );
        const assembledView = floatView(
          `tile-${tileIndex}-halo-view`,
          assembledBuffer,
          plan.width * plan.height
        );
        const assembledValidityView = wordView(
          `tile-${tileIndex}-halo-validity-view`,
          assembledValidityBuffer,
          plan.width * plan.height
        );
        new GPURasterTileHaloFill({
          id: `tile-${tileIndex}-halo-fill`,
          pixelBounds: plan.availablePixelBounds,
          sources,
          output: assembledView,
          outputValidity: assembledValidityView
        }).addToGraph(graph);
        const blurredBuffer = resources.createBuffer(
          `tile-${tileIndex}-blur`,
          plan.width * plan.height * 4
        );
        const blurredValidityBuffer = resources.createBuffer(
          `tile-${tileIndex}-blur-validity`,
          plan.width * plan.height * 4
        );
        const input: GPURasterBufferBand<'float32'> = {
          id: `tile-${tileIndex}-assembled`,
          format: 'float32',
          storage: {kind: 'buffer', values: assembledView},
          validity: assembledValidityView
        };
        const blurProps = {
          id: `tile-${tileIndex}-gaussian-radius-${options.radius}`,
          width: plan.width,
          height: plan.height,
          input,
          output: floatView(`tile-${tileIndex}-blur-view`, blurredBuffer, plan.width * plan.height),
          outputValidity: wordView(
            `tile-${tileIndex}-blur-mask-view`,
            blurredValidityBuffer,
            plan.width * plan.height
          ),
          radius: Number(options.radius),
          sigma: Math.max(0.5, Number(options.radius) / 2),
          borderMode: options.edgePolicy,
          noDataPolicy: options.noDataPolicy
        } as const;
        new GPURasterGaussianBlur(blurProps).addToGraph(graph);
        const isolatedBuffer = resources.createBuffer(
          `tile-${tileIndex}-isolated-blur`,
          TILE_SIZE * TILE_SIZE * 4
        );
        const isolatedValidityBuffer = resources.createBuffer(
          `tile-${tileIndex}-isolated-validity`,
          TILE_SIZE * TILE_SIZE * 4
        );
        const isolatedInput = {
          id: `tile-${tileIndex}-core-only`,
          format: 'float32' as const,
          storage: {
            kind: 'buffer' as const,
            values: floatView(
              `tile-${tileIndex}-core-source`,
              haloLease.core.bands[0].buffer,
              TILE_SIZE * TILE_SIZE
            )
          },
          validity: haloLease.core.bands[0].validity
            ? wordView(
                `tile-${tileIndex}-core-source-validity`,
                haloLease.core.bands[0].validity,
                TILE_SIZE * TILE_SIZE
              )
            : undefined
        };
        new GPURasterGaussianBlur({
          ...blurProps,
          id: `tile-${tileIndex}-deliberately-isolated-core`,
          width: TILE_SIZE,
          height: TILE_SIZE,
          input: isolatedInput,
          output: floatView(
            `tile-${tileIndex}-isolated-view`,
            isolatedBuffer,
            TILE_SIZE * TILE_SIZE
          ),
          outputValidity: wordView(
            `tile-${tileIndex}-isolated-mask-view`,
            isolatedValidityBuffer,
            TILE_SIZE * TILE_SIZE
          ),
          borderMode: options.edgePolicy
        }).addToGraph(graph);
        isolatedTileViews.push(
          floatView(`tile-${tileIndex}-isolated-stitch-view`, isolatedBuffer, TILE_SIZE * TILE_SIZE)
        );
        const tileOutputBuffer = resources.createBuffer(
          `tile-${tileIndex}-owned-output`,
          TILE_SIZE * TILE_SIZE * 4
        );
        const tileOutputValidityBuffer = resources.createBuffer(
          `tile-${tileIndex}-owned-output-validity`,
          TILE_SIZE * TILE_SIZE * 4
        );
        const tileOutput = floatView(
          `tile-${tileIndex}-owned-output-view`,
          tileOutputBuffer,
          TILE_SIZE * TILE_SIZE
        );
        const tileOutputValidity = wordView(
          `tile-${tileIndex}-owned-output-validity-view`,
          tileOutputValidityBuffer,
          TILE_SIZE * TILE_SIZE
        );
        new GPURasterTileCoreExtract({
          id: `tile-${tileIndex}-core-extract`,
          availablePixelBounds: plan.availablePixelBounds,
          corePixelBounds: plan.corePixelBounds,
          input: {
            id: `tile-${tileIndex}-processed-halo`,
            format: 'float32',
            storage: {
              kind: 'buffer',
              values: floatView(
                `tile-${tileIndex}-blur-input`,
                blurredBuffer,
                plan.width * plan.height
              )
            },
            validity: wordView(
              `tile-${tileIndex}-blur-input-validity`,
              blurredValidityBuffer,
              plan.width * plan.height
            )
          },
          output: tileOutput,
          outputValidity: tileOutputValidity
        }).addToGraph(graph);
        tileOutputViews.push(tileOutput);
        tileValidityViews.push(tileOutputValidity);
        const ownedSource = haloLease.core.bands[0];
        const ownedSourceBand: GPURasterBufferBand<'float32'> = {
          id: `tile-${tileIndex}-source-core`,
          format: 'float32',
          storage: {
            kind: 'buffer',
            values: floatView(
              `tile-${tileIndex}-owned-source-values`,
              ownedSource.buffer,
              TILE_SIZE * TILE_SIZE
            )
          },
          validity: ownedSource.validity
            ? wordView(
                `tile-${tileIndex}-owned-source-validity`,
                ownedSource.validity,
                TILE_SIZE * TILE_SIZE
              )
            : undefined
        };
        new GPURasterGlobalStatisticsMerge({
          id: `global-statistics-core-${tileIndex}`,
          width: TILE_SIZE,
          height: TILE_SIZE,
          input: ownedSourceBand,
          accumulator
        }).addToGraph(graph);
        globalSourceCoreBands.push(ownedSourceBand);
      }
      // The first pass fixes the complete global range before any tile contributes histogram bins.
      for (const [tileIndex, sourceCore] of globalSourceCoreBands.entries()) {
        new GPURasterGlobalHistogramMerge({
          id: `global-histogram-core-${tileIndex}`,
          width: TILE_SIZE,
          height: TILE_SIZE,
          input: sourceCore,
          accumulator
        }).addToGraph(graph);
      }
      new GPURasterGlobalPercentile({
        accumulator,
        percentile: Number(options.percentile),
        output: floatView('global-percentile-output', percentile, 1),
        outputValidity: wordView('global-percentile-validity-output', percentileValidity, 1)
      }).addToGraph(graph);
      const mosaic = view('tiled-output-mosaic', output, 'float32');
      const mosaicValidity = view('tiled-output-mosaic-validity', validity, 'uint32');
      const isolatedMosaic = view('isolated-mosaic', isolated, 'float32');
      const differences = view('seam-differences', difference, 'float32');
      for (const [tileIndex] of tileOutputViews.entries()) {
        const tileColumn = tileIndex % 2;
        const tileRow = Math.floor(tileIndex / 2);
        const source = tileOutputViews[tileIndex];
        const sourceValidity = tileValidityViews[tileIndex];
        addKernelPass(graph, {
          id: `stitch-core-${tileIndex}`,
          invocationCount: TILE_SIZE * TILE_SIZE,
          bindings: [
            {name: 'tile', view: source, type: 'f32', access: 'read'},
            {name: 'tileValidity', view: sourceValidity, type: 'u32', access: 'read'},
            {name: 'mosaic', view: mosaic, type: 'f32', access: 'read_write'},
            {name: 'mosaicValidity', view: mosaicValidity, type: 'u32', access: 'read_write'},
            {name: 'isolated', view: isolatedTileViews[tileIndex], type: 'f32', access: 'read'},
            {name: 'isolatedMosaic', view: isolatedMosaic, type: 'f32', access: 'read_write'},
            {name: 'difference', view: differences, type: 'f32', access: 'read_write'}
          ],
          body: `let localRow=index/${TILE_SIZE}u;let localColumn=index%${TILE_SIZE}u;let mosaicIndex=(${tileRow * TILE_SIZE}u+localRow)*${CROP_SIZE}u+${tileColumn * TILE_SIZE}u+localColumn;mosaic[mosaicOffset+mosaicIndex]=tile[tileOffset+index];mosaicValidity[mosaicValidityOffset+mosaicIndex]=tileValidity[tileValidityOffset+index];isolatedMosaic[isolatedMosaicOffset+mosaicIndex]=isolated[isolatedOffset+index];difference[differenceOffset+mosaicIndex]=tile[tileOffset+index]-isolated[isolatedOffset+index];`
        });
      }
      addKernelPass(graph, {
        id: 'tiled-wholecrop-gaussian-difference',
        invocationCount: CELL_COUNT,
        bindings: [
          {name: 'tiled', view: mosaic, type: 'f32', access: 'read'},
          {name: 'tiledValidity', view: mosaicValidity, type: 'u32', access: 'read'},
          {
            name: 'wholecrop',
            view: floatView('wholecrop-difference-input', wholecropOutput),
            type: 'f32',
            access: 'read'
          },
          {
            name: 'wholecropValidity',
            view: wordView('wholecrop-difference-validity', wholecropValidity),
            type: 'u32',
            access: 'read'
          },
          {
            name: 'difference',
            view: floatView('wholecrop-difference-output', wholecropDifference),
            type: 'f32',
            access: 'read_write'
          }
        ],
        body: 'let valid=tiledValidity[tiledValidityOffset+index]!=0u&&wholecropValidity[wholecropValidityOffset+index]!=0u;difference[differenceOffset+index]=select(bitcast<f32>(0x7fc00000u | (index & 0u)),tiled[tiledOffset+index]-wholecrop[wholecropOffset+index],valid);'
      });
      const rasterMetadata = makeGPURasterMetadata({
        width: CROP_SIZE,
        height: CROP_SIZE,
        cellSize: 60,
        origin: [0, CROP_SIZE * 60]
      });
      const destinationMetadata = {
        ...rasterMetadata,
        affine: [60, 0, 30, 0, -60, CROP_SIZE * 60 - 30] as const
      };
      const reprojectionSources: GPURasterReprojectionSourceTile[] = haloLeases.flatMap(
        (haloLease, tileIndex) => {
          const tileColumn = tileIndex % 2;
          const tileRow = Math.floor(tileIndex / 2);
          if (options.omitEast && tileColumn === 1) return [];
          const band = haloLease.core.bands[0];
          return [
            {
              pixelBounds: [
                tileColumn * TILE_SIZE,
                tileRow * TILE_SIZE,
                (tileColumn + 1) * TILE_SIZE,
                (tileRow + 1) * TILE_SIZE
              ],
              input: {
                id: `reprojection-core-${tileIndex}`,
                format: 'float32',
                storage: {
                  kind: 'buffer',
                  values: floatView(
                    `reprojection-core-values-${tileIndex}`,
                    band.buffer,
                    TILE_SIZE * TILE_SIZE
                  )
                },
                validity: band.validity
                  ? wordView(
                      `reprojection-core-validity-${tileIndex}`,
                      band.validity,
                      TILE_SIZE * TILE_SIZE
                    )
                  : undefined
              }
            }
          ];
        }
      );
      const referenceValues = Float32Array.from(source.values);
      const referenceValidity = Uint32Array.from(source.validity);
      if (options.omitEast) {
        for (let row = 0; row < CROP_SIZE; row++) {
          referenceValidity.fill(0, row * CROP_SIZE + TILE_SIZE, (row + 1) * CROP_SIZE);
        }
      }
      const referenceValueBuffer = resources.createBuffer(
        'reprojection-reference-source',
        referenceValues
      );
      const referenceValidityBuffer = resources.createBuffer(
        'reprojection-reference-source-validity',
        referenceValidity
      );
      const sourceReferenceBand: GPURasterBufferBand<'float32'> = {
        id: 'reprojection-reference-source',
        format: 'float32',
        storage: {
          kind: 'buffer',
          values: floatView('reprojection-reference-source-view', referenceValueBuffer)
        },
        validity: wordView('reprojection-reference-source-validity-view', referenceValidityBuffer)
      };
      const resamplingOutput = {
        values: floatView('resampling-output', resampling),
        validity: wordView('resampling-output-validity', resamplingValidity)
      };
      const referenceOutput = {
        values: floatView('resampling-reference-output', resamplingReference),
        validity: wordView('resampling-reference-output-validity', resamplingReferenceValidity)
      };
      const interpolationMethod = options.resampling === 'nearest' ? 'nearest' : 'bilinear';
      const interpolationNoDataPolicy =
        options.resampling === 'bilinear-strict' ? 'strict' : 'renormalize';
      graph.add(
        resources.track(
          new GPURasterTileReprojection({
            id: 'half-cell-shift-tiled-reprojection',
            sourceGrid: rasterMetadata,
            destinationGrid: destinationMetadata,
            sources: reprojectionSources,
            method: interpolationMethod,
            noDataPolicy: interpolationNoDataPolicy,
            output: resamplingOutput
          })
        )
      );
      graph.add(
        resources.track(
          new GPURasterReprojection({
            id: 'half-cell-shift-untiled-reference',
            sourceGrid: rasterMetadata,
            destinationGrid: destinationMetadata,
            input: sourceReferenceBand,
            method: interpolationMethod,
            noDataPolicy: interpolationNoDataPolicy,
            output: referenceOutput
          })
        )
      );
      addKernelPass(graph, {
        id: 'tiled-resampling-reference-difference',
        invocationCount: CELL_COUNT,
        bindings: [
          {name: 'tiled', view: resamplingOutput.values, type: 'f32', access: 'read'},
          {name: 'tiledValidity', view: resamplingOutput.validity, type: 'u32', access: 'read'},
          {name: 'reference', view: referenceOutput.values, type: 'f32', access: 'read'},
          {name: 'referenceValidity', view: referenceOutput.validity, type: 'u32', access: 'read'},
          {
            name: 'difference',
            view: floatView('resampling-difference-view', resamplingDifference),
            type: 'f32',
            access: 'read_write'
          }
        ],
        body: 'let valid=tiledValidity[tiledValidityOffset+index]!=0u&&referenceValidity[referenceValidityOffset+index]!=0u;difference[differenceOffset+index]=select(bitcast<f32>(0x7fc00000u | (index & 0u)),tiled[tiledOffset+index]-reference[referenceOffset+index],valid);'
      });
      const overviewMetadata = makeRasterOverviewMetadata(
        rasterMetadata,
        Number(options.overviewScale)
      );
      new GPURasterOverview({
        metadata: rasterMetadata,
        scale: Number(options.overviewScale),
        input: {
          id: 'overview-source',
          format: 'float32',
          storage: {kind: 'buffer', values: mosaic},
          validity: mosaicValidity
        },
        output: view(
          'weighted-overview-output',
          overview,
          'float32',
          overviewMetadata.width * overviewMetadata.height
        ),
        outputValidity: view(
          'weighted-overview-validity',
          resources.createBuffer('weighted-overview-validity', overviewWidth * overviewWidth * 4),
          'uint32',
          overviewWidth * overviewWidth
        ),
        sum: view(
          'weighted-overview-sum',
          resources.createBuffer('weighted-overview-sum', overviewWidth * overviewWidth * 4),
          'float32',
          overviewWidth * overviewWidth
        ),
        validCount: view(
          'weighted-overview-count',
          overviewCount,
          'uint32',
          overviewWidth * overviewWidth
        )
      }).addToGraph(graph);
      addKernelPass(graph, {
        id: 'overview-support-display',
        invocationCount: overviewWidth * overviewWidth,
        bindings: [
          {
            name: 'support',
            view: view(
              'overview-support-source',
              overviewCount,
              'uint32',
              overviewWidth * overviewWidth
            ),
            type: 'u32',
            access: 'read'
          },
          {
            name: 'display',
            view: view(
              'overview-support-display-view',
              overviewSupport,
              'float32',
              overviewWidth * overviewWidth
            ),
            type: 'f32',
            access: 'read_write'
          }
        ],
        body: 'display[displayOffset+index]=f32(support[supportOffset+index]);'
      });
      const factorTwoMetadata = makeRasterOverviewMetadata(rasterMetadata, 2);
      const factorFourMetadata = makeRasterOverviewMetadata(rasterMetadata, 4);
      const factorTwoCount = factorTwoMetadata.width * factorTwoMetadata.height;
      const factorFourCount = factorFourMetadata.width * factorFourMetadata.height;
      const levelTwoMean = resources.createBuffer('overview-level-two-mean', factorTwoCount * 4);
      const levelTwoSum = resources.createBuffer('overview-level-two-sum', factorTwoCount * 4);
      const levelTwoValidCount = resources.createBuffer(
        'overview-level-two-valid-count',
        factorTwoCount * 4
      );
      const levelTwoValidity = resources.createBuffer(
        'overview-level-two-validity',
        factorTwoCount * 4
      );
      const directFour = resources.createBuffer('overview-direct-four-mean', factorFourCount * 4);
      const directFourSum = resources.createBuffer('overview-direct-four-sum', factorFourCount * 4);
      const directFourCount = resources.createBuffer(
        'overview-direct-four-count',
        factorFourCount * 4
      );
      const directFourValidity = resources.createBuffer(
        'overview-direct-four-validity',
        factorFourCount * 4
      );
      const cascadedFour = resources.createBuffer(
        'overview-cascaded-four-mean',
        factorFourCount * 4
      );
      const cascadedFourSum = resources.createBuffer(
        'overview-cascaded-four-sum',
        factorFourCount * 4
      );
      const cascadedFourCount = resources.createBuffer(
        'overview-cascaded-four-count',
        factorFourCount * 4
      );
      const cascadedFourValidity = resources.createBuffer(
        'overview-cascaded-four-validity',
        factorFourCount * 4
      );
      const overviewAgreement = resources.createBuffer(
        'overview-direct-cascaded-difference',
        factorFourCount * 4
      );
      const overviewSumAgreement = resources.createBuffer(
        'overview-direct-cascaded-sum-difference',
        factorFourCount * 4
      );
      new GPURasterOverview({
        id: 'overview-level-two-for-cascade',
        metadata: rasterMetadata,
        scale: 2,
        input: {
          id: 'overview-cascade-source',
          format: 'float32',
          storage: {kind: 'buffer', values: mosaic},
          validity: mosaicValidity
        },
        output: floatView('overview-level-two-mean-view', levelTwoMean, factorTwoCount),
        outputValidity: wordView(
          'overview-level-two-validity-view',
          levelTwoValidity,
          factorTwoCount
        ),
        sum: floatView('overview-level-two-sum-view', levelTwoSum, factorTwoCount),
        validCount: wordView('overview-level-two-count-view', levelTwoValidCount, factorTwoCount)
      }).addToGraph(graph);
      new GPURasterOverview({
        id: 'overview-direct-factor-four',
        metadata: rasterMetadata,
        scale: 4,
        input: {
          id: 'overview-direct-source',
          format: 'float32',
          storage: {kind: 'buffer', values: mosaic},
          validity: mosaicValidity
        },
        output: floatView('overview-direct-four-view', directFour, factorFourCount),
        outputValidity: wordView(
          'overview-direct-four-validity-view',
          directFourValidity,
          factorFourCount
        ),
        sum: floatView('overview-direct-four-sum-view', directFourSum, factorFourCount),
        validCount: wordView('overview-direct-four-count-view', directFourCount, factorFourCount)
      }).addToGraph(graph);
      new GPURasterOverview({
        id: 'overview-cascaded-factor-four',
        metadata: factorTwoMetadata,
        scale: 2,
        input: {
          id: 'overview-cascade-level-two',
          format: 'float32',
          storage: {
            kind: 'buffer',
            values: floatView('overview-level-two-input', levelTwoMean, factorTwoCount)
          }
        },
        inputSum: floatView('overview-level-two-input-sum', levelTwoSum, factorTwoCount),
        inputValidCount: wordView(
          'overview-level-two-input-count',
          levelTwoValidCount,
          factorTwoCount
        ),
        maximumInputValidCount: 4,
        output: floatView('overview-cascaded-four-view', cascadedFour, factorFourCount),
        outputValidity: wordView(
          'overview-cascaded-four-validity-view',
          cascadedFourValidity,
          factorFourCount
        ),
        sum: floatView('overview-cascaded-four-sum-view', cascadedFourSum, factorFourCount),
        validCount: wordView(
          'overview-cascaded-four-count-view',
          cascadedFourCount,
          factorFourCount
        )
      }).addToGraph(graph);
      addKernelPass(graph, {
        id: 'overview-direct-cascade-agreement',
        invocationCount: factorFourCount,
        bindings: [
          {
            name: 'direct',
            view: floatView('overview-agreement-direct', directFour, factorFourCount),
            type: 'f32',
            access: 'read'
          },
          {
            name: 'directCount',
            view: wordView('overview-agreement-direct-count', directFourCount, factorFourCount),
            type: 'u32',
            access: 'read'
          },
          {
            name: 'cascade',
            view: floatView('overview-agreement-cascade', cascadedFour, factorFourCount),
            type: 'f32',
            access: 'read'
          },
          {
            name: 'cascadeCount',
            view: wordView('overview-agreement-cascade-count', cascadedFourCount, factorFourCount),
            type: 'u32',
            access: 'read'
          },
          {
            name: 'agreement',
            view: floatView('overview-agreement-output', overviewAgreement, factorFourCount),
            type: 'f32',
            access: 'read_write'
          }
        ],
        body: 'agreement[agreementOffset+index]=select(0.0,direct[directOffset+index]-cascade[cascadeOffset+index],directCount[directCountOffset+index]==cascadeCount[cascadeCountOffset+index]&&directCount[directCountOffset+index]!=0u);'
      });
      addKernelPass(graph, {
        id: 'overview-direct-cascade-sum-agreement',
        invocationCount: factorFourCount,
        bindings: [
          {
            name: 'direct',
            view: floatView('overview-sum-agreement-direct', directFourSum, factorFourCount),
            type: 'f32',
            access: 'read'
          },
          {
            name: 'cascade',
            view: floatView('overview-sum-agreement-cascade', cascadedFourSum, factorFourCount),
            type: 'f32',
            access: 'read'
          },
          {
            name: 'agreement',
            view: floatView('overview-sum-agreement-output', overviewSumAgreement, factorFourCount),
            type: 'f32',
            access: 'read_write'
          }
        ],
        body: 'agreement[agreementOffset+index]=direct[directOffset+index]-cascade[cascadeOffset+index];'
      });
      const categoricalLabels = resources.createBuffer('nir-bin-labels', CELL_COUNT * 4);
      const categoricalLabelValidity = resources.createBuffer(
        'nir-bin-label-validity',
        CELL_COUNT * 4
      );
      const categorical = resources.createBuffer('nir-bin-overview', factorFourCount * 4);
      const categoricalValidity = resources.createBuffer(
        'nir-bin-overview-validity',
        factorFourCount * 4
      );
      const categoricalCount = resources.createBuffer(
        'nir-bin-overview-support',
        factorFourCount * 4
      );
      const categoricalDisplay = resources.createBuffer(
        'nir-bin-overview-display',
        factorFourCount * 4
      );
      addKernelPass(graph, {
        id: 'nir-reflectance-to-explicit-integer-bins',
        invocationCount: CELL_COUNT,
        bindings: [
          {name: 'values', view: mosaic, type: 'f32', access: 'read'},
          {name: 'validity', view: mosaicValidity, type: 'u32', access: 'read'},
          {
            name: 'labels',
            view: wordView('nir-bin-label-output', categoricalLabels),
            type: 'u32',
            access: 'read_write'
          },
          {
            name: 'labelValidity',
            view: wordView('nir-bin-label-validity-output', categoricalLabelValidity),
            type: 'u32',
            access: 'read_write'
          }
        ],
        body: 'let value=values[valuesOffset+index];let valid=validity[validityOffset+index]!=0u&&value==value&&abs(value)<=3.402823e38;labels[labelsOffset+index]=select(select(select(0u,1u,value>=0.1),2u,value>=0.2),3u,value>=0.3);labelValidity[labelValidityOffset+index]=select(0u,1u,valid);'
      });
      const categoricalInput: GPURasterBufferBand<'uint32'> = {
        id: 'explicit-nir-reflectance-bin',
        format: 'uint32',
        storage: {kind: 'buffer', values: wordView('nir-bin-label-input', categoricalLabels)},
        validity: wordView('nir-bin-label-validity-input', categoricalLabelValidity)
      };
      new GPURasterCategoricalOverview({
        id: 'nir-bin-mode-overview',
        metadata: rasterMetadata,
        scale: 4,
        input: categoricalInput,
        policy: options.categoricalPolicy,
        output: wordView('nir-bin-mode-output', categorical, factorFourCount),
        outputValidity: wordView('nir-bin-mode-validity', categoricalValidity, factorFourCount),
        validCount: wordView('nir-bin-mode-count', categoricalCount, factorFourCount)
      }).addToGraph(graph);
      addKernelPass(graph, {
        id: 'nir-bin-label-display',
        invocationCount: factorFourCount,
        bindings: [
          {
            name: 'labels',
            view: wordView('nir-bin-label-display-input', categorical, factorFourCount),
            type: 'u32',
            access: 'read'
          },
          {
            name: 'validity',
            view: wordView('nir-bin-label-display-validity', categoricalValidity, factorFourCount),
            type: 'u32',
            access: 'read'
          },
          {
            name: 'display',
            view: wordView('nir-bin-display-output', categoricalDisplay, factorFourCount),
            type: 'u32',
            access: 'read_write'
          }
        ],
        body: 'display[displayOffset+index]=select(0xffffffffu,labels[labelsOffset+index],validity[validityOffset+index]!=0u);'
      });
      const compiled = resources.track(await graph.compileAsync());
      const summaryReader = new SummaryReader(
        resources,
        'tile-summary',
        [
          {buffer: global.extent, size: 8},
          {buffer: global.count, size: 4},
          {buffer: global.sum, size: 4},
          {buffer: global.histogram, size: Number(options.binCount) * 4},
          {buffer: global.overflow, size: 4},
          {buffer: percentile, size: 4},
          {buffer: percentileValidity, size: 4},
          {buffer: difference, size: difference.byteLength},
          {buffer: wholecropDifference, size: wholecropDifference.byteLength},
          {buffer: resampling, size: resampling.byteLength},
          {buffer: resamplingValidity, size: resamplingValidity.byteLength},
          {buffer: resamplingReference, size: resamplingReference.byteLength},
          {buffer: resamplingReferenceValidity, size: resamplingReferenceValidity.byteLength},
          {buffer: resamplingDifference, size: resamplingDifference.byteLength},
          {buffer: directFour, size: directFour.byteLength},
          {buffer: directFourCount, size: directFourCount.byteLength},
          {buffer: cascadedFour, size: cascadedFour.byteLength},
          {buffer: cascadedFourCount, size: cascadedFourCount.byteLength},
          {buffer: overviewAgreement, size: overviewAgreement.byteLength},
          {buffer: overviewSumAgreement, size: overviewSumAgreement.byteLength},
          {buffer: categorical, size: categorical.byteLength},
          {buffer: categoricalValidity, size: categoricalValidity.byteLength},
          {buffer: categoricalCount, size: categoricalCount.byteLength},
          ...topology.summaryBuffers.map(buffer => ({buffer, size: buffer.byteLength}))
        ],
        bytes => {
          if (
            disposed ||
            active?.generation !== sceneGeneration ||
            active.key !== requestedKey ||
            keyFor(ctx.options) !== requestedKey
          )
            return;
          const words = new Uint32Array(bytes);
          const floats = new Float32Array(bytes);
          const countIndex = 2;
          const sumIndex = 3;
          const count = words[countIndex];
          const sum = floats[sumIndex];
          const sourceSum = source.values.reduce(
            (total, value, index) => total + (source.validity[index] ? value : 0),
            0
          );
          const differenceOffset = 16 + Number(options.binCount) * 4 + 12;
          const seamDifferences = new Float32Array(bytes, differenceOffset, CELL_COUNT);
          let maximumDifference = 0;
          let squaredDifference = 0;
          let seamCount = 0;
          for (const value of seamDifferences) {
            if (!Number.isFinite(value)) continue;
            maximumDifference = Math.max(maximumDifference, Math.abs(value));
            squaredDifference += value * value;
            seamCount++;
          }
          ctx.setReadout('validCells', count);
          ctx.setReadout('globalMean', count ? sum / count : Number.NaN);
          ctx.setReadout(
            'sourceSumCheck',
            `${sum.toFixed(4)} GPU · ${sourceSum.toFixed(4)} CPU reference · Δ ${(sum - sourceSum).toExponential(2)}`
          );
          ctx.setReadout(
            'seamError',
            `${seamCount} core cells · max ${maximumDifference.toFixed(5)} · RMS ${seamCount ? Math.sqrt(squaredDifference / seamCount).toFixed(5) : 'n/a'}`
          );
          const wholecropDifferenceOffset = differenceOffset + difference.byteLength;
          const wholecropDifferences = new Float32Array(
            bytes,
            wholecropDifferenceOffset,
            CELL_COUNT
          );
          let wholecropMaximumDifference = 0;
          let wholecropSquaredDifference = 0;
          let wholecropCount = 0;
          for (const value of wholecropDifferences) {
            if (!Number.isFinite(value)) continue;
            wholecropMaximumDifference = Math.max(wholecropMaximumDifference, Math.abs(value));
            wholecropSquaredDifference += value * value;
            wholecropCount++;
          }
          ctx.setReadout(
            'wholecropSeamError',
            `${wholecropCount} shared-valid cells · tiled vs whole-crop GPU · max ${wholecropMaximumDifference.toFixed(6)} · RMS ${wholecropCount ? Math.sqrt(wholecropSquaredDifference / wholecropCount).toFixed(6) : 'n/a'}`
          );
          const resamplingOffset = wholecropDifferenceOffset + difference.byteLength;
          const resampledValues = new Float32Array(bytes, resamplingOffset, CELL_COUNT);
          const resampledValidity = new Uint32Array(
            bytes,
            resamplingOffset + difference.byteLength,
            CELL_COUNT
          );
          const referenceOffset = resamplingOffset + difference.byteLength * 2;
          const referenceValidityOffset = referenceOffset + difference.byteLength;
          const resamplingDifferenceOffset = referenceValidityOffset + difference.byteLength;
          const resamplingDifferences = new Float32Array(
            bytes,
            resamplingDifferenceOffset,
            CELL_COUNT
          );
          let resampleCount = 0;
          let referenceCount = 0;
          let supportMismatchCount = 0;
          let resampleAgreementCount = 0;
          let resampleMaximum = 0;
          let resampleSquared = 0;
          const referenceValidity = new Uint32Array(bytes, referenceValidityOffset, CELL_COUNT);
          for (let index = 0; index < CELL_COUNT; index++) {
            referenceCount += referenceValidity[index] ? 1 : 0;
            supportMismatchCount += Number(
              Boolean(resampledValidity[index]) !== Boolean(referenceValidity[index])
            );
            if (!resampledValidity[index]) continue;
            resampleCount++;
            if (!referenceValidity[index]) continue;
            resampleAgreementCount++;
            const error = resamplingDifferences[index];
            resampleMaximum = Math.max(resampleMaximum, Math.abs(error));
            resampleSquared += error * error;
          }
          void resampledValues;
          ctx.setReadout(
            'resampleSupport',
            `${resampleCount} tiled destinations · ${referenceCount} reference destinations · ${supportMismatchCount} support mismatches · ${resampleAgreementCount} shared support · max |Δ| ${resampleMaximum.toExponential(2)} · RMS ${resampleAgreementCount ? Math.sqrt(resampleSquared / resampleAgreementCount).toExponential(2) : 'n/a'}`
          );
          const diagnosticsOffset = resamplingDifferenceOffset + difference.byteLength;
          const directMeanValues = new Float32Array(bytes, diagnosticsOffset, factorFourCount);
          const directCounts = new Uint32Array(
            bytes,
            diagnosticsOffset + factorFourCount * 4,
            factorFourCount
          );
          const cascadeMeanOffset = diagnosticsOffset + factorFourCount * 8;
          const cascadeMeanValues = new Float32Array(bytes, cascadeMeanOffset, factorFourCount);
          const cascadeCounts = new Uint32Array(
            bytes,
            cascadeMeanOffset + factorFourCount * 4,
            factorFourCount
          );
          const agreementValues = new Float32Array(
            bytes,
            cascadeMeanOffset + factorFourCount * 8,
            factorFourCount
          );
          const sumAgreementValues = new Float32Array(
            bytes,
            cascadeMeanOffset + factorFourCount * 12,
            factorFourCount
          );
          const labelOffset = cascadeMeanOffset + factorFourCount * 16;
          const labels = new Uint32Array(bytes, labelOffset, factorFourCount);
          const labelValidity = new Uint32Array(
            bytes,
            labelOffset + factorFourCount * 4,
            factorFourCount
          );
          const labelSupport = new Uint32Array(
            bytes,
            labelOffset + factorFourCount * 8,
            factorFourCount
          );
          let overviewMaximumDifference = 0;
          let overviewMaximumSumDifference = 0;
          let overviewCountAgreement = true;
          for (let index = 0; index < factorFourCount; index++) {
            overviewCountAgreement &&= directCounts[index] === cascadeCounts[index];
            if (!directCounts[index] || !cascadeCounts[index]) continue;
            overviewMaximumDifference = Math.max(
              overviewMaximumDifference,
              Math.abs(agreementValues[index])
            );
            overviewMaximumSumDifference = Math.max(
              overviewMaximumSumDifference,
              Math.abs(sumAgreementValues[index])
            );
          }
          const labelCounts = [0, 0, 0, 0];
          let labelCells = 0;
          let labelSupportTotal = 0;
          for (let index = 0; index < factorFourCount; index++) {
            if (!labelValidity[index]) continue;
            labelCells++;
            labelSupportTotal += labelSupport[index];
            labelCounts[Math.min(3, labels[index])]++;
          }
          void directMeanValues;
          void cascadeMeanValues;
          ctx.setReadout(
            'overviewAgreement',
            `direct 4×4 vs 2×2→2×2 · max |Δsum| ${overviewMaximumSumDifference.toExponential(2)} · max |Δmean| ${overviewMaximumDifference.toExponential(2)} · counts ${overviewCountAgreement ? 'agree exactly' : 'DIFFER'}`
          );
          ctx.setReadout(
            'categoricalOverview',
            `${labelCells} valid labels · support ${labelSupportTotal} · bins 0:${labelCounts[0]} 1:${labelCounts[1]} 2:${labelCounts[2]} 3:${labelCounts[3]} · ${options.categoricalPolicy === 'mode' ? 'mode ties choose lower label' : 'upper-left center on ties; missing centers stay missing'}`
          );
          const topologyWordOffset = (bytes.byteLength - topology.summaryByteLength) / 4;
          const topologyWords = new Uint32Array(bytes, topologyWordOffset * 4);
          let topologyCursor = 0;
          const localLow = Array.from({length: 4}, () => {
            const values = topologyWords.slice(topologyCursor, topologyCursor + 4);
            topologyCursor += 4;
            return values;
          });
          const localHigh = Array.from({length: 4}, () => {
            const values = topologyWords.slice(topologyCursor, topologyCursor + 4);
            topologyCursor += 4;
            return values;
          });
          const lowGlobal = topologyWords.slice(topologyCursor, topologyCursor + 4);
          topologyCursor += 4;
          const lowPopulation = topologyWords.slice(topologyCursor, topologyCursor + 128);
          topologyCursor += 128;
          const highGlobal = topologyWords.slice(topologyCursor, topologyCursor + 4);
          topologyCursor += 4;
          const highPopulation = topologyWords.slice(topologyCursor, topologyCursor + 128);
          topologyCursor += 128;
          const adjacency = topologyWords.slice(
            topologyCursor,
            topologyCursor + topology.cellCount * topology.cellCount
          );
          topologyCursor += topology.cellCount * topology.cellCount;
          const mapping = topologyWords.slice(topologyCursor, topologyCursor + topology.cellCount);
          topologyCursor += topology.cellCount;
          const regionStatus = topologyWords[topologyCursor] ?? 1;
          const localStatuses = [...localLow, ...localHigh];
          const localReady = localStatuses.every(
            status =>
              status[2] === 1 && status[3] === 0 && status[0] <= TILED_TOPOLOGY_REGION_CAPACITY
          );
          const globalReady =
            lowGlobal[2] === 1 &&
            lowGlobal[3] === 0 &&
            highGlobal[2] === 1 &&
            highGlobal[3] === 0 &&
            regionStatus === 0;
          const lowLocalCount = localLow.reduce((total, status) => total + status[0]!, 0);
          const highLocalCount = localHigh.reduce((total, status) => total + status[0]!, 0);
          const lowPopulationCount = lowPopulation.reduce((total, value) => total + value, 0);
          const highPopulationCount = highPopulation.reduce((total, value) => total + value, 0);
          if (localReady && globalReady) {
            acceptedTopologyGeneration = sceneGeneration;
            const lowSeamMerges = lowLocalCount - lowGlobal[0]!;
            const highSeamMerges = highLocalCount - highGlobal[0]!;
            ctx.setReadout(
              'topologyComponents',
              `8-connected · low: ${lowLocalCount} core-local → ${lowGlobal[0]} global (${lowSeamMerges} seam joins) · high: ${highLocalCount} → ${highGlobal[0]} (${highSeamMerges}) · 128 slots/class · all local/global passes converged, no overflow`
            );
            ctx.setReadout(
              'topologyPopulation',
              `${lowPopulationCount + highPopulationCount} valid cells · low < ${TILED_TOPOLOGY_THRESHOLD}: ${lowPopulationCount} · high ≥ ${TILED_TOPOLOGY_THRESHOLD}: ${highPopulationCount} · 64 × 64 focus in crop rows 82–145, cols 92–155`
            );
            let crossClassPairs = 0;
            let crossClassContacts = 0;
            let activeRegions = 0;
            let mappedRegions = 0;
            const mappedRoots = new Set<number>();
            for (let region = 0; region < topology.cellCount; region++) {
              const count = region < 128 ? lowPopulation[region]! : highPopulation[region - 128]!;
              if (!count) continue;
              activeRegions++;
              mappedRoots.add(mapping[region]!);
            }
            mappedRegions = mappedRoots.size;
            for (let lowRegion = 0; lowRegion < 128; lowRegion++) {
              for (let highRegion = 128; highRegion < topology.cellCount; highRegion++) {
                const contacts = adjacency[lowRegion * topology.cellCount + highRegion]!;
                if (contacts === 0) continue;
                crossClassPairs++;
                crossClassContacts += contacts;
              }
            }
            ctx.setReadout(
              'regionAdjacency',
              `${crossClassPairs} low/high region pairs · ${crossClassContacts} shared 4-edge contacts · ${activeRegions} active IDs → ${mappedRegions} after fixed 0.0 mean threshold · complete after 32 rounds`
            );
          } else {
            acceptedTopologyGeneration = -1;
            ctx.setReadout(
              'topologyComponents',
              `withheld · local ready ${localReady} · global/graph ready ${globalReady} · low overflow ${lowGlobal[3]} · high overflow ${highGlobal[3]} · region graph status ${regionStatus}`
            );
            ctx.setReadout('topologyPopulation', 'withheld until every owned core is valid');
            ctx.setReadout(
              'regionAdjacency',
              'withheld until component and region-graph stages converge'
            );
          }
          ctx.requestLayers();
          const histogramOffset = 16;
          const histogramBinCount = Number(options.binCount);
          const histogramCounts = words.slice(
            histogramOffset / 4,
            histogramOffset / 4 + histogramBinCount
          );
          const extentMinimum = floats[0];
          const extentMaximum = floats[1];
          const histogramBinWidth = (extentMaximum - extentMinimum) / histogramBinCount;
          const overflow = words[histogramOffset / 4 + histogramBinCount];
          const percentileValidityFlag = words[histogramOffset / 4 + histogramBinCount + 2];
          ctx.setChart('histogram', {
            kind: 'histogram',
            values: Array.from(histogramCounts),
            xDomain: [extentMinimum, extentMaximum]
          });
          ctx.setReadout(
            'percentile',
            percentileValidityFlag
              ? `${options.percentile} ≈ ${floats[5 + histogramBinCount].toFixed(3)}`
              : `${options.percentile} unavailable · empty population or histogram overflow`
          );
          ctx.setReadout(
            'histogramDetails',
            `range ${extentMinimum.toFixed(4)}–${extentMaximum.toFixed(4)} · ${histogramBinCount} bins · width ${histogramBinWidth.toFixed(6)} · counted ${histogramCounts.reduce((total, value) => total + value, 0)} · overflow ${overflow}`
          );
          ctx.setReadout(
            'tileCache',
            `hit ${cache.stats.tileHits} · miss ${cache.stats.tileMisses} · pinned ${cache.stats.pinnedTiles}`
          );
        }
      );
      return {
        generation: sceneGeneration,
        key: keyFor(options),
        resources,
        compiled,
        output,
        validity,
        isolated,
        difference,
        wholecropDifference,
        overview,
        overviewCount,
        overviewSupport,
        overviewDirect4: directFour,
        overviewDirect4Count: directFourCount,
        overviewCascaded4: cascadedFour,
        overviewCascaded4Count: cascadedFourCount,
        overviewAgreement,
        overviewSumAgreement,
        categorical,
        categoricalValidity,
        categoricalCount,
        categoricalDisplay,
        resampling,
        resamplingValidity,
        resamplingReference,
        resamplingReferenceValidity,
        resamplingDifference,
        global,
        percentile,
        reader: summaryReader,
        windows,
        haloLeases,
        topology
      };
    } catch (error) {
      for (const lease of haloLeases) lease.release();
      resources.destroy();
      throw error;
    }
  };
  await graphSwap.request({
    build: async () => compile(ctx.options),
    commit: variant => {
      releaseVariant(active);
      active = variant;
      acceptedTopologyGeneration = -1;
      requestedKey = keyFor(ctx.options);
      dirty = true;
      clearTiledSummary(ctx);
      ctx.requestLayers();
    },
    discard: variant => {
      releaseVariant(variant);
    }
  });
  const sourceNir = {
    july: resourcesForLayer(rootResources, 'july', sourceByDate.july),
    september: resourcesForLayer(rootResources, 'september', sourceByDate.september)
  };
  const tileLineSegments = new Float32Array(32);
  const tileEdges = [
    [frame.cellToMeters(-0.5, -0.5), frame.cellToMeters(CROP_SIZE - 0.5, -0.5)],
    [
      frame.cellToMeters(CROP_SIZE - 0.5, -0.5),
      frame.cellToMeters(CROP_SIZE - 0.5, CROP_SIZE - 0.5)
    ],
    [
      frame.cellToMeters(CROP_SIZE - 0.5, CROP_SIZE - 0.5),
      frame.cellToMeters(-0.5, CROP_SIZE - 0.5)
    ],
    [frame.cellToMeters(-0.5, CROP_SIZE - 0.5), frame.cellToMeters(-0.5, -0.5)],
    [
      frame.cellToMeters(TILE_SIZE - 0.5, -0.5),
      frame.cellToMeters(TILE_SIZE - 0.5, CROP_SIZE - 0.5)
    ],
    [
      frame.cellToMeters(-0.5, TILE_SIZE - 0.5),
      frame.cellToMeters(CROP_SIZE - 0.5, TILE_SIZE - 0.5)
    ]
  ];
  for (let edgeIndex = 0; edgeIndex < tileEdges.length; edgeIndex++) {
    const [start, end] = tileEdges[edgeIndex];
    tileLineSegments.set([start[0], start[1], end[0], end[1]], edgeIndex * 4);
  }
  const tileLines = rootResources.createBuffer('ownership-lines', tileLineSegments, Buffer.VERTEX);
  const selectedCoreCorners = [
    frame.cellToMeters(-0.5, -0.5),
    frame.cellToMeters(TILE_SIZE - 0.5, -0.5),
    frame.cellToMeters(TILE_SIZE - 0.5, TILE_SIZE - 0.5),
    frame.cellToMeters(-0.5, TILE_SIZE - 0.5)
  ];
  const selectedCoreSegments = new Float32Array(16);
  for (let cornerIndex = 0; cornerIndex < 4; cornerIndex++) {
    const start = selectedCoreCorners[cornerIndex];
    const end = selectedCoreCorners[(cornerIndex + 1) % 4];
    selectedCoreSegments.set([start[0], start[1], end[0], end[1]], cornerIndex * 4);
  }
  const selectedCoreOutline = rootResources.createBuffer(
    'selected-core-outline',
    selectedCoreSegments,
    Buffer.VERTEX
  );
  const [focusLeft, focusTop, focusRight, focusBottom] = getTiledObservationTopologyFocusBounds();
  const focusEdges = [
    [
      frame.cellToMeters(focusLeft - 0.5, focusTop - 0.5),
      frame.cellToMeters(focusRight - 0.5, focusTop - 0.5)
    ],
    [
      frame.cellToMeters(focusRight - 0.5, focusTop - 0.5),
      frame.cellToMeters(focusRight - 0.5, focusBottom - 0.5)
    ],
    [
      frame.cellToMeters(focusRight - 0.5, focusBottom - 0.5),
      frame.cellToMeters(focusLeft - 0.5, focusBottom - 0.5)
    ],
    [
      frame.cellToMeters(focusLeft - 0.5, focusBottom - 0.5),
      frame.cellToMeters(focusLeft - 0.5, focusTop - 0.5)
    ],
    [frame.cellToMeters(63.5, focusTop - 0.5), frame.cellToMeters(63.5, focusBottom - 0.5)],
    [frame.cellToMeters(focusLeft - 0.5, 63.5), frame.cellToMeters(focusRight - 0.5, 63.5)]
  ];
  const focusSegments = new Float32Array(focusEdges.length * 4);
  for (let edgeIndex = 0; edgeIndex < focusEdges.length; edgeIndex++) {
    const [start, end] = focusEdges[edgeIndex]!;
    focusSegments.set([start[0], start[1], end[0], end[1]], edgeIndex * 4);
  }
  const topologyFocusOutline = rootResources.createBuffer(
    'topology-focus-outline',
    focusSegments,
    Buffer.VERTEX
  );
  const scene: SceneInstance<TiledObservationsOptions> = {
    getCompiledGraphs: () => (active ? [active.compiled] : []),
    encode(commandEncoder, frameState) {
      if (!active || active.key !== keyFor(ctx.options)) return;
      if (frameState.frameIndex === 0 || dirty) {
        active.compiled.encode(commandEncoder, {parameters: undefined});
        active.reader.request(commandEncoder);
        dirty = false;
      } else active.reader.flush(commandEncoder);
    },
    getLayers() {
      if (!active || active.key !== keyFor(ctx.options)) return [];
      if (ctx.options.mapLayer === 'topology' && acceptedTopologyGeneration !== active.generation)
        return [];
      const showingOverview =
        ctx.options.mapLayer === 'overview' ||
        ctx.options.mapLayer === 'support' ||
        ctx.options.mapLayer === 'categorical';
      const overviewSize =
        ctx.options.mapLayer === 'categorical'
          ? CROP_SIZE / 4
          : CROP_SIZE / Number(ctx.options.overviewScale);
      const mapOverviewScale =
        ctx.options.mapLayer === 'categorical' ? 4 : Number(ctx.options.overviewScale);
      const showingResampled =
        ctx.options.mapLayer === 'resampled' || ctx.options.mapLayer === 'resampleDifference';
      const mapFrame = showingOverview
        ? createUtmFrame(dataset.defaultOrigin, {
            boundsProjected: [
              PROJECTED_BOUNDS[0] + CROP_COLUMN * CELL_SIZE_METERS,
              PROJECTED_BOUNDS[1] + (SOURCE_SIZE - CROP_ROW - CROP_SIZE) * CELL_SIZE_METERS,
              PROJECTED_BOUNDS[0] + (CROP_COLUMN + CROP_SIZE) * CELL_SIZE_METERS,
              PROJECTED_BOUNDS[3] - CROP_ROW * CELL_SIZE_METERS
            ],
            cellSizeM: CELL_SIZE_METERS * mapOverviewScale,
            width: overviewSize,
            height: overviewSize
          })
        : showingResampled
          ? createUtmFrame(dataset.defaultOrigin, {
              boundsProjected: [
                PROJECTED_BOUNDS[0] + CROP_COLUMN * CELL_SIZE_METERS + CELL_SIZE_METERS / 2,
                PROJECTED_BOUNDS[1] +
                  (SOURCE_SIZE - CROP_ROW - CROP_SIZE) * CELL_SIZE_METERS -
                  CELL_SIZE_METERS / 2,
                PROJECTED_BOUNDS[0] +
                  (CROP_COLUMN + CROP_SIZE) * CELL_SIZE_METERS +
                  CELL_SIZE_METERS / 2,
                PROJECTED_BOUNDS[3] - CROP_ROW * CELL_SIZE_METERS - CELL_SIZE_METERS / 2
              ],
              cellSizeM: CELL_SIZE_METERS,
              width: CROP_SIZE,
              height: CROP_SIZE
            })
          : frame;
      const mapBuffer =
        ctx.options.mapLayer === 'source'
          ? sourceNir[ctx.options.date]
          : ctx.options.mapLayer === 'isolated'
            ? active.isolated
            : ctx.options.mapLayer === 'difference'
              ? active.difference
              : ctx.options.mapLayer === 'wholecropDifference'
                ? active.wholecropDifference
                : ctx.options.mapLayer === 'overview'
                  ? active.overview
                  : ctx.options.mapLayer === 'support'
                    ? active.overviewSupport
                    : ctx.options.mapLayer === 'categorical'
                      ? active.categoricalDisplay
                      : ctx.options.mapLayer === 'resampled'
                        ? active.resampling
                        : ctx.options.mapLayer === 'resampleDifference'
                          ? active.resamplingDifference
                          : ctx.options.mapLayer === 'topology'
                            ? active.topology.display
                            : active.output;
      return [
        new UtmRasterLayer({
          id: 'tiled-observations-map',
          frame: mapFrame,
          values: mapBuffer,
          valueFormat:
            ctx.options.mapLayer === 'categorical' || ctx.options.mapLayer === 'topology'
              ? 'uint32'
              : 'float32',
          valueRange:
            ctx.options.mapLayer === 'difference' ||
            ctx.options.mapLayer === 'wholecropDifference' ||
            ctx.options.mapLayer === 'resampleDifference'
              ? [-0.2, 0.2]
              : ctx.options.mapLayer === 'topology'
                ? [0, 255]
                : ctx.options.mapLayer === 'support'
                  ? [0, Number(ctx.options.overviewScale) ** 2]
                  : [0, 0.5],
          colormap:
            ctx.options.mapLayer === 'difference' ||
            ctx.options.mapLayer === 'wholecropDifference' ||
            ctx.options.mapLayer === 'resampleDifference'
              ? 'diverging'
              : ctx.options.mapLayer === 'categorical' || ctx.options.mapLayer === 'topology'
                ? 'category'
                : 'batlow',
          ...(ctx.options.mapLayer === 'categorical'
            ? {
                palette: [
                  [102, 194, 165, 255],
                  [252, 141, 98, 255],
                  [141, 160, 203, 255],
                  [231, 138, 195, 255]
                ] as const
              }
            : {}),
          ...(ctx.options.mapLayer === 'topology' ? {palette: TILED_TOPOLOGY_PALETTE} : {}),
          opacity: 0.9
        }),
        new SpatialAnalysisSegmentLayer({
          id: 'tiled-observations-core-ownership',
          coordinateOrigin: [frame.origin[0], frame.origin[1], 0],
          segments: tileLines,
          instanceCount: tileEdges.length,
          widthPixels: 1.5,
          color: [255, 221, 130, 240],
          outlineColor: [35, 40, 45, 240],
          outlineWidthPixels: 1
        }),
        new SpatialAnalysisSegmentLayer({
          id: 'tiled-observations-selected-core',
          coordinateOrigin: [frame.origin[0], frame.origin[1], 0],
          segments: selectedCoreOutline,
          instanceCount: 4,
          widthPixels: 2.5,
          color: [255, 115, 55, 255],
          outlineColor: [25, 25, 25, 255],
          outlineWidthPixels: 1
        }),
        ...(ctx.options.mapLayer === 'topology'
          ? [
              new SpatialAnalysisSegmentLayer({
                id: 'tiled-observations-topology-focus',
                coordinateOrigin: [frame.origin[0], frame.origin[1], 0],
                segments: topologyFocusOutline,
                instanceCount: focusEdges.length,
                widthPixels: 2,
                color: [255, 255, 255, 255],
                outlineColor: [30, 35, 42, 255],
                outlineWidthPixels: 1
              })
            ]
          : [])
      ];
    },
    setOption(id, _value, state) {
      if (id === 'step' || id === 'mapLayer') {
        ctx.requestLayers();
        return;
      }
      const nextKey = keyFor(state);
      if (nextKey === requestedKey) return;
      requestedKey = nextKey;
      clearTiledSummary(ctx);
      ctx.requestLayers();
      if (id === 'date') {
        const centerValue = sourceByDate[state.date].values[64 * CROP_SIZE + 64];
        ctx.setReadout(
          'probe',
          `crop cell 64,64 · ${Number.isFinite(centerValue) ? centerValue.toFixed(3) : 'missing'} reflectance (click to inspect)`
        );
      }
      void graphSwap
        .request({
          build: async () => compile(state),
          commit: variant => {
            releaseVariant(active);
            active = variant;
            acceptedTopologyGeneration = -1;
            dirty = true;
            clearTiledSummary(ctx);
            ctx.requestLayers();
          }
        })
        .catch(error => {
          if (disposed || requestedKey !== nextKey) return;
          ctx.setStatus(`Tile graph rebuild failed: ${String(error)}`);
        });
    },
    destroy() {
      disposed = true;
      releaseVariant(active);
      rootResources.destroy();
    },
    onClick(info) {
      if (!info.coordinate) return false;
      const coordinate = frame.lngLatToCell(info.coordinate[0], info.coordinate[1]);
      const column = Math.floor(coordinate[0]);
      const row = Math.floor(coordinate[1]);
      if (column < 0 || row < 0 || column >= CROP_SIZE || row >= CROP_SIZE) return false;
      const sample = sourceByDate[ctx.options.date].values[row * CROP_SIZE + column];
      ctx.setReadout(
        'probe',
        `cell ${column},${row} · ${Number.isFinite(sample) ? sample.toFixed(3) : 'missing'}`
      );
      return true;
    }
  };
  const centerValue = sourceByDate[ctx.options.date].values[64 * CROP_SIZE + 64];
  ctx.setReadout(
    'probe',
    `crop cell 64,64 · ${Number.isFinite(centerValue) ? centerValue.toFixed(3) : 'missing'} reflectance (click to inspect)`
  );
  ctx.setReadout(
    'cropSupport',
    '128 × 128 cells · four owned 64 × 64 cores · orange outline: NW core'
  );
  return scene;
}

function resourcesForLayer(
  resources: SpatialAnalysisResources,
  date: string,
  source: SourceArrays
): Buffer {
  return resources.createBuffer(`source-map-${date}`, source.values);
}

function releaseVariant(variant: TileGraph | null): void {
  if (!variant) return;
  variant.reader.release();
  for (const lease of variant.haloLeases) lease.release();
  variant.resources.destroy();
}

function clearTiledSummary(ctx: SceneContext<TiledObservationsOptions>): void {
  ctx.setReadout('validCells', 'pending');
  ctx.setReadout('globalMean', 'pending');
  ctx.setReadout('sourceSumCheck', 'pending');
  ctx.setReadout('seamError', 'pending');
  ctx.setReadout('wholecropSeamError', 'pending');
  ctx.setReadout('percentile', 'pending');
  ctx.setReadout('tileCache', 'pending');
  ctx.setReadout('resampleSupport', 'pending');
  ctx.setReadout('histogramDetails', 'pending');
  ctx.setReadout('overviewAgreement', 'pending');
  ctx.setReadout('categoricalOverview', 'pending');
  ctx.setReadout('topologyComponents', 'pending · waiting for local and seam reconciliation');
  ctx.setReadout('topologyPopulation', 'pending · waiting for region measurements');
  ctx.setReadout('regionAdjacency', 'pending · waiting for the region graph');
  ctx.setChart('histogram', null);
}

function keyFor(options: TiledObservationsOptions): string {
  return [
    options.date,
    options.radius,
    options.noDataPolicy,
    options.edgePolicy,
    options.binCount,
    options.percentile,
    options.overviewScale,
    options.categoricalPolicy,
    options.resampling,
    options.omitEast
  ].join('|');
}

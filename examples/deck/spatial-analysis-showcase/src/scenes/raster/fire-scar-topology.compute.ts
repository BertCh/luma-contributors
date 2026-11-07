// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer} from '@luma.gl/core';
import {GPUCommandGraph, GPUReduction, type CompiledGPUCommandGraph, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {getGPURasterArithmeticParameterValues, GPURasterArithmetic, GPURasterMedialAxis, GPURasterRankFilter, GPURasterRegionGraph, GPURasterSkeleton, GPURasterWatershedSegmentation, makeGPURasterMetadata} from '@luma-contributors/analysis/raster-analysis';
import {GPURasterClosing, GPURasterConnectedComponents, GPURasterDenseComponents, GPURasterHistogram, GPURasterOtsuThreshold, GPURasterOpening, GPURasterRegionMeasurements, GPURasterThreshold, type GPURasterBufferBand} from '@luma.gl/graph/gpu-raster';
import {SpatialAnalysisResources, formatCount} from '../../engine/resources';
import {addKernelPass} from '../../engine/mode-kernels';
import {fetchBytes} from '../../data/loaders';
import {GraphSwap} from '../../engine/graph-swap';
import {SummaryReader} from '../../engine/summary-reader';
import type {SceneContext, SceneInstance} from '../scene';
import {createGraphViewer, createUtmFrame, UtmRasterLayer} from './b16-common';

const DATASET_ID = 'dixie-spectral';
const COMPONENT_CAPACITY = 4096;
const DNBR_DOMAIN: readonly [number, number] = [-2, 2];
const HISTOGRAM_BINS = 256;
const REGION_COLORS = [
  [61, 106, 163, 255],
  [221, 137, 47, 255],
  [113, 158, 96, 255],
  [181, 101, 139, 255],
  [84, 159, 164, 255],
  [195, 177, 82, 255],
  [130, 113, 182, 255],
  [209, 107, 84, 255]
] as const;

export type FireScarTopologyOptions = {
  mapLayer:
    | 'support'
    | 'nbrBefore'
    | 'nbrAfter'
    | 'dnbr'
    | 'threshold'
    | 'cleaned'
    | 'regions'
    | 'classContacts'
    | 'focusMask'
    | 'skeleton'
    | 'medialAxis'
    | 'watershed'
    | 'watershedMerged'
    | 'watershedMarkers';
  thresholdMethod: 'fixed' | 'otsu' | 'source-probe';
  thresholdOperation: 'above' | 'below' | 'range';
  thresholdInclusive: boolean;
  thresholdValue: number;
  rangeEndpoint: number;
  cleanup: 'none' | 'median' | 'opening' | 'closing';
  radius: number;
  connectivity: '4' | '8';
  regionMergeMode: 'fixed' | 'adaptive';
  regionMergeThreshold: number;
  regionMergeRounds: number;
};

const FOCUS_WIDTH = 64;
const FOCUS_HEIGHT = 64;
const FOCUS_COLUMN = 176;
const FOCUS_ROW = 56;
const FOCUS_CELL_COUNT = FOCUS_WIDTH * FOCUS_HEIGHT;
const SOURCE_PROBE_ROW = 26488;

type VariantState = Pick<
  FireScarTopologyOptions,
  | 'thresholdMethod'
  | 'thresholdOperation'
  | 'thresholdInclusive'
  | 'cleanup'
  | 'radius'
  | 'connectivity'
  | 'regionMergeMode'
  | 'regionMergeThreshold'
  | 'regionMergeRounds'
>;

type TopologyBuffers = {
  mask: Buffer;
  thresholdCellCount: Buffer;
  cleanedMask: Buffer;
  sparseLabels: Buffer;
  sparseValidity: Buffer;
  converged: Buffer;
  iterations: Buffer;
  denseLabels: Buffer;
  denseValidity: Buffer;
  componentCount: Buffer;
  requiredComponentCount: Buffer;
  componentOverflow: Buffer;
  measurementPixelCounts: Buffer;
  measurementAreas: Buffer;
  classLabels: Buffer;
  classAdjacency: Buffer;
  classWeights: Buffer;
  classStatus: Buffer;
};

type TopologyVariant = {
  key: string;
  compiled: CompiledGPUCommandGraph<void>;
  buffers: TopologyBuffers;
  ownedBuffers: Buffer[];
  reader: SummaryReader;
  requestedSignature: string;
  focusRequestedSignature: string;
  focusCompiled: CompiledGPUCommandGraph<void>;
  focusBuffers: {
    mask: Buffer;
    skeleton: Buffer;
    skeletonConverged: Buffer;
    skeletonCount: Buffer;
    medialAxis: Buffer;
    medialRadius: Buffer;
    maximumRadius: Buffer;
    watershed: Buffer;
    watershedConverged: Buffer;
    regionLabels: Buffer;
    regionAdjacency: Buffer;
    regionWeights: Buffer;
    regionMapping: Buffer;
    regionStatus: Buffer;
    basinCounts: Buffer;
    basinChecksums: Buffer;
    watershedMerged: Buffer;
    markers: Buffer;
  };
  focusReader: SummaryReader;
  focusOwnedBuffers: Buffer[];
};

function makeVariantKey(state: VariantState): string {
  return `${state.thresholdMethod}-${state.thresholdOperation}-${state.thresholdInclusive}-${state.cleanup}-${state.radius}-${state.connectivity}-${state.regionMergeMode}-${state.regionMergeThreshold}-${state.regionMergeRounds}`;
}

function makeReadbackSignature(options: FireScarTopologyOptions): string {
  return `${makeVariantKey(options)}-${options.thresholdValue}-${options.rangeEndpoint}`;
}

function describeThreshold(options: FireScarTopologyOptions, otsuValue: number): string {
  if (options.thresholdOperation === 'range') {
    const lower = Math.min(options.thresholdValue, options.rangeEndpoint);
    const upper = Math.max(options.thresholdValue, options.rangeEndpoint);
    return `manual ${options.thresholdInclusive ? 'inclusive' : 'exclusive'} [${lower.toFixed(2)}, ${upper.toFixed(2)}]`;
  }
  const threshold =
    options.thresholdMethod === 'otsu'
      ? Number.isFinite(otsuValue)
        ? `Otsu ${otsuValue.toFixed(3)}`
        : 'Otsu pending'
      : options.thresholdMethod === 'source-probe'
        ? `GPU source-cell threshold · row ${SOURCE_PROBE_ROW.toLocaleString()}`
        : options.thresholdValue.toFixed(2);
  return `${options.thresholdOperation} ${options.thresholdInclusive ? 'inclusive' : 'exclusive'} · ${threshold}`;
}

function setTopologyReadoutsUpdating(ctx: SceneContext<FireScarTopologyOptions>): void {
  ctx.setReadout('thresholdCells', 'updating…');
  ctx.setReadout('componentCount', 'updating…');
  ctx.setReadout('selectedCells', 'updating…');
  ctx.setReadout('selectedArea', 'updating…');
  ctx.setReadout('largestPatch', 'updating…');
  ctx.setReadout('topologyStatus', 'recomputing the selected topology');
  ctx.setReadout('classBoundaryLength', 'updating…');
  ctx.setReadout('classMeanDifference', 'updating…');
  ctx.setReadout('classContactStatus', 'updating class graph…');
  ctx.setReadout('skeletonPixels', 'updating…');
  ctx.setReadout('medialRadius', 'updating…');
  ctx.setReadout('mergedBasinCount', 'updating…');
  ctx.setReadout('watershedStatus', 'updating…');
  ctx.setReadout('focusRegionMapping', 'updating watershed merge…');
}

function isFocusMapLayer(mapLayer: FireScarTopologyOptions['mapLayer']): boolean {
  return (
    mapLayer === 'focusMask' ||
    mapLayer === 'skeleton' ||
    mapLayer === 'medialAxis' ||
    mapLayer === 'watershed' ||
    mapLayer === 'watershedMerged' ||
    mapLayer === 'watershedMarkers'
  );
}

/** Load the bundled byte validity mask without decoding or changing its source flags. */
async function loadValidity(
  dataset: ReturnType<SceneContext<FireScarTopologyOptions>['datasets']['get']>,
  file: string,
  signal: AbortSignal
): Promise<Uint8Array> {
  return new Uint8Array(await fetchBytes(dataset.fileUrl(file), signal));
}

function requireFloatRaster(values: ArrayBufferView, name: string): Float32Array {
  // The manifest declares every reflectance raster as little-endian float32-bin.
  if (!(values instanceof Float32Array)) throw new Error(`${name} must decode as float32`);
  return values;
}

function createFocusMarkers(validity: Uint32Array): {markers: Uint32Array; count: number} {
  const markers = new Uint32Array(FOCUS_CELL_COUNT);
  let markerCount = 0;
  for (let row = 8; row < FOCUS_HEIGHT; row += 16) {
    for (let column = 8; column < FOCUS_WIDTH; column += 16) {
      for (let searchRadius = 0; searchRadius <= 4; searchRadius++) {
        let foundIndex = -1;
        for (
          let offsetRow = -searchRadius;
          offsetRow <= searchRadius && foundIndex < 0;
          offsetRow++
        ) {
          for (let offsetColumn = -searchRadius; offsetColumn <= searchRadius; offsetColumn++) {
            const candidateColumn = column + offsetColumn;
            const candidateRow = row + offsetRow;
            if (
              candidateColumn < 0 ||
              candidateRow < 0 ||
              candidateColumn >= FOCUS_WIDTH ||
              candidateRow >= FOCUS_HEIGHT
            ) {
              continue;
            }
            const candidateIndex = candidateRow * FOCUS_WIDTH + candidateColumn;
            if (validity[candidateIndex] !== 0 && markers[candidateIndex] === 0) {
              foundIndex = candidateIndex;
              break;
            }
          }
        }
        if (foundIndex >= 0) {
          markers[foundIndex] = ++markerCount;
          break;
        }
      }
    }
  }
  return {markers, count: markerCount};
}

function addRasterCropCopies<Format extends 'float32' | 'uint32'>(
  graph: GPUCommandGraph<void>,
  source: GraphDataView<Format>,
  destination: GraphDataView<Format>,
  sourceWidth: number,
  sourceName: string
): void {
  for (let row = 0; row < FOCUS_HEIGHT; row++) {
    const sourceOffset = ((FOCUS_ROW + row) * sourceWidth + FOCUS_COLUMN) * 4;
    const destinationOffset = row * FOCUS_WIDTH * 4;
    graph.addCopyPass({
      id: `crop-focus-${sourceName}-row-${row}`,
      resources: [
        {buffer: source, usage: 'copy-source'},
        {buffer: destination, usage: 'copy-destination'}
      ],
      compile: () => ({
        encode: ({commandEncoder, getBuffer}) =>
          commandEncoder.copyBufferToBuffer({
            sourceBuffer: getBuffer(source),
            sourceOffset,
            destinationBuffer: getBuffer(destination),
            destinationOffset,
            size: FOCUS_WIDTH * 4
          })
      })
    });
  }
}

export async function createFireScarTopology(
  ctx: SceneContext<FireScarTopologyOptions>
): Promise<SceneInstance<FireScarTopologyOptions>> {
  const dataset = ctx.datasets.get(DATASET_ID);
  const {device} = ctx;
  ctx.setStatus('Loading paired Sentinel-2 bands and source validity masks');
  const [
    nirBeforeRaster,
    swirBeforeRaster,
    nirAfterRaster,
    swirAfterRaster,
    validBeforeBytes,
    validAfterBytes
  ] = await Promise.all([
    dataset.loadRaster('nir_before', ctx.signal),
    dataset.loadRaster('swir22_before', ctx.signal),
    dataset.loadRaster('nir_after', ctx.signal),
    dataset.loadRaster('swir22_after', ctx.signal),
    loadValidity(dataset, 'land_valid_before.bin', ctx.signal),
    loadValidity(dataset, 'land_valid_after.bin', ctx.signal)
  ]);
  ctx.signal.throwIfAborted();
  const width = nirBeforeRaster.width;
  const height = nirBeforeRaster.height;
  const cellCount = width * height;
  const nirBeforeValues = requireFloatRaster(nirBeforeRaster.values, 'nir_before');
  const swirBeforeValues = requireFloatRaster(swirBeforeRaster.values, 'swir22_before');
  const nirAfterValues = requireFloatRaster(nirAfterRaster.values, 'nir_after');
  const swirAfterValues = requireFloatRaster(swirAfterRaster.values, 'swir22_after');
  const validBefore = validBeforeBytes;
  const validAfter = validAfterBytes;
  if (validBefore.length !== cellCount || validAfter.length !== cellCount) {
    throw new Error('dixie-spectral paired land-valid masks must match the reflectance grid');
  }
  const commonValidity = new Uint32Array(cellCount);
  for (let index = 0; index < cellCount; index++) {
    commonValidity[index] = validBefore[index] !== 0 && validAfter[index] !== 0 ? 1 : 0;
  }
  const focusSupport = new Uint32Array(FOCUS_CELL_COUNT);
  for (let row = 0; row < FOCUS_HEIGHT; row++) {
    const sourceOffset = (FOCUS_ROW + row) * width + FOCUS_COLUMN;
    focusSupport.set(
      commonValidity.subarray(sourceOffset, sourceOffset + FOCUS_WIDTH),
      row * FOCUS_WIDTH
    );
  }
  const focusMarkerData = createFocusMarkers(focusSupport);
  // Mask the derived continuous view before upload so invalid land-support pixels draw transparent.
  const maskedNirBefore = nirBeforeValues.slice();
  const maskedSwirBefore = swirBeforeValues.slice();
  const maskedNirAfter = nirAfterValues.slice();
  const maskedSwirAfter = swirAfterValues.slice();
  for (let index = 0; index < cellCount; index++) {
    if (!commonValidity[index]) {
      maskedNirBefore[index] = Number.NaN;
      maskedSwirBefore[index] = Number.NaN;
      maskedNirAfter[index] = Number.NaN;
      maskedSwirAfter[index] = Number.NaN;
    }
  }
  const commonValidCount = commonValidity.reduce((total, value) => total + value, 0);
  const focusSupportCount = focusSupport.reduce((total, value) => total + value, 0);
  const rasterSpec = dataset.rasterSpecs.nir_before;
  const boundsProjected = rasterSpec?.boundsProjected;
  const cellSizeM = rasterSpec?.cellSizeM;
  if (!Array.isArray(boundsProjected) || typeof cellSizeM !== 'number') {
    throw new Error('dixie-spectral requires projected raster bounds and cell size');
  }
  if (boundsProjected.length !== 4 || !boundsProjected.every(value => typeof value === 'number')) {
    throw new Error('dixie-spectral projected raster bounds must contain four numeric coordinates');
  }
  const [minimumEasting, minimumNorthing, maximumEasting, maximumNorthing] = [
    boundsProjected[0],
    boundsProjected[1],
    boundsProjected[2],
    boundsProjected[3]
  ];
  const frame = createUtmFrame(dataset.defaultOrigin, {
    boundsProjected: [minimumEasting, minimumNorthing, maximumEasting, maximumNorthing],
    cellSizeM,
    width,
    height
  });
  const focusBoundsProjected: readonly [number, number, number, number] = [
    minimumEasting + FOCUS_COLUMN * cellSizeM,
    maximumNorthing - (FOCUS_ROW + FOCUS_HEIGHT) * cellSizeM,
    minimumEasting + (FOCUS_COLUMN + FOCUS_WIDTH) * cellSizeM,
    maximumNorthing - FOCUS_ROW * cellSizeM
  ];
  const focusFrame = createUtmFrame(dataset.defaultOrigin, {
    boundsProjected: focusBoundsProjected,
    cellSizeM,
    width: FOCUS_WIDTH,
    height: FOCUS_HEIGHT
  });
  const focusMetadata = makeGPURasterMetadata({
    width: FOCUS_WIDTH,
    height: FOCUS_HEIGHT,
    cellSize: cellSizeM,
    origin: [-frame.halfWidth + FOCUS_COLUMN * cellSizeM, frame.halfHeight - FOCUS_ROW * cellSizeM]
  });
  const metadata = makeGPURasterMetadata({
    width,
    height,
    cellSize: cellSizeM,
    origin: [-frame.halfWidth, frame.halfHeight]
  });
  const resources = new SpatialAnalysisResources(device, 'fire-scar-topology');
  const inputBuffers = {
    nirBefore: resources.createBuffer('nir-before', maskedNirBefore),
    swirBefore: resources.createBuffer('swir22-before', maskedSwirBefore),
    nirAfter: resources.createBuffer('nir-after', maskedNirAfter),
    swirAfter: resources.createBuffer('swir22-after', maskedSwirAfter),
    commonValidity: resources.createBuffer('common-land-validity', commonValidity),
    focusSupport: resources.createBuffer('focus-support', focusSupport),
    focusMarkers: resources.createBuffer('focus-analyst-markers', focusMarkerData.markers),
    nbrBefore: resources.createBuffer('nbr-before', cellCount * 4),
    nbrAfter: resources.createBuffer('nbr-after', cellCount * 4),
    dnbr: resources.createBuffer('dnbr', cellCount * 4),
    histogram: resources.createBuffer('dnbr-histogram', HISTOGRAM_BINS * 4),
    otsuThreshold: resources.createBuffer('otsu-threshold', 4),
    validCount: resources.createBuffer('common-valid-count', 4)
  };
  const arithmeticParameters = resources.createParameterBuffer(
    'arithmetic-parameters',
    'float32',
    8
  );
  arithmeticParameters.write(
    getGPURasterArithmeticParameterValues({operation: 'normalizedDifference'})
  );
  const mainGraph = new GPUCommandGraph<void>(device, {id: 'fire-scar-dnbr-and-otsu'});
  const mainView = createGraphViewer(mainGraph);
  const sourceViews = {
    nirBefore: mainView('nir-before', inputBuffers.nirBefore, 'float32', cellCount),
    swirBefore: mainView('swir22-before', inputBuffers.swirBefore, 'float32', cellCount),
    nirAfter: mainView('nir-after', inputBuffers.nirAfter, 'float32', cellCount),
    swirAfter: mainView('swir22-after', inputBuffers.swirAfter, 'float32', cellCount),
    validity: mainView('common-land-validity', inputBuffers.commonValidity, 'uint32', cellCount),
    nbrBefore: mainView('nbr-before', inputBuffers.nbrBefore, 'float32', cellCount),
    nbrAfter: mainView('nbr-after', inputBuffers.nbrAfter, 'float32', cellCount),
    dnbr: mainView('dnbr', inputBuffers.dnbr, 'float32', cellCount),
    histogram: mainView('dnbr-histogram', inputBuffers.histogram, 'uint32', HISTOGRAM_BINS),
    otsuThreshold: mainView('otsu-threshold', inputBuffers.otsuThreshold, 'float32', 1),
    validCount: mainView('common-valid-count', inputBuffers.validCount, 'uint32', 1)
  };
  const arithmeticView = arithmeticParameters.importToGraph(mainGraph);
  mainGraph.add(
    new GPURasterArithmetic({
      id: 'nbr-before',
      cellCount,
      a: sourceViews.nirBefore,
      b: sourceViews.swirBefore,
      parameters: arithmeticView,
      output: {values: sourceViews.nbrBefore}
    })
  );
  mainGraph.add(
    new GPURasterArithmetic({
      id: 'nbr-after',
      cellCount,
      a: sourceViews.nirAfter,
      b: sourceViews.swirAfter,
      parameters: arithmeticView,
      output: {values: sourceViews.nbrAfter}
    })
  );
  const subtractParameters = resources.createParameterBuffer(
    'difference-parameters',
    'float32',
    8,
    getGPURasterArithmeticParameterValues({operation: 'subtract'})
  );
  mainGraph.add(
    new GPURasterArithmetic({
      id: 'dnbr-before-minus-after',
      cellCount,
      a: sourceViews.nbrBefore,
      b: sourceViews.nbrAfter,
      parameters: subtractParameters.importToGraph(mainGraph),
      output: {values: sourceViews.dnbr}
    })
  );
  const dnbrBand: GPURasterBufferBand<'float32'> = {
    id: 'dNBR',
    format: 'float32',
    storage: {kind: 'buffer', values: sourceViews.dnbr},
    validity: sourceViews.validity
  };
  new GPURasterHistogram({
    id: 'valid-dnbr-histogram',
    input: dnbrBand,
    output: sourceViews.histogram,
    domain: DNBR_DOMAIN
  }).addToGraph(mainGraph);
  new GPURasterOtsuThreshold({
    id: 'otsu-dnbr-split',
    histogram: sourceViews.histogram,
    domain: DNBR_DOMAIN,
    output: sourceViews.otsuThreshold
  }).addToGraph(mainGraph);
  mainGraph.add(
    new GPUReduction({
      id: 'count-common-valid-pixels',
      input: sourceViews.validity,
      output: sourceViews.validCount,
      operation: 'sum'
    })
  );
  const mainCompiled = resources.track(await mainGraph.compileAsync());
  const thresholdParameters = resources.createParameterBuffer('manual-threshold', 'float32', 1);
  const rangeParameters = resources.createParameterBuffer('manual-range', 'float32', 2);
  let latestVariant: TopologyVariant | null = null;
  let variantDirty = true;
  let readbackDirty = true;
  let focusDirty = true;
  let focusReadbackDirty = true;
  let otsuThresholdValue = Number.NaN;
  let nextVariantId = 0;
  const variantSwap = resources.track(new GraphSwap());
  const dnbrBufferBandForVariant = (graph: GPUCommandGraph<void>) => {
    const graphView = createGraphViewer(graph);
    const dnbr = graphView('variant-dnbr', inputBuffers.dnbr, 'float32', cellCount);
    const validity = graphView(
      'variant-validity',
      inputBuffers.commonValidity,
      'uint32',
      cellCount
    );
    return {
      graphView,
      band: {
        id: 'variant-dNBR',
        format: 'float32' as const,
        storage: {kind: 'buffer' as const, values: dnbr},
        validity
      } satisfies GPURasterBufferBand<'float32'>
    };
  };

  const buildVariant = async (state: VariantState): Promise<TopologyVariant> => {
    const serial = nextVariantId++;
    const graph = new GPUCommandGraph<void>(device, {id: `fire-scar-topology-${serial}`});
    const {graphView, band} = dnbrBufferBandForVariant(graph);
    const sourceProbeThreshold = graph.createDataView(band.storage.values.buffer, {
      format: 'float32',
      length: 1,
      byteOffset: SOURCE_PROBE_ROW * 4
    });
    const ownedBuffers: Buffer[] = [];
    const createBuffer = (name: string, byteLength: number) => {
      const buffer = resources.createBuffer(`v${serial}-${name}`, byteLength);
      ownedBuffers.push(buffer);
      return buffer;
    };
    const maskBuffer = createBuffer('threshold-mask', cellCount * 4);
    const thresholdCellCountBuffer = createBuffer('threshold-cell-count', 4);
    const cleanedMaskBuffer =
      state.cleanup === 'none' ? maskBuffer : createBuffer('cleaned-mask', cellCount * 4);
    const sparseLabelsBuffer = createBuffer('sparse-labels', cellCount * 4);
    const sparseValidityBuffer = createBuffer('sparse-validity', cellCount * 4);
    const convergedBuffer = createBuffer('converged', 4);
    const iterationsBuffer = createBuffer('iterations', 4);
    const denseLabelsBuffer = createBuffer('dense-labels', cellCount * 4);
    const denseValidityBuffer = createBuffer('dense-validity', cellCount * 4);
    const componentCountBuffer = createBuffer('component-count', 4);
    const requiredComponentCountBuffer = createBuffer('required-component-count', 4);
    const componentOverflowBuffer = createBuffer('component-overflow', 4);
    const measurementPixelCountsBuffer = createBuffer(
      'measurement-pixel-counts',
      COMPONENT_CAPACITY * 4
    );
    const measurementAreasBuffer = createBuffer('measurement-areas', COMPONENT_CAPACITY * 4);
    const classLabelsBuffer = createBuffer('class-labels', cellCount * 4);
    const classAdjacencyBuffer = createBuffer('class-adjacency', 4 * 4);
    const classWeightsBuffer = createBuffer('class-mean-differences', 4 * 4);
    const classMappingBuffer = createBuffer('class-mapping', 2 * 4);
    const classStatusBuffer = createBuffer('class-merge-check-status', 4);
    const views = {
      thresholdMask: graphView('threshold-mask', maskBuffer, 'uint32', cellCount),
      thresholdCellCount: graphView('threshold-cell-count', thresholdCellCountBuffer, 'uint32', 1),
      cleanedMask: graphView('cleaned-mask', cleanedMaskBuffer, 'uint32', cellCount),
      sparseLabels: graphView('sparse-labels', sparseLabelsBuffer, 'uint32', cellCount),
      sparseValidity: graphView('sparse-validity', sparseValidityBuffer, 'uint32', cellCount),
      converged: graphView('converged', convergedBuffer, 'uint32', 1),
      iterations: graphView('iterations', iterationsBuffer, 'uint32', 1),
      denseLabels: graphView('dense-labels', denseLabelsBuffer, 'uint32', cellCount),
      denseValidity: graphView('dense-validity', denseValidityBuffer, 'uint32', cellCount),
      componentCount: graphView('component-count', componentCountBuffer, 'uint32', 1),
      requiredComponentCount: graphView(
        'required-component-count',
        requiredComponentCountBuffer,
        'uint32',
        1
      ),
      componentOverflow: graphView('component-overflow', componentOverflowBuffer, 'uint32', 1),
      measurementPixelCounts: graphView(
        'measurement-pixel-counts',
        measurementPixelCountsBuffer,
        'uint32',
        COMPONENT_CAPACITY
      ),
      measurementAreas: graphView(
        'measurement-areas',
        measurementAreasBuffer,
        'float32',
        COMPONENT_CAPACITY
      ),
      classLabels: graphView('class-labels', classLabelsBuffer, 'uint32', cellCount),
      classAdjacency: graphView('class-adjacency', classAdjacencyBuffer, 'uint32', 4),
      classWeights: graphView('class-mean-differences', classWeightsBuffer, 'float32', 4),
      classMapping: graphView('class-mapping', classMappingBuffer, 'uint32', 2),
      classStatus: graphView('class-merge-check-status', classStatusBuffer, 'uint32', 1)
    };
    const thresholdView =
      state.thresholdOperation === 'range'
        ? rangeParameters.importToGraph(graph, 'manual-range')
        : state.thresholdMethod === 'otsu'
          ? graphView('otsu-threshold', inputBuffers.otsuThreshold, 'float32', 1)
          : state.thresholdMethod === 'source-probe'
            ? sourceProbeThreshold
            : thresholdParameters.importToGraph(graph, 'manual-threshold');
    new GPURasterThreshold({
      id: 'fire-scar-threshold',
      width,
      height,
      input: band,
      output: views.thresholdMask,
      threshold: thresholdView,
      operation: state.thresholdOperation,
      inclusive: state.thresholdInclusive
    }).addToGraph(graph);
    graph.add(
      new GPUReduction({
        id: 'count-threshold-selected-cells',
        input: views.thresholdMask,
        output: views.thresholdCellCount,
        operation: 'sum'
      })
    );
    const maskBand: GPURasterBufferBand<'uint32'> = {
      id: 'threshold-mask-band',
      format: 'uint32',
      storage: {kind: 'buffer', values: views.thresholdMask},
      validity: band.validity
    };
    let cleanedValidity = band.validity!;
    if (state.cleanup === 'median') {
      const medianBuffer = createBuffer('median-dnbr', cellCount * 4);
      const medianValidityBuffer = createBuffer('median-validity', cellCount * 4);
      const medianValues = graphView('median-dnbr', medianBuffer, 'float32', cellCount);
      cleanedValidity = graphView('median-validity', medianValidityBuffer, 'uint32', cellCount);
      graph.add(
        new GPURasterRankFilter({
          id: 'median-filter-dnbr',
          grid: metadata,
          input: band,
          radius: state.radius,
          quantile: 0.5,
          borderMode: 'nodata',
          noDataPolicy: 'ignore',
          output: {values: medianValues, validity: cleanedValidity}
        })
      );
      new GPURasterThreshold({
        id: 'threshold-median-dnbr',
        width,
        height,
        input: {
          id: 'median-dNBR-band',
          format: 'float32',
          storage: {kind: 'buffer', values: medianValues},
          validity: cleanedValidity
        },
        output: views.cleanedMask,
        threshold: thresholdView,
        operation: state.thresholdOperation,
        inclusive: state.thresholdInclusive
      }).addToGraph(graph);
    } else if (state.cleanup === 'opening' || state.cleanup === 'closing') {
      const cleanupValidityBuffer = createBuffer('cleanup-validity', cellCount * 4);
      cleanedValidity = graphView('cleanup-validity', cleanupValidityBuffer, 'uint32', cellCount);
      const morphologyProps = {
        id: `${state.cleanup}-binary-mask`,
        width,
        height,
        input: maskBand,
        radius: state.radius,
        mode: 'binary' as const,
        output: views.cleanedMask,
        outputValidity: cleanedValidity,
        noDataPolicy: 'ignore' as const,
        borderMode: 'nodata' as const
      };
      if (state.cleanup === 'opening') new GPURasterOpening(morphologyProps).addToGraph(graph);
      else new GPURasterClosing(morphologyProps).addToGraph(graph);
    }
    const cleanedBand: GPURasterBufferBand<'uint32'> = {
      id: 'cleaned-mask-band',
      format: 'uint32',
      storage: {kind: 'buffer', values: views.cleanedMask},
      validity: cleanedValidity
    };
    addKernelPass(graph, {
      id: 'label-valid-dnbr-classes',
      invocationCount: cellCount,
      bindings: [
        {name: 'mask', view: views.cleanedMask, type: 'u32', access: 'read'},
        {name: 'validity', view: cleanedValidity, type: 'u32', access: 'read'},
        {name: 'labels', view: views.classLabels, type: 'u32', access: 'read_write'}
      ],
      body: 'labels[labelsOffset + index] = select(0xffffffffu, min(mask[maskOffset + index], 1u), validity[validityOffset + index] != 0u);'
    });
    graph.add(
      new GPURasterRegionGraph({
        id: 'valid-dnbr-class-contact-graph',
        grid: metadata,
        labels: views.classLabels,
        values: band.storage.values,
        regionCount: 2,
        mergeMode: 'fixed',
        mergeThreshold: 0,
        maximumIterations: 1,
        output: {
          adjacency: views.classAdjacency,
          weights: views.classWeights,
          mapping: views.classMapping,
          status: views.classStatus
        }
      })
    );
    new GPURasterConnectedComponents({
      id: 'fire-scar-connected-components',
      width,
      height,
      input: cleanedBand,
      output: views.sparseLabels,
      outputValidity: views.sparseValidity,
      converged: views.converged,
      iterationCount: views.iterations,
      connectivity: state.connectivity === '4' ? 4 : 8,
      maximumIterations: Math.min(64, Math.ceil(Math.log2(cellCount)) + 2)
    }).addToGraph(graph);
    new GPURasterDenseComponents({
      id: 'dense-fire-scar-patch-ids',
      width,
      height,
      input: views.sparseLabels,
      inputValidity: views.sparseValidity,
      converged: views.converged,
      output: views.denseLabels,
      outputValidity: views.denseValidity,
      componentCount: views.componentCount,
      requiredComponentCount: views.requiredComponentCount,
      overflow: views.componentOverflow,
      capacity: COMPONENT_CAPACITY
    }).addToGraph(graph);
    new GPURasterRegionMeasurements({
      id: 'fire-scar-patch-measurements',
      metadata,
      labels: views.denseLabels,
      labelValidity: views.denseValidity,
      converged: views.converged,
      componentCount: views.componentCount,
      overflow: views.componentOverflow,
      intensity: band,
      output: {
        pixelCounts: views.measurementPixelCounts,
        intensityCounts: graphView(
          'unused-intensity-counts',
          createBuffer('measurement-intensity-counts', COMPONENT_CAPACITY * 4),
          'uint32',
          COMPONENT_CAPACITY
        ),
        intensitySums: graphView(
          'unused-intensity-sums',
          createBuffer('measurement-intensity-sums', COMPONENT_CAPACITY * 4),
          'float32',
          COMPONENT_CAPACITY
        ),
        intensityMinimums: graphView(
          'unused-intensity-minimums',
          createBuffer('measurement-intensity-minimums', COMPONENT_CAPACITY * 4),
          'float32',
          COMPONENT_CAPACITY
        ),
        intensityMaximums: graphView(
          'unused-intensity-maximums',
          createBuffer('measurement-intensity-maximums', COMPONENT_CAPACITY * 4),
          'float32',
          COMPONENT_CAPACITY
        ),
        intensityMeans: graphView(
          'unused-intensity-means',
          createBuffer('measurement-intensity-means', COMPONENT_CAPACITY * 4),
          'float32',
          COMPONENT_CAPACITY
        ),
        columnSums: graphView(
          'unused-column-sums',
          createBuffer('measurement-column-sums', COMPONENT_CAPACITY * 4),
          'float32',
          COMPONENT_CAPACITY
        ),
        rowSums: graphView(
          'unused-row-sums',
          createBuffer('measurement-row-sums', COMPONENT_CAPACITY * 4),
          'float32',
          COMPONENT_CAPACITY
        ),
        centroidColumns: graphView(
          'unused-centroid-columns',
          createBuffer('measurement-centroid-columns', COMPONENT_CAPACITY * 4),
          'float32',
          COMPONENT_CAPACITY
        ),
        centroidRows: graphView(
          'unused-centroid-rows',
          createBuffer('measurement-centroid-rows', COMPONENT_CAPACITY * 4),
          'float32',
          COMPONENT_CAPACITY
        ),
        areas: views.measurementAreas
      },
      capacity: COMPONENT_CAPACITY
    }).addToGraph(graph);
    const focusGraph = new GPUCommandGraph<void>(device, {id: `fire-scar-focus-${serial}`});
    const focusView = createGraphViewer(focusGraph);
    const focusOwnedBuffers: Buffer[] = [];
    const createFocusBuffer = (
      name: string,
      byteLength: number | Float32Array | Uint32Array | Int32Array
    ) => {
      const buffer = resources.createBuffer(`v${serial}-focus-${name}`, byteLength);
      focusOwnedBuffers.push(buffer);
      return buffer;
    };
    const focusMaskBuffer = createFocusBuffer('mask', FOCUS_CELL_COUNT * 4);
    const focusDnbrBuffer = createFocusBuffer('dnbr', FOCUS_CELL_COUNT * 4);
    const focusTopographyBuffer = createFocusBuffer('negative-dnbr', FOCUS_CELL_COUNT * 4);
    const skeletonBuffer = createFocusBuffer('skeleton', FOCUS_CELL_COUNT * 4);
    const skeletonConvergedBuffer = createFocusBuffer('skeleton-converged', 4);
    const skeletonCountBuffer = createFocusBuffer('skeleton-count', 4);
    const medialAxisBuffer = createFocusBuffer('medial-axis', FOCUS_CELL_COUNT * 4);
    const medialRadiusBuffer = createFocusBuffer('medial-radius', FOCUS_CELL_COUNT * 4);
    const maximumRadiusBuffer = createFocusBuffer('maximum-radius', 4);
    const watershedBuffer = createFocusBuffer('watershed', FOCUS_CELL_COUNT * 4);
    const watershedConvergedBuffer = createFocusBuffer('watershed-converged', 4);
    const focusRegionCount = focusMarkerData.count;
    const regionLabelsBuffer = createFocusBuffer('watershed-region-labels', FOCUS_CELL_COUNT * 4);
    const regionAdjacencyBuffer = createFocusBuffer(
      'watershed-region-adjacency',
      focusRegionCount ** 2 * 4
    );
    const regionWeightsBuffer = createFocusBuffer(
      'watershed-region-weights',
      focusRegionCount ** 2 * 4
    );
    const regionMappingBuffer = createFocusBuffer('watershed-region-mapping', focusRegionCount * 4);
    const regionStatusBuffer = createFocusBuffer('watershed-region-status', 4);
    const basinCountsBuffer = createFocusBuffer('watershed-basin-counts', focusRegionCount * 4);
    const basinChecksumsBuffer = createFocusBuffer(
      'watershed-basin-checksums',
      focusRegionCount * 4
    );
    const watershedMergedBuffer = createFocusBuffer('watershed-merged', FOCUS_CELL_COUNT * 4);
    const focusSourceViews = {
      fullMask: focusView('full-cleaned-mask', cleanedMaskBuffer, 'uint32', cellCount),
      fullDnbr: focusView('full-dnbr', inputBuffers.dnbr, 'float32', cellCount),
      mask: focusView('focus-mask', focusMaskBuffer, 'uint32', FOCUS_CELL_COUNT),
      dnbr: focusView('focus-dnbr', focusDnbrBuffer, 'float32', FOCUS_CELL_COUNT),
      topography: focusView(
        'focus-negative-dnbr',
        focusTopographyBuffer,
        'float32',
        FOCUS_CELL_COUNT
      ),
      support: focusView('focus-support', inputBuffers.focusSupport, 'uint32', FOCUS_CELL_COUNT),
      markers: focusView('focus-markers', inputBuffers.focusMarkers, 'uint32', FOCUS_CELL_COUNT),
      skeleton: focusView('focus-skeleton', skeletonBuffer, 'uint32', FOCUS_CELL_COUNT),
      skeletonConverged: focusView(
        'focus-skeleton-converged',
        skeletonConvergedBuffer,
        'uint32',
        1
      ),
      skeletonCount: focusView('focus-skeleton-count', skeletonCountBuffer, 'uint32', 1),
      medialAxis: focusView('focus-medial-axis', medialAxisBuffer, 'uint32', FOCUS_CELL_COUNT),
      medialRadius: focusView(
        'focus-medial-radius',
        medialRadiusBuffer,
        'float32',
        FOCUS_CELL_COUNT
      ),
      maximumRadius: focusView('focus-maximum-radius', maximumRadiusBuffer, 'float32', 1),
      watershed: focusView('focus-watershed', watershedBuffer, 'uint32', FOCUS_CELL_COUNT),
      watershedConverged: focusView(
        'focus-watershed-converged',
        watershedConvergedBuffer,
        'uint32',
        1
      ),
      regionLabels: focusView(
        'focus-watershed-region-labels',
        regionLabelsBuffer,
        'uint32',
        FOCUS_CELL_COUNT
      ),
      regionAdjacency: focusView(
        'focus-watershed-region-adjacency',
        regionAdjacencyBuffer,
        'uint32',
        focusRegionCount ** 2
      ),
      regionWeights: focusView(
        'focus-watershed-region-weights',
        regionWeightsBuffer,
        'float32',
        focusRegionCount ** 2
      ),
      regionMapping: focusView(
        'focus-watershed-region-mapping',
        regionMappingBuffer,
        'uint32',
        focusRegionCount
      ),
      regionStatus: focusView('focus-watershed-region-status', regionStatusBuffer, 'uint32', 1),
      basinCounts: focusView(
        'focus-watershed-basin-counts',
        basinCountsBuffer,
        'uint32',
        focusRegionCount
      ),
      basinChecksums: focusView(
        'focus-watershed-basin-checksums',
        basinChecksumsBuffer,
        'uint32',
        focusRegionCount
      ),
      watershedMerged: focusView(
        'focus-watershed-merged',
        watershedMergedBuffer,
        'uint32',
        FOCUS_CELL_COUNT
      )
    };
    addRasterCropCopies(
      focusGraph,
      focusSourceViews.fullMask,
      focusSourceViews.mask,
      width,
      'mask'
    );
    addRasterCropCopies(
      focusGraph,
      focusSourceViews.fullDnbr,
      focusSourceViews.dnbr,
      width,
      'dnbr'
    );
    const negativeDnbParameters = createFocusBuffer(
      'negate-parameters',
      getGPURasterArithmeticParameterValues({operation: 'multiply', constantB: -1})
    );
    focusGraph.add(
      new GPURasterArithmetic({
        id: 'focus-negative-dnbr',
        cellCount: FOCUS_CELL_COUNT,
        a: focusSourceViews.dnbr,
        parameters: focusView('focus-negate-parameters', negativeDnbParameters, 'float32', 8),
        output: {values: focusSourceViews.topography}
      })
    );
    focusGraph.add(
      new GPURasterSkeleton({
        id: 'focus-zhang-suen-skeleton',
        grid: focusMetadata,
        input: focusSourceViews.mask,
        maximumIterations: 64,
        output: {
          skeleton: focusSourceViews.skeleton,
          status: {converged: focusSourceViews.skeletonConverged}
        }
      })
    );
    focusGraph.add(
      new GPURasterMedialAxis({
        id: 'focus-maximal-inscribed-disks',
        grid: focusMetadata,
        input: focusSourceViews.mask,
        output: {medialAxis: focusSourceViews.medialAxis, radius: focusSourceViews.medialRadius}
      })
    );
    focusGraph.add(
      new GPUReduction({
        id: 'count-focus-skeleton-cells',
        input: focusSourceViews.skeleton,
        output: focusSourceViews.skeletonCount,
        operation: 'sum'
      })
    );
    focusGraph.add(
      new GPUReduction({
        id: 'maximum-focus-medial-radius',
        input: focusSourceViews.medialRadius,
        output: focusSourceViews.maximumRadius,
        operation: 'max'
      })
    );
    const focusWatershedBand: GPURasterBufferBand<'float32'> = {
      id: 'focus-negative-dNBR',
      format: 'float32',
      storage: {kind: 'buffer', values: focusSourceViews.topography},
      validity: focusSourceViews.support
    };
    focusGraph.add(
      new GPURasterWatershedSegmentation({
        id: 'focus-dnbr-marker-watershed',
        grid: focusMetadata,
        input: focusWatershedBand,
        markers: focusSourceViews.markers,
        connectivity: state.connectivity === '4' ? 4 : 8,
        maximumSteps: FOCUS_CELL_COUNT,
        output: {
          labels: focusSourceViews.watershed,
          status: {converged: focusSourceViews.watershedConverged}
        }
      })
    );
    addKernelPass(focusGraph, {
      id: 'normalize-focus-watershed-basin-ids',
      invocationCount: FOCUS_CELL_COUNT,
      bindings: [
        {name: 'watershed', view: focusSourceViews.watershed, type: 'u32', access: 'read'},
        {name: 'converged', view: focusSourceViews.watershedConverged, type: 'u32', access: 'read'},
        {name: 'labels', view: focusSourceViews.regionLabels, type: 'u32', access: 'read_write'}
      ],
      body: `let label = watershed[watershedOffset + index]; labels[labelsOffset + index] = select(0xffffffffu, label - 1u, converged[convergedOffset] != 0u && label > 0u && label <= ${focusRegionCount}u);`
    });
    addKernelPass(focusGraph, {
      id: 'summarize-focus-watershed-basins',
      invocationCount: focusRegionCount,
      bindings: [
        {name: 'labels', view: focusSourceViews.regionLabels, type: 'u32', access: 'read'},
        {name: 'counts', view: focusSourceViews.basinCounts, type: 'u32', access: 'read_write'},
        {
          name: 'checksums',
          view: focusSourceViews.basinChecksums,
          type: 'u32',
          access: 'read_write'
        }
      ],
      body: `var cellCount = 0u; var weightedLabelSum = 0u; for (var cell = 0u; cell < ${FOCUS_CELL_COUNT}u; cell++) { if (labels[labelsOffset + cell] == index) { cellCount++; weightedLabelSum += (cell + 1u) * (index + 1u); } } counts[countsOffset + index] = cellCount; checksums[checksumsOffset + index] = weightedLabelSum;`
    });
    focusGraph.add(
      new GPURasterRegionGraph({
        id: 'merge-focus-watershed-basins',
        grid: focusMetadata,
        labels: focusSourceViews.regionLabels,
        values: focusSourceViews.dnbr,
        regionCount: focusRegionCount,
        mergeMode: state.regionMergeMode,
        mergeThreshold: state.regionMergeThreshold,
        maximumIterations: state.regionMergeRounds,
        output: {
          adjacency: focusSourceViews.regionAdjacency,
          weights: focusSourceViews.regionWeights,
          mapping: focusSourceViews.regionMapping,
          status: focusSourceViews.regionStatus
        }
      })
    );
    addKernelPass(focusGraph, {
      id: 'apply-focus-watershed-region-mapping',
      invocationCount: FOCUS_CELL_COUNT,
      bindings: [
        {name: 'watershed', view: focusSourceViews.watershed, type: 'u32', access: 'read'},
        {name: 'converged', view: focusSourceViews.watershedConverged, type: 'u32', access: 'read'},
        {name: 'mapping', view: focusSourceViews.regionMapping, type: 'u32', access: 'read'},
        {name: 'merged', view: focusSourceViews.watershedMerged, type: 'u32', access: 'read_write'}
      ],
      body: `let label = watershed[watershedOffset + index]; if (converged[convergedOffset] == 0u || label == 0u || label > ${focusRegionCount}u) { merged[mergedOffset + index] = 0u; return; } merged[mergedOffset + index] = mapping[mappingOffset + label - 1u] + 1u;`
    });
    let compiled: CompiledGPUCommandGraph<void> | null = null;
    let focusCompiled: CompiledGPUCommandGraph<void> | null = null;
    try {
      compiled = await graph.compileAsync();
      focusCompiled = await focusGraph.compileAsync();
    } catch (error) {
      if (compiled) resources.release(compiled);
      for (const buffer of ownedBuffers) resources.release(buffer);
      for (const buffer of focusOwnedBuffers) resources.release(buffer);
      throw error;
    }
    if (!compiled || !focusCompiled) throw new Error('Dixie topology graphs did not compile');
    resources.track(compiled);
    resources.track(focusCompiled);
    const buffers: TopologyBuffers = {
      mask: maskBuffer,
      thresholdCellCount: thresholdCellCountBuffer,
      cleanedMask: cleanedMaskBuffer,
      sparseLabels: sparseLabelsBuffer,
      sparseValidity: sparseValidityBuffer,
      converged: convergedBuffer,
      iterations: iterationsBuffer,
      denseLabels: denseLabelsBuffer,
      denseValidity: denseValidityBuffer,
      componentCount: componentCountBuffer,
      requiredComponentCount: requiredComponentCountBuffer,
      componentOverflow: componentOverflowBuffer,
      measurementPixelCounts: measurementPixelCountsBuffer,
      measurementAreas: measurementAreasBuffer,
      classLabels: classLabelsBuffer,
      classAdjacency: classAdjacencyBuffer,
      classWeights: classWeightsBuffer,
      classStatus: classStatusBuffer
    };
    let variant!: TopologyVariant;
    let requestedSignature = '';
    let focusRequestedSignature = '';
    const focusBuffers = {
      mask: focusMaskBuffer,
      skeleton: skeletonBuffer,
      skeletonConverged: skeletonConvergedBuffer,
      skeletonCount: skeletonCountBuffer,
      medialAxis: medialAxisBuffer,
      medialRadius: medialRadiusBuffer,
      maximumRadius: maximumRadiusBuffer,
      watershed: watershedBuffer,
      watershedConverged: watershedConvergedBuffer,
      regionLabels: regionLabelsBuffer,
      regionAdjacency: regionAdjacencyBuffer,
      regionWeights: regionWeightsBuffer,
      regionMapping: regionMappingBuffer,
      regionStatus: regionStatusBuffer,
      basinCounts: basinCountsBuffer,
      basinChecksums: basinChecksumsBuffer,
      watershedMerged: watershedMergedBuffer,
      markers: inputBuffers.focusMarkers
    };
    const reader = new SummaryReader(
      resources,
      `topology-${serial}`,
      [
        {buffer: componentCountBuffer, size: 4},
        {buffer: requiredComponentCountBuffer, size: 4},
        {buffer: componentOverflowBuffer, size: 4},
        {buffer: convergedBuffer, size: 4},
        {buffer: iterationsBuffer, size: 4},
        {buffer: measurementPixelCountsBuffer, size: COMPONENT_CAPACITY * 4},
        {buffer: measurementAreasBuffer, size: COMPONENT_CAPACITY * 4},
        {buffer: classAdjacencyBuffer, size: 4 * 4},
        {buffer: classWeightsBuffer, size: 4 * 4},
        {buffer: classStatusBuffer, size: 4},
        {buffer: thresholdCellCountBuffer, size: 4}
      ],
      bytes => {
        if (
          latestVariant !== variant ||
          variant.key !== makeVariantKey(ctx.options) ||
          requestedSignature !== makeReadbackSignature(ctx.options)
        ) {
          return;
        }
        const view = new DataView(bytes);
        const count = view.getUint32(0, true);
        const requiredCount = view.getUint32(4, true);
        const overflow = view.getUint32(8, true) !== 0;
        const converged = view.getUint32(12, true) !== 0;
        const iterations = view.getUint32(16, true);
        const pixelCountsOffset = 20;
        const areasOffset = pixelCountsOffset + COMPONENT_CAPACITY * 4;
        let largestArea = 0;
        let selectedCells = 0;
        let selectedArea = 0;
        for (let index = 0; index < count; index++) {
          const area = view.getFloat32(areasOffset + index * 4, true);
          largestArea = Math.max(largestArea, area);
          selectedArea += area;
          selectedCells += view.getUint32(pixelCountsOffset + index * 4, true);
        }
        ctx.setReadout(
          'componentCount',
          !converged
            ? 'withheld (connectivity did not converge)'
            : overflow
              ? `${count} (capacity reached)`
              : formatCount(count)
        );
        ctx.setReadout(
          'largestPatch',
          !converged
            ? 'withheld (connectivity did not converge)'
            : overflow
              ? 'withheld (component capacity exceeded)'
              : `${(largestArea / 10000).toFixed(1)} ha`
        );
        ctx.setReadout(
          'selectedCells',
          !converged || overflow ? 'withheld (incomplete topology)' : formatCount(selectedCells)
        );
        ctx.setReadout(
          'selectedArea',
          !converged || overflow
            ? 'withheld (incomplete topology)'
            : `${(selectedArea / 10000).toFixed(1)} ha`
        );
        ctx.setReadout(
          'topologyStatus',
          !converged
            ? `not converged in ${iterations} rounds · measurements withheld`
            : overflow
              ? `capacity overflow · ${formatCount(requiredCount)} required, ${COMPONENT_CAPACITY} published`
              : `converged in ${iterations} rounds · within component capacity`
        );
        const contactOffset = 20 + COMPONENT_CAPACITY * 8;
        const sharedEdges = view.getUint32(contactOffset + 4, true);
        const meanDifference = view.getFloat32(contactOffset + 16 + 4, true);
        const mergePending = view.getUint32(contactOffset + 32, true) !== 0;
        const thresholdCells = view.getUint32(contactOffset + 36, true);
        ctx.setReadout('thresholdCells', `${formatCount(thresholdCells)} valid cells selected`);
        ctx.setReadout(
          'classBoundaryLength',
          `${((sharedEdges * cellSizeM) / 1000).toFixed(2)} km · ${formatCount(sharedEdges)} shared edges`
        );
        ctx.setReadout(
          'classMeanDifference',
          Number.isFinite(meanDifference)
            ? meanDifference.toFixed(3)
            : 'unavailable · no adjacent class means'
        );
        ctx.setReadout(
          'classContactStatus',
          `Observed class adjacency · no mapping applied${mergePending ? ' · zero-threshold candidate' : ''}`
        );
        ctx.requestLayers();
      }
    );
    const focusReader = new SummaryReader(
      resources,
      `focus-${serial}`,
      [
        {buffer: skeletonConvergedBuffer, size: 4},
        {buffer: skeletonCountBuffer, size: 4},
        {buffer: maximumRadiusBuffer, size: 4},
        {buffer: watershedConvergedBuffer, size: 4},
        {buffer: regionMappingBuffer, size: focusRegionCount * 4},
        {buffer: regionStatusBuffer, size: 4},
        {buffer: basinCountsBuffer, size: focusRegionCount * 4},
        {buffer: basinChecksumsBuffer, size: focusRegionCount * 4}
      ],
      bytes => {
        if (
          latestVariant !== variant ||
          variant.key !== makeVariantKey(ctx.options) ||
          focusRequestedSignature !== makeReadbackSignature(ctx.options)
        ) {
          return;
        }
        const view = new DataView(bytes);
        const skeletonConverged = view.getUint32(0, true) !== 0;
        const skeletonCount = view.getUint32(4, true);
        const maximumRadiusPixels = view.getFloat32(8, true);
        const watershedConverged = view.getUint32(12, true) !== 0;
        const regionMappings = Array.from(
          new Uint32Array(bytes.slice(16, 16 + focusRegionCount * 4))
        );
        let focusOffset = 16 + focusRegionCount * 4;
        const mergePending = view.getUint32(focusOffset, true) !== 0;
        focusOffset += 4;
        const basinCounts = Array.from(
          new Uint32Array(bytes.slice(focusOffset, focusOffset + focusRegionCount * 4))
        );
        focusOffset += focusRegionCount * 4;
        const basinChecksums = new Uint32Array(
          bytes.slice(focusOffset, focusOffset + focusRegionCount * 4)
        );
        const mergedRoots = new Set(regionMappings).size;
        const labeledCells = basinCounts.reduce((total, count) => total + count, 0);
        const weightedLabelSum = basinChecksums.reduce((total, checksum) => total + checksum, 0);
        ctx.setReadout(
          'skeletonPixels',
          skeletonConverged
            ? `${formatCount(skeletonCount)} skeleton cells · converged; max disk radius ${(maximumRadiusPixels * cellSizeM).toFixed(1)} m`
            : `skeleton withheld · not converged; max disk radius ${(maximumRadiusPixels * cellSizeM).toFixed(1)} m`
        );
        ctx.setReadout(
          'medialRadius',
          `${(maximumRadiusPixels * cellSizeM).toFixed(1)} m maximum discrete radius`
        );
        ctx.setReadout(
          'watershedStatus',
          watershedConverged
            ? mergePending
              ? `complete queue · merge incomplete at ${state.regionMergeRounds} rounds`
              : 'complete queue · no eligible merge remains'
            : 'incomplete watershed · merge withheld'
        );
        ctx.setReadout(
          'mergedBasinCount',
          watershedConverged
            ? `${mergedRoots} / ${focusMarkerData.count} roots`
            : 'withheld · incomplete'
        );
        ctx.setReadout(
          'focusRegionMapping',
          !watershedConverged
            ? 'withheld · watershed did not converge'
            : `${formatCount(labeledCells)} labeled cells · checksum ${formatCount(weightedLabelSum)} · cells by marker [${basinCounts.join(', ')}] · mapping ${regionMappings.map((root, index) => `${index + 1}→${root + 1}`).join(' ')} · ${state.regionMergeMode}, Δmean ≤ ${state.regionMergeThreshold.toFixed(3)}`
        );
        ctx.requestLayers();
      }
    );
    variant = {
      key: makeVariantKey(state),
      compiled,
      buffers,
      ownedBuffers,
      reader,
      focusCompiled,
      focusBuffers,
      focusReader,
      focusOwnedBuffers,
      get requestedSignature() {
        return requestedSignature;
      },
      set requestedSignature(value: string) {
        requestedSignature = value;
      },
      get focusRequestedSignature() {
        return focusRequestedSignature;
      },
      set focusRequestedSignature(value: string) {
        focusRequestedSignature = value;
      }
    };
    return variant;
  };

  const releaseVariant = (variant: TopologyVariant | null) => {
    if (!variant) return;
    variant.reader.release();
    variant.focusReader.release();
    resources.release(variant.compiled);
    resources.release(variant.focusCompiled);
    for (const buffer of variant.ownedBuffers) resources.release(buffer);
    for (const buffer of variant.focusOwnedBuffers) resources.release(buffer);
  };
  const startingState: VariantState = {
    thresholdMethod: ctx.options.thresholdMethod,
    thresholdOperation: ctx.options.thresholdOperation,
    thresholdInclusive: ctx.options.thresholdInclusive,
    cleanup: ctx.options.cleanup,
    radius: ctx.options.radius,
    connectivity: ctx.options.connectivity,
    regionMergeMode: ctx.options.regionMergeMode,
    regionMergeThreshold: ctx.options.regionMergeThreshold,
    regionMergeRounds: ctx.options.regionMergeRounds
  };
  latestVariant = await buildVariant(startingState);
  const baseReader = new SummaryReader(
    resources,
    'dnbr-summary',
    [
      {buffer: inputBuffers.validCount, size: 4},
      {buffer: inputBuffers.otsuThreshold, size: 4},
      {buffer: inputBuffers.histogram, size: HISTOGRAM_BINS * 4}
    ],
    bytes => {
      const view = new DataView(bytes);
      const validCount = view.getUint32(0, true);
      const otsuThreshold = view.getFloat32(4, true);
      const histogram = Uint32Array.from(new Uint32Array(bytes.slice(8, 8 + HISTOGRAM_BINS * 4)));
      ctx.setReadout('validPixels', `${formatCount(validCount)} / ${formatCount(cellCount)}`);
      ctx.setReadout(
        'otsuThreshold',
        Number.isFinite(otsuThreshold) ? otsuThreshold.toFixed(3) : 'n/a'
      );
      otsuThresholdValue = otsuThreshold;
      ctx.setReadout('activeThreshold', describeThreshold(ctx.options, otsuThreshold));
      ctx.setChart('dnbrHistogram', {
        kind: 'histogram',
        title: 'Valid dNBR values',
        xLabel: 'dNBR',
        yLabel: 'Cells',
        values: Float32Array.from(histogram),
        xDomain: DNBR_DOMAIN,
        description:
          'GPU histogram over the common valid land cells; the histogram domain is −2 to 2 dNBR.'
      });
      ctx.requestLayers();
    }
  );
  ctx.setReadout(
    'validPixels',
    `${formatCount(commonValidCount)} / ${formatCount(cellCount)} source-valid cells`
  );
  ctx.setReadout(
    'focusSupport',
    `${formatCount(focusSupportCount)} / ${FOCUS_CELL_COUNT} valid cells`
  );
  ctx.setReadout('focusMarkers', formatCount(focusMarkerData.count));
  ctx.setStatus(`${width} × ${height} cells · ${cellSizeM} m · EPSG:32610`);
  let baseEncoded = false;

  const requestVariant = () => {
    const nextState: VariantState = {
      thresholdMethod: ctx.options.thresholdMethod,
      thresholdOperation: ctx.options.thresholdOperation,
      thresholdInclusive: ctx.options.thresholdInclusive,
      cleanup: ctx.options.cleanup,
      radius: ctx.options.radius,
      connectivity: ctx.options.connectivity,
      regionMergeMode: ctx.options.regionMergeMode,
      regionMergeThreshold: ctx.options.regionMergeThreshold,
      regionMergeRounds: ctx.options.regionMergeRounds
    };
    const nextKey = makeVariantKey(nextState);
    if (latestVariant?.key === nextKey && !variantSwap.pending) return;
    void variantSwap
      .request({
        build: () => buildVariant(nextState),
        commit: next => {
          const previous = latestVariant;
          latestVariant = next;
          variantDirty = true;
          readbackDirty = true;
          focusDirty = true;
          focusReadbackDirty = true;
          releaseVariant(previous);
          ctx.requestLayers();
        },
        discard: releaseVariant
      })
      .catch(error => {
        ctx.setStatus(`Topology graph build failed: ${String(error)}`);
      });
  };

  return {
    getCompiledGraphs: () => [
      mainCompiled,
      ...(latestVariant ? [latestVariant.compiled, latestVariant.focusCompiled] : [])
    ],
    encode(commandEncoder) {
      if (!baseEncoded) {
        mainCompiled.encode(commandEncoder, {parameters: undefined});
        baseEncoded = true;
        baseReader.request(commandEncoder);
      }
      const variant = latestVariant;
      if (!variant) return;
      thresholdParameters.write(new Float32Array([ctx.options.thresholdValue]));
      rangeParameters.write(
        new Float32Array([
          Math.min(ctx.options.thresholdValue, ctx.options.rangeEndpoint),
          Math.max(ctx.options.thresholdValue, ctx.options.rangeEndpoint)
        ])
      );
      if (variantDirty) {
        variant.compiled.encode(commandEncoder, {parameters: undefined});
        variantDirty = false;
        focusDirty = true;
        focusReadbackDirty = true;
      }
      if (readbackDirty && !variant.reader.isPending) {
        variant.requestedSignature = makeReadbackSignature(ctx.options);
        variant.reader.request(commandEncoder);
        readbackDirty = false;
      } else variant.reader.flush(commandEncoder);
      if (isFocusMapLayer(ctx.options.mapLayer)) {
        if (focusDirty) {
          variant.focusCompiled.encode(commandEncoder, {parameters: undefined});
          variant.focusReader.markStale();
          focusDirty = false;
          focusReadbackDirty = true;
        }
        if (focusReadbackDirty && !variant.focusReader.isPending) {
          variant.focusRequestedSignature = makeReadbackSignature(ctx.options);
          variant.focusReader.request(commandEncoder);
          focusReadbackDirty = false;
        } else variant.focusReader.flush(commandEncoder);
      }
    },
    getLayers() {
      const variant = latestVariant;
      if (!variant) return [];
      if (
        ctx.options.mapLayer === 'dnbr' ||
        ctx.options.mapLayer === 'nbrBefore' ||
        ctx.options.mapLayer === 'nbrAfter'
      ) {
        const values =
          ctx.options.mapLayer === 'nbrBefore'
            ? inputBuffers.nbrBefore
            : ctx.options.mapLayer === 'nbrAfter'
              ? inputBuffers.nbrAfter
              : inputBuffers.dnbr;
        const valueRange: readonly [number, number] =
          ctx.options.mapLayer === 'dnbr' ? DNBR_DOMAIN : [-1, 1];
        return [
          new UtmRasterLayer({
            id: `fire-scar-${ctx.options.mapLayer}`,
            frame,
            values,
            valueFormat: 'float32',
            colormap: 'diverging',
            valueRange,
            opacity: 0.9,
            noDataColor: [0, 0, 0, 0]
          })
        ];
      }
      if (ctx.options.mapLayer === 'support') {
        return [
          new UtmRasterLayer({
            id: 'fire-scar-common-land-support',
            frame,
            values: inputBuffers.commonValidity,
            valueFormat: 'uint32',
            colormap: 'mask',
            color: [80, 171, 222, 220],
            noDataValue: 0,
            opacity: 0.9,
            noDataColor: [0, 0, 0, 0]
          })
        ];
      }
      if (ctx.options.mapLayer === 'classContacts') {
        return [
          new UtmRasterLayer({
            id: 'fire-scar-valid-dnbr-classes',
            frame,
            values: variant.buffers.classLabels,
            valueFormat: 'uint32',
            colormap: 'category',
            palette: [
              [85, 102, 116, 235],
              [242, 92, 54, 230]
            ],
            noDataValue: 0xffffffff,
            opacity: 0.9,
            noDataColor: [0, 0, 0, 0]
          })
        ];
      }
      if (isFocusMapLayer(ctx.options.mapLayer)) {
        const focusValues =
          ctx.options.mapLayer === 'focusMask'
            ? variant.focusBuffers.mask
            : ctx.options.mapLayer === 'skeleton'
              ? variant.focusBuffers.skeleton
              : ctx.options.mapLayer === 'medialAxis'
                ? variant.focusBuffers.medialAxis
                : ctx.options.mapLayer === 'watershed'
                  ? variant.focusBuffers.watershed
                  : ctx.options.mapLayer === 'watershedMerged'
                    ? variant.focusBuffers.watershedMerged
                    : variant.focusBuffers.markers;
        if (ctx.options.mapLayer === 'watershed' || ctx.options.mapLayer === 'watershedMerged') {
          return [
            new UtmRasterLayer({
              id: `fire-scar-focus-${ctx.options.mapLayer}`,
              frame: focusFrame,
              values: focusValues,
              valueFormat: 'uint32',
              colormap: 'category',
              palette: REGION_COLORS,
              discardAtOrBelow: 0,
              noDataValue: 0xfffffffe,
              opacity: 0.9,
              noDataColor: [0, 0, 0, 0]
            })
          ];
        }
        const color: readonly [number, number, number, number] =
          ctx.options.mapLayer === 'skeleton'
            ? [75, 28, 113, 255]
            : ctx.options.mapLayer === 'medialAxis'
              ? [66, 214, 177, 255]
              : ctx.options.mapLayer === 'watershedMarkers'
                ? [0, 71, 171, 255]
                : [242, 92, 54, 230];
        return [
          new UtmRasterLayer({
            id: `fire-scar-focus-${ctx.options.mapLayer}`,
            frame: focusFrame,
            values: focusValues,
            valueFormat: 'uint32',
            colormap: 'mask',
            color,
            noDataValue: ctx.options.mapLayer === 'skeleton' ? 0xfffffffe : 0,
            opacity: 0.94,
            noDataColor: [0, 0, 0, 0]
          })
        ];
      }
      const values =
        ctx.options.mapLayer === 'threshold'
          ? variant.buffers.mask
          : ctx.options.mapLayer === 'cleaned'
            ? variant.buffers.cleanedMask
            : variant.buffers.denseLabels;
      if (ctx.options.mapLayer === 'regions') {
        return [
          new UtmRasterLayer({
            id: `fire-scar-${ctx.options.mapLayer}`,
            frame,
            values,
            valueFormat: 'uint32',
            colormap: 'category',
            palette: REGION_COLORS,
            noDataValue: 0,
            opacity: 0.9,
            noDataColor: [0, 0, 0, 0]
          })
        ];
      }
      return [
        new UtmRasterLayer({
          id: `fire-scar-${ctx.options.mapLayer}`,
          frame,
          values,
          valueFormat: 'uint32',
          colormap: 'mask',
          color: [242, 92, 54, 230],
          noDataValue: 0,
          opacity: 0.92,
          noDataColor: [0, 0, 0, 0]
        })
      ];
    },
    setOption(id) {
      if (id === 'thresholdValue' || id === 'rangeEndpoint') {
        variantDirty = true;
        readbackDirty = true;
        focusDirty = true;
        focusReadbackDirty = true;
        latestVariant?.reader.markStale();
        latestVariant?.focusReader.markStale();
        setTopologyReadoutsUpdating(ctx);
        ctx.setReadout('activeThreshold', describeThreshold(ctx.options, otsuThresholdValue));
        ctx.requestLayers();
      } else if (
        id === 'thresholdMethod' ||
        id === 'thresholdOperation' ||
        id === 'thresholdInclusive' ||
        id === 'cleanup' ||
        id === 'radius' ||
        id === 'connectivity' ||
        id === 'regionMergeMode' ||
        id === 'regionMergeThreshold' ||
        id === 'regionMergeRounds'
      ) {
        latestVariant?.reader.markStale();
        latestVariant?.focusReader.markStale();
        readbackDirty = true;
        focusDirty = true;
        focusReadbackDirty = true;
        setTopologyReadoutsUpdating(ctx);
        ctx.setReadout('activeThreshold', describeThreshold(ctx.options, otsuThresholdValue));
        requestVariant();
      } else {
        if (id === 'mapLayer' && isFocusMapLayer(ctx.options.mapLayer)) {
          focusDirty = true;
          focusReadbackDirty = true;
          latestVariant?.focusReader.markStale();
          ctx.setReadout('skeletonPixels', 'updating…');
          ctx.setReadout('medialRadius', 'updating…');
          ctx.setReadout('mergedBasinCount', 'updating…');
          ctx.setReadout('watershedStatus', 'updating…');
          ctx.setReadout('focusRegionMapping', 'updating watershed merge…');
        }
        ctx.requestLayers();
      }
    },
    onThemeChange() {
      ctx.requestLayers();
    },
    destroy() {
      baseReader.release();
      releaseVariant(latestVariant);
      latestVariant = null;
      resources.destroy();
    }
  };
}

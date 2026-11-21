// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  getViewBindingRange,
  GPUScan,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {GPURaster, type GPURasterBand, type GPURasterMetadata} from '@luma.gl/graph/gpu-raster';
import type {GPUApproximationStatusPort} from '../../analysis-core/contracts/status';
import {
  getGraphOutputViews,
  validateCapacity,
  validateMaximumCount,
  validateFiniteNumber,
  validateRequirement,
  validateCommandNodeBudget,
  validateGPUStorageBindingByteLength,
  validateSinglePackedView,
  validateViewLength
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  createFillNode,
  createWGSLKernelNode,
  type WGSLKernelBinding
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {GET_NAN_WGSL, IS_FINITE_VALUE_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {UINT64_WGSL} from '../../analysis-core/kernels/uint64-wgsl';
import {SCALED_QUOTIENT_WGSL} from '../../analysis-core/kernels/scaled-quotient-wgsl';
import {
  getRasterBandViews,
  getRasterBufferBandNodes,
  validateRasterBandBelongsToGraph,
  validateRasterGrid,
  validateRasterParameters
} from '../raster-band-utils';
import {
  requireRasterLength,
  RASTER_BAND_CANONICALIZATION_NODE_COUNT
} from '../raster-expansion-utils';
import {GPU_RASTER_CONTRAST_EQUALIZATION_PARAMETER_LENGTH} from './raster-contrast-equalization-parameters';
import {CLIP_CONTRAST_HISTOGRAM_WGSL} from './contrast-equalization-wgsl';
import {validateGraphChunkBindingRanges} from '../../analysis-core/kernels/graph-chunk-dispatch';
import {
  validateRasterFractionalBindingByteLength,
  validateRasterFractionalWritableViews
} from '../fractional-coverage/raster-fractional-tile-kernel';

/** Cell and reflected-image counts are accumulated exactly as uint32; storage limits are device checks. */
const MAXIMUM_CELL_COUNT = 0xffffffff;
/** Histogram addressing is uint32; actual histogram, CDF and mapping bindings are device checked. */
const MAXIMUM_HISTOGRAM_LENGTH = 0xffffffff;
/** Each tile clips a bounded histogram serially; never a cell-sized loop. */
const MAXIMUM_BIN_COUNT = 256;
/** Bounds reflected tile area to 65,536 virtual pixels and integer map products below 2^31. */
const MAXIMUM_TILE_SIDE = 256;
/** Each productive redistribution sweep removes at least one excess count from a bounded tile. */
const MAXIMUM_REDISTRIBUTION_SWEEPS = MAXIMUM_TILE_SIDE * MAXIMUM_TILE_SIDE;
/** A packed uint32-length scan has at most four 256-way levels, within this allowance. */
const SCAN_NODE_ALLOWANCE = 16;
const OPERATION = 'GPURasterContrastEqualization';

export type GPURasterContrastEqualizationOutput = {
  /** Dense row-major output: normalized CLAHE mapping, or values in the reference domain. */
  values: GraphDataView<'float32'>;
  /** Invalid source cells (and a missing reference distribution) produce zero and NaN values. */
  validity: GraphDataView<'uint32'>;
  /** Quantization is deliberate; approximation is always 1. */
  status: GPUApproximationStatusPort & {approximation: GraphDataView<'uint32'>};
};

export type GPURasterContrastEqualizationProps = {
  id?: string;
  grid: GPURasterMetadata;
  input: GPURasterBand;
  /** Compile-time method. Defaults to clahe. */
  mode?: 'clahe' | 'match-histograms';
  /** Required only for matching; its grid and cell count may differ from the input. */
  reference?: {grid: GPURasterMetadata; input: GPURasterBand};
  /** Compile-time bins in [2, 256], default 256. */
  binCount?: number;
  /** Compile-time tile width/height in [1, 256], default 32 each; used only by CLAHE. */
  tileWidth?: number;
  tileHeight?: number;
  /** Per-frame settings, packed by getGPURasterContrastEqualizationParameterValues. */
  parameters: GraphDataView<'float32'>;
  output: GPURasterContrastEqualizationOutput;
};

/**
 * CLAHE with reflected edge tiles and bilinear tile mappings, or quantized CDF histogram matching.
 * Per-frame: band/validity contents and parameter-buffer domains/clipLimit. Compile-time: grids,
 * band metadata/calibration, mode, tile dimensions, binCount and view topology. No ground distances
 * or world coordinates are read. Canonical raster bands honor validity, nodata and calibration.
 *
 * CLAHE follows scikit-image 0.24's 14-bit binning and integer mappings, but takes an explicit
 * domain and returns the raw mapping divided by 16383 (no global output min/max rescale). Empty
 * tiles are omitted from interpolation and remaining weights renormalized. Matching linearly
 * interpolates the inverse reference CDF at occupied bins, with endpoint clamping like np.interp.
 * It matches quantized bin representatives, rather than every unique float. Invalid cells stay
 * invalid; empty reference distributions invalidate all matching outputs. See the API page for
 * parameter mapping and oracle differences. Large images must be explicitly tiled by the caller.
 */
export class GPURasterContrastEqualization implements GPUCommandNodeProducer {
  readonly id: string;
  readonly props: GPURasterContrastEqualizationProps;
  readonly binCount: number;
  readonly tileWidth: number;
  readonly tileHeight: number;
  readonly tileColumns: number;
  readonly tileRows: number;
  readonly histogramLength: number;
  readonly estimatedCommandNodeCount: number;

  constructor(props: GPURasterContrastEqualizationProps) {
    this.id = props.id ?? 'raster-contrast-equalization';
    this.props = props;
    const {id} = this;
    const matching = props.mode === 'match-histograms';
    validateRequirement(
      id,
      props.mode === undefined || props.mode === 'clahe' || matching,
      'mode must be clahe or match-histograms'
    );
    validateRequirement(
      id,
      matching === Boolean(props.reference),
      'reference must exist exactly when mode is match-histograms'
    );
    const inputs = [
      ...getRasterBandViews(props.input),
      props.parameters,
      ...(props.reference ? getRasterBandViews(props.reference.input) : [])
    ];
    const cellCount = validateRasterGrid(id, props.grid);
    validateMaximumCount(id, 'grid cell count', cellCount, MAXIMUM_CELL_COUNT);
    for (const [name, band, grid] of [
      ['input', props.input, props.grid],
      ...(props.reference
        ? [['reference.input', props.reference.input, props.reference.grid] as const]
        : [])
    ] as const) {
      validateFiniteNumber(id, `${name}.scale`, band.scale ?? 1);
      validateFiniteNumber(id, `${name}.offset`, band.offset ?? 0);
      const count = validateRasterGrid(id, grid);
      validateMaximumCount(id, `${name} cell count`, count, MAXIMUM_CELL_COUNT);
      if (band.storage.kind === 'buffer') {
        validateSinglePackedView(
          band.storage.values,
          ['float32', 'sint32', 'uint32'],
          `${id} ${name}.values`
        );
        validateViewLength(id, `${name}.values`, band.storage.values, count);
      }
      if (band.validity) {
        validateSinglePackedView(band.validity, ['uint32'], `${id} ${name}.validity`);
        validateViewLength(id, `${name}.validity`, band.validity, count);
      }
      // Public non-owning raster validation checks declared formats, native nodata, texture
      // channels/extents and validity ownership without declaring any graph resources.
      new GPURaster({id: `${id} ${name}`, metadata: grid, bands: [band]});
    }
    this.binCount = props.binCount ?? 256;
    this.tileWidth = props.tileWidth ?? 32;
    this.tileHeight = props.tileHeight ?? 32;
    validateCapacity(id, 'binCount', this.binCount, 2, MAXIMUM_BIN_COUNT);
    validateCapacity(id, 'tileWidth', this.tileWidth, 1, MAXIMUM_TILE_SIDE);
    validateCapacity(id, 'tileHeight', this.tileHeight, 1, MAXIMUM_TILE_SIDE);
    this.tileColumns = matching ? 1 : Math.ceil(props.grid.width / this.tileWidth);
    this.tileRows = matching ? 1 : Math.ceil(props.grid.height / this.tileHeight);
    validateMaximumCount(
      id,
      'reflected tile cell count',
      matching ? cellCount : this.tileColumns * this.tileRows * this.tileWidth * this.tileHeight,
      MAXIMUM_CELL_COUNT
    );
    this.histogramLength = (matching ? 2 : this.tileColumns * this.tileRows) * this.binCount;
    validateMaximumCount(id, 'histogramLength', this.histogramLength, MAXIMUM_HISTOGRAM_LENGTH);
    // The single CDF scan spans both matching planes. Bound its total before allocating scratch;
    // each plane's counts remain integers, and no prefix relies on uint32 wraparound.
    validateMaximumCount(
      id,
      'cumulative histogram count',
      matching
        ? cellCount + props.reference!.grid.width * props.reference!.grid.height
        : this.tileColumns * this.tileRows * this.tileWidth * this.tileHeight,
      MAXIMUM_CELL_COUNT
    );
    validateRasterParameters(
      id,
      props.parameters,
      GPU_RASTER_CONTRAST_EQUALIZATION_PARAMETER_LENGTH
    );
    for (const [name, view, format, count] of [
      ['output.values', props.output.values, 'float32', cellCount],
      ['output.validity', props.output.validity, 'uint32', cellCount],
      ['output.status.approximation', props.output.status.approximation, 'uint32', 1]
    ] as const) {
      validateSinglePackedView(view, [format], `${id} ${name}`);
      requireRasterLength(view, count, id, name);
    }
    validateRasterFractionalWritableViews(id, props.output, inputs, [
      {
        writes: [props.output.values, props.output.validity, props.output.status.approximation],
        reads: [props.parameters]
      }
    ]);
    this.estimatedCommandNodeCount =
      (matching ? 2 : 1) * RASTER_BAND_CANONICALIZATION_NODE_COUNT + SCAN_NODE_ALLOWANCE + 6;
    validateCommandNodeBudget(id, this.estimatedCommandNodeCount);
  }

  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, binCount, histogramLength, tileColumns, tileRows, tileWidth, tileHeight} =
      this;
    // Device-dependent checks precede all node/scratch declarations.
    const views = [
      ...getRasterBandViews(props.input),
      props.parameters,
      ...getGraphOutputViews(props.output),
      ...(props.reference ? getRasterBandViews(props.reference.input) : [])
    ];
    validateGraphChunkBindingRanges(
      graph,
      id,
      Object.fromEntries(views.map((view, index) => [`view.${index}`, view]))
    );
    for (const view of views) {
      validateGPUStorageBindingByteLength(
        graph.device,
        id,
        'packed view',
        getViewBindingRange(view).size,
        'reduce the packed view length'
      );
    }
    validateGPUStorageBindingByteLength(
      graph.device,
      id,
      'histogramLength scratch',
      histogramLength * 4,
      'reduce the grid or binCount'
    );
    for (const view of views) {
      validateRasterFractionalBindingByteLength(
        graph.device,
        id,
        view.buffer.id,
        getViewBindingRange(view).size,
        'reduce the packed view length',
        view.buffer.byteLength
      );
    }
    for (const [name, length] of [
      ['source canonicalization', props.grid.width * props.grid.height],
      [
        'reference canonicalization',
        props.reference ? props.reference.grid.width * props.reference.grid.height : 0
      ],
      ['histogram/CDF/mapping scratch', histogramLength],
      ['tile counts', props.mode === 'match-histograms' ? 2 : tileColumns * tileRows]
    ] as const)
      validateRasterFractionalBindingByteLength(
        graph.device,
        id,
        name,
        length * 4,
        'reduce grid dimensions or binCount'
      );
    validateRasterBandBelongsToGraph(id, graph, props.input, []);
    if (props.reference) validateRasterBandBelongsToGraph(id, graph, props.reference.input, []);
    // Includes outputs and parameters; buffer bands were checked above as well.
    for (const view of views)
      validateRequirement(id, view.buffer.graph === graph, 'views must belong to the target graph');
    const matching = props.mode === 'match-histograms';
    const source = getRasterBufferBandNodes(graph, `${id}-source`, props.input, props.grid, true);
    const nodes: GPUCommandNode<Parameters>[] = [...source.nodes];
    const tileCount = matching ? 2 : tileColumns * tileRows;
    const histogram = createTransientView(graph, `${id}-histogram`, 'uint32', histogramLength);
    const counts = createTransientView(graph, `${id}-counts`, 'uint32', tileCount);
    const cumulative = createTransientView(graph, `${id}-cumulative`, 'uint32', histogramLength);
    const mapping = createTransientView(graph, `${id}-mapping`, 'float32', histogramLength);
    const parameters: WGSLKernelBinding = {
      name: 'parameters',
      view: props.parameters,
      type: 'f32',
      access: 'read'
    };
    const declarations = `${IS_FINITE_VALUE_WGSL}
${SCALED_QUOTIENT_WGSL}
${GET_NAN_WGSL}
const BIN_COUNT: u32 = ${binCount}u;
fn getBin(value: f32, domain: u32) -> u32 {
  let minimum = parameters[parametersOffset + domain];
  let maximum = parameters[parametersOffset + domain + 1u];
  let normalized = clamp(getScaledQuotient(clamp(value, minimum, maximum) - minimum, maximum - minimum, 1.0, 0), 0.0, 1.0);
  ${matching ? 'return min(BIN_COUNT - 1u, u32(floor(normalized * f32(BIN_COUNT - 1u) + 0.5)));' : 'return min(BIN_COUNT - 1u, u32(round(normalized * 16383.0)) / (1u + 16384u / BIN_COUNT));'}
}
fn reflectCell(coordinate: u32, extent: u32) -> u32 {
  if (extent == 1u) { return 0u; }
  let period = 2u * (extent - 1u);
  let folded = coordinate % period;
  return min(folded, period - folded);
}`;
    nodes.push(
      createFillNode(graph, {
        id: `${id}-clear-histogram`,
        operation: OPERATION,
        view: histogram,
        type: 'u32',
        value: '0u'
      }),
      createFillNode(graph, {
        id: `${id}-clear-counts`,
        operation: OPERATION,
        view: counts,
        type: 'u32',
        value: '0u'
      })
    );
    const addHistogram = (
      band: typeof source.band,
      grid: GPURasterMetadata,
      reference: boolean
    ): void => {
      const virtualWidth = matching ? grid.width : tileColumns * tileWidth;
      const virtualHeight = matching ? grid.height : tileRows * tileHeight;
      nodes.push(
        createWGSLKernelNode(graph, {
          id: `${id}-${reference ? 'reference' : 'source'}-histogram`,
          operation: OPERATION,
          variant: 'histogram',
          invocationCount: virtualWidth * virtualHeight,
          declarations,
          bindings: [
            parameters,
            {name: 'source', view: band.storage.values, type: 'f32', access: 'read'},
            {name: 'sourceValidity', view: band.validity!, type: 'u32', access: 'read'},
            {name: 'histogram', view: histogram, type: 'atomic<u32>', access: 'read_write'},
            {name: 'counts', view: counts, type: 'atomic<u32>', access: 'read_write'}
          ],
          body: `let column = index % ${virtualWidth}u;
let row = index / ${virtualWidth}u;
let cell = reflectCell(row, ${grid.height}u) * ${grid.width}u + reflectCell(column, ${grid.width}u);
let value = source[sourceOffset + cell];
if (sourceValidity[sourceValidityOffset + cell] != 0u && isFiniteValue(value)) {
  let tile = ${matching ? (reference ? '1u' : '0u') : `(row / ${tileHeight}u) * ${tileColumns}u + column / ${tileWidth}u`};
  let bin = getBin(value, ${reference ? '2u' : '0u'});
  atomicAdd(&histogram[histogramOffset + tile * BIN_COUNT + bin], 1u);
  atomicAdd(&counts[countsOffset + tile], 1u);
}`
        })
      );
    };
    addHistogram(source.band, props.grid, false);
    if (props.reference) {
      const reference = getRasterBufferBandNodes(
        graph,
        `${id}-reference`,
        props.reference.input,
        props.reference.grid,
        true
      );
      nodes.push(...reference.nodes);
      addHistogram(reference.band, props.reference.grid, true);
    } else {
      nodes.push(
        createWGSLKernelNode(graph, {
          id: `${id}-clip`,
          operation: OPERATION,
          variant: 'clip',
          invocationCount: tileCount,
          declarations: `const BIN_COUNT: u32 = ${binCount}u;
const MAXIMUM_REDISTRIBUTION_SWEEPS: u32 = ${MAXIMUM_REDISTRIBUTION_SWEEPS}u;`,
          bindings: [
            parameters,
            {name: 'histogram', view: histogram, type: 'u32', access: 'read_write'},
            {name: 'counts', view: counts, type: 'u32', access: 'read'}
          ],
          body: CLIP_CONTRAST_HISTOGRAM_WGSL
        })
      );
    }
    // One global inclusive scan; subtract the preceding tile's final prefix in getCumulative.
    nodes.push(
      ...new GPUScan({
        id: `${id}-scan`,
        input: histogram,
        output: cumulative,
        mode: 'inclusive'
      }).getCommandNodes(graph)
    );
    nodes.push(
      createWGSLKernelNode(graph, {
        id: `${id}-mapping`,
        operation: OPERATION,
        variant: 'mapping',
        invocationCount: histogramLength,
        declarations: `${
          matching
            ? `${declarations}\n${UINT64_WGSL}`
            : `const BIN_COUNT: u32 = ${binCount}u;
${GET_NAN_WGSL}`
        }
fn getCumulative(tile: u32, bin: u32) -> u32 {
  var preceding = 0u;
  if (tile > 0u) { preceding = cumulative[cumulativeOffset + tile * BIN_COUNT - 1u]; }
  return cumulative[cumulativeOffset + tile * BIN_COUNT + bin] - preceding;
}`,
        bindings: [
          ...(matching ? [parameters] : []),
          {name: 'counts', view: counts, type: 'u32', access: 'read'},
          {name: 'cumulative', view: cumulative, type: 'u32', access: 'read'},
          {name: 'mapping', view: mapping, type: 'f32', access: 'read_write'}
        ],
        body: matching
          ? `if (index >= BIN_COUNT) { return; }
var result = getNaN();
if (counts[countsOffset] > 0u && counts[countsOffset + 1u] > 0u) {
  let sourceCount = counts[countsOffset];
  let referenceCount = counts[countsOffset + 1u];
  // Exact wide products of uint32 counts. Compare before any float conversion:
  // adjacent normalized CDF knots can round to the same f32 near one.
  let sourceProduct = multiplyUint32Wide(getCumulative(0u, index), referenceCount);
  var previousCount = 0u;
  var previousBin = 0u;
  var found = false;
  for (var bin = 0u; bin < BIN_COUNT; bin++) {
    let count = getCumulative(1u, bin);
    if (count == previousCount) { continue; }
    let upperProduct = multiplyUint32Wide(count, sourceCount);
    if (!found) { result = f32(bin); }
    if (isLessOrEqualUint64(sourceProduct, upperProduct)) {
      if (found) {
        // Subtract exact integer products first; converting the two knots and then subtracting
        // loses low-count gaps. The remaining ratio is bounded in [0,1] without cancellation.
        let lowerProduct = multiplyUint32Wide(previousCount, sourceCount);
        let numerator = subtractUint64(sourceProduct, lowerProduct);
        let denominator = multiplyUint32Wide(count - previousCount, sourceCount);
        let fraction = clamp(uint64ToFloat(numerator) / uint64ToFloat(denominator), 0.0, 1.0);
        result = mix(f32(previousBin), f32(bin), fraction);
      }
      break;
    }
    result = f32(bin); previousCount = count; previousBin = bin; found = true;
  }
  result = mix(parameters[parametersOffset + 2u], parameters[parametersOffset + 3u], result / f32(BIN_COUNT - 1u));
}
mapping[mappingOffset + index] = result;`
          : `let tile = index / BIN_COUNT;
let count = counts[countsOffset + tile];
var result = getNaN();
if (count > 0u) {
  // Integer division prevents an approximate float division crossing an integer mapping boundary.
  result = f32(min(16383u, (getCumulative(tile, index % BIN_COUNT) * 16383u) / count)) / 16383.0;
}
mapping[mappingOffset + index] = result;`
      })
    );
    nodes.push(
      createWGSLKernelNode(graph, {
        id: `${id}-apply`,
        operation: OPERATION,
        variant: 'apply',
        invocationCount: props.grid.width * props.grid.height,
        declarations,
        bindings: [
          parameters,
          {name: 'source', view: source.band.storage.values, type: 'f32', access: 'read'},
          {name: 'sourceValidity', view: source.band.validity!, type: 'u32', access: 'read'},
          {name: 'mapping', view: mapping, type: 'f32', access: 'read'},
          {name: 'values', view: props.output.values, type: 'f32', access: 'read_write'},
          {name: 'validity', view: props.output.validity, type: 'u32', access: 'read_write'},
          {
            name: 'approximation',
            view: props.output.status.approximation,
            type: 'u32',
            access: 'read_write'
          }
        ],
        body: `let value = source[sourceOffset + index];
var result = getNaN();
if (sourceValidity[sourceValidityOffset + index] != 0u && isFiniteValue(value)) {
  let bin = getBin(value, 0u);
  ${
    matching
      ? 'result = mapping[mappingOffset + bin];'
      : `
  let column = index % ${props.grid.width}u + ${Math.floor(tileWidth / 2)}u;
  let row = index / ${props.grid.width}u + ${Math.floor(tileHeight / 2)}u;
  let left = i32(column / ${tileWidth}u) - 1;
  let top = i32(row / ${tileHeight}u) - 1;
  let horizontal = f32(column % ${tileWidth}u) / ${tileWidth}.0;
  let vertical = f32(row % ${tileHeight}u) / ${tileHeight}.0;
  var weighted = 0.0; var weightSum = 0.0;
  for (var corner = 0u; corner < 4u; corner++) {
    let right = corner % 2u; let bottom = corner / 2u;
    let tileColumn = u32(clamp(left + i32(right), 0, ${tileColumns - 1}));
    let tileRow = u32(clamp(top + i32(bottom), 0, ${tileRows - 1}));
    let mapped = mapping[mappingOffset + (tileRow * ${tileColumns}u + tileColumn) * BIN_COUNT + bin];
    let weight = select(1.0 - horizontal, horizontal, right != 0u) * select(1.0 - vertical, vertical, bottom != 0u);
    if (isFiniteValue(mapped)) { weighted += mapped * weight; weightSum += weight; }
  }
  if (weightSum > 0.0) { result = weighted / weightSum; }
  `
  }
}
values[valuesOffset + index] = result;
validity[validityOffset + index] = u32(isFiniteValue(result));
if (index == 0u) { approximation[approximationOffset] = 1u; }`
      })
    );
    return nodes;
  }
}

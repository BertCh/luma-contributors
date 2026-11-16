// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {
  GraphDataView,
  GPUCommandGraph,
  GPUCommandNode,
  GPUCommandNodeProducer
} from '@luma.gl/graph/gpu-core';
import {
  createRasterTimeSeriesNodes,
  planRasterTimeSeriesChunks,
  rasterTimeSeriesBinding as rasterBinding,
  validateRasterTimeSeriesViews as validateRasterViews,
  validateRasterTimeSeriesGraph as validateRasterGraph,
  type RasterTimeSeriesScalarView as RasterScalarView,
  type RasterTimeSeriesView,
  type RasterTimeSeriesChunkPlan
} from './raster-time-series-chunks-b';
import {
  validateCapacity,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {RASTER_FINITE_WGSL} from '../remote-sensing/raster-kernel-utils';

/** Slices: lag loops run over the series inside one invocation. */
export const MAXIMUM_LAGGED_CORRELATION_SLICE_COUNT = 256;
/** Lags: each lag adds a full pass over the slices per cell. */
export const MAXIMUM_LAGGED_CORRELATION_LAG_COUNT = 64;
/** Float32 elements of the per-frame parameters: smallest usable pair count. */
export const GPU_RASTER_LAGGED_CORRELATION_PARAMETER_LENGTH = 1;

/** Per-frame settings of {@link GPURasterLaggedCorrelation}. */
export type GPURasterLaggedCorrelationSettings = {
  /** Fewest valid `(index, cell)` pairs for a correlation; defaults to 3. */
  minimumPairCount?: number;
};

/** Packs {@link GPURasterLaggedCorrelationSettings}: `[minimumPairCount]`. */
export function getGPURasterLaggedCorrelationParameterValues(
  settings: GPURasterLaggedCorrelationSettings = {},
  target: Float32Array = new Float32Array(GPU_RASTER_LAGGED_CORRELATION_PARAMETER_LENGTH)
): Float32Array {
  target[0] = settings.minimumPairCount ?? 3;
  return target;
}

/** Properties of {@link GPURasterLaggedCorrelation}. */
export type GPURasterLaggedCorrelationProps<Rows extends RasterScalarView = GraphDataView> = {
  id?: string;
  cellCount: number;
  sliceCount: number;
  /** Cell-major `cell * sliceCount + slice`; nonfinite samples are missing. */
  values: RasterTimeSeriesView<'float32', Rows>;
  /** Per-observation QA; zero rejects the sample. */
  mask?: RasterTimeSeriesView<'uint32', Rows>;
  /** The reference series, one value per slice, nonfinite where missing (for example Nino3.4). */
  referenceIndex: GraphDataView<'float32'>;
  /** Float32, see {@link getGPURasterLaggedCorrelationParameterValues}. */
  parameters: GraphDataView<'float32'>;
  /**
   * Slice lags, in `(-sliceCount, sliceCount)`. At lag `L` the correlation pairs `index[t]` with
   * `cell[t + L]`, so a positive lag means the index leads the cell. Slices are regular slots.
   */
  lags: readonly number[];
  output: {
    /** `cell * lags.length + lagIndex`: Pearson correlation on the pairwise-complete samples; NaN unless status is 0. */
    correlations: RasterTimeSeriesView<'float32', Rows>;
    /** Same layout: 0 computed, 1 fewer than `minimumPairCount` pairs, 2 zero variance. */
    status: RasterTimeSeriesView<'uint32', Rows>;
    /** Optional, same layout: valid pairs used. */
    pairCount?: RasterTimeSeriesView<'uint32', Rows>;
  };
};

/**
 * Pearson correlation of each cell's series with a supplied index at a fixed set of slice lags,
 * for teleconnection maps. Each lag uses only pairs where both the index and the cell are valid
 * (pairwise-complete); means and variances are taken over those pairs, not the whole series.
 *
 * Per-frame: `parameters`, `referenceIndex`. Compile-time: `cellCount`, `sliceCount`, `lags`,
 * which optional views exist.
 *
 * Cell-major views accept matching chunks of complete cell records; shared tables stay packed.
 */
export class GPURasterLaggedCorrelation<Rows extends RasterScalarView = GraphDataView>
  implements GPUCommandNodeProducer
{
  readonly id: string;
  readonly props: GPURasterLaggedCorrelationProps<Rows>;
  readonly chunkPlan: RasterTimeSeriesChunkPlan;
  readonly estimatedCommandNodeCount: number;

  constructor(props: GPURasterLaggedCorrelationProps<Rows> & {values: Rows}) {
    this.props = props;
    const id = props.id ?? 'raster-lagged-correlation';
    this.id = id;
    const {cellCount, sliceCount, lags} = props;
    validateCapacity(id, 'cellCount', cellCount);
    validateCapacity(id, 'sliceCount', sliceCount, 2, MAXIMUM_LAGGED_CORRELATION_SLICE_COUNT);
    validateCapacity(id, 'lags length', lags.length, 1, MAXIMUM_LAGGED_CORRELATION_LAG_COUNT);
    validateRequirement(
      id,
      lags.every(lag => Number.isInteger(lag) && Math.abs(lag) < sliceCount),
      'lags must be integers in (-sliceCount, sliceCount)'
    );
    const observations = cellCount * sliceCount;
    validateCapacity(id, 'cellCount times sliceCount', observations);
    validateRequirement(
      id,
      [props.values, props.mask].every(view => !view || view.length >= observations),
      'values and mask must hold cellCount times sliceCount entries'
    );
    validateRequirement(
      id,
      props.referenceIndex.length >= sliceCount,
      'referenceIndex must hold one entry per slice'
    );
    const lagEntries = cellCount * lags.length;
    validateRequirement(
      id,
      [props.output.correlations, props.output.status, props.output.pairCount].every(
        view => !view || view.length >= lagEntries
      ),
      'output views must hold one entry per cell and lag'
    );
    validateRequirement(
      id,
      props.parameters.length >= GPU_RASTER_LAGGED_CORRELATION_PARAMETER_LENGTH,
      `parameters must hold ${GPU_RASTER_LAGGED_CORRELATION_PARAMETER_LENGTH} entries`
    );
    this.chunkPlan = planRasterTimeSeriesChunks(this.id, props.cellCount, {
      values: {view: props.values, entriesPerCell: props.sliceCount},
      mask: {view: props.mask, entriesPerCell: props.sliceCount},
      'output.correlations': {view: props.output.correlations, entriesPerCell: props.lags.length},
      'output.status': {view: props.output.status, entriesPerCell: props.lags.length},
      'output.pairCount': {view: props.output.pairCount, entriesPerCell: props.lags.length}
    });
    this.estimatedCommandNodeCount = this.chunkPlan.estimatedCommandNodeCount;
    validateRasterViews(this.id, this.getInputs(), this.getOutputs(), this.chunkPlan);
  }

  private getInputs(): RasterScalarView[] {
    const {values, mask, referenceIndex, parameters} = this.props;
    return [values, referenceIndex, parameters, ...(mask ? [mask] : [])];
  }

  private getOutputs(): RasterScalarView[] {
    const {correlations, status, pairCount} = this.props.output;
    return [correlations, status, ...(pairCount ? [pairCount] : [])];
  }

  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {props, id} = this;
    const {sliceCount: slices, values, mask, referenceIndex, parameters, lags, output} = props;
    validateRasterGraph(id, graph, this.getInputs(), this.getOutputs());
    const lagCount = lags.length;
    return createRasterTimeSeriesNodes<Parameters>(graph, this.chunkPlan, {
      id,
      operation: 'GPURasterLaggedCorrelation',
      invocationCount: props.cellCount,
      workgroupSize: 16,
      bindings: [
        rasterBinding('values', values),
        rasterBinding('referenceIndex', referenceIndex),
        rasterBinding('parameters', parameters),
        ...(mask ? [rasterBinding('mask', mask)] : []),
        rasterBinding('correlations', output.correlations, true),
        rasterBinding('status', output.status, true),
        ...(output.pairCount ? [rasterBinding('pairCount', output.pairCount, true)] : [])
      ],
      declarations: `${RASTER_FINITE_WGSL}
const LAGS = array<i32, ${lagCount}>(${lags.map(lag => `${lag}`).join(', ')});`,
      body: `let minimumPairs = parameters[parametersOffset];
for (var lagIndex = 0u; lagIndex < ${lagCount}u; lagIndex++) {
  let lag = LAGS[lagIndex];
  var count = 0u;
  var indexTotal = 0.0;
  var cellTotal = 0.0;
  for (var slice = 0i; slice < ${slices}i; slice++) {
    let targetSlice = slice + lag;
    if (targetSlice < 0i || targetSlice >= ${slices}i) { continue; }
    let indexValue = referenceIndex[referenceIndexOffset + u32(slice)];
    let source = index * ${slices}u + u32(targetSlice);
    let cellValue = values[valuesOffset + source];
    if (!finite(indexValue) || !finite(cellValue) ${mask ? '|| mask[maskOffset + source] == 0u' : ''}) { continue; }
    count++;
    indexTotal += indexValue;
    cellTotal += cellValue;
  }
  var code = 1u;
  var correlation = missing();
  if (f32(count) >= minimumPairs) {
    let indexMean = indexTotal / f32(count);
    let cellMean = cellTotal / f32(count);
    var crossTotal = 0.0;
    var indexSquares = 0.0;
    var cellSquares = 0.0;
    for (var slice = 0i; slice < ${slices}i; slice++) {
      let targetSlice = slice + lag;
      if (targetSlice < 0i || targetSlice >= ${slices}i) { continue; }
      let indexValue = referenceIndex[referenceIndexOffset + u32(slice)];
      let source = index * ${slices}u + u32(targetSlice);
      let cellValue = values[valuesOffset + source];
      if (!finite(indexValue) || !finite(cellValue) ${mask ? '|| mask[maskOffset + source] == 0u' : ''}) { continue; }
      let indexAnomaly = indexValue - indexMean;
      let cellAnomaly = cellValue - cellMean;
      crossTotal += indexAnomaly * cellAnomaly;
      indexSquares += indexAnomaly * indexAnomaly;
      cellSquares += cellAnomaly * cellAnomaly;
    }
    // Float32 means carry rounding, so variance below it counts as zero rather than as a correlation.
    if (indexSquares > f32(count) * indexMean * indexMean * 1e-12 && cellSquares > f32(count) * cellMean * cellMean * 1e-12) {
      code = 0u;
      correlation = clamp(crossTotal / sqrt(indexSquares * cellSquares), -1.0, 1.0);
    } else {
      code = 2u;
    }
  }
  correlations[correlationsOffset + index * ${lagCount}u + lagIndex] = correlation;
  status[statusOffset + index * ${lagCount}u + lagIndex] = code;
  ${output.pairCount ? `pairCount[pairCountOffset + index * ${lagCount}u + lagIndex] = count;` : ''}
}`
    });
  }
}

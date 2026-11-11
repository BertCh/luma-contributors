// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import type {GPUCommandNodeProducer} from '@luma.gl/graph/gpu-core';
import {
  validateCapacity,
  validateGPUStorageBindingByteLength,
  validateGraphOutputsDisjointFromInputs,
  validateSinglePackedView,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {getColumnQuantileNodes, type GPUColumnQuantilesOutput} from './column-quantiles-nodes';
import {
  getGPUColumnQuantilesParameterLength,
  getGPUColumnQuantilesRationalProbabilityLength,
  GPU_COLUMN_QUANTILES_MAXIMUM_QUANTILE_COUNT
} from './column-quantiles-parameters';

export type {GPUColumnQuantilesOutput} from './column-quantiles-nodes';

const OPERATION = 'GPUColumnQuantiles';
/**
 * Largest column of the float32 rank: row counts and ranks are held in float32, which is exact
 * only up to 2^24. The percentile filter ranks are float32 on both paths.
 */
const MAXIMUM_ROW_COUNT = 2 ** 24;
/**
 * Largest column with `rationalProbabilities`: counts and ranks are `u32` there, and the runtime
 * zero that blocks multiply-add fusion (`validCount >> 31`) needs a count below 2^31. The device's
 * storage binding size is the practical limit (2^25 rows at the default 128 MiB).
 */
const MAXIMUM_RATIONAL_ROW_COUNT = 2 ** 31 - 1;

/**
 * Properties for {@link GPUColumnQuantiles}.
 *
 * Per-frame (no recompile): the contents of `parameters` (probabilities, interpolation, filter
 * range), of `rationalProbabilities` and of `values` and `mask`. Topology (needs a new graph):
 * `quantileCount`, view lengths, whether `mask` and `rationalProbabilities` are present, and which
 * optional outputs are present.
 */
export type GPUColumnQuantilesProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'column-quantiles'`. */
  id?: string;
  /**
   * Packed float32 column. NaN and +-Infinity rows are missing and skipped. At most 2^24 rows;
   * with `rationalProbabilities` and no filter output, as many rows as one storage binding holds.
   */
  values: GraphDataView<'float32'>;
  /** Optional packed `uint32` row mask; zero skips the row. */
  mask?: GraphDataView<'uint32'>;
  /**
   * Float32 parameters written with `getGPUColumnQuantilesParameterValues`:
   * `[interpolationCode, filterLower, filterUpper, 0, p_0 ... p_{quantileCount-1}]`.
   */
  parameters: GraphDataView<'float32'>;
  /**
   * Optional probabilities as exact ratios, written with
   * `getGPUColumnQuantilesRationalProbabilityValues`: `2 * quantileCount` uint32 words
   * `[numerator_0, denominator_0, ...]`, `p_k = numerator_k / denominator_k`. When present they
   * replace the float32 probabilities of `parameters` (which are then ignored; the interpolation
   * and the filter range are still read from `parameters`), and the rank
   * `floor((n - 1) numerator / denominator)` and the interpolation fraction are computed with
   * 64-bit integers: exact at any row count. A zero denominator or `numerator > denominator`
   * gives NaN. Omitted (the default), the rank is the float32 product described on the class.
   */
  rationalProbabilities?: GraphDataView<'uint32'>;
  /** Number of probability slots, compile-time, 1 to 64. */
  quantileCount: number;
  /** Caller-owned outputs. */
  output: GPUColumnQuantilesOutput;
};

/**
 * Exact quantiles of a float32 column by GPU radix selection, plus a per-frame percentile-range
 * filter mask, with no readback and no sort.
 *
 * With `n` valid rows (unmasked and finite: NaN and +-Infinity are missing, as in every
 * interpolating kernel; `-0` equals `+0`) sorted ascending as `x[0..n-1]` and probability `p`,
 * `h = fround(fround(n - 1) * p)`, `lo = floor(h)`, `hi = min(lo + 1, n - 1)`:
 * `lower` is `x[lo]`, `higher` is `x[ceil(h)]`, `nearest` is `x[round-half-to-even(h)]` (numpy),
 * `linear` is `x[lo] + (x[hi] - x[lo]) * (h - lo)` (numpy `linear`, d3 `quantile`, R-7), and
 * `midpoint` is `(x[lo] + x[ceil(h)]) / 2`. When the two order statistics compare equal the first
 * is returned unchanged. A probability that is NaN or outside `[0, 1]`, or `n == 0`, gives NaN.
 * This is `numpy.quantile(x[numpy.isfinite(x)], p, method=...)`; numpy keeps infinities and
 * raises on an invalid `p`. `GPUClassBreaks` and the `GPUGroupStatistics` medians and percentiles
 * use the same policy and the same `linear` arithmetic. When `x[hi] - x[lo]` overflows f32, the
 * convex form `x[lo] * (1 - t) + x[hi] * t` keeps the result finite.
 *
 * The percentile filter follows deck.gl `lowerPercentile` / `upperPercentile` semantics with
 * fractions `fl`, `fu` in `[0, 1]` (NaN counts as 0 and 1): `lowerIndex = clamp(floor(n * fl), 0,
 * n - 1)` and `upperIndex = clamp(ceil(n * fu) - 1, 0, n - 1)`, with the products rounded to f32.
 * `filterBounds` is `[x[lowerIndex], x[upperIndex]]` and `outputMask[row]` is 1 when the row is
 * valid and `lowerBound <= value <= upperBound` in ordered-key order. With `n == 0` the bounds are
 * NaN and the mask is all 0.
 *
 * Algorithm: four 8-bit digit passes over order-preserving u32 keys, most significant first. Each
 * pass builds histograms with integer atomics (workgroup-local, flushed to global) per distinct
 * active prefix and one single-workgroup kernel narrows every target rank to a bucket. Targets are
 * the needed order statistics (two per probability plus two for the filter). The result key is
 * exact, so every output, including `linear` and `midpoint` (each f32 step is rounded separately,
 * never fused into a multiply-add), is bitwise equal to a sort-based `Math.fround` oracle and
 * independent of GPU thread order.
 *
 * Which rank is exact. There are two paths, chosen by whether `rationalProbabilities` is given:
 * - Default, float32 probability (`parameters`): the rank `h` above is a float32 product. It is
 *   not exact. Its fraction has a step of `2^(e - 23)` for `h` in `[2^e, 2^(e + 1))` and is lost
 *   from `h = 2^23` (8,388,608): `linear` then returns `x[lo]` where numpy, with `h` in float64,
 *   interpolates (16,777,216 rows at `p = 0.95`: fraction 0 here, 0.25 in numpy); the error is at
 *   most that step times `x[hi] - x[lo]`. At any row count the float32 `p` is not the decimal the
 *   caller wrote, so `floor(h)` can be one rank away from numpy's (101 rows at `p = 0.59`: rank
 *   58 here, 59 in numpy; 40 of 6,912 combinations of 27 row counts and 256 probabilities in the
 *   committed fixture).
 * - `rationalProbabilities`, `p = numerator / denominator`: `lo = floor((n - 1) numerator /
 *   denominator)` and the remainder come from an exact 64-bit integer product and division, so
 *   the ranks of all five methods are exact for every row count, and the `linear` fraction
 *   `remainder / denominator` is the correctly rounded float32 of the exact ratio. The value
 *   arithmetic (`x[lo] + (x[hi] - x[lo]) * fraction`) is still float32. This is
 *   `numpy.quantile(x, numerator / denominator)` evaluated without rounding the rank: it agrees
 *   with numpy 2.5.4 for `linear` everywhere tested (to float32 rounding of the value), and for
 *   the other four methods except where numpy's own float64 product `(n - 1) * q` lands just
 *   below or just above an integer that the exact product equals. Just below, `lower` and
 *   `midpoint` are one rank off (101 rows, `q = 0.29`: the product is 28.999999999999996, numpy
 *   `lower` returns `x[28]`, the exact rank is 29); just above, `higher` and `midpoint` are
 *   (101 rows, `q = 0.55`: 55.00000000000001, numpy `higher` returns `x[56]`, the exact rank is
 *   55). 22 of 34,560 cases in the fixture, 6 below and 16 above, all at 101 rows. Checked against numpy and
 *   exact rational arithmetic up to 2^25 rows on the GPU, and the rank arithmetic alone up to a
 *   count of 2^32 - 1 (`test/oracles/dataframe-quantile-rational-rank.py`).
 *
 * Limits: 64 probabilities. At most 2^24 rows on the default path (row counts and ranks use f32
 * arithmetic). With `rationalProbabilities` the column may be longer, up to one storage binding
 * of the device (2^25 rows at the default 128 MiB; never more than 2^31 - 1), provided no filter
 * output (`outputMask`, `filterBounds`) is requested: the percentile filter ranks are float32 on
 * both paths.
 * Inputs must be single packed views (chunked vectors are rejected).
 */
export class GPUColumnQuantiles implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUColumnQuantilesProps;

  constructor(props: GPUColumnQuantilesProps) {
    this.id = props.id ?? 'column-quantiles';
    this.props = props;
    const id = this.id;
    const {quantileCount, output} = props;
    validateCapacity(
      id,
      'quantileCount',
      quantileCount,
      1,
      GPU_COLUMN_QUANTILES_MAXIMUM_QUANTILE_COUNT
    );
    const rows = props.values.length;
    if (rows < 1) {
      throw new Error(`${id} needs at least one row`);
    }
    if (props.rationalProbabilities) {
      validateSinglePackedView(
        props.rationalProbabilities,
        ['uint32'],
        `${id} rationalProbabilities`
      );
      const rationalLength = getGPUColumnQuantilesRationalProbabilityLength(quantileCount);
      validateRequirement(
        id,
        props.rationalProbabilities.length >= rationalLength,
        `rationalProbabilities must hold ${rationalLength} uint32 values (a numerator and a denominator per quantile)`
      );
      validateRequirement(
        id,
        rows <= MAXIMUM_RATIONAL_ROW_COUNT,
        `values length must be at most ${MAXIMUM_RATIONAL_ROW_COUNT} (2^31 - 1) with rationalProbabilities: row counts are uint32 and the kernel needs them below 2^31`
      );
      validateRequirement(
        id,
        rows <= MAXIMUM_ROW_COUNT || !(output.outputMask || output.filterBounds),
        `values length must be at most ${MAXIMUM_ROW_COUNT} (2^24) when output.outputMask or output.filterBounds is requested: the percentile filter ranks are held in float32`
      );
    } else {
      validateRequirement(
        id,
        rows <= MAXIMUM_ROW_COUNT,
        `values length must be at most ${MAXIMUM_ROW_COUNT} (2^24): row counts and ranks are held in float32 and are exact only up to 2^24`
      );
    }
    validateSinglePackedView(props.values, ['float32'], `${id} values`);
    if (props.mask) {
      validateSinglePackedView(props.mask, ['uint32'], `${id} mask`);
      if (props.mask.length !== rows) {
        throw new Error(`${id} mask length must equal values length`);
      }
    }
    validateSinglePackedView(props.parameters, ['float32'], `${id} parameters`);
    const parameterLength = getGPUColumnQuantilesParameterLength(quantileCount);
    if (props.parameters.length < parameterLength) {
      throw new Error(`${id} parameters must hold ${parameterLength} float32 values`);
    }
    validateSinglePackedView(output.quantiles, ['float32'], `${id} output.quantiles`);
    if (output.quantiles.length < quantileCount) {
      throw new Error(`${id} output.quantiles must hold quantileCount rows`);
    }
    validateSinglePackedView(output.validCount, ['uint32'], `${id} output.validCount`);
    if (output.validCount.length < 1) {
      throw new Error(`${id} output.validCount must hold one row`);
    }
    if (output.outputMask) {
      validateSinglePackedView(output.outputMask, ['uint32'], `${id} output.outputMask`);
      if (output.outputMask.length < rows) {
        throw new Error(`${id} output.outputMask must hold values.length rows`);
      }
    }
    if (output.filterBounds) {
      validateSinglePackedView(output.filterBounds, ['float32'], `${id} output.filterBounds`);
      if (output.filterBounds.length < 2) {
        throw new Error(`${id} output.filterBounds must hold two rows`);
      }
    }
    validateGraphOutputsDisjointFromInputs(
      id,
      [output.quantiles, output.validCount, output.outputMask, output.filterBounds],
      [props.values, props.mask, props.parameters, props.rationalProbabilities]
    );
  }

  /**
   * Returns init, four histogram and select pairs, finish, and optionally filter-mask nodes.
   * Node IDs are `${id}-init`, `${id}-histogram-0..3`, `${id}-select-0..3`, `${id}-finish`, and
   * `${id}-filter-mask`.
   */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {props, id} = this;
    if (props.values.length > MAXIMUM_ROW_COUNT) {
      // Only reachable with rationalProbabilities; shorter columns fit the core binding limit.
      validateGPUStorageBindingByteLength(
        graph.device,
        id,
        'values',
        props.values.length * Float32Array.BYTES_PER_ELEMENT,
        'use a shorter column or a device with a larger maxStorageBufferBindingSize'
      );
    }
    return getColumnQuantileNodes(graph, {
      id,
      operation: OPERATION,
      values: props.values,
      mask: props.mask,
      parameters: props.parameters,
      rationalProbabilities: props.rationalProbabilities,
      quantileCount: props.quantileCount,
      output: props.output
    });
  }
}

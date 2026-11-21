// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createClassificationKernelNode as createWGSLKernelNode} from './classification-kernel-node';
import {
  DEFAULT_NATURAL_BREAKS_BIN_COUNT,
  getClassBreaksProvenance,
  resolveClassBreaksMethods,
  validateClassBreaksNaturalOptions
} from './class-breaks-input';
import {Buffer} from '@luma.gl/core';
import {
  createTransientView,
  getBoundedDispatchLayout,
  getViewBindingRange,
  GPUSort,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import type {GPUCommandNodeProducer} from '@luma.gl/graph/gpu-core';
import type {GPUAnalysisRoute} from '../../analysis-core/contracts/analysis-execution-planning';
import type {WGSLKernelBinding} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  getGraphViewChunks,
  isGraphVectorView,
  validateCapacity,
  validateGPUStorageBindingByteLength,
  validateGraphOutputsDisjointFromInputs,
  validateSinglePackedView,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  CLASS_BREAKS_EXTREMES,
  CLASS_BREAKS_HEAD_TAIL,
  CLASS_BREAKS_STATE,
  CLASS_BREAKS_STATE_LENGTH,
  getClassBreaksIndirectGate,
  getClassBreaksMomentNodes,
  getClassBreaksMomentTileCount,
  MOMENT_STRIDE,
  MOMENT_WORKGROUP_SIZE
} from './class-breaks-kernels';
import {
  getGPUClassBreaksParameterLength,
  GPU_CLASS_BREAKS_BOX_PLOT_CLASS_COUNT,
  GPU_CLASS_BREAKS_METHOD_CODES,
  GPU_CLASS_BREAKS_PARAMETER_HEADER_LENGTH,
  type GPUClassBreaksMethod
} from './class-breaks-parameters';
import {COLUMN_ORDERED_KEY_WGSL} from './column-classification-shared';
import {getColumnQuantileNodes} from './column-quantiles-nodes';
import {getNaturalBreaksNodes, getNaturalBreaksDispatchWordCount} from './natural-breaks-nodes';
import {getExactNaturalBreaksNodes} from './exact-natural-breaks-nodes';
import {getMonotoneNaturalBreaksNodes} from './monotone-natural-breaks-nodes';

import {
  type ClassificationRows,
  validateClassificationOutputs,
  validateClassificationScratch,
  validateClassificationBindings,
  validateClassificationRows
} from './column-classification-rows';

import {snapshotClassBreaksProps} from './class-breaks-operation-snapshot';
import {
  CLASS_BREAKS_LIMITS,
  getClassBreaksExecutionPlan,
  type GPUClassBreaksShape
} from './class-breaks-planning';
export type {GPUClassBreaksShape, GPUClassBreaksDeviceLimits} from './class-breaks-planning';

const OPERATION = 'GPUClassBreaks';
/** Largest supported class count; the quantile pass refines at most 64 targets. */
const MAXIMUM_CLASS_COUNT = 64;
/** Largest row count: moment counts are f32 and quantile ranks are exact up to 2^24. */
const MAXIMUM_ROW_COUNT = 2 ** 24;
const MAXIMUM_NATURAL_BREAKS_BIN_COUNT = 2048;
/** Quantile parameter code for linear (R-7) interpolation, as numpy and mapclassify use. */
const LINEAR_INTERPOLATION_CODE = 3;
const INVALID_METHOD = 0xffffffff;
const CODES = GPU_CLASS_BREAKS_METHOD_CODES;

/** Caller-owned outputs of {@link GPUClassBreaks}. */
export type GPUClassBreaksOutput = {
  /**
   * Class edges `e[0..k]` (`maximumClassCount + 1` rows): `e[0]` is the smallest and `e[k]` the
   * largest finite value, inner edges ascend. Rows past `k` are NaN. Rows `e[1..k-1]` feed
   * `GPUColorScale` or any threshold scale directly as a domain.
   */
  breaks: GraphDataView<'float32'>;
  /** One row receiving the produced class count `k` (0 when no finite value is selected). */
  classCount: GraphDataView<'uint32'>;
  /**
   * Optional rows per class (`maximumClassCount` rows), counted with the stream's class rule: a
   * value is in class `i` when `i` inner edges have an ordered key at or below its key; values
   * outside `[e[0], e[k]]` (such as infinities) clamp to the first or last class.
   */
  classCounts?: GraphDataView<'uint32'>;
};

/**
 * Properties for {@link GPUClassBreaks}.
 *
 * Per-frame (no recompile): the contents of `parameters` (method, class count, method settings,
 * custom edges), `values`, and `mask`. Topology (needs a new graph): `maximumClassCount`, `methods`,
 * `naturalBreaksBinCount`, `naturalBreaksMode`, view lengths, and whether `mask` and `output.classCounts` are present.
 */
export type GPUClassBreaksProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'class-breaks'`. */
  id?: string;
  /**
   * Packed or chunked float32 column. Its logical view length is the live row count: use a
   * grid-sized prefix when a raster allocation has spare rows. An explicit full view includes
   * every row. NaN rows are skipped; infinities count only toward the end classes.
   * A `uint32` or `sint32` column (for example point counts) is also accepted: one internal pass
   * converts it to f32 (exact up to 2^24 in magnitude, rounded to nearest beyond) before the
   * classification, which then behaves as for the converted float column. Topology: the format.
   */
  values: ClassificationRows<'float32' | 'uint32' | 'sint32'>;
  /** Optional row-aligned packed or chunked `uint32` mask; zero skips the row. */
  mask?: ClassificationRows<'uint32'>;
  /**
   * Float32 parameter view of `getGPUClassBreaksParameterLength(maximumClassCount)` elements
   * written with `getGPUClassBreaksParameterValues`.
   */
  parameters: GraphDataView<'float32'>;
  /** Compile-time class capacity, 1 to 64 (at least 6 when `'box-plot'` is compiled). */
  maximumClassCount: number;
  /**
   * Methods compiled into the graph; the per-frame method code chooses among them and every other
   * method's kernels return immediately. Defaults to every method compatible with the class capacity and input layout. `'maximum-breaks'` adds two
   * full radix sorts that run every frame whatever the selected method, so leave it out unless it
   * is needed.
   */
  methods?: readonly GPUClassBreaksMethod[];
  /**
   * Equal-width bins of the coarse natural-breaks histogram, 2 to 2048. Default 2048. More bins
   * resolve finer structure for the coarse dynamic program; the breaks themselves are refined on
   * the data, so the bin count only decides where that refinement starts. It also caps the
   * class count: the coarse program cannot place more classes than bins, so in approximate mode
   * (the default) with `'natural-breaks'` compiled, `maximumClassCount` must not exceed it
   * (rejected at construction). Exact and monotone modes do not bin and have no such cap.
   */
  naturalBreaksBinCount?: number;
  /**
   * Default approximate histogram/refinement; exact exhaustively optimizes the normalized f32
   * recurrence (up to 8192 total packed rows when natural-breaks is enabled). Monotone uses
   * full source-value boundaries with 20-bit positions and divide-and-conquer pruning
   * (up to 262144 total packed rows when enabled). Quantization and rounded score
   * comparisons mean monotone is an approximation, not an exact-mode substitute.
   */
  naturalBreaksMode?: 'approximate' | 'exact' | 'monotone';
  /**
   * Exact mode only. Auto materializes interval costs through 1024 rows, otherwise recomputes
   * them per DP layer. Recompute removes quadratic storage but retains O(k n²) work. Explicit
   * With natural-breaks enabled, materialized storage supports at most 1024 total packed rows;
   * recompute supports at most 8192. These route limits do not apply when natural-breaks is disabled.
   */
  naturalBreaksExactStorage?: 'auto' | 'materialized' | 'recompute';
  /** Caller-owned outputs. */
  output: GPUClassBreaksOutput;
};

/** First word of the head/tail break values in the head/tail buffer, after the state words. */
const HEAD_TAIL_BREAKS_OFFSET = 4;

/**
 * Computes choropleth class breaks of one column on the GPU every frame, with no readback: equal
 * interval, quantile, standard deviation, head/tail, box plot, maximum breaks, natural breaks
 * (Fisher-Jenks), and custom edges.
 *
 * Methods (`n` unmasked non-NaN rows, `min`/`max` over the finite ones, `k` classes):
 * - `'equal-interval'`: `e[i] = min + ((max - min) / k) * i` in f32. When `max - min` is past the
 *   float32 range, `e[i] = 2 * (min / 2 + ((max / 2 - min / 2) / k) * i)`, so the cuts stay inside.
 * - `'quantile'`: `e[i]` is the linear (R-7, numpy and d3 default) quantile of the finite values at
 *   `fround(i / k)`, computed exactly by the `GPUColumnQuantiles` radix select and its
 *   order-statistic policy. Ties may repeat edges (empty classes).
 * - `'standard-deviation'`: inner edges at `mean + (j - (k - 2) / 2) * interval * sd` for
 *   `j = 0..k-2`, centred on the mean, with the sample standard deviation `sqrt(M2 / (n - 1))`
 *   (mapclassify `StdMean`, R classInt `'sd'`; `M2` from a fixed-order Chan merge). With
 *   `classCount` k this is mapclassify `StdMean(multiples=[(j - (k - 2) / 2) * interval])`.
 * - `'head-tail'`: Jiang's head/tail breaks. Splits at the mean of the current head, then keeps
 *   the values above it, while the next head holds at most `headTailRatio` of the current rows, has
 *   more than one row, and fewer than `k` classes exist. Produces a data-dependent class count.
 * - `'box-plot'`: six classes with edges `[min, q1 - h * iqr, q1, median, q3, q3 + h * iqr, max]`.
 * - `'maximum-breaks'`: sorts the column, takes the `k - 1` largest gaps between consecutive
 *   distinct values (ties to the lower position), and breaks at their float32 midpoints, using the
 *   upper observation if rounding reaches the lower one. Fewer classes when there are fewer distinct values.
 * - `'natural-breaks'`: a histogram approximation to Fisher-Jenks (minimum within-class squared
 *   deviation), followed by local refinement; it does not guarantee the global optimum. A
 *   monotone divide-and-conquer dynamic program over `naturalBreaksBinCount` equal-width bins of
 *   `[min, max]` (O(k B log B), exact 64-bit integer prefix sums, no cost table) gives the coarse
 *   partition. Each coarse range and refinement break chooses scaled
 *   coordinates for its own magnitude, converting saved widths and movement between scales; published edges retain input units. Red-black
 *   coordinate descent on the rows then moves every break onto the data:
 *   each break takes the best split of the values in its two adjacent classes, in windows that
 *   shrink around the cut until a cell is below an f32 ulp. Every inner edge is therefore an
 *   actual data value, the smallest value of its class (mapclassify and jenkspy return the
 *   largest value of the class below it). A class count above the number of distinct values yields one class per distinct value. The refinement is a local
 *   search around the coarse partition: it equals the exact optimum (jenkspy, mapclassify) on 32
 *   of the 34 reference cases. The other two require moving adjacent cuts jointly: no single-cut
 *   move improves their cost, even with an unlimited refinement window. Their observed goodness
 *   of variance fit gaps are 3.44e-8 and 6.95e-13; these are fixture measurements, not an error bound.
 *   When an extreme outlier collapses the bulk of the data into one bin, the coarse stage runs again over the range of that class. Cost: the
 *   extents pass plus two passes over the rows per round, at most eight rounds, usually two.
 * - `'custom'`: the edges from the parameter view, used as given.
 *
 * Kind of edge per method. It decides which class a value equal to an inner edge belongs to when
 * the edges are applied to data (`GPUClassAssignment` `intervalClosure`); `classCounts` here always
 * counts with the lower-edge rule (`'left'`).
 * - Lower edges, classes `[e[j], e[j + 1])`, `intervalClosure: 'left'`: `'natural-breaks'` (each
 *   inner edge is the smallest value of its class; `'right'` would put that value one class low).
 * - The same numbers as mapclassify's upper-bound `bins`, classes `(e[j], e[j + 1]]` there,
 *   `intervalClosure: 'right'` to reproduce mapclassify's or giddy's classes: `'quantile'`,
 *   `'equal-interval'`, `'standard-deviation'`, `'head-tail'`, `'box-plot'`. Used as a d3
 *   threshold or quantile domain the same numbers are lower edges (`'left'`).
 * - `'maximum-breaks'`: float32 midpoints between consecutive distinct values. If a midpoint
 *   rounds to the lower observation, the edge uses the upper observation so the lower-edge
 *   rule (`'left'`) separates them. Closure matters when an edge equals an observation.
 * - `'custom'`: whichever the caller's edges are.
 *
 * Every computed inner edge is clamped to `[min, max]`. With no finite values, or when any edge is
 * NaN (only a hand-written parameter view with a NaN probability, hinge, interval or custom edge can
 * cause that), the class count is 0 and every edge is NaN, so every row is unclassified; when every finite value is equal the class count is 1 (except `'custom'`).
 * Signed zeros are numerically equal for membership and maximum-breaks distinctness: both
 * enter the upper class at a zero inner edge. Raw keys still preserve bits for extrema.
 * Infinite values are missing for every edge computation (moments, extremes, quantiles, gaps,
 * histogram), so `e[0]` and `e[k]` bound every inner edge; they still classify into the end
 * classes in `classCounts`.
 *
 * Chunked input: values and mask may be GraphVectorView with matching ordered chunk lengths.
 * Compile `methods` without maximum-breaks; exact and monotone natural-breaks remain packed-only
 * because they sort all rows. The default method set omits these packed-only methods for vectors. Approximate natural-breaks histograms/refinement, extrema, quantiles and
 * class counts accumulate all chunks before consuming each reduction, bitwise equal to packed
 * input. Standard-deviation and head-tail materialize packed GPU scratch to preserve the exact
 * global moment tile/tree order across arbitrary chunk boundaries. Integer input also uses the
 * existing packed f32 conversion. Total row limits are unchanged; controls/outputs remain packed.
 * Chunk topology is compile-time; buffer contents remain per-frame. Construction rejects an
 * estimated graph above 32,768 command nodes in total. The approximate natural-break route
 * includes three coordinate/cut stages: with only that method, 18 classes and 660 nonempty
 * chunks estimate 32,771 nodes and are rejected (659 chunks estimate 32,725).
 *
 * Exact mode uses normalized f32 interval costs and exhaustive DP; see the API reference for
 * arithmetic and tie semantics. It is globally optimal for that recurrence, not f64 arithmetic.
 *
 * Determinism (approximate mode): extremes, quantiles, counts, histograms, and sorts are integer-exact. Moments are
 * reduced in a fixed tree order and the natural-breaks sums are 64-bit integers, so every output
 * is bitwise identical across runs. Edges that come out of f32 arithmetic (equal interval,
 * standard deviation, box-plot whiskers) may differ from a CPU f32 evaluation by an ulp where a
 * driver fuses a multiply-add.
 */
export class GPUClassBreaks implements GPUCommandNodeProducer {
  /** Conditional domain and binding limits; inspect a shape and device with getExecutionPlan. */
  static readonly limits: readonly GPUAnalysisRoute[] = CLASS_BREAKS_LIMITS;
  /** Read-only route admission with the same row, method, command and scratch checks as construction. */
  static readonly getExecutionPlan = getClassBreaksExecutionPlan;
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUClassBreaksProps;
  /** Compiled methods. */
  readonly methods: readonly GPUClassBreaksMethod[];
  /** Resolved natural-breaks mode and GPU algorithm; per-frame method still comes from parameters. */
  readonly provenance: ReturnType<typeof getClassBreaksProvenance>;
  /** Number of equal-width bins of the coarse natural-breaks histogram. */
  readonly naturalBreaksBinCount: number;
  /** Number of quantile targets refined per frame. */
  readonly quantileCount: number;
  /** Conservative upper bound including chunk passes, refinement and published sort nodes. */
  readonly estimatedCommandNodeCount: number;

  constructor(inputProps: GPUClassBreaksProps) {
    const props = snapshotClassBreaksProps(inputProps);
    this.id = props.id ?? 'class-breaks';
    this.props = props;
    const id = this.id;
    const {maximumClassCount, output} = props;
    validateCapacity(id, 'maximumClassCount', maximumClassCount, 1, MAXIMUM_CLASS_COUNT);
    this.methods = resolveClassBreaksMethods(id, props, isGraphVectorView(props.values));
    this.provenance = getClassBreaksProvenance({...props, methods: this.methods}, 'gpu');
    this.naturalBreaksBinCount = props.naturalBreaksBinCount ?? DEFAULT_NATURAL_BREAKS_BIN_COUNT;
    validateCapacity(
      id,
      'naturalBreaksBinCount',
      this.naturalBreaksBinCount,
      2,
      MAXIMUM_NATURAL_BREAKS_BIN_COUNT
    );
    // The approximate coarse program cannot place more classes than bins; its kernels clamp the
    // per-frame class count to the bin count, which would silently return fewer classes.
    validateRequirement(
      id,
      !this.methods.includes('natural-breaks') ||
        (props.naturalBreaksMode ?? 'approximate') !== 'approximate' ||
        maximumClassCount <= this.naturalBreaksBinCount,
      `maximumClassCount ${maximumClassCount} must not exceed naturalBreaksBinCount ${this.naturalBreaksBinCount} in approximate natural-breaks mode`
    );
    this.quantileCount = Math.max(maximumClassCount - 1, this.methods.includes('box-plot') ? 3 : 1);
    // Full-row sorts retain their packed-only contract, including when compiled but inactive.
    if (
      this.methods.includes('maximum-breaks') ||
      (this.methods.includes('natural-breaks') &&
        props.naturalBreaksMode &&
        props.naturalBreaksMode !== 'approximate')
    ) {
      validateSinglePackedView(
        props.values,
        ['float32', 'uint32', 'sint32'],
        `${id} values (compiled sort methods)`
      );
    }
    validateClassificationRows(id, 'values', props.values, ['float32', 'uint32', 'sint32']);
    const rowCount = props.values.length;
    validateClassBreaksNaturalOptions(id, props);
    validateRequirement(
      id,
      rowCount >= 1,
      `values length received ${rowCount}; skip GPU class breaks for empty input or use CPUClassBreaksOperation`
    );
    validateRequirement(
      id,
      rowCount <= MAXIMUM_ROW_COUNT,
      `values length must be in [1, ${MAXIMUM_ROW_COUNT}] (2^24): moment counts and quantile ranks are held in float32 and are exact only up to 2^24`
    );
    if (props.mask) {
      validateClassificationRows(id, 'mask', props.mask, ['uint32'], props.values);
      if (props.mask.length !== rowCount) {
        throw new Error(`${id} mask length must equal values length`);
      }
    }
    validateSinglePackedView(props.parameters, ['float32'], `${id} parameters`);
    const parameterLength = getGPUClassBreaksParameterLength(maximumClassCount);
    if (props.parameters.length < parameterLength) {
      throw new Error(`${id} parameters must hold ${parameterLength} float32 values`);
    }
    validateSinglePackedView(output.breaks, ['float32'], `${id} output.breaks`);
    if (output.breaks.length < maximumClassCount + 1) {
      throw new Error(`${id} output.breaks must hold maximumClassCount + 1 rows`);
    }
    validateSinglePackedView(output.classCount, ['uint32'], `${id} output.classCount`);
    if (output.classCount.length < 1) {
      throw new Error(`${id} output.classCount must hold one row`);
    }
    if (output.classCounts) {
      validateSinglePackedView(output.classCounts, ['uint32'], `${id} output.classCounts`);
      if (output.classCounts.length < maximumClassCount) {
        throw new Error(`${id} output.classCounts must hold maximumClassCount rows`);
      }
    }
    validateClassificationOutputs(id, [output.breaks, output.classCount, output.classCounts]);
    validateGraphOutputsDisjointFromInputs(
      id,
      [output.breaks, output.classCount, output.classCounts],
      [props.values, props.mask, props.parameters]
    );
    const plan = getClassBreaksExecutionPlan(this.getExecutionShape());
    this.validateExecutionPlan(plan);
    this.estimatedCommandNodeCount = plan.estimatedCommandNodeCount;
  }

  /** Physical binding sizes include the alignment prefix and backing allocation. */
  private getExecutionShape(): GPUClassBreaksShape {
    const {props} = this;
    return {
      ...props,
      rowCount: props.values.length,
      valuesFormat: props.values.format,
      methods: this.methods,
      mask: Boolean(props.mask),
      chunkRowCounts: isGraphVectorView(props.values)
        ? getGraphViewChunks(props.values).map(chunk => chunk.length)
        : undefined,
      bindings: Object.entries({
        values: props.values,
        mask: props.mask,
        parameters: props.parameters,
        'output.breaks': props.output.breaks,
        'output.classCount': props.output.classCount,
        'output.classCounts': props.output.classCounts
      }).flatMap(([name, view]) =>
        view
          ? getGraphViewChunks(view).map((chunk, index) => ({
              name: name === 'values' || name === 'mask' ? `${name}.${index}` : name,
              byteLength: getViewBindingRange(chunk).size,
              bufferByteLength: chunk.buffer.byteLength
            }))
          : []
      )
    };
  }

  private validateExecutionPlan(plan: ReturnType<typeof getClassBreaksExecutionPlan>): void {
    for (const constraint of plan.constraints)
      validateRequirement(
        this.id,
        constraint.satisfied,
        `${constraint.name} ${constraint.quantity} received ${constraint.received}; requires [${constraint.minimum ?? 0}, ${constraint.maximum ?? 'unbounded'}] when ${constraint.appliesWhen}; ${constraint.alternative ? `use ${constraint.alternative}` : constraint.accuracyImplications}`
      );
  }

  /** Returns prepare, extremes, per-method, finish, publish, and class-count nodes in order. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, methods, quantileCount} = this;
    const {parameters, output, maximumClassCount} = props;
    const nodes: GPUCommandNode<Parameters>[] = [];

    const plan = getClassBreaksExecutionPlan(this.getExecutionShape(), graph.device.limits);
    if (
      methods.includes('natural-breaks') &&
      (props.naturalBreaksMode ?? 'approximate') === 'approximate'
    ) {
      validateGPUStorageBindingByteLength(
        graph.device,
        id,
        'natural-breaks refinement dispatch scratch',
        getNaturalBreaksDispatchWordCount(getGraphViewChunks(props.values).length) * 4,
        'use fewer source chunks'
      );
    }
    const needsMoments = methods.includes('standard-deviation') || methods.includes('head-tail');
    validateClassificationBindings(id, graph, {
      values: props.values,
      mask: props.mask,
      parameters,
      'output.breaks': output.breaks,
      'output.classCount': output.classCount,
      'output.classCounts': output.classCounts
    });
    validateClassificationScratch(graph, id, plan.scratchWordCounts);
    this.validateExecutionPlan(plan);
    const rowCount = props.values.length;
    const integerFormat = props.values.format === 'float32' ? undefined : props.values.format;
    // Moment tiles must preserve the packed tree's global row positions, including chunks that
    // split a tile. Materialize only for those methods (and the existing integer conversion).
    const materialize = (
      name: string,
      source: ClassificationRows<'float32' | 'uint32' | 'sint32'>,
      format: 'float32' | 'uint32'
    ) => {
      if (!isGraphVectorView(source) && source.format === format)
        return source as GraphDataView<typeof format>;
      const packed = createTransientView(graph, `${id}-${name}`, format, source.length);
      let rowOffset = 0;
      for (const [chunkIndex, chunk] of getGraphViewChunks(source).entries()) {
        if (chunk.length > 0) {
          const destination = graph.createDataView(packed.buffer, {
            format,
            length: chunk.length,
            byteOffset: rowOffset * 4
          });
          nodes.push(
            createWGSLKernelNode<Parameters>(graph, {
              id: `${id}-${name}-${chunkIndex}`,
              operation: OPERATION,
              variant: 'pack-convert',
              invocationCount: chunk.length,
              bindings: [
                {
                  name: 'source',
                  view: chunk,
                  type:
                    chunk.format === 'float32' ? 'f32' : chunk.format === 'uint32' ? 'u32' : 'i32',
                  access: 'read'
                },
                {
                  name: 'destination',
                  view: destination,
                  type: format === 'float32' ? 'f32' : 'u32',
                  access: 'read_write'
                }
              ],
              body: `destination[destinationOffset + index] = ${format === 'float32' ? 'f32' : 'u32'}(source[sourceOffset + index]);`
            })
          );
        }
        rowOffset += chunk.length;
      }
      return packed;
    };
    const packedValues =
      needsMoments || integerFormat
        ? (materialize('float-values', props.values, 'float32') as GraphDataView<'float32'>)
        : undefined;
    const packedMask =
      props.mask && (needsMoments || integerFormat)
        ? (materialize('packed-mask', props.mask, 'uint32') as GraphDataView<'uint32'>)
        : undefined;
    const values = integerFormat ? packedValues! : (props.values as ClassificationRows<'float32'>);
    const mask = integerFormat ? packedMask : props.mask;
    const valueChunks = getGraphViewChunks(values);
    const maskChunks = mask ? getGraphViewChunks(mask) : undefined;
    const has = (method: GPUClassBreaksMethod) => methods.includes(method);
    const usesQuantiles = has('quantile') || has('box-plot');
    const usesStandardDeviation = has('standard-deviation');
    const usesHeadTail = has('head-tail');
    const headTailRounds = maximumClassCount - 1;
    const u32 = (name: string, length: number) =>
      createTransientView(graph, `${id}-${name}`, 'uint32', length);
    const f32 = (name: string, length: number) =>
      createTransientView(graph, `${id}-${name}`, 'float32', length);
    const read = (name: string, view: GraphDataView, type: 'u32' | 'f32'): WGSLKernelBinding => ({
      name,
      view,
      type,
      access: 'read'
    });
    const write = (
      name: string,
      view: GraphDataView,
      type: 'u32' | 'f32' | 'atomic<u32>' = 'u32'
    ): WGSLKernelBinding => ({name, view, type, access: 'read_write'});
    const maskTest = mask ? 'rowMask[rowMaskOffset + index] != 0u' : 'true';

    const state = u32('state', CLASS_BREAKS_STATE_LENGTH);
    const extremes = u32('extremes', 4);
    const edges = f32('edges', maximumClassCount + 1);
    const edgeState = u32('edge-state', 1);
    const quantileGate = usesQuantiles ? u32('quantile-gate', 1) : undefined;
    const quantileParameters = usesQuantiles
      ? f32('quantile-parameters', 4 + quantileCount)
      : undefined;
    const quantileValues = usesQuantiles ? f32('quantile-values', quantileCount) : undefined;
    const usesMoments = usesStandardDeviation || usesHeadTail;
    // Slot 0 holds the current head/tail round (consumed by its split kernel before the next round
    // overwrites it); the last slot holds the standard-deviation moments.
    const momentSlotCount = (usesHeadTail ? 1 : 0) + 1;
    const moments = usesMoments ? f32('moments', momentSlotCount * MOMENT_STRIDE) : undefined;
    const partials = usesMoments
      ? f32('moment-partials', getClassBreaksMomentTileCount(rowCount) * MOMENT_STRIDE)
      : undefined;
    // Head/tail state words, then the break values as f32 bit patterns. One buffer keeps the
    // finish kernel within the default limit of 8 storage buffers per stage.
    const headTail = usesHeadTail
      ? u32('head-tail', HEAD_TAIL_BREAKS_OFFSET + Math.max(headTailRounds, 1))
      : undefined;
    // Indirect dispatch triples of the head/tail moment kernels: slot 0 the tile kernel, slot 1
    // the merge kernel. Prepare arms them for the head/tail method and a split kernel disarms them
    // once the run is done, so the remaining rounds launch nothing (rule 13).
    const headTailDispatch = usesHeadTail
      ? createTransientView(
          graph,
          `${id}-head-tail-dispatch`,
          'uint32',
          6,
          Buffer.STORAGE | Buffer.INDIRECT
        )
      : undefined;
    const headTailLayouts = (
      usesHeadTail
        ? [getClassBreaksMomentTileCount(rowCount) * MOMENT_WORKGROUP_SIZE, MOMENT_WORKGROUP_SIZE]
        : []
    ).map(invocationCount =>
      getBoundedDispatchLayout(
        OPERATION,
        invocationCount,
        MOMENT_WORKGROUP_SIZE,
        graph.device.limits.maxComputeWorkgroupsPerDimension
      )
    );
    const headTailDispatchValues = headTailLayouts.flatMap(layout => [
      layout.x,
      layout.y,
      layout.z
    ]);
    const headTailDispatchWGSL = (activeExpression: string) =>
      [0, 1]
        .map(
          slot => `
  dispatch[dispatchOffset + ${slot * 3}u] = select(0u, getChunkParameter(${36 + slot * 3}u), ${activeExpression});
  dispatch[dispatchOffset + ${slot * 3 + 1}u] = getChunkParameter(${37 + slot * 3}u);
  dispatch[dispatchOffset + ${slot * 3 + 2}u] = getChunkParameter(${38 + slot * 3}u);`
        )
        .join('\n');
    const compiledMask = methods.reduce((bits, method) => bits | (1 << CODES[method]), 0);
    const methodRead = `state[stateOffset + ${CLASS_BREAKS_STATE.method}u]`;
    const classCountRead = `state[stateOffset + ${CLASS_BREAKS_STATE.classCount}u]`;
    // 1. Decode the per-frame method and class count, and reset every accumulator.
    const prepareBindings: WGSLKernelBinding[] = [
      read('params', parameters, 'f32'),
      write('state', state),
      write('extremes', extremes),
      write('edgeState', edgeState),
      ...(quantileGate && quantileParameters
        ? [
            write('quantileGate', quantileGate),
            write('quantileParameters', quantileParameters, 'f32')
          ]
        : []),
      ...(headTail ? [write('headTail', headTail)] : []),
      ...(headTailDispatch ? [write('dispatch', headTailDispatch)] : [])
    ];
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-prepare`,
        operation: OPERATION,
        variant: 'prepare',
        parameterValues: headTailDispatchValues,
        bindings: prepareBindings,
        invocationCount: 1,
        declarations: `const COMPILED_METHODS: u32 = ${compiledMask}u;
const MAXIMUM_CLASS_COUNT: u32 = ${maximumClassCount}u;
const QUANTILE_COUNT: u32 = ${quantileCount}u;
const HEADER: u32 = ${GPU_CLASS_BREAKS_PARAMETER_HEADER_LENGTH}u;`,
        body: `let methodValue = params[paramsOffset];
  var method = ${INVALID_METHOD}u;
  if (methodValue >= 0.0 && methodValue < 32.0 && floor(methodValue) == methodValue) {
    let code = u32(methodValue);
    if ((COMPILED_METHODS & (1u << code)) != 0u) {
      method = code;
    }
  }
  let classValue = params[paramsOffset + 1u];
  var classCount = 1u;
  if (classValue >= 1.0) {
    classCount = u32(min(classValue, f32(MAXIMUM_CLASS_COUNT)));
  }
  if (method == ${CODES['box-plot']}u) {
    classCount = ${GPU_CLASS_BREAKS_BOX_PLOT_CLASS_COUNT}u;
  }
  state[stateOffset + ${CLASS_BREAKS_STATE.method}u] = method;
  state[stateOffset + ${CLASS_BREAKS_STATE.classCount}u] = classCount;
  state[stateOffset + ${CLASS_BREAKS_STATE.layerCursor}u] = 0u;
  state[stateOffset + ${CLASS_BREAKS_STATE.refinementCursor}u] = 0u;
  state[stateOffset + ${CLASS_BREAKS_STATE.refinementMoved}u] = 0u;
  state[stateOffset + ${CLASS_BREAKS_STATE.refinementDone}u] = 0u;
  extremes[extremesOffset + ${CLASS_BREAKS_EXTREMES.minimumKey}u] = 0xffffffffu;
  extremes[extremesOffset + ${CLASS_BREAKS_EXTREMES.maximumKey}u] = 0u;
  extremes[extremesOffset + ${CLASS_BREAKS_EXTREMES.validCount}u] = 0u;
  extremes[extremesOffset + ${CLASS_BREAKS_EXTREMES.finiteCount}u] = 0u;
  edgeState[edgeStateOffset] = 0u;
  ${
    quantileGate
      ? `quantileGate[quantileGateOffset] = select(0u, 1u, method == ${CODES.quantile}u || method == ${CODES['box-plot']}u);
  quantileParameters[quantileParametersOffset] = ${LINEAR_INTERPOLATION_CODE}.0;
  quantileParameters[quantileParametersOffset + 1u] = 0.0;
  quantileParameters[quantileParametersOffset + 2u] = 1.0;
  quantileParameters[quantileParametersOffset + 3u] = 0.0;
  for (var slot = 0u; slot < QUANTILE_COUNT; slot = slot + 1u) {
    quantileParameters[quantileParametersOffset + 4u + slot] = params[paramsOffset + HEADER + slot];
  }`
      : ''
  }
  ${
    headTail
      ? `headTail[headTailOffset + ${CLASS_BREAKS_HEAD_TAIL.done}u] = 0u;
  headTail[headTailOffset + ${CLASS_BREAKS_HEAD_TAIL.breakCount}u] = 0u;
  headTail[headTailOffset + ${CLASS_BREAKS_HEAD_TAIL.thresholdKey}u] = 0u;
  headTail[headTailOffset + ${CLASS_BREAKS_HEAD_TAIL.previousCount}u] = 0u;
  ${headTailDispatchWGSL(`method == ${CODES['head-tail']}u`)}`
      : ''
  }`
      })
    );

    // 2. Exact extremes of the finite values and counts, on ordered keys.
    for (const [chunkIndex, values] of valueChunks.entries()) {
      if (values.length === 0) continue;
      const mask = maskChunks?.[chunkIndex];
      const maskBinding = mask ? [read('rowMask', mask, 'u32')] : [];
      const rowCount = values.length;
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: valueChunks.length === 1 ? `${id}-extremes` : `${id}-extremes-chunk-${chunkIndex}`,
          operation: OPERATION,
          variant: 'extremes',
          bindings: [
            read('values', values, 'f32'),
            ...maskBinding,
            write('extremes', extremes, 'atomic<u32>')
          ],
          invocationCount: rowCount,
          declarations: COLUMN_ORDERED_KEY_WGSL,
          body: `let value = values[valuesOffset + index];
  if (isNanBits(value) || !(${maskTest})) {
    return;
  }
  atomicAdd(&extremes[extremesOffset + ${CLASS_BREAKS_EXTREMES.validCount}u], 1u);
  if ((bitcast<u32>(value) & 0x7f800000u) == 0x7f800000u) {
    return;
  }
  let key = getOrderedKey(value);
  atomicAdd(&extremes[extremesOffset + ${CLASS_BREAKS_EXTREMES.finiteCount}u], 1u);
  atomicMin(&extremes[extremesOffset + ${CLASS_BREAKS_EXTREMES.minimumKey}u], key);
  atomicMax(&extremes[extremesOffset + ${CLASS_BREAKS_EXTREMES.maximumKey}u], key);`
        })
      );
    }

    // 3. Exact linear quantiles for 'quantile' and 'box-plot', gated per frame.
    if (quantileGate && quantileParameters && quantileValues) {
      nodes.push(
        ...getColumnQuantileNodes<Parameters>(graph, {
          id: `${id}-quantiles`,
          operation: OPERATION,
          values,
          mask,
          parameters: quantileParameters,
          quantileCount,
          output: {quantiles: quantileValues, validCount: u32('quantile-valid-count', 1)},
          gate: {view: quantileGate, value: 1}
        })
      );
    }

    // 4. Fixed-order moments for 'standard-deviation' (the last slot).
    if (moments && partials && usesStandardDeviation) {
      nodes.push(
        ...getClassBreaksMomentNodes<Parameters>(graph, {
          id: `${id}-moments`,
          operation: OPERATION,
          values: packedValues!,
          mask: packedMask,
          state,
          partials,
          moments,
          slot: momentSlotCount - 1,
          gate: {kind: 'method', methodCode: CODES['standard-deviation']}
        })
      );
    }

    // 5. Head/tail rounds: the mean of the current head, then a one-thread split decision. Every
    // round compiles to the same four pipelines: the round index lives in the head/tail state
    // (rule 12), and the split kernel disarms the moment kernels' dispatch once done (rule 13).
    if (moments && partials && headTail && headTailDispatch) {
      for (let round = 0; round < headTailRounds; round++) {
        const roundId = `${id}-head-tail-${round}`;
        nodes.push(
          ...getClassBreaksMomentNodes<Parameters>(graph, {
            id: roundId,
            operation: OPERATION,
            values: packedValues!,
            mask: packedMask,
            state,
            partials,
            moments,
            slot: 0,
            gate: {kind: 'head-tail', methodCode: CODES['head-tail'], headTail},
            indirectGates: {
              tile: getClassBreaksIndirectGate<Parameters>(`${roundId}-tile`, headTailDispatch, 0),
              merge: getClassBreaksIndirectGate<Parameters>(`${roundId}-merge`, headTailDispatch, 1)
            }
          })
        );
        nodes.push(
          createWGSLKernelNode<Parameters>(graph, {
            id: `${roundId}-split`,
            operation: OPERATION,
            variant: 'head-tail-split',
            parameterValues: headTailDispatchValues,
            bindings: [
              read('params', parameters, 'f32'),
              read('state', state, 'u32'),
              read('moments', moments, 'f32'),
              write('headTail', headTail),
              write('dispatch', headTailDispatch)
            ],
            invocationCount: 1,
            declarations: COLUMN_ORDERED_KEY_WGSL,
            body: headTailSplitBody(methodRead, classCountRead, headTailDispatchWGSL('false'))
          })
        );
      }
    }

    // 6. Maximum breaks: sort the finite keys, sort the gaps between distinct neighbours.
    if (has('maximum-breaks')) {
      nodes.push(
        ...this.getMaximumBreaksNodes(graph, {
          values: values as GraphDataView<'float32'>,
          state,
          extremes,
          edges,
          edgeState
        })
      );
    }

    // 7. Sort-based methods remain packed; approximate histograms/refinement stream chunks.
    if (has('natural-breaks')) {
      const shared = {
        id,
        operation: OPERATION,
        state,
        extremes,
        edges,
        edgeState,
        maximumClassCount,
        binCount: this.naturalBreaksBinCount
      };
      if (props.naturalBreaksMode === 'exact' || props.naturalBreaksMode === 'monotone') {
        const getNodes =
          props.naturalBreaksMode === 'exact'
            ? getExactNaturalBreaksNodes<Parameters>
            : getMonotoneNaturalBreaksNodes<Parameters>;
        nodes.push(
          ...getNodes(graph, {
            ...shared,
            values: values as GraphDataView<'float32'>,
            mask: mask as GraphDataView<'uint32'> | undefined,
            exactStorage: props.naturalBreaksExactStorage
          })
        );
      } else {
        nodes.push(...getNaturalBreaksNodes(graph, {...shared, values, mask}));
      }
    }

    // 8. Closed-form edges for equal interval, quantile, standard deviation, box plot, custom.
    const finishBindings: WGSLKernelBinding[] = [
      read('params', parameters, 'f32'),
      read('state', state, 'u32'),
      read('extremes', extremes, 'u32'),
      ...(quantileValues ? [read('quantileValues', quantileValues, 'f32')] : []),
      // Head/tail rounds use moments as scratch, but finish reads their edges from headTail.
      // Supplying an unused moments binding disagrees with WebGPU's inferred shader layout.
      ...(usesStandardDeviation && moments ? [read('moments', moments, 'f32')] : []),
      ...(headTail ? [read('headTail', headTail, 'u32')] : []),
      write('edges', edges, 'f32')
    ];
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-finish`,
        operation: OPERATION,
        variant: 'finish',
        bindings: [...finishBindings, write('edgeState', edgeState)],
        invocationCount: 1,
        declarations: `${COLUMN_ORDERED_KEY_WGSL}
const HEADER: u32 = ${GPU_CLASS_BREAKS_PARAMETER_HEADER_LENGTH}u;`,
        body: this.getFinishBody({
          methodRead,
          classCountRead,
          hasQuantiles: Boolean(quantileValues),
          momentsSlot: usesStandardDeviation ? momentSlotCount - 1 : -1,
          hasHeadTail: Boolean(headTail)
        })
      })
    );

    // 9. Publish: clamp, fill the end edges, apply the degenerate-data rules, NaN the tail.
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-publish`,
        operation: OPERATION,
        variant: 'publish',
        bindings: [
          read('state', state, 'u32'),
          read('extremes', extremes, 'u32'),
          read('edges', edges, 'f32'),
          read('edgeState', edgeState, 'u32'),
          write('breaksOut', output.breaks, 'f32'),
          write('classCountOut', output.classCount)
        ],
        invocationCount: 1,
        declarations: `${COLUMN_ORDERED_KEY_WGSL}
const MAXIMUM_CLASS_COUNT: u32 = ${maximumClassCount}u;`,
        body: `let method = ${methodRead};
  let finiteCount = extremes[extremesOffset + ${CLASS_BREAKS_EXTREMES.finiteCount}u];
  let minimum = decodeOrderedKey(extremes[extremesOffset + ${CLASS_BREAKS_EXTREMES.minimumKey}u]);
  let maximum = decodeOrderedKey(extremes[extremesOffset + ${CLASS_BREAKS_EXTREMES.maximumKey}u]);
  var classes = edgeState[edgeStateOffset];
  let isCustom = method == ${CODES.custom}u;
  if (method == ${INVALID_METHOD}u) {
    classes = 0u;
  } else if (!isCustom && finiteCount == 0u) {
    classes = 0u;
  } else if (!isCustom && minimum == maximum) {
    classes = 1u;
  }
  classes = min(classes, MAXIMUM_CLASS_COUNT);
  // A NaN edge (a hand-written parameter view with a NaN probability, hinge, interval or custom
  // edge) makes the classification undefined: report 0 classes and NaN edges, the no-data result,
  // instead of clamping NaN to an end edge and assigning a silent wrong class.
  for (var edge = 0u; edge <= classes; edge = edge + 1u) {
    let isComputedEnd = !isCustom && (edge == 0u || edge == classes);
    if (!isComputedEnd && isNanBits(edges[edgesOffset + edge])) {
      classes = 0u;
    }
  }
  let nan = getNaN();
  for (var edge = 0u; edge <= MAXIMUM_CLASS_COUNT; edge = edge + 1u) {
    var value = nan;
    if (classes > 0u && edge <= classes) {
      if (isCustom) {
        value = edges[edgesOffset + edge];
      } else if (edge == 0u) {
        value = minimum;
      } else if (edge == classes) {
        value = maximum;
      } else {
        value = clamp(edges[edgesOffset + edge], minimum, maximum);
      }
    }
    breaksOut[breaksOutOffset + edge] = value;
  }
  classCountOut[classCountOutOffset] = classes;`
      })
    );

    // 10. Optional per-class row counts with the stream's class rule.
    if (output.classCounts) {
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-class-counts-clear`,
          operation: OPERATION,
          variant: 'class-counts-clear',
          bindings: [write('classCounts', output.classCounts)],
          invocationCount: maximumClassCount,
          body: 'classCounts[classCountsOffset + index] = 0u;'
        })
      );
      for (const [chunkIndex, values] of valueChunks.entries()) {
        if (values.length === 0) continue;
        const mask = maskChunks?.[chunkIndex];
        const maskBinding = mask ? [read('rowMask', mask, 'u32')] : [];
        const rowCount = values.length;
        nodes.push(
          createWGSLKernelNode<Parameters>(graph, {
            id:
              valueChunks.length === 1
                ? `${id}-class-counts`
                : `${id}-class-counts-chunk-${chunkIndex}`,
            operation: OPERATION,
            variant: 'class-counts',
            bindings: [
              read('values', values, 'f32'),
              ...maskBinding,
              read('breaks', output.breaks, 'f32'),
              read('classCountIn', output.classCount, 'u32'),
              write('classCounts', output.classCounts, 'atomic<u32>')
            ],
            invocationCount: rowCount,
            declarations: COLUMN_ORDERED_KEY_WGSL,
            body: `let value = values[valuesOffset + index];
  let classes = classCountIn[classCountInOffset];
  if (classes == 0u || isNanBits(value) || !(${maskTest})) {
    return;
  }
  let key = getComparisonKey(value);
  // Binary search for the number of inner edges e[1..classes-1] whose key is at or below key.
  var low = 1u;
  var high = classes;
  while (low < high) {
    let middle = (low + high) >> 1u;
    if (getComparisonKey(breaks[breaksOffset + middle]) <= key) {
      low = middle + 1u;
    } else {
      high = middle;
    }
  }
  atomicAdd(&classCounts[classCountsOffset + low - 1u], 1u);`
          })
        );
      }
    }
    return nodes;
  }

  /** Builds the WGSL body of the closed-form finish kernel. */
  private getFinishBody(props: {
    methodRead: string;
    classCountRead: string;
    hasQuantiles: boolean;
    momentsSlot: number;
    hasHeadTail: boolean;
  }): string {
    const {methodRead, classCountRead, hasQuantiles, momentsSlot, hasHeadTail} = props;
    const minimumKey = `extremes[extremesOffset + ${CLASS_BREAKS_EXTREMES.minimumKey}u]`;
    const maximumKey = `extremes[extremesOffset + ${CLASS_BREAKS_EXTREMES.maximumKey}u]`;
    const momentBase = momentsSlot * MOMENT_STRIDE;
    return `let method = ${methodRead};
  let classCount = ${classCountRead};
  let minimum = decodeOrderedKey(${minimumKey});
  let maximum = decodeOrderedKey(${maximumKey});
  if (method == ${CODES['equal-interval']}u) {
    let span = maximum - minimum;
    let width = span / f32(classCount);
    // Finite extrema can span more than float32 holds ([-3e38, 3e38]); then the edges are built
    // from halved endpoints, where neither the span nor a partial sum can overflow, and doubled
    // exactly. The ordinary rounding is unchanged when the span is finite.
    let isSpanFinite = (bitcast<u32>(span) & 0x7f800000u) != 0x7f800000u;
    let halfMinimum = minimum * 0.5;
    let halfWidth = (maximum * 0.5 - halfMinimum) / f32(classCount);
    for (var edge = 1u; edge < classCount; edge = edge + 1u) {
      edges[edgesOffset + edge] = select(
        2.0 * (halfMinimum + halfWidth * f32(edge)),
        minimum + width * f32(edge),
        isSpanFinite
      );
    }
    edgeState[edgeStateOffset] = classCount;
  }
  ${
    hasQuantiles
      ? `if (method == ${CODES.quantile}u) {
    for (var edge = 1u; edge < classCount; edge = edge + 1u) {
      edges[edgesOffset + edge] = quantileValues[quantileValuesOffset + edge - 1u];
    }
    edgeState[edgeStateOffset] = classCount;
  }
  if (method == ${CODES['box-plot']}u) {
    let hinge = params[paramsOffset + 4u];
    let lowerQuartile = quantileValues[quantileValuesOffset];
    let median = quantileValues[quantileValuesOffset + 1u];
    let upperQuartile = quantileValues[quantileValuesOffset + 2u];
    let spread = upperQuartile - lowerQuartile;
    edges[edgesOffset + 1u] = lowerQuartile - hinge * spread;
    edges[edgesOffset + 2u] = lowerQuartile;
    edges[edgesOffset + 3u] = median;
    edges[edgesOffset + 4u] = upperQuartile;
    edges[edgesOffset + 5u] = upperQuartile + hinge * spread;
    edgeState[edgeStateOffset] = ${GPU_CLASS_BREAKS_BOX_PLOT_CLASS_COUNT}u;
  }`
      : ''
  }
  ${
    momentsSlot >= 0
      ? `if (method == ${CODES['standard-deviation']}u) {
    let count = moments[momentsOffset + ${momentBase}u];
    let mean = moments[momentsOffset + ${momentBase + 1}u];
    // Sample standard deviation (n - 1), as mapclassify StdMean and R classInt 'sd'.
    let deviation = sqrt(moments[momentsOffset + ${momentBase + 2}u] / max(count - 1.0, 1.0));
    let interval = params[paramsOffset + 2u] * deviation;
    let center = f32(classCount - 2u) * 0.5;
    for (var edge = 1u; edge < classCount; edge = edge + 1u) {
      edges[edgesOffset + edge] = mean + (f32(edge - 1u) - center) * interval;
    }
    edgeState[edgeStateOffset] = classCount;
  }`
      : ''
  }
  ${
    hasHeadTail
      ? `if (method == ${CODES['head-tail']}u) {
    let breakCount = headTail[headTailOffset + ${CLASS_BREAKS_HEAD_TAIL.breakCount}u];
    for (var edge = 1u; edge <= breakCount; edge = edge + 1u) {
      edges[edgesOffset + edge] = bitcast<f32>(
        headTail[headTailOffset + ${HEAD_TAIL_BREAKS_OFFSET - 1}u + edge]
      );
    }
    edgeState[edgeStateOffset] = breakCount + 1u;
  }`
      : ''
  }
  if (method == ${CODES.custom}u) {
    for (var edge = 0u; edge <= classCount; edge = edge + 1u) {
      edges[edgesOffset + edge] = params[paramsOffset + HEADER + edge];
    }
    edgeState[edgeStateOffset] = classCount;
  }`;
  }

  /** Sort-based exact maximum breaks. */
  private getMaximumBreaksNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>,
    shared: {
      /** The float32 view of the values (converted when the column is integer). */
      values: GraphDataView<'float32'>;
      state: GraphDataView<'uint32'>;
      extremes: GraphDataView<'uint32'>;
      edges: GraphDataView<'float32'>;
      edgeState: GraphDataView<'uint32'>;
    }
  ): GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const {values} = shared;
    const {maximumClassCount} = props;
    const mask = props.mask as GraphDataView<'uint32'> | undefined;
    const rowCount = values.length;
    const prefix = `${id}-maximum-breaks`;
    const u32 = (name: string) =>
      createTransientView(graph, `${prefix}-${name}`, 'uint32', rowCount);
    const keys = u32('keys');
    const rows = u32('rows');
    const sortedKeys = u32('sorted-keys');
    const sortedRows = u32('sorted-rows');
    const gapKeys = u32('gap-keys');
    const gapPositions = u32('gap-positions');
    const sortedGapKeys = u32('sorted-gap-keys');
    const sortedGapPositions = u32('sorted-gap-positions');
    const gate = `state[stateOffset + ${CLASS_BREAKS_STATE.method}u] != ${CODES['maximum-breaks']}u`;
    const nodes: GPUCommandNode<Parameters>[] = [];
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${prefix}-keys`,
        operation: OPERATION,
        variant: 'maximum-breaks-keys',
        bindings: [
          {name: 'values', view: values, type: 'f32', access: 'read'},
          ...(mask ? [{name: 'rowMask', view: mask, type: 'u32', access: 'read'} as const] : []),
          {name: 'state', view: shared.state, type: 'u32', access: 'read'},
          {name: 'keys', view: keys, type: 'u32', access: 'read_write'},
          {name: 'rows', view: rows, type: 'u32', access: 'read_write'}
        ],
        invocationCount: rowCount,
        declarations: COLUMN_ORDERED_KEY_WGSL,
        body: `if (${gate}) {
    return;
  }
  let value = values[valuesOffset + index];
  var include = (bitcast<u32>(value) & 0x7f800000u) != 0x7f800000u;
  ${mask ? 'include = include && rowMask[rowMaskOffset + index] != 0u;' : ''}
  // Excluded rows take the NaN key 0xffffffff, which no finite value has, and sort last.
  keys[keysOffset + index] = select(0xffffffffu, getComparisonKey(value), include);
  rows[rowsOffset + index] = index;`
      })
    );
    nodes.push(
      ...new GPUSort({
        id: `${prefix}-sort`,
        keys,
        values: rows,
        outputKeys: sortedKeys,
        outputValues: sortedRows,
        algorithm: 'radix'
      }).getCommandNodes(graph)
    );
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${prefix}-gaps`,
        operation: OPERATION,
        variant: 'maximum-breaks-gaps',
        bindings: [
          {name: 'state', view: shared.state, type: 'u32', access: 'read'},
          {name: 'extremes', view: shared.extremes, type: 'u32', access: 'read'},
          {name: 'sortedKeys', view: sortedKeys, type: 'u32', access: 'read'},
          {name: 'gapKeys', view: gapKeys, type: 'u32', access: 'read_write'},
          {name: 'gapPositions', view: gapPositions, type: 'u32', access: 'read_write'}
        ],
        invocationCount: rowCount,
        declarations: `${COLUMN_ORDERED_KEY_WGSL}
// Return the positive gap's IEEE magnitude without subnormal floating-point arithmetic.
// Zero means this pair has a normal gap and may use the ordinary subtraction below.
fn getSmallGapMagnitude(lowerBits: u32, upperBits: u32) -> u32 {
  let lowerMagnitude = lowerBits & 0x7fffffffu;
  let upperMagnitude = upperBits & 0x7fffffffu;
  let largerMagnitude = max(lowerMagnitude, upperMagnitude);
  let smallerMagnitude = min(lowerMagnitude, upperMagnitude);
  var difference: u32;
  var exponent = 1u;
  if (((lowerBits ^ upperBits) & 0x80000000u) != 0u) {
    // These first two binades have integer magnitudes in units of 2^-149.
    if (largerMagnitude >= 0x01000000u) { return 0u; }
    difference = largerMagnitude + smallerMagnitude;
  } else {
    let largerExponent = max(largerMagnitude >> 23u, 1u);
    let smallerExponent = max(smallerMagnitude >> 23u, 1u);
    // A subnormal gap is possible only in these neighboring small binades.
    if (largerExponent > 24u || largerExponent - smallerExponent > 1u) { return 0u; }
    let largerSignificand = (largerMagnitude & 0x007fffffu) |
      select(0u, 0x00800000u, largerMagnitude >= 0x00800000u);
    let smallerSignificand = (smallerMagnitude & 0x007fffffu) |
      select(0u, 0x00800000u, smallerMagnitude >= 0x00800000u);
    difference = (largerSignificand << (largerExponent - smallerExponent)) - smallerSignificand;
    exponent = smallerExponent;
  }
  if (difference == 0u) { return 0u; }
  let leadingBit = firstLeadingBit(difference);
  if (leadingBit + exponent < 24u) {
    return difference << (exponent - 1u);
  }
  // At most 25 significant bits: a single discarded bit uses ties-to-even rounding.
  var significand: u32;
  if (leadingBit > 23u) {
    significand = (difference >> 1u) + select(0u, 1u, (difference & 3u) == 3u);
  } else {
    significand = difference << (23u - leadingBit);
  }
  return ((leadingBit + exponent - 24u) << 23u) + significand;
}
fn getHalfGapEndpoint(value: f32) -> f32 {
  let bits = bitcast<u32>(value);
  // Adjust the exponent as an integer so Metal cannot reassociate this into a subtract-first
  // half gap. The large-gap route has at least one huge normal endpoint; a tiny other endpoint
  // cannot affect its rounded gap, so underflow on that endpoint is harmless here.
  if ((bits & 0x7fffffffu) >= 0x01000000u) {
    return bitcast<f32>(bits - 0x00800000u);
  }
  return value * 0.5;
}`,
        body: `if (${gate}) {
    return;
  }
  let finiteCount = extremes[extremesOffset + ${CLASS_BREAKS_EXTREMES.finiteCount}u];
  var gapKey = 0xffffffffu;
  if (index > 0u && index < finiteCount) {
    let lowerKey = sortedKeys[sortedKeysOffset + index - 1u];
    let upperKey = sortedKeys[sortedKeysOffset + index];
    if (upperKey != lowerKey) {
      let lower = decodeOrderedKey(lowerKey);
      let upper = decodeOrderedKey(upperKey);
      var magnitude = getSmallGapMagnitude(bitcast<u32>(lower), bitcast<u32>(upper));
      // Gate on endpoint magnitude before subtraction: no overflow test can survive Metal's
      // reciprocal/reassociation optimizations. Large endpoints have no subnormal gap.
      if (magnitude != 0u) {
        // Already encoded exactly from endpoint bits, including subnormal gap magnitudes.
      } else if (max(abs(lower), abs(upper)) >= 0x1p126) {
        let halfGap = getHalfGapEndpoint(upper) - getHalfGapEndpoint(lower);
        // An extra exponent bit preserves distinct gaps above the finite f32 range.
        magnitude = bitcast<u32>(halfGap) + 0x00800000u;
      } else {
        magnitude = bitcast<u32>(upper - lower);
      }
      // Positive IEEE magnitudes ascend; the stable sort keeps lower positions first.
      gapKey = ~magnitude;
    }
  }
  gapKeys[gapKeysOffset + index] = gapKey;
  gapPositions[gapPositionsOffset + index] = index;`
      })
    );
    nodes.push(
      ...new GPUSort({
        id: `${prefix}-gap-sort`,
        keys: gapKeys,
        values: gapPositions,
        outputKeys: sortedGapKeys,
        outputValues: sortedGapPositions,
        algorithm: 'radix'
      }).getCommandNodes(graph)
    );
    // One invocation per selected gap; no legacy subtract-first midpoint is evaluated.
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${prefix}-midpoints`,
        operation: OPERATION,
        variant: 'maximum-breaks-midpoints',
        invocationCount: maximumClassCount,
        bindings: [
          {name: 'state', view: shared.state, type: 'u32', access: 'read'},
          {name: 'sortedKeys', view: sortedKeys, type: 'u32', access: 'read'},
          {name: 'sortedGapKeys', view: sortedGapKeys, type: 'u32', access: 'read'},
          {name: 'sortedGapPositions', view: sortedGapPositions, type: 'u32', access: 'read'},
          {name: 'edges', view: shared.edges, type: 'f32', access: 'read_write'},
          {name: 'edgeState', view: shared.edgeState, type: 'u32', access: 'read_write'}
        ],
        declarations: COLUMN_ORDERED_KEY_WGSL,
        body: `if (${gate}) { return; }
  let classCount = state[stateOffset + ${CLASS_BREAKS_STATE.classCount}u];
  let available = arrayLength(&sortedGapKeys) - sortedGapKeysOffset;
  if (index == 0u) {
    var found = 0u;
    for (var rank = 0u; rank + 1u < classCount && rank < available; rank = rank + 1u) {
      if (sortedGapKeys[sortedGapKeysOffset + rank] == 0xffffffffu) { break; }
      found = found + 1u;
    }
    edgeState[edgeStateOffset] = found + 1u;
  }
  if (index + 1u >= classCount || index >= available) { return; }
  if (sortedGapKeys[sortedGapKeysOffset + index] == 0xffffffffu) { return; }
  let position = sortedGapPositions[sortedGapPositionsOffset + index];
  let lower = decodeOrderedKey(sortedKeys[sortedKeysOffset + position - 1u]);
  let upper = decodeOrderedKey(sortedKeys[sortedKeysOffset + position]);
  var midpoint: f32;
  if ((bitcast<u32>(lower) ^ bitcast<u32>(upper)) >= 0x80000000u) {
    midpoint = (lower + upper) * 0.5;
  } else {
    midpoint = lower + (upper - lower) * 0.5;
  }
  // Compare stored float32 bits: Metal may reassociate the midpoint expression.
  if (getComparisonKey(midpoint) <= getComparisonKey(lower)) {
    midpoint = upper;
  }
  var slot = 1u;
  for (var rank = 0u; rank + 1u < classCount && rank < available; rank = rank + 1u) {
    if (sortedGapKeys[sortedGapKeysOffset + rank] == 0xffffffffu) { break; }
    if (sortedGapPositions[sortedGapPositionsOffset + rank] < position) { slot = slot + 1u; }
  }
  edges[edgesOffset + slot] = midpoint;`
      })
    );
    return nodes;
  }
}

/**
 * One head/tail split decision: take the mean of the current head as a break. The round is the
 * break count so far (equal to the round index while the run is not done), so every round shares
 * one pipeline. Stopping disarms the moment kernels of the remaining rounds.
 */
function headTailSplitBody(methodRead: string, classCountRead: string, disarmWGSL: string): string {
  const word = (name: keyof typeof CLASS_BREAKS_HEAD_TAIL) =>
    `headTail[headTailOffset + ${CLASS_BREAKS_HEAD_TAIL[name]}u]`;
  const done = word('done');
  return `if (${methodRead} != ${CODES['head-tail']}u || ${done} != 0u) {
    return;
  }
  let round = ${word('breakCount')};
  let count = moments[momentsOffset];
  let mean = moments[momentsOffset + 1u];
  let minimum = moments[momentsOffset + 3u];
  let maximum = moments[momentsOffset + 4u];
  var ratio = params[paramsOffset + 3u];
  if (!(ratio >= 0.0)) {
    ratio = 0.4;
  }
  // At most k - 1 breaks; stop on an empty or single-valued head.
  var stop = round + 1u >= ${classCountRead} || count < 1.0 || !(minimum < maximum);
  if (!stop && round > 0u) {
    let previousCount = bitcast<f32>(${word('previousCount')});
    // The head kept from the previous split must be a minority (the 40% rule).
    stop = count > ratio * previousCount || count <= 1.0;
  }
  if (stop) {
    ${done} = 1u;
    ${disarmWGSL}
    return;
  }
  headTail[headTailOffset + ${HEAD_TAIL_BREAKS_OFFSET}u + round] = bitcast<u32>(mean);
  ${word('breakCount')} = round + 1u;
  ${word('thresholdKey')} = getOrderedKey(mean);
  ${word('previousCount')} = bitcast<u32>(count);`;
}

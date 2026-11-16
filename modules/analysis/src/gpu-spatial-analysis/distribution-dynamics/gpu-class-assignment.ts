// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createGraphChunkGroupKernelNode,
  validateGraphChunkGroupKernelSupport
} from '../../analysis-core/kernels/graph-chunk-groups';

import {
  validatePackedUint32View,
  getBoundedDispatchLayout,
  type GraphVectorView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import type {GPUCommandNodeProducer} from '@luma.gl/graph/gpu-core';
import {type WGSLKernelBinding} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  validateGraphOutputsDisjointFromInputs,
  getGraphViewChunks,
  validateCommandNodeBudget,
  validateSinglePackedView,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  planGraphChunkDispatch,
  validateDisjointWritableViews,
  validateGraphChunkBindingRanges,
  validateGraphChunkDispatchBindings,
  type GraphChunkDispatchPlan
} from '../../analysis-core/kernels/graph-chunk-dispatch';
import type {GPUClassifiedValuesPort} from '../../analysis-core/contracts/index';

const OPERATION = 'GPUClassAssignment';
/** Fixed workgroup size keeps all row chunks on the same pipeline specialization. */
const WORKGROUP_SIZE = 256;

/** Class value written for a masked row or a NaN value. */
export const GPU_CLASS_ASSIGNMENT_NO_CLASS = 0xffffffff;

/**
 * Which end of a class interval is closed, that is, which class a value equal to an inner edge
 * belongs to. It depends on what the edges are:
 * - `'left'`: classes are `[e[j], e[j + 1])`. An edge is the lower edge of the class above it, so a
 *   value equal to an edge belongs to the upper class. Use it for lower edges: the
 *   `'natural-breaks'` edges of `GPUClassBreaks` (each is the smallest value of its class), d3
 *   threshold and quantile scales (`bisectRight`), and to reproduce `GPUClassBreaks.classCounts`.
 * - `'right'`: classes are `(e[j], e[j + 1]]`. An edge is the upper bound of the class below it, so
 *   a value equal to an edge belongs to the lower class, and the minimum belongs to the first
 *   class. Use it for upper-bound bins: mapclassify classifiers and giddy, which classifies with
 *   them (`numpy.digitize(values, bins, right=True)`).
 */
export type GPUClassIntervalClosure = 'left' | 'right';

/** Preserves packed row types by default and the caller's chunked view kind when requested. */
type GPUClassAssignmentRowView<
  Format extends GraphDataView['format'],
  Rows extends GraphDataView | GraphVectorView
> = Rows extends GraphVectorView ? GraphVectorView<Format> : GraphDataView<Format>;

/**
 * Properties for {@link GPUClassAssignment}.
 *
 * Per-frame: the contents of every view. Topology: view lengths and whether `mask` is present.
 */
export type GPUClassAssignmentProps<Rows extends GraphDataView | GraphVectorView = GraphDataView> =
  {
    /** Prefix for generated node IDs. Defaults to `'class-assignment'`. */
    id?: string;
    /**
     * Packed float32 column. For distribution dynamics this is the pooled column of every period
     * (`rows * periods` entries, period-major), so that all periods share one set of breaks.
     */
    values: GPUClassAssignmentRowView<'float32', Rows>;
    /**
     * Class edges `e[0..k]`, at least `classCount + 1` rows, for example the `breaks` of a
     * `GPUClassBreaks` run on `values`. Only inner edges `e[1..k-1]` are compared.
     */
    breaks: GraphDataView<'float32'>;
    /** One row holding the class count `k`, for example the `classCount` of `GPUClassBreaks`. */
    classCount: GraphDataView<'uint32'>;
    /** Optional per-row selection (same length as `values`): zero rows get {@link GPU_CLASS_ASSIGNMENT_NO_CLASS}. */
    mask?: GPUClassAssignmentRowView<'uint32', Rows>;
    /**
     * Compile-time side on which class intervals are closed, see {@link GPUClassIntervalClosure}.
     * Defaults to `'left'` (lower edges, the rule of `GPUClassBreaks.classCounts`). Pass `'right'`
     * when the edges are upper-bound bins and the classes must equal mapclassify's or giddy's.
     */
    intervalClosure?: GPUClassIntervalClosure;
    /** Caller-owned class per row, `0..k-1`. */
    output: GPUClassAssignmentRowView<'uint32', Rows>;
  };

/**
 * Assigns each row of a column the class of the shared breaks, clamped to `0..k-1` (infinities
 * clamp to the end classes). NaN and masked rows get {@link GPU_CLASS_ASSIGNMENT_NO_CLASS}. When
 * `k` is 0 every row gets that value.
 *
 * The class is a count of inner edges `e[1..k-1]`, and `intervalClosure` decides what happens to
 * a value equal to an edge:
 * - `'left'` (default): the number of inner edges less than or equal to the value, so the value
 *   goes to the upper class. This is the class rule of `GPUClassBreaks.classCounts` and of d3
 *   `bisectRight`, and the right one for lower edges such as `'natural-breaks'`.
 * - `'right'`: the number of inner edges strictly less than the value, so the value stays in the
 *   lower class. This is the rule of mapclassify (`x <= upper bound`) and therefore of giddy.
 *
 * The two differ only for rows exactly equal to an inner edge. Quantile edges land exactly on data
 * values whenever the data are tied at the quantile (rounded rates) or the quantile position is a
 * whole number, so with `'quantile'` breaks pass `'right'` to count the transitions giddy counts.
 *
 * Running one `GPUClassBreaks` over the pooled column of all periods and assigning every period
 * with these breaks keeps a single legend across a time slider, as giddy's pooled classification
 * does. The result feeds `GPUTransitionMatrix`.
 *
 * Integer-exact: a pure function of the float values and the edges. Row values, masks and outputs
 * may be chunked with matching ordered chunk lengths. Each chunk must fit one storage binding;
 * total rows fit the shared uint32 chunk limit. Breaks and classCount stay packed.
 */
export class GPUClassAssignment<
  Props extends GPUClassAssignmentProps<GraphDataView | GraphVectorView> = GPUClassAssignmentProps
> implements GPUCommandNodeProducer
{
  /** Prefix for every node ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: Props & GPUClassAssignmentProps<Props['values']>;
  /** Canonical classified-values port consumed by distribution and transition workflows. */
  readonly classifiedValues: GPUClassifiedValuesPort & {
    classes: GPUClassAssignmentProps<Props['values']>['output'];
    mask?: GPUClassAssignmentProps<Props['values']>['mask'];
  };

  readonly estimatedCommandNodeCount: number;
  private readonly chunkPlan: GraphChunkDispatchPlan;

  constructor(props: Props) {
    const id = props.id ?? 'class-assignment';
    this.id = id;
    this.props = props as Props & GPUClassAssignmentProps<Props['values']>;
    this.classifiedValues = {
      classes: this.props.output,
      mask: this.props.mask,
      classCount: props.classCount
    };
    const intervalClosure = props.intervalClosure ?? 'left';
    validateRequirement(
      id,
      intervalClosure === 'left' || intervalClosure === 'right',
      "intervalClosure must be 'left' or 'right'"
    );
    for (const chunk of getGraphViewChunks(props.values))
      validateSinglePackedView(chunk, ['float32'], `${id} values`);
    validateSinglePackedView(props.breaks, ['float32'], `${id} breaks`);
    validatePackedUint32View(props.classCount, `${id} classCount`);
    for (const chunk of getGraphViewChunks(props.output))
      validatePackedUint32View(chunk, `${id} output`);
    if (props.classCount.length < 1) {
      throw new Error(`${id} classCount must hold one row`);
    }
    if (props.breaks.length < 2) {
      throw new Error(`${id} breaks must hold at least two rows`);
    }
    if (props.output.length !== props.values.length) {
      throw new Error(`${id} output length must equal values length`);
    }
    if (props.mask) {
      for (const chunk of getGraphViewChunks(props.mask))
        validatePackedUint32View(chunk, `${id} mask`);
      if (props.mask.length !== props.values.length) {
        throw new Error(`${id} mask length must equal values length`);
      }
    }
    validateGraphOutputsDisjointFromInputs(
      id,
      [props.output],
      [props.values, props.breaks, props.classCount, props.mask]
    );
    validateDisjointWritableViews(id, {output: props.output});
    this.chunkPlan = planGraphChunkDispatch(id, {
      values: props.values,
      mask: props.mask,
      output: props.output
    });
    this.estimatedCommandNodeCount = this.chunkPlan.nonemptyChunkCount;
    validateCommandNodeBudget(id, this.estimatedCommandNodeCount);
  }

  /** Returns the assignment node. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, chunkPlan} = this;
    const {breaks, classCount, mask} = props;
    validateGraphChunkDispatchBindings(graph, id, chunkPlan);
    validateGraphChunkBindingRanges(graph, id, {breaks, classCount});
    validateGraphChunkGroupKernelSupport(graph, id, 4 + Number(Boolean(mask)), WORKGROUP_SIZE);
    for (const chunk of chunkPlan.chunks) {
      if (chunk.length > 0)
        getBoundedDispatchLayout(
          id,
          chunk.length,
          WORKGROUP_SIZE,
          graph.device.limits.maxComputeWorkgroupsPerDimension
        );
    }

    // 'left': an edge at or below the value is below it. 'right': only an edge strictly below it.
    const isRightClosed = props.intervalClosure === 'right';
    const edgeIsBelowWGSL = isRightClosed
      ? 'breaks[breaksOffset + edge] < value'
      : 'breaks[breaksOffset + edge] <= value';
    return [
      ...chunkPlan.chunks
        .filter(chunk => chunk.length > 0)
        .map(chunk =>
          createGraphChunkGroupKernelNode<Parameters>(graph, {
            id: `${id}-assign-chunk-${chunk.chunkIndex}`,
            operation: OPERATION,
            ...(isRightClosed ? {variant: 'right-closed'} : {}),
            bindings: [
              {name: 'values', view: chunk.views['values'], type: 'f32', access: 'read'},
              {name: 'breaks', view: breaks, type: 'f32', access: 'read'},
              {name: 'classCount', view: classCount, type: 'u32', access: 'read'},
              ...(mask
                ? [
                    {
                      name: 'mask',
                      view: chunk.views['mask'],
                      type: 'u32' as const,
                      access: 'read' as const
                    }
                  ]
                : []),
              {name: 'output', view: chunk.views['output'], type: 'u32', access: 'read_write'}
            ] as WGSLKernelBinding[],
            invocationCount: chunk.length,
            workgroupSize: WORKGROUP_SIZE,
            body: `if (index >= getChunkParameter(1u)) { return; }
  let valuesBase = arrayLength(&values) - getChunkParameter(1u);
  let outputBase = arrayLength(&output) - getChunkParameter(1u);
  ${mask ? 'let maskBase = arrayLength(&mask) - getChunkParameter(1u);' : ''}
  let value = values[valuesBase + index];
  let count = min(classCount[classCountOffset], arrayLength(&breaks) - breaksOffset - 1u);
  var assigned = ${GPU_CLASS_ASSIGNMENT_NO_CLASS}u;
  // WGSL gives no NaN guarantees to comparisons, so test the exponent and mantissa bits.
  let isNaN = (bitcast<u32>(value) & 0x7fffffffu) > 0x7f800000u;
  if (count > 0u && !isNaN && ${mask ? 'mask[maskBase + index] != 0u' : 'true'}) {
    var below = 0u;
    for (var edge = 1u; edge < count; edge++) {
      if (${edgeIsBelowWGSL}) {
        below++;
      }
    }
    assigned = below;
  }
  output[outputBase + index] = assigned;`
          })
        )
    ];
  }
}

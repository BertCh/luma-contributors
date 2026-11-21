// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  captureDataframeConstructorProps,
  type DataframePropKeys
} from '../calendar-buckets/dataframe-constructor-contracts';

import {
  createGraphChunkGroupKernelNode,
  validateGraphChunkGroupKernelSupport,
  GRAPH_CHUNK_GROUP_EXTRA_PARAMETER_OFFSET
} from '../../analysis-core/kernels/graph-chunk-groups';

import {
  getBoundedDispatchLayout,
  type GraphVectorView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  resolveGPUTemporalContext,
  validateGPUTemporalContext,
  type GPUTemporalContext,
  type GPUTemporalContextSupport
} from '../../analysis-core/contracts/temporal-context';
import {
  getGraphViewChunks,
  validateCommandNodeBudget,
  validateGraphOutputsDisjointFromInputs,
  validateSinglePackedView,
  validateRequirement,
  validateViewLength
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  planGraphChunkDispatch,
  validateDisjointWritableViews,
  validateGraphChunkDispatchBindings,
  type GraphChunkDispatchPlan
} from '../../analysis-core/kernels/graph-chunk-dispatch';
import {type WGSLKernelBinding} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {TIME_WORDS_WGSL} from '../time-window-filter/time-words';

const OPERATION = 'GPUTimeWordsRelative';
/** Fixed workgroup size keeps all row chunks on the same pipeline specialization. */
const WORKGROUP_SIZE = 256;
const TEMPORAL_CONTEXT_SUPPORT: GPUTemporalContextSupport = Object.freeze({
  encodings: Object.freeze(['int64-words'] as const)
});
const INT64_MAXIMUM = (1n << 63n) - 1n;
const INT64_MINIMUM = -(1n << 63n);

/** Preserves packed row types by default and the caller's chunked view kind when requested. */
type GPUTimeWordsRelativeRowView<
  Format extends GraphDataView['format'],
  Rows extends GraphDataView | GraphVectorView
> = Rows extends GraphVectorView ? GraphVectorView<Format> : GraphDataView<Format>;

/** Properties for {@link GPUTimeWordsRelative}. */
export type GPUTimeWordsRelativeProps<
  Rows extends GraphDataView | GraphVectorView = GraphDataView
> = {
  /** Prefix for generated node IDs. Defaults to `'time-words-relative'`. */
  id?: string;
  /** Int64 words, `(low, high)` per row, counted from the epoch of `temporalContext`. */
  timestamps: GPUTimeWordsRelativeRowView<'uint32x2', Rows>;
  /** Optional chunk-aligned uint32 mask; zero writes NaN for a missing time. */
  mask?: GPUTimeWordsRelativeRowView<'uint32', Rows>;
  /** Temporal context of `timestamps`. It must be `'int64-words'`; any unit and epoch. */
  temporalContext: GPUTemporalContext;
  /**
   * The instant the output counts from, as signed Int64 ticks of the context unit after the Unix
   * epoch (compile-time). It becomes the `epoch` of {@link GPUTimeWordsRelative.temporalContext}.
   */
  origin: bigint;
  /** Caller-owned `float32` column with one row per timestamp: ticks after `origin`. */
  output: GPUTimeWordsRelativeRowView<'float32', Rows>;
};

/**
 * Turns Int64 word times into `float32` ticks after an origin on the GPU, so they can feed
 * contributors that read relative times (`GPUKnoxTest`, `GPUMantelTest`, `GPUSpaceTimeVariogram`,
 * float-mode `GPUTemporalReduction` and the float trajectory paths) with no readback and no CPU
 * subtraction.
 *
 * The difference to the origin is taken exactly on the words with `timeWordsDifferenceToF32` and
 * only then rounded: the result is exact for `|time - origin| < 2^24` ticks and has a relative
 * error below 2^-23 beyond, with the right sign for every Int64 pair (times before the origin are
 * negative). Choose the origin near the data and the unit coarse enough that the span stays within
 * 2^24 ticks when exact ticks matter.
 *
 * The context of the output is `temporalContext` with `encoding: 'float32-relative'` and
 * `epoch: origin`; its unit, time zone and axis are the input's. Nothing is converted between
 * units. A zero mask row writes NaN; downstream algorithms that require finite times must
 * compact these missing rows first. Without a mask every word pair, including zero, is a time.
 *
 * Per-frame (no recompile): the contents of `timestamps` and `output`. Topology (needs a new
 * graph): the row count and chunk boundaries, `origin` and `temporalContext`. Row inputs and
 * outputs and the optional missing-time mask may be chunked with matching ordered chunk lengths. Each chunk must fit one storage
 * binding; the total row count is bounded by the shared uint32 chunk limit.
 */
export class GPUTimeWordsRelative<
  Props extends GPUTimeWordsRelativeProps<
    GraphDataView | GraphVectorView
  > = GPUTimeWordsRelativeProps
> implements GPUCommandNodeProducer
{
  /** Prefix for every node ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: Props & GPUTimeWordsRelativeProps<Props['timestamps']>;
  /** Number of rows. */
  readonly rowCount: number;
  /** Context of `output`: the input's with `encoding: 'float32-relative'` and `epoch: origin`. */
  readonly outputTemporalContext: GPUTemporalContext;
  /** `origin` as ticks after the input epoch, the words subtracted from every time. */
  private readonly offsetTicks: bigint;

  readonly estimatedCommandNodeCount: number;
  private readonly chunkPlan: GraphChunkDispatchPlan;

  constructor(
    input: Props &
      Record<
        Exclude<
          DataframePropKeys<Props>,
          DataframePropKeys<GPUTimeWordsRelativeProps<GraphDataView | GraphVectorView>>
        >,
        never
      >
  ) {
    const props = captureDataframeConstructorProps(input.id ?? 'time-words-relative', input, [
      'id',
      'timestamps',
      'mask',
      'temporalContext',
      'origin',
      'output'
    ]);
    this.id = props.id ?? 'time-words-relative';
    this.props = props as Props & GPUTimeWordsRelativeProps<Props['timestamps']>;
    const id = this.id;
    const {timestamps, output, origin, mask} = props;
    const inputContext = resolveGPUTemporalContext(
      id,
      props.temporalContext,
      TEMPORAL_CONTEXT_SUPPORT
    );
    validateRequirement(
      id,
      typeof origin === 'bigint' && origin >= INT64_MINIMUM && origin <= INT64_MAXIMUM,
      'origin must be a signed Int64 bigint'
    );
    // The words count from the input epoch, so the origin in those words is origin - epoch.
    this.offsetTicks = origin - (inputContext.epoch === 'unix' ? 0n : inputContext.epoch);
    validateRequirement(
      id,
      this.offsetTicks >= INT64_MINIMUM && this.offsetTicks <= INT64_MAXIMUM,
      'origin lies more than 2^63 ticks from the input epoch'
    );
    this.outputTemporalContext = {...inputContext, encoding: 'float32-relative', epoch: origin};
    validateGPUTemporalContext(id, this.outputTemporalContext);
    for (const chunk of getGraphViewChunks(timestamps))
      validateSinglePackedView(chunk, ['uint32x2'], `${id} timestamps`);
    for (const chunk of getGraphViewChunks(output))
      validateSinglePackedView(chunk, ['float32'], `${id} output`);
    this.rowCount = timestamps.length;
    validateRequirement(id, this.rowCount >= 1, 'needs at least one row');
    validateViewLength(id, 'output', output, this.rowCount);
    if (mask) {
      for (const chunk of getGraphViewChunks(mask))
        validateSinglePackedView(chunk, ['uint32'], `${id} mask`);
      validateViewLength(id, 'mask', mask, this.rowCount);
    }
    validateGraphOutputsDisjointFromInputs(id, [output], [timestamps, mask]);
    validateDisjointWritableViews(id, {output});
    this.chunkPlan = planGraphChunkDispatch(id, {timestamps, output, mask});
    this.estimatedCommandNodeCount = this.chunkPlan.nonemptyChunkCount;
    validateCommandNodeBudget(id, this.estimatedCommandNodeCount);
  }

  /** Returns the one conversion kernel. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, offsetTicks, chunkPlan} = this;
    validateGraphChunkDispatchBindings(graph, id, chunkPlan);
    validateGraphChunkGroupKernelSupport(
      graph,
      id,
      2 + Number(Boolean(this.props.mask)),
      WORKGROUP_SIZE
    );
    for (const chunk of chunkPlan.chunks) {
      if (chunk.length > 0)
        getBoundedDispatchLayout(
          id,
          chunk.length,
          WORKGROUP_SIZE,
          graph.device.limits.maxComputeWorkgroupsPerDimension
        );
    }

    const unsigned = BigInt.asUintN(64, offsetTicks);
    return [
      ...chunkPlan.chunks
        .filter(chunk => chunk.length > 0)
        .map(chunk =>
          createGraphChunkGroupKernelNode<Parameters>(graph, {
            id: `${id}-convert-chunk-${chunk.chunkIndex}`,
            operation: OPERATION,
            bindings: [
              {name: 'timestamps', view: chunk.views['timestamps'], type: 'u32', access: 'read'},
              ...(this.props.mask
                ? [{name: 'timeValidity', view: chunk.views['mask'], type: 'u32', access: 'read'}]
                : []),
              {name: 'relative', view: chunk.views['output'], type: 'f32', access: 'read_write'}
            ] as WGSLKernelBinding[],
            invocationCount: chunk.length,
            workgroupSize: WORKGROUP_SIZE,
            parameterValues: [Number(unsigned & 0xffffffffn), Number(unsigned >> 32n)],
            declarations: TIME_WORDS_WGSL,
            body: `if (index >= getChunkParameter(1u)) { return; }
  let timestampsBase = timestampsOffset;
  let relativeBase = relativeOffset;
  ${
    this.props.mask
      ? `let validityBase = timeValidityOffset;
  if (timeValidity[validityBase + index] == 0u) {
    relative[relativeBase + index] = bitcast<f32>((index & 0u) | 0x7fc00000u);
    return;
  }`
      : ''
  }
  let time = vec2<u32>(timestamps[timestampsBase + 2u * index], timestamps[timestampsBase + 2u * index + 1u]);
  relative[relativeBase + index] = timeWordsDifferenceToF32(time, vec2<u32>(getChunkParameter(${GRAPH_CHUNK_GROUP_EXTRA_PARAMETER_OFFSET}u), getChunkParameter(${GRAPH_CHUNK_GROUP_EXTRA_PARAMETER_OFFSET + 1}u)));`
          })
        )
    ];
  }
}

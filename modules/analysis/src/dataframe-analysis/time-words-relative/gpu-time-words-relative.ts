// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
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
  validateGPUStorageBindingByteLength,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validateSinglePackedView,
  validateRequirement,
  validateViewLength
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {TIME_WORDS_WGSL} from '../time-window-filter/time-words';

const OPERATION = 'GPUTimeWordsRelative';
const TEMPORAL_CONTEXT_SUPPORT: GPUTemporalContextSupport = Object.freeze({
  encodings: Object.freeze(['int64-words'] as const)
});
const INT64_MAXIMUM = (1n << 63n) - 1n;
const INT64_MINIMUM = -(1n << 63n);

/** Properties for {@link GPUTimeWordsRelative}. */
export type GPUTimeWordsRelativeProps = {
  /** Prefix for generated node IDs. Defaults to `'time-words-relative'`. */
  id?: string;
  /** Int64 words, `(low, high)` per row, counted from the epoch of `temporalContext`. */
  timestamps: GraphDataView<'uint32x2'>;
  /** Temporal context of `timestamps`. It must be `'int64-words'`; any unit and epoch. */
  temporalContext: GPUTemporalContext;
  /**
   * The instant the output counts from, as signed Int64 ticks of the context unit after the Unix
   * epoch (compile-time). It becomes the `epoch` of {@link GPUTimeWordsRelative.temporalContext}.
   */
  origin: bigint;
  /** Caller-owned `float32` column with one row per timestamp: ticks after `origin`. */
  output: GraphDataView<'float32'>;
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
 * units.
 *
 * Per-frame (no recompile): the contents of `timestamps` and `output`. Topology (needs a new
 * graph): the row count, `origin` and `temporalContext`. The `uint32x2` timestamps must fit one
 * storage binding (16,777,216 rows on a default device).
 */
export class GPUTimeWordsRelative implements GPUCommandNodeProducer {
  /** Prefix for every node ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUTimeWordsRelativeProps;
  /** Number of rows. */
  readonly rowCount: number;
  /** Context of `output`: the input's with `encoding: 'float32-relative'` and `epoch: origin`. */
  readonly outputTemporalContext: GPUTemporalContext;
  /** `origin` as ticks after the input epoch, the words subtracted from every time. */
  private readonly offsetTicks: bigint;

  constructor(props: GPUTimeWordsRelativeProps) {
    this.id = props.id ?? 'time-words-relative';
    this.props = props;
    const id = this.id;
    const {timestamps, output, origin} = props;
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
    validateSinglePackedView(timestamps, ['uint32x2'], `${id} timestamps`);
    validateSinglePackedView(output, ['float32'], `${id} output`);
    this.rowCount = timestamps.length;
    validateRequirement(id, this.rowCount >= 1, 'needs at least one row');
    validateViewLength(id, 'output', output, this.rowCount);
    validateGraphOutputsDisjointFromInputs(id, [output], [timestamps]);
  }

  /** Returns the one conversion kernel. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {props, id, rowCount, offsetTicks} = this;
    const {timestamps, output} = props;
    validateGraphViewsBelongToGraph(id, graph, [timestamps, output]);
    validateGPUStorageBindingByteLength(
      graph.device,
      id,
      'timestamps',
      rowCount * 8,
      'convert fewer rows per graph'
    );
    const unsigned = BigInt.asUintN(64, offsetTicks);
    return [
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-convert`,
        operation: OPERATION,
        bindings: [
          {name: 'timestamps', view: timestamps, type: 'u32', access: 'read'},
          {name: 'relative', view: output, type: 'f32', access: 'read_write'}
        ],
        invocationCount: rowCount,
        // The origin words are overrides so that graphs differing only in origin share a module.
        declarations: `${TIME_WORDS_WGSL}
override ORIGIN_LOW: u32;
override ORIGIN_HIGH: u32;`,
        constants: {
          ORIGIN_LOW: Number(unsigned & 0xffffffffn),
          ORIGIN_HIGH: Number(unsigned >> 32n)
        },
        body: `let time = vec2<u32>(timestamps[timestampsOffset + 2u * index], timestamps[timestampsOffset + 2u * index + 1u]);
  relative[relativeOffset + index] = timeWordsDifferenceToF32(time, vec2<u32>(ORIGIN_LOW, ORIGIN_HIGH));`
      })
    ];
  }
}

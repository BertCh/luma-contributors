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
  validateGPUStatusPort,
  type GPUInvalidStatusPort
} from '../../analysis-core/contracts/status';
import {
  validateCapacity,
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
  validateGraphChunkBindingRanges,
  validateGraphChunkDispatchBindings,
  type GraphChunkDispatchPlan
} from '../../analysis-core/kernels/graph-chunk-dispatch';
import {
  createFillNode,
  type WGSLKernelBinding
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  getGPUCellPeriodKeyPeriodBits,
  GPU_CELL_PERIOD_KEY_MAXIMUM_PERIOD_BITS,
  type GPUCellPeriodKeyLayout,
  type GPUCellPeriodKeyFill
} from './cell-period-key-packing';

const OPERATION = 'GPUCellPeriodKeys';
/** Fixed workgroup size keeps all row chunks on the same pipeline specialization. */
const WORKGROUP_SIZE = 256;
/** Periods are `uint32` and the all-ones period is the "already excluded" marker of `GPUPeriodIndex`. */
const MAXIMUM_PERIOD_COUNT = 0xfffffffd;

/** Preserves packed row types by default and the caller's chunked view kind when requested. */
type GPUCellPeriodKeysRowView<
  Format extends GraphDataView['format'],
  Rows extends GraphDataView | GraphVectorView
> = Rows extends GraphVectorView ? GraphVectorView<Format> : GraphDataView<Format>;

/** Caller-owned outputs of {@link GPUCellPeriodKeys}. */
export type GPUCellPeriodKeysOutput<Rows extends GraphDataView | GraphVectorView = GraphDataView> =
  {
    /**
     * One packed `uint32x2` `(low, high)` key per input row, the `keys` input of `GPUGroupStatistics`.
     * A row that is not packed holds the reserved all-ones key, which `GPUGroupStatistics` skips.
     */
    keys: GPUCellPeriodKeysRowView<'uint32x2', Rows>;
    /**
     * `status.invalidCount`: rows that passed the mask and carry a period and a cell key, but were not
     * packed because the period is out of range or the low bits of the cell key do not hold the
     * `fill` pattern (a cell finer than the chosen resolution), or the encoded key would be the
     * reserved all-ones missing sentinel. Never a silent collision.
     */
    status?: GPUInvalidStatusPort;
  };

/**
 * Properties for {@link GPUCellPeriodKeys}.
 *
 * Per-frame (no recompile): the contents of `cells`, `periods` and `mask`. Topology (needs a new
 * graph): the row count, `periodCount`, `fill`, and whether `mask` and the status are present.
 */
export type GPUCellPeriodKeysProps<Rows extends GraphDataView | GraphVectorView = GraphDataView> = {
  /** Prefix for generated node IDs. Defaults to `'cell-period-keys'`. */
  id?: string;
  /**
   * 64-bit cell key per row as `uint32x2` little-endian `(low, high)` words (Arrow `Uint64`): H3,
   * S2 or Quadbin, for example the keys written by `GPUPointToCell`. The all-ones key means "no
   * cell" and is skipped silently.
   */
  cells: GPUCellPeriodKeysRowView<'uint32x2', Rows>;
  /**
   * Period per row, the `periods` output of `GPUPeriodIndex`. `0xffffffff` (a row excluded
   * upstream) is skipped silently; a period at or above `periodCount` is skipped and counted.
   */
  periods: GPUCellPeriodKeysRowView<'uint32', Rows>;
  /** Number of periods, 1 to `2^32 - 3`. The layout uses at most 32 period bits. */
  periodCount: number;
  /**
   * What the low bits reserved by the layout must hold: `'ones'` for H3 and
   * Quadbin, `'zeros'` for S2. See {@link GPUCellPeriodKeyFill}.
   */
  fill: GPUCellPeriodKeyFill;
  /** Explicit key layout. Identity requires periodCount = 1. */
  layout: GPUCellPeriodKeyLayout;
  /** Optional `uint32` row mask; zero skips the row silently. */
  mask?: GPUCellPeriodKeysRowView<'uint32', Rows>;
  /** Caller-owned outputs. */
  output: GPUCellPeriodKeysOutput<Rows>;
};

/**
 * Packs a 64-bit DGGS cell key and a period into one 64-bit group key, so that
 * `GPUGroupStatistics` (which groups by `uint32x2` keys) computes a sparse DGGS cell by period
 * cube: count, sum, mean, minimum, maximum, variance and percentiles per (cell, period) in a
 * capacity-bounded table sorted by cell, then period.
 *
 * Packing: explicitly select `layout: 'zero-period-bit-identity'` for one period to preserve every
 * cell bit (`b = 0`). Use `layout: 'legacy-one-bit'` for multiple periods or to match persisted
 * one-bit keys, such as cell 3 and period 0 stored as key 2 with 'ones' fill. Identity with more
 * than one period is rejected at construction. With `b = max(1, ceil(log2(periodCount)))` legacy
 * period bits, the key is `(cellKey & ~(2^b - 1)) |
 * period`. The period replaces the low `b` bits of the cell key. Those bits are not free in
 * general, so the contributor requires them to equal the `fill` pattern (all ones for H3 and
 * Quadbin, zeros for S2), which holds exactly for cells coarser than the resolution that leaves `b`
 * free bits: H3 `3 * (15 - resolution) >= b`, Quadbin `52 - 2 * zoom >= b`, S2 `60 - 2 * level >=
 * b`. Within that limit the packing is lossless and reversible
 * (`unpackGPUCellPeriodKeyChecked`): no information of the cell is dropped and no two (cell, period)
 * pairs share a key. A row is rejected when its encoded key would be the reserved all-ones
 * sentinel or the low `b` bits of its cell key differ from
 * the `fill` pattern, which is what a cell finer than the limit has in nearly every case: the row
 * is not packed (its key is all ones, so `GPUGroupStatistics` skips it) and is counted in
 * `output.status.invalidCount`. A finer cell whose extra bits happen to equal the fill pattern is
 * packed, and still unpacks to itself, because the replaced bits are then known. Bits are never
 * truncated silently and two different cells never share a packed key (the map from accepted cell
 * keys to packed keys is injective for each period). For example 365 daily periods need 9 bits:
 * H3 resolution 12 or coarser, Quadbin zoom 21 or coarser, S2 level 25 or coarser, are always
 * accepted.
 *
 * The packed key sorts by cell and then period (the cell bits are the high bits), so the group
 * table of `GPUGroupStatistics` is in cell-major order. A packed key is not a valid cell id; unpack
 * the group table keys with `unpackGPUCellPeriodKeyChecked`.
 *
 * Row inputs and keys may be chunked with matching ordered chunk lengths. Each chunk must fit
 * one storage binding; total rows fit the shared uint32 chunk limit. Status stays packed.
 *
 * Limits: `periodCount` at most `2^32 - 3`. Downstream, `GPUGroupStatistics` keeps at most
 * `getGPUGroupStatisticsMaximumGroupCount` (8,388,608 on a default device) distinct (cell, period)
 * groups. Its `sumValues` and `means` use a unit that follows the data, as `GPUValueCube` does;
 * only its `sums` words have the fixed resolution `1 / sumScale`.
 */
export class GPUCellPeriodKeys<
  Props extends GPUCellPeriodKeysProps<GraphDataView | GraphVectorView> = GPUCellPeriodKeysProps
> implements GPUCommandNodeProducer
{
  /** Prefix for every node ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: Props & GPUCellPeriodKeysProps<Props['cells']>;
  /** Number of rows. */
  readonly rowCount: number;
  /** Low key bits replaced by the period. */
  readonly periodBits: number;

  readonly estimatedCommandNodeCount: number;
  private readonly chunkPlan: GraphChunkDispatchPlan;

  constructor(
    input: Props &
      Record<
        Exclude<
          DataframePropKeys<Props>,
          DataframePropKeys<GPUCellPeriodKeysProps<GraphDataView | GraphVectorView>>
        >,
        never
      >
  ) {
    const props = captureDataframeConstructorProps(input.id ?? 'cell-period-keys', input, [
      'id',
      'cells',
      'periods',
      'periodCount',
      'fill',
      'layout',
      'mask',
      'output'
    ]);
    this.id = props.id ?? 'cell-period-keys';
    this.props = props as Props & GPUCellPeriodKeysProps<Props['cells']>;
    const id = this.id;
    const {cells, periods, mask, output} = props;
    validateCapacity(id, 'periodCount', props.periodCount, 1, MAXIMUM_PERIOD_COUNT);
    validateRequirement(
      id,
      props.layout === 'legacy-one-bit' || props.layout === 'zero-period-bit-identity',
      `layout must be 'legacy-one-bit' or 'zero-period-bit-identity'; received ${props.layout}`
    );
    validateRequirement(
      id,
      props.layout !== 'zero-period-bit-identity' || props.periodCount === 1,
      `layout 'zero-period-bit-identity' requires periodCount = 1; received ${props.periodCount}`
    );
    this.periodBits = getGPUCellPeriodKeyPeriodBits(props.periodCount, props.layout);
    // Always true while the periodCount limit holds; stated because the kernel reads the low word only.
    validateRequirement(
      id,
      this.periodBits <= GPU_CELL_PERIOD_KEY_MAXIMUM_PERIOD_BITS,
      'periodCount needs at most 32 key bits'
    );
    validateRequirement(
      id,
      props.fill === 'ones' || props.fill === 'zeros',
      "fill must be 'ones' or 'zeros'"
    );
    for (const chunk of getGraphViewChunks(cells))
      validateSinglePackedView(chunk, ['uint32x2'], `${id} cells`);
    for (const chunk of getGraphViewChunks(periods))
      validateSinglePackedView(chunk, ['uint32'], `${id} periods`);
    this.rowCount = cells.length;
    validateRequirement(id, this.rowCount >= 1, 'needs at least one row');
    validateViewLength(id, 'periods', periods, this.rowCount);
    if (mask) {
      for (const chunk of getGraphViewChunks(mask))
        validateSinglePackedView(chunk, ['uint32'], `${id} mask`);
      validateViewLength(id, 'mask', mask, this.rowCount);
    }
    for (const chunk of getGraphViewChunks(output.keys))
      validateSinglePackedView(chunk, ['uint32x2'], `${id} output.keys`);
    validateViewLength(id, 'output.keys', output.keys, this.rowCount);
    validateGPUStatusPort(id, {invalidCount: output.status?.invalidCount});
    validateGraphOutputsDisjointFromInputs(id, this.getOutputViews(), this.getInputViews());
    validateDisjointWritableViews(id, {
      keys: output.keys,
      invalidCount: output.status?.invalidCount
    });
    this.chunkPlan = planGraphChunkDispatch(id, {
      cells,
      periods,
      rowMask: mask,
      keysOut: output.keys
    });
    this.estimatedCommandNodeCount =
      Number(Boolean(output.status?.invalidCount)) + this.chunkPlan.nonemptyChunkCount;
    validateCommandNodeBudget(id, this.estimatedCommandNodeCount);
  }

  /** Returns the optional status clear and the one packing kernel. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {props, id, periodBits, chunkPlan} = this;
    const {mask, output, periodCount, fill} = props;
    validateGraphChunkDispatchBindings(graph, id, chunkPlan);
    validateGraphChunkBindingRanges(graph, id, {invalidCount: output.status?.invalidCount});
    validateGraphChunkGroupKernelSupport(
      graph,
      id,
      3 + Number(Boolean(mask)) + Number(Boolean(output.status?.invalidCount)),
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

    const invalidCount = output.status?.invalidCount;
    const nodes: GPUCommandNode<Parameters>[] = [];
    if (invalidCount) {
      nodes.push(
        createFillNode<Parameters>(graph, {
          id: `${id}-clear-invalid-count`,
          operation: OPERATION,
          view: invalidCount,
          componentCount: 1,
          type: 'u32',
          value: '0u'
        })
      );
    }
    const periodMask = periodBits === 32 ? 0xffffffff : 2 ** periodBits - 1;
    for (const chunk of chunkPlan.chunks) {
      if (chunk.length === 0) continue;
      const bindings: WGSLKernelBinding[] = [
        {name: 'cells', view: chunk.views['cells'], type: 'u32', access: 'read'},
        {name: 'periods', view: chunk.views['periods'], type: 'u32', access: 'read'},
        ...(mask
          ? [
              {
                name: 'rowMask',
                view: chunk.views['rowMask'],
                type: 'u32' as const,
                access: 'read' as const
              }
            ]
          : []),
        {name: 'keysOut', view: chunk.views['keysOut'], type: 'u32', access: 'read_write'},
        ...(invalidCount
          ? [
              {
                name: 'invalidCountOut',
                view: invalidCount,
                type: 'atomic<u32>' as const,
                access: 'read_write' as const
              }
            ]
          : [])
      ];
      nodes.push(
        createGraphChunkGroupKernelNode<Parameters>(graph, {
          id: `${id}-pack-chunk-${chunk.chunkIndex}`,
          operation: OPERATION,
          bindings: bindings,
          invocationCount: chunk.length,
          workgroupSize: WORKGROUP_SIZE,
          parameterValues: [periodMask, fill === 'ones' ? periodMask : 0, periodCount],
          body: `let periodMask = getChunkParameter(${GRAPH_CHUNK_GROUP_EXTRA_PARAMETER_OFFSET}u);
  let expectedFill = getChunkParameter(${GRAPH_CHUNK_GROUP_EXTRA_PARAMETER_OFFSET + 1}u);
  let periodCount = getChunkParameter(${GRAPH_CHUNK_GROUP_EXTRA_PARAMETER_OFFSET + 2}u);
  if (index >= getChunkParameter(1u)) { return; }
  let cellsBase = arrayLength(&cells) - 2u * getChunkParameter(1u);
  let periodsBase = arrayLength(&periods) - getChunkParameter(1u);
  let keysOutBase = arrayLength(&keysOut) - 2u * getChunkParameter(1u);
  ${mask ? 'let rowMaskBase = arrayLength(&rowMask) - getChunkParameter(1u);' : ''}
  let low = cells[cellsBase + 2u * index];
  let high = cells[cellsBase + 2u * index + 1u];
  let period = periods[periodsBase + index];
  var keyLow = 0xffffffffu;
  var keyHigh = 0xffffffffu;
  let isCell = !(low == 0xffffffffu && high == 0xffffffffu);
  if (isCell && period != 0xffffffffu && ${mask ? 'rowMask[rowMaskBase + index] != 0u' : 'true'}) {
    let packedLow = (low & ~periodMask) | period;
    let isPackedMissing = packedLow == 0xffffffffu && high == 0xffffffffu;
    if (period < periodCount && (low & periodMask) == expectedFill && !isPackedMissing) {
      keyLow = (low & ~periodMask) | period;
      keyHigh = high;
    }${
      invalidCount
        ? ` else {
      atomicAdd(&invalidCountOut[invalidCountOutOffset], 1u);
    }`
        : ''
    }
  }
  keysOut[keysOutBase + 2u * index] = keyLow;
  keysOut[keysOutBase + 2u * index + 1u] = keyHigh;`
        })
      );
    }
    return nodes;
  }

  private getInputViews(): (GraphDataView | GraphVectorView)[] {
    const {cells, periods, mask} = this.props;
    return [cells, periods, mask].filter(view => Boolean(view)) as (
      | GraphDataView
      | GraphVectorView
    )[];
  }

  private getOutputViews(): (GraphDataView | GraphVectorView)[] {
    const {output} = this.props;
    return [output.keys, output.status?.invalidCount].filter(view => Boolean(view)) as (
      | GraphDataView
      | GraphVectorView
    )[];
  }
}

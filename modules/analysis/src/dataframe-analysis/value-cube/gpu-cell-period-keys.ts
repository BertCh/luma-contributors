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
  validateGPUStatusPort,
  type GPUInvalidStatusPort
} from '../../analysis-core/contracts/status';
import {
  validateCapacity,
  validateGPUStorageBindingByteLength,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validateSinglePackedView,
  validateRequirement,
  validateViewLength
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  createFillNode,
  createWGSLKernelNode,
  type WGSLKernelBinding
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  getGPUCellPeriodKeyPeriodBits,
  GPU_CELL_PERIOD_KEY_MAXIMUM_PERIOD_BITS,
  type GPUCellPeriodKeyFill
} from './cell-period-key-packing';

const OPERATION = 'GPUCellPeriodKeys';
/** Periods are `uint32` and the all-ones period is the "already excluded" marker of `GPUPeriodIndex`. */
const MAXIMUM_PERIOD_COUNT = 0xfffffffd;

/** Caller-owned outputs of {@link GPUCellPeriodKeys}. */
export type GPUCellPeriodKeysOutput = {
  /**
   * One packed `uint32x2` `(low, high)` key per input row, the `keys` input of `GPUGroupStatistics`.
   * A row that is not packed holds the reserved all-ones key, which `GPUGroupStatistics` skips.
   */
  keys: GraphDataView<'uint32x2'>;
  /**
   * `status.invalidCount`: rows that passed the mask and carry a period and a cell key, but were not
   * packed because the period is out of range or the low bits of the cell key do not hold the
   * `fill` pattern (a cell finer than the chosen resolution). Never a silent collision.
   */
  status?: GPUInvalidStatusPort;
};

/**
 * Properties for {@link GPUCellPeriodKeys}.
 *
 * Per-frame (no recompile): the contents of `cells`, `periods` and `mask`. Topology (needs a new
 * graph): the row count, `periodCount`, `fill`, and whether `mask` and the status are present.
 */
export type GPUCellPeriodKeysProps = {
  /** Prefix for generated node IDs. Defaults to `'cell-period-keys'`. */
  id?: string;
  /**
   * 64-bit cell key per row as `uint32x2` little-endian `(low, high)` words (Arrow `Uint64`): H3,
   * S2 or Quadbin, for example the keys written by `GPUPointToCell`. The all-ones key means "no
   * cell" and is skipped silently.
   */
  cells: GraphDataView<'uint32x2'>;
  /**
   * Period per row, the `periods` output of `GPUPeriodIndex`. `0xffffffff` (a row excluded
   * upstream) is skipped silently; a period at or above `periodCount` is skipped and counted.
   */
  periods: GraphDataView<'uint32'>;
  /** Number of periods, 1 to `2^32 - 3` (the period needs `ceil(log2(periodCount))` key bits, at most 32). */
  periodCount: number;
  /**
   * What the low `ceil(log2(periodCount))` bits of every cell key must hold: `'ones'` for H3 and
   * Quadbin, `'zeros'` for S2. See {@link GPUCellPeriodKeyFill}.
   */
  fill: GPUCellPeriodKeyFill;
  /** Optional `uint32` row mask; zero skips the row silently. */
  mask?: GraphDataView<'uint32'>;
  /** Caller-owned outputs. */
  output: GPUCellPeriodKeysOutput;
};

/**
 * Packs a 64-bit DGGS cell key and a period into one 64-bit group key, so that
 * `GPUGroupStatistics` (which groups by `uint32x2` keys) computes a sparse DGGS cell by period
 * cube: count, sum, mean, minimum, maximum, variance and percentiles per (cell, period) in a
 * capacity-bounded table sorted by cell, then period.
 *
 * Packing: with `b = ceil(log2(periodCount))` period bits, the key is `(cellKey & ~(2^b - 1)) |
 * period`. The period replaces the low `b` bits of the cell key. Those bits are not free in
 * general, so the contributor requires them to equal the `fill` pattern (all ones for H3 and
 * Quadbin, zeros for S2), which holds exactly for cells coarser than the resolution that leaves `b`
 * free bits: H3 `3 * (15 - resolution) >= b`, Quadbin `52 - 2 * zoom >= b`, S2 `60 - 2 * level >=
 * b`. Within that limit the packing is lossless and reversible
 * (`unpackGPUCellPeriodKey`): no information of the cell is dropped and no two (cell, period)
 * pairs share a key. A row is rejected exactly when the low `b` bits of its cell key differ from
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
 * the group table keys with `unpackGPUCellPeriodKey`.
 *
 * Limits: `periodCount` at most `2^32 - 3`. Downstream, `GPUGroupStatistics` keeps at most
 * `getGPUGroupStatisticsMaximumGroupCount` (8,388,608 on a default device) distinct (cell, period)
 * groups. Its `sumValues` and `means` use a unit that follows the data, as `GPUValueCube` does;
 * only its `sums` words have the fixed resolution `1 / sumScale`.
 */
export class GPUCellPeriodKeys implements GPUCommandNodeProducer {
  /** Prefix for every node ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUCellPeriodKeysProps;
  /** Number of rows. */
  readonly rowCount: number;
  /** Low key bits replaced by the period. */
  readonly periodBits: number;

  constructor(props: GPUCellPeriodKeysProps) {
    this.id = props.id ?? 'cell-period-keys';
    this.props = props;
    const id = this.id;
    const {cells, periods, mask, output} = props;
    validateCapacity(id, 'periodCount', props.periodCount, 1, MAXIMUM_PERIOD_COUNT);
    this.periodBits = getGPUCellPeriodKeyPeriodBits(props.periodCount);
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
    validateSinglePackedView(cells, ['uint32x2'], `${id} cells`);
    validateSinglePackedView(periods, ['uint32'], `${id} periods`);
    this.rowCount = cells.length;
    validateRequirement(id, this.rowCount >= 1, 'needs at least one row');
    validateViewLength(id, 'periods', periods, this.rowCount);
    if (mask) {
      validateSinglePackedView(mask, ['uint32'], `${id} mask`);
      validateViewLength(id, 'mask', mask, this.rowCount);
    }
    validateSinglePackedView(output.keys, ['uint32x2'], `${id} output.keys`);
    validateViewLength(id, 'output.keys', output.keys, this.rowCount);
    validateGPUStatusPort(id, {invalidCount: output.status?.invalidCount});
    validateGraphOutputsDisjointFromInputs(id, this.getOutputViews(), this.getInputViews());
    if (output.status?.invalidCount?.buffer === output.keys.buffer) {
      throw new Error(`${id} outputs must not share buffers with each other`);
    }
  }

  /** Returns the optional status clear and the one packing kernel. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {props, id, rowCount, periodBits} = this;
    const {cells, periods, mask, output, periodCount, fill} = props;
    validateGraphViewsBelongToGraph(id, graph, [...this.getInputViews(), ...this.getOutputViews()]);
    validateGPUStorageBindingByteLength(
      graph.device,
      id,
      'cells',
      rowCount * 8,
      'split the rows over several graphs'
    );
    const invalidCount = output.status?.invalidCount;
    const nodes: GPUCommandNode<Parameters>[] = [];
    if (invalidCount) {
      nodes.push(
        createFillNode<Parameters>(graph, {
          id: `${id}-clear-invalid-count`,
          operation: OPERATION,
          view: invalidCount,
          type: 'u32',
          value: '0u'
        })
      );
    }
    const periodMask = periodBits === 32 ? 0xffffffff : 2 ** periodBits - 1;
    const bindings: WGSLKernelBinding[] = [
      {name: 'cells', view: cells, type: 'u32', access: 'read'},
      {name: 'periods', view: periods, type: 'u32', access: 'read'},
      ...(mask
        ? [{name: 'rowMask', view: mask, type: 'u32' as const, access: 'read' as const}]
        : []),
      {name: 'keysOut', view: output.keys, type: 'u32', access: 'read_write'},
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
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-pack`,
        operation: OPERATION,
        bindings,
        invocationCount: rowCount,
        // Shape values are overrides so that cubes of different shapes share one shader module.
        declarations: `override PERIOD_MASK: u32;
override EXPECTED_FILL: u32;
override PERIOD_COUNT: u32;`,
        constants: {
          PERIOD_MASK: periodMask,
          EXPECTED_FILL: fill === 'ones' ? periodMask : 0,
          PERIOD_COUNT: periodCount
        },
        body: `let low = cells[cellsOffset + 2u * index];
  let high = cells[cellsOffset + 2u * index + 1u];
  let period = periods[periodsOffset + index];
  var keyLow = 0xffffffffu;
  var keyHigh = 0xffffffffu;
  let isCell = !(low == 0xffffffffu && high == 0xffffffffu);
  if (isCell && period != 0xffffffffu && ${mask ? 'rowMask[rowMaskOffset + index] != 0u' : 'true'}) {
    if (period < PERIOD_COUNT && (low & PERIOD_MASK) == EXPECTED_FILL) {
      keyLow = (low & ~PERIOD_MASK) | period;
      keyHigh = high;
    }${
      invalidCount
        ? ` else {
      atomicAdd(&invalidCountOut[invalidCountOutOffset], 1u);
    }`
        : ''
    }
  }
  keysOut[keysOutOffset + 2u * index] = keyLow;
  keysOut[keysOutOffset + 2u * index + 1u] = keyHigh;`
      })
    );
    return nodes;
  }

  private getInputViews(): GraphDataView[] {
    const {cells, periods, mask} = this.props;
    return [cells, periods, mask].filter(view => Boolean(view)) as GraphDataView[];
  }

  private getOutputViews(): GraphDataView[] {
    const {output} = this.props;
    return [output.keys, output.status?.invalidCount].filter(view =>
      Boolean(view)
    ) as GraphDataView[];
  }
}

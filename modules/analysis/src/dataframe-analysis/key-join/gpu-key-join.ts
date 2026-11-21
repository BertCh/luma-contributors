// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getSumOverflowNodes} from './sum-overflow-nodes';
import {FINITE_MEAN_WGSL} from './finite-mean-wgsl';
import {snapshotConstructionFields} from '../../analysis-core/kernels/construction-snapshot';

import {
  createTransientView,
  GPUScan,
  GPUUint32Gather,
  type GPUCommandGraph,
  type GPUCommandNode,
  GraphDataView,
  GraphVectorView
} from '@luma.gl/graph/gpu-core';
import {createStableLexicographicIndexSortNodes} from '../../analysis-core/kernels/stable-lexicographic-index-sort';
import type {GPUCompactOutput} from '../../analysis-core/kernels/gpu-contributor-types';
import type {GPUCommandNodeProducer} from '@luma.gl/graph/gpu-core';
import {
  createFillNode,
  createWGSLKernelNode,
  createPublishNode,
  type WGSLKernelBinding
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  validateGraphOutputsDisjointFromInputs,
  validateGPUStorageBindingByteLength,
  validateCapacity,
  validateMaximumCount,
  validateCompactOutput,
  validateSinglePackedView,
  validateRequirement,
  validateCommandNodeBudget
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  validateGraphChunkBindingRanges,
  validateDisjointWritableViews
} from '../../analysis-core/kernels/graph-chunk-dispatch';
import {
  getFixedPointWGSL,
  GPU_CELL_DEFAULT_SUM_SCALE,
  ORDERED_KEY_WGSL,
  validateSumScale
} from '../../analysis-core/kernels/cell-key-wgsl';
import {
  FIXED_POINT_WGSL,
  getFixedPointHeadroomBits
} from '../../analysis-core/kernels/fixed-point-accumulator';

import {createKeyLookupDistinctNodes} from './key-join-distinct';

const OPERATION = 'GPUKeyLookup';
/** Base key sorts and scans, including published radix hierarchy at the maximum supported row count. */
const BASE_COMMAND_NODE_ALLOWANCE = 2048;
/** Two full-width stable radix sorts plus distinct-key preparation, counting and publication. */
const DISTINCT_COMMAND_NODE_ALLOWANCE = 1056;
/** Clear, optional scale reduction, accumulation and finish per ordinary aggregate. */
const AGGREGATE_COMMAND_NODE_ALLOWANCE = 8;
/** Published GPUSort accepts at most 2^31 keys; reject larger metadata before scratch allocation. */
const MAXIMUM_RIGHT_ROW_COUNT = 0x80000000;
/** Left IDs and compacted row counts are uint32. */
const MAXIMUM_LEFT_ROW_COUNT = 0xffffffff;
const MAXIMUM_GATHER_COLUMNS_PER_KERNEL = 3;
/** Workgroup size of the pass that reduces the largest exponent of an aggregate column. */
const SUM_SCALE_WORKGROUP_SIZE = 256;

/** Operation of a 1:n {@link GPUKeyLookupAggregate} over all right rows matching a left row's key. */
export type GPUKeyLookupAggregateOperation =
  | 'count'
  | 'sum'
  | 'mean'
  | 'minimum'
  | 'maximum'
  | 'count-distinct';

/** One 1:1 gathered right column: the first (smallest-row) matching right row's value. */
export type GPUKeyLookupGather = {
  /** Right-aligned source column. */
  column: GraphDataView<'float32'> | GraphDataView<'uint32'>;
  /**
   * Left-aligned destination of the same format. Unmatched rows receive NaN (`float32`) or
   * `0xffffffff` (`uint32`). The value is copied bit for bit.
   */
  output: GraphDataView<'float32'> | GraphDataView<'uint32'>;
  /** Fill for an unmatched row, in the destination format. Defaults to NaN or 0xffffffff. */
  unmatchedValue?: number;
};

/** One 1:n aggregate over every right row that matches a left row's key. */
type GPUKeyLookupAggregateFields = {
  /**
   * Right-aligned values. Required except for `'count'`. Rows with a non-finite value are
   * excluded from this aggregate only.
   */
  column?: GraphDataView<'float32'>;
  /**
   * Optional left-aligned fixed-point sums in units of `1 / sumScale` (`'sum'` and `'mean'`
   * only), signed 64-bit little-endian words equal to the sum of
   * `roundHalfEven(fround(value * sumScale))`. Each value is rounded to the unit first, so a
   * value below half a unit adds nothing; `output` holds the sum of the values themselves. Zero
   * for unmatched rows. Scaled rows clamp just below 2^62 and totals wrap at 64 bits; request
   * `status.sumOverflow` to detect either loss.
   */
  sums?: GraphDataView<'uint32x2'>;
  status?: {
    /** One-word flag for row clamping or signed-64-bit sum overflow among right keys
     * matched by a valid, unmasked left row. Requires `sums`. */
    sumOverflow?: GraphDataView<'uint32'>;
  };
};

/**
 * Aggregate of matching right rows, published in left-row order. Count counts matching rows
 * (finite values if column is supplied); count-distinct counts unique finite values, merging +/-0.
 * Counts may publish exactly as uint32 or rounded as float32; unmatched counts are zero.
 * Sum/mean use adaptive 96-bit totals independently of sumScale. Minimum/maximum are exact.
 * Unmatched float aggregates are NaN; a matched group with no finite values has sum 0 and
 * mean/minimum/maximum NaN. Caller-scale words and their loss status are independent.
 */
export type GPUKeyLookupAggregate = GPUKeyLookupAggregateFields &
  (
    | {
        operation: 'count' | 'count-distinct';
        output: GraphDataView<'float32'> | GraphDataView<'uint32'>;
      }
    | {
        operation: Exclude<GPUKeyLookupAggregateOperation, 'count' | 'count-distinct'>;
        output: GraphDataView<'float32'>;
      }
  );

/** Outputs of {@link GPUKeyLookup}. Left-aligned unless noted. */
export type GPUKeyLookupOutput = {
  /** Whole-table certification, required when rightTable is supplied. matched still describes retained keys only. */
  status?: {complete: GraphDataView<'uint32'>; candidateOverflow?: GraphDataView<'uint32'>};
  /** Smallest matching right row per left row, `0xffffffff` when unmatched. */
  rightRows?: GraphDataView<'uint32'>;
  /** Number of right rows matching each left row's key. */
  matchCounts?: GraphDataView<'uint32'>;
  /** 1 when the left row matched, else 0; usable as a crossfilter live mask. */
  matched?: GraphDataView<'uint32'>;
  /** Right-aligned: 1 when the right row's key is matched by some valid left row. */
  rightMatched?: GraphDataView<'uint32'>;
  /** Required when compactMatchedLeftRows is true: ascending matched left row indices, capacity bounded. */
  rows?: GPUCompactOutput;
};

/**
 * Properties for {@link GPUKeyLookup}.
 *
 * Per-frame (no recompile): the contents of every input buffer. Topology (needs a new graph):
 * `compactMatchedLeftRows`, `sumScale`, view lengths, key format, and which optional views and aggregates exist.
 */
export type GPUKeyLookupProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'key-join'`. */
  id?: string;
  /**
   * Left keys: `uint32` or 64-bit `uint32x2` little-endian `(low, high)` words. The all-ones key
   * (`0xffffffff`, or both words `0xffffffff`) means "no key" and never matches.
   */
  leftKeys: GraphDataView<'uint32'> | GraphDataView<'uint32x2'>;
  /** Right keys, in the same format as `leftKeys`. */
  rightKeys: GraphDataView<'uint32'> | GraphDataView<'uint32x2'>;
  /** Grouped right keys and producer loss. keys must be the same view as rightKeys.
   * Requires output.status.complete; 0 means the lookup cannot certify genuine misses. */
  rightTable?: {
    keys: GraphDataView<'uint32'> | GraphDataView<'uint32x2'>;
    status: {overflow: GraphDataView<'uint32'>; candidateOverflow?: GraphDataView<'uint32'>};
  };
  /** Optional left mask; zero rows never match. */
  leftMask?: GraphDataView<'uint32'>;
  /** Optional right mask; zero rows are ignored by every output. */
  rightMask?: GraphDataView<'uint32'>;
  /** Also publish compact matched left row indices in output.rows. Defaults to false. */
  compactMatchedLeftRows?: boolean;
  /** 1:1 gathered right columns (first matching right row wins). */
  gather?: readonly GPUKeyLookupGather[];
  /** 1:n aggregates over all matching right rows. */
  aggregates?: readonly GPUKeyLookupAggregate[];
  /**
   * Fixed-point scale of the optional `sums` words of an aggregate only, a positive float32: one
   * unit is `1 / sumScale`. Defaults to 65536. The float `'sum'` and `'mean'` do not use it.
   */
  sumScale?: number;
  /** Isolate adaptive float sum/mean units and finite-count headroom per right key. Defaults to shared. */
  scaleMode?: 'shared' | 'per-group';
  /** Caller-owned outputs. */
  output: GPUKeyLookupOutput;
};

/**
 * Looks up each left row's key in a right table, entirely on the GPU.
 *
 * Valid right rows (mask set, key not all-ones) are sorted by key with a stable radix sort
 * (one pass per key word), so equal keys stay in ascending right-row order. Segment heads and a
 * scan give the unique right keys and their row ranges. Each left row binary-searches the unique
 * keys on the unsigned `(high, low)` order, then gathers the first matching right row's columns
 * and 1:n aggregates. Aggregates accumulate once per unique key with integer atomics (96-bit
 * fixed-point sums with a data-derived unit and explicit carry, ordered-key extremes), so every
 * output is independent of thread order and bitwise reproducible. Counts and extremes are exact;
 * a sum is the exact integer sum rounded once to float32 (a value keeps all its bits while its
 * magnitude is at least `max * 2^(h - 71)`, `h` the bit length of the right row count). All
 * inputs must be single packed views.
 * Left-aligned outputs keep one slot per left row in source order. With
 * `compactMatchedLeftRows: true`, `output.rows` also lists the matched left row indices in
 * ascending order, once each.
 *
 * Float sums and means default to one adaptive scale across right keys. With values [1, 2^100]
 * at distinct keys, the shared unit is 256 and the singleton 1 rounds to zero. Set
 * scaleMode: 'per-group' to isolate units by key. Caller-scale sums and status.sumOverflow remain
 * independent: the flag covers row clamping and signed-64 loss for valid unmasked left matches.
 */
export class GPUKeyLookup implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUKeyLookupProps;
  /** Whether to publish compact matched left row indices in output.rows. */
  readonly compactMatchedLeftRows: boolean;
  /** Fixed-point scale of the `sums` words. */
  readonly sumScale: number;
  readonly estimatedCommandNodeCount: number;

  constructor(props: GPUKeyLookupProps) {
    const inputProps = props;
    props = snapshotConstructionFields(props, [
      'id',
      'leftKeys',
      'rightKeys',
      'rightTable',
      'leftMask',
      'rightMask',
      'compactMatchedLeftRows',
      'gather',
      'aggregates',
      'sumScale',
      'scaleMode',
      'output'
    ]);
    const id = props.id ?? 'key-join';
    for (const legacyProp of ['kind', 'how']) {
      validateRequirement(
        id,
        !(legacyProp in inputProps || Reflect.get(inputProps, legacyProp) !== undefined),
        `${legacyProp} was removed; use compactMatchedLeftRows`
      );
    }
    const {compactMatchedLeftRows} = props;
    validateRequirement(
      id,
      compactMatchedLeftRows === undefined || typeof compactMatchedLeftRows === 'boolean',
      `compactMatchedLeftRows must be a boolean; received ${String(compactMatchedLeftRows)}`
    );
    const declaredOutput = snapshotConstructionFields(props.output, [
      'rightRows',
      'matchCounts',
      'matched',
      'rightMatched',
      'rows',
      'status'
    ]);
    const compact = declaredOutput.rows
      ? snapshotConstructionFields(declaredOutput.rows, ['ids', 'status'])
      : undefined;
    props = Object.freeze({
      ...props,
      rightTable: props.rightTable
        ? Object.freeze({
            keys: props.rightTable.keys,
            status: snapshotConstructionFields(props.rightTable.status, [
              'overflow',
              'candidateOverflow'
            ])
          })
        : undefined,
      gather: props.gather
        ? Object.freeze(
            props.gather.map(entry =>
              snapshotConstructionFields(entry, ['column', 'output', 'unmatchedValue'])
            )
          )
        : undefined,
      aggregates: props.aggregates
        ? Object.freeze(
            props.aggregates.map((entry, index) => {
              validateRequirement(
                id,
                !('sumOverflow' in entry || Reflect.get(entry, 'sumOverflow') !== undefined),
                `aggregates[${index}].sumOverflow was removed; use aggregates[${index}].status.sumOverflow`
              );
              entry = snapshotConstructionFields(entry, [
                'operation',
                'column',
                'output',
                'sums',
                'status'
              ]);
              return Object.freeze({
                ...entry,
                status: entry.status
                  ? snapshotConstructionFields(entry.status, ['sumOverflow'])
                  : undefined
              });
            })
          )
        : undefined,
      output: Object.freeze({
        ...declaredOutput,
        status: declaredOutput.status
          ? snapshotConstructionFields(declaredOutput.status, ['complete', 'candidateOverflow'])
          : undefined,
        rows: compact
          ? Object.freeze({
              ...compact,
              status: snapshotConstructionFields(compact.status, [
                'count',
                'overflow',
                'candidateOverflow',
                'requiredCount'
              ])
            })
          : undefined
      })
    });
    this.id = id;
    this.props = props;
    this.compactMatchedLeftRows = compactMatchedLeftRows ?? false;
    this.sumScale = props.sumScale ?? GPU_CELL_DEFAULT_SUM_SCALE;
    validateSumScale(id, this.sumScale);
    validateRequirement(
      id,
      props.scaleMode === undefined ||
        props.scaleMode === 'shared' ||
        props.scaleMode === 'per-group',
      `scaleMode must be shared or per-group; received ${props.scaleMode}`
    );
    const {output} = props;
    if (props.rightTable) {
      validateRequirement(
        id,
        props.rightTable.keys === props.rightKeys,
        'rightTable.keys must equal rightKeys'
      );
      validateRequirement(
        id,
        Boolean(output.status?.complete),
        'rightTable requires output.status.complete'
      );
      validateRequirement(
        id,
        Boolean(props.rightTable.status.overflow),
        'rightTable.status.overflow is required'
      );
      for (const [name, view] of Object.entries(props.rightTable.status)) {
        if (!view) continue;
        validateSinglePackedView(view, ['uint32'], `${id} rightTable.status.${name}`);
        validateRequirement(id, view.length === 1, `rightTable.status.${name} must hold one row`);
      }
    }
    for (const [name, view] of Object.entries(output.status ?? {})) {
      if (!view) continue;
      validateSinglePackedView(view, ['uint32'], `${id} output.status.${name}`);
      validateRequirement(id, view.length === 1, `output.status.${name} must hold one row`);
    }
    validateDisjointWritableViews(id, {
      ...output.status,
      localOverflow: output.status ? output.rows?.status.overflow : undefined
    });
    const gather = props.gather ?? [];
    const aggregates = props.aggregates ?? [];
    const keyFormat = props.leftKeys.format;
    if (keyFormat !== 'uint32' && keyFormat !== 'uint32x2') {
      throw new Error(`${id} leftKeys must be uint32 or uint32x2`);
    }
    validateSinglePackedView(props.leftKeys, [keyFormat], `${id} leftKeys`);
    validateSinglePackedView(props.rightKeys, [keyFormat], `${id} rightKeys`);
    const leftCount = props.leftKeys.length;
    const rightCount = props.rightKeys.length;
    validateCapacity(id, 'leftKeys length', leftCount, 1, MAXIMUM_LEFT_ROW_COUNT);
    validateCapacity(id, 'rightKeys length', rightCount, 1, MAXIMUM_RIGHT_ROW_COUNT);
    for (const [name, view, count] of [
      ['leftMask', props.leftMask, leftCount],
      ['rightMask', props.rightMask, rightCount]
    ] as const) {
      if (view) {
        validateSinglePackedView(view, ['uint32'], `${id} ${name}`);
        if (view.length !== count) {
          throw new Error(`${id} ${name} length must equal the ${name.slice(0, -4)} row count`);
        }
      }
    }
    for (const [index, entry] of gather.entries()) {
      const name = `gather[${index}]`;
      const format = entry.column.format;
      validateSinglePackedView(entry.column, ['float32', 'uint32'], `${id} ${name}.column`);
      validateSinglePackedView(entry.output, [format], `${id} ${name}.output`);
      if (entry.unmatchedValue !== undefined) {
        validateRequirement(
          id,
          format === 'uint32'
            ? Number.isInteger(entry.unmatchedValue) &&
                entry.unmatchedValue >= 0 &&
                entry.unmatchedValue <= 0xffffffff
            : typeof entry.unmatchedValue === 'number' &&
                (!Number.isFinite(entry.unmatchedValue) ||
                  Number.isFinite(Math.fround(entry.unmatchedValue))),
          `${name}.unmatchedValue must fit ${format}; received ${entry.unmatchedValue}`
        );
      }
      if (entry.column.length !== rightCount) {
        throw new Error(`${id} ${name}.column length must equal the right row count`);
      }
      if (entry.output.length !== leftCount) {
        throw new Error(`${id} ${name}.output length must equal the left row count`);
      }
    }
    for (const [index, entry] of aggregates.entries()) {
      const name = `aggregates[${index}]`;
      if (
        !['count', 'sum', 'mean', 'minimum', 'maximum', 'count-distinct'].includes(entry.operation)
      ) {
        throw new Error(`${id} ${name}.operation is not supported`);
      }
      if (entry.operation !== 'count' && !entry.column) {
        throw new Error(`${id} ${name} needs a column for ${entry.operation}`);
      }
      if (entry.column) {
        validateSinglePackedView(entry.column, ['float32'], `${id} ${name}.column`);
        if (entry.column.length !== rightCount) {
          throw new Error(`${id} ${name}.column length must equal the right row count`);
        }
      }
      validateSinglePackedView(
        entry.output,
        entry.operation === 'count' || entry.operation === 'count-distinct'
          ? ['float32', 'uint32']
          : ['float32'],
        `${id} ${name}.output`
      );
      if (entry.output.length !== leftCount) {
        throw new Error(`${id} ${name}.output length must equal the left row count`);
      }
      const sumOverflow = entry.status?.sumOverflow;
      if (sumOverflow) {
        validateRequirement(id, Boolean(entry.sums), `${name}.status.sumOverflow requires sums`);
        validateSinglePackedView(sumOverflow, ['uint32'], `${id} ${name}.status.sumOverflow`);
        validateRequirement(
          id,
          sumOverflow.length === 1,
          `${name}.status.sumOverflow must hold one uint32 row`
        );
      }
      if (entry.sums) {
        if (entry.operation !== 'sum' && entry.operation !== 'mean') {
          throw new Error(`${id} ${name}.sums needs the sum or mean operation`);
        }
        validateSinglePackedView(entry.sums, ['uint32x2'], `${id} ${name}.sums`);
        if (entry.sums.length !== leftCount) {
          throw new Error(`${id} ${name}.sums length must equal the left row count`);
        }
      }
    }
    for (const name of ['rightRows', 'matchCounts', 'matched'] as const) {
      const view = output[name];
      if (view) {
        validateSinglePackedView(view, ['uint32'], `${id} output.${name}`);
        if (view.length !== leftCount) {
          throw new Error(`${id} output.${name} length must equal the left row count`);
        }
      }
    }
    if (output.rightMatched) {
      validateSinglePackedView(output.rightMatched, ['uint32'], `${id} output.rightMatched`);
      if (output.rightMatched.length !== rightCount) {
        throw new Error(`${id} output.rightMatched length must equal the right row count`);
      }
    }
    if (this.compactMatchedLeftRows) {
      validateRequirement(
        id,
        output.rows !== undefined,
        'output.rows is required when compactMatchedLeftRows is true'
      );
      validateCompactOutput(id, output.rows);
      if (output.rows.ids.length < 1) {
        throw new Error(`${id} output.rows.ids must hold at least one row`);
      }
    } else {
      validateRequirement(
        id,
        output.rows === undefined,
        'output.rows is only supported when compactMatchedLeftRows is true'
      );
    }
    const inputs = [
      props.leftKeys,
      props.rightKeys,
      props.leftMask,
      props.rightMask,
      props.rightTable?.status.overflow,
      props.rightTable?.status.candidateOverflow,
      ...gather.map(entry => entry.column),
      ...aggregates.map(entry => entry.column)
    ];
    const outputs = [
      output.status?.complete,
      output.status?.candidateOverflow,
      output.rightRows,
      output.matchCounts,
      output.matched,
      output.rightMatched,
      output.rows?.ids,
      output.rows?.status.count,
      output.rows?.status.overflow,
      output.rows?.status.candidateOverflow,
      output.rows?.status.requiredCount,
      ...gather.map(entry => entry.output),
      ...aggregates.flatMap(entry => [entry.output, entry.sums, entry.status?.sumOverflow])
    ];
    const publicationPasses: Readonly<Record<string, GraphDataView | undefined>>[] = [
      {
        'output.rightRows': output.rightRows,
        'output.matchCounts': output.matchCounts,
        'output.matched': output.matched
      },
      {
        'output.rows.status.count': output.rows?.status.count,
        'output.rows.status.overflow': output.rows?.status.overflow,
        'output.rows.status.candidateOverflow': output.rows?.status.candidateOverflow,
        'output.rows.status.requiredCount': output.rows?.status.requiredCount
      },
      ...aggregates.map((entry, index) => ({
        [`aggregates[${index}].output`]: entry.output,
        [`aggregates[${index}].sums`]: entry.sums
      }))
    ];
    for (let first = 0; first < gather.length; first += MAXIMUM_GATHER_COLUMNS_PER_KERNEL) {
      publicationPasses.push({
        'output.rightRows': output.rightRows,
        ...Object.fromEntries(
          gather
            .slice(first, first + MAXIMUM_GATHER_COLUMNS_PER_KERNEL)
            .map((entry, index) => [`gather[${first + index}].output`, entry.output])
        )
      });
    }
    if (output.rows)
      publicationPasses.push({
        'output.matched': output.matched,
        'output.rows.ids': output.rows.ids
      });
    {
      const writableOutputs: Readonly<Record<string, GraphDataView | GraphVectorView | undefined>> =
        {
          'output.status.complete': output.status?.complete,
          'output.status.candidateOverflow': output.status?.candidateOverflow,
          'output.rightRows': output.rightRows,
          'output.matchCounts': output.matchCounts,
          'output.matched': output.matched,
          'output.rightMatched': output.rightMatched,
          'output.rows.ids': output.rows?.ids,
          'output.rows.status.count': output.rows?.status.count,
          'output.rows.status.overflow': output.rows?.status.overflow,
          'output.rows.status.candidateOverflow': output.rows?.status.candidateOverflow,
          'output.rows.status.requiredCount': output.rows?.status.requiredCount,
          ...Object.fromEntries(
            gather.map((entry, index) => [`gather[${index}].output`, entry.output])
          ),
          ...Object.fromEntries(
            aggregates.flatMap((entry, index) => [
              [`aggregates[${index}].output`, entry.output],
              [`aggregates[${index}].sums`, entry.sums],
              [`aggregates[${index}].status.sumOverflow`, entry.status?.sumOverflow]
            ])
          )
        };
      validateDisjointWritableViews(id, writableOutputs, {separateDispatches: true});
      for (const fields of publicationPasses) validateDisjointWritableViews(id, fields);
    }
    this.estimatedCommandNodeCount =
      BASE_COMMAND_NODE_ALLOWANCE +
      Math.ceil(gather.length / MAXIMUM_GATHER_COLUMNS_PER_KERNEL) +
      aggregates.reduce(
        (total, entry) =>
          total +
          (entry.operation === 'count-distinct'
            ? DISTINCT_COMMAND_NODE_ALLOWANCE
            : AGGREGATE_COMMAND_NODE_ALLOWANCE + (entry.status?.sumOverflow ? 4 : 0)),
        0
      );
    validateCommandNodeBudget(id, this.estimatedCommandNodeCount);
    validateGraphOutputsDisjointFromInputs(id, outputs, inputs);
  }

  /** Returns build, probe, gather, aggregate, right-matched and compaction nodes in order. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, compactMatchedLeftRows} = this;
    const output = props.output;
    const gather = props.gather ?? [];
    const aggregates = props.aggregates ?? [];
    const boundViews = [
      props.leftKeys,
      props.rightKeys,
      props.leftMask,
      props.rightMask,
      props.rightTable?.status.overflow,
      props.rightTable?.status.candidateOverflow,
      output.status?.complete,
      output.status?.candidateOverflow,
      ...gather.flatMap(entry => [entry.column, entry.output]),
      ...aggregates.flatMap(entry => [
        entry.column,
        entry.output,
        entry.sums,
        entry.status?.sumOverflow
      ]),
      output.rightRows,
      output.matchCounts,
      output.matched,
      output.rightMatched,
      output.rows?.ids,
      output.rows?.status.count,
      output.rows?.status.overflow,
      output.rows?.status.candidateOverflow,
      output.rows?.status.requiredCount
    ];
    validateGraphChunkBindingRanges(
      graph,
      id,
      Object.fromEntries(boundViews.map((view, index) => [`view[${index}]`, view]))
    );
    const rightRows = props.rightKeys.length;
    const scratchLengths = {
      leftRows: props.leftKeys.length,
      uniqueStart: rightRows + 1,
      sumWords: aggregates.some(entry => entry.operation === 'sum' || entry.operation === 'mean')
        ? 2 * rightRows
        : 0
    };
    for (const [name, length] of Object.entries(scratchLengths)) {
      if (length > 0) {
        validateMaximumCount(
          id,
          `${name} scratch byteLength`,
          length * 4,
          graph.device.limits.maxBufferSize
        );
        validateGPUStorageBindingByteLength(
          graph.device,
          id,
          `${name} scratch`,
          length * 4,
          'reduce the left or right row count'
        );
      }
    }
    const twoWords = props.leftKeys.format === 'uint32x2';
    const leftCount = props.leftKeys.length;
    const rightCount = props.rightKeys.length;
    const u32 = (name: string, length: number) =>
      createTransientView(graph, `${id}-${name}`, 'uint32', length);
    const read = (name: string, view: GraphDataView, type: 'u32' | 'f32' = 'u32') =>
      ({name, view, type, access: 'read'}) as WGSLKernelBinding;
    const write = (name: string, view: GraphDataView, type: 'u32' | 'f32' = 'u32') =>
      ({name, view, type, access: 'read_write'}) as WGSLKernelBinding;
    const atomic = (name: string, view: GraphDataView) =>
      ({
        name,
        view,
        type: 'atomic<u32>',
        access: 'read_write'
      }) as WGSLKernelBinding;
    const node = (
      step: string,
      bindings: WGSLKernelBinding[],
      invocationCount: number,
      body: string,
      declarations = ''
    ) =>
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-${step}`,
        operation: OPERATION,
        variant: step,
        bindings,
        invocationCount,
        declarations,
        body
      });
    const nodes: GPUCommandNode<Parameters>[] = [];
    const keyDeclarations = /* wgsl */ `
const NO_KEY: u32 = 0xffffffffu;
fn isNoKey(key: vec2u) -> bool {
  return key.y == NO_KEY && ${twoWords ? 'key.x == NO_KEY' : 'true'};
}
fn isKeyLess(left: vec2u, right: vec2u) -> bool {
  return left.x < right.x || (left.x == right.x && left.y < right.y);
}`;
    // Keys as vec2u(high, low) from an input key view.
    const readKey = (name: string, row: string) =>
      twoWords
        ? `vec2u(${name}[${name}Offset + 2u * (${row}) + 1u], ${name}[${name}Offset + 2u * (${row})])`
        : `vec2u(0u, ${name}[${name}Offset + (${row})])`;

    // 1. Right keys: invalid rows (mask or reserved key) become the all-ones key and sort last.
    const keyLow = u32('key-low', rightCount);
    const keyHigh = twoWords ? u32('key-high', rightCount) : undefined;
    const rowIds = u32('row-ids', rightCount);
    nodes.push(
      node(
        'right-keys',
        [
          read('rightKeys', props.rightKeys),
          ...(props.rightMask ? [read('rightMask', props.rightMask)] : []),
          write('keyLow', keyLow),
          ...(keyHigh ? [write('keyHigh', keyHigh)] : []),
          write('rowIds', rowIds)
        ],
        rightCount,
        `var key = ${readKey('rightKeys', 'index')};
  ${props.rightMask ? 'if (rightMask[rightMaskOffset + index] == 0u) {\n    key = vec2u(NO_KEY, NO_KEY);\n  }' : ''}
  if (isNoKey(key)) {
    key = vec2u(NO_KEY, NO_KEY);
  }
  keyLow[keyLowOffset + index] = key.y;
  ${keyHigh ? 'keyHigh[keyHighOffset + index] = key.x;' : ''}
  rowIds[rowIdsOffset + index] = index;`,
        keyDeclarations
      )
    );

    // 2. Stable ascending (high, low) order; gather the low word in the final order.
    const sorted = createStableLexicographicIndexSortNodes(graph, {
      id: `${id}-sort`,
      operation: OPERATION,
      indices: rowIds,
      indicesAreIdentity: true,
      keys: [...(keyHigh ? [{view: keyHigh}] : []), {view: keyLow}]
    });
    nodes.push(...sorted.nodes);
    const sortedRows = sorted.sortedIndices;
    const sortedHigh = keyHigh ? sorted.sortedPrimaryKeys : undefined;
    const sortedLow = keyHigh ? u32('sorted-low', rightCount) : sorted.sortedPrimaryKeys;
    if (keyHigh) {
      nodes.push(
        ...new GPUUint32Gather({
          id: `${id}-gather-low`,
          source: keyLow,
          indices: sortedRows,
          output: sortedLow
        }).getCommandNodes(graph)
      );
    }
    const sortedKeyBindings = (): WGSLKernelBinding[] => [
      read('sortedLow', sortedLow),
      ...(sortedHigh ? [read('sortedHigh', sortedHigh)] : [])
    ];
    const sortedKeyAt = (row: string) =>
      twoWords
        ? `vec2u(sortedHigh[sortedHighOffset + ${row}], sortedLow[sortedLowOffset + ${row}])`
        : `vec2u(0u, sortedLow[sortedLowOffset + ${row}])`;

    // 3. Segment heads of valid rows (a prefix of the sorted order) and the valid row count.
    const heads = u32('heads', rightCount);
    const validCount = u32('valid-count', 1);
    nodes.push(
      node(
        'heads',
        [...sortedKeyBindings(), write('heads', heads), write('validCount', validCount)],
        rightCount,
        `let key = ${sortedKeyAt('index')};
  let isValid = !isNoKey(key);
  var isHead = isValid;
  if (isValid && index > 0u) {
    isHead = any(key != ${sortedKeyAt('index - 1u')});
  }
  heads[headsOffset + index] = select(0u, 1u, isHead);
  let isLastValid = isValid && (index + 1u == ROW_COUNT || isNoKey(${sortedKeyAt('index + 1u')}));
  if (isLastValid) {
    validCount[validCountOffset] = index + 1u;
  } else if (index == 0u && !isValid) {
    validCount[validCountOffset] = 0u;
  }`,
        `const ROW_COUNT: u32 = ${rightCount}u;\n${keyDeclarations}`
      )
    );

    // 4. Exclusive scan of heads gives every head its unique-key index.
    const uniqueIndices = u32('unique-indices', rightCount);
    nodes.push(
      ...new GPUScan({
        id: `${id}-scan`,
        input: heads,
        output: uniqueIndices,
        mode: 'exclusive'
      }).getCommandNodes(graph)
    );

    // 5. Unique keys with the first sorted position of each; `uniqueStart[U]` is the valid count.
    const uniqueLow = u32('unique-low', rightCount);
    const uniqueHigh = twoWords ? u32('unique-high', rightCount) : undefined;
    const uniqueStart = u32('unique-start', rightCount + 1);
    nodes.push(
      node(
        'unique-keys',
        [
          ...sortedKeyBindings(),
          read('heads', heads),
          read('uniqueIndices', uniqueIndices),
          read('validCount', validCount),
          write('uniqueLow', uniqueLow),
          ...(uniqueHigh ? [write('uniqueHigh', uniqueHigh)] : []),
          write('uniqueStart', uniqueStart)
        ],
        rightCount,
        `if (heads[headsOffset + index] != 0u) {
    let unique = uniqueIndices[uniqueIndicesOffset + index];
    let key = ${sortedKeyAt('index')};
    uniqueLow[uniqueLowOffset + unique] = key.y;
    ${uniqueHigh ? 'uniqueHigh[uniqueHighOffset + unique] = key.x;' : ''}
    uniqueStart[uniqueStartOffset + unique] = index;
  }
  if (index == 0u) {
    let last = ROW_COUNT - 1u;
    let uniqueTotal = uniqueIndices[uniqueIndicesOffset + last] + heads[headsOffset + last];
    uniqueStart[uniqueStartOffset + uniqueTotal] = validCount[validCountOffset];
  }`,
        `const ROW_COUNT: u32 = ${rightCount}u;`
      )
    );
    // Unique-key total and the unique index of every sorted row (NO_UNIQUE for invalid rows).
    const uniqueTotal = u32('unique-total', 1);
    const rowUnique = u32('row-unique', rightCount);
    nodes.push(
      node(
        'row-unique',
        [
          read('heads', heads),
          read('uniqueIndices', uniqueIndices),
          read('validCount', validCount),
          write('rowUnique', rowUnique),
          write('uniqueTotal', uniqueTotal)
        ],
        rightCount,
        `let isValid = index < validCount[validCountOffset];
  rowUnique[rowUniqueOffset + index] = select(
    0xffffffffu,
    uniqueIndices[uniqueIndicesOffset + index] + heads[headsOffset + index] - 1u,
    isValid
  );
  if (index == 0u) {
    let last = ROW_COUNT - 1u;
    uniqueTotal[uniqueTotalOffset] =
      uniqueIndices[uniqueIndicesOffset + last] + heads[headsOffset + last];
  }`,
        `const ROW_COUNT: u32 = ${rightCount}u;`
      )
    );

    // 6. Probe: binary search of every left key over the unique keys.
    const leftUnique = u32('left-unique', leftCount);
    const uniqueMatched =
      output.rightMatched || aggregates.some(aggregate => aggregate.status?.sumOverflow)
        ? u32('unique-matched', rightCount)
        : undefined;
    if (uniqueMatched) {
      nodes.push(
        createFillNode<Parameters>(graph, {
          id: `${id}-clear-unique-matched`,
          operation: OPERATION,
          view: uniqueMatched,
          type: 'u32',
          value: '0u'
        })
      );
    }
    nodes.push(
      node(
        'probe',
        [
          read('leftKeys', props.leftKeys),
          ...(props.leftMask ? [read('leftMask', props.leftMask)] : []),
          read('uniqueLow', uniqueLow),
          ...(uniqueHigh ? [read('uniqueHigh', uniqueHigh)] : []),
          read('uniqueTotal', uniqueTotal),
          write('leftUnique', leftUnique),
          ...(uniqueMatched ? [atomic('uniqueMatched', uniqueMatched)] : [])
        ],
        leftCount,
        `var key = ${readKey('leftKeys', 'index')};
  ${props.leftMask ? 'if (leftMask[leftMaskOffset + index] == 0u) {\n    key = vec2u(NO_KEY, NO_KEY);\n  }' : ''}
  var found = 0xffffffffu;
  if (!isNoKey(key)) {
    let total = uniqueTotal[uniqueTotalOffset];
    var low = 0u;
    var high = total;
    while (low < high) {
      let middle = (low + high) >> 1u;
      let candidate = ${
        twoWords
          ? 'vec2u(uniqueHigh[uniqueHighOffset + middle], uniqueLow[uniqueLowOffset + middle])'
          : 'vec2u(0u, uniqueLow[uniqueLowOffset + middle])'
      };
      if (isKeyLess(candidate, key)) {
        low = middle + 1u;
      } else {
        high = middle;
      }
    }
    if (low < total) {
      let candidate = ${
        twoWords
          ? 'vec2u(uniqueHigh[uniqueHighOffset + low], uniqueLow[uniqueLowOffset + low])'
          : 'vec2u(0u, uniqueLow[uniqueLowOffset + low])'
      };
      if (all(candidate == key)) {
        found = low;
      }
    }
  }
  leftUnique[leftUniqueOffset + index] = found;
  ${uniqueMatched ? 'if (found != 0xffffffffu) {\n    atomicMax(&uniqueMatched[uniqueMatchedOffset + found], 1u);\n  }' : ''}`,
        keyDeclarations
      )
    );

    // 7. Left-aligned flags, first rows and match counts.
    const needsMatched = Boolean(output.matched) || compactMatchedLeftRows;
    const needsFirst = Boolean(output.rightRows) || gather.length > 0;
    const matchedView = output.matched ?? (needsMatched ? u32('matched', leftCount) : undefined);
    const firstView = output.rightRows ?? (needsFirst ? u32('first-rows', leftCount) : undefined);
    if (matchedView || firstView || output.matchCounts) {
      nodes.push(
        node(
          'left-rows',
          [
            read('leftUnique', leftUnique),
            read('uniqueStart', uniqueStart),
            ...(firstView ? [read('sortedRows', sortedRows)] : []),
            ...(matchedView ? [write('matchedOut', matchedView)] : []),
            ...(firstView ? [write('firstOut', firstView)] : []),
            ...(output.matchCounts ? [write('countsOut', output.matchCounts)] : [])
          ],
          leftCount,
          `let unique = leftUnique[leftUniqueOffset + index];
  let isMatched = unique != 0xffffffffu;
  ${matchedView ? 'matchedOut[matchedOutOffset + index] = select(0u, 1u, isMatched);' : ''}
  var begin = 0u;
  var end = 0u;
  if (isMatched) {
    begin = uniqueStart[uniqueStartOffset + unique];
    end = uniqueStart[uniqueStartOffset + unique + 1u];
  }
  ${firstView ? 'firstOut[firstOutOffset + index] = select(0xffffffffu, sortedRows[sortedRowsOffset + begin], isMatched);' : ''}
  ${output.matchCounts ? 'countsOut[countsOutOffset + index] = end - begin;' : ''}`
        )
      );
    }

    // 8. 1:1 gathers, at most three columns per kernel (bit copies; NaN or all-ones if unmatched).
    for (let first = 0; first < gather.length; first += MAXIMUM_GATHER_COLUMNS_PER_KERNEL) {
      const group = gather.slice(first, first + MAXIMUM_GATHER_COLUMNS_PER_KERNEL);
      nodes.push(
        node(
          `gather-${first}`,
          [
            // Retained row IDs and gathered outputs may share one writable buffer.
            write('firstRows', firstView!),
            ...group.flatMap((entry, index) => [
              read(`column${index}`, entry.column),
              write(`out${index}`, entry.output)
            ])
          ],
          leftCount,
          `let row = firstRows[firstRowsOffset + index];
  let isMatched = row != 0xffffffffu;
  ${group
    .map(
      (entry, index) =>
        `out${index}[out${index}Offset + index] = select(${entry.unmatchedValue === undefined ? (entry.column.format === 'float32' ? '0x7fc00000u' : '0xffffffffu') : `${entry.column.format === 'float32' ? new Uint32Array(Float32Array.of(entry.unmatchedValue).buffer)[0] : entry.unmatchedValue}u`}, column${index}[column${index}Offset + select(0u, row, isMatched)], isMatched);`
    )
    .join('\n  ')}`
        )
      );
    }

    // 9. 1:n aggregates: per-unique integer accumulation, then a left-aligned finish.
    for (const [index, aggregate] of aggregates.entries()) {
      const name = `aggregate-${index}`;
      const {operation, column} = aggregate;
      if (operation === 'count-distinct') {
        nodes.push(
          ...createKeyLookupDistinctNodes(graph, {
            id: `${id}-${name}-distinct`,
            column: column!,
            sortedRows,
            rowUnique,
            leftUnique,
            output: aggregate.output
          })
        );
        continue;
      }
      const wantsSums = operation === 'sum' || operation === 'mean';
      const wantsCounts =
        operation === 'mean' ||
        (operation === 'count' && Boolean(column)) ||
        (wantsSums && props.scaleMode === 'per-group');
      const isMinimum = operation === 'minimum';
      const isMaximum = operation === 'maximum';
      // The float `'sum'` and `'mean'` come from the shared 96-bit fixed-point accumulator, whose
      // unit follows the largest finite value among the valid right rows: three words per unique
      // key. They are split over two views so that the row limit stays at one storage binding
      // of two words per row, as it was with the 64-bit sums: the low and middle words here, the
      // high word in `sumHighWords`, followed by one word that holds the largest biased exponent
      // (the scale of the unit).
      const sumWords = wantsSums ? u32(`${name}-sum-words`, 2 * rightCount) : undefined;
      const sumHighWords = wantsSums
        ? u32(
            `${name}-sum-high-words`,
            rightCount + (props.scaleMode === 'per-group' ? rightCount : 1)
          )
        : undefined;
      // The 64-bit words in units of `1 / sumScale` are accumulated only for `aggregate.sums`.
      const sums =
        wantsSums && aggregate.sums
          ? createTransientView(graph, `${id}-${name}-sums`, 'uint32x2', rightCount)
          : undefined;
      const counts = wantsCounts ? u32(`${name}-counts`, rightCount) : undefined;
      const extremes = isMinimum || isMaximum ? u32(`${name}-extremes`, rightCount) : undefined;
      const fixedPointDeclarations = /* wgsl */ `
const SUM_SCALE_INDEX: u32 = ${rightCount}u;
const FIXED_POINT_HEADROOM_BITS: u32 = ${getFixedPointHeadroomBits(rightCount)}u;
${FIXED_POINT_WGSL}`;
      // The carry chain of `getFixedPointAtomicAddWGSL`, with the high word in its own view.
      const sumWordsAddWGSL = /* wgsl */ `
fn addFixedPointToSumWords(cell: u32, value: vec3<u32>) {
  let base = sumWordsOffset + cell * 2u;
  var lowCarry = 0u;
  if (value.x != 0u) {
    let oldLow = atomicAdd(&sumWords[base], value.x);
    lowCarry = select(0u, 1u, oldLow + value.x < oldLow);
  }
  let middleAddend = value.y + lowCarry;
  var middleCarry = select(0u, 1u, middleAddend < value.y);
  if (middleAddend != 0u) {
    let oldMiddle = atomicAdd(&sumWords[base + 1u], middleAddend);
    middleCarry = middleCarry + select(0u, 1u, oldMiddle + middleAddend < oldMiddle);
  }
  let highAddend = value.z + middleCarry;
  if (highAddend != 0u) {
    atomicAdd(&sumHighWords[sumHighWordsOffset + cell], highAddend);
  }
}`;
      if (sumWords || sums || counts || extremes) {
        nodes.push(
          node(
            `${name}-clear`,
            [
              ...(sumWords ? [write('sumWords', sumWords)] : []),
              ...(sumHighWords ? [write('sumHighWords', sumHighWords)] : []),
              ...(sums ? [write('sums', sums)] : []),
              ...(counts ? [write('counts', counts)] : []),
              ...(extremes ? [write('extremes', extremes)] : [])
            ],
            rightCount,
            `${
              sumWords
                ? `sumWords[sumWordsOffset + 2u * index] = 0u;
  sumWords[sumWordsOffset + 2u * index + 1u] = 0u;
  sumHighWords[sumHighWordsOffset + index] = 0u;
  ${props.scaleMode === 'per-group' ? `sumHighWords[sumHighWordsOffset + ${rightCount}u + index] = 0u;` : `if (index == 0u) { sumHighWords[sumHighWordsOffset + ${rightCount}u] = 0u; }`}`
                : ''
            }
  ${sums ? 'sums[sumsOffset + 2u * index] = 0u;\n  sums[sumsOffset + 2u * index + 1u] = 0u;' : ''}
  ${counts ? 'counts[countsOffset + index] = 0u;' : ''}
  ${extremes ? `extremes[extremesOffset + index] = ${isMinimum ? '0xffffffffu' : '0u'};` : ''}`
          )
        );
        if (sumWords) {
          // Scale of the unit: the largest biased exponent of the values the accumulate pass
          // adds (the same predicate), one `atomicMax` per workgroup. Masked rows and rows
          // without a key (a sentinel such as 1e30) do not raise the unit.
          nodes.push(
            createWGSLKernelNode<Parameters>(graph, {
              id: `${id}-${name}-sum-scale`,
              operation: OPERATION,
              variant: `${name}-sum-scale`,
              bindings: [
                read('sortedRows', sortedRows),
                read('rowUnique', rowUnique),
                read('values', column!, 'f32'),
                atomic('sumHighWords', sumHighWords!),
                ...(props.scaleMode === 'per-group' ? [atomic('counts', counts!)] : [])
              ],
              invocationCount: rightCount,
              guardIndex: false,
              workgroupSize: SUM_SCALE_WORKGROUP_SIZE,
              declarations: `${fixedPointDeclarations}
var<workgroup> workgroupExponentBits: atomic<u32>;`,
              // No early return: every invocation of a workgroup must reach the barriers.
              body:
                props.scaleMode === 'per-group'
                  ? `if (index < INVOCATION_COUNT) {
    let group = rowUnique[rowUniqueOffset + index];
    let value = values[valuesOffset + sortedRows[sortedRowsOffset + index]];
    if (group != 0xffffffffu && (bitcast<u32>(value) & 0x7fffffffu) < 0x7f800000u) {
      atomicAdd(&counts[countsOffset + group], 1u);
      atomicMax(&sumHighWords[sumHighWordsOffset + SUM_SCALE_INDEX + group], getFixedPointExponentBits(values[valuesOffset + sortedRows[sortedRowsOffset + index]]));
    }
  }`
                  : `if (localInvocationIndex == 0u) {
    atomicStore(&workgroupExponentBits, 0u);
  }
  workgroupBarrier();
  if (index < INVOCATION_COUNT && rowUnique[rowUniqueOffset + index] != 0xffffffffu) {
    // 0 for zero and for non-finite values, which the accumulate pass skips.
    let exponentBits = getFixedPointExponentBits(
      values[valuesOffset + sortedRows[sortedRowsOffset + index]]
    );
    if (exponentBits != 0u) {
      atomicMax(&workgroupExponentBits, exponentBits);
    }
  }
  workgroupBarrier();
  if (localInvocationIndex == 0u) {
    let workgroupMaximum = atomicLoad(&workgroupExponentBits);
    if (workgroupMaximum != 0u) {
      atomicMax(&sumHighWords[sumHighWordsOffset + SUM_SCALE_INDEX], workgroupMaximum);
    }
  }`
            })
          );
        }
        nodes.push(
          node(
            `${name}-accumulate`,
            [
              read('sortedRows', sortedRows),
              read('rowUnique', rowUnique),
              read('values', column!, 'f32'),
              ...(sumWords ? [atomic('sumWords', sumWords)] : []),
              ...(sumHighWords ? [atomic('sumHighWords', sumHighWords)] : []),
              ...(sums ? [atomic('sums', sums)] : []),
              ...(counts ? [atomic('counts', counts)] : []),
              ...(extremes ? [atomic('extremes', extremes)] : [])
            ],
            rightCount,
            `let unique = rowUnique[rowUniqueOffset + index];
  if (unique == 0xffffffffu) {
    return;
  }
  let value = values[valuesOffset + sortedRows[sortedRowsOffset + index]];
  if ((bitcast<u32>(value) & 0x7fffffffu) >= 0x7f800000u) {
    return;
  }
  ${counts && !(wantsSums && props.scaleMode === 'per-group') ? 'atomicAdd(&counts[countsOffset + unique], 1u);' : ''}
  ${
    sumWords
      ? `let unitExponent = getFixedPointUnitExponent(
    atomicLoad(&sumHighWords[sumHighWordsOffset + SUM_SCALE_INDEX${props.scaleMode === 'per-group' ? ' + unique' : ''}]),
    ${props.scaleMode === 'per-group' ? 'firstLeadingBit(max(atomicLoad(&counts[countsOffset + unique]), 1u)) + 1u' : 'FIXED_POINT_HEADROOM_BITS'}
  );
  addFixedPointToSumWords(unique, quantizeFixedPoint(value, unitExponent));`
      : ''
  }
  ${
    sums
      ? `let scaled = cellScaleValue(value);
  let previous = atomicAdd(&sums[sumsOffset + 2u * unique], scaled.y);
  let carried = scaled.x + select(0u, 1u, previous + scaled.y < previous);
  if (carried != 0u) {
    atomicAdd(&sums[sumsOffset + 2u * unique + 1u], carried);
  }`
      : ''
  }
  ${isMinimum ? 'atomicMin(&extremes[extremesOffset + unique], getOrderedKey(value));' : ''}
  ${isMaximum ? 'atomicMax(&extremes[extremesOffset + unique], getOrderedKey(value));' : ''}`,
            `${getFixedPointWGSL(this.sumScale)}\n${ORDERED_KEY_WGSL}${
              sumWords ? `\n${fixedPointDeclarations}\n${sumWordsAddWGSL}` : ''
            }`
          )
        );
      }
      if (aggregate.status?.sumOverflow) {
        nodes.push(
          ...getSumOverflowNodes(graph, {
            id: `${id}-${name}-sum-overflow`,
            capacity: rightCount,
            sumScale: this.sumScale,
            retainedGroups: uniqueMatched,
            chunks: [{values: column!, groups: rowUnique, permutation: sortedRows}],
            output: aggregate.status.sumOverflow
          })
        );
      }
      // The exact integer sum of a unique key, rounded once to the nearest float32.
      const floatTotal = `let base = sumWordsOffset + 2u * unique;
  let totalWords = vec3<u32>(sumWords[base], sumWords[base + 1u], sumHighWords[sumHighWordsOffset + unique]);
  let unitExponent = getFixedPointUnitExponent(
    sumHighWords[sumHighWordsOffset + SUM_SCALE_INDEX${props.scaleMode === 'per-group' ? ' + unique' : ''}],
    ${props.scaleMode === 'per-group' ? 'firstLeadingBit(max(counts[countsOffset + unique], 1u)) + 1u' : 'FIXED_POINT_HEADROOM_BITS'}
  );
  let floatTotal = getFixedPointFloat(totalWords, unitExponent);`;
      let finishBody: string;
      if (operation === 'count') {
        finishBody = counts
          ? 'let value = f32(counts[countsOffset + unique]);'
          : 'let value = f32(uniqueStart[uniqueStartOffset + unique + 1u] - uniqueStart[uniqueStartOffset + unique]);';
      } else if (operation === 'sum') {
        finishBody = `${floatTotal}
  let value = floatTotal;`;
      } else if (operation === 'mean') {
        finishBody = `${floatTotal}
  let n = counts[countsOffset + unique];
  let value = getFiniteFixedPointMean(totalWords, unitExponent, n);`;
      } else {
        finishBody = `let key = extremes[extremesOffset + unique];
    let value = select(decodeOrderedKey(key), getNaN(), key == ${isMinimum ? '0xffffffffu' : '0u'});`;
      }
      if (operation === 'count' && aggregate.output.format === 'uint32') {
        finishBody = finishBody.replace('f32(', '(');
      }
      const needsUniqueStart = operation === 'count' && !counts;
      nodes.push(
        node(
          `${name}-finish`,
          [
            read('leftUnique', leftUnique),
            ...(needsUniqueStart ? [read('uniqueStart', uniqueStart)] : []),
            ...(sumWords ? [read('sumWords', sumWords)] : []),
            ...(sumHighWords ? [read('sumHighWords', sumHighWords)] : []),
            ...(sums ? [read('sums', sums)] : []),
            ...(counts ? [read('counts', counts)] : []),
            ...(extremes ? [read('extremes', extremes)] : []),
            write(
              'valueOut',
              aggregate.output,
              aggregate.output.format === 'uint32' ? 'u32' : 'f32'
            ),
            ...(aggregate.sums ? [write('sumsOut', aggregate.sums)] : [])
          ],
          leftCount,
          `let unique = leftUnique[leftUniqueOffset + index];
  if (unique == 0xffffffffu) {
    valueOut[valueOutOffset + index] = ${operation === 'count' ? (aggregate.output.format === 'uint32' ? '0u' : '0.0') : 'getNaN()'};
    ${aggregate.sums ? 'sumsOut[sumsOutOffset + 2u * index] = 0u;\n    sumsOut[sumsOutOffset + 2u * index + 1u] = 0u;' : ''}
    return;
  }
  ${sums ? 'let total = vec2u(sums[sumsOffset + 2u * unique + 1u], sums[sumsOffset + 2u * unique]);' : ''}
  ${finishBody}
  valueOut[valueOutOffset + index] = value;
  ${aggregate.sums ? 'sumsOut[sumsOutOffset + 2u * index] = total.y;\n  sumsOut[sumsOutOffset + 2u * index + 1u] = total.x;' : ''}`,
          `${ORDERED_KEY_WGSL}${sumWords ? `\n${fixedPointDeclarations}` : ''}${operation === 'mean' ? `\n${FINITE_MEAN_WGSL}` : ''}`
        )
      );
    }

    // 10. Right-aligned matched flags: every sorted position writes its source row once.
    if (output.rightMatched && uniqueMatched) {
      nodes.push(
        node(
          'right-matched',
          [
            read('sortedRows', sortedRows),
            read('rowUnique', rowUnique),
            read('uniqueMatched', uniqueMatched),
            write('rightMatchedOut', output.rightMatched)
          ],
          rightCount,
          `let unique = rowUnique[rowUniqueOffset + index];
  let isMatched = unique != 0xffffffffu && uniqueMatched[uniqueMatchedOffset + unique] != 0u;
  rightMatchedOut[rightMatchedOutOffset + sortedRows[sortedRowsOffset + index]] =
    select(0u, 1u, isMatched);`
        )
      );
    }

    // 11. Compact matched left row indices in ascending order, bounded by capacity.
    if (compactMatchedLeftRows && output.rows && matchedView) {
      const positions = u32('positions', leftCount);
      const total = u32('matched-total', 1);
      const capacity = output.rows.ids.length;
      nodes.push(
        ...new GPUScan({
          id: `${id}-scan-matched`,
          input: matchedView,
          output: positions,
          mode: 'exclusive'
        }).getCommandNodes(graph)
      );
      nodes.push(
        node(
          'compact',
          [
            // Retained flags and compact IDs may share one writable buffer.
            write('matched', matchedView),
            read('positions', positions),
            write('idsOut', output.rows.ids),
            write('totalOut', total)
          ],
          Math.max(leftCount, capacity),
          `let last = LEFT_COUNT - 1u;
  let total = positions[positionsOffset + last] + matched[matchedOffset + last];
  if (index == 0u) {
    totalOut[totalOutOffset] = total;
  }
  if (index < LEFT_COUNT && matched[matchedOffset + index] != 0u) {
    let position = positions[positionsOffset + index];
    if (position < CAPACITY) {
      idsOut[idsOutOffset + position] = index;
    }
  }
  if (index < CAPACITY && index >= total) {
    idsOut[idsOutOffset + index] = 0xffffffffu;
  }`,
          `const LEFT_COUNT: u32 = ${leftCount}u;\nconst CAPACITY: u32 = ${capacity}u;`
        )
      );
      nodes.push(
        createPublishNode<Parameters>(graph, {
          id: `${id}-publish`,
          operation: OPERATION,
          requiredCount: total,
          output: output.rows,
          candidateOverflowSources:
            output.rows.status.candidateOverflow && props.rightTable
              ? [
                  props.rightTable.status.overflow,
                  ...(props.rightTable.status.candidateOverflow
                    ? [props.rightTable.status.candidateOverflow]
                    : [])
                ]
              : []
        })
      );
    }
    if (output.status) {
      nodes.push(
        node(
          'certify',
          [
            ...(props.rightTable
              ? [
                  read('upstreamOverflow', props.rightTable.status.overflow),
                  ...(props.rightTable.status.candidateOverflow
                    ? [read('upstreamCandidateOverflow', props.rightTable.status.candidateOverflow)]
                    : [])
                ]
              : []),
            write('complete', output.status.complete),
            ...(output.rows ? [write('localOverflow', output.rows.status.overflow)] : []),
            ...(output.status.candidateOverflow
              ? [write('candidateOverflow', output.status.candidateOverflow)]
              : [])
          ],
          1,
          `let loss = ${props.rightTable ? `upstreamOverflow[upstreamOverflowOffset] != 0u${props.rightTable.status.candidateOverflow ? ' || upstreamCandidateOverflow[upstreamCandidateOverflowOffset] != 0u' : ''}` : 'false'};
  complete[completeOffset] = select(1u, 0u, loss${output.rows ? ' || localOverflow[localOverflowOffset] != 0u' : ''});
  ${output.status.candidateOverflow ? 'candidateOverflow[candidateOverflowOffset] = select(0u, 1u, loss);' : ''}`
        )
      );
    }
    return nodes;
  }
}

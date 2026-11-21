// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUVisibilityWorkflow,
  GPUReduction,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import type {GPUCompactOutput} from '../../analysis-core/kernels/gpu-contributor-types';
import {snapshotConstructionFields} from '../../analysis-core/kernels/construction-snapshot';
import {
  validateCapacity,
  validateCompactOutput,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validateRequirement,
  validateSinglePackedView
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  createWGSLKernelNode,
  type WGSLKernelBinding
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {validateGPUCompactPairPort} from '../../analysis-core/contracts/index';
import type {GPUCompactPairPort} from '../../analysis-core/contracts/index';
import {validateGraphChunkBindingRanges} from '../../analysis-core/kernels/graph-chunk-dispatch';
import {validateGPUStorageBindingByteLength} from '../../analysis-core/kernels/gpu-contributor-utils';
import {validateGeographicDistributionViews} from '../geographic-distribution/geographic-distribution-validation';
import {validateGeometryReadWriteViews} from '../geometry-edit/geometry-storage-validation';

const OPERATION = 'GPUPairGather';
const NO_FEATURE = 0xffffffff;

/** Value written to `rightRows` for unmatched left rows of a left join (and to unused output rows). */
export const GPU_PAIR_GATHER_NO_ROW = 0xffffffff;

/** Join type of {@link GPUPairGather}: `'inner'` keeps matched pairs, `'left'` also keeps unmatched left rows. */
export type GPUPairGatherHow = 'inner' | 'left';

/** One attribute column gathered by {@link GPUPairGather}. */
export type GPUPairGatherColumn = {
  /** Packed source column. */
  source: GraphDataView<'float32'> | GraphDataView<'uint32'>;
  /** Caller-owned destination with the source format and one row per output row capacity. */
  output: GraphDataView<'float32'> | GraphDataView<'uint32'>;
};

/**
 * Dense `[query, slot]` neighbor lists as written by `GPUNearestFeatureJoin` in neighbors mode;
 * flattened to one pair row per filled slot (GeoPandas `sjoin_nearest`).
 */
export type GPUPairGatherNeighbors = {
  /**
   * Neighbor feature row at `query * capacity + slot`, or `GPU_SPATIAL_JOIN_NO_FEATURE`.
   * Use nearest join's `neighborRows`, never its application-valued `neighborIds`.
   */
  rows: GraphDataView<'uint32'>;
  /** Number of filled slots per query. Its length is the query (left row) count. */
  counts: GraphDataView<'uint32'>;
  /** Slots per query: `rows.length` is `counts.length * capacity`. */
  capacity: number;
  /** Nearest producer completeness. Its aggregate overflow includes BVH, candidate and tie loss. */
  status?: {
    overflow: GraphDataView<'uint32'>;
    candidateOverflow?: GraphDataView<'uint32'>;
  };
  /** Nearest producer per-query search loss; one uint32 per query. */
  candidateTruncated?: GraphDataView<'uint32'>;
};

/** Caller-owned row outputs of {@link GPUPairGather}. */
export type GPUPairGatherOutput = {
  /** Left row per output row. Its length is the output row capacity. */
  leftRows: GraphDataView<'uint32'>;
  /** Right row per output row, {@link GPU_PAIR_GATHER_NO_ROW} for unmatched left rows. Same length. */
  rightRows: GraphDataView<'uint32'>;
  /** Completeness and diagnostic scalars. */
  status: {
    /** One-row live gathered count, including emitted unmatched rows and clamped to output capacity. */
    count: GraphDataView<'uint32'>;
    /** One-row scalar receiving `1` when final output capacity (including the inputs') was too small. */
    overflow: GraphDataView<'uint32'>;
    /** Candidate loss inherited from inputs; required when any supplied input carries candidateOverflow. */
    candidateOverflow?: GraphDataView<'uint32'>;
    /** Optional one-row input matched pair count, before clamping to output capacity. */
    matchedCount?: GraphDataView<'uint32'>;
  };
};

/**
 * Properties for {@link GPUPairGather}.
 *
 * Per-frame (no recompile): the contents of every input buffer. Compile-time: view lengths,
 * `how`, `neighbors.capacity` and which optional views are present.
 */
export type GPUPairGatherProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'pair-gather'`. */
  id?: string;
  /** `'inner'` (default) or `'left'`. */
  how?: GPUPairGatherHow;
  /** Matched pairs of a predicate join. Give exactly one of `pairs` and `neighbors`. */
  pairs?: GPUCompactPairPort;
  /** Dense neighbor lists of a nearest join. Give exactly one of `pairs` and `neighbors`. */
  neighbors?: GPUPairGatherNeighbors;
  /**
   * Left rows without a match, for `how: 'left'`: the `unmatched` output of a predicate join with
   * `how: 'anti'` or an inner join with output.unmatched. With `neighbors` it is optional:
   * queries with no neighbor are derived.
   */
  unmatchedLeft?: GPUCompactOutput;
  /** Left attribute columns, gathered by `leftRows`. Unmatched rows are never missing on the left. */
  leftColumns?: readonly GPUPairGatherColumn[];
  /** Right attribute columns, gathered by `rightRows`; unmatched left rows get NaN or `0xffffffff`. */
  rightColumns?: readonly GPUPairGatherColumn[];
  /**
   * Columns aligned with the input pair slots (`pairs.leftRows` rows, or `rows` slots for
   * `neighbors`) and carried to the output pair rows, for example `neighborDistances`
   * (`sjoin_nearest(distance_col=...)`) or join weights. Unmatched rows get NaN or `0xffffffff`.
   */
  slotColumns?: readonly GPUPairGatherColumn[];
  /** Extra final-output capacity flags ORed into `output.status.overflow`. */
  overflowSources?: readonly GraphDataView<'uint32'>[];
  /** Extra intermediate-loss flags propagated separately from final-output overflow. */
  candidateOverflowSources?: readonly GraphDataView<'uint32'>[];
  /** Caller-owned bounded row outputs. */
  output: GPUPairGatherOutput;
};

/**
 * Merges join output with attribute columns: the dataframe half of GeoPandas `sjoin` and
 * `sjoin_nearest` (`how='inner'` and `how='left'`, `distance_col`).
 *
 * Output rows `[0, matched)` are the matched pairs in input order (a predicate join emits them
 * sorted by `(left, right)`; neighbor lists by query then distance), followed, for `how: 'left'`,
 * by one row per unmatched left row in ascending order with `rightRows = GPU_PAIR_GATHER_NO_ROW`
 * and NaN (`float32`) or `0xffffffff` (`uint32`) in every right and slot column. This is a
 * concatenation, not GeoPandas' left-row order; sort by `(leftRows, rightRows)` to compare.
 *
 * With `neighbors`, filled slots (`slot < counts[query]`, feature not `GPU_SPATIAL_JOIN_NO_FEATURE`)
 * are flattened by a stable compaction. Columns are copied as raw 32-bit words, so NaN payloads
 * survive. Supplied nearest BVH, candidate and tie loss propagates as candidateOverflow.
 * Derived unmatched rows exclude queries flagged by candidateTruncated; without per-query flags,
 * aggregate loss withholds all derived unmatched rows. Legacy candidateOverflowSources also
 * withholds all derived unmatched rows because those flags have no query attribution.
 * Bounded by `output.leftRows.length`; surplus rows set `output.status.overflow`. Deterministic, no atomics.
 *
 * Consumes join row indices and attribute columns only, so it takes no `spatialContext`.
 */
export class GPUPairGather implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUPairGatherProps;
  /** Resolved join type. */
  readonly how: GPUPairGatherHow;
  /** The captured output.status.count view. */
  readonly outputCount: GraphDataView<'uint32'>;

  constructor(props: GPUPairGatherProps) {
    // Capture compile-time maps and scalar options; retain view identities for per-frame writes.
    props = snapshotConstructionFields(props, [
      'id',
      'how',
      'pairs',
      'neighbors',
      'unmatchedLeft',
      'leftColumns',
      'rightColumns',
      'slotColumns',
      'overflowSources',
      'candidateOverflowSources',
      'output'
    ]);
    const id = props.id ?? 'pair-gather';
    validateRequirement(
      id,
      !props.pairs || (!('leftIds' in props.pairs) && !('rightIds' in props.pairs)),
      'pairs.leftRows and pairs.rightRows are required; leftIds and rightIds are unsupported application-ID names'
    );
    validateRequirement(
      id,
      !props.neighbors || !('ids' in props.neighbors),
      'neighbors.rows is required and neighbors.ids is unsupported; supply feature rows, not application IDs'
    );
    const pairs =
      props.pairs && snapshotConstructionFields(props.pairs, ['leftRows', 'rightRows', 'status']);
    const unmatchedLeft =
      props.unmatchedLeft && snapshotConstructionFields(props.unmatchedLeft, ['ids', 'status']);
    const outputMap =
      props.output && snapshotConstructionFields(props.output, ['leftRows', 'rightRows', 'status']);
    const outputStatus =
      outputMap?.status &&
      snapshotConstructionFields(outputMap.status, [
        'count',
        'overflow',
        'candidateOverflow',
        'matchedCount'
      ]);
    validateRequirement(id, Boolean(outputStatus?.count), 'output.status.count is required');
    validateRequirement(
      id,
      !('rowCount' in props.output),
      'output.rowCount was removed; use output.status.count'
    );
    const neighborsMap =
      props.neighbors &&
      snapshotConstructionFields(props.neighbors, [
        'rows',
        'counts',
        'capacity',
        'status',
        'candidateTruncated'
      ]);
    props = Object.freeze({
      ...props,
      ...(neighborsMap
        ? {
            neighbors: Object.freeze({
              ...neighborsMap,
              ...(neighborsMap.status
                ? {
                    status: snapshotConstructionFields(neighborsMap.status, [
                      'overflow',
                      'candidateOverflow'
                    ])
                  }
                : {})
            })
          }
        : {}),
      ...(pairs
        ? {
            pairs: Object.freeze({
              ...pairs,
              status: snapshotConstructionFields(pairs.status, [
                'count',
                'overflow',
                'requiredCount',
                'candidateOverflow'
              ])
            })
          }
        : {}),
      ...(unmatchedLeft
        ? {
            unmatchedLeft: Object.freeze({
              ...unmatchedLeft,
              status: snapshotConstructionFields(unmatchedLeft.status, [
                'count',
                'overflow',
                'requiredCount',
                'candidateOverflow'
              ])
            })
          }
        : {}),
      ...(props.leftColumns
        ? {
            leftColumns: Object.freeze(
              props.leftColumns.map(column =>
                snapshotConstructionFields(column, ['source', 'output'])
              )
            )
          }
        : {}),
      ...(props.rightColumns
        ? {
            rightColumns: Object.freeze(
              props.rightColumns.map(column =>
                snapshotConstructionFields(column, ['source', 'output'])
              )
            )
          }
        : {}),
      ...(props.slotColumns
        ? {
            slotColumns: Object.freeze(
              props.slotColumns.map(column =>
                snapshotConstructionFields(column, ['source', 'output'])
              )
            )
          }
        : {}),
      ...(props.overflowSources
        ? {overflowSources: Object.freeze([...props.overflowSources])}
        : {}),
      ...(props.candidateOverflowSources
        ? {candidateOverflowSources: Object.freeze([...props.candidateOverflowSources])}
        : {}),
      output: Object.freeze({
        ...outputMap,
        status: outputStatus
      })
    });
    this.id = id;
    this.props = props;
    this.how = props.how ?? 'inner';
    const {how} = this;
    const {output} = props;
    this.outputCount = output.status.count;
    if (how !== 'inner' && how !== 'left') {
      throw new Error(`${id} how must be 'inner' or 'left'`);
    }
    if (Boolean(props.pairs) === Boolean(props.neighbors)) {
      throw new Error(`${id} needs exactly one of pairs and neighbors`);
    }
    if (how === 'left' && !props.unmatchedLeft && !props.neighbors) {
      throw new Error(
        `${id} how 'left' with pairs needs unmatchedLeft (predicate join output.unmatched)`
      );
    }
    if (how === 'inner' && props.unmatchedLeft) {
      throw new Error(`${id} unmatchedLeft needs how 'left'`);
    }
    for (const [name, view] of [
      ['leftRows', output.leftRows],
      ['rightRows', output.rightRows],
      ['status.count', this.outputCount],
      ['status.overflow', output.status.overflow],
      ['status.candidateOverflow', output.status.candidateOverflow],
      ['status.matchedCount', output.status.matchedCount]
    ] as const) {
      if (view) {
        validateSinglePackedView(view, ['uint32'], `${id} output.${name}`);
      }
    }
    const capacity = output.leftRows.length;
    if (capacity < 1) {
      throw new Error(`${id} output.leftRows must contain at least one row`);
    }
    if (output.rightRows.length !== capacity) {
      throw new Error(`${id} output.rightRows length must equal output.leftRows length`);
    }
    for (const [name, view] of [
      ['status.count', this.outputCount],
      ['status.overflow', output.status.overflow],
      ['status.candidateOverflow', output.status.candidateOverflow],
      ['status.matchedCount', output.status.matchedCount]
    ] as const) {
      if (view && view.length < 1) {
        throw new Error(`${id} output.${name} must contain one uint32 row`);
      }
    }
    let slotCount: number;
    if (props.pairs) {
      const {pairs} = props;
      validateGPUCompactPairPort(id, pairs);
      validateSinglePackedView(pairs.leftRows, ['uint32'], `${id} pairs.leftRows`);
      validateSinglePackedView(pairs.rightRows, ['uint32'], `${id} pairs.rightRows`);
      validateSinglePackedView(pairs.status.count, ['uint32'], `${id} pairs.count`);
      validateSinglePackedView(pairs.status.overflow, ['uint32'], `${id} pairs.overflow`);
      if (pairs.rightRows.length !== pairs.leftRows.length) {
        throw new Error(`${id} pairs.rightRows length must equal pairs.leftRows length`);
      }
      slotCount = pairs.leftRows.length;
    } else {
      const {neighbors} = props;
      validateRequirement(
        id,
        Boolean(neighbors?.rows) && !('ids' in neighbors!),
        'neighbors.rows is required and neighbors.ids is unsupported; supply feature rows, not application IDs'
      );
      validateSinglePackedView(neighbors!.rows, ['uint32'], `${id} neighbors.rows`);
      validateSinglePackedView(neighbors!.counts, ['uint32'], `${id} neighbors.counts`);
      validateCapacity(id, 'neighbors.capacity', neighbors!.capacity);
      if (neighbors!.rows.length !== neighbors!.counts.length * neighbors!.capacity) {
        throw new Error(`${id} neighbors.rows length must equal counts length * capacity`);
      }
      if (neighbors!.candidateTruncated) {
        validateSinglePackedView(
          neighbors!.candidateTruncated,
          ['uint32'],
          `${id} neighbors.candidateTruncated`
        );
        validateRequirement(
          id,
          neighbors!.candidateTruncated.length === neighbors!.counts.length,
          'neighbors.candidateTruncated needs one row per query'
        );
      }
      slotCount = neighbors!.rows.length;
    }
    if (slotCount < 1) {
      throw new Error(`${id} needs at least one pair slot`);
    }
    if (props.unmatchedLeft) {
      validateCompactOutput(id, props.unmatchedLeft);
    }
    for (const [name, columns, rows] of [
      ['leftColumns', props.leftColumns, capacity],
      ['rightColumns', props.rightColumns, capacity],
      ['slotColumns', props.slotColumns, capacity]
    ] as const) {
      for (const [index, column] of (columns ?? []).entries()) {
        const label = `${id} ${name}[${index}]`;
        validateSinglePackedView(column.source, ['float32', 'uint32'], label);
        if (column.source.format !== 'float32' && column.source.format !== 'uint32') {
          throw new Error(`${label} must be float32 or uint32`);
        }
        if (column.output.format !== column.source.format) {
          throw new Error(`${label} output format must equal source format`);
        }
        validateSinglePackedView(column.output, ['float32', 'uint32'], `${label} output`);
        if (column.output.length !== rows) {
          throw new Error(`${label} output length must equal output.leftRows length`);
        }
        if (name === 'slotColumns' && column.source.length < slotCount) {
          throw new Error(`${label} source must hold one row per pair slot (${slotCount})`);
        }
      }
    }
    for (const flag of props.overflowSources ?? []) {
      validateSinglePackedView(flag, ['uint32'], `${id} overflowSources`);
    }
    const candidateFlags = this._getCandidateFlags();
    validateRequirement(
      id,
      (candidateFlags.length === 0 && !props.neighbors?.candidateTruncated) ||
        Boolean(output.status.candidateOverflow),
      'output.status.candidateOverflow is required when inputs supply candidate overflow'
    );
    for (const flag of candidateFlags) {
      validateSinglePackedView(flag, ['uint32'], `${id} candidateOverflowSources`);
      validateRequirement(
        id,
        flag.length >= 1,
        'candidateOverflowSources must contain one uint32 row'
      );
    }
    validateGraphOutputsDisjointFromInputs(id, this._getOutputViews(), this._getInputViews());
    validateGeographicDistributionViews(
      id,
      {},
      Object.fromEntries(this._getOutputViews().map((view, index) => [`output.${index}`, view])),
      [
        {reads: {}, writes: {leftRows: output.leftRows, rightRows: output.rightRows}},
        {
          reads: {},
          writes: {
            'output.status.count': this.outputCount,
            'output.status.overflow': output.status.overflow,
            'output.status.matchedCount': output.status.matchedCount
          }
        }
      ]
    );
    validateGeometryReadWriteViews(
      id,
      [output.leftRows],
      (props.leftColumns ?? []).map(column => column.output)
    );
    validateGeometryReadWriteViews(
      id,
      [output.rightRows],
      (props.rightColumns ?? []).map(column => column.output)
    );
  }

  private _getOutputViews(): GraphDataView[] {
    const {output} = this.props;
    return [
      output.leftRows,
      output.rightRows,
      this.outputCount,
      output.status.overflow,
      ...(output.status.candidateOverflow ? [output.status.candidateOverflow] : []),
      ...(output.status.matchedCount ? [output.status.matchedCount] : []),
      ...[
        ...(this.props.leftColumns ?? []),
        ...(this.props.rightColumns ?? []),
        ...(this.props.slotColumns ?? [])
      ].map(column => column.output as GraphDataView)
    ];
  }

  private _getInputViews(): GraphDataView[] {
    const {props} = this;
    return [
      ...(props.pairs
        ? [
            props.pairs.leftRows,
            props.pairs.rightRows,
            props.pairs.status.count,
            props.pairs.status.overflow
          ]
        : [props.neighbors!.rows, props.neighbors!.counts]),
      ...(props.unmatchedLeft
        ? [
            props.unmatchedLeft.ids,
            props.unmatchedLeft.status.count,
            props.unmatchedLeft.status.overflow
          ]
        : []),
      ...(props.overflowSources ?? []),
      ...this._getCandidateFlags(),
      ...(props.neighbors?.candidateTruncated ? [props.neighbors.candidateTruncated] : []),
      ...[
        ...(props.leftColumns ?? []),
        ...(props.rightColumns ?? []),
        ...(props.slotColumns ?? [])
      ].map(column => column.source as GraphDataView)
    ];
  }

  private _getCandidateFlags(): GraphDataView<'uint32'>[] {
    const {props} = this;
    return [
      ...(props.pairs?.status.candidateOverflow ? [props.pairs.status.candidateOverflow] : []),
      ...(props.unmatchedLeft?.status.candidateOverflow
        ? [props.unmatchedLeft.status.candidateOverflow]
        : []),
      ...(props.neighbors?.status ? [props.neighbors.status.overflow] : []),
      ...(props.neighbors?.status?.candidateOverflow
        ? [props.neighbors.status.candidateOverflow]
        : []),
      ...(props.candidateOverflowSources ?? [])
    ];
  }

  /** Returns the optional flatten nodes, the row kernel, one kernel per column and the publish node. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {props, id, how} = this;
    const {output} = props;
    validateGraphChunkBindingRanges(
      graph,
      id,
      Object.fromEntries(
        [...this._getInputViews(), ...this._getOutputViews()].map((view, index) => [
          `binding.${index}`,
          view
        ])
      )
    );
    // Visibility creates scalar masks/identity/compaction offsets. Scan summaries shrink from
    // those planes. No nested allocation exceeds the larger neighbor-slot/query plane.
    const scratchRows = props.neighbors
      ? Math.max(props.neighbors.rows.length, props.neighbors.counts.length)
      : 1;
    validateGPUStorageBindingByteLength(
      graph.device,
      id,
      'pair visibility scratch',
      4 * Math.max(scratchRows, 1),
      'reduce neighbor slots or query count'
    );
    validateGraphViewsBelongToGraph(id, graph, [
      ...this._getInputViews(),
      ...this._getOutputViews()
    ]);
    const capacity = output.leftRows.length;
    const nodes: GPUCommandNode<Parameters>[] = [];

    // Stage 1 (neighbors): flatten filled slots to a compact slot list and count them.
    let slotIds: GraphDataView<'uint32'> | undefined;
    let matched: GraphDataView<'uint32'>;
    const neighbors = props.neighbors;
    if (neighbors) {
      const slotCount = neighbors.rows.length;
      const slotMask = createTransientView(graph, `${id}-slot-mask`, 'uint32', slotCount);
      slotIds = createTransientView(graph, `${id}-slot-ids`, 'uint32', slotCount);
      matched = createTransientView(graph, `${id}-matched`, 'uint32', 1);
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-slot-mask`,
          operation: OPERATION,
          variant: 'neighbors',
          bindings: [
            {name: 'neighborRows', view: neighbors.rows, type: 'u32', access: 'read'},
            {name: 'neighborCounts', view: neighbors.counts, type: 'u32', access: 'read'},
            {name: 'maskOut', view: slotMask, type: 'u32', access: 'read_write'}
          ],
          invocationCount: slotCount,
          declarations: `const CAPACITY: u32 = ${neighbors.capacity}u;
const NO_FEATURE: u32 = ${NO_FEATURE}u;`,
          body: `let query = index / CAPACITY;
  let slot = index % CAPACITY;
  let filled = slot < neighborCounts[neighborCountsOffset + query] &&
    neighborRows[neighborRowsOffset + index] != NO_FEATURE;
  maskOut[maskOutOffset + index] = select(0u, 1u, filled);`
        }),
        ...new GPUVisibilityWorkflow({
          id: `${id}-slots`,
          predicates: [{kind: 'selection', mask: slotMask}],
          output: slotIds,
          count: matched
        }).getCommandNodes(graph)
      );
    } else {
      matched = props.pairs!.status.count;
    }

    // Unmatched left rows: given, or derived from queries with no neighbor.
    let unmatchedIds: GraphDataView<'uint32'> | undefined;
    let unmatchedCount: GraphDataView<'uint32'> | undefined;
    const overflowFlags: GraphDataView<'uint32'>[] = [
      ...(props.pairs ? [props.pairs.status.overflow] : []),
      ...(props.overflowSources ?? [])
    ];
    if (how === 'left') {
      if (props.unmatchedLeft) {
        unmatchedIds = props.unmatchedLeft.ids;
        unmatchedCount = props.unmatchedLeft.status.count;
        overflowFlags.push(props.unmatchedLeft.status.overflow);
      } else {
        const queryCount = neighbors!.counts.length;
        const emptyMask = createTransientView(graph, `${id}-empty-mask`, 'uint32', queryCount);
        unmatchedIds = createTransientView(graph, `${id}-unmatched-ids`, 'uint32', queryCount);
        unmatchedCount = createTransientView(graph, `${id}-unmatched-count`, 'uint32', 1);
        nodes.push(
          createWGSLKernelNode<Parameters>(graph, {
            id: `${id}-empty-mask`,
            operation: OPERATION,
            variant: 'unmatched',
            bindings: [
              {name: 'neighborCounts', view: neighbors!.counts, type: 'u32', access: 'read'},
              ...(neighbors!.candidateTruncated
                ? [
                    {
                      name: 'truncated',
                      view: neighbors!.candidateTruncated,
                      type: 'u32',
                      access: 'read'
                    } as WGSLKernelBinding
                  ]
                : []),
              {name: 'maskOut', view: emptyMask, type: 'u32', access: 'read_write'}
            ],
            invocationCount: queryCount,
            body: `maskOut[maskOutOffset + index] =
    select(0u, 1u, neighborCounts[neighborCountsOffset + index] == 0u${neighbors!.candidateTruncated ? ' && truncated[truncatedOffset + index] == 0u' : ''});`
          })
        );
        // Per-query flags establish absence for unaffected queries. Extra legacy loss sources
        // have no row attribution and conservatively withhold every derived unmatched row.
        const globalLoss = [
          ...(neighbors!.candidateTruncated
            ? []
            : [
                ...(neighbors!.status ? [neighbors!.status.overflow] : []),
                ...(neighbors!.status?.candidateOverflow
                  ? [neighbors!.status.candidateOverflow]
                  : [])
              ]),
          ...(props.candidateOverflowSources ?? [])
        ];
        for (let start = 0; start < globalLoss.length; start += 7) {
          const group = globalLoss.slice(start, start + 7);
          nodes.push(
            createWGSLKernelNode<Parameters>(graph, {
              id: `${id}-unmatched-loss-${start / 7}`,
              operation: OPERATION,
              variant: 'unmatched-loss',
              invocationCount: queryCount,
              bindings: [
                ...group.map(
                  (view, flagIndex): WGSLKernelBinding => ({
                    name: `loss${flagIndex}`,
                    view,
                    type: 'u32',
                    access: 'read'
                  })
                ),
                {name: 'maskOut', view: emptyMask, type: 'u32', access: 'read_write'}
              ],
              body: `if (${group.map((_, flagIndex) => `loss${flagIndex}[loss${flagIndex}Offset] != 0u`).join(' || ')}) { maskOut[maskOutOffset + index] = 0u; }`
            })
          );
        }
        nodes.push(
          ...new GPUVisibilityWorkflow({
            id: `${id}-unmatched`,
            predicates: [{kind: 'selection', mask: emptyMask}],
            output: unmatchedIds,
            count: unmatchedCount
          }).getCommandNodes(graph)
        );
      }
    }

    // Stage 2: left and right row per output row.
    const rowBindings: WGSLKernelBinding[] = [
      {name: 'matchedIn', view: matched, type: 'u32', access: 'read'}
    ];
    let pairLookup: string;
    if (slotIds) {
      rowBindings.push({name: 'slotIds', view: slotIds, type: 'u32', access: 'read'});
    }
    if (neighbors) {
      rowBindings.push({name: 'neighborRows', view: neighbors.rows, type: 'u32', access: 'read'});
      pairLookup = `let slot = slotIds[slotIdsOffset + index];
    left = slot / ${neighbors.capacity}u;
    right = neighborRows[neighborRowsOffset + slot];`;
    } else {
      rowBindings.push(
        {name: 'pairLeft', view: props.pairs!.leftRows, type: 'u32', access: 'read'},
        {name: 'pairRight', view: props.pairs!.rightRows, type: 'u32', access: 'read'}
      );
      pairLookup = `left = pairLeft[pairLeftOffset + index];
    right = pairRight[pairRightOffset + index];`;
    }
    if (unmatchedIds && unmatchedCount) {
      rowBindings.push(
        {name: 'unmatchedIds', view: unmatchedIds, type: 'u32', access: 'read'},
        {name: 'unmatchedCount', view: unmatchedCount, type: 'u32', access: 'read'}
      );
    }
    rowBindings.push(
      {name: 'leftOut', view: output.leftRows, type: 'u32', access: 'read_write'},
      {name: 'rightOut', view: output.rightRows, type: 'u32', access: 'read_write'}
    );
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-rows`,
        operation: OPERATION,
        variant: 'rows',
        bindings: rowBindings,
        invocationCount: capacity,
        declarations: `const NO_ROW: u32 = ${GPU_PAIR_GATHER_NO_ROW}u;`,
        body: `let matched = matchedIn[matchedInOffset];
  var left = NO_ROW;
  var right = NO_ROW;
  if (index < matched) {
    ${pairLookup}
  }${
    unmatchedIds
      ? ` else {
    let unmatchedIndex = index - matched;
    if (unmatchedIndex < ${unmatchedIds.length}u &&
        unmatchedIndex < unmatchedCount[unmatchedCountOffset]) {
      left = unmatchedIds[unmatchedIdsOffset + unmatchedIndex];
    }
  }`
      : ''
  }
  leftOut[leftOutOffset + index] = left;
  rightOut[rightOutOffset + index] = right;`
      })
    );

    // Stage 3: columns, copied as raw words by one local kernel for both formats. The published
    // gather would write 0 instead of NaN for an unmatched float32 row.
    const columnNode = (name: string, column: GPUPairGatherColumn, rows: GraphDataView<'uint32'>) =>
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-${name}`,
        operation: OPERATION,
        variant: 'column',
        bindings: [
          {name: 'rowsIn', view: rows, type: 'u32', access: 'read'},
          {name: 'sourceIn', view: column.source as GraphDataView, type: 'u32', access: 'read'},
          {
            name: 'columnOut',
            view: column.output as GraphDataView,
            type: 'u32',
            access: 'read_write'
          }
        ],
        invocationCount: capacity,
        declarations: `const NO_ROW: u32 = ${GPU_PAIR_GATHER_NO_ROW}u;
const SOURCE_LENGTH: u32 = ${column.source.length}u;
const FILL: u32 = ${column.source.format === 'float32' ? '0x7fc00000u' : '0xffffffffu'};`,
        body: `let row = rowsIn[rowsInOffset + index];
  var value = FILL;
  if (row != NO_ROW && row < SOURCE_LENGTH) {
    value = sourceIn[sourceInOffset + row];
  }
  columnOut[columnOutOffset + index] = value;`
      });
    (props.leftColumns ?? []).forEach((column, index) =>
      nodes.push(columnNode(`left-column-${index}`, column, output.leftRows))
    );
    (props.rightColumns ?? []).forEach((column, index) =>
      nodes.push(columnNode(`right-column-${index}`, column, output.rightRows))
    );
    (props.slotColumns ?? []).forEach((column, index) => {
      const slotBindings: WGSLKernelBinding[] = [
        {name: 'matchedIn', view: matched, type: 'u32', access: 'read'},
        {name: 'sourceIn', view: column.source as GraphDataView, type: 'u32', access: 'read'},
        {name: 'columnOut', view: column.output as GraphDataView, type: 'u32', access: 'read_write'}
      ];
      if (slotIds) {
        slotBindings.push({name: 'slotIds', view: slotIds, type: 'u32', access: 'read'});
      }
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-slot-column-${index}`,
          operation: OPERATION,
          variant: 'slot-column',
          bindings: slotBindings,
          invocationCount: capacity,
          declarations: `const FILL: u32 = ${column.source.format === 'float32' ? '0x7fc00000u' : '0xffffffffu'};`,
          body: `var value = FILL;
  if (index < matchedIn[matchedInOffset]) {
    ${slotIds ? 'let slot = slotIds[slotIdsOffset + index];' : 'let slot = index;'}
    value = sourceIn[sourceInOffset + slot];
  }
  columnOut[columnOutOffset + index] = value;`
        })
      );
    });

    // Stage 4: clamp counts and publish flags. The publish kernel binds 3 to 5 fixed views plus
    // one per flag; above the core limit of 8, the flags are first OR-ed into one word, 7 at a time.
    const fixedBindingCount = 3 + (unmatchedCount ? 1 : 0) + (output.status.matchedCount ? 1 : 0);
    if (fixedBindingCount + overflowFlags.length > 8) {
      const combined = createTransientView(graph, `${id}-overflow-any`, 'uint32', 1);
      for (let start = 0; start < overflowFlags.length; start += 7) {
        const group = overflowFlags.slice(start, start + 7);
        const anyFlag = group
          .map((_, flagIndex) => `flagIn${flagIndex}[flagIn${flagIndex}Offset] != 0u`)
          .join(' || ');
        nodes.push(
          createWGSLKernelNode<Parameters>(graph, {
            id: `${id}-overflow-any-${start / 7}`,
            operation: OPERATION,
            variant: 'overflow-any',
            bindings: [
              ...group.map(
                (flag, flagIndex): WGSLKernelBinding => ({
                  name: `flagIn${flagIndex}`,
                  view: flag,
                  type: 'u32',
                  access: 'read'
                })
              ),
              {name: 'anyOut', view: combined, type: 'u32', access: 'read_write'}
            ],
            invocationCount: 1,
            body: `anyOut[anyOutOffset] = ${start === 0 ? '' : 'anyOut[anyOutOffset] | '}select(0u, 1u, ${anyFlag});`
          })
        );
      }
      overflowFlags.splice(0, overflowFlags.length, combined);
    }
    const publishBindings: WGSLKernelBinding[] = [
      {name: 'matchedIn', view: matched, type: 'u32', access: 'read'}
    ];
    if (unmatchedCount) {
      publishBindings.push({
        name: 'unmatchedCount',
        view: unmatchedCount,
        type: 'u32',
        access: 'read'
      });
    }
    overflowFlags.forEach((flag, flagIndex) =>
      publishBindings.push({name: `flagIn${flagIndex}`, view: flag, type: 'u32', access: 'read'})
    );
    publishBindings.push(
      {name: 'rowCountOut', view: this.outputCount, type: 'u32', access: 'read_write'},
      {name: 'overflowOut', view: output.status.overflow, type: 'u32', access: 'read_write'}
    );
    if (output.status.matchedCount) {
      publishBindings.push({
        name: 'matchedOut',
        view: output.status.matchedCount,
        type: 'u32',
        access: 'read_write'
      });
    }
    const overflowExpression = [
      'total > CAPACITY',
      ...overflowFlags.map((_, flagIndex) => `flagIn${flagIndex}[flagIn${flagIndex}Offset] != 0u`)
    ].join(' || ');
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-publish`,
        operation: OPERATION,
        variant: 'publish',
        bindings: publishBindings,
        invocationCount: 1,
        declarations: `const CAPACITY: u32 = ${capacity}u;`,
        body: `let matched = matchedIn[matchedInOffset];
  let total = matched + ${unmatchedCount ? 'unmatchedCount[unmatchedCountOffset]' : '0u'};
  rowCountOut[rowCountOutOffset] = min(total, CAPACITY);
  overflowOut[overflowOutOffset] = select(0u, 1u, ${overflowExpression});
  ${output.status.matchedCount ? 'matchedOut[matchedOutOffset] = matched;' : ''}`
      })
    );
    // Candidate loss cannot be recovered by growing the gathered row capacity. Publish it
    // separately, seven inputs per pass so each pass stays within eight storage bindings.
    if (output.status.candidateOverflow) {
      const candidateFlags = this._getCandidateFlags();
      if (neighbors?.candidateTruncated) {
        const truncatedMaximum = createTransientView(graph, `${id}-truncated-maximum`, 'uint32', 1);
        nodes.push(
          ...new GPUReduction({
            id: `${id}-truncated`,
            input: neighbors.candidateTruncated,
            output: truncatedMaximum,
            operation: 'max'
          }).getCommandNodes(graph)
        );
        candidateFlags.push(truncatedMaximum);
      }
      for (let start = 0; start < Math.max(candidateFlags.length, 1); start += 7) {
        const group = candidateFlags.slice(start, start + 7);
        const expression =
          group
            .map((_, flagIndex) => `candidateIn${flagIndex}[candidateIn${flagIndex}Offset] != 0u`)
            .join(' || ') || 'false';
        nodes.push(
          createWGSLKernelNode<Parameters>(graph, {
            id: `${id}-candidate-overflow-${start / 7}`,
            operation: OPERATION,
            variant: 'candidate-overflow',
            bindings: [
              ...group.map(
                (flag, flagIndex): WGSLKernelBinding => ({
                  name: `candidateIn${flagIndex}`,
                  view: flag,
                  type: 'u32',
                  access: 'read'
                })
              ),
              {
                name: 'candidateOverflowOut',
                view: output.status.candidateOverflow,
                type: 'u32',
                access: 'read_write'
              }
            ],
            invocationCount: 1,
            body: `candidateOverflowOut[candidateOverflowOutOffset] = ${start === 0 ? '' : 'candidateOverflowOut[candidateOverflowOutOffset] | '}select(0u, 1u, ${expression});`
          })
        );
      }
    }
    return nodes;
  }
}

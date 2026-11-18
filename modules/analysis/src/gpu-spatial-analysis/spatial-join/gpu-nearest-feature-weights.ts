// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {validateGraphChunkBindingRanges} from '../../analysis-core/kernels/graph-chunk-dispatch';

import {
  getStorageTopKAccessors,
  createStorageTopKNode
} from '../../analysis-core/kernels/storage-backed-top-k';

import {
  GPU_PLANAR_OR_GREAT_CIRCLE_SPATIAL_CONTEXT_SUPPORT,
  resolveGPUSpatialContext,
  type GPUSpatialContext
} from '../../analysis-core/contracts/spatial-context';
import {
  createTransientView,
  GPUScan,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import type {GPUCommandNodeProducer} from '@luma.gl/graph/gpu-core';
import {
  getWGSLFloatLiteral,
  type WGSLKernelProps
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  validateGPUStorageBindingByteLength,
  validateMaximumCount,
  validateGraphViewsBelongToGraph,
  validateSinglePackedView,
  validateCapacity,
  validateCommandNodeBudget,
  validateRequirement,
  validateFiniteNumber
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  describeGPUSpatialWeights,
  type GPUSpatialWeights,
  type GPUSpatialWeightsOutput,
  validateGPUSpatialWeights
} from '../spatial-weights/spatial-weights';
import {GPU_INVERSE_DISTANCE_DEFAULT_DISTANCE_FLOOR} from '../neighbor-search/neighbor-search-parameters';
import {GPU_SPATIAL_JOIN_NO_FEATURE} from './spatial-join-types';
import {validateStatusScalars} from './spatial-join-geometry';
import {validateDisjointOutputs} from './spatial-join-passes';
import {
  snapshotConstructionFields,
  snapshotConstructionSpatialContext
} from '../../analysis-core/kernels/construction-snapshot';
import type {
  GPUCandidateStatusPort,
  GPUOverflowStatusPort
} from '../../analysis-core/contracts/status';
import {validateSpatialJoinReadWriteBindings} from './spatial-join-binding-preflight';
import {snapshotJoinWeights} from './spatial-join-construction-snapshot';
import {
  NEAREST_PRIVATE_NEIGHBOR_CAPACITY,
  NEAREST_MAXIMUM_NEIGHBOR_CAPACITY
} from './nearest-types';

/** Contexts of the producing join's distances and this adapter's distance floor. */
const NEAREST_FEATURE_WEIGHTS_SPATIAL_CONTEXT_SUPPORT = Object.freeze({
  ...GPU_PLANAR_OR_GREAT_CIRCLE_SPATIAL_CONTEXT_SUPPORT,
  measure: 'distance'
} as const);

const OPERATION = 'GPUNearestFeatureWeights';

/** Slot indices use the producing join's packed uint32 scratch layout. */
const MAXIMUM_SLOTS = NEAREST_MAXIMUM_NEIGHBOR_CAPACITY;
/** The maximum CSR total must fit uint32 before its exclusive scan. */
const MAXIMUM_TOTAL_SLOTS = 0xffffffff;
/** A storage-sort batch must contain at least one query to advance its cursor. */
const MINIMUM_SCRATCH_ROW_COUNT = 1;
/** Covers the recursive published scan plus counts and offset publication for uint32 rows. */
const SCAN_COMMAND_NODE_ALLOWANCE = 64;
/** POWER and FLOOR are WGSL f32 literals, so finite JS doubles alone are insufficient. */
const MAXIMUM_SHADER_FLOAT = 3.4028234663852886e38;
/** Positive weight parameters must stay nonzero after conversion to f32. */
const MINIMUM_SHADER_POSITIVE_FLOAT = 1.401298464324817e-45;

/**
 * Properties for {@link GPUNearestFeatureWeights}.
 *
 * The `neighbor*` views are the outputs of a {@link GPUNearestFeatureJoin} in neighbors mode,
 * passed with the join's `neighborCapacity` as `slotCapacity`. Compile-time: every prop except the
 * contents of the input views.
 */
export type GPUNearestFeatureWeightsProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'nearest-feature-weights'`. */
  id?: string;
  /** Resolved context of the producing join; distanceFloor uses its length unit. Defaults to planar. */
  spatialContext?: GPUSpatialContext;
  /** `neighborRows` of the join: `queryCount * slotCapacity` feature row indices, padded with `GPU_SPATIAL_JOIN_NO_FEATURE`. Never supply remapped application IDs. */
  neighborRows: GraphDataView<'uint32'>;
  /** `neighborCounts` of the join: filled slots per query. Its length is the row count. */
  neighborCounts: GraphDataView<'uint32'>;
  /** Producing join completeness. Any upstream loss becomes output.status.candidateOverflow; supplying this port requires that output scalar. */
  neighborStatus?: GPUOverflowStatusPort & GPUCandidateStatusPort;
  /**
   * `neighborDistances` of the join. Required for `'inverse-distance'` weights and for
   * `weights.distances`.
   */
  neighborDistances?: GraphDataView<'float32'>;
  /** The join's `neighborCapacity`: slots per query in the input layout. A positive integer; above 64 the adapter sorts in storage scratch. */
  slotCapacity: number;
  /** Compile-time maximum queries per storage-sort dispatch; defaults to the device-sized batch. */
  scratchChunkRowCount?: number;
  /**
   * Neighbors kept per row, taken in the join's `(distance, feature row)` order. Defaults to
   * `slotCapacity`, or to `slotCapacity - 1` with `excludeSelf`. Rows with fewer valid slots keep
   * fewer (no padding entries are written).
   */
  k?: number;
  /**
   * Self-join: drop the slot whose feature row equals the query row (libpysal `KNN` never lists a
   * row as its own neighbor). Run the join with `k + 1` neighbors so `k` remain.
   * Defaults to false (cross weights: query rows to feature rows).
   */
  excludeSelf?: boolean;
  /**
   * Weight rule. `'binary'`: 1 per neighbor (libpysal `KNN` default). `'inverse-distance'`:
   * `1 / max(d, distanceFloor)^power`. Defaults to `'binary'`.
   */
  weightType?: 'binary' | 'inverse-distance';
  /** Exponent of the inverse-distance rule. A positive finite number. Defaults to 1. */
  power?: number;
  /**
   * Lower bound on the distance in the inverse-distance rule, so coincident features get a finite
   * weight (libpysal would divide by zero). A positive finite number. Defaults to `1e-6`, the
   * default of `GPUNeighborSearch` inverse-distance weights too: coincident features weigh
   * `1e6^power`.
   */
  distanceFloor?: number;
  /**
   * Caller-owned output CSR with `neighborCounts.length + 1` offsets. Neighbor rows ascend within
   * each row. `distances` is filled when present. The capacity (`neighbors.length`) must be at least
   * `1`; `rows * (k)` always suffices. `status.overflow` is 1 when the kept neighbors did not fit;
   * `status.candidateOverflow` receives upstream join loss (including truncated tie slots), or zero
   * without neighborStatus. It is required when neighborStatus is supplied. Omit status.requiredCount.
   */
  output: GPUSpatialWeightsOutput;
};

/**
 * Adapter from the k-nearest output of {@link GPUNearestFeatureJoin} to a {@link GPUSpatialWeights}
 * CSR: libpysal 4.15 `KNN` / `Graph.build_knn` for features of any kind (points, lines, polygons),
 * with the join's exact geometry distances.
 *
 * Per row it keeps the first `k` valid slots in the join's `(distance, feature row)` order, drops
 * padding (`GPU_SPATIAL_JOIN_NO_FEATURE`, slots past `neighborCounts`) and, with `excludeSelf`, the
 * row's own ID, then sorts the kept IDs ascending as the CSR invariants require. Rows with fewer
 * than `k` available neighbors are simply shorter. The result is deterministic and exact.
 *
 * Distances and `distanceFloor` use the producing join's length unit. Pass its resolved
 * `spatialContext` to carry that unit into `outputWeights` and subsequent transforms.
 *
 * The weights are directed: a nearest-neighbor relation is generally not symmetric, so this
 * output does not meet the symmetric-pattern requirement of the `'symmetrize'` transform. Take the
 * union of the weights with their transpose (`GPUSpatialWeightsAlgebra` `'union'` with
 * `GPUSpatialWeightsTranspose`) first. Deviation from libpysal: binary weights are 1 and
 * inverse-distance weights floor the distance instead of dividing by zero.
 */
export class GPUNearestFeatureWeights implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUNearestFeatureWeightsProps;
  /** Number of rows (queries). */
  readonly rows: number;
  /** Resolved neighbors kept per row. */
  readonly k: number;
  /** Upper bound including every storage sorting chunk and the scan. */
  readonly estimatedCommandNodeCount: number;
  /** Resolved context of the producing join. */
  readonly spatialContext: GPUSpatialContext;
  /** Output CSR carrying the producing join context and weights metadata. */
  readonly outputWeights: GPUSpatialWeights;

  constructor(props: GPUNearestFeatureWeightsProps) {
    props = snapshotConstructionFields(props, [
      'id',
      'spatialContext',
      'neighborRows',
      'neighborIds',
      'neighborCounts',
      'neighborDistances',
      'neighborStatus',
      'slotCapacity',
      'scratchChunkRowCount',
      'k',
      'excludeSelf',
      'weightType',
      'power',
      'distanceFloor',
      'output'
    ] as readonly (keyof GPUNearestFeatureWeightsProps)[]);
    const id = props.id ?? 'nearest-feature-weights';
    const {slotCapacity} = props;
    validateCapacity(id, 'slotCapacity', slotCapacity, 1, MAXIMUM_SLOTS);
    const output = snapshotJoinWeights(props.output);
    props = Object.freeze({
      ...props,
      slotCapacity,
      ...(props.spatialContext
        ? {spatialContext: snapshotConstructionSpatialContext(props.spatialContext)}
        : {}),
      ...(props.neighborStatus
        ? {
            neighborStatus: snapshotConstructionFields(props.neighborStatus, [
              'overflow',
              'candidateOverflow'
            ])
          }
        : {}),
      output
    });
    this.id = id;
    validateRequirement(
      id,
      Boolean(props.neighborRows) && !('neighborIds' in props),
      'neighborRows is required and neighborIds is unsupported; supply feature rows, not application IDs'
    );
    this.props = props;
    this.spatialContext = resolveGPUSpatialContext(
      id,
      props.spatialContext,
      NEAREST_FEATURE_WEIGHTS_SPATIAL_CONTEXT_SUPPORT
    );
    if (props.scratchChunkRowCount !== undefined) {
      validateCapacity(
        id,
        'scratchChunkRowCount',
        props.scratchChunkRowCount,
        MINIMUM_SCRATCH_ROW_COUNT,
        MAXIMUM_TOTAL_SLOTS
      );
    }
    this.k = props.k ?? (props.excludeSelf ? Math.max(slotCapacity - 1, 1) : slotCapacity);
    validateCapacity(id, 'k', this.k, 1, slotCapacity);
    validateSinglePackedView(props.neighborRows, ['uint32'], `${id} neighborRows`);
    validateSinglePackedView(props.neighborCounts, ['uint32'], `${id} neighborCounts`);
    this.rows = props.neighborCounts.length;
    validateMaximumCount(id, 'rows * slotCapacity', this.rows * slotCapacity, MAXIMUM_TOTAL_SLOTS);
    if (this.rows < 1) {
      throw new Error(`${id} neighborCounts needs at least one row`);
    }
    if (props.neighborRows.length !== this.rows * slotCapacity) {
      throw new Error(`${id} neighborRows needs rows * slotCapacity rows`);
    }
    const scratchRows = Math.max(
      1,
      Math.floor(
        ((props.neighborRows.buffer.graph as GPUCommandGraph).device.limits
          .maxStorageBufferBindingSize /
          4 -
          1) /
          (this.k * 2)
      )
    );
    this.estimatedCommandNodeCount =
      SCAN_COMMAND_NODE_ALLOWANCE +
      Number(Boolean(props.output.status.candidateOverflow)) +
      (slotCapacity > NEAREST_PRIVATE_NEIGHBOR_CAPACITY
        ? 2 *
          Math.ceil(this.rows / Math.min(scratchRows, props.scratchChunkRowCount ?? scratchRows))
        : 1);
    validateCommandNodeBudget(id, this.estimatedCommandNodeCount);
    const weightType = props.weightType ?? 'binary';
    if (weightType !== 'binary' && weightType !== 'inverse-distance') {
      throw new Error(`${id} weightType must be 'binary' or 'inverse-distance'`);
    }
    if (props.neighborDistances) {
      validateSinglePackedView(props.neighborDistances, ['float32'], `${id} neighborDistances`);
      if (props.neighborDistances.length !== props.neighborRows.length) {
        throw new Error(`${id} neighborDistances needs one row per slot`);
      }
    }
    if ((weightType === 'inverse-distance' || props.output.distances) && !props.neighborDistances) {
      throw new Error(`${id} needs neighborDistances for distances`);
    }
    for (const [name, value] of [
      ['power', props.power],
      ['distanceFloor', props.distanceFloor]
    ] as const) {
      if (value !== undefined) {
        validateFiniteNumber(id, name, value, MINIMUM_SHADER_POSITIVE_FLOAT, MAXIMUM_SHADER_FLOAT);
      }
    }
    if (validateGPUSpatialWeights(id, props.output) !== this.rows) {
      throw new Error(`${id} output.offsets needs neighborCounts.length + 1 rows`);
    }
    this.outputWeights = describeGPUSpatialWeights(props.output, {
      spatialContext: this.spatialContext,
      information: {
        rowCount: this.rows,
        transform: weightType === 'binary' ? 'binary' : 'raw',
        symmetric: 'unknown',
        hasSelfWeights: !props.excludeSelf,
        provenance: {source: OPERATION, criterion: 'knn', k: this.k}
      }
    });
    const {status} = props.output;
    if (status.requiredCount) {
      throw new Error(`${id} does not write output.status.requiredCount`);
    }
    validateRequirement(
      id,
      !props.neighborStatus || Boolean(props.neighborStatus.overflow),
      'neighborStatus.overflow is required when neighborStatus is supplied'
    );
    validateRequirement(
      id,
      !props.neighborStatus || Boolean(status.candidateOverflow),
      'output.status.candidateOverflow is required with neighborStatus'
    );
    validateStatusScalars(id, {
      overflow: status.overflow,
      candidateOverflow: status.candidateOverflow
    });
    if (props.neighborStatus) validateStatusScalars(`${id} neighborStatus`, props.neighborStatus);
    validateSpatialJoinReadWriteBindings(
      id,
      [props.neighborStatus?.overflow, props.neighborStatus?.candidateOverflow],
      [status.candidateOverflow]
    );
    validateSpatialJoinReadWriteBindings(
      id,
      [props.output.offsets],
      [props.output.neighbors, props.output.weights, props.output.distances]
    );
    validateDisjointOutputs(
      id,
      [
        props.neighborRows,
        props.neighborCounts,
        props.neighborDistances,
        props.neighborStatus?.overflow,
        props.neighborStatus?.candidateOverflow
      ],
      [
        props.output.offsets,
        props.output.neighbors,
        props.output.weights,
        props.output.distances,
        status.overflow,
        status.candidateOverflow
      ],
      [
        {offsets: props.output.offsets, overflow: status.overflow},
        {
          neighbors: props.output.neighbors,
          weights: props.output.weights,
          distances: props.output.distances
        },
        {candidateOverflow: status.candidateOverflow}
      ]
    );
  }

  /** Returns the adapter nodes in dependency order. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, rows, k} = this;
    const {output: weights, slotCapacity} = props;
    validateGraphViewsBelongToGraph(id, graph, [
      props.neighborRows,
      props.neighborCounts,
      props.neighborDistances,
      weights.offsets,
      weights.neighbors,
      weights.weights,
      weights.distances,
      weights.status.overflow,
      weights.status.candidateOverflow,
      props.neighborStatus?.overflow,
      props.neighborStatus?.candidateOverflow
    ]);
    const storageBacked = slotCapacity > NEAREST_PRIVATE_NEIGHBOR_CAPACITY;
    validateGraphChunkBindingRanges(graph, id, {
      neighborRows: props.neighborRows,
      neighborCounts: props.neighborCounts,
      neighborDistances: props.neighborDistances,
      offsets: weights.offsets,
      neighbors: weights.neighbors,
      weights: weights.weights,
      distances: weights.distances,
      overflow: weights.status.overflow,
      candidateOverflow: weights.status.candidateOverflow,
      neighborOverflow: props.neighborStatus?.overflow,
      neighborCandidateOverflow: props.neighborStatus?.candidateOverflow
    });
    validateGPUStorageBindingByteLength(
      graph.device,
      id,
      'counts and scan scratch',
      rows * 4,
      'reduce neighborCounts length'
    );
    const scratchRows = Math.min(
      rows,
      props.scratchChunkRowCount ?? rows,
      Math.max(1, Math.floor((graph.device.limits.maxStorageBufferBindingSize / 4 - 1) / (k * 2)))
    );
    const cursorWord = scratchRows * k * 2;
    validateGPUStorageBindingByteLength(
      graph.device,
      id,
      'kept neighbor scratch',
      storageBacked ? (cursorWord + 1) * 4 : 0,
      'reduce rows or k'
    );
    const keptScratch = storageBacked
      ? createTransientView(graph, `${id}-kept`, 'uint32', cursorWord + 1)
      : undefined;
    const storageList = getStorageTopKAccessors(
      'keptScratch',
      'keptScratchOffset + scratchIndex * KEEP * 2u'
    );
    const readId = (slot: string) => (keptScratch ? storageList.readRow(slot) : `keptIds[${slot}]`);
    const readDistance = (slot: string) =>
      keptScratch ? storageList.readDistance(slot) : `keptDistances[${slot}]`;
    const writeId = (slot: string, value: string) =>
      keptScratch ? storageList.writeRow(slot, value) : `keptIds[${slot}] = ${value};`;
    const writeDistance = (slot: string, value: string) =>
      keptScratch ? storageList.writeDistance(slot, value) : `keptDistances[${slot}] = ${value};`;
    const capacity = weights.neighbors.length;
    const counts = createTransientView(graph, `${id}-counts`, 'uint32', rows);
    const starts = createTransientView(graph, `${id}-starts`, 'uint32', rows);
    const inverse = (props.weightType ?? 'binary') === 'inverse-distance';
    const declarations = `const SLOTS: u32 = ${slotCapacity}u;
const KEEP: u32 = ${k}u;
const EXCLUDE_SELF: bool = ${props.excludeSelf === true};
const NO_FEATURE: u32 = ${GPU_SPATIAL_JOIN_NO_FEATURE}u;
const POWER: f32 = ${getWGSLFloatLiteral(props.power ?? 1)};
const FLOOR: f32 = ${getWGSLFloatLiteral(props.distanceFloor ?? GPU_INVERSE_DISTANCE_DEFAULT_DISTANCE_FLOOR)};`;
    const inputBindings = [
      {name: 'ids', view: props.neighborRows, type: 'u32' as const, access: 'read' as const},
      {
        name: 'filled',
        view: props.neighborCounts,
        type: 'u32' as const,
        access: 'read' as const
      },
      ...(props.neighborDistances
        ? [
            {
              name: 'slotDistances',
              view: props.neighborDistances,
              type: 'f32' as const,
              access: 'read' as const
            }
          ]
        : [])
    ];
    // Visits the valid slots of row `index` in join order, stopping after KEEP kept neighbors.
    const visit = (onKeep: string) => `let filledCount = min(filled[filledOffset + index], SLOTS);
  var kept = 0u;
  for (var slot = 0u; slot < filledCount && kept < KEEP; slot++) {
    let neighbor = ids[idsOffset + index * SLOTS + slot];
    if (neighbor == NO_FEATURE || (EXCLUDE_SELF && neighbor == index)) {
      continue;
    }
    ${onKeep}
    kept++;
  }`;
    const nodes = [
      createStorageTopKNode<Parameters>(graph, {
        id: `${id}-counts`,
        operation: OPERATION,
        variant: 'counts',
        bindings: [
          ...inputBindings.slice(0, 2),
          {name: 'counts', view: counts, type: 'u32', access: 'read_write'}
        ],
        invocationCount: rows,
        declarations,
        body: `${visit('')}
  counts[countsOffset + index] = kept;`
      }),
      ...new GPUScan({
        id: `${id}-scan`,
        input: counts,
        output: starts,
        mode: 'exclusive'
      }).getCommandNodes(graph),
      createStorageTopKNode<Parameters>(graph, {
        id: `${id}-offsets`,
        operation: OPERATION,
        variant: 'offsets',
        bindings: [
          {name: 'counts', view: counts, type: 'u32', access: 'read'},
          {name: 'starts', view: starts, type: 'u32', access: 'read'},
          {name: 'offsets', view: weights.offsets, type: 'u32', access: 'read_write'},
          {name: 'overflow', view: weights.status.overflow, type: 'u32', access: 'read_write'}
        ],
        invocationCount: rows + 1,
        declarations: `const ROWS: u32 = ${rows}u;
const CAPACITY: u32 = ${capacity}u;`,
        body: `if (index < ROWS) {
    offsets[offsetsOffset + index] = min(starts[startsOffset + index], CAPACITY);
  } else {
    let total = starts[startsOffset + ROWS - 1u] + counts[countsOffset + ROWS - 1u];
    offsets[offsetsOffset + ROWS] = min(total, CAPACITY);
    overflow[overflowOffset] = select(0u, 1u, total > CAPACITY);
  }`
      })
    ];
    if (weights.status.candidateOverflow) {
      nodes.push(
        createStorageTopKNode<Parameters>(graph, {
          id: `${id}-candidate-status`,
          operation: OPERATION,
          variant: 'candidate-status',
          bindings: [
            ...(props.neighborStatus
              ? [
                  {
                    name: 'upstreamOverflow',
                    view: props.neighborStatus.overflow,
                    type: 'u32' as const,
                    access: 'read' as const
                  }
                ]
              : []),
            ...(props.neighborStatus?.candidateOverflow
              ? [
                  {
                    name: 'upstreamCandidateOverflow',
                    view: props.neighborStatus.candidateOverflow,
                    type: 'u32' as const,
                    access: 'read' as const
                  }
                ]
              : []),
            {
              name: 'candidateOverflow',
              view: weights.status.candidateOverflow,
              type: 'u32',
              access: 'read_write'
            }
          ],
          invocationCount: 1,
          body: `candidateOverflow[candidateOverflowOffset] = select(0u, 1u, ${props.neighborStatus ? 'upstreamOverflow[upstreamOverflowOffset] != 0u' : 'false'} || ${props.neighborStatus?.candidateOverflow ? 'upstreamCandidateOverflow[upstreamCandidateOverflowOffset] != 0u' : 'false'});`
        })
      );
    }
    if (keptScratch) {
      nodes.push(
        createStorageTopKNode(graph, {
          id: `${id}-reset-cursor`,
          operation: OPERATION,
          variant: 'reset-sort-cursor',
          bindings: [{name: 'keptScratch', view: keptScratch, type: 'u32', access: 'read_write'}],
          invocationCount: 1,
          body: `keptScratch[keptScratchOffset + ${cursorWord}u] = 0u;`
        })
      );
    }

    const emitProps: WGSLKernelProps<Parameters> = {
      id: `${id}-emit`,
      operation: OPERATION,
      variant: 'emit',
      bindings: [
        ...inputBindings,
        ...(keptScratch
          ? [{name: 'keptScratch', view: keptScratch, type: 'u32', access: 'read_write'} as const]
          : []),
        {name: 'offsets', view: weights.offsets, type: 'u32', access: 'read'},
        {name: 'neighbors', view: weights.neighbors, type: 'u32', access: 'read_write'},
        {name: 'values', view: weights.weights, type: 'f32', access: 'read_write'},
        ...(weights.distances
          ? [
              {
                name: 'outDistances',
                view: weights.distances,
                type: 'f32' as const,
                access: 'read_write' as const
              }
            ]
          : [])
      ],
      invocationCount: keptScratch ? scratchRows : rows,
      declarations,
      body: `${keptScratch ? '' : `var keptIds: array<u32, ${NEAREST_PRIVATE_NEIGHBOR_CAPACITY}>;\n  var keptDistances: array<f32, ${NEAREST_PRIVATE_NEIGHBOR_CAPACITY}>;`}
  ${visit(`var position = kept;
    let distance = ${props.neighborDistances ? 'slotDistances[slotDistancesOffset + index * SLOTS + slot]' : '0.0'};
    // Insertion sort by neighbor ID keeps the CSR rows ascending.
    while (position > 0u && ${readId('position - 1u')} > neighbor) {
      ${writeId('position', readId('position - 1u'))}
      ${writeDistance('position', readDistance('position - 1u'))}
      position--;
    }
    ${writeId('position', 'neighbor')}
    ${writeDistance('position', 'distance')}`)}
  let begin = offsets[offsetsOffset + index];
  let end = offsets[offsetsOffset + index + 1u];
  for (var position = 0u; position < kept && begin + position < end; position++) {
    let outSlot = begin + position;
    neighbors[neighborsOffset + outSlot] = ${readId('position')};
    ${
      inverse
        ? `let safeDistance = max(${readDistance('position')}, FLOOR);\n    let weight = 1.0 / pow(safeDistance, POWER);\n    values[valuesOffset + outSlot] = select(0.0, weight, weight < 3.0e38);`
        : 'values[valuesOffset + outSlot] = 1.0;'
    }
    ${weights.distances ? `outDistances[outDistancesOffset + outSlot] = ${readDistance('position')};` : ''}
  }`
    };

    for (let chunk = 0; chunk < (keptScratch ? Math.ceil(rows / scratchRows) : 1); chunk++) {
      nodes.push(
        createStorageTopKNode(graph, {
          ...emitProps,
          id: keptScratch ? `${id}-emit-${chunk}` : `${id}-emit`,
          invocationCount: keptScratch ? Math.min(scratchRows, rows - chunk * scratchRows) : rows,
          body: keptScratch
            ? `{ let scratchIndex = index;
  let index = index + keptScratch[keptScratchOffset + ${cursorWord}u];
  if (index >= ${rows}u) { return; }
  ${emitProps.body}
}`
            : emitProps.body
        })
      );
      if (keptScratch && chunk + 1 < Math.ceil(rows / scratchRows)) {
        nodes.push(
          createStorageTopKNode(graph, {
            id: `${id}-advance-${chunk}`,
            operation: OPERATION,
            variant: 'advance-sort-cursor',
            bindings: [{name: 'keptScratch', view: keptScratch, type: 'u32', access: 'read_write'}],
            invocationCount: 1,
            body: `keptScratch[keptScratchOffset + ${cursorWord}u] += ${scratchRows}u;`
          })
        );
      }
    }
    return nodes;
  }
}

// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {validateRegionalizationWeightsCompleteness} from './adjacency-loss-suppression';
import {validateSpatialOutputViews} from '../spatial-clustering/spatial-output-validation';
import {
  snapshotWeightsFields,
  snapshotWeightsPort
} from '../spatial-weights/weights-construction-snapshot';
import {
  createWeightsIncompleteNodes,
  validateWeightsLossSources
} from '../spatial-weights/spatial-weights-loss';
import {
  createTransientView,
  GPUCompaction,
  GPUUint32Gather,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  createPublishNode,
  createWGSLKernelNode
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  validateCapacity,
  validateCompactOutput,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validateSinglePackedView,
  validateGPUStorageBindingByteLength,
  validateCommandNodeBudget
} from '../../analysis-core/kernels/gpu-contributor-utils';
import type {GPUCompactOutput} from '../../analysis-core/kernels/gpu-contributor-types';
import {
  type GPUSpatialWeights,
  validateGPUSpatialWeights
} from '../spatial-weights/spatial-weights';
import {IS_FINITE_FLOAT_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {validateGraphChunkBindingRanges} from '../../analysis-core/kernels/graph-chunk-dispatch';
import {createStandardizeColumnsNode} from './regionalization-inputs';
import {REGIONALIZATION_CSR_WGSL, REGIONALIZATION_NONE_WGSL} from './regionalization-wgsl';

const OPERATION = 'GPUSpatialWeightsMinimumSpanningTree';
/** Standardization, compaction and final publishing, including published primitive scans. */
const SPANNING_TREE_FIXED_NODE_ALLOWANCE = 128;
/** Each Boruvka round selects/hooks/relabels, with no more than log2(rows)+1 pointer jumps. */
const SPANNING_TREE_ROUND_NODE_ALLOWANCE = 7;

/** Largest supported attribute column count. */
export const GPU_MINIMUM_SPANNING_TREE_MAXIMUM_COLUMNS = 32;

/** Packed words of {@link GPUSpatialWeightsMinimumSpanningTreeOutput.status}. */
export const GPU_MINIMUM_SPANNING_TREE_STATUS_LAYOUT = {
  /** Number of non-finite entries of `values`; those entries are skipped (see the class). */
  invalidCount: 0,
  /** Either upstream adjacency loss cause, normalized to 0 or 1, independently of compact edges. */
  candidateOverflow: 1,
  length: 2
} as const;

/**
 * Properties for {@link GPUSpatialWeightsMinimumSpanningTree}.
 *
 * Compile-time: row count, slot capacity, `columnCount`, `standardize` and which outputs exist.
 * Per-frame: the contents of `values` and `weights` (the whole forest is rebuilt on every
 * encoding, so a changed attribute selection is a buffer write).
 */
export type GPUSpatialWeightsMinimumSpanningTreeProps = {
  /** Prefix for generated node IDs. Defaults to `'minimum-spanning-tree'`. */
  id?: string;
  /**
   * Symmetric contiguity or neighbor weights. Every listed slot is a link; the weight value is
   * ignored (cost is attribute dissimilarity). Every link must be listed in both rows.
   * `GPUContiguityWeights` supplies reciprocal links. For directed weights, take the union with
   * their transpose before constructing the tree.
   */
  weights: GPUSpatialWeights;
  /** Attribute table, row-major with `columnCount` columns: `rows * columnCount` entries. */
  values: GraphDataView<'float32'>;
  /** Number of attribute columns. Defaults to 1; at most 32. */
  columnCount?: number;
  /**
   * When true (the default), each column is standardised to mean 0 and population standard
   * deviation 1 before costs are taken (a constant column becomes all zero). Set false when
   * `values` is already standardised or when exact integer costs matter.
   */
  standardize?: boolean;
  /** Caller-owned outputs. */
  output: GPUSpatialWeightsMinimumSpanningTreeOutput;
};

/** Caller-owned outputs of {@link GPUSpatialWeightsMinimumSpanningTree}. */
export type GPUSpatialWeightsMinimumSpanningTreeOutput = {
  /**
   * Flags, one per weights slot (`weights.neighbors.length` entries). Slot `s` is
   * `1` when it is the lower-row slot of an MST edge, otherwise `0`. The slot index is the
   * stable edge ID used by every other output.
   */
  treeEdgeFlags: GraphDataView<'uint32'>;
  /**
   * Forest component label per row (`rows` entries): the smallest row index of the
   * row's tree. Isolated rows label themselves. Incomplete weights publish 0xffffffff for every row.
   */
  componentLabels: GraphDataView<'uint32'>;
  /**
   * Optional copy of the standardised attribute table (`rows * columnCount`
   * entries), so {@link GPUSkaterRegions} cuts under the same metric. Requires `standardize`.
   */
  standardizedValues?: GraphDataView<'float32'>;
  /**
   * Optional bounded compact edge list: `ids` hold the edge slots in ascending order.
   * `ids.length` should be at least `rows - 1`; smaller capacities clamp and set `overflow`.
   */
  edges?: GPUCompactOutput;
  /**
   * Optional packed words indexed by {@link GPU_MINIMUM_SPANNING_TREE_STATUS_LAYOUT}: `invalidCount`
   * counts non-finite attribute entries before standardisation; `candidateOverflow` reports either
   * upstream adjacency loss cause, even without compact edges. Attribute invalidity alone still
   * builds the tree. Adjacency loss suppresses the forest. Spare words are untouched.
   */
  status?: GraphDataView<'uint32'>;
  /** Optional `[row, neighbor]` pair per compact edge (`2 * edges.ids.length` entries). */
  edgeEndpoints?: GraphDataView<'uint32'>;
  /** Optional squared attribute dissimilarity per compact edge (`edges.ids.length` entries). */
  edgeDissimilarities?: GraphDataView<'float32'>;
};

/**
 * Minimum spanning forest over a {@link GPUSpatialWeights} graph, with edge cost the squared
 * Euclidean distance between the attribute rows of the two neighbors (the metric of SKATER).
 *
 * The solver is Borůvka: each round, every tree selects its cheapest outgoing link, the
 * selections merge trees, and labels are flattened by pointer jumping, for
 * `ceil(log2(rows))` unrolled rounds. A disconnected graph yields a spanning forest with one
 * tree per connected component.
 * The forest accepts 1–32 attribute columns and at most 32,768 estimated command nodes.
 * The attribute table, row scratch and slot scratch must each fit one storage binding
 * (33,554,432 32-bit entries at 128 MiB).
 * Either input loss flag suppresses the forest: flags are zero, component labels are 0xffffffff,
 * and compact count/requiredCount are zero. Compact candidateOverflow carries either upstream
 * cause; compact overflow remains destination-capacity loss. The complete publication port is
 * available as output; an omitted compact candidate flag is graph-owned. Clearing input loss
 * rebuilds and publishes the forest on the next submission.
 *
 * **Exactness and ties.** Costs are totally ordered by `(cost, edge ID)` where the edge ID is
 * the lower-row CSR slot, so the minimum spanning forest is unique and equals Kruskal's under
 * the same order. Selection uses integer atomic minima over the cost bits (non-negative f32 bits
 * order like the values), so results are deterministic. Other libraries break cost ties by their
 * own rules (scipy by traversal order), so on data with exact cost ties the edge sets can differ
 * while the total cost is identical.
 *
 * **Non-finite attributes.** As in `GPUAZPRegions`, `GPUWardRegions` and `GPUMaxPRegions`, the
 * standardisation statistics use the finite entries of each column only, so one NaN does not turn
 * the column (and every edge cost) into NaN; the non-finite entry itself stays non-finite in
 * `standardizedValues`. Every link that touches a row with a non-finite entry costs `+Infinity`,
 * so it is chosen last and the forest still spans the graph. `output.status` reports how many
 * entries were skipped. Callers that cut the tree (SKATER) should check it.
 *
 * Non-goals: asymmetric weights (a link listed in one row only is scanned from one end, so the
 * forest is not guaranteed minimum; build the undirected union with `GPUSpatialWeightsTranspose` first), and edge costs other than squared Euclidean distance.
 */
export class GPUSpatialWeightsMinimumSpanningTree implements GPUCommandNodeProducer {
  /** Prefix for every node ID. */
  readonly id: string;
  readonly estimatedCommandNodeCount: number;
  /** Validated properties. */
  readonly props: GPUSpatialWeightsMinimumSpanningTreeProps;
  /** Complete publication port, including a producer-owned compact candidate flag when needed. */
  readonly output: GPUSpatialWeightsMinimumSpanningTreeOutput;

  constructor(props: GPUSpatialWeightsMinimumSpanningTreeProps) {
    const captured = snapshotWeightsFields(props, [
      'id',
      'weights',
      'values',
      'columnCount',
      'standardize',
      'output'
    ]);
    const outputSnapshot = snapshotWeightsFields(captured.output, [
      'treeEdgeFlags',
      'componentLabels',
      'standardizedValues',
      'edges',
      'status',
      'edgeEndpoints',
      'edgeDissimilarities'
    ]);
    const edges = outputSnapshot.edges
      ? snapshotWeightsFields(outputSnapshot.edges, ['ids', 'status'])
      : undefined;
    props = Object.freeze({
      ...captured,
      weights: snapshotWeightsPort(captured.weights),
      output: Object.freeze({
        ...outputSnapshot,
        ...(edges
          ? {
              edges: Object.freeze({
                ...edges,
                status: snapshotWeightsFields(edges.status, [
                  'count',
                  'requiredCount',
                  'overflow',
                  'candidateOverflow'
                ])
              })
            }
          : {})
      })
    });
    const id = props.id ?? 'minimum-spanning-tree';
    this.id = id;
    this.props = props;
    const rows = validateGPUSpatialWeights(id, props.weights);
    validateRegionalizationWeightsCompleteness(id, props.weights);
    const columnCount = props.columnCount ?? 1;
    const {output} = props;
    validateCapacity(id, 'columnCount', columnCount, 1, GPU_MINIMUM_SPANNING_TREE_MAXIMUM_COLUMNS);
    validateSinglePackedView(props.values, ['float32'], `${id} values`);
    if (props.values.length !== rows * columnCount) {
      throw new Error(`${id} values length must equal rows * columnCount`);
    }
    validateSinglePackedView(output.treeEdgeFlags, ['uint32'], `${id} output.treeEdgeFlags`);
    if (output.treeEdgeFlags.length !== props.weights.neighbors.length) {
      throw new Error(`${id} output.treeEdgeFlags length must equal the weights slot capacity`);
    }
    validateSinglePackedView(output.componentLabels, ['uint32'], `${id} output.componentLabels`);
    if (output.componentLabels.length !== rows) {
      throw new Error(`${id} output.componentLabels length must equal the weights row count`);
    }
    if (output.standardizedValues) {
      validateSinglePackedView(
        output.standardizedValues,
        ['float32'],
        `${id} output.standardizedValues`
      );
      if (output.standardizedValues.length !== rows * columnCount) {
        throw new Error(`${id} output.standardizedValues length must equal rows * columnCount`);
      }
      if (props.standardize === false) {
        throw new Error(`${id} output.standardizedValues requires standardize`);
      }
    }
    if (output.status) {
      validateSinglePackedView(output.status, ['uint32'], `${id} output.status`);
      if (output.status.length < GPU_MINIMUM_SPANNING_TREE_STATUS_LAYOUT.length) {
        throw new Error(
          `${id} output.status must hold ${GPU_MINIMUM_SPANNING_TREE_STATUS_LAYOUT.length} word`
        );
      }
    }
    if (output.edges) {
      validateCompactOutput(id, output.edges);
    } else if (output.edgeEndpoints || output.edgeDissimilarities) {
      throw new Error(`${id} edge details require output.edges`);
    }
    const capacity = output.edges?.ids.length ?? 0;
    if (output.edgeEndpoints) {
      validateSinglePackedView(output.edgeEndpoints, ['uint32'], `${id} output.edgeEndpoints`);
      if (output.edgeEndpoints.length < 2 * capacity) {
        throw new Error(`${id} output.edgeEndpoints must hold two entries per edge slot`);
      }
    }
    if (output.edgeDissimilarities) {
      validateSinglePackedView(
        output.edgeDissimilarities,
        ['float32'],
        `${id} output.edgeDissimilarities`
      );
      if (output.edgeDissimilarities.length < capacity) {
        throw new Error(`${id} output.edgeDissimilarities must hold one entry per edge slot`);
      }
    }
    const rounds = Math.max(1, Math.ceil(Math.log2(rows)));
    this.estimatedCommandNodeCount =
      SPANNING_TREE_FIXED_NODE_ALLOWANCE + rounds * (SPANNING_TREE_ROUND_NODE_ALLOWANCE + rounds);
    validateCommandNodeBudget(id, this.estimatedCommandNodeCount);
    validateSpatialOutputViews(
      id,
      {
        treeEdgeFlags: output.treeEdgeFlags,
        componentLabels: output.componentLabels,
        standardizedValues: output.standardizedValues,
        status: output.status,
        edgeIds: output.edges?.ids,
        count: output.edges?.status.count,
        overflow: output.edges?.status.overflow,
        requiredCount: output.edges?.status.requiredCount,
        candidateOverflow: output.edges?.status.candidateOverflow,
        edgeEndpoints: output.edgeEndpoints,
        edgeDissimilarities: output.edgeDissimilarities
      },
      [
        [
          output.edges?.ids.length ? output.edges.ids : undefined,
          output.edges?.status.count,
          output.edges?.status.overflow,
          output.edges?.status.requiredCount,
          output.edges?.status.candidateOverflow
        ],
        ...(output.edgeEndpoints || output.edgeDissimilarities
          ? [
              {
                read: [output.edges?.ids, output.edges?.status.count],
                write: [output.edgeEndpoints, output.edgeDissimilarities]
              }
            ]
          : [])
      ]
    );
    validateGraphOutputsDisjointFromInputs(
      id,
      [
        output.treeEdgeFlags,
        output.componentLabels,
        output.standardizedValues,
        output.status,
        output.edges?.ids,
        output.edges?.status.count,
        output.edges?.status.overflow,
        output.edges?.status.requiredCount,
        output.edges?.status.candidateOverflow,
        output.edgeEndpoints,
        output.edgeDissimilarities
      ],
      [
        props.values,
        props.weights.offsets,
        props.weights.neighbors,
        ...validateWeightsLossSources(id, [props.weights])
      ]
    );
    const candidateOverflow =
      output.edges && (props.weights.status?.overflow || props.weights.status?.candidateOverflow)
        ? (output.edges.status.candidateOverflow ??
          createTransientView(
            output.treeEdgeFlags.buffer.graph as GPUCommandGraph,
            `${id}-${output.treeEdgeFlags.buffer.id}-${output.treeEdgeFlags.byteOffset}-candidate-loss`,
            'uint32',
            1
          ))
        : output.edges?.status.candidateOverflow;
    this.output = Object.freeze({
      ...output,
      ...(output.edges
        ? {
            edges: Object.freeze({
              ...output.edges,
              status: Object.freeze({...output.edges.status, candidateOverflow})
            })
          }
        : {})
    });
    this.props = Object.freeze({...props, output: this.output});
  }

  /** Returns the standardisation, Borůvka round, label and edge-list nodes. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const {weights, output} = props;
    const {treeEdgeFlags, componentLabels} = output;
    validateGraphViewsBelongToGraph(id, graph, [
      props.values,
      treeEdgeFlags,
      componentLabels,
      output.standardizedValues,
      output.status,
      output.edges?.ids,
      output.edges?.status.count,
      output.edges?.status.overflow,
      output.edges?.status.requiredCount,
      output.edges?.status.candidateOverflow,
      output.edgeEndpoints,
      output.edgeDissimilarities,
      weights.offsets,
      weights.neighbors,
      ...validateWeightsLossSources(id, [weights])
    ]);
    const rows = weights.offsets.length - 1;
    const capacity = weights.neighbors.length;
    const columnCount = props.columnCount ?? 1;
    validateGraphChunkBindingRanges(graph, id, {
      values: props.values,
      offsets: weights.offsets,
      neighbors: weights.neighbors,
      treeEdgeFlags: output.treeEdgeFlags,
      componentLabels: output.componentLabels,
      standardizedValues: output.standardizedValues,
      status: output.status,
      edgeIds: output.edges?.ids,
      count: output.edges?.status.count,
      overflow: output.edges?.status.overflow,
      requiredCount: output.edges?.status.requiredCount,
      edgeEndpoints: output.edgeEndpoints,
      edgeDissimilarities: output.edgeDissimilarities,
      inputOverflow: weights.status?.overflow,
      inputCandidateOverflow: weights.status?.candidateOverflow,
      candidateOverflow: output.edges?.status.candidateOverflow
    });
    for (const [name, byteLength] of [
      ['standardized table', rows * columnCount * 4],
      ['row scratch', Math.max(1, rows) * 4],
      ['slot costs and compaction', Math.max(1, capacity) * 4]
    ] as const)
      validateGPUStorageBindingByteLength(
        graph.device,
        id,
        name,
        byteLength,
        'reduce row or neighbor slot counts'
      );
    const standardize = props.standardize !== false;
    const csrBindings = [
      {name: 'offsets', view: weights.offsets, type: 'u32' as const, access: 'read' as const},
      {name: 'neighbors', view: weights.neighbors, type: 'u32' as const, access: 'read' as const}
    ];
    const rowDeclarations = `const ROWS: u32 = ${rows}u;
const COLUMNS: u32 = ${columnCount}u;
${IS_FINITE_FLOAT_WGSL}
${REGIONALIZATION_NONE_WGSL}
${REGIONALIZATION_CSR_WGSL}`;
    const loss = createWeightsIncompleteNodes(graph, id, [weights]);
    const nodes: GPUCommandNode<Parameters>[] = [...loss.nodes];

    // Attribute table the costs read.
    let table = props.values;
    if (standardize) {
      table =
        output.standardizedValues ??
        createTransientView(graph, `${id}-table`, 'float32', rows * columnCount);
      // The one standardizer of the regionalization family (AZP, Ward and max-p use the same
      // node): one workgroup per column, a constant column becomes exactly zero.
      nodes.push(
        createStandardizeColumnsNode<Parameters>(graph, {
          id: `${id}-standardize`,
          operation: OPERATION,
          values: props.values,
          scaled: table,
          rowCount: rows,
          columnCount
        })
      );
    }

    if (output.status) {
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-clear-invalid`,
          operation: OPERATION,
          variant: 'clear-invalid',
          bindings: [
            {name: 'status', view: output.status, type: 'u32', access: 'read_write'},
            ...(loss.view
              ? [{name: 'loss', view: loss.view, type: 'u32' as const, access: 'read' as const}]
              : [])
          ],
          invocationCount: 1,
          body: `status[statusOffset] = 0u;
  status[statusOffset + ${GPU_MINIMUM_SPANNING_TREE_STATUS_LAYOUT.candidateOverflow}u] = ${loss.view ? 'loss[lossOffset]' : '0u'};`
        }),
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-count-invalid`,
          operation: OPERATION,
          variant: 'count-invalid',
          bindings: [
            {name: 'values', view: props.values, type: 'f32', access: 'read'},
            {name: 'status', view: output.status, type: 'atomic<u32>', access: 'read_write'}
          ],
          invocationCount: rows,
          declarations: `const COLUMNS: u32 = ${columnCount}u;
${IS_FINITE_FLOAT_WGSL}`,
          body: `var invalid = 0u;
  for (var column = 0u; column < COLUMNS; column++) {
    invalid += select(1u, 0u, isFiniteFloat(values[valuesOffset + index * COLUMNS + column]));
  }
  if (invalid > 0u) { atomicAdd(&status[statusOffset], invalid); }`
        })
      );
    }

    const slotCost = createTransientView(graph, `${id}-slot-cost`, 'float32', capacity);
    const slotEdge = createTransientView(graph, `${id}-slot-edge`, 'uint32', capacity);
    const component = createTransientView(graph, `${id}-component`, 'uint32', rows);
    const bestCost = createTransientView(graph, `${id}-best-cost`, 'uint32', rows);
    const bestEdge = createTransientView(graph, `${id}-best-edge`, 'uint32', rows);
    const parentA = createTransientView(graph, `${id}-parent-a`, 'uint32', rows);
    const parentB = createTransientView(graph, `${id}-parent-b`, 'uint32', rows);
    const minimumLabel = createTransientView(graph, `${id}-minimum-label`, 'uint32', rows);

    nodes.push(
      // Zero the flag capacity so unused tail slots are well defined, and seed one tree per row.
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-clear-flags`,
        operation: OPERATION,
        variant: 'clear',
        bindings: [{name: 'flags', view: treeEdgeFlags, type: 'u32', access: 'read_write'}],
        invocationCount: capacity,
        body: 'flags[flagsOffset + index] = 0u;'
      }),
      // Seeds one tree per row and clears round 0's selections; later rounds are cleared by the
      // previous round's relabel kernel, so no round needs a separate reset pass.
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-seed-components`,
        operation: OPERATION,
        variant: 'seed',
        bindings: [
          {name: 'component', view: component, type: 'u32', access: 'read_write'},
          {name: 'bestCost', view: bestCost, type: 'u32', access: 'read_write'},
          {name: 'bestEdge', view: bestEdge, type: 'u32', access: 'read_write'}
        ],
        invocationCount: rows,
        body: `component[componentOffset + index] = index;
  bestCost[bestCostOffset + index] = 0xffffffffu;
  bestEdge[bestEdgeOffset + index] = 0xffffffffu;`
      }),
      // Per-slot cost (squared distance of the standardised rows) and canonical edge ID.
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-slot-costs`,
        operation: OPERATION,
        variant: 'costs',
        bindings: [
          ...csrBindings,
          {name: 'table', view: table, type: 'f32', access: 'read'},
          {name: 'slotCost', view: slotCost, type: 'f32', access: 'read_write'},
          {name: 'slotEdge', view: slotEdge, type: 'u32', access: 'read_write'}
        ],
        invocationCount: rows,
        declarations: rowDeclarations,
        body: `for (var slot = offsets[offsetsOffset + index]; slot < offsets[offsetsOffset + index + 1u]; slot++) {
    let neighbor = neighbors[neighborsOffset + slot];
    let low = min(index, neighbor);
    let high = max(index, neighbor);
    var cost = 0.0;
    var finite = true;
    for (var column = 0u; column < COLUMNS; column++) {
      let lowValue = table[tableOffset + low * COLUMNS + column];
      let highValue = table[tableOffset + high * COLUMNS + column];
      finite = finite && isFiniteFloat(lowValue) && isFiniteFloat(highValue);
      let delta = lowValue - highValue;
      cost += delta * delta;
    }
    // A link touching a non-finite entry costs +Infinity: chosen last, never a NaN bit pattern. The
    // runtime operand keeps the infinity out of constant evaluation.
    slotCost[slotCostOffset + slot] = select(bitcast<f32>(0x7f800000u | (index & 0u)), cost, finite);
    var edge = slot;
    if (neighbor < index) {
      let twin = findSlot(neighbor, index);
      if (twin != NONE) {
        edge = twin;
      }
    }
    slotEdge[slotEdgeOffset + slot] = edge;
  }`
      })
    );

    const roundCount = Math.max(1, Math.ceil(Math.log2(rows)));
    for (let round = 0; round < roundCount; round++) {
      const roundId = `${id}-round-${round}`;
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${roundId}-cheapest-cost`,
          operation: OPERATION,
          variant: 'cheapest-cost',
          bindings: [
            ...csrBindings,
            {name: 'component', view: component, type: 'u32', access: 'read'},
            {name: 'slotCost', view: slotCost, type: 'f32', access: 'read'},
            {name: 'bestCost', view: bestCost, type: 'atomic<u32>', access: 'read_write'}
          ],
          invocationCount: rows,
          declarations: `const ROWS: u32 = ${rows}u;`,
          body: `let own = component[componentOffset + index];
  for (var slot = offsets[offsetsOffset + index]; slot < offsets[offsetsOffset + index + 1u]; slot++) {
    if (component[componentOffset + neighbors[neighborsOffset + slot]] != own) {
      atomicMin(&bestCost[bestCostOffset + own], bitcast<u32>(slotCost[slotCostOffset + slot]));
    }
  }`
        }),
        createWGSLKernelNode<Parameters>(graph, {
          id: `${roundId}-cheapest-edge`,
          operation: OPERATION,
          variant: 'cheapest-edge',
          bindings: [
            ...csrBindings,
            {name: 'component', view: component, type: 'u32', access: 'read'},
            {name: 'slotCost', view: slotCost, type: 'f32', access: 'read'},
            {name: 'slotEdge', view: slotEdge, type: 'u32', access: 'read'},
            {name: 'bestCost', view: bestCost, type: 'u32', access: 'read'},
            {name: 'bestEdge', view: bestEdge, type: 'atomic<u32>', access: 'read_write'}
          ],
          invocationCount: rows,
          declarations: `const ROWS: u32 = ${rows}u;`,
          body: `let own = component[componentOffset + index];
  for (var slot = offsets[offsetsOffset + index]; slot < offsets[offsetsOffset + index + 1u]; slot++) {
    if (component[componentOffset + neighbors[neighborsOffset + slot]] != own &&
        bitcast<u32>(slotCost[slotCostOffset + slot]) == bestCost[bestCostOffset + own]) {
      atomicMin(&bestEdge[bestEdgeOffset + own], slotEdge[slotEdgeOffset + slot]);
    }
  }`
        }),
        // Each tree points at the tree across its cheapest link and flags that edge. Two trees
        // that chose the same edge would point at each other; with the total (cost, edge ID) order
        // that is exactly "the far tree's cheapest edge is this edge", and the lower label becomes
        // the root, so no separate pair-breaking pass is needed.
        createWGSLKernelNode<Parameters>(graph, {
          id: `${roundId}-hook`,
          operation: OPERATION,
          variant: 'hook',
          bindings: [
            ...csrBindings,
            {name: 'component', view: component, type: 'u32', access: 'read'},
            {name: 'bestEdge', view: bestEdge, type: 'u32', access: 'read'},
            {name: 'parentOut', view: parentB, type: 'u32', access: 'read_write'},
            {name: 'flags', view: treeEdgeFlags, type: 'atomic<u32>', access: 'read_write'}
          ],
          invocationCount: rows,
          declarations: rowDeclarations,
          body: `var parent = index;
  let edge = bestEdge[bestEdgeOffset + index];
  if (component[componentOffset + index] == index && edge != NONE) {
    let first = component[componentOffset + getRowOfSlot(edge)];
    let second = component[componentOffset + neighbors[neighborsOffset + edge]];
    parent = select(first, second, first == index);
    let isMutual = bestEdge[bestEdgeOffset + parent] == edge;
    parent = select(parent, index, isMutual && index < parent);
    atomicStore(&flags[flagsOffset + edge], 1u);
  }
  parentOut[parentOutOffset + index] = parent;`
        })
      );
      // Active trees at most halve each round, so chains are at most rows / 2^round long.
      const jumpCount =
        Math.max(1, Math.ceil(Math.log2(Math.max(2, Math.ceil(rows / 2 ** round))))) + 1;
      // The last pointer-jumping step is folded into the relabel kernel (a double indirection), so
      // `jumpCount - 1` separate passes remain. The relabel also clears the next round's
      // selections, which no kernel reads after the hook.
      let source = parentB;
      let destination = parentA;
      for (let jump = 0; jump < jumpCount - 1; jump++) {
        nodes.push(
          createWGSLKernelNode<Parameters>(graph, {
            id: `${roundId}-jump-${jump}`,
            operation: OPERATION,
            variant: 'jump',
            bindings: [
              {name: 'parentIn', view: source, type: 'u32', access: 'read'},
              {name: 'parentOut', view: destination, type: 'u32', access: 'read_write'}
            ],
            invocationCount: rows,
            body: `parentOut[parentOutOffset + index] =
    parentIn[parentInOffset + parentIn[parentInOffset + index]];`
          })
        );
        [source, destination] = [destination, source];
      }
      const isLastRound = round === roundCount - 1;
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${roundId}-relabel`,
          operation: OPERATION,
          variant: 'relabel',
          bindings: [
            {name: 'root', view: source, type: 'u32', access: 'read'},
            {name: 'component', view: component, type: 'u32', access: 'read_write'},
            ...(isLastRound
              ? [
                  {
                    name: 'minimumLabel',
                    view: minimumLabel,
                    type: 'u32' as const,
                    access: 'read_write' as const
                  }
                ]
              : [
                  {
                    name: 'bestCost',
                    view: bestCost,
                    type: 'u32' as const,
                    access: 'read_write' as const
                  },
                  {
                    name: 'bestEdge',
                    view: bestEdge,
                    type: 'u32' as const,
                    access: 'read_write' as const
                  }
                ])
          ],
          invocationCount: rows,
          body: `let label = component[componentOffset + index];
  component[componentOffset + index] = root[rootOffset + root[rootOffset + label]];
  ${
    isLastRound
      ? 'minimumLabel[minimumLabelOffset + index] = 0xffffffffu;'
      : `bestCost[bestCostOffset + index] = 0xffffffffu;
  bestEdge[bestEdgeOffset + index] = 0xffffffffu;`
  }`
        })
      );
    }

    // Canonical labels: the smallest row index of each tree.
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-label-minimum`,
        operation: OPERATION,
        variant: 'label-minimum',
        bindings: [
          {name: 'component', view: component, type: 'u32', access: 'read'},
          {name: 'minimumLabel', view: minimumLabel, type: 'atomic<u32>', access: 'read_write'}
        ],
        invocationCount: rows,
        body: 'atomicMin(&minimumLabel[minimumLabelOffset + component[componentOffset + index]], index);'
      }),
      // Every `component` entry is a tree root, a row index below `rows`, so none is out of range.
      ...new GPUUint32Gather({
        id: `${id}-label-write`,
        source: minimumLabel,
        indices: component,
        output: componentLabels
      }).getCommandNodes(graph)
    );

    if (loss.view) {
      for (const [name, view, value] of [
        ['tree-flags', treeEdgeFlags, '0u'],
        ['component-labels', componentLabels, '0xffffffffu']
      ] as const)
        nodes.push(
          createWGSLKernelNode(graph, {
            id: `${id}-incomplete-${name}`,
            operation: OPERATION,
            invocationCount: view.length,
            bindings: [
              {name: 'loss', view: loss.view, type: 'u32', access: 'read'},
              {name: 'result', view, type: 'u32', access: 'read_write'}
            ],
            body: `if (loss[lossOffset] != 0u) { result[resultOffset + index] = ${value}; }`
          })
        );
    }

    if (output.edges) {
      const {edges} = output;
      const slotIds = createTransientView(graph, `${id}-slot-ids`, 'uint32', capacity);
      const compactSlots = createTransientView(graph, `${id}-compact-slots`, 'uint32', capacity);
      const requiredCount = createTransientView(graph, `${id}-edge-total`, 'uint32', 1);
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-slot-ids`,
          operation: OPERATION,
          variant: 'slot-ids',
          bindings: [{name: 'slotIds', view: slotIds, type: 'u32', access: 'read_write'}],
          invocationCount: capacity,
          body: 'slotIds[slotIdsOffset + index] = index;'
        }),
        ...new GPUCompaction({
          id: `${id}-compaction`,
          input: slotIds,
          flags: treeEdgeFlags,
          output: compactSlots,
          count: requiredCount
        }).getCommandNodes(graph),
        createPublishNode<Parameters>(graph, {
          id: `${id}-publish`,
          operation: OPERATION,
          requiredCount,
          compactIds: compactSlots,
          candidateOverflowSources: loss.view ? [loss.view] : [],
          output: edges
        })
      );
      if (output.edgeEndpoints || output.edgeDissimilarities) {
        const bindings = [
          ...csrBindings,
          {name: 'edgeIds', view: edges.ids, type: 'u32' as const, access: 'read' as const},
          {
            name: 'edgeCount',
            view: edges.status.count,
            type: 'u32' as const,
            access: 'read' as const
          },
          {name: 'slotCost', view: slotCost, type: 'f32' as const, access: 'read' as const},
          ...(output.edgeEndpoints
            ? [
                {
                  name: 'endpoints',
                  view: output.edgeEndpoints,
                  type: 'u32' as const,
                  access: 'read_write' as const
                }
              ]
            : []),
          ...(output.edgeDissimilarities
            ? [
                {
                  name: 'dissimilarities',
                  view: output.edgeDissimilarities,
                  type: 'f32' as const,
                  access: 'read_write' as const
                }
              ]
            : [])
        ];
        nodes.push(
          createWGSLKernelNode<Parameters>(graph, {
            id: `${id}-edge-details`,
            operation: OPERATION,
            variant: 'edge-details',
            bindings,
            invocationCount: edges.ids.length,
            declarations: rowDeclarations,
            body: `if (index >= edgeCount[edgeCountOffset]) {
    return;
  }
  let slot = edgeIds[edgeIdsOffset + index];
  ${
    output.edgeEndpoints
      ? `endpoints[endpointsOffset + 2u * index] = getRowOfSlot(slot);
  endpoints[endpointsOffset + 2u * index + 1u] = neighbors[neighborsOffset + slot];`
      : ''
  }
  ${output.edgeDissimilarities ? 'dissimilarities[dissimilaritiesOffset + index] = slotCost[slotCostOffset + slot];' : ''}`
          })
        );
      }
    }
    return nodes;
  }
}

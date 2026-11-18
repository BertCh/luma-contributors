// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {validateSpatialOutputViews} from '../spatial-clustering/spatial-output-validation';
import {
  snapshotWeightsFields,
  snapshotWeightsPort
} from '../spatial-weights/weights-construction-snapshot';
import {validateGraphChunkBindingRanges} from '../../analysis-core/kernels/graph-chunk-dispatch';
import {getGraphOutputViews} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  createTransientView,
  GPUSort,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {createMaskedWeakComponentNodes} from '../../analysis-core/kernels/masked-weak-components';
import {createFillNode, createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  validateCapacity,
  validateCommandNodeBudget,
  validateGPUStorageBindingByteLength,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validateSinglePackedView,
  validateViewLength,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import type {GPUIterationStatusPort} from '../../analysis-core/contracts/status';
import {
  type GPUSpatialWeights,
  validateGPUSpatialWeights
} from '../spatial-weights/spatial-weights';
import {
  GPU_MINIMUM_SPANNING_TREE_MAXIMUM_COLUMNS,
  GPU_MINIMUM_SPANNING_TREE_STATUS,
  GPUSpatialWeightsMinimumSpanningTree
} from '../spatial-regionalization/gpu-spatial-weights-minimum-spanning-tree';

const OPERATION = 'GPURedcapSingleLinkage';

/** Words of {@link GPURedcapSingleLinkageProps.parameters}. */
export const GPU_REDCAP_SINGLE_LINKAGE_PARAMETER_LENGTH = 1;

/**
 * Default `componentIterations`: label propagation with pointer shortcuts, the same bound as the
 * published connected components primitive. `output.status.converged` reports whether it was enough.
 */
export const GPU_REDCAP_SINGLE_LINKAGE_DEFAULT_COMPONENT_ITERATIONS = 32;

/**
 * Largest `componentIterations`. Each iteration adds two command nodes, so this keeps the node
 * count of the label stage well inside the shared command node budget.
 */
export const GPU_REDCAP_SINGLE_LINKAGE_MAXIMUM_COMPONENT_ITERATIONS = 256;

/** Command nodes allowed for the minimum spanning tree and the stable sort (an upper bound). */
const SPANNING_TREE_AND_SORT_NODE_ALLOWANCE = 1024;

/** Per-frame settings packed by {@link getGPURedcapSingleLinkageParameterValues}. */
export type GPURedcapSingleLinkageSettings = {
  /**
   * Target region count, at least 1. A value at or below the number of trees of the spanning
   * forest (the connected components of `weights`) cuts nothing; a value above the row count
   * cuts every edge.
   */
  regionCount: number;
};

/** Packs {@link GPURedcapSingleLinkageSettings} into the `uint32` words of `parameters`. */
export function getGPURedcapSingleLinkageParameterValues(
  settings: GPURedcapSingleLinkageSettings
): Uint32Array {
  validateCapacity(OPERATION, 'regionCount', settings.regionCount);
  return new Uint32Array([settings.regionCount]);
}

/**
 * Properties for {@link GPURedcapSingleLinkage}.
 *
 * Compile-time: row count, slot capacity, `columnCount`, `standardize`, `componentIterations` and
 * which outputs exist. Per-frame: the contents of `values`, `weights` and `parameters`, so a
 * region-count slider is a buffer write.
 */
export type GPURedcapSingleLinkageProps = {
  /** Prefix for generated node IDs. Defaults to `'redcap-single-linkage'`. */
  id?: string;
  /** Symmetric contiguity weights, as for {@link GPUSpatialWeightsMinimumSpanningTree}. */
  weights: GPUSpatialWeights;
  /** Attribute table, row-major with `columnCount` columns (`rows * columnCount` entries). */
  values: GraphDataView<'float32'>;
  /** Number of attribute columns. Defaults to 1; at most 32. */
  columnCount?: number;
  /** Standardise each column to mean 0 and population SD 1 first. Defaults to true. */
  standardize?: boolean;
  /** Per-frame `uint32` words, see {@link getGPURedcapSingleLinkageParameterValues}. */
  parameters: GraphDataView<'uint32'>;
  /**
   * Iterations of the label propagation that names the regions. Defaults to
   * {@link GPU_REDCAP_SINGLE_LINKAGE_DEFAULT_COMPONENT_ITERATIONS}. A region that is a long
   * thin path may need more; check `output.status.converged`.
   */
  componentIterations?: number;
  /** Caller-owned outputs. */
  output: GPURedcapSingleLinkageOutput;
};

/** Caller-owned outputs of {@link GPURedcapSingleLinkage}. */
export type GPURedcapSingleLinkageOutput = {
  /** Region label per row: smallest row index, or 0xffffffff throughout after supplied weights loss. */
  labels: GraphDataView<'uint32'>;
  /** Optional one-row region count after the cuts; 0 after supplied weights loss. */
  regionCount?: GraphDataView<'uint32'>;
  /**
   * Optional status scalars (one uint32 row each): `converged` is 1 when the label propagation
   * finished within `componentIterations` and the supplied weights are complete.
   * `iterationLimitReached` is 1 when the component budget was insufficient; both flags are 0
   * after supplied weights overflow/candidateOverflow suppresses the partition.
   * `iterationCount` is the executed label-propagation rounds (always `componentIterations`;
   * every round executes even after labels stabilize). It excludes the spanning-tree rounds.
   * `invalidCount` counts non-finite attribute entries as `GPUSpatialWeightsMinimumSpanningTree`.
   */
  status?: GPUIterationStatusPort & {invalidCount?: GraphDataView<'uint32'>};
};

/**
 * Contiguity-constrained single-linkage clustering: the single-linkage dendrogram of the
 * contiguity graph cut at `k` regions, the region count chosen per frame. This is the first step
 * of first-order single-linkage REDCAP (Guo 2008), the tree, with a different second step, so it
 * is not the REDCAP partition (see "What this is not").
 *
 * The minimum spanning tree of the contiguity graph under attribute dissimilarity
 * ({@link GPUSpatialWeightsMinimumSpanningTree}, squared Euclidean distance of the standardised
 * attributes) holds every single-linkage merge of the constrained agglomeration: cutting its
 * `k - c` most dissimilar edges (`c` trees in the forest) leaves `k` contiguous regions, the
 * constrained single-linkage dendrogram cut at `k`. The tree edges are sorted by `(cost, edge
 * slot)` with a stable radix sort, the heaviest are dropped from a masked copy of the weights, and
 * the regions are the weak components of what remains. Every step is parallel; there is no
 * per-edge loop. Region counts are a prefix family: the regions for `k` refine those for `k - 1`.
 *
 * **What this is not.** REDCAP as Guo (2008) describes it has two steps: build a spatially
 * contiguous tree by constrained agglomerative clustering, then partition that tree top-down,
 * each time removing the edge whose removal reduces the within-region sum of squared deviations
 * (SSD) the most. For first-order single linkage the tree is the minimum spanning tree used here,
 * and the SSD partition of it is what {@link GPUSkaterRegions} computes (GeoDa documents
 * first-order single-linkage REDCAP as identical to SKATER). This class keeps the tree and cuts
 * its `k - c` heaviest edges instead, which looks only at the dissimilarity across one edge and
 * not at the regions the cut leaves. The two differ on ordinary data. On a path of six rows with
 * values `[0, 1, 2.1, 3.3, 4.6, 6.0]` and `k = 2` the heaviest edge is the last one, so this class
 * returns `{0..4}, {5}` (within-region SSD 13.26); the SSD partition is `{0, 1, 2}, {3, 4, 5}`
 * (5.85). Single linkage also chains: it tends to peel off single outlying rows. For REDCAP
 * regions use `GPUSpatialWeightsMinimumSpanningTree` followed by `GPUSkaterRegions`. The class
 * name is kept for compatibility. (The description of Guo's second step is from the literature as
 * recalled in the 2026-11-10 math review, not re-read against the paper.)
 *
 * **Non-goals.** Minimum region size or bound (the GeoDa and spopt style floor) is not supported,
 * and neither are the full-order, average and complete linkage REDCAP variants, which need the
 * tree re-evaluated per region. Weights must be symmetric.
 *
 * **Reference.** There is no pinned REDCAP package. The reference is the definition of
 * constrained single linkage: the same partition as `sklearn.cluster.AgglomerativeClustering(
 * linkage='single', connectivity=W)`, evaluated as a minimum spanning tree plus a cut by SciPy
 * in binary64. Edge costs that tie make several partitions valid; compare on distinct costs.
 *
 * **float32 limits.** Costs are float32 squared distances, so two edges whose true costs differ
 * by less than about one part in `1e7` of the larger can swap order.
 */
export class GPURedcapSingleLinkage implements GPUCommandNodeProducer {
  /** Prefix for every node ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPURedcapSingleLinkageProps;
  /** Upper bound of the command nodes returned by `getCommandNodes`. */
  readonly estimatedCommandNodeCount: number;

  constructor(props: GPURedcapSingleLinkageProps) {
    const captured = snapshotWeightsFields(props, [
      'id',
      'weights',
      'values',
      'columnCount',
      'standardize',
      'parameters',
      'componentIterations',
      'output'
    ]);
    const output = snapshotWeightsFields(captured.output, ['labels', 'regionCount', 'status']);
    props = Object.freeze({
      ...captured,
      weights: snapshotWeightsPort(captured.weights),
      output: Object.freeze({
        ...output,
        status: output.status
          ? snapshotWeightsFields(output.status, [
              'converged',
              'iterationLimitReached',
              'iterationCount',
              'invalidCount'
            ])
          : undefined
      })
    });
    const id = props.id ?? 'redcap-single-linkage';
    this.id = id;
    this.props = props;
    const rows = validateGPUSpatialWeights(id, props.weights);
    const columnCount = props.columnCount ?? 1;
    validateCapacity(id, 'columnCount', columnCount, 1, GPU_MINIMUM_SPANNING_TREE_MAXIMUM_COLUMNS);
    validateSinglePackedView(props.values, ['float32'], `${id} values`);
    validateRequirement(
      id,
      props.values.length === rows * columnCount,
      'values length must equal rows * columnCount'
    );
    validateSinglePackedView(props.parameters, ['uint32'], `${id} parameters`);
    validateRequirement(
      id,
      props.parameters.length >= GPU_REDCAP_SINGLE_LINKAGE_PARAMETER_LENGTH,
      `parameters must hold ${GPU_REDCAP_SINGLE_LINKAGE_PARAMETER_LENGTH} word`
    );
    validateCapacity(
      id,
      'componentIterations',
      props.componentIterations ?? GPU_REDCAP_SINGLE_LINKAGE_DEFAULT_COMPONENT_ITERATIONS,
      1,
      GPU_REDCAP_SINGLE_LINKAGE_MAXIMUM_COMPONENT_ITERATIONS
    );
    validateSinglePackedView(props.output.labels, ['uint32'], `${id} output.labels`);
    validateRequirement(
      id,
      props.output.labels.length === rows,
      'output.labels length must equal the weights row count'
    );
    if (props.output.status?.iterationCount) {
      validateSinglePackedView(
        props.output.status.iterationCount,
        ['uint32'],
        `${id} output.status.iterationCount`
      );
      validateViewLength(id, 'output.status.iterationCount', props.output.status.iterationCount, 1);
    }
    if (props.output.status?.invalidCount) {
      validateSinglePackedView(
        props.output.status.invalidCount,
        ['uint32'],
        `${id} output.status.invalidCount`
      );
      validateCapacity(
        id,
        'output.status.invalidCount length',
        props.output.status.invalidCount.length,
        1
      );
    }
    this.estimatedCommandNodeCount =
      SPANNING_TREE_AND_SORT_NODE_ALLOWANCE +
      6 +
      2 * (props.componentIterations ?? GPU_REDCAP_SINGLE_LINKAGE_DEFAULT_COMPONENT_ITERATIONS) +
      1 +
      (props.output.status?.iterationCount ? 1 : 0) +
      (props.output.status?.invalidCount ? 1 : 0) +
      (getRedcapWeightLossFlags(props.weights).length
        ? 1 +
          Number(Boolean(props.output.regionCount)) +
          Number(Boolean(props.output.status?.converged)) +
          Number(Boolean(props.output.status?.iterationLimitReached))
        : 0);
    validateCommandNodeBudget(id, this.estimatedCommandNodeCount);
    validateSpatialOutputViews(
      this.id,
      Object.fromEntries(
        getGraphOutputViews({
          ...props.output,
          status: {
            converged: props.output.status?.converged,
            iterationCount: props.output.status?.iterationCount,
            iterationLimitReached: props.output.status?.iterationLimitReached,
            invalidCount: props.output.status?.invalidCount
          }
        }).map((view, index) => [`output[${index}]`, view])
      ),
      [
        {
          read: [props.output.status?.converged],
          write: [props.output.status?.iterationLimitReached]
        }
      ]
    );
    validateGraphOutputsDisjointFromInputs(
      id,
      [props.output.labels, props.output.regionCount, ...Object.values(props.output.status ?? {})],
      [
        props.values,
        props.parameters,
        props.weights.offsets,
        props.weights.neighbors,
        props.weights.weights,
        ...getRedcapWeightLossFlags(props.weights)
      ]
    );
  }

  /** Returns the spanning tree, edge sort, masked-weights and component nodes. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const {weights, output} = props;
    const lossFlags = getRedcapWeightLossFlags(weights);
    validateGraphViewsBelongToGraph(id, graph, [
      props.values,
      props.parameters,
      weights.offsets,
      weights.neighbors,
      ...getRedcapWeightLossFlags(weights),
      output.labels,
      output.regionCount,
      ...Object.values(output.status ?? {})
    ]);
    const rows = weights.offsets.length - 1;
    const slots = weights.neighbors.length;
    const edgeCapacity = Math.max(rows - 1, 1);
    validateGraphChunkBindingRanges(graph, id, {
      values: props.values,
      parameters: props.parameters,
      offsets: weights.offsets,
      neighbors: weights.neighbors,
      ...Object.fromEntries(
        getRedcapWeightLossFlags(weights).map((view, index) => [`loss${index}`, view])
      ),
      ...Object.fromEntries(
        getGraphOutputViews({
          ...output,
          status: {
            converged: output.status?.converged,
            iterationCount: output.status?.iterationCount,
            iterationLimitReached: output.status?.iterationLimitReached,
            invalidCount: output.status?.invalidCount
          }
        }).map((view, index) => [`output[${index}]`, view])
      )
    });
    for (const [name, byteLength] of [
      ['standardized table', rows * (props.columnCount ?? 1) * 4],
      ['row scratch', rows * 4],
      ['neighbor scratch', Math.max(1, slots) * 4],
      // GPUSort's radix histogram is smaller than the edge plane; small sorts stay local.
      ['edge scratch', edgeCapacity * 4]
    ] as const)
      validateGPUStorageBindingByteLength(
        graph.device,
        id,
        name,
        byteLength,
        'reduce row or neighbor slot counts'
      );
    const iterations =
      props.componentIterations ?? GPU_REDCAP_SINGLE_LINKAGE_DEFAULT_COMPONENT_ITERATIONS;
    const treeEdgeFlags = createTransientView(graph, `${id}-tree-edge-flags`, 'uint32', slots);
    const treeLabels = createTransientView(graph, `${id}-tree-labels`, 'uint32', rows);
    const treeStatus = output.status?.invalidCount
      ? createTransientView(
          graph,
          `${id}-tree-status`,
          'uint32',
          GPU_MINIMUM_SPANNING_TREE_STATUS.length
        )
      : undefined;
    const edgeIds = createTransientView(graph, `${id}-edge-ids`, 'uint32', edgeCapacity);
    const edgeCount = createTransientView(graph, `${id}-edge-count`, 'uint32', 1);
    const edgeOverflow = createTransientView(graph, `${id}-edge-overflow`, 'uint32', 1);
    const edgeCosts = createTransientView(graph, `${id}-edge-costs`, 'float32', edgeCapacity);
    const sortKeys = createTransientView(graph, `${id}-sort-keys`, 'uint32', edgeCapacity);
    const sortSlots = createTransientView(graph, `${id}-sort-slots`, 'uint32', edgeCapacity);
    const sortedKeys = createTransientView(graph, `${id}-sorted-keys`, 'uint32', edgeCapacity);
    const sortedSlots = createTransientView(graph, `${id}-sorted-slots`, 'uint32', edgeCapacity);
    const keptNeighbors = createTransientView(graph, `${id}-kept-neighbors`, 'uint32', slots);
    const converged =
      output.status?.converged ??
      (output.status?.iterationLimitReached
        ? createTransientView(graph, `${id}-converged`, 'uint32', 1)
        : undefined);
    const declarations = `const ROWS: u32 = ${rows}u;
const EDGE_CAPACITY: u32 = ${edgeCapacity}u;`;
    // The target region count minus the trees of the forest, clamped to the tree edges.
    const cutCountWGSL = `fn getCutCount(edgeTotal: u32, regionCount: u32) -> u32 {
  let trees = ROWS - edgeTotal;
  return min(select(0u, regionCount - trees, regionCount > trees), edgeTotal);
}`;
    const nodes: GPUCommandNode<Parameters>[] = [
      ...new GPUSpatialWeightsMinimumSpanningTree({
        id: `${id}-tree`,
        weights,
        values: props.values,
        columnCount: props.columnCount,
        standardize: props.standardize,
        output: {
          treeEdgeFlags,
          componentLabels: treeLabels,
          status: treeStatus,
          edges: {ids: edgeIds, status: {count: edgeCount, overflow: edgeOverflow}},
          edgeDissimilarities: edgeCosts
        }
      }).getCommandNodes(graph),
      // Non-negative float32 bit patterns order like the values; padding sorts last.
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-sort-keys`,
        operation: OPERATION,
        variant: 'sort-keys',
        bindings: [
          {name: 'edgeIds', view: edgeIds, type: 'u32', access: 'read'},
          {name: 'edgeCount', view: edgeCount, type: 'u32', access: 'read'},
          {name: 'edgeCosts', view: edgeCosts, type: 'f32', access: 'read'},
          {name: 'sortKeys', view: sortKeys, type: 'u32', access: 'read_write'},
          {name: 'sortSlots', view: sortSlots, type: 'u32', access: 'read_write'}
        ],
        invocationCount: edgeCapacity,
        declarations,
        body: `let isEdge = index < edgeCount[edgeCountOffset];
  sortKeys[sortKeysOffset + index] = select(0xffffffffu, bitcast<u32>(edgeCosts[edgeCostsOffset + index]), isEdge);
  sortSlots[sortSlotsOffset + index] = edgeIds[edgeIdsOffset + index];`
      }),
      ...new GPUSort({
        id: `${id}-sort`,
        keys: sortKeys,
        values: sortSlots,
        outputKeys: sortedKeys,
        outputValues: sortedSlots,
        keyBits: 32
      }).getCommandNodes(graph),
      createFillNode<Parameters>(graph, {
        id: `${id}-clear-kept`,
        operation: OPERATION,
        view: keptNeighbors,
        type: 'u32',
        value: '0xffffffffu'
      }),
      // The first `edgeCount - cutCount` sorted edges survive.
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-keep-edges`,
        operation: OPERATION,
        variant: 'keep-edges',
        bindings: [
          {name: 'sortedSlots', view: sortedSlots, type: 'u32', access: 'read'},
          {name: 'edgeCount', view: edgeCount, type: 'u32', access: 'read'},
          {name: 'parameters', view: props.parameters, type: 'u32', access: 'read'},
          {name: 'neighbors', view: weights.neighbors, type: 'u32', access: 'read'},
          {name: 'keptNeighbors', view: keptNeighbors, type: 'u32', access: 'read_write'}
        ],
        invocationCount: edgeCapacity,
        declarations: `${declarations}
${cutCountWGSL}`,
        body: `let edgeTotal = edgeCount[edgeCountOffset];
  let cutCount = getCutCount(edgeTotal, parameters[parametersOffset]);
  if (index < edgeTotal - cutCount) {
    let slot = sortedSlots[sortedSlotsOffset + index];
    keptNeighbors[keptNeighborsOffset + slot] = neighbors[neighborsOffset + slot];
  }`
      }),
      ...createMaskedWeakComponentNodes<Parameters>(graph, {
        id: `${id}-regions`,
        operation: OPERATION,
        nodeCount: rows,
        offsets: weights.offsets,
        neighbors: keptNeighbors,
        labels: output.labels,
        converged,
        iterations
      }),
      ...(output.status?.iterationCount
        ? [
            createFillNode<Parameters>(graph, {
              id: `${id}-iteration-count`,
              operation: OPERATION,
              view: output.status.iterationCount,
              type: 'u32',
              value: `${iterations}u`
            })
          ]
        : []),
      ...(output.status?.iterationLimitReached && converged
        ? [
            createWGSLKernelNode<Parameters>(graph, {
              id: `${id}-iteration-limit`,
              operation: OPERATION,
              variant: 'iteration-limit',
              bindings: [
                {name: 'converged', view: converged, type: 'u32', access: 'read'},
                {
                  name: 'iterationLimitReached',
                  view: output.status.iterationLimitReached,
                  type: 'u32',
                  access: 'read_write'
                }
              ],
              invocationCount: 1,
              body: 'iterationLimitReached[iterationLimitReachedOffset] = 1u - min(converged[convergedOffset], 1u);'
            })
          ]
        : []),
      ...(output.regionCount
        ? [
            createWGSLKernelNode<Parameters>(graph, {
              id: `${id}-region-count`,
              operation: OPERATION,
              variant: 'region-count',
              bindings: [
                {name: 'edgeCount', view: edgeCount, type: 'u32', access: 'read'},
                {name: 'parameters', view: props.parameters, type: 'u32', access: 'read'},
                {name: 'regionCount', view: output.regionCount, type: 'u32', access: 'read_write'}
              ],
              invocationCount: 1,
              declarations: `${declarations}
${cutCountWGSL}`,
              body: `let edgeTotal = edgeCount[edgeCountOffset];
  regionCount[regionCountOffset] = ROWS - edgeTotal + getCutCount(edgeTotal, parameters[parametersOffset]);`
            })
          ]
        : [])
    ];
    if (treeStatus && output.status?.invalidCount)
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-invalid-count`,
          operation: OPERATION,
          invocationCount: 1,
          bindings: [
            {name: 'treeStatus', view: treeStatus, type: 'u32', access: 'read'},
            {
              name: 'invalidCount',
              view: output.status.invalidCount,
              type: 'u32',
              access: 'read_write'
            }
          ],
          body: `invalidCount[invalidCountOffset] = treeStatus[treeStatusOffset + ${GPU_MINIMUM_SPANNING_TREE_STATUS.invalidCount}u];`
        })
      );
    for (const [name, view, value] of [
      ['labels', output.labels, '0xffffffffu'],
      ['regionCount', output.regionCount, '0u'],
      ['converged', output.status?.converged, '0u'],
      ['iterationLimitReached', output.status?.iterationLimitReached, '0u']
    ] as const) {
      if (!view || lossFlags.length === 0) continue;
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-suppress-${name}`,
          operation: OPERATION,
          variant: 'incomplete-input',
          bindings: [
            ...lossFlags.map((flag, index) => ({
              name: `loss${index}`,
              view: flag,
              type: 'u32' as const,
              access: 'read' as const
            })),
            {name: 'result', view, type: 'u32', access: 'read_write'}
          ],
          invocationCount: name === 'labels' ? rows : 1,
          body: `if (${lossFlags.map((_, index) => `loss${index}[loss${index}Offset] != 0u`).join(' || ')}) {
  result[resultOffset + index] = ${value};
}`
        })
      );
    }
    return nodes;
  }
}

/** Flags that make the retained contiguity graph incomplete. */
function getRedcapWeightLossFlags(weights: GPUSpatialWeights): GraphDataView<'uint32'>[] {
  return [weights.status?.overflow, weights.status?.candidateOverflow].filter(
    (view): view is GraphDataView<'uint32'> => view !== undefined
  );
}

// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  validateGraphViewsBelongToGraph,
  validateSinglePackedView,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {SPATIAL_STATISTICS_COUNTS_WGSL} from '../spatial-weights/spatial-statistics-counts-wgsl';
import {getPairwiseSumWGSL} from '../spatial-regression/pairwise-sum-wgsl';
import {GET_NAN_WGSL} from '../../analysis-core/kernels/wgsl-common';
import type {GPUSpatialWeights} from '../spatial-weights/spatial-weights';
import {
  GPUCategoryPairCounts,
  GPU_CATEGORY_PAIR_COUNTS_MAXIMUM_CATEGORY_COUNT
} from './gpu-category-pair-counts';

import {validateSpatialStatisticsBindings} from '../spatial-weights/spatial-statistics-bindings';
import {validateDisjointWritableViews} from '../../analysis-core/kernels/graph-chunk-dispatch';
import {validateGPUSpatialWeights} from '../spatial-weights/spatial-weights';
import {snapshotCategoricalStatisticsProps} from './categorical-construction-snapshot';
import {
  validateCapacity,
  validateGraphOutputsDisjointFromInputs
} from '../../analysis-core/kernels/gpu-contributor-utils';

const OPERATION = 'GPUSpatialEntropy';

/** Layout of the `output.summary` words written by {@link GPUSpatialEntropy}. All entropies are in nats. */
export const GPU_SPATIAL_ENTROPY_LAYOUT = {
  /** Shannon entropy `-sum_a p_a ln p_a` of the category shares of the rows (Batty's aspatial term). */
  shannon: 0,
  /** Entropy of the category of the origin row of a link. */
  originEntropy: 1,
  /** Entropy of the category of the destination row of a link. */
  destinationEntropy: 2,
  /**
   * Co-occurrence entropy `-sum_ab p_ab ln p_ab` of the ordered category pair of a link
   * (Leibovici 2009), `p_ab = n_ab / L` with `L` the number of valid links.
   */
  cooccurrence: 3,
  /**
   * Endpoint-category mutual information `originEntropy + destinationEntropy - cooccurrence`:
   * the information the origin category carries about the
   * destination category. 0 for a spatially random map up to sampling noise; at most
   * `min(originEntropy, destinationEntropy)`.
   */
  mutualInformation: 4,
  /** `mutualInformation / destinationEntropy`, the share of the destination entropy that is spatial; NaN if that entropy is 0. */
  relativeMutualInformation: 5,
  /** Number of valid ordered links `L`. */
  linkCount: 6,
  /** Number of rows with a valid category. */
  rowCount: 7,
  /** Total word count. */
  length: 8
} as const;

/** Properties for {@link GPUSpatialEntropy}. */
export type GPUSpatialEntropyProps = {
  /** Prefix for generated node IDs. Defaults to `'spatial-entropy'`. */
  id?: string;
  /**
   * Square weights; each active slot (positive weight, another row: the shared neighbor rule) is
   * one ordered link. Self slots and zero-weight slots are not links.
   * Supplied completeness status views are rejected because these outputs cannot qualify loss;
   * verify the producer's evidence before supplying a complete CSR without those status ports.
   */
  weights: GPUSpatialWeights;
  /** Category per row, `0xffffffff` for none. */
  categories: GraphDataView<'uint32'>;
  /** Number of categories, at most {@link GPU_CATEGORY_PAIR_COUNTS_MAXIMUM_CATEGORY_COUNT}. */
  categoryCount: number;
  /** Caller-owned outputs. */
  output: GPUSpatialEntropyOutput;
};

/** Caller-owned outputs of {@link GPUSpatialEntropy}. */
export type GPUSpatialEntropyOutput = {
  /** Exact uint32 [validRowCount, validOrderedLinkCount], matching the float32 summary counters. */
  counts?: GraphDataView<'uint32'>;
  /** Summary words, see {@link GPU_SPATIAL_ENTROPY_LAYOUT}. */
  summary: GraphDataView<'float32'>;
  /** Optional ordered link counts, `categoryCount ^ 2` entries (see `GPUCategoryPairCounts`). */
  pairCounts?: GraphDataView<'uint32'>;
};

/**
 * Global spatial entropy of a categorical map over a weights graph: the Shannon entropy of the
 * categories, the co-occurrence entropy of the category pairs at the links (Leibovici 2009) and
 * the mutual information between a link's two category labels. Use contiguity weights, or
 * distance-band weights per distance class for the Leibovici distance-class form.
 *
 * Counts come from {@link GPUCategoryPairCounts} (exact integer atomics). Eight publication
 * invocations publish pre-reduced terms. Four reduction invocations each compute one entropy,
 * visiting at most 32 x 32 counters, bounded by GPU_CATEGORY_PAIR_COUNTS_MAXIMUM_CATEGORY_COUNT = 32. Entropy terms use fixed pairwise sums
 * in float32. An absolute mutual-information tolerance near zero requires device measurement.
 *
 * **Reference.** Shannon category entropy and the ordered co-occurrence definition above;
 * expectations are independent binary64 evaluations, not a pinned R package run.
 * Batty's area-weighted entropy and Altieri's decomposition into mutual information and
 * conditional entropy between unordered category pairs and distance classes are unavailable:
 * this interface has neither area nor distance-class inputs. Supply separate distance-band
 * graphs to compare ordered pair entropies across bands.
 */
export class GPUSpatialEntropy implements GPUCommandNodeProducer {
  /** Prefix for every node ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUSpatialEntropyProps;

  constructor(props: GPUSpatialEntropyProps) {
    props = snapshotCategoricalStatisticsProps(props, ['summary', 'counts', 'pairCounts']);
    const id = props.id ?? 'spatial-entropy';
    this.id = id;
    this.props = props;
    validateSinglePackedView(props.output.summary, ['float32'], `${id} output.summary`);
    validateRequirement(
      id,
      props.output.summary.length >= GPU_SPATIAL_ENTROPY_LAYOUT.length,
      `output.summary must hold ${GPU_SPATIAL_ENTROPY_LAYOUT.length} words`
    );
    validateRequirement(
      id,
      props.categoryCount <= GPU_CATEGORY_PAIR_COUNTS_MAXIMUM_CATEGORY_COUNT,
      `categoryCount must be at most ${GPU_CATEGORY_PAIR_COUNTS_MAXIMUM_CATEGORY_COUNT}`
    );
    validateCapacity(
      id,
      'categoryCount',
      props.categoryCount,
      1,
      GPU_CATEGORY_PAIR_COUNTS_MAXIMUM_CATEGORY_COUNT
    );
    const rows = validateGPUSpatialWeights(id, props.weights);
    validateRequirement(
      id,
      !props.weights.status?.overflow &&
        !props.weights.status?.candidateOverflow &&
        !props.weights.status?.requiredCount &&
        !props.weights.status?.complete &&
        !props.weights.status?.converged &&
        !props.weights.status?.iterationLimitReached &&
        !props.weights.status?.unresolvedCount,
      'weights.status overflow, candidateOverflow, requiredCount, complete, converged, iterationLimitReached and unresolvedCount ports cannot be checked or propagated by the entropy outputs; provide a complete CSR after verifying all supplied completeness descriptors, then omit those ports'
    );
    validateSinglePackedView(props.categories, ['uint32'], `${id} categories`);
    validateRequirement(
      id,
      props.categories.length === rows,
      'categories length must equal the weights row count'
    );
    if (props.output.pairCounts) {
      validateSinglePackedView(props.output.pairCounts, ['uint32'], `${id} output.pairCounts`);
      validateRequirement(
        id,
        props.output.pairCounts.length >= props.categoryCount ** 2,
        'output.pairCounts must hold categoryCount squared entries'
      );
    }
    if (props.output.counts) {
      validateSinglePackedView(props.output.counts, ['uint32'], `${id} output.counts`);
      validateCapacity(id, 'output.counts length', props.output.counts.length, 2, 2);
    }
    validateDisjointWritableViews(id, props.output);
    validateGraphOutputsDisjointFromInputs(id, Object.values(props.output), [
      props.categories,
      props.weights.offsets,
      props.weights.neighbors,
      props.weights.weights
    ]);
  }

  /** Returns the pair count nodes and the finalize node. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    validateSpatialStatisticsBindings(graph, id, props, {
      'pair count scratch': props.categoryCount ** 2 * 4,
      'category scratch': props.categoryCount * 4,
      'entropy totals': 8,
      'entropy terms': 16
    });
    const {categoryCount, output} = props;
    validateGraphViewsBelongToGraph(id, graph, [output.summary, output.pairCounts]);
    const pairCounts =
      output.pairCounts ??
      createTransientView(graph, `${id}-pair-counts`, 'uint32', categoryCount ** 2);
    const categoryRowCounts = createTransientView(
      graph,
      `${id}-category-row-counts`,
      'uint32',
      categoryCount
    );
    const totals = createTransientView(graph, `${id}-summary-totals`, 'uint32', 2);
    const entropies = createTransientView(graph, `${id}-summary-entropies`, 'float32', 4);
    return [
      ...new GPUCategoryPairCounts({
        id: `${id}-pairs`,
        weights: props.weights,
        categories: props.categories,
        categoryCount,
        output: {pairCounts, categoryRowCounts}
      }).getCommandNodes(graph),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-summary-totals`,
        operation: OPERATION,
        variant: 'summary-totals',
        invocationCount: 2,
        bindings: [
          {name: 'pairCounts', view: pairCounts, type: 'u32', access: 'read'},
          {name: 'categoryRowCounts', view: categoryRowCounts, type: 'u32', access: 'read'},
          {name: 'totals', view: totals, type: 'u32', access: 'read_write'}
        ],
        body: `var total = 0u;
  if (index == 0u) { for (var category = 0u; category < ${categoryCount}u; category++) { total += categoryRowCounts[categoryRowCountsOffset + category]; } }
  else { for (var pair = 0u; pair < ${categoryCount ** 2}u; pair++) { total += pairCounts[pairCountsOffset + pair]; } }
  totals[totalsOffset + index] = total;`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-summary-entropies`,
        operation: OPERATION,
        variant: 'summary',
        bindings: [
          {name: 'pairCounts', view: pairCounts, type: 'u32', access: 'read'},
          {name: 'categoryRowCounts', view: categoryRowCounts, type: 'u32', access: 'read'},
          {name: 'totals', view: totals, type: 'u32', access: 'read'},
          {name: 'entropies', view: entropies, type: 'f32', access: 'read_write'}
        ],
        invocationCount: 4,
        declarations: `const CATEGORIES: u32 = ${categoryCount}u;
${GET_NAN_WGSL}
${SPATIAL_STATISTICS_COUNTS_WGSL}
fn getTerm(count: u32, total: u32) -> f32 {
  if (count == 0u) {
    return 0.0;
  }
  let share = divideSpatialCounts(count, total);
  return -share * log(share);
}`,
        body: `let rowTotal = totals[totalsOffset]; let linkTotal = totals[totalsOffset + 1u];
  if (index == 0u) {
  ${getPairwiseSumWGSL({count: 'CATEGORIES', name: 'shannon', term: 'getTerm(categoryRowCounts[categoryRowCountsOffset + shannonItem], rowTotal)'})}
    entropies[entropiesOffset] = shannon; return;
  }
  if (index == 3u) {
  ${getPairwiseSumWGSL({count: 'CATEGORIES * CATEGORIES', name: 'joint', term: 'getTerm(pairCounts[pairCountsOffset + jointItem], linkTotal)'})}
    entropies[entropiesOffset + 3u] = joint; return;
  }
  if (index == 1u) {
  ${getPairwiseSumWGSL({
    count: 'CATEGORIES',
    name: 'origin',
    setup: `
    var originCount = 0u;
    for (var b = 0u; b < CATEGORIES; b++) {
      originCount += pairCounts[pairCountsOffset + originItem * CATEGORIES + b];
    }`,
    term: 'getTerm(originCount, linkTotal)'
  })}
    entropies[entropiesOffset + 1u] = origin; return;
  }
  ${getPairwiseSumWGSL({
    count: 'CATEGORIES',
    name: 'destination',
    setup: `
    var destinationCount = 0u;
    for (var b = 0u; b < CATEGORIES; b++) {
      destinationCount += pairCounts[pairCountsOffset + b * CATEGORIES + destinationItem];
    }`,
    term: 'getTerm(destinationCount, linkTotal)'
  })}
  entropies[entropiesOffset + 2u] = destination;`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-summary`,
        operation: OPERATION,
        variant: 'publish-summary',
        invocationCount: GPU_SPATIAL_ENTROPY_LAYOUT.length,
        bindings: [
          {name: 'entropies', view: entropies, type: 'f32', access: 'read'},
          {name: 'totals', view: totals, type: 'u32', access: 'read'},
          {name: 'summary', view: output.summary, type: 'f32', access: 'read_write'},
          ...(output.counts
            ? [
                {
                  name: 'counts',
                  view: output.counts,
                  type: 'u32' as const,
                  access: 'read_write' as const
                }
              ]
            : [])
        ],
        declarations: GET_NAN_WGSL,
        body: `let shannon = entropies[entropiesOffset]; let origin = entropies[entropiesOffset + 1u]; let destination = entropies[entropiesOffset + 2u]; let joint = entropies[entropiesOffset + 3u];
  let rowTotal = totals[totalsOffset]; let linkTotal = totals[totalsOffset + 1u];
  ${output.counts ? 'if (index < 2u) { counts[countsOffset + index] = totals[totalsOffset + index]; }' : ''}
  let nan = getNaN();
  let hasLinks = linkTotal > 0u;
  let mutual = select(nan, max(origin + destination - joint, 0.0), hasLinks);
  let result = array<f32, 8>(select(nan, shannon, rowTotal > 0u), select(nan, origin, hasLinks),
    select(nan, destination, hasLinks), select(nan, joint, hasLinks), mutual,
    select(nan, mutual / destination, hasLinks && destination > 0.0), f32(linkTotal), f32(rowTotal));
  summary[summaryOffset + index] = result[index];`
      })
    ];
  }
}

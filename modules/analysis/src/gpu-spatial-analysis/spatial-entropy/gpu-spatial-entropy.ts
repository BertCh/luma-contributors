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
import {GET_NAN_WGSL} from '../../analysis-core/kernels/wgsl-common';
import type {GPUSpatialWeights} from '../spatial-weights/spatial-weights';
import {
  GPUCategoryPairCounts,
  GPU_CATEGORY_PAIR_COUNTS_MAXIMUM_CATEGORY_COUNT
} from './gpu-category-pair-counts';

import {validateSpatialStatisticsBindings} from '../spatial-weights/spatial-statistics-bindings';
import {validateDisjointWritableViews} from '../../analysis-core/kernels/graph-chunk-dispatch';
import {validateGPUSpatialWeights} from '../spatial-weights/spatial-weights';
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
   * Spatial mutual information `originEntropy + destinationEntropy - cooccurrence`
   * (Altieri, Cocchi and Roli 2018): the information the origin category carries about the
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
 * the spatial mutual information between a link's two ends, which is the part of the entropy that
 * is spatial in the decomposition of Altieri, Cocchi and Roli (2018). Use contiguity weights, or
 * distance-band weights per distance class for the Leibovici distance-class form.
 *
 * Counts come from {@link GPUCategoryPairCounts} (exact integer atomics); one finalize invocation
 * then loops over the at most `32 x 32` counters. Entropies are evaluated in float32 from exact
 * counts, so they agree with binary64 to about `1e-6` relative.
 *
 * **Reference.** The R package `spatentropy` (GPL, not installed) defines these measures; no
 * pinned package was run. The reference is a binary64 CPU evaluation of the formulas above.
 * Batty's area-weighted spatial entropy and the Altieri conditional (distance-class) terms are not
 * implemented.
 */
export class GPUSpatialEntropy implements GPUCommandNodeProducer {
  /** Prefix for every node ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUSpatialEntropyProps;

  constructor(props: GPUSpatialEntropyProps) {
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
      'category scratch': props.categoryCount * 4
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
    return [
      ...new GPUCategoryPairCounts({
        id: `${id}-pairs`,
        weights: props.weights,
        categories: props.categories,
        categoryCount,
        output: {pairCounts, categoryRowCounts}
      }).getCommandNodes(graph),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-summary`,
        operation: OPERATION,
        variant: 'summary',
        bindings: [
          {name: 'pairCounts', view: pairCounts, type: 'u32', access: 'read'},
          {name: 'categoryRowCounts', view: categoryRowCounts, type: 'u32', access: 'read'},
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
        invocationCount: 1,
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
        body: `var rowTotal = 0u;
  var linkTotal = 0u;
  for (var a = 0u; a < CATEGORIES; a++) {
    rowTotal += categoryRowCounts[categoryRowCountsOffset + a];
    for (var b = 0u; b < CATEGORIES; b++) {
      linkTotal += pairCounts[pairCountsOffset + a * CATEGORIES + b];
    }
  }
  ${output.counts ? 'counts[countsOffset] = rowTotal; counts[countsOffset + 1u] = linkTotal;' : ''}
  var shannon = 0.0;
  var origin = 0.0;
  var destination = 0.0;
  var joint = 0.0;
  for (var a = 0u; a < CATEGORIES; a++) {
    shannon += getTerm(categoryRowCounts[categoryRowCountsOffset + a], rowTotal);
    var originCount = 0u;
    var destinationCount = 0u;
    for (var b = 0u; b < CATEGORIES; b++) {
      originCount += pairCounts[pairCountsOffset + a * CATEGORIES + b];
      destinationCount += pairCounts[pairCountsOffset + b * CATEGORIES + a];
      joint += getTerm(pairCounts[pairCountsOffset + a * CATEGORIES + b], linkTotal);
    }
    origin += getTerm(originCount, linkTotal);
    destination += getTerm(destinationCount, linkTotal);
  }
  let nan = getNaN();
  let hasLinks = linkTotal > 0u;
  let mutual = select(nan, max(origin + destination - joint, 0.0), hasLinks);
  summary[summaryOffset + ${GPU_SPATIAL_ENTROPY_LAYOUT.shannon}u] = select(nan, shannon, rowTotal > 0u);
  summary[summaryOffset + ${GPU_SPATIAL_ENTROPY_LAYOUT.originEntropy}u] = select(nan, origin, hasLinks);
  summary[summaryOffset + ${GPU_SPATIAL_ENTROPY_LAYOUT.destinationEntropy}u] = select(nan, destination, hasLinks);
  summary[summaryOffset + ${GPU_SPATIAL_ENTROPY_LAYOUT.cooccurrence}u] = select(nan, joint, hasLinks);
  summary[summaryOffset + ${GPU_SPATIAL_ENTROPY_LAYOUT.mutualInformation}u] = mutual;
  summary[summaryOffset + ${GPU_SPATIAL_ENTROPY_LAYOUT.relativeMutualInformation}u] = select(nan, mutual / destination, hasLinks && destination > 0.0);
  summary[summaryOffset + ${GPU_SPATIAL_ENTROPY_LAYOUT.linkCount}u] = f32(linkTotal);
  summary[summaryOffset + ${GPU_SPATIAL_ENTROPY_LAYOUT.rowCount}u] = f32(rowTotal);`
      })
    ];
  }
}

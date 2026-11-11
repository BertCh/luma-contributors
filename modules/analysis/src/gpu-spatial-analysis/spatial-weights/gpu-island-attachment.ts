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
import {
  createFillNode,
  createWGSLKernelNode,
  type WGSLKernelBinding
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  validateCapacity,
  validateRequirement,
  validateSinglePackedView,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph
} from '../../analysis-core/kernels/gpu-contributor-utils';
import type {GPUUncertaintyStatusPort} from '../../analysis-core/contracts/status';
import {IS_FINITE_BITS_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {GPUNeighborSearch} from '../neighbor-search/gpu-neighbor-search';
import {createNeighborSearchDerivedParameterNodes} from '../neighbor-search/neighbor-search-derived-parameters';
import {getSlotGroupingNodes} from './slot-grouping';
import {createCSRStages} from './spatial-weights-algebra-wgsl';
import {
  type GPUSpatialWeights,
  type GPUSpatialWeightsOutput,
  describeGPUSpatialWeights,
  validateGPUSpatialWeights,
  validateGPUSpatialWeightsOutput,
  getGPUSpatialWeightsViews,
  getGPUSpatialWeightsStatusViews
} from './spatial-weights';

/**
 * Largest row count for which {@link GPUIslandAttachmentProps.nearestSearch} `'auto'` scans every
 * representative per island. Above it the candidates come from a `GPUNeighborSearch` grid. The value
 * follows the cut-over of local kriging (`GPU_LOCAL_KRIGING_BRUTE_FORCE_MAXIMUM_SAMPLE_COUNT`) and
 * of the segment-BVH contributors.
 */
export const GPU_ISLAND_ATTACHMENT_SCAN_MAXIMUM_ROW_COUNT = 4096;

/** How {@link GPUIslandAttachment} finds the nearest representative of each island. */
export type GPUIslandAttachmentNearestSearch = 'auto' | 'scan' | 'grid';

/**
 * Nearest candidates (the island itself included) the grid returns per island. The attachment is
 * decided among them with the scan's own comparison; an island whose list cannot prove the answer
 * (see {@link GPUIslandAttachment}) rescans every representative.
 */
const ISLAND_ATTACHMENT_CANDIDATE_COUNT = 16;

/**
 * Relative margin by which the farthest listed candidate must exceed the chosen one before the
 * list proves the answer. It covers the float32 rounding differences between the grid's
 * `dx^2 + dy^2` ranking and the scan's scaled comparison.
 */
const ISLAND_ATTACHMENT_PROOF_MARGIN = 1.00001;

export type GPUIslandAttachmentProps = {
  id?: string;
  weights: GPUSpatialWeights;
  /** Planar point representatives, e.g. explicitly chosen polygon centroids. */
  positions: GraphDataView<'float32x2'>;
  /** Add reciprocal edges by default. False adds only island -> nearest. */
  reciprocal?: boolean;
  /**
   * How the nearest representative of each island is found. Compile-time, default `'auto'`. Every
   * choice selects the same attachment (the nearest finite representative by the scan's comparison,
   * ties to the lowest row), so results do not depend on it, only speed. `'scan'` tests every
   * representative per island; `'grid'` takes the nearest candidates from a `GPUNeighborSearch`
   * grid; `'auto'` scans up to {@link GPU_ISLAND_ATTACHMENT_SCAN_MAXIMUM_ROW_COUNT} rows and uses
   * the grid above.
   */
  nearestSearch?: GPUIslandAttachmentNearestSearch;
  output: GPUSpatialWeightsOutput & {
    /**
     * `uncertainCount` is optional: the exact number of islands whose nearest representative the
     * grid's candidate list could not prove, so that island scanned every representative in one
     * invocation (see {@link GPUIslandAttachment}). Always 0 for `nearestSearch: 'scan'`.
     */
    status: GPUSpatialWeightsOutput['status'] & GPUUncertaintyStatusPort;
    /** Per row: selected nearest row, or 0xffffffff when no attachment was needed/possible. */
    attachment: GraphDataView<'uint32'>;
    /** Per written edge: 1 added by this operation, 0 copied. */
    added: GraphDataView<'uint32'>;
  };
};

/**
 * Attach original empty CSR rows to their nearest finite planar representative (ties: lowest ID),
 * retaining original weights and adding binary edges. Existing CSR must satisfy sorted-unique,
 * finite positive off-diagonal input invariants; zero-weight-only rows are not treated as empty.
 * Selection is simultaneous against the ORIGINAL graph, so two islands can attach to each other.
 * Output is sorted, unique and deterministic under truncation. Never silently row-standardizes.
 * Nonfinite representatives cannot acquire an attachment, but existing edges are preserved.
 * With `nearestSearch: 'scan'` each island scans every representative for its nearest one. With
 * `'grid'` (and `'auto'` above {@link GPU_ISLAND_ATTACHMENT_SCAN_MAXIMUM_ROW_COUNT} rows) a
 * `GPUNeighborSearch` k-nearest cross join over a grid lists the 16 nearest representatives of each
 * island by float32 `dx^2 + dy^2`, ties to the lowest row; the island's own row is skipped, and the
 * kernel then chooses among them with the scan's comparison, so ties still go to the lowest row.
 * The list proves the choice when it holds every finite representative, or when its farthest entry
 * is farther than the chosen one by more than a relative 1e-5 (the two rankings can differ only by
 * float32 rounding). Otherwise, for example when 16 or more representatives are tied or
 * coincident at the nearest distance, or when coordinates are so large that the bounds or squared
 * distances overflow, that island rescans every representative, so the answer always equals the
 * scan's. That rescan is a loop over every row inside one invocation, so its cost is visible, not
 * silent: the rule is that the result stays exact and `output.status.uncertainCount`, when given,
 * counts the islands that used it (0 with `'scan'`; with `'grid'` or `'auto'` above the cut-over a
 * nonzero count means those islands each paid one pass over all rows, up to 65,535, and a count
 * near the number of islands means the representatives are too clustered or too large for the grid
 * list to help: choose `'scan'` or deduplicate coincident representatives). The output rows then
 * merge the original row, the row's attachment and the islands attached to it (grouped by a stable
 * sort), so rows that are not islands cost their own degree rather than a scan of all rows.
 */
export class GPUIslandAttachment implements GPUCommandNodeProducer {
  readonly id: string;
  readonly outputWeights: GPUSpatialWeights;
  constructor(readonly props: GPUIslandAttachmentProps) {
    this.id = props.id ?? 'island-attachment';
    const nearestSearch = props.nearestSearch ?? 'auto';
    validateRequirement(
      this.id,
      ['auto', 'scan', 'grid'].includes(nearestSearch),
      "nearestSearch must be 'auto', 'scan' or 'grid'"
    );
    const rows = validateGPUSpatialWeights(this.id, props.weights);
    validateCapacity(this.id, 'rows', rows, 1, 65535);
    if (
      validateGPUSpatialWeightsOutput(this.id, props.output) !== rows ||
      props.positions.length !== rows ||
      props.output.attachment.length !== rows ||
      props.output.added.length !== props.output.neighbors.length
    )
      throw new Error(`${this.id}: shape mismatch`);
    if (props.output.distances) throw new Error(`${this.id}: distances output is not supported`);
    validateSinglePackedView(props.positions, ['float32x2'], this.id);
    validateSinglePackedView(props.output.attachment, ['uint32'], this.id);
    validateSinglePackedView(props.output.added, ['uint32'], this.id);
    validateGraphOutputsDisjointFromInputs(
      this.id,
      [
        ...getGPUSpatialWeightsViews(props.output),
        ...getGPUSpatialWeightsStatusViews(props.output),
        props.output.attachment,
        props.output.added,
        props.output.status.uncertainCount
      ],
      [...getGPUSpatialWeightsViews(props.weights), props.positions]
    );
    this.outputWeights = describeGPUSpatialWeights(props.output, {
      information: {
        rowCount: rows,
        transform: 'raw',
        symmetric:
          props.reciprocal !== false && props.weights.information?.symmetric === true
            ? true
            : 'unknown',
        hasSelfWeights: false,
        provenance: {source: 'GPUIslandAttachment', criterion: 'nearest-original-island'}
      },
      status: props.output.status
    });
  }
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {props, id} = this;
    const rows = props.positions.length;
    const capacity = props.output.neighbors.length;
    const binding = (
      name: string,
      view: GraphDataView,
      access: 'read' | 'read_write' = 'read',
      type: 'f32' | 'u32' | 'atomic<u32>' = 'u32'
    ): WGSLKernelBinding => ({name, view, access, type});
    const node = (
      name: string,
      invocationCount: number,
      bindings: WGSLKernelBinding[],
      body: string,
      declarations = ''
    ) =>
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-${name}`,
        operation: 'GPUIslandAttachment',
        variant: name,
        invocationCount,
        bindings,
        declarations,
        body
      });
    validateGraphViewsBelongToGraph(id, graph, [
      ...getGPUSpatialWeightsViews(props.weights),
      props.positions,
      ...getGPUSpatialWeightsViews(props.output),
      ...getGPUSpatialWeightsStatusViews(props.output),
      props.output.attachment,
      props.output.added,
      props.output.status.uncertainCount
    ]);
    const reciprocal = props.reciprocal !== false;
    const nearestSearch = props.nearestSearch ?? 'auto';
    const useGrid =
      nearestSearch === 'grid' ||
      (nearestSearch === 'auto' && rows > GPU_ISLAND_ATTACHMENT_SCAN_MAXIMUM_ROW_COUNT);
    // Halving before subtraction prevents overflow for opposite finite coordinates. Compare both
    // distances at their shared scale so squared norms neither overflow nor collapse to zero.
    // Keep the scale multiplier normal even on devices that flush subnormal factors.
    const nearestDeclarations = `${IS_FINITE_BITS_WGSL}
const NO_ATTACHMENT: u32 = 0xffffffffu;
fn isCloserDifference(difference: vec2<f32>, bestDifference: vec2<f32>, isFirst: bool) -> bool {
 let scale = max(max(abs(difference.x), abs(difference.y)), max(abs(bestDifference.x), abs(bestDifference.y)));
 var closer = isFirst;
 if (scale > 0.0) { let exponent = clamp(i32((bitcast<u32>(scale) >> 23u) & 255u) - 127, -120, 120); let normalizer = bitcast<f32>(u32(127 - exponent) << 23u); let normalized = difference * normalizer; let bestNormalized = bestDifference * normalizer; closer = closer || dot(normalized, normalized) < dot(bestNormalized, bestNormalized); }
 return closer;
}`;
    const scanWGSL = `for (var other = 0u; other < ${rows}u; other++) { let candidate = vec2<f32>(positions[positionsOffset + 2u * other], positions[positionsOffset + 2u * other + 1u]);
 if (other != index && isFiniteBits(candidate.x) && isFiniteBits(candidate.y)) {
 let difference = position * 0.5 - candidate * 0.5;
 if (isCloserDifference(difference, bestDifference, nearest == NO_ATTACHMENT)) { bestDifference = difference; nearest = other; }
 }
}`;
    const nodes: GPUCommandNode<Parameters>[] = [];
    const uncertainCount = props.output.status.uncertainCount;
    const clearUncertainCount = () =>
      uncertainCount
        ? [
            createFillNode<Parameters>(graph, {
              id: `${id}-uncertain-count-clear`,
              operation: 'GPUIslandAttachment',
              view: uncertainCount,
              type: 'u32',
              value: '0u',
              componentCount: 1
            })
          ]
        : [];
    if (!useGrid) {
      nodes.push(
        ...clearUncertainCount(),
        node(
          'nearest',
          rows,
          [
            binding('positions', props.positions, 'read', 'f32'),
            binding('offsets', props.weights.offsets),
            binding('attachment', props.output.attachment, 'read_write')
          ],
          `var nearest = NO_ATTACHMENT; var bestDifference = vec2<f32>(0.0);
let position = vec2<f32>(positions[positionsOffset + 2u * index], positions[positionsOffset + 2u * index + 1u]);
if (offsets[offsetsOffset + index] == offsets[offsetsOffset + index + 1u] && isFiniteBits(position.x) && isFiniteBits(position.y)) {
${scanWGSL}
}
attachment[attachmentOffset + index] = nearest;`,
          nearestDeclarations
        )
      );
    } else {
      // Islands are the query rows of a k-nearest cross join over every representative; the grid
      // bounds are the extent of the finite representatives, so every one of them is a target.
      const islandFlags = createTransientView(graph, `${id}-island-flags`, 'uint32', rows);
      const search = createNeighborSearchDerivedParameterNodes<Parameters>(graph, {
        id: `${id}-index-parameters`,
        operation: 'GPUIslandAttachment',
        positions: props.positions,
        radius: Infinity
      });
      const candidateOffsets = createTransientView(
        graph,
        `${id}-index-offsets`,
        'uint32',
        rows + 1
      );
      const candidateNeighbors = createTransientView(
        graph,
        `${id}-index-neighbors`,
        'uint32',
        rows * ISLAND_ATTACHMENT_CANDIDATE_COUNT
      );
      const candidateDistances = createTransientView(
        graph,
        `${id}-index-distances`,
        'float32',
        rows * ISLAND_ATTACHMENT_CANDIDATE_COUNT
      );
      nodes.push(
        node(
          'island-flags',
          rows,
          [
            binding('offsets', props.weights.offsets),
            binding('islandFlags', islandFlags, 'read_write')
          ],
          `islandFlags[islandFlagsOffset + index] = select(0u, 1u, offsets[offsetsOffset + index] == offsets[offsetsOffset + index + 1u]);`
        ),
        ...search.nodes,
        ...new GPUNeighborSearch({
          id: `${id}-index`,
          mode: 'knn',
          k: ISLAND_ATTACHMENT_CANDIDATE_COUNT,
          positions: props.positions,
          queryPositions: props.positions,
          queryMask: islandFlags,
          parameters: search.parameters,
          gridSize: search.gridSize,
          output: {
            offsets: candidateOffsets,
            neighbors: candidateNeighbors,
            weights: createTransientView(
              graph,
              `${id}-index-weights`,
              'float32',
              rows * ISLAND_ATTACHMENT_CANDIDATE_COUNT
            ),
            distances: candidateDistances,
            status: {overflow: createTransientView(graph, `${id}-index-overflow`, 'uint32', 1)}
          }
        }).getCommandNodes(graph),
        ...clearUncertainCount(),
        node(
          'nearest',
          rows,
          [
            binding('positions', props.positions, 'read', 'f32'),
            binding('offsets', props.weights.offsets),
            binding('candidateOffsets', candidateOffsets),
            binding('candidateNeighbors', candidateNeighbors),
            binding('candidateDistances', candidateDistances, 'read', 'f32'),
            binding('attachment', props.output.attachment, 'read_write'),
            ...(uncertainCount
              ? [binding('uncertainCount', uncertainCount, 'read_write', 'atomic<u32>')]
              : [])
          ],
          `var nearest = NO_ATTACHMENT; var bestDifference = vec2<f32>(0.0);
let position = vec2<f32>(positions[positionsOffset + 2u * index], positions[positionsOffset + 2u * index + 1u]);
if (offsets[offsetsOffset + index] == offsets[offsetsOffset + index + 1u] && isFiniteBits(position.x) && isFiniteBits(position.y)) {
 let begin = candidateOffsets[candidateOffsetsOffset + index];
 let found = candidateOffsets[candidateOffsetsOffset + index + 1u] - begin;
 var farthest = 0.0;
 for (var slot = 0u; slot < found; slot++) {
  let other = candidateNeighbors[candidateNeighborsOffset + begin + slot];
  farthest = max(farthest, candidateDistances[candidateDistancesOffset + begin + slot]);
  let candidate = vec2<f32>(positions[positionsOffset + 2u * other], positions[positionsOffset + 2u * other + 1u]);
  if (other != index && isFiniteBits(candidate.x) && isFiniteBits(candidate.y)) {
   let difference = position * 0.5 - candidate * 0.5;
   if (isCloserDifference(difference, bestDifference, nearest == NO_ATTACHMENT)) { bestDifference = difference; nearest = other; }
  }
 }
 // The list is complete when it holds fewer than the candidate count (an empty list means the grid
 // was unusable, for example overflowing bounds). Otherwise the answer is proven when the
 // farthest entry is clearly farther than the chosen one; if not, rescan every representative.
 var proven = found > 0u && found < ${ISLAND_ATTACHMENT_CANDIDATE_COUNT}u;
 if (found == ${ISLAND_ATTACHMENT_CANDIDATE_COUNT}u && nearest != NO_ATTACHMENT) {
  let chosenDistance = 2.0 * length(bestDifference);
  proven = isFiniteBits(chosenDistance) && isFiniteBits(farthest) && farthest > chosenDistance * ${ISLAND_ATTACHMENT_PROOF_MARGIN};
 }
 if (!proven) {
  ${uncertainCount ? 'atomicAdd(&uncertainCount[uncertainCountOffset], 1u);' : ''}
  nearest = NO_ATTACHMENT; bestDifference = vec2<f32>(0.0);
${scanWGSL}
 }
}
attachment[attachmentOffset + index] = nearest;`,
          nearestDeclarations
        )
      );
    }
    // Islands grouped by the row they attach to: a stable sort by target keeps ascending island
    // order inside each group, so every row merges three ascending lists without scanning all rows.
    let inverse: {offsets: GraphDataView<'uint32'>; islands: GraphDataView<'uint32'>} | undefined;
    if (reciprocal) {
      const inverseOffsets = createTransientView(
        graph,
        `${id}-inverse-offsets`,
        'uint32',
        rows + 1
      );
      const slotKeys = createTransientView(graph, `${id}-inverse-keys`, 'uint32', rows);
      const slotIndices = createTransientView(graph, `${id}-inverse-indices`, 'uint32', rows);
      nodes.push(
        node(
          'inverse-keys',
          rows,
          [
            binding('attachment', props.output.attachment),
            binding('slotKeys', slotKeys, 'read_write'),
            binding('slotIndices', slotIndices, 'read_write')
          ],
          `let attachedRow = attachment[attachmentOffset + index];
slotKeys[slotKeysOffset + index] = select(${rows}u, attachedRow, attachedRow < ${rows}u);
slotIndices[slotIndicesOffset + index] = index;`
        )
      );
      const grouping = getSlotGroupingNodes<Parameters>(graph, {
        id: `${id}-inverse`,
        operation: 'GPUIslandAttachment',
        slotKeys,
        slotIndices,
        columns: rows,
        columnOffsets: inverseOffsets
      });
      nodes.push(...grouping.nodes);
      inverse = {offsets: inverseOffsets, islands: grouping.sortedSlots};
    }
    // Ascending merge of the original row, its own attachment and the islands attached to it.
    // Original edges keep their slot (and weight); the others are added edges to other rows.
    const visit = (onItem: string) => `var originalNext = offsets[offsetsOffset + index];
let originalEnd = offsets[offsetsOffset + index + 1u];
var attached = attachment[attachmentOffset + index];
${
  reciprocal
    ? `var inverseNext = inverseOffsets[inverseOffsetsOffset + index];
let inverseEnd = inverseOffsets[inverseOffsetsOffset + index + 1u];`
    : ''
}
loop {
 var originalId = NO_SLOT; if (originalNext < originalEnd) { originalId = neighbors[neighborsOffset + originalNext]; }
 var inverseId = NO_SLOT;${reciprocal ? ' if (inverseNext < inverseEnd) { inverseId = inverseIslands[inverseIslandsOffset + inverseNext]; }' : ''}
 let other = min(min(originalId, attached), inverseId);
 if (other == NO_SLOT) { break; }
 var originalSlot = NO_SLOT;
 if (originalId == other) { originalSlot = originalNext; while (originalNext < originalEnd && neighbors[neighborsOffset + originalNext] == other) { originalNext++; } }
 if (attached == other) { attached = NO_SLOT; }
${reciprocal ? ' if (inverseId == other) { inverseNext++; }\n' : ''} if (other < ROWS && (originalSlot != NO_SLOT || other != index)) { ${onItem} }
}`;
    nodes.push(
      ...createCSRStages<Parameters>(graph, {
        id: `${id}-edges`,
        operation: 'GPUIslandAttachment',
        rows,
        targetOffsets: props.output.offsets,
        capacity,
        overflow: props.output.status.overflow,
        initializesOverflow: true,
        totalNeighbors: props.output.status.requiredCount,
        bindings: [
          binding('offsets', props.weights.offsets),
          binding('neighbors', props.weights.neighbors),
          binding('attachment', props.output.attachment),
          ...(inverse
            ? [
                binding('inverseOffsets', inverse.offsets),
                binding('inverseIslands', inverse.islands)
              ]
            : [])
        ],
        visit,
        attributes: [
          {name: 'neighbors', output: props.output.neighbors, type: 'u32', value: 'other'},
          {
            name: 'weights',
            output: props.output.weights,
            type: 'f32',
            value: 'getOriginalWeight(originalSlot)',
            bindings: [binding('originalWeights', props.weights.weights, 'read', 'f32')],
            declarations: `fn getOriginalWeight(slot: u32) -> f32 {
 if (slot == NO_SLOT) { return 1.0; }
 return originalWeights[originalWeightsOffset + slot];
}`
          },
          {
            name: 'added',
            output: props.output.added,
            type: 'u32',
            value: 'select(0u, 1u, originalSlot == NO_SLOT)'
          }
        ]
      })
    );
    return nodes;
  }
}

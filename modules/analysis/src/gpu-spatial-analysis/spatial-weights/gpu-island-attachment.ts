// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  getViewBindingRange,
  doGraphDataViewsOverlap,
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
  validateMaximumCount,
  validateGPUStorageBindingByteLength,
  validateCommandNodeBudget,
  getGPUGridIndexMaximumCellCount,
  validateRequirement,
  validateSinglePackedView,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph
} from '../../analysis-core/kernels/gpu-contributor-utils';
import type {GPUUncertaintyStatusPort} from '../../analysis-core/contracts/status';
import {GPUNeighborSearch} from '../neighbor-search/gpu-neighbor-search';
import {createNeighborSearchDerivedParameterNodes} from '../neighbor-search/neighbor-search-derived-parameters';
import {
  ISLAND_ATTACHMENT_NEAREST_DECLARATIONS,
  createIslandAttachmentFallbackNodes,
  getIslandAttachmentFallbackBatchCount
} from './island-attachment-fallback';
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

import {validateSpatialStatisticsBindings} from './spatial-statistics-bindings';
import {getCoreDefaultBuffer} from '../../analysis-core/contracts/network-topology';

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
 * (see {@link GPUIslandAttachment}) uses a bounded cooperative reduction over representatives.
 */
const ISLAND_ATTACHMENT_CANDIDATE_COUNT = 16;

/** uint32 CSR offsets and the missing-row sentinel reserve the final row-count word. */
const ISLAND_ATTACHMENT_MAXIMUM_ROW_COUNT = 0xfffffffe;
/** CSR counts and candidate slot addressing must fit uint32 without wrapping their scans. */
const ISLAND_ATTACHMENT_MAXIMUM_SLOT_COUNT = 0xffffffff;
/** Retains the old reference scan coverage; large inputs must use the indexed route. */
const ISLAND_ATTACHMENT_SCAN_MAXIMUM_ROW_COUNT = 65535;
/**
 * Conservative allowance for published grid builds, compaction, scans and radix sort: at most
 * 32 radix bits and logarithmic scan levels in the uint32 row space. Node specs check the bound.
 */
const ISLAND_ATTACHMENT_PRIMITIVE_NODE_ALLOWANCE = 4096;

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
   * the grid above. The explicit scan retains its reference limit of 65,535 rows.
   */
  nearestSearch?: GPUIslandAttachmentNearestSearch;
  output: GPUSpatialWeightsOutput & {
    /**
     * `uncertainCount` is optional: the exact number of islands whose nearest representative the
     * grid's candidate list could not prove, so that island used a bounded cooperative reduction
     * (see {@link GPUIslandAttachment}). Always 0 for `nearestSearch: 'scan'`.
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
 * With `nearestSearch: 'scan'` (at most 65,535 rows) each island scans every representative. With
 * `'grid'` (and `'auto'` above {@link GPU_ISLAND_ATTACHMENT_SCAN_MAXIMUM_ROW_COUNT} rows) a
 * `GPUNeighborSearch` k-nearest cross join over a grid lists the 16 nearest representatives of each
 * island by float32 `dx^2 + dy^2`, ties to the lowest row; the island's own row is skipped, and the
 * kernel then chooses among them with the scan's comparison, so ties still go to the lowest row.
 * The list proves the choice when it holds every finite representative, or when its farthest entry
 * is farther than the chosen one by more than a relative 1e-5 (the two rankings can differ only by
 * float32 rounding). Otherwise, for example when 16 or more representatives are tied or
 * coincident at the nearest distance, or when coordinates are so large that the bounds or squared
 * distances overflow, a dedicated pass compacts the unresolved islands and gives each one a
 * 256-lane workgroup. Representatives are processed in batches of at most 65,536; each lane
 * tests at most 256 per dispatch and a tree reduction retains the best across batches. Query
 * dispatches are bounded by the device's 3D dispatch layout and uint32 invocation space. GPU
 * cursors supply batch bases, so shaders and pipelines do not grow with the number of batches.
 * The indexed route is bounded by uint32 CSR indexing and device storage bindings, including
 * 16 candidate slots per row. `output.status.uncertainCount`, when given, counts islands sent
 * to this pass (0 with `'scan'`). Degenerate inputs can require exhaustive work across batches.
 * `nearestSearch` and `reciprocal` are compile-time props; there are no per-frame scalar props.
 * The output rows then merge the original row, its attachment and the islands attached to it
 * (grouped by a stable sort), so rows that are not islands cost their own degree rather than a
 * scan of all rows.
 */
export class GPUIslandAttachment implements GPUCommandNodeProducer {
  readonly id: string;
  readonly outputWeights: GPUSpatialWeights;
  readonly estimatedCommandNodeCount: number;
  constructor(readonly props: GPUIslandAttachmentProps) {
    this.id = props.id ?? 'island-attachment';
    const nearestSearch = props.nearestSearch ?? 'auto';
    validateRequirement(
      this.id,
      ['auto', 'scan', 'grid'].includes(nearestSearch),
      "nearestSearch must be 'auto', 'scan' or 'grid'"
    );
    const rows = validateGPUSpatialWeights(this.id, props.weights);
    validateCapacity(this.id, 'rows', rows, 1, ISLAND_ATTACHMENT_MAXIMUM_ROW_COUNT);
    if (nearestSearch === 'scan') {
      validateMaximumCount(
        this.id,
        'rows for nearestSearch scan',
        rows,
        ISLAND_ATTACHMENT_SCAN_MAXIMUM_ROW_COUNT
      );
    }
    const useGrid =
      nearestSearch === 'grid' ||
      (nearestSearch === 'auto' && rows > GPU_ISLAND_ATTACHMENT_SCAN_MAXIMUM_ROW_COUNT);
    if (useGrid) {
      validateMaximumCount(
        this.id,
        'candidate slots (rows * 16)',
        rows * ISLAND_ATTACHMENT_CANDIDATE_COUNT,
        ISLAND_ATTACHMENT_MAXIMUM_SLOT_COUNT
      );
    }
    // Every island adds at most one outward edge and one reciprocal edge. This upper bound
    // keeps all CSR prefix sums exact and monotone even when caller output capacity is smaller.
    validateMaximumCount(
      this.id,
      'maximum output edge count',
      props.weights.neighbors.length + rows * (props.reciprocal !== false ? 2 : 1),
      ISLAND_ATTACHMENT_MAXIMUM_SLOT_COUNT
    );
    this.estimatedCommandNodeCount =
      ISLAND_ATTACHMENT_PRIMITIVE_NODE_ALLOWANCE +
      (useGrid ? 3 * getIslandAttachmentFallbackBatchCount(rows) : 0);
    validateCommandNodeBudget(this.id, this.estimatedCommandNodeCount);
    if (
      validateGPUSpatialWeightsOutput(this.id, props.output) !== rows ||
      props.positions.length !== rows ||
      props.output.attachment.length !== rows ||
      props.output.added.length !== props.output.neighbors.length
    )
      throw new Error(`${this.id}: shape mismatch`);
    if (props.output.distances) throw new Error(`${this.id}: distances output is not supported`);
    validateSinglePackedView(props.positions, ['float32x2'], `${this.id} positions`);
    validateSinglePackedView(props.output.attachment, ['uint32'], `${this.id} output.attachment`);
    validateSinglePackedView(props.output.added, ['uint32'], `${this.id} output.added`);
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
    const outputs = [
      ...getGPUSpatialWeightsViews(props.output),
      ...getGPUSpatialWeightsStatusViews(props.output),
      props.output.attachment,
      props.output.added,
      props.output.status.uncertainCount
    ].filter((view): view is GraphDataView<'uint32'> | GraphDataView<'float32'> => Boolean(view));
    if (props.output.status.uncertainCount) {
      validateSinglePackedView(
        props.output.status.uncertainCount,
        ['uint32'],
        `${this.id} output.status.uncertainCount`
      );
      validateCapacity(
        this.id,
        'output.status.uncertainCount length',
        props.output.status.uncertainCount.length,
        1,
        1
      );
    }
    for (let first = 0; first < outputs.length; first++) {
      for (let second = first + 1; second < outputs.length; second++) {
        validateRequirement(
          this.id,
          !doGraphDataViewsOverlap(outputs[first], outputs[second]),
          'output writable byte ranges must not overlap'
        );
      }
    }
    // CSR offsets/status share one write pass. Emit passes read offsets and attachments while
    // writing one attribute. WebGPU forbids mixed storage access to one physical buffer even
    // for disjoint binding ranges; co-bound writable ranges only need aligned separation.
    const passBindings: readonly (readonly (readonly [GraphDataView | undefined, boolean])[])[] = [
      [
        [props.output.offsets, true],
        [props.output.status.overflow, true],
        [props.output.status.requiredCount, true]
      ],
      ...[props.output.neighbors, props.output.weights, props.output.added].map(
        view =>
          [
            [props.output.offsets, false],
            [props.output.attachment, false],
            [view, true]
          ] as const
      )
    ];
    for (const bindings of passBindings) {
      for (let first = 0; first < bindings.length; first++) {
        for (let second = first + 1; second < bindings.length; second++) {
          const [firstView, firstWritable] = bindings[first];
          const [secondView, secondWritable] = bindings[second];
          if (!firstView || !secondView || !(firstWritable || secondWritable)) continue;
          const firstBuffer = firstView.buffer.defaultBuffer
            ? getCoreDefaultBuffer(firstView.buffer.defaultBuffer)
            : firstView.buffer;
          const secondBuffer = secondView.buffer.defaultBuffer
            ? getCoreDefaultBuffer(secondView.buffer.defaultBuffer)
            : secondView.buffer;
          if (firstBuffer !== secondBuffer) continue;
          validateRequirement(
            this.id,
            firstWritable === secondWritable,
            'output read-only and writable bindings must use separate physical buffers'
          );
          validateRequirement(
            this.id,
            firstView.buffer === secondView.buffer,
            'output views share a physical buffer: import it once and use offset views'
          );
          const firstRange = getViewBindingRange(firstView);
          const secondRange = getViewBindingRange(secondView);
          validateRequirement(
            this.id,
            firstRange.offset + firstRange.size <= secondRange.offset ||
              secondRange.offset + secondRange.size <= firstRange.offset,
            'output aligned binding ranges used together must not overlap'
          );
        }
      }
    }
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
    validateSpatialStatisticsBindings(graph, id, props);
    const rows = props.positions.length;
    const capacity = props.output.neighbors.length;
    // Check every aligned caller binding and all derived scratch before declaring graph resources.
    const views: readonly (readonly [string, GraphDataView | undefined])[] = [
      ['weights.offsets', props.weights.offsets],
      ['weights.neighbors', props.weights.neighbors],
      ['weights.weights', props.weights.weights],
      ['weights.distances', props.weights.distances],
      ['positions', props.positions],
      ['output.offsets', props.output.offsets],
      ['output.neighbors', props.output.neighbors],
      ['output.weights', props.output.weights],
      ['output.distances', props.output.distances],
      ['output.attachment', props.output.attachment],
      ['output.added', props.output.added],
      ['output.status.overflow', props.output.status.overflow],
      ['output.status.requiredCount', props.output.status.requiredCount],
      ['output.status.candidateOverflow', props.output.status.candidateOverflow],
      ['output.status.uncertainCount', props.output.status.uncertainCount]
    ];
    for (const [name, view] of views) {
      if (view)
        validateGPUStorageBindingByteLength(
          graph.device,
          id,
          name,
          getViewBindingRange(view).size,
          'reduce rows or output capacity'
        );
    }
    const nearestSearch = props.nearestSearch ?? 'auto';
    const useGrid =
      nearestSearch === 'grid' ||
      (nearestSearch === 'auto' && rows > GPU_ISLAND_ATTACHMENT_SCAN_MAXIMUM_ROW_COUNT);
    const scratch = [
      ['CSR counts and starts', rows * 4],
      ['CSR scan total', 4]
    ] as [string, number][];
    if (props.reciprocal !== false) {
      scratch.push(['inverse offsets', (rows + 1) * 4], ['inverse sort rows', rows * 4]);
      if (rows > 256) scratch.push(['inverse radix histograms', Math.ceil(rows / 256) * 16 * 4]);
    }
    if (useGrid) {
      const columns = Math.max(1, Math.ceil(Math.sqrt(rows / 2)));
      validateMaximumCount(
        id,
        'gridSize cells',
        columns * columns,
        getGPUGridIndexMaximumCellCount(graph.device)
      );
      scratch.push(
        ['candidate lists and outputs', rows * ISLAND_ATTACHMENT_CANDIDATE_COUNT * 4],
        ['neighbor records', rows * 3 * 4],
        ['neighbor grid positions', rows * 2 * 4],
        ['neighbor grid offsets', (columns * columns + 1) * 4],
        ['neighbor bounds partials', Math.min(256, Math.ceil(rows / 4096)) * 16],
        ['fallback cursor', 20],
        ['fallback dispatch', 12],
        ['neighbor parameters', 12 * 4]
      );
    }
    for (const [name, byteLength] of scratch) {
      validateGPUStorageBindingByteLength(graph.device, id, name, byteLength, 'reduce rows');
    }
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
    // Halving before subtraction prevents overflow for opposite finite coordinates. Compare both
    // distances at their shared scale so squared norms neither overflow nor collapse to zero.
    // Keep the scale multiplier normal even on devices that flush subnormal factors.
    const nearestDeclarations = ISLAND_ATTACHMENT_NEAREST_DECLARATIONS;
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
      const unresolved = createTransientView(graph, `${id}-unresolved`, 'uint32', rows);
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
        node(
          'nearest',
          rows,
          [
            binding('positions', props.positions, 'read', 'f32'),
            binding('offsets', props.weights.offsets),
            binding('candidateOffsets', candidateOffsets),
            binding('candidateNeighbors', candidateNeighbors),
            binding('candidateDistances', candidateDistances, 'read', 'f32'),
            binding('unresolved', unresolved, 'read_write'),
            binding('attachment', props.output.attachment, 'read_write')
          ],
          `unresolved[unresolvedOffset + index] = 0u;
var nearest = NO_ATTACHMENT; var bestDifference = vec2<f32>(0.0);
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
 // farthest entry is clearly farther than the chosen one; otherwise use the cooperative pass.
 var proven = found > 0u && found < ${ISLAND_ATTACHMENT_CANDIDATE_COUNT}u;
 if (found == ${ISLAND_ATTACHMENT_CANDIDATE_COUNT}u && nearest != NO_ATTACHMENT) {
  let chosenDistance = 2.0 * length(bestDifference);
  proven = isFiniteBits(chosenDistance) && isFiniteBits(farthest) && farthest > chosenDistance * ${ISLAND_ATTACHMENT_PROOF_MARGIN};
 }
 if (!proven) {
  unresolved[unresolvedOffset + index] = 1u;
  nearest = NO_ATTACHMENT;
 }
}
attachment[attachmentOffset + index] = nearest;`,
          nearestDeclarations
        ),
        ...createIslandAttachmentFallbackNodes(graph, {
          id,
          positions: props.positions,
          unresolved,
          attachment: props.output.attachment,
          uncertainCount,
          nearestDeclarations
        })
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

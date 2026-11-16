// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  createWGSLKernelNode,
  type WGSLKernelBinding
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  createSpatialJoinCandidateQueryRange,
  type SpatialJoinCandidatePage
} from './spatial-join-candidate-pages';
import {createGraphChunkGroupKernelNode} from '../../analysis-core/kernels/graph-chunk-groups';
import {createSaturatedCountScanNodes} from '../../analysis-core/kernels/scan-total';
import {getCandidateTruncationNodes, SPATIAL_JOIN_WGSL_HELPERS} from './spatial-join-passes';
import {NODE_OVERLAPS_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {GEODESIC_WGSL} from '../geometry-measures/geodesic-wgsl';
import {SCALED_QUOTIENT_WGSL} from '../../analysis-core/kernels/scaled-quotient-wgsl';

const OPERATION = 'GPUSpatialJoinCandidates';

/** BVH facts the candidate probe needs. @internal */
export type SpatialJoinCandidateTree = {
  /** Node minima of the right-hand BVH. */
  nodeMinima: GraphDataView<'float32x2'>;
  /** Node maxima of the right-hand BVH. */
  nodeMaxima: GraphDataView<'float32x2'>;
  /** Leaf slot to right feature row. */
  leafIds: GraphDataView<'uint32'>;
  /** Number of internal nodes. */
  internalNodeCount: number;
  /** Whether leaf traversal needs restoring to source-row order. */
  spatialSort?: boolean;
};

/** Formats a float as a WGSL literal. @internal */
export function formatSpatialJoinFloat(value: number): string {
  const text = String(Math.fround(value));
  return /[.eE]/.test(text) ? text : `${text}.0`;
}

/**
 * Returns the shared bounding-box candidate stage: a counted and scanned BVH probe from every left
 * box (expanded by `margin`), writing `(left, right)` rows ordered by left row and then right row.
 *
 * `state[0]` receives the unclamped candidate total. Slots past the total keep whatever the caller
 * cleared them to; slots past `candidateCapacity` are dropped. Nodes: count, scan, total, write.
 *
 * @internal
 */
export function getSpatialJoinCandidateNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    leftCount: number;
    rightCount: number;
    leftMinima: GraphDataView<'float32x2'>;
    leftMaxima: GraphDataView<'float32x2'>;
    tree: SpatialJoinCandidateTree;
    candidateCapacity: number;
    page?: SpatialJoinCandidatePage;
    /**
     * Box expansion in coordinate units: a compile-time number, or a one-row float32 view read at
     * every encoding (a negative, NaN or infinite value counts as 0).
     */
    margin: number | GraphDataView<'float32'>;
    /**
     * When set, boxes hold longitude/latitude degrees and `margin` is a great-circle distance in
     * meters on a sphere of this radius. Each left box is then expanded by the angular radius in
     * latitude and by the widest longitude reach of that radius at the box's far latitude (the
     * whole longitude range when the expansion reaches a pole), and a box that crosses the
     * antimeridian also probes its copy shifted by 360 degrees. The expansion is conservative by
     * 0.1% plus 1e-5 degrees.
     */
    sphereRadius?: number;
    state: GraphDataView<'uint32'>;
    candidatePairs: GraphDataView<'uint32x2'>;
    /**
     * Optional per-left-row flag, 1 for the rows that lost candidates to `candidateCapacity`: the
     * row that crosses the capacity and every later row with candidates (see
     * {@link getCandidateTruncationNodes}).
     */
    candidateTruncated?: GraphDataView<'uint32'>;
    /** Optional one-row flag, 1 when the candidate total exceeded `candidateCapacity`. */
    candidateOverflow?: GraphDataView<'uint32'>;
    /**
     * Optional one-row BVH overflow flag of `tree`. When it is nonzero every left row is flagged
     * in `candidateTruncated` and `candidateOverflow` is 1.
     */
    bvhOverflow?: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters>[] {
  const {id, leftCount, rightCount, candidateCapacity, tree, state, candidatePairs} = props;
  if (leftCount < 1) {
    // The generated WGSL computes leftCount - 1u; an empty left side must be handled by the caller.
    throw new Error(`${id} requires at least one left feature`);
  }
  const nodes: GPUCommandNode<Parameters>[] = [];
  const leftCounts =
    props.page?.counts ?? createTransientView(graph, `${id}-left-counts`, 'uint32', leftCount);
  const leftOffsets =
    props.page?.offsets ?? createTransientView(graph, `${id}-left-offsets`, 'uint32', leftCount);
  const marginView = typeof props.margin === 'number' ? undefined : props.margin;
  const {sphereRadius} = props;
  const probeBindings = (extra: WGSLKernelBinding[]): WGSLKernelBinding[] => [
    {name: 'leftMinima', view: props.leftMinima, type: 'f32', access: 'read'},
    {name: 'leftMaxima', view: props.leftMaxima, type: 'f32', access: 'read'},
    {name: 'nodeMinima', view: tree.nodeMinima, type: 'f32', access: 'read'},
    {name: 'nodeMaxima', view: tree.nodeMaxima, type: 'f32', access: 'read'},
    {name: 'leafIds', view: tree.leafIds, type: 'u32', access: 'read'},
    ...(marginView
      ? [{name: 'marginRow', view: marginView, type: 'f32', access: 'read'} as const]
      : []),
    ...extra
  ];
  const marginWGSL = marginView
    ? `fn getMargin() -> f32 {
  let margin = marginRow[marginRowOffset];
  return select(0.0, margin, margin >= 0.0 && margin <= 3.4028234e38);
}`
    : `fn getMargin() -> f32 { return ${formatSpatialJoinFloat(Math.fround(props.margin as number))}; }`;
  const probeDeclarations = `${SPATIAL_JOIN_WGSL_HELPERS}
const INTERNAL_NODE_COUNT: u32 = ${tree.internalNodeCount}u;
const RIGHT_COUNT: u32 = ${rightCount}u;
const CANDIDATE_CAPACITY: u32 = ${candidateCapacity}u;
${marginWGSL}
${NODE_OVERLAPS_WGSL}
${
  sphereRadius === undefined
    ? ''
    : `${GEODESIC_WGSL}
${SCALED_QUOTIENT_WGSL}
const SPHERE_RADIUS: f32 = ${formatSpatialJoinFloat(sphereRadius)};
const WRAP_SHIFT: vec2f = vec2f(360.0, 0.0);`
}`;
  const expansion =
    sphereRadius === undefined
      ? `let queryMinimum = boxMinimum - vec2f(margin);
    let queryMaximum = boxMaximum + vec2f(margin);`
      : `var queryMinimum = boxMinimum;
    var queryMaximum = boxMaximum;
    var wrapDown = false;
    var wrapUp = false;
    if (margin > 0.0) {
      let angle = getScaledQuotient(margin, SPHERE_RADIUS, 1.0, 0) * 1.001;
      let deltaLatitude = angle * GEODESIC_RADIANS_TO_DEGREES + 1.0e-5;
      queryMinimum.y = boxMinimum.y - deltaLatitude;
      queryMaximum.y = boxMaximum.y + deltaLatitude;
      let farLatitude = max(abs(queryMinimum.y), abs(queryMaximum.y));
      var ratio = 2.0;
      if (farLatitude < 90.0) { ratio = sin(0.5 * angle) / geodesicCosLatitude(farLatitude); }
      if (ratio < 1.0) {
        let deltaLongitude = 2.0 * asin(ratio) * GEODESIC_RADIANS_TO_DEGREES + 1.0e-5;
        queryMinimum.x = boxMinimum.x - deltaLongitude;
        queryMaximum.x = boxMaximum.x + deltaLongitude;
        wrapDown = queryMaximum.x > 180.0;
        wrapUp = queryMinimum.x < -180.0;
      } else {
        // The cap reaches a pole (or is wider than a hemisphere in longitude): every longitude.
        queryMinimum.x = -FLOAT32_MAXIMUM;
        queryMaximum.x = FLOAT32_MAXIMUM;
      }
    }`;
  const overlaps =
    sphereRadius === undefined
      ? 'nodeOverlaps(node, queryMinimum, queryMaximum)'
      : `(nodeOverlaps(node, queryMinimum, queryMaximum) ||
        (wrapDown && nodeOverlaps(node, queryMinimum - WRAP_SHIFT, queryMaximum - WRAP_SHIFT)) ||
        (wrapUp && nodeOverlaps(node, queryMinimum + WRAP_SHIFT, queryMaximum + WRAP_SHIFT)))`;
  const probeBody = (
    leaf: string,
    prologue: string,
    epilogue: string
  ) => `let boxMinimum = vec2f(leftMinima[leftMinimaOffset + index * 2u], leftMinima[leftMinimaOffset + index * 2u + 1u]);
  let boxMaximum = vec2f(leftMaxima[leftMaximaOffset + index * 2u], leftMaxima[leftMaximaOffset + index * 2u + 1u]);
  ${prologue}
  if (boxMinimum.x <= boxMaximum.x && boxMinimum.y <= boxMaximum.y) {
    let margin = getMargin();
    ${expansion}
    var node = 0u;
    loop {
      if (${overlaps}) {
        if (node < INTERNAL_NODE_COUNT) {
          node = node * 2u + 1u;
          continue;
        }
        let rightRow = leafIds[leafIdsOffset + node - INTERNAL_NODE_COUNT];
        if (rightRow < RIGHT_COUNT) {
          ${leaf}
        }
      }
      // Climb while the node is a right child, then step to the right sibling.
      loop {
        if (node == 0u || (node & 1u) == 1u) { break; }
        node = (node - 1u) / 2u;
      }
      if (node == 0u) { break; }
      node = node + 1u;
    }
  }
  ${epilogue}`;
  if (!props.page || props.page.initialize) {
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-count-candidates`,
        operation: OPERATION,
        variant: 'count-candidates',
        bindings: probeBindings([
          {name: 'leftCounts', view: leftCounts, type: 'u32', access: 'read_write'}
        ]),
        invocationCount: leftCount,
        declarations: probeDeclarations,
        body: probeBody(
          'found = found + 1u;',
          'var found = 0u;',
          'leftCounts[leftCountsOffset + index] = found;'
        )
      })
    );
    nodes.push(
      ...createSaturatedCountScanNodes(graph, {
        id: `${id}-scan-candidates`,
        operation: OPERATION,
        counts: leftCounts,
        offsets: leftOffsets,
        rowCount: leftCount,
        total: state
      })
    );
  }
  // One invocation owns each left row's candidate segment. A bounded max heap retains the
  // smallest source IDs even when capacity cuts through that row; sorting only the first visited
  // leaves would return a different overflow prefix from the unsorted tree. No extra scratch or
  // storage bindings are needed. The ordinary source-ordered traversal keeps its fast path.
  const pageRankDeclarations =
    props.page && tree.spatialSort
      ? `
fn countRightsThrough(index: u32, maximumRight: u32) -> u32 {
${probeBody('if (rightRow <= maximumRight) { found++; }', 'var found = 0u;', 'return found;')}
}
fn findFirstPageRight(index: u32, rank: u32) -> u32 {
  // The first retained candidate needs no rank selection, including every single-page query.
  if (rank == 0u) { return 0u; }
  var lower = 0u;
  var upper = RIGHT_COUNT - 1u;
  // At most 32 bisections of uint32 source IDs; no candidate-prefix allocation is needed.
  for (var step = 0u; step < 32u; step++) {
    if (lower >= upper) { break; }
    let middle = lower + (upper - lower) / 2u;
    if (countRightsThrough(index, middle) > rank) { upper = middle; }
    else { lower = middle + 1u; }
  }
  return lower;
}`
      : '';
  const sortedDeclarations = tree.spatialSort
    ? `
fn readCandidateRight(start: u32, slot: u32) -> u32 {
  return candidatePairs[candidatePairsOffset + (start + slot) * 2u + 1u];
}
fn writeCandidateRight(start: u32, slot: u32, value: u32) {
  candidatePairs[candidatePairsOffset + (start + slot) * 2u + 1u] = value;
}
fn siftCandidateDown(start: u32, size: u32, value: u32) {
  var parent = 0u;
  loop {
    var child = parent * 2u + 1u;
    if (child >= size) { break; }
    if (child + 1u < size && readCandidateRight(start, child + 1u) > readCandidateRight(start, child)) {
      child = child + 1u;
    }
    let childValue = readCandidateRight(start, child);
    if (childValue <= value) { break; }
    writeCandidateRight(start, parent, childValue);
    parent = child;
  }
  writeCandidateRight(start, parent, value);
}`
    : '';
  const body = probeBody(
    tree.spatialSort
      ? `${props.page ? 'if (rightRow >= minimumRight) {' : ''}
          if (found < slotEnd - slotStart) {
            var child = found;
            found = found + 1u;
            loop {
              if (child == 0u) { break; }
              let parent = (child - 1u) / 2u;
              let parentValue = readCandidateRight(slotStart, parent);
              if (parentValue >= rightRow) { break; }
              writeCandidateRight(slotStart, child, parentValue);
              child = parent;
            }
            writeCandidateRight(slotStart, child, rightRow);
          } else if (rightRow < readCandidateRight(slotStart, 0u)) {
            siftCandidateDown(slotStart, found, rightRow);
          }
          ${props.page ? '}' : ''}`
      : `${props.page ? 'if (found >= skipped) {' : ''}
          let slot = slotStart + found${props.page ? ' - skipped' : ''};
          candidatePairs[candidatePairsOffset + slot * 2u] = index;
          candidatePairs[candidatePairsOffset + slot * 2u + 1u] = rightRow;
          if (slot + 1u >= slotEnd) { break; }
          ${props.page ? '}' : ''}
          found = found + 1u;`,
    // Left features with no candidates, or whose slots all fall past the capacity, skip the
    // second traversal; source-ordered probes stop at their last retained candidate. The last
    // row's end comes from the capacity, keeping this pass within 8 storage bindings.
    props.page
      ? `var found = 0u;
  let globalStart = leftOffsets[leftOffsetsOffset + index];
  var globalEnd = CANDIDATE_CAPACITY;
  if (index + 1u < ${leftCount}u) { globalEnd = min(leftOffsets[leftOffsetsOffset + index + 1u], CANDIDATE_CAPACITY); }
  let pageFirst = getChunkParameter(36u);
  let pageEnd = pageFirst + getChunkParameter(37u);
  if (globalStart >= pageEnd || globalEnd <= pageFirst) { return; }
  let skipped = max(globalStart, pageFirst) - globalStart;
  let slotStart = max(globalStart, pageFirst) - pageFirst;
  let slotEnd = min(globalEnd, pageEnd) - pageFirst;
  ${tree.spatialSort ? 'let minimumRight = findFirstPageRight(index, skipped);' : ''}`
      : `var found = 0u;
  let slotStart = leftOffsets[leftOffsetsOffset + index];
  var slotEnd = CANDIDATE_CAPACITY;
  if (index + 1u < ${leftCount}u) {
    slotEnd = min(leftOffsets[leftOffsetsOffset + index + 1u], CANDIDATE_CAPACITY);
  }
  if (slotStart >= slotEnd) { return; }`,
    tree.spatialSort
      ? `var remaining = found;
  loop {
    if (remaining <= 1u) { break; }
    remaining = remaining - 1u;
    let largest = readCandidateRight(slotStart, 0u);
    let replacement = readCandidateRight(slotStart, remaining);
    writeCandidateRight(slotStart, remaining, largest);
    siftCandidateDown(slotStart, remaining, replacement);
  }
  for (var slot = 0u; slot < found; slot = slot + 1u) {
    candidatePairs[candidatePairsOffset + (slotStart + slot) * 2u] = index;
  }`
      : ''
  );
  const queryRange =
    props.page?.metadataRow === undefined
      ? undefined
      : createSpatialJoinCandidateQueryRange(graph, {
          id,
          offsets: leftOffsets,
          total: state,
          firstSlot: props.page.firstSlot,
          length: props.page.length,
          candidateCapacity,
          queryFirst: 0,
          queryLength: leftCount,
          queryCount: leftCount,
          metadataRow: props.page.metadataRow
        });
  if (queryRange) nodes.push(queryRange.updateNode);
  nodes.push(
    (props.page ? createGraphChunkGroupKernelNode<Parameters> : createWGSLKernelNode<Parameters>)(
      graph,
      {
        ...(props.page
          ? {
              parameterValues: [
                props.page.firstSlot,
                props.page.length,
                props.page.metadataRow ?? 0
              ]
            }
          : {}),
        ...(queryRange ? {condition: queryRange.condition} : {}),
        id: `${id}-write-candidates`,
        operation: OPERATION,
        variant: 'write-candidates',
        bindings: probeBindings([
          {name: 'leftOffsets', view: leftOffsets, type: 'u32', access: 'read'},
          {name: 'candidatePairs', view: candidatePairs, type: 'u32', access: 'read_write'}
        ]),
        invocationCount: leftCount,
        declarations:
          probeDeclarations +
          sortedDeclarations +
          pageRankDeclarations +
          (queryRange
            ? `
fn writeCandidateQuery(index: u32) { ${body} }`
            : ''),
        body: queryRange
          ? `let metadata = leftOffsetsOffset + getChunkParameter(38u);
  if (index >= leftOffsets[metadata + 1u]) { return; }
  writeCandidateQuery(leftOffsets[metadata] + index);`
          : body
      }
    )
  );
  if (!props.page || props.page.initialize)
    nodes.push(
      ...getCandidateTruncationNodes<Parameters>(graph, {
        id,
        operation: OPERATION,
        counts: leftCounts,
        offsets: leftOffsets,
        rowCount: leftCount,
        candidateCapacity,
        total: state,
        candidateTruncated: props.candidateTruncated,
        candidateOverflow: props.candidateOverflow,
        bvhOverflow: props.bvhOverflow
      })
    );
  return nodes;
}

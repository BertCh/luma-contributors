// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  resolveGeometryUint32StatusView,
  validateGeometryRetainedViews
} from '../geometry-edit/geometry-storage-validation';
import type {
  GPUCommandGraph,
  GPUCommandNode,
  GPUCommandNodeProducer,
  GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  resolveGPUSpatialContext,
  type GPUSpatialContext
} from '../../analysis-core/contracts/spatial-context';
import {
  validateSinglePackedView,
  validateCapacity,
  validateGraphOutputsDisjointFromInputs,
  validateFiniteNumber,
  validateRequirement,
  validateCommandNodeBudget,
  validateGPUStorageBindingByteLength
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  createWGSLKernelNode,
  getWGSLFloatLiteral
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {validateGraphChunkBindingRanges} from '../../analysis-core/kernels/graph-chunk-dispatch';
import {validateOutlineOutputPasses} from '../outline-geometry/validate-outline-output-passes';
import {
  createSegmentSnapCandidates,
  DEFAULT_SEGMENT_SNAP_CANDIDATE_CAPACITY,
  SEGMENT_SNAP_GRID_WIDTH
} from './segment-snap-candidates';
import {SCALED_QUOTIENT_WGSL} from '../../analysis-core/kernels/scaled-quotient-wgsl';
import {getReadPointWGSL} from '../shared/spatial-wgsl-point-helpers';

/** The inclusive tolerance is emitted as an f32 literal and must remain finite after rounding. */
const MAXIMUM_TOLERANCE = 3.4028234663852886e38;
/** Required insertion counts use uint32 and must not wrap, even when fixed output slots overflow. */
const MAXIMUM_REQUIRED_VERTEX_COUNT = 0xffffffff;
/** Two scalar coordinates per output row must be addressable with uint32 binding indices. */
const MAXIMUM_POSITION_COUNT = 0x7fffffff;
/** Four scalar status words per line must be addressable with uint32 binding indices. */
const MAXIMUM_LINE_COUNT = 0x3fffffff;
/** Candidate record indexing includes two header words and one global nonfinite flag. */
const MAXIMUM_CANDIDATE_WORD_COUNT = 0xffffffff;
/** Grid build, hierarchical scan, bounds reduction and publication have a fixed node budget. */
const MAXIMUM_GRID_COMMAND_NODE_COUNT = 32;
/** One workgroup per batch keeps dispatch/pipelines fixed while bounding reusable list scratch. */
const MAXIMUM_BATCH_LINE_COUNT = 256;
/** Exhaustive insertion includes cubic duplicate searches; bound its worst-case scalar work. */
export const MAXIMUM_SEGMENT_SNAP_EXHAUSTIVE_WORK = 16_777_216;

/** Shared with the CPU twin: every exhaustive route, including overflow, has the same work bound. */
export function canExhaustivelySnap(
  positionCount: number,
  referenceCount: number,
  lineCount = 1
): boolean {
  // Even empty lines scan references for nonfinite input on the explicit exhaustive route.
  return (
    (positionCount + lineCount) * (referenceCount + 1) ** 3 <= MAXIMUM_SEGMENT_SNAP_EXHAUSTIVE_WORK
  );
}

/** Per-line candidate diagnostics; each column has lineOffsets.length - 1 rows. */
export type GPUSegmentSnapStatus = {
  candidateOverflow?: GraphDataView<'uint32'>;
};

export type GPUSegmentSnapProps = {
  id?: string;
  spatialContext?: GPUSpatialContext;
  positions: GraphDataView<'float32x2'>;
  lineOffsets: GraphDataView<'uint32'>;
  referencePositions: GraphDataView<'float32x2'>;
  /** Default grid route reuses bounded candidate batches. Explicit brute-force is work limited.
   * Overflowing lists use exhaustive insertion only within MAXIMUM_SEGMENT_SNAP_EXHAUSTIVE_WORK;
   * otherwise the line has zero written/required vertices, capacity overflow=1 and validity bit 4.
   * Increase candidateCapacity or split the reference set before consuming incomplete geometry. */
  searchStructure?: 'grid' | 'brute-force';
  /** Reference IDs retained per line. Default 1024. Overflow is never silently truncated. */
  candidateCapacity?: number;
  /** Default bounded-brute-force: exact fallback within the named work limit, otherwise incomplete.
   * incomplete: report every list overflow without attempting exhaustive insertion. */
  candidateOverflowPolicy?: 'bounded-brute-force' | 'incomplete';
  /** Nonnegative inclusive distance in planar units. */
  tolerance: number;
  /** Fixed output slots per input line. */
  maximumVerticesPerLine: number;
  output: {
    positions: GraphDataView<'float32x2'>;
    /** Original segment's first vertex index for every emitted vertex. */
    sourceEdges: GraphDataView<'uint32'>;
    /** Per line [written vertices, required vertices, capacity overflow, nonfinite input]. */
    counts: GraphDataView<'uint32x4'>;
    /** Optional line validity bits: too few distinct vertices=1, nonfinite=2, incomplete=4. Self-crossing is valid for lines. */
    validity?: GraphDataView<'uint32'>;
    /** Per-line diagnostics, independent of the counts column's final-capacity overflow. */
    status?: GPUSegmentSnapStatus;
    /** @deprecated Use status.candidateOverflow. Both spellings must identify the same view. */
    candidateOverflow?: GPUSegmentSnapStatus['candidateOverflow'];
  };
};

/**
 * Snap endpoints to nearest reference vertices, then insert all reference vertices
 * within tolerance of each snapped segment in projection order. Distance ties use
 * reference row order; identical references insert only once. This intentionally
 * has no GEOS safety heuristic and does not certify topology preservation.
 * Output line i occupies slots i * maximumVerticesPerLine through its written
 * count. Required counts saturate at uint32 maximum; suffix vertices are dropped.
 */
export class GPUSegmentSnap implements GPUCommandNodeProducer {
  readonly id: string;
  readonly props: GPUSegmentSnapProps;
  readonly spatialContext: GPUSpatialContext;
  readonly estimatedCommandNodeCount: number;
  private readonly outputViews: {
    positions: GraphDataView<'float32x2'>;
    sourceEdges: GraphDataView<'uint32'>;
    counts: GraphDataView<'uint32x4'>;
    validity?: GraphDataView<'uint32'>;
    candidateOverflow?: GraphDataView<'uint32'>;
  };
  constructor(props: GPUSegmentSnapProps) {
    this.id = props.id ?? 'segment-snap';
    this.props = props;
    this.outputViews = {
      positions: props.output.positions,
      sourceEdges: props.output.sourceEdges,
      counts: props.output.counts,
      validity: props.output.validity,
      candidateOverflow: resolveGeometryUint32StatusView(
        this.id,
        'output.status.candidateOverflow',
        props.output.status?.candidateOverflow,
        props.output.candidateOverflow
      )
    };
    this.spatialContext = resolveGPUSpatialContext(this.id, props.spatialContext);
    validateCapacity(
      this.id,
      'maximumVerticesPerLine',
      props.maximumVerticesPerLine,
      1,
      MAXIMUM_POSITION_COUNT
    );
    this.estimatedCommandNodeCount =
      props.searchStructure === 'brute-force'
        ? 2
        : MAXIMUM_GRID_COMMAND_NODE_COUNT + 5 * (props.lineOffsets.length - 1);
    validateRequirement(
      this.id,
      props.searchStructure === undefined ||
        props.searchStructure === 'grid' ||
        props.searchStructure === 'brute-force',
      'searchStructure must be grid or brute-force'
    );
    validateRequirement(
      this.id,
      props.candidateOverflowPolicy === undefined ||
        props.candidateOverflowPolicy === 'bounded-brute-force' ||
        props.candidateOverflowPolicy === 'incomplete',
      'candidateOverflowPolicy must be bounded-brute-force or incomplete'
    );
    validateCapacity(
      this.id,
      'candidateCapacity',
      props.candidateCapacity ?? DEFAULT_SEGMENT_SNAP_CANDIDATE_CAPACITY,
      1,
      MAXIMUM_CANDIDATE_WORD_COUNT - 5
    );
    if (props.searchStructure === 'brute-force')
      validateCapacity(
        this.id,
        'positions.length * (referencePositions.length + 1)',
        props.positions.length * (props.referencePositions.length + 1),
        0,
        MAXIMUM_REQUIRED_VERTEX_COUNT
      );
    if (props.searchStructure === 'brute-force')
      validateRequirement(
        this.id,
        canExhaustivelySnap(
          props.positions.length,
          props.referencePositions.length,
          props.lineOffsets.length - 1
        ),
        `(positions.length + lineCount) * (referencePositions.length + 1)^3 must not exceed MAXIMUM_SEGMENT_SNAP_EXHAUSTIVE_WORK (${MAXIMUM_SEGMENT_SNAP_EXHAUSTIVE_WORK})`
      );
    validateFiniteNumber(this.id, 'tolerance', props.tolerance, 0, MAXIMUM_TOLERANCE);
    for (const view of [props.positions, props.referencePositions, props.output.positions]) {
      validateSinglePackedView(view, ['float32x2'], this.id);
    }
    for (const view of [props.lineOffsets, props.output.sourceEdges]) {
      validateSinglePackedView(view, ['uint32'], this.id);
    }
    validateSinglePackedView(props.output.counts, ['uint32x4'], this.id);
    if (props.output.validity) {
      validateSinglePackedView(props.output.validity, ['uint32'], this.id);
      if (props.output.validity.length !== props.lineOffsets.length - 1) {
        validateRequirement(
          this.id,
          false,
          'output.validity.length must equal lineOffsets.length - 1'
        );
      }
    }
    validateCapacity(
      this.id,
      'lineOffsets.length',
      props.lineOffsets.length,
      2,
      MAXIMUM_LINE_COUNT + 1
    );
    validateCapacity(
      this.id,
      'positions.length',
      props.positions.length,
      0,
      MAXIMUM_POSITION_COUNT
    );
    validateCapacity(
      this.id,
      'referencePositions.length',
      props.referencePositions.length,
      0,
      MAXIMUM_POSITION_COUNT
    );
    const lines = props.lineOffsets.length - 1;
    validateCapacity(
      this.id,
      'lineCount * maximumVerticesPerLine',
      lines * props.maximumVerticesPerLine,
      1,
      MAXIMUM_POSITION_COUNT
    );
    if (
      props.output.counts.length !== lines ||
      props.output.positions.length !== lines * props.maximumVerticesPerLine ||
      props.output.sourceEdges.length !== props.output.positions.length
    ) {
      validateRequirement(
        this.id,
        false,
        'output lengths must match lineCount and maximumVerticesPerLine'
      );
    }
    if (this.outputViews.candidateOverflow) {
      validateSinglePackedView(this.outputViews.candidateOverflow, ['uint32'], this.id);
      validateRequirement(
        this.id,
        this.outputViews.candidateOverflow.length === lines,
        'output.status.candidateOverflow.length must equal lineCount'
      );
    }
    validateOutlineOutputPasses(this.id, this.outputViews, [
      ['positions', 'sourceEdges', 'counts', 'validity'],
      ['candidateOverflow']
    ]);
    const inputs = [props.positions, props.lineOffsets, props.referencePositions];
    // The diagnostic is published after insertion, with only private candidate scratch bound.
    // Other output fields share the insertion dispatch with the caller's read-only inputs.
    const {candidateOverflow, ...insertionOutputs} = this.outputViews;
    validateGraphOutputsDisjointFromInputs(this.id, Object.values(insertionOutputs), inputs);
    validateGeometryRetainedViews(this.id, inputs, [candidateOverflow]);
  }
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {props, id} = this;
    validateGraphChunkBindingRanges(graph, id, {
      positions: props.positions,
      lineOffsets: props.lineOffsets,
      referencePositions: props.referencePositions,
      ...Object.fromEntries(
        Object.entries(this.outputViews).map(([name, view]) => [`output.${name}`, view])
      )
    });
    const candidateCapacity = props.candidateCapacity ?? DEFAULT_SEGMENT_SNAP_CANDIDATE_CAPACITY;
    const lineCount = props.lineOffsets.length - 1;
    const grid = props.searchStructure !== 'brute-force' && props.referencePositions.length > 0;
    const maximumWords = Math.min(
      MAXIMUM_CANDIDATE_WORD_COUNT,
      Math.floor(
        Math.min(
          graph.device.limits.maxStorageBufferBindingSize,
          graph.device.limits.maxBufferSize
        ) / 4
      )
    );
    const batchLineCapacity = Math.max(
      1,
      Math.min(
        MAXIMUM_BATCH_LINE_COUNT,
        lineCount,
        Math.floor((maximumWords - 3) / (candidateCapacity + 2))
      )
    );
    const candidateByteLength = (3 + batchLineCapacity * (candidateCapacity + 2)) * 4;
    validateCommandNodeBudget(
      id,
      grid ? MAXIMUM_GRID_COMMAND_NODE_COUNT + 5 * Math.ceil(lineCount / batchLineCapacity) : 2
    );
    if (grid) {
      // Includes the public grid builder's counts, cursors, per-row ranks/IDs and scan summaries,
      // plus the bounds reducer's (at most 256) vec4 partials. Every scratch view is aligned.
      for (const [name, byteLength] of Object.entries({
        candidates: candidateByteLength,
        normalizedReferences: Math.max(1, props.referencePositions.length) * 8,
        gridRowScratch: Math.max(1, props.referencePositions.length) * 4,
        gridCellsAndScan: (SEGMENT_SNAP_GRID_WIDTH ** 2 + 1) * 4,
        boundsPartials:
          Math.min(256, Math.max(1, Math.ceil(props.referencePositions.length / 4096))) * 16
      })) {
        validateGPUStorageBindingByteLength(
          graph.device,
          id,
          name,
          byteLength,
          'reduce candidateCapacity or split the input'
        );
        validateCapacity(
          id,
          `${name} byteLength`,
          byteLength,
          1,
          graph.device.limits.maxBufferSize
        );
      }
    }
    const built = grid
      ? createSegmentSnapCandidates(graph, {...props, id, candidateCapacity, batchLineCapacity})
      : undefined;
    const candidates = built?.candidates;
    const nodes = built?.nodes ?? [];
    for (
      let firstLine = 0;
      firstLine < lineCount;
      firstLine += grid ? batchLineCapacity : lineCount
    ) {
      if (built)
        nodes.push(
          ...built.createBatchNodes(firstLine, Math.min(batchLineCapacity, lineCount - firstLine))
        );
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-insert-${firstLine}`,
          operation: 'GPUSegmentSnap',
          variant: 'sorted-insertions',
          invocationCount: grid ? MAXIMUM_BATCH_LINE_COUNT : lineCount,
          workgroupSize: grid ? MAXIMUM_BATCH_LINE_COUNT : undefined,
          guardIndex: !grid,
          bindings: [
            {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
            {name: 'lineOffsets', view: props.lineOffsets, type: 'u32', access: 'read'},
            {name: 'references', view: props.referencePositions, type: 'f32', access: 'read'},
            ...(candidates
              ? [
                  {
                    name: 'candidates',
                    view: candidates,
                    type: 'u32' as const,
                    access: 'read' as const
                  }
                ]
              : []),
            {name: 'outPositions', view: props.output.positions, type: 'f32', access: 'read_write'},
            {
              name: 'sourceEdges',
              view: props.output.sourceEdges,
              type: 'u32',
              access: 'read_write'
            },
            {name: 'counts', view: props.output.counts, type: 'u32', access: 'read_write'},
            ...(props.output.validity
              ? [
                  {
                    name: 'validity',
                    view: props.output.validity,
                    type: 'u32' as const,
                    access: 'read_write' as const
                  }
                ]
              : [])
          ],
          declarations: `const CAPACITY: u32 = ${props.maximumVerticesPerLine}u;
const REFERENCE_COUNT: u32 = ${props.referencePositions.length}u;
const TOLERANCE: f32 = ${getWGSLFloatLiteral(props.tolerance)};
${getReadPointWGSL('point', 'positions')}
${SCALED_QUOTIENT_WGSL}
const EXHAUSTIVE_ALLOWED: bool = ${props.candidateOverflowPolicy !== 'incomplete' && canExhaustivelySnap(props.positions.length, props.referencePositions.length, props.lineOffsets.length - 1)};
${
  candidates
    ? `const CANDIDATE_CAPACITY: u32 = ${candidateCapacity}u;
fn requiresDuplicateSearch(line: u32) -> bool { return candidates[candidatesOffset + 4u + (line - candidates[candidatesOffset + 1u]) * (CANDIDATE_CAPACITY + 2u)] != 0u; }
fn candidateCount(line: u32) -> u32 { let base = candidatesOffset + 3u + (line - candidates[candidatesOffset + 1u]) * (CANDIDATE_CAPACITY + 2u); return select(candidates[base], REFERENCE_COUNT, candidates[base + 1u] != 0u); }
fn candidateRow(line: u32, slot: u32) -> u32 { let base = candidatesOffset + 3u + (line - candidates[candidatesOffset + 1u]) * (CANDIDATE_CAPACITY + 2u); if (candidates[base + 1u] != 0u) { return slot; } return candidates[base + 2u + slot]; }`
    : 'fn requiresDuplicateSearch(line: u32) -> bool { return true; }\nfn candidateCount(line: u32) -> u32 { return REFERENCE_COUNT; }\nfn candidateRow(line: u32, slot: u32) -> u32 { return slot; }'
}
fn scaledProductQuotient(value: f32, scale: f32, first: f32, second: f32) -> f32 {
  if (value == 0.0) { return value; }
  let scaling = frexp(scale);
  return getScaledQuotient(value * scaling.fract, first, second, scaling.exp);
}
fn reference(vertex: u32) -> vec2f { return vec2f(references[referencesOffset + vertex * 2u], references[referencesOffset + vertex * 2u + 1u]); }
fn finitePoint(value: vec2f) -> bool { return all((bitcast<vec2u>(value) & vec2u(0x7f800000u)) != vec2u(0x7f800000u)); }
fn pointDistance(first: vec2f, second: vec2f) -> f32 {
  let difference = first - second;
  let magnitude = max(abs(difference.x), abs(difference.y));
  if (magnitude == 0.0) { return 0.0; }
  // A nonfinite magnitude is larger than every finite tolerance. Avoid inf / inf below.
  if ((bitcast<u32>(magnitude) & 0x7f800000u) == 0x7f800000u) { return magnitude; }
  return magnitude * length(vec2f(getScaledQuotient(difference.x, magnitude, 1.0, 0), getScaledQuotient(difference.y, magnitude, 1.0, 0)));
}
fn snapPoint(line: u32, value: vec2f) -> vec2f {
  var result = value; var minimum = TOLERANCE; var found = false; var winner = 0xffffffffu;
  for (var slot = 0u; slot < candidateCount(line); slot++) {
    let candidate = candidateRow(line, slot);
    let distance = pointDistance(reference(candidate), value);
    if (distance <= TOLERANCE && (!found || distance < minimum || (distance == minimum && candidate < winner))) { result = reference(candidate); minimum = distance; found = true; winner = candidate; }
  }
  return result;
}
fn writeVertex(line: u32, slot: u32, value: vec2f, edge: u32) {
  if (slot >= CAPACITY) { return; }
  let destination = line * CAPACITY + slot;
  outPositions[outPositionsOffset + destination * 2u] = value.x;
  outPositions[outPositionsOffset + destination * 2u + 1u] = value.y;
  sourceEdges[sourceEdgesOffset + destination] = edge;
}`,
          body: `${candidates ? 'if (index >= candidates[candidatesOffset + 2u]) { return; }' : ''}
let line = index ${candidates ? '+ candidates[candidatesOffset + 1u]' : ''};
let incomplete = ${candidates ? '!EXHAUSTIVE_ALLOWED && requiresDuplicateSearch(line)' : 'false'};
let first = min(lineOffsets[lineOffsetsOffset + line], ${props.positions.length}u);
let last = min(lineOffsets[lineOffsetsOffset + line + 1u], ${props.positions.length}u);
var required = 0u;
var invalid = false;
${candidates ? 'invalid = candidates[candidatesOffset] != 0u;' : 'for (var candidate = 0u; candidate < REFERENCE_COUNT; candidate++) { invalid = invalid || !finitePoint(reference(candidate)); }'}
for (var vertex = first; vertex < last; vertex++) { invalid = invalid || !finitePoint(point(vertex)); }
if (!invalid && !incomplete && last > first) {
  var start = snapPoint(line, point(first));
  writeVertex(line, required, start, first); required += min(1u, 0xffffffffu - required);
  for (var vertex = first; vertex + 1u < last; vertex++) {
    let end = snapPoint(line, point(vertex + 1u));
    let halfEdge = end * 0.5 - start * 0.5;
    let magnitude = max(abs(halfEdge.x), abs(halfEdge.y));
    let divisor = max(magnitude, 1.175494351e-38);
    let edge = vec2f(getScaledQuotient(halfEdge.x, divisor, 1.0, 0), getScaledQuotient(halfEdge.y, divisor, 1.0, 0));
    let denominator = dot(edge, edge);
    var previousFraction = 0.0; var previousReference = 0xffffffffu;
    var previousPoint = start;
    if (denominator > 0.0) {
      for (var inserted = 0u; inserted < candidateCount(line); inserted++) {
        var winner = 0xffffffffu; var winnerFraction = 2.0;
        for (var slot = 0u; slot < candidateCount(line); slot++) {
          let candidate = candidateRow(line, slot);
          let current = reference(candidate);
          var duplicate = false;
          if (requiresDuplicateSearch(line)) {
            for (var earlierSlot = 0u; earlierSlot < candidateCount(line); earlierSlot++) {
              let earlier = candidateRow(line, earlierSlot);
              if (earlier < candidate && all(reference(earlier) == current)) { duplicate = true; break; }
            }
          }
          if (duplicate) { continue; }
          let halfRelative = current * 0.5 - start * 0.5;
          let relativeScale = max(max(abs(halfRelative.x), abs(halfRelative.y)), 1.175494351e-38);
          let relative = vec2f(getScaledQuotient(halfRelative.x, relativeScale, 1.0, 0), getScaledQuotient(halfRelative.y, relativeScale, 1.0, 0));
          let fraction = scaledProductQuotient(dot(relative, edge), relativeScale, divisor, denominator);
          // Perpendicular distance avoids reconstructing and rounding the projected coordinate.
          // The determinant is exactly zero for translated axis-collinear references at zero tolerance.
          let perpendicular = abs(relative.x * edge.y - relative.y * edge.x);
          let distance = scaledProductQuotient(perpendicular, relativeScale, sqrt(denominator) * 0.5, 1.0);
          if (fraction <= 0.0 || fraction >= 1.0 || distance > TOLERANCE) { continue; }
          if (fraction < previousFraction || (fraction == previousFraction && candidate <= previousReference)) { continue; }
          if (fraction < winnerFraction || (fraction == winnerFraction && candidate < winner)) { winner = candidate; winnerFraction = fraction; }
        }
        if (winner == 0xffffffffu) { break; }
        let current = reference(winner);
        if (any(current != previousPoint)) { writeVertex(line, required, current, vertex); required += min(1u, 0xffffffffu - required); previousPoint = current; }
        previousFraction = winnerFraction; previousReference = winner;
      }
    }
    if (any(end != previousPoint)) { writeVertex(line, required, end, vertex); required += min(1u, 0xffffffffu - required); }
    start = end;
  }
}
counts[countsOffset + line * 4u] = min(required, CAPACITY);
counts[countsOffset + line * 4u + 1u] = required;
counts[countsOffset + line * 4u + 2u] = select(0u, 1u, required > CAPACITY || incomplete);
counts[countsOffset + line * 4u + 3u] = select(0u, 1u, invalid);
${props.output.validity ? 'validity[validityOffset + line] = select(0u, 1u, required == 1u) | select(0u, 2u, invalid) | select(0u, 4u, required > CAPACITY || incomplete);' : ''}`
        })
      );
      if (this.outputViews.candidateOverflow)
        nodes.push(
          createWGSLKernelNode(graph, {
            id: `${id}-candidate-status-${firstLine}`,
            operation: 'GPUSegmentSnap',
            variant: 'candidate-status',
            invocationCount: grid ? MAXIMUM_BATCH_LINE_COUNT : lineCount,
            workgroupSize: grid ? MAXIMUM_BATCH_LINE_COUNT : undefined,
            guardIndex: !grid,
            bindings: [
              ...(candidates
                ? [
                    {
                      name: 'candidates',
                      view: candidates,
                      type: 'u32' as const,
                      access: 'read' as const
                    }
                  ]
                : []),
              {
                name: 'candidateOverflow',
                view: this.outputViews.candidateOverflow,
                type: 'u32',
                access: 'read_write'
              }
            ],
            body: candidates
              ? `if (index >= candidates[candidatesOffset + 2u]) { return; }
candidateOverflow[candidateOverflowOffset + candidates[candidatesOffset + 1u] + index] = candidates[candidatesOffset + 4u + index * ${candidateCapacity + 2}u];`
              : 'candidateOverflow[candidateOverflowOffset + index] = 0u;'
          })
        );
    }
    return nodes;
  }
}

// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {
  createGPUCopyCommandNode,
  createTransientView,
  GPUGridIndex,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {GPUParameterBuffer} from '../../analysis-core/kernels/gpu-contributor-utils';
import {SCALED_QUOTIENT_WGSL} from '../../analysis-core/kernels/scaled-quotient-wgsl';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {getBoundsReductionNodes} from '../hilbert-keys/bounds-reduction';

/** A fixed grid keeps build scratch independent of the number of line queries. */
export const SEGMENT_SNAP_GRID_WIDTH = 256;
/** Default per-line list; overflow is reported independently from output capacity. */
export const DEFAULT_SEGMENT_SNAP_CANDIDATE_CAPACITY = 1024;

// Half-coordinate subtraction avoids overflow for opposite, finite f32 extremes. Both the index
// and queries use this mapping. A one-cell halo conservatively covers its rounding at cell edges.
const NORMALIZE_POINT = `${SCALED_QUOTIENT_WGSL}
fn normalizeCoordinate(value: f32, minimum: f32, extent: f32) -> f32 {
  if (extent < 1.175494351e-38) { return 0.5; }
  // A conservative query expansion can overflow even though all caller coordinates are finite.
  // Never pass infinity into the finite-input exponent-scaled quotient helper.
  if ((bitcast<u32>(value) & 0x7f800000u) == 0x7f800000u) { return select(0.0, 1.0, value > 0.0); }
  return clamp(getScaledQuotient(value * 0.5 - minimum, extent, 1.0, 0), 0.0, 1.0);
}
fn normalizePoint(value: vec2f) -> vec2f {
  let minimum = vec2f(bounds[boundsOffset], bounds[boundsOffset + 1u]) * 0.5;
  let maximum = vec2f(bounds[boundsOffset + 2u], bounds[boundsOffset + 3u]) * 0.5;
  let extent = maximum - minimum;
  return vec2f(normalizeCoordinate(value.x, minimum.x, extent.x), normalizeCoordinate(value.y, minimum.y, extent.y));
}
fn finitePoint(value: vec2f) -> bool { return all((bitcast<vec2u>(value) & vec2u(0x7f800000u)) != vec2u(0x7f800000u)); }`;

/** Builds once, then gathers a conservative candidate superset for every line. */
export function createSegmentSnapCandidates<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    positions: GraphDataView<'float32x2'>;
    lineOffsets: GraphDataView<'uint32'>;
    referencePositions: GraphDataView<'float32x2'>;
    tolerance: number;
    candidateCapacity: number;
    batchLineCapacity: number;
  }
): {
  nodes: GPUCommandNode<Parameters>[];
  candidates: GraphDataView<'uint32'>;
  createBatchNodes: (firstLine: number, lineCount: number) => GPUCommandNode<Parameters>[];
} {
  const {id, candidateCapacity, referencePositions} = props;

  // Header: global nonfinite flag, first line in this batch, and batch length.
  // Reuse the records for every batch; they contain [required, overflow, ids...].
  const candidates = createTransientView(
    graph,
    `${id}-candidates`,
    'uint32',
    3 + props.batchLineCapacity * (candidateCapacity + 2),
    Buffer.STORAGE | Buffer.COPY_DST
  );
  const bounds = createTransientView(graph, `${id}-bounds`, 'float32', 4);
  const normalized = createTransientView(
    graph,
    `${id}-normalized`,
    'float32x2',
    referencePositions.length
  );
  const cellOffsets = createTransientView(
    graph,
    `${id}-cells`,
    'uint32',
    SEGMENT_SNAP_GRID_WIDTH ** 2 + 1
  );
  const objectIds = createTransientView(
    graph,
    `${id}-objects`,
    'uint32',
    referencePositions.length
  );
  const count = createTransientView(graph, `${id}-grid-count`, 'uint32', 1);
  const overflow = createTransientView(graph, `${id}-grid-overflow`, 'uint32', 1);
  const nodes: GPUCommandNode<Parameters>[] = [
    createWGSLKernelNode(graph, {
      id: `${id}-reset-candidates`,
      operation: 'GPUSegmentSnap',
      variant: 'reset-candidates',
      invocationCount: 1,
      bindings: [{name: 'candidates', view: candidates, type: 'u32', access: 'read_write'}],
      body: 'candidates[candidatesOffset] = 0u;'
    })
  ];
  nodes.push(
    ...getBoundsReductionNodes(graph, {
      id,
      operation: 'GPUSegmentSnap',
      variant: 'reference-bounds',
      itemCount: referencePositions.length,
      bindings: [{name: 'references', view: referencePositions, type: 'f32', access: 'read'}],
      declarations: `fn readItem(row: u32) -> vec4f {
      let value = vec2f(references[referencesOffset + row * 2u], references[referencesOffset + row * 2u + 1u]);
      let finite = all((bitcast<vec2u>(value) & vec2u(0x7f800000u)) != vec2u(0x7f800000u));
      return vec4f(value, select(0.0, 1.0, finite), 0.0);
    }`,
      output: bounds
    })
  );
  nodes.push(
    createWGSLKernelNode(graph, {
      id: `${id}-normalize`,
      operation: 'GPUSegmentSnap',
      variant: 'normalize-references',
      invocationCount: referencePositions.length,
      bindings: [
        {name: 'references', view: referencePositions, type: 'f32', access: 'read'},
        {name: 'bounds', view: bounds, type: 'f32', access: 'read'},
        {name: 'normalized', view: normalized, type: 'f32', access: 'read_write'},
        {name: 'candidates', view: candidates, type: 'atomic<u32>', access: 'read_write'}
      ],
      declarations: NORMALIZE_POINT,
      body: `let value = vec2f(references[referencesOffset + index * 2u], references[referencesOffset + index * 2u + 1u]);
      if (!finitePoint(value)) { atomicOr(&candidates[candidatesOffset], 1u); }
      let mapped = select(vec2f(-1.0), normalizePoint(value), finitePoint(value));
      normalized[normalizedOffset + index * 2u] = mapped.x;
      normalized[normalizedOffset + index * 2u + 1u] = mapped.y;`
    })
  );
  nodes.push(
    ...new GPUGridIndex({
      id: `${id}-grid`,
      positions: normalized,
      gridSize: [SEGMENT_SNAP_GRID_WIDTH, SEGMENT_SNAP_GRID_WIDTH],
      bounds: [0, 0, 1, 1],
      cellOffsets,
      objectIds,
      count,
      overflow
    }).getCommandNodes(graph)
  );
  const createBatchNodes = (firstLine: number, lineCount: number): GPUCommandNode<Parameters>[] => {
    const batchId = `${id}-batch-${firstLine}`;
    const batchNodes: GPUCommandNode<Parameters>[] = [
      createGPUCopyCommandNode({
        id: `${batchId}-parameters`,
        resources: [{buffer: candidates, usage: 'copy-destination'}],
        compile: ({device}) => {
          const parameters = new GPUParameterBuffer(device, {
            id: `${batchId}-defaults`,
            format: 'uint32',
            length: 2,
            values: new Uint32Array([firstLine, lineCount])
          });
          return {
            encode: ({commandEncoder, getBuffer}) =>
              commandEncoder.copyBufferToBuffer({
                sourceBuffer: parameters.buffer,
                sourceOffset: 0,
                destinationBuffer: getBuffer(candidates),
                destinationOffset: candidates.byteOffset + 4,
                size: 8
              }),
            destroy: () => parameters.destroy()
          };
        }
      })
    ];
    batchNodes.push(
      createWGSLKernelNode(graph, {
        id: `${batchId}-gather-candidates`,
        operation: 'GPUSegmentSnap',
        variant: 'line-grid-candidates',
        invocationCount: 256,
        workgroupSize: 256,
        guardIndex: false,
        bindings: [
          {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
          {name: 'lineOffsets', view: props.lineOffsets, type: 'u32', access: 'read'},
          {name: 'bounds', view: bounds, type: 'f32', access: 'read'},
          {name: 'cellOffsets', view: cellOffsets, type: 'u32', access: 'read'},
          {name: 'objectIds', view: objectIds, type: 'u32', access: 'read'},
          {name: 'candidates', view: candidates, type: 'u32', access: 'read_write'}
        ],
        declarations: `${NORMALIZE_POINT}
const CANDIDATE_CAPACITY: u32 = ${candidateCapacity}u;
const GRID_WIDTH: u32 = ${SEGMENT_SNAP_GRID_WIDTH}u;
const TOLERANCE: f32 = ${Math.fround(props.tolerance)};`,
        body: `if (index >= candidates[candidatesOffset + 2u]) { return; }
      let record = candidatesOffset + 3u + index * (CANDIDATE_CAPACITY + 2u);
      let rowCount = (arrayLength(&positions) - positionsOffset) / 2u;
      let first = min(lineOffsets[lineOffsetsOffset + candidates[candidatesOffset + 1u] + index], rowCount);
      let last = min(lineOffsets[lineOffsetsOffset + candidates[candidatesOffset + 1u] + index + 1u], rowCount);
      var minimum = vec2f(1.0); var maximum = vec2f(0.0);
      for (var row = first; row < last; row++) {
        let value = vec2f(positions[positionsOffset + row * 2u], positions[positionsOffset + row * 2u + 1u]);
        // Endpoints may move by tolerance before insertion, so the conservative radius is twice
        // tolerance. Add in half units to avoid overflowing the literal or multiplication.
        // Cover f32 rounding in snapping and convex interpolation too. Otherwise a reference
        // one ulp outside a very narrow line box can pass the exact kernel's rounded predicate.
        let roundingMargin = max(abs(value), vec2f(TOLERANCE)) * 9.5367431640625e-7 + vec2f(1.175494351e-38);
        let low = normalizePoint(value - vec2f(TOLERANCE) - vec2f(TOLERANCE) - roundingMargin);
        let high = normalizePoint(value + vec2f(TOLERANCE) + vec2f(TOLERANCE) + roundingMargin);
        minimum = min(minimum, low); maximum = max(maximum, high);
        if (!finitePoint(low) || !finitePoint(high)) { minimum = vec2f(0.0); maximum = vec2f(1.0); }
      }
      let lowerCell = vec2u(max(vec2f(0.0), floor(minimum * f32(GRID_WIDTH)) - vec2f(1.0)));
      let upperCell = vec2u(min(vec2f(f32(GRID_WIDTH - 1u)), floor(maximum * f32(GRID_WIDTH)) + vec2f(1.0)));
      var required = 0u;
      if (last > first) {
        for (var vertical = lowerCell.y; vertical <= upperCell.y; vertical++) {
          for (var horizontal = lowerCell.x; horizontal <= upperCell.x; horizontal++) {
            let cell = vertical * GRID_WIDTH + horizontal;
            let start = cellOffsets[cellOffsetsOffset + cell];
            let end = cellOffsets[cellOffsetsOffset + cell + 1u];
            let available = min(end - start, CANDIDATE_CAPACITY - min(required, CANDIDATE_CAPACITY));
            for (var slot = 0u; slot < available; slot++) {
              candidates[record + 2u + required + slot] = objectIds[objectIdsOffset + start + slot];
            }
            required += min(end - start, 0xffffffffu - required);
          }
        }
      }
      candidates[record] = required;
      candidates[record + 1u] = select(0u, 1u, required > CANDIDATE_CAPACITY);`
      })
    );
    batchNodes.push(
      createWGSLKernelNode(graph, {
        id: `${batchId}-deduplicate-candidates`,
        operation: 'GPUSegmentSnap',
        variant: 'unique-line-candidates',
        invocationCount: 256,
        workgroupSize: 256,
        guardIndex: false,
        bindings: [
          {name: 'references', view: referencePositions, type: 'f32', access: 'read'},
          {name: 'candidates', view: candidates, type: 'u32', access: 'read_write'}
        ],
        declarations: `const CANDIDATE_CAPACITY: u32 = ${candidateCapacity}u;
fn referencePoint(row: u32) -> vec2f { return vec2f(references[referencesOffset + row * 2u], references[referencesOffset + row * 2u + 1u]); }`,
        body: `if (index >= candidates[candidatesOffset + 2u]) { return; }
let record = candidatesOffset + 3u + index * (CANDIDATE_CAPACITY + 2u);
if (candidates[candidatesOffset] != 0u || candidates[record + 1u] != 0u) { return; }
let required = candidates[record];
var written = 0u;
for (var slot = 0u; slot < required; slot++) {
  let row = candidates[record + 2u + slot];
  let point = referencePoint(row);
  var duplicate = false;
  for (var other = 0u; other < required; other++) {
    let earlierRow = candidates[record + 2u + other];
    if (earlierRow < row && all(referencePoint(earlierRow) == point)) { duplicate = true; break; }
  }
  // Compaction only overwrites an earlier slot. Every coordinate's minimum source row remains
  // in the prefix or the unread suffix, so duplicate decisions retain the original row tie break.
  if (!duplicate) { candidates[record + 2u + written] = row; written++; }
}
candidates[record] = written;`
      })
    );
    return batchNodes;
  };
  return {nodes, candidates, createBatchNodes};
}

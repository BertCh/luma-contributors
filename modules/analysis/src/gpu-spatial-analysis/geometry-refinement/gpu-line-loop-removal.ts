// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {validateGraphChunkBindingRanges} from '../../analysis-core/kernels/graph-chunk-dispatch';
import {validateOutlineOutputPasses} from '../outline-geometry/validate-outline-output-passes';
import {
  validateRefinementScratchByteLengths,
  validateRefinementLineNodingScratch
} from './refinement-storage-validation';
import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import type {GPULineGeometryPort, GPUPackedGeometryPort} from '../../analysis-core/contracts/index';
import {
  resolveGPUSpatialContext,
  type GPUSpatialContext
} from '../../analysis-core/contracts/spatial-context';
import {GPULineSplit} from '../line-split/gpu-line-split';
import {
  validateCapacity,
  validateSinglePackedView,
  validateGraphOutputsDisjointFromInputs
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';

export type GPULineLoopRemovalProps = {
  id?: string;
  spatialContext?: GPUSpatialContext;
  /** Explicitly prepared offset curves or other open planar lines. */
  lines: GPUPackedGeometryPort<GPULineGeometryPort>;
  intersectionCapacity: number;
  output: {
    positions: GraphDataView<'float32x2'>;
    lineOffsets: GraphDataView<'uint32'>;
    /** First input vertex of the source edge for each output vertex. */
    sourceEdges: GraphDataView<'uint32'>;
    /** [writtenVertices, requiredVertices, outputOverflow, incompleteNoding]. */
    status: GraphDataView<'uint32x4'>;
  };
};

/**
 * Chronological loop erasure of noded open curves. Exact-predicate line splitting
 * inserts crossing vertices; repeated visits erase the intervening path suffix.
 * Source line identity, direction and edge provenance are retained. This is the
 * loop-removal stage for raw offset curves, not a claim of full GEOS offset_curve
 * component selection: choosing components by distance/side remains separate.
 */
export class GPULineLoopRemoval implements GPUCommandNodeProducer {
  readonly id: string;
  readonly props: GPULineLoopRemovalProps;
  constructor(props: GPULineLoopRemovalProps) {
    this.id = props.id ?? 'line-loop-removal';
    this.props = props;
    resolveGPUSpatialContext(this.id, props.spatialContext);
    validateSinglePackedView(props.lines.positions, ['float32x2'], this.id);
    validateSinglePackedView(props.lines.lineOffsets, ['uint32'], this.id);
    validateCapacity(this.id, 'lines.lineOffsets.length', props.lines.lineOffsets.length, 2);
    validateCapacity(this.id, 'intersectionCapacity', props.intersectionCapacity, 1);
    validateSinglePackedView(props.output.positions, ['float32x2'], this.id);
    validateSinglePackedView(props.output.lineOffsets, ['uint32'], this.id);
    validateSinglePackedView(props.output.sourceEdges, ['uint32'], this.id);
    validateSinglePackedView(props.output.status, ['uint32x4'], this.id);
    if (
      props.output.lineOffsets.length !== props.lines.lineOffsets.length ||
      props.output.sourceEdges.length !== props.output.positions.length ||
      props.output.status.length !== 1
    ) {
      throw new Error('Loop output capacity mismatch');
    }
    validateOutlineOutputPasses(this.id, props.output, [
      ['positions', 'sourceEdges'],
      ['lineOffsets', 'status']
    ]);
    validateGraphOutputsDisjointFromInputs(this.id, Object.values(props.output), [
      props.lines.positions,
      props.lines.lineOffsets
    ]);
  }
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {props, id} = this;
    validateGraphChunkBindingRanges(
      graph,
      id,
      Object.fromEntries(
        [props.lines.positions, props.lines.lineOffsets, ...Object.values(props.output)].map(
          (view, index) => [`binding.${index}`, view]
        )
      )
    );
    validateRefinementLineNodingScratch(
      graph,
      id,
      props.lines.positions.length,
      props.lines.lineOffsets.length - 1,
      props.intersectionCapacity
    );
    validateRefinementScratchByteLengths(graph, id, {
      stack: (props.lines.positions.length + props.intersectionCapacity * 8) * 16,
      lineState: (props.lines.lineOffsets.length - 1) * 8
    });
    const lineCount = props.lines.lineOffsets.length - 1;
    const pieceCapacity = lineCount + 4 * props.intersectionCapacity;
    const vertexCapacity = props.lines.positions.length + 8 * props.intersectionCapacity;
    const positions = createTransientView(
      graph,
      `${id}-split-positions`,
      'float32x2',
      vertexCapacity
    );
    const offsets = createTransientView(graph, `${id}-split-offsets`, 'uint32', pieceCapacity + 1);
    const sourceIds = createTransientView(graph, `${id}-split-sources`, 'uint32', pieceCapacity);
    const sourceEdges = createTransientView(graph, `${id}-split-edges`, 'uint32', vertexCapacity);
    const count = createTransientView(graph, `${id}-split-count`, 'uint32', 1);
    const requiredCount = createTransientView(graph, `${id}-split-required`, 'uint32', 1);
    const overflow = createTransientView(graph, `${id}-split-overflow`, 'uint32', 1);
    const candidateOverflow = createTransientView(
      graph,
      `${id}-split-candidate-overflow`,
      'uint32',
      1
    );
    const stack = createTransientView(graph, `${id}-stack`, 'uint32x4', vertexCapacity);
    const lineState = createTransientView(graph, `${id}-line-state`, 'uint32x2', lineCount);
    const outputStart = createTransientView(graph, `${id}-output-start`, 'uint32', lineCount);
    return [
      ...new GPULineSplit({
        id: `${id}-split`,
        lines: props.lines,
        spatialContext: props.spatialContext,
        intersectionCapacity: props.intersectionCapacity,
        output: {
          pieces: {
            geometry: {kind: 'lines', positions, lineOffsets: offsets},
            sourceIds,
            vertexSourceEdges: sourceEdges,
            status: {count, requiredCount, overflow, candidateOverflow}
          }
        }
      }).getCommandNodes(graph),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-erase`,
        operation: 'GPULineLoopRemoval',
        variant: 'chronological-erasure',
        // One thread per source line; split pieces are ordered by source line, so each thread
        // binary-searches its contiguous piece range and uses its own vertex region of the stack.
        invocationCount: lineCount,
        bindings: [
          {name: 'positions', view: positions, type: 'f32', access: 'read'},
          {name: 'offsets', view: offsets, type: 'u32', access: 'read'},
          {name: 'sourceIds', view: sourceIds, type: 'u32', access: 'read'},
          {name: 'sourceEdges', view: sourceEdges, type: 'u32', access: 'read'},
          {name: 'count', view: count, type: 'u32', access: 'read'},
          {name: 'stack', view: stack, type: 'u32', access: 'read_write'},
          {name: 'lineState', view: lineState, type: 'u32', access: 'read_write'}
        ],
        declarations: `fn findFirstPiece(line: u32) -> u32 {
  var low = 0u; var high = count[countOffset];
  while (low < high) {
    let middle = (low + high) >> 1u;
    if (sourceIds[sourceIdsOffset + middle] < line) { low = middle + 1u; } else { high = middle; }
  }
  return low;
}`,
        body: `let line = index;
let firstPiece = findFirstPiece(line); let endPiece = findFirstPiece(line + 1u);
let base = offsets[offsetsOffset + firstPiece];
var kept = 0u;
for (var piece = firstPiece; piece < endPiece; piece++) {
  if (sourceIds[sourceIdsOffset + piece] != line) { continue; }
  let first = offsets[offsetsOffset + piece]; let last = offsets[offsetsOffset + piece + 1u];
  for (var vertex = first; vertex < last; vertex++) {
    let current = vec2f(positions[positionsOffset + vertex * 2u], positions[positionsOffset + vertex * 2u + 1u]);
    var found = kept;
    for (var previous = 0u; previous < kept; previous++) {
      let destination = stackOffset + (base + previous) * 4u;
      let oldPoint = vec2f(bitcast<f32>(stack[destination]), bitcast<f32>(stack[destination + 1u]));
      if (all(current == oldPoint)) { found = previous; break; }
    }
    if (found < kept) { kept = found + 1u; }
    else { kept++; }
    let destination = stackOffset + (base + kept - 1u) * 4u;
    stack[destination] = bitcast<u32>(current.x); stack[destination + 1u] = bitcast<u32>(current.y);
    if (vertex + 1u < last) { stack[destination + 2u] = sourceEdges[sourceEdgesOffset + vertex] & 0x7fffffffu; }
    else if (kept > 1u) { stack[destination + 2u] = stack[destination - 2u]; }
  }
}
if (kept < 2u) { kept = 0u; }
lineState[lineStateOffset + line * 2u] = base; lineState[lineStateOffset + line * 2u + 1u] = kept;`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-publish-offsets`,
        operation: 'GPULineLoopRemoval',
        variant: 'publish-offsets',
        // Whole-line capacity dropping is order dependent (a later short line can still fit), so
        // only this O(lines) bookkeeping loop is sequential; vertex copies run per line below.
        invocationCount: 1,
        bindings: [
          {name: 'lineState', view: lineState, type: 'u32', access: 'read'},
          {name: 'nodingOverflow', view: overflow, type: 'u32', access: 'read'},
          {name: 'candidateOverflow', view: candidateOverflow, type: 'u32', access: 'read'},
          {name: 'outputStart', view: outputStart, type: 'u32', access: 'read_write'},
          {name: 'outOffsets', view: props.output.lineOffsets, type: 'u32', access: 'read_write'},
          {name: 'status', view: props.output.status, type: 'u32', access: 'read_write'}
        ],
        body: `var required = 0u; var written = 0u;
for (var line = 0u; line < ${lineCount}u; line++) {
  let length = lineState[lineStateOffset + line * 2u + 1u];
  outOffsets[outOffsetsOffset + line] = written; required += length;
  // Drop whole lines that would exceed capacity, preserving valid path boundaries.
  if (written + length > ${props.output.positions.length}u) { outputStart[outputStartOffset + line] = 0xffffffffu; continue; }
  outputStart[outputStartOffset + line] = written; written += length;
}
outOffsets[outOffsetsOffset + ${lineCount}u] = written;
status[statusOffset] = written; status[statusOffset + 1u] = required;
status[statusOffset + 2u] = select(0u, 1u, written < required); status[statusOffset + 3u] = nodingOverflow[nodingOverflowOffset] | candidateOverflow[candidateOverflowOffset];`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-publish`,
        operation: 'GPULineLoopRemoval',
        variant: 'publish',
        invocationCount: lineCount,
        bindings: [
          {name: 'stack', view: stack, type: 'u32', access: 'read'},
          {name: 'lineState', view: lineState, type: 'u32', access: 'read'},
          {name: 'outputStart', view: outputStart, type: 'u32', access: 'read'},
          {name: 'outPositions', view: props.output.positions, type: 'f32', access: 'read_write'},
          {name: 'outEdges', view: props.output.sourceEdges, type: 'u32', access: 'read_write'}
        ],
        body: `let start = outputStart[outputStartOffset + index];
if (start == 0xffffffffu) { return; }
let base = lineState[lineStateOffset + index * 2u]; let length = lineState[lineStateOffset + index * 2u + 1u];
for (var vertex = 0u; vertex < length; vertex++) {
  let source = stackOffset + (base + vertex) * 4u; let written = start + vertex;
  outPositions[outPositionsOffset + written * 2u] = bitcast<f32>(stack[source]);
  outPositions[outPositionsOffset + written * 2u + 1u] = bitcast<f32>(stack[source + 1u]);
  outEdges[outEdgesOffset + written] = stack[source + 2u];
}`
      })
    ];
  }
}

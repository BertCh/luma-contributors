// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {
  snapshotLineFeatureGeometry,
  type GPULineFeatureGeometry
} from '../../analysis-core/contracts/line-feature-provenance';
import {snapshotConstructionFields} from '../../analysis-core/kernels/construction-snapshot';

import {validateGraphChunkBindingRanges} from '../../analysis-core/kernels/graph-chunk-dispatch';
import {validateOutlineOutputPasses} from '../outline-geometry/validate-outline-output-passes';
import {
  validateRefinementScratchByteLengths,
  validateRefinementLineNodingScratch
} from './refinement-storage-validation';
import {
  createTransientView,
  GPUScan,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
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

/** Maximum source lines considered by one ordered capacity-selection dispatch. */
export const MAXIMUM_LOOP_REMOVAL_SELECTION_LINES_PER_DISPATCH = 256;

export type GPULineLoopRemovalProps = {
  id?: string;
  spatialContext?: GPUSpatialContext;
  /** Explicitly prepared offset curves or other open planar lines. */
  lines: GPULineFeatureGeometry;
  intersectionCapacity: number;
  output: {
    positions: GraphDataView<'float32x2'>;
    lineOffsets: GraphDataView<'uint32'>;
    /** Optional unchanged feature-to-part offsets; erased/truncated parts keep empty line rows. */
    featureOffsets?: GraphDataView<'uint32'>;
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
 * component selection: choosing components by distance/side remains separate. Capacity selection
 * considers at most 256 source lines per ordered dispatch, preserving whole-line first-fit order;
 * offset publication uses a parallel prefix scan with no total source-line cap.
 */
export class GPULineLoopRemoval implements GPUCommandNodeProducer {
  readonly id: string;
  readonly props: GPULineLoopRemovalProps;
  constructor(props: GPULineLoopRemovalProps) {
    this.id = props.id ?? 'line-loop-removal';
    resolveGPUSpatialContext(this.id, props.spatialContext);
    const snapshot = snapshotConstructionFields(props, [
      'id',
      'spatialContext',
      'lines',
      'intersectionCapacity',
      'output'
    ]);
    props = Object.freeze({
      ...snapshot,
      lines: snapshotLineFeatureGeometry(this.id, snapshot.lines),
      output: snapshotConstructionFields(snapshot.output, [
        'positions',
        'lineOffsets',
        'featureOffsets',
        'sourceEdges',
        'status'
      ])
    });
    this.props = props;
    validateSinglePackedView(props.lines.positions, ['float32x2'], this.id);
    validateSinglePackedView(props.lines.lineOffsets, ['uint32'], this.id);
    validateCapacity(this.id, 'lines.lineOffsets.length', props.lines.lineOffsets.length, 2);
    validateCapacity(this.id, 'intersectionCapacity', props.intersectionCapacity, 1);
    validateSinglePackedView(props.output.positions, ['float32x2'], this.id);
    validateSinglePackedView(props.output.lineOffsets, ['uint32'], this.id);
    if (props.output.featureOffsets) {
      validateSinglePackedView(
        props.output.featureOffsets,
        ['uint32'],
        `${this.id} output.featureOffsets`
      );
      validateCapacity(
        this.id,
        'output.featureOffsets.length',
        props.output.featureOffsets.length,
        (props.lines.featureOffsets ?? props.lines.lineOffsets).length,
        (props.lines.featureOffsets ?? props.lines.lineOffsets).length
      );
    }
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
      props.lines.lineOffsets,
      props.lines.featureOffsets
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
        [
          props.lines.positions,
          props.lines.lineOffsets,
          props.lines.featureOffsets,
          ...Object.values(props.output)
        ]
          .filter(Boolean)
          .map((view, index) => [`binding.${index}`, view])
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
      lineState: (props.lines.lineOffsets.length - 1) * 8,
      publicationPrefix: props.lines.lineOffsets.length * 4
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
    const outputStart = createTransientView(graph, `${id}-output-start`, 'uint32', lineCount + 1);
    const selectedLengths = createTransientView(
      graph,
      `${id}-selected-lengths`,
      'uint32',
      lineCount + 1
    );
    const selectionState = createTransientView(graph, `${id}-selection-state`, 'uint32', 2);
    return [
      ...new GPULineSplit({
        id: `${id}-split`,
        lines: props.lines,
        spatialContext: props.spatialContext,
        intersectionCapacity: props.intersectionCapacity,
        output: {
          pieces: {
            geometry: {kind: 'lines', positions, lineOffsets: offsets},
            sourceIds: createTransientView(
              graph,
              `${id}-split-identities`,
              'uint32',
              pieceCapacity
            ),
            sourcePartRows: sourceIds,
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
      createWGSLKernelNode(graph, {
        id: `${id}-clear-selection`,
        operation: 'GPULineLoopRemoval',
        invocationCount: 1,
        bindings: [
          {name: 'state', view: selectionState, type: 'u32', access: 'read_write'},
          {name: 'lengths', view: selectedLengths, type: 'u32', access: 'read_write'}
        ],
        body: `state[stateOffset]=0u;state[stateOffset+1u]=0u;lengths[lengthsOffset+${lineCount}u]=0u;`
      }),
      // A later short line can still fit after a dropped line. Preserve that first-fit rule in
      // bounded ordered chunks; only their capacity decisions are serial. Offsets use GPUScan.
      ...Array.from(
        {length: Math.ceil(lineCount / MAXIMUM_LOOP_REMOVAL_SELECTION_LINES_PER_DISPATCH)},
        (_, chunk) =>
          createWGSLKernelNode(graph, {
            id: `${id}-select-lines-${chunk}`,
            operation: 'GPULineLoopRemoval',
            // contract-allow: single-invocation bounded by MAXIMUM_LOOP_REMOVAL_SELECTION_LINES_PER_DISPATCH = 256 source lines per dispatch.
            invocationCount: 1,
            bindings: [
              {name: 'lineState', view: lineState, type: 'u32', access: 'read'},
              {name: 'state', view: selectionState, type: 'u32', access: 'read_write'},
              {name: 'lengths', view: selectedLengths, type: 'u32', access: 'read_write'}
            ],
            body: `var written=state[stateOffset];var required=state[stateOffset+1u];
for(var local=0u;local<${MAXIMUM_LOOP_REMOVAL_SELECTION_LINES_PER_DISPATCH}u;local++) {
 let line=${chunk * MAXIMUM_LOOP_REMOVAL_SELECTION_LINES_PER_DISPATCH}u+local;if(line>=${lineCount}u) {break;}
 let length=lineState[lineStateOffset+line*2u+1u];required+=length;
 let fits=length<=${props.output.positions.length}u-written;
 lengths[lengthsOffset+line]=select(0u,length,fits);if(fits) {written+=length;}
}
state[stateOffset]=written;state[stateOffset+1u]=required;`
          })
      ),
      ...new GPUScan({
        id: `${id}-offset-prefix`,
        input: selectedLengths,
        output: outputStart,
        mode: 'exclusive'
      }).getCommandNodes(graph),
      createWGSLKernelNode(graph, {
        id: `${id}-publish-offsets`,
        operation: 'GPULineLoopRemoval',
        variant: 'publish-offsets',
        invocationCount: lineCount + 1,
        bindings: [
          {name: 'starts', view: outputStart, type: 'u32', access: 'read'},
          {name: 'outOffsets', view: props.output.lineOffsets, type: 'u32', access: 'read_write'}
        ],
        body: 'outOffsets[outOffsetsOffset+index]=starts[startsOffset+index];'
      }),
      createWGSLKernelNode(graph, {
        id: `${id}-publish-status`,
        operation: 'GPULineLoopRemoval',
        invocationCount: 1,
        bindings: [
          {name: 'state', view: selectionState, type: 'u32', access: 'read'},
          {name: 'nodingOverflow', view: overflow, type: 'u32', access: 'read'},
          {name: 'candidateOverflow', view: candidateOverflow, type: 'u32', access: 'read'},
          {name: 'status', view: props.output.status, type: 'u32', access: 'read_write'}
        ],
        body: `let written=state[stateOffset];let required=state[stateOffset+1u];
status[statusOffset]=written;status[statusOffset+1u]=required;status[statusOffset+2u]=select(0u,1u,written<required);status[statusOffset+3u]=nodingOverflow[nodingOverflowOffset]|candidateOverflow[candidateOverflowOffset];`
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
          {name: 'lengths', view: selectedLengths, type: 'u32', access: 'read'},
          {name: 'outPositions', view: props.output.positions, type: 'f32', access: 'read_write'},
          {name: 'outEdges', view: props.output.sourceEdges, type: 'u32', access: 'read_write'}
        ],
        body: `let start = outputStart[outputStartOffset + index];
if (lengths[lengthsOffset + index] == 0u) { return; }
let base = lineState[lineStateOffset + index * 2u]; let length = lineState[lineStateOffset + index * 2u + 1u];
for (var vertex = 0u; vertex < length; vertex++) {
  let source = stackOffset + (base + vertex) * 4u; let written = start + vertex;
  outPositions[outPositionsOffset + written * 2u] = bitcast<f32>(stack[source]);
  outPositions[outPositionsOffset + written * 2u + 1u] = bitcast<f32>(stack[source + 1u]);
  outEdges[outEdgesOffset + written] = stack[source + 2u];
}`
      }),
      ...(props.output.featureOffsets
        ? [
            createWGSLKernelNode(graph, {
              id: `${id}-feature-offsets`,
              operation: 'GPULineLoopRemoval',
              invocationCount: props.output.featureOffsets.length,
              bindings: [
                ...(props.lines.featureOffsets
                  ? [
                      {
                        name: 'features',
                        view: props.lines.featureOffsets,
                        type: 'u32',
                        access: 'read'
                      } as const
                    ]
                  : []),
                {
                  name: 'output',
                  view: props.output.featureOffsets,
                  type: 'u32',
                  access: 'read_write'
                }
              ],
              body: `output[outputOffset+index]=${props.lines.featureOffsets ? 'features[featuresOffset+index]' : 'index'};`
            })
          ]
        : [])
    ];
  }
}

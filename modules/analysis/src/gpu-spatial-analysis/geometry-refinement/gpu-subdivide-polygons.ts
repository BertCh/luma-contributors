// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  snapshotConstructionFields,
  snapshotConstructionSpatialContext
} from '../../analysis-core/kernels/construction-snapshot';
import {resolveGPUSpatialContext} from '../../analysis-core/contracts/spatial-context';
import {validateGraphChunkBindingRanges} from '../../analysis-core/kernels/graph-chunk-dispatch';
import {validateOutlineOutputPasses} from '../outline-geometry/validate-outline-output-passes';
import {validateRefinementScratchByteLengths} from './refinement-storage-validation';
import {getRectilinearSubdivisionScratchByteLengths} from './rectilinear-subdivision';
import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import type {GPUPolygonGeometryFields} from '../../analysis-core/contracts/index';
import type {GPUSpatialContext} from '../../analysis-core/contracts/spatial-context';
import {
  GPUPolygonTriangulation,
  getPolygonTriangulationIndexCount,
  validatePolygonTriangulationScratch,
  validatePolygonTriangulationWorkLimits
} from '../polygon-triangulation/gpu-polygon-triangulation';
import {
  validateCapacity,
  validateOption,
  validateSinglePackedView,
  validateGraphOutputsDisjointFromInputs
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {GPUScan} from '@luma.gl/graph/gpu-core';
import {createFillNode, createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {getRectilinearSubdivisionNodes} from './rectilinear-subdivision';
import {EXACT_ORIENTATION_WGSL} from '../../analysis-core/kernels/exact-orientation-wgsl';
import {getReadPointWGSL} from '../shared/spatial-wgsl-point-helpers';

export type GPUSubdividePolygonsProps = GPUPolygonGeometryFields & {
  id?: string;
  spatialContext?: GPUSpatialContext;
  /** Rectilinear mode clips a hole-aware convex decomposition with an explicit GPU work queue. */
  method?: 'convex' | 'rectilinear';
  /** Maximum queued split operations per convex seed in rectilinear mode. */
  maximumSplits?: number;
  /** Maximum vertices in each convex output piece, between 3 and 128. */
  maximumVertices?: number;
  /** Forwarded triangulation work budget per polygon. */
  maximumWork?: number;
  output: {
    /** Fixed piece blocks of maximumVertices positions; only metadata vertexCount is active. */
    positions: GraphDataView<'float32x2'>;
    /** [vertexCount, sourceFeature, sourcePolygon, sourceSeedTriangle] per piece. */
    pieces: GraphDataView<'uint32x4'>;
    /** One row [writtenPieces, requiredPieces, capacityOverflow, invalidPolygonCount]. Bit 31 of the last word reports incomplete rectilinear splitting. */
    status: GraphDataView<'uint32x4'>;
  };
};

/**
 * Bounded convex subdivision: triangulate polygon interiors (including holes),
 * then merge edge-adjacent triangles while the union remains convex and within
 * the vertex limit. The piece union reconstructs every successfully triangulated
 * source polygon. This is a triangulate-and-merge decomposition, not PostGIS's
 * rectilinear partition or an optimization of its exact piece ordering.
 */
export class GPUSubdividePolygons implements GPUCommandNodeProducer {
  readonly id: string;
  readonly props: GPUSubdividePolygonsProps;
  readonly maximumVertices: number;
  constructor(props: GPUSubdividePolygonsProps) {
    props = snapshotConstructionFields(props, [
      'id',
      'spatialContext',
      'method',
      'maximumSplits',
      'maximumVertices',
      'maximumWork',
      'positions',
      'ringOffsets',
      'polygonOffsets',
      'featureOffsets',
      'output'
    ]);
    props = Object.freeze({
      ...props,
      spatialContext:
        props.spatialContext && snapshotConstructionSpatialContext(props.spatialContext),
      output: snapshotConstructionFields(props.output, ['positions', 'pieces', 'status'])
    });
    this.id = props.id ?? 'subdivide-polygons';
    this.props = props;
    resolveGPUSpatialContext(this.id, props.spatialContext);
    this.maximumVertices = props.maximumVertices ?? 32;
    validateOption(this.id, 'method', props.method === undefined ? 'convex' : props.method, [
      'convex',
      'rectilinear'
    ]);
    validatePolygonTriangulationWorkLimits(this.id, props);
    const identity = {id: this.id, contributor: 'GPUSubdividePolygons'};
    validateSinglePackedView(props.positions, ['float32x2'], identity, 'positions');
    for (const name of ['ringOffsets', 'polygonOffsets', 'featureOffsets'] as const) {
      validateSinglePackedView(props[name], ['uint32'], identity, name);
      validateCapacity(this.id, `${name}.length (including terminal offset)`, props[name].length);
    }
    validateCapacity(
      this.id,
      'maximumVertices',
      this.maximumVertices,
      props.method === 'rectilinear' ? 4 : 3,
      128
    );
    validateCapacity(this.id, 'maximumSplits', props.maximumSplits ?? 4096, 1, 65536);
    validateCapacity(this.id, 'positions', props.positions.length, 1, 0x7fffffff);
    validateSinglePackedView(props.output.positions, ['float32x2'], identity, 'output.positions');
    validateSinglePackedView(props.output.pieces, ['uint32x4'], identity, 'output.pieces');
    validateSinglePackedView(props.output.status, ['uint32x4'], identity, 'output.status');
    if (
      props.output.positions.length !== props.output.pieces.length * this.maximumVertices ||
      props.output.status.length !== 1
    ) {
      throw new Error('Subdivision output capacity mismatch');
    }
    validateOutlineOutputPasses(this.id, props.output, [['positions', 'pieces'], ['status']]);
    validateGraphOutputsDisjointFromInputs(this.id, Object.values(props.output), [
      props.positions,
      props.ringOffsets,
      props.polygonOffsets,
      props.featureOffsets
    ]);
  }
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {props, id, maximumVertices} = this;
    validateGraphChunkBindingRanges(
      graph,
      id,
      Object.fromEntries(
        [
          props.positions,
          props.ringOffsets,
          props.polygonOffsets,
          props.featureOffsets,
          ...Object.values(props.output)
        ].map((view, index) => [`binding.${index}`, view])
      )
    );
    const inputPolygonCount = props.polygonOffsets.length - 1;
    if (inputPolygonCount === 0) {
      return [
        createZeroPiecesNode(graph, id, props.output.pieces),
        createFillNode(graph, {
          id: `${id}-empty-status`,
          operation: 'GPUSubdividePolygons',
          view: props.output.status,
          type: 'u32',
          value: '0u'
        })
      ];
    }
    validatePolygonTriangulationScratch(
      graph.device,
      id,
      props.positions.length,
      props.ringOffsets.length - 1,
      inputPolygonCount
    );
    validateRefinementScratchByteLengths(graph, id, {
      indices:
        Math.max(
          3,
          getPolygonTriangulationIndexCount(
            props.positions.length,
            props.ringOffsets.length - 1,
            inputPolygonCount
          )
        ) * 4,
      polygonInfo: inputPolygonCount * 8,
      triangleOffsets: (inputPolygonCount + 1) * 4,
      ...(props.method === 'rectilinear'
        ? getRectilinearSubdivisionScratchByteLengths(
            props.positions.length,
            props.ringOffsets.length - 1
          )
        : {})
    });
    if (props.method === 'rectilinear') return getRectilinearSubdivisionNodes(graph, this);
    const polygonCount = props.polygonOffsets.length - 1;
    const indexCapacity = Math.max(
      3,
      getPolygonTriangulationIndexCount(
        props.positions.length,
        props.ringOffsets.length - 1,
        polygonCount
      )
    );
    const indices = createTransientView(graph, `${id}-indices`, 'uint32', indexCapacity);
    const valid = createTransientView(graph, `${id}-valid`, 'uint32', polygonCount);
    const triangleOffsets = createTransientView(
      graph,
      `${id}-triangle-offsets`,
      'uint32',
      polygonCount + 1
    );
    const info = createTransientView(
      graph,
      `${id}-polygon-info`,
      'uint32',
      Math.max(polygonCount * 2, 1)
    );
    const triangleSlotCount = Math.max(Math.floor(indexCapacity / 3), 1);
    const consumed = createTransientView(graph, `${id}-consumed`, 'uint32', triangleSlotCount);
    const pieceVertexCounts = createTransientView(
      graph,
      `${id}-piece-vertex-counts`,
      'uint32',
      triangleSlotCount
    );
    const pieceFlags = createTransientView(graph, `${id}-piece-flags`, 'uint32', triangleSlotCount);
    const pieceOffsets = createTransientView(
      graph,
      `${id}-piece-offsets`,
      'uint32',
      triangleSlotCount
    );
    // One invocation per polygon runs the greedy merge of its own triangles. Pieces are numbered
    // by a scan over the seed triangles, so a first `count` pass fixes the piece starts, a scan
    // numbers them, and a second `write` pass repeats the same merge to write each piece to its
    // numbered slot. Pieces keep the polygon-major, seed-order sequence of the serial algorithm.
    const getMergePass = (pass: 'count' | 'write') =>
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-merge-${pass}`,
        operation: 'GPUSubdividePolygons',
        variant: `convex-merge-${pass}`,
        invocationCount: polygonCount,
        bindings: [
          {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
          {name: 'indices', view: indices, type: 'u32', access: 'read'},
          {name: 'consumed', view: consumed, type: 'u32', access: 'read_write'},
          {name: 'info', view: info, type: 'u32', access: 'read'},
          {name: 'triangleOffsets', view: triangleOffsets, type: 'u32', access: 'read'},
          ...(pass === 'count'
            ? [
                {
                  name: 'pieceVertexCounts',
                  view: pieceVertexCounts,
                  type: 'u32' as const,
                  access: 'read_write' as const
                }
              ]
            : [
                {
                  name: 'pieceOffsets',
                  view: pieceOffsets,
                  type: 'u32' as const,
                  access: 'read' as const
                },
                {
                  name: 'outPositions',
                  view: props.output.positions,
                  type: 'f32' as const,
                  access: 'read_write' as const
                },
                {
                  name: 'pieces',
                  view: props.output.pieces,
                  type: 'u32' as const,
                  access: 'read_write' as const
                }
              ])
        ],
        declarations: `${EXACT_ORIENTATION_WGSL}
const MAXIMUM_VERTICES: u32 = ${maximumVertices}u;
const PIECE_CAPACITY: u32 = ${props.output.pieces.length}u;
${getReadPointWGSL('point', 'positions')}
fn triangle(row: u32) -> vec3u { return vec3u(indices[indicesOffset + row * 3u] & 0x7fffffffu, indices[indicesOffset + row * 3u + 1u], indices[indicesOffset + row * 3u + 2u]); }
fn turn(first: u32, second: u32, third: u32) -> i32 {
  return orientSign(point(first), point(second), point(third));
}`,
        body: `let polygon = index;
if (info[infoOffset + polygon * 2u] != 0u) { return; }
let first = triangleOffsets[triangleOffsetsOffset + polygon];
let last = triangleOffsets[triangleOffsetsOffset + polygon + 1u];
let feature = info[infoOffset + polygon * 2u + 1u];
for (var seed = first; seed < last; seed++) {
  if (consumed[consumedOffset + seed] != 0u) { continue; }
    let initial = triangle(seed); let orientation = turn(initial.x, initial.y, initial.z);
    if (orientation == 0) { continue; }
    var boundary: array<u32, MAXIMUM_VERTICES>;
    boundary[0] = initial.x; boundary[1] = initial.y; boundary[2] = initial.z;
    var vertexCount = 3u;
    consumed[consumedOffset + seed] = 1u;
    loop {
      if (vertexCount >= MAXIMUM_VERTICES) { break; }
      var merged = false;
      for (var candidate = first; candidate < last; candidate++) {
        if (consumed[consumedOffset + candidate] != 0u) { continue; }
        let current = triangle(candidate);
        if (turn(current.x, current.y, current.z) == 0) { continue; }
        for (var edge = 0u; edge < vertexCount; edge++) {
          for (var side = 0u; side < 3u; side++) {
            if (boundary[edge] != current[(side + 1u) % 3u] || boundary[(edge + 1u) % vertexCount] != current[side]) { continue; }
            let inserted = current[(side + 2u) % 3u];
            var proposed: array<u32, MAXIMUM_VERTICES>;
            for (var vertex = 0u; vertex <= vertexCount; vertex++) {
              if (vertex <= edge) { proposed[vertex] = boundary[vertex]; }
              else if (vertex == edge + 1u) { proposed[vertex] = inserted; }
              else { proposed[vertex] = boundary[vertex - 1u]; }
            }
            var convex = true;
            for (var vertex = 0u; vertex <= vertexCount; vertex++) {
              let direction = turn(proposed[vertex], proposed[(vertex + 1u) % (vertexCount + 1u)], proposed[(vertex + 2u) % (vertexCount + 1u)]);
              if (direction == 2 || direction * orientation < 0) { convex = false; }
              if (vertex != edge + 1u && proposed[vertex] == inserted) { convex = false; }
            }
            if (!convex) { continue; }
            for (var vertex = 0u; vertex <= vertexCount; vertex++) { boundary[vertex] = proposed[vertex]; }
            vertexCount++; consumed[consumedOffset + candidate] = 1u; merged = true; break;
          }
          if (merged) { break; }
        }
        if (merged) { break; }
      }
      if (!merged) { break; }
    }
  ${
    pass === 'count'
      ? 'pieceVertexCounts[pieceVertexCountsOffset + seed] = vertexCount;'
      : `let required = pieceOffsets[pieceOffsetsOffset + seed];
  if (required < PIECE_CAPACITY) {
    pieces[piecesOffset + required * 4u] = vertexCount;
    pieces[piecesOffset + required * 4u + 1u] = feature;
    pieces[piecesOffset + required * 4u + 2u] = polygon;
    pieces[piecesOffset + required * 4u + 3u] = seed;
    for (var vertex = 0u; vertex < vertexCount; vertex++) {
      let value = point(boundary[vertex]); let destination = (required * MAXIMUM_VERTICES + vertex) * 2u;
      outPositions[outPositionsOffset + destination] = value.x; outPositions[outPositionsOffset + destination + 1u] = value.y;
    }
  }`
  }
}`
      });
    return [
      ...new GPUPolygonTriangulation({
        ...props,
        id: `${id}-triangulation`,
        output: {indices, valid, triangleOffsets}
      }).getCommandNodes(graph),
      createZeroPiecesNode(graph, id, props.output.pieces),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-polygon-info`,
        operation: 'GPUSubdividePolygons',
        variant: 'polygon-info',
        invocationCount: polygonCount,
        bindings: [
          {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
          {name: 'indices', view: indices, type: 'u32', access: 'read'},
          {name: 'valid', view: valid, type: 'u32', access: 'read'},
          {name: 'triangleOffsets', view: triangleOffsets, type: 'u32', access: 'read'},
          {name: 'featureOffsets', view: props.featureOffsets, type: 'u32', access: 'read'},
          {name: 'info', view: info, type: 'u32', access: 'read_write'}
        ],
        declarations: `${EXACT_ORIENTATION_WGSL}
${getReadPointWGSL('point', 'positions')}`,
        // Per-polygon triangle orientation checks and feature lookup run in parallel; the merge
        // below only consumes the result.
        body: `var rejected = valid[validOffset + index] == 0u;
if (!rejected) {
  for (var candidate = triangleOffsets[triangleOffsetsOffset + index]; candidate < triangleOffsets[triangleOffsetsOffset + index + 1u]; candidate++) {
    let current = vec3u(indices[indicesOffset + candidate * 3u] & 0x7fffffffu, indices[indicesOffset + candidate * 3u + 1u], indices[indicesOffset + candidate * 3u + 2u]);
    if (orientSign(point(current.x), point(current.y), point(current.z)) == 2) { rejected = true; }
  }
}
// Largest feature whose first polygon is at or before this polygon (feature 0 is the default).
var low = 1u; var high = ${props.featureOffsets.length}u;
while (low < high) {
  let middle = (low + high) >> 1u;
  if (featureOffsets[featureOffsetsOffset + middle] <= index) { low = middle + 1u; } else { high = middle; }
}
info[infoOffset + index * 2u] = select(0u, 1u, rejected);
info[infoOffset + index * 2u + 1u] = low - 1u;`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-consumed-reset`,
        operation: 'GPUSubdividePolygons',
        variant: 'convex-consumed-reset',
        invocationCount: triangleSlotCount,
        bindings: [
          {name: 'indices', view: indices, type: 'u32', access: 'read'},
          {name: 'consumed', view: consumed, type: 'u32', access: 'read_write'},
          {name: 'pieceVertexCounts', view: pieceVertexCounts, type: 'u32', access: 'read_write'}
        ],
        // A triangle the triangulation flagged (bit 31) is never a seed or a merge candidate.
        body: `consumed[consumedOffset + index] = indices[indicesOffset + index * 3u] >> 31u;
pieceVertexCounts[pieceVertexCountsOffset + index] = 0u;`
      }),
      getMergePass('count'),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-piece-flags`,
        operation: 'GPUSubdividePolygons',
        variant: 'convex-piece-flags',
        invocationCount: triangleSlotCount,
        bindings: [
          {name: 'pieceVertexCounts', view: pieceVertexCounts, type: 'u32', access: 'read'},
          {name: 'pieceFlags', view: pieceFlags, type: 'u32', access: 'read_write'}
        ],
        body: 'pieceFlags[pieceFlagsOffset + index] = select(0u, 1u, pieceVertexCounts[pieceVertexCountsOffset + index] != 0u);'
      }),
      ...new GPUScan({
        id: `${id}-piece-scan`,
        input: pieceFlags,
        output: pieceOffsets,
        mode: 'exclusive'
      }).getCommandNodes(graph),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-consumed-reset-write`,
        operation: 'GPUSubdividePolygons',
        variant: 'convex-consumed-reset',
        invocationCount: triangleSlotCount,
        bindings: [
          {name: 'indices', view: indices, type: 'u32', access: 'read'},
          {name: 'consumed', view: consumed, type: 'u32', access: 'read_write'},
          {name: 'pieceVertexCounts', view: pieceVertexCounts, type: 'u32', access: 'read_write'}
        ],
        body: `consumed[consumedOffset + index] = indices[indicesOffset + index * 3u] >> 31u;
pieceVertexCounts[pieceVertexCountsOffset + index] = 0u;`
      }),
      getMergePass('write'),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-merge-status`,
        operation: 'GPUSubdividePolygons',
        variant: 'convex-merge-status',
        invocationCount: 1,
        bindings: [
          {name: 'pieceFlags', view: pieceFlags, type: 'u32', access: 'read'},
          {name: 'pieceOffsets', view: pieceOffsets, type: 'u32', access: 'read'},
          {name: 'status', view: props.output.status, type: 'u32', access: 'read_write'}
        ],
        declarations: `const PIECE_CAPACITY: u32 = ${props.output.pieces.length}u;`,
        body: `let required = pieceOffsets[pieceOffsetsOffset + ${triangleSlotCount - 1}u] + pieceFlags[pieceFlagsOffset + ${triangleSlotCount - 1}u];
status[statusOffset] = min(required, PIECE_CAPACITY); status[statusOffset + 1u] = required;
status[statusOffset + 2u] = select(0u, 1u, required > PIECE_CAPACITY); status[statusOffset + 3u] = 0u;`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-invalid-count`,
        operation: 'GPUSubdividePolygons',
        variant: 'convex-invalid-count',
        invocationCount: polygonCount,
        bindings: [
          {name: 'info', view: info, type: 'u32', access: 'read'},
          {name: 'status', view: props.output.status, type: 'atomic<u32>', access: 'read_write'}
        ],
        body: 'if (info[infoOffset + index * 2u] != 0u) { atomicAdd(&status[statusOffset + 3u], 1u); }'
      })
    ];
  }
}

/** Zeroes the piece table in parallel before a serial merge or queue kernel fills it. @internal */
export function createZeroPiecesNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  pieces: GraphDataView<'uint32x4'>
): GPUCommandNode<Parameters> {
  return createWGSLKernelNode<Parameters>(graph, {
    id: `${id}-zero-pieces`,
    operation: 'GPUSubdividePolygons',
    variant: 'zero-pieces',
    invocationCount: pieces.length * 4,
    bindings: [{name: 'pieces', view: pieces, type: 'u32', access: 'read_write'}],
    body: 'pieces[piecesOffset + index] = 0u;'
  });
}

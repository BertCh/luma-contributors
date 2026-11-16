// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPU_LONGITUDE_LATITUDE_SPATIAL_CONTEXT,
  type GPUSpatialContext,
  type GPUSpatialEdgeModel
} from '../../analysis-core/contracts/spatial-context';
import {
  createGPUComputeCommandNode,
  getBoundedDispatchLayout,
  getViewBinding,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView,
  type GraphVectorView,
  type GPUCommandNodeProducer
} from '@luma.gl/graph/gpu-core';
import {Kernel} from '@luma.gl/graph';
import {dggs} from '@luma.gl/shaders';

import type {WGSLKernelBinding} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  createGraphChunkDispatchParameters,
  planGraphChunkDispatch,
  validateDisjointWritableViews,
  validateGraphChunkBindingRanges,
  validateGraphChunkDispatchBindings,
  type GraphChunkDispatchPlan
} from '../../analysis-core/kernels/graph-chunk-dispatch';
import {
  validateGraphOutputsDisjointFromInputs,
  getGraphViewChunks,
  validateCapacity,
  validateCommandNodeBudget,
  validateMaximumCount,
  validateRequirement,
  validateSinglePackedView
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {CELL_KEY_WGSL} from '../cell-aggregation/cell-keys';
import {H3_BOUNDARY_WGSL} from './h3-boundary-wgsl';
import type {GPUCellWordOrder} from '../cell-aggregation/index';
import type {GPUCellIndexFamily} from './cell-index-families';

const OPERATION = 'GPUCellGeometry';

/** Families {@link GPUCellGeometry} decodes: the indexing families plus A5. */
export type GPUCellGeometryFamily = GPUCellIndexFamily | 'a5';

/**
 * Boundary vertices each family produces: 4 for the rectangular families (Quadbin, quadkey,
 * geohash, S2), 6 for H3 hexagons and 5 for A5 pentagons. H3 cells that cross an icosahedron edge
 * and pentagons have up to 10 vertices: use `maximumVertexCount: 10` for H3 to keep them all
 * (see {@link GPU_CELL_GEOMETRY_H3_MAXIMUM_VERTEX_COUNT}); the 6 here is the hexagon minimum.
 */
export const GPU_CELL_GEOMETRY_VERTEX_COUNTS: Readonly<Record<GPUCellGeometryFamily, number>> = {
  quadbin: 4,
  h3: 6,
  quadkey: 4,
  geohash: 4,
  s2: 4,
  a5: 5
};

/** Most boundary vertices an H3 cell has (distortion vertices included): use as `maximumVertexCount`. */
export const GPU_CELL_GEOMETRY_H3_MAXIMUM_VERTEX_COUNT = 10;

/** Largest accepted `maximumVertexCount`. */
const MAXIMUM_VERTEX_COUNT_LIMIT = 16;

/** Storage array component indices are uint32, including each fixed-stride boundary record. */
const MAXIMUM_CHUNK_COMPONENT_COUNT = 0xffffffff;
/** Fixed workgroup size keeps the shader and pipeline independent of chunk lengths. */
const CHUNK_WORKGROUP_SIZE = 256;
/** Guard before multiplying the workgroup index, so padded 3D dispatches cannot wrap uint32. */
const MAXIMUM_LINEAR_WORKGROUP_COUNT = 16777216;

/** Caller-owned outputs of {@link GPUCellGeometry}, all row-aligned with the input cells. */
export type GPUCellGeometryOutput = {
  /**
   * One longitude/latitude pair per cell. Invalid, zero and unsupported cells get `NaN` in both
   * components.
   */
  centers?: GraphDataView<'float32x2'> | GraphVectorView<'float32x2'>;
  /**
   * `cells.length * maximumVertexCount` longitude/latitude pairs; cell `row` owns pairs
   * `[row * maximumVertexCount, (row + 1) * maximumVertexCount)`. Pairs at and beyond the cell's
   * vertex count are zero.
   */
  boundaries?: GraphDataView<'float32x2'> | GraphVectorView<'float32x2'>;
  /**
   * One vertex count per cell: `GPU_CELL_GEOMETRY_VERTEX_COUNTS[family]`, or `0` for invalid
   * cells. H3 counts follow h3-js `cellToBoundary` (6, 5 for Class II pentagons, up to 10 with
   * distortion vertices) and are clamped to `maximumVertexCount`. Requires `boundaries`.
   */
  vertexCounts?: GraphDataView<'uint32'> | GraphVectorView<'uint32'>;
  /**
   * One flag per cell: 1 when the key is a valid cell of the family, 0 otherwise (the zero key
   * included), the convention of published `GPUDGGSCellProjection` (`uint32`, nonzero is valid).
   * It is the test that decides the NaN centres and zero vertex counts, so those outputs are the
   * same with or without it.
   */
  validity?: GraphDataView<'uint32'> | GraphVectorView<'uint32'>;
};

/**
 * Properties for {@link GPUCellGeometry}.
 *
 * Per-frame (no recompile): the contents of `cells`. Topology (needs a new graph): `family`,
 * `wordOrder`, `maximumVertexCount`, the row count and which outputs are present.
 */
export type GPUCellGeometryProps = {
  /** Prefix for generated node IDs. Defaults to `'cell-geometry'`. */
  id?: string;
  /** Family of the input keys. */
  family: GPUCellGeometryFamily;
  /** One 64-bit key per row as two `uint32` words. */
  cells: GraphDataView<'uint32x2'> | GraphVectorView<'uint32x2'>;
  /** Word order of `cells`. Defaults to `'little-endian'` (`(low, high)`, Arrow layout). */
  wordOrder?: GPUCellWordOrder;
  /**
   * Fixed boundary stride per cell, at least `GPU_CELL_GEOMETRY_VERTEX_COUNTS[family]` and at most
   * 16. Defaults to the family's vertex count. Required to be consistent with `output.boundaries`.
   * H3 cells have up to 10 vertices; a smaller stride truncates them (the count is clamped).
   */
  maximumVertexCount?: number;
  /** Caller-owned outputs; at least one of `centers` and `boundaries`. */
  output: GPUCellGeometryOutput;
};

/**
 * Decodes 64-bit cell keys to centers and boundary polygons in longitude/latitude degrees.
 *
 * One compute kernel per encoding reads each key (little-endian `(low, high)` words unless
 * `wordOrder` says otherwise) and writes row-aligned `centers`, fixed-stride `boundaries` and
 * `vertexCounts`. It reuses the `@luma.gl/shaders` `dggs` decoders for H3
 * (`dggs_h3_get_center_lnglat`, boundary vertices), S2, geohash, quadkey and A5, plus a small
 * Quadbin bit-layout decoder. Nothing here compiles, submits or reads back.
 *
 * Conventions:
 * - Rectangular families list the vertices clockwise from the north-west corner (Quadbin, quadkey
 *   and geohash: NW, NE, SE, SW) or as `(i, j), (i, j+1), (i+1, j+1), (i+1, j)` face corners (S2).
 *   Quadbin and quadkey centers are the Web Mercator tile center (the middle of the tile in
 *   projected space, not the mean latitude); geohash centers are the middle of the bounds; S2
 *   centers are the face `(i + 0.5, j + 0.5)` point. Edges are straight in lng/lat for the
 *   rectangular families; S2 and H3 and A5 vertices are the exact great-circle corner positions.
 * - H3 boundaries use a WGSL port of H3's `_faceIjkToCellBoundary` and
 *   `_faceIjkPentToCellBoundary` ({@link H3_BOUNDARY_WGSL}): every valid cell gets a boundary
 *   (hexagons, pentagons, cells across icosahedron edges, with the extra edge-crossing vertices)
 *   in h3-js `cellToBoundary` order, up to 10 vertices. Set `maximumVertexCount: 10` to keep them.
 * - A5 boundaries have five vertices (three at resolution 1), in the order of the dggs decoder.
 * - Invalid keys (including the zero key): center `NaN`, vertex count 0, boundary rows zero, and
 *   0 in the optional `output.validity` (1 for a valid cell). Published `GPUDGGSCellProjection`
 *   decodes H3 and A5 centres with the same shader functions and the same validity test; it
 *   writes a zero centre for an invalid cell where this class writes `NaN`.
 * - Precision is f32: Quadbin and quadkey coordinates lose precision above zoom 24 (the tile
 *   column itself exceeds the f32 mantissa), H3, S2 and A5 inherit the accuracy of WGSL `sin`,
 *   `cos` and `atan2`, which the WGSL spec leaves implementation defined.
 *
 * Packed or chunked inputs are supported. Row-aligned outputs mirror the input chunks; each
 * boundary chunk holds whole fixed-stride cell records. Outputs must not share buffers with `cells`.
 */
export class GPUCellGeometry implements GPUCommandNodeProducer {
  /** Prefix for every node ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUCellGeometryProps;
  /** Boundary stride per cell. */
  readonly maximumVertexCount: number;
  /** Boundary edges: great-circle arcs for H3, S2 and A5; linear in degrees otherwise. */
  readonly edgeModel: GPUSpatialEdgeModel;

  /**
   * Context of the decoded `centers` and `boundaries` (longitude/latitude, no metric), including
   * the family's `edgeModel`. The class has no `spatialContext` prop: its input is cell identifiers,
   * which carry no coordinates.
   */
  readonly outputSpatialContext: GPUSpatialContext;

  /** Upload plus one decode dispatch per nonempty chunk. */
  readonly estimatedCommandNodeCount: number;
  private readonly chunkPlan: GraphChunkDispatchPlan;
  private readonly boundaryChunks: readonly GraphDataView[];

  constructor(props: GPUCellGeometryProps) {
    this.id = props.id ?? 'cell-geometry';
    this.props = props;
    const id = this.id;
    validateRequirement(
      id,
      Object.prototype.hasOwnProperty.call(GPU_CELL_GEOMETRY_VERTEX_COUNTS, props.family),
      `family must be one of ${Object.keys(GPU_CELL_GEOMETRY_VERTEX_COUNTS).join(', ')}`
    );
    this.edgeModel =
      props.family === 'h3' || props.family === 's2' || props.family === 'a5'
        ? 'great-circle'
        : 'linear';
    this.outputSpatialContext = {
      ...GPU_LONGITUDE_LATITUDE_SPATIAL_CONTEXT,
      edgeModel: this.edgeModel
    };
    validateRequirement(
      id,
      props.wordOrder === undefined || ['little-endian', 'high-low'].includes(props.wordOrder),
      "wordOrder must be 'little-endian' or 'high-low'"
    );
    const required = GPU_CELL_GEOMETRY_VERTEX_COUNTS[props.family];
    this.maximumVertexCount = props.maximumVertexCount ?? required;
    validateCapacity(
      id,
      'maximumVertexCount',
      this.maximumVertexCount,
      required,
      MAXIMUM_VERTEX_COUNT_LIMIT
    );
    const {output} = props;
    validateRequirement(
      id,
      !output.vertexCounts || Boolean(output.boundaries),
      'output.vertexCounts needs output.boundaries'
    );
    validateRequirement(
      id,
      Boolean(output.centers || output.boundaries),
      'output needs centers or boundaries'
    );
    this.chunkPlan = planGraphChunkDispatch(id, {
      cells: props.cells,
      centers: output.centers,
      vertexCounts: output.vertexCounts,
      validity: output.validity
    });
    validateCapacity(id, 'cells row count', this.chunkPlan.totalLength);
    for (const chunk of this.chunkPlan.chunks) {
      validateSinglePackedView(
        chunk.views['cells'],
        ['uint32x2'],
        `${id} cells chunk ${chunk.chunkIndex}`
      );
      for (const [name, format] of [
        ['centers', 'float32x2'],
        ['vertexCounts', 'uint32'],
        ['validity', 'uint32']
      ] as const) {
        const view = chunk.views[name];
        if (view)
          validateSinglePackedView(
            view,
            [format],
            `${id} output.${name} chunk ${chunk.chunkIndex}`
          );
      }
      validateMaximumCount(
        id,
        `cells chunk ${chunk.chunkIndex} components`,
        chunk.length * 2,
        MAXIMUM_CHUNK_COMPONENT_COUNT
      );
    }
    this.boundaryChunks = output.boundaries ? getGraphViewChunks(output.boundaries) : [];
    if (output.boundaries) {
      validateRequirement(
        id,
        output.boundaries.length === this.chunkPlan.totalLength * this.maximumVertexCount,
        'output.boundaries length must equal cells rows * maximumVertexCount'
      );
      validateRequirement(
        id,
        this.boundaryChunks.length === this.chunkPlan.chunks.length,
        'output.boundaries must match cells chunk topology'
      );
      for (const chunk of this.chunkPlan.chunks) {
        const boundaries = this.boundaryChunks[chunk.chunkIndex];
        validateSinglePackedView(
          boundaries,
          ['float32x2'],
          `${id} output.boundaries chunk ${chunk.chunkIndex}`
        );
        validateRequirement(
          id,
          boundaries.length === chunk.length * this.maximumVertexCount,
          `output.boundaries chunk ${chunk.chunkIndex} length must equal cells chunk rows * maximumVertexCount (${chunk.length * this.maximumVertexCount})`
        );
        validateMaximumCount(
          id,
          `output.boundaries chunk ${chunk.chunkIndex} components`,
          boundaries.length * 2,
          MAXIMUM_CHUNK_COMPONENT_COUNT
        );
      }
    }
    validateDisjointWritableViews(id, output);
    this.estimatedCommandNodeCount = 1 + this.chunkPlan.nonemptyChunkCount;
    validateCommandNodeBudget(id, this.estimatedCommandNodeCount);
    validateGraphOutputsDisjointFromInputs(
      id,
      [output.centers, output.boundaries, output.vertexCounts, output.validity],
      [props.cells]
    );
  }

  /** Uploads immutable chunk parameters and decodes each nonempty chunk. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, maximumVertexCount} = this;
    const {family, output} = props;
    validateMaximumCount(
      id,
      'storage bindings',
      2 + Object.values(output).filter(Boolean).length,
      graph.device.limits.maxStorageBuffersPerShaderStage
    );
    validateGraphChunkDispatchBindings(graph, id, this.chunkPlan);
    validateGraphChunkBindingRanges(graph, id, {boundaries: output.boundaries});
    // Resolve every dispatch before the upload helper declares its parameter scratch.
    const layouts = this.chunkPlan.chunks.map(chunk =>
      getBoundedDispatchLayout(
        id,
        Math.max(chunk.length, 1),
        CHUNK_WORKGROUP_SIZE,
        graph.device.limits.maxComputeWorkgroupsPerDimension
      )
    );
    const {slots, nodes} = createGraphChunkDispatchParameters(graph, {
      id: `${id}-chunks`,
      plan: this.chunkPlan
    });
    // Every chunk closes over the same source string, including assembled fp64 modules.
    let sharedShaderSource: string | undefined;
    const chunkNodes = this.chunkPlan.chunks
      .filter(chunk => chunk.length > 0)
      .map(chunk => {
        const views = chunk.views;
        const bindings: WGSLKernelBinding[] = [
          {name: 'chunkParameters', view: slots[chunk.chunkIndex], type: 'u32', access: 'read'},
          {name: 'cells', view: views['cells'], type: 'u32', access: 'read'}
        ];
        if (output.centers)
          bindings.push({
            name: 'centersOut',
            view: views['centers'],
            type: 'f32',
            access: 'read_write'
          });
        if (output.boundaries)
          bindings.push({
            name: 'boundariesOut',
            view: this.boundaryChunks[chunk.chunkIndex],
            type: 'f32',
            access: 'read_write'
          });
        if (output.vertexCounts)
          bindings.push({
            name: 'vertexCountsOut',
            view: views['vertexCounts'],
            type: 'u32',
            access: 'read_write'
          });
        if (output.validity)
          bindings.push({
            name: 'validityOut',
            view: views['validity'],
            type: 'u32',
            access: 'read_write'
          });
        const centerBody = output.centers
          ? `let center = select(vec2f(getNotANumber(index)), geometryGetCenter(key), isValid);
  centersOut[centersOutOffset + 2u * index] = center.x;
  centersOut[centersOutOffset + 2u * index + 1u] = center.y;`
          : '';
        const boundaryBody = output.boundaries
          ? `var points: array<vec2f, 16>;
  var vertexCount = 0u;
  if (isValid) {
    vertexCount = geometryFillBoundary(key, &points);
  }
  vertexCount = min(vertexCount, MAXIMUM_VERTEX_COUNT);
  for (var vertex = 0u; vertex < MAXIMUM_VERTEX_COUNT; vertex++) {
    let point = points[vertex];
    let slot = (index * MAXIMUM_VERTEX_COUNT + vertex) * 2u;
    boundariesOut[boundariesOutOffset + slot] = point.x;
    boundariesOut[boundariesOutOffset + slot + 1u] = point.y;
  }
  ${output.vertexCounts ? 'vertexCountsOut[vertexCountsOutOffset + index] = vertexCount;' : ''}`
          : '';
        const declarations = bindings
          .map(
            (binding, location) =>
              `@group(0) @binding(${location}) var<storage, ${binding.access}> ${binding.name}: array<${binding.type}>;`
          )
          .join('\n');
        // getViewBinding exposes exactly prefix + logical components. Subtracting the chunk's
        // logical size recovers any uint32-aligned view offset without specializing the shader.
        const source =
          sharedShaderSource ??
          `${declarations}
const MAXIMUM_VERTEX_COUNT: u32 = ${maximumVertexCount}u;
fn getNotANumber(runtimeZero: u32) -> f32 {
  return bitcast<f32>(0x7fc00000u | (runtimeZero & 0u));
}
${dggs.source}
${CELL_KEY_WGSL}
${CELL_GEOMETRY_COMMON_WGSL}
${FAMILY_GEOMETRY_WGSL[family]}
${family === 'h3' ? '' : GENERIC_FILL_BOUNDARY_WGSL}
@compute @workgroup_size(${CHUNK_WORKGROUP_SIZE})
fn main(
  @builtin(workgroup_id) workgroupId: vec3u,
  @builtin(num_workgroups) workgroupCount: vec3u,
  @builtin(local_invocation_index) localInvocationIndex: u32
) {
  let workgroupIndex = (workgroupId.z * workgroupCount.y + workgroupId.y) * workgroupCount.x + workgroupId.x;
  if (workgroupIndex >= ${MAXIMUM_LINEAR_WORKGROUP_COUNT}u) { return; }
  let index = workgroupIndex * ${CHUNK_WORKGROUP_SIZE}u + localInvocationIndex;
  let rowCount = chunkParameters[1];
  if (index >= rowCount) { return; }
  let cellsOffset = arrayLength(&cells) - rowCount * 2u;
  ${output.centers ? 'let centersOutOffset = arrayLength(&centersOut) - rowCount * 2u;' : ''}
  ${output.boundaries ? 'let boundariesOutOffset = arrayLength(&boundariesOut) - rowCount * MAXIMUM_VERTEX_COUNT * 2u;' : ''}
  ${output.vertexCounts ? 'let vertexCountsOutOffset = arrayLength(&vertexCountsOut) - rowCount;' : ''}
  ${output.validity ? 'let validityOutOffset = arrayLength(&validityOut) - rowCount;' : ''}
  let words = vec2u(cells[cellsOffset + 2u * index], cells[cellsOffset + 2u * index + 1u]);
  let key = ${props.wordOrder === 'high-low' ? 'words' : 'words.yx'};
  let isValid = geometryIsValid(key);
  ${output.validity ? 'validityOut[validityOutOffset + index] = select(0u, 1u, isValid);' : ''}
  ${centerBody}
  ${boundaryBody}
}`;
        sharedShaderSource = source;
        const layout = layouts[chunk.chunkIndex];
        return createGPUComputeCommandNode<Parameters>({
          id: `${id}-geometry-${chunk.chunkIndex}`,
          resources: bindings.map(binding => ({
            buffer: binding.view,
            usage: binding.access === 'read' ? 'storage-read' : 'storage-read-write'
          })),
          workload: {
            operation: OPERATION,
            variant: family,
            commandCount: 1,
            maximumWorkgroupCount: layout.x * layout.y * layout.z,
            maximumInvocationCount: chunk.length
          },
          compile: ({device}) => {
            const kernel = new Kernel(device, {
              id: `${OPERATION}-${family}-geometry`,
              source,
              shaderLayout: {
                bindings: bindings.map((binding, location) => ({
                  name: binding.name,
                  group: 0,
                  location,
                  type: binding.access === 'read' ? 'read-only-storage' : 'storage'
                }))
              }
            });
            return {
              encode: ({computePass, getBuffer}) =>
                kernel.dispatch(computePass, {
                  bindings: Object.fromEntries(
                    bindings.map(binding => [binding.name, getViewBinding(binding.view, getBuffer)])
                  ),
                  ...layout
                }),
              destroy: () => kernel.destroy()
            };
          }
        });
      });
    return [...nodes, ...chunkNodes];
  }
}

/** Web Mercator tile helpers shared by the Quadbin and quadkey decoders. */
const CELL_GEOMETRY_COMMON_WGSL = /* wgsl */ `
fn geometryGetTileBounds(tile: vec3u) -> vec4f {
  let tileScale = f32(1u << tile.z);
  let west = f32(tile.x) / tileScale * 360.0 - 180.0;
  let east = f32(tile.x + 1u) / tileScale * 360.0 - 180.0;
  let north = dggs_web_mercator_tile_y_to_latitude(f32(tile.y), tileScale);
  let south = dggs_web_mercator_tile_y_to_latitude(f32(tile.y + 1u), tileScale);
  return vec4f(west, south, east, north);
}

fn geometryGetTileCenter(tile: vec3u) -> vec2f {
  let tileScale = f32(1u << tile.z);
  return vec2f(
    (f32(tile.x) + 0.5) / tileScale * 360.0 - 180.0,
    dggs_web_mercator_tile_y_to_latitude(f32(tile.y) + 0.5, tileScale)
  );
}

fn geometryGetRectangleVertex(bounds: vec4f, vertex: u32) -> vec2f {
  return dggs_bounds_get_boundary_point(bounds, vertex);
}

/** Even bits of a 32-bit value packed into 16 bits (inverse of the Morton spread). */
fn geometryCompactEvenBits(value: u32) -> u32 {
  var x = value & 0x55555555u;
  x = (x | (x >> 1u)) & 0x33333333u;
  x = (x | (x >> 2u)) & 0x0f0f0f0fu;
  x = (x | (x >> 4u)) & 0x00ff00ffu;
  x = (x | (x >> 8u)) & 0x0000ffffu;
  return x;
}
`;

const QUADBIN_GEOMETRY_WGSL = /* wgsl */ `
fn geometryGetQuadbinTile(key: vec2u) -> vec3u {
  let resolution = (key.x >> 20u) & 0x1fu;
  let morton = cellGetCompactKey(key, 52u - 2u * resolution, 2u * resolution);
  let x = geometryCompactEvenBits(morton.y) | (geometryCompactEvenBits(morton.x) << 16u);
  let y = geometryCompactEvenBits(morton.y >> 1u) | (geometryCompactEvenBits(morton.x >> 1u) << 16u);
  return vec3u(x, y, resolution);
}
fn geometryIsValid(key: vec2u) -> bool { return cellIsValidQuadbin(key); }
fn geometryGetCenter(key: vec2u) -> vec2f { return geometryGetTileCenter(geometryGetQuadbinTile(key)); }
fn geometryGetVertexCount(key: vec2u) -> u32 { return 4u; }
fn geometryGetVertex(key: vec2u, vertex: u32) -> vec2f {
  return geometryGetRectangleVertex(geometryGetTileBounds(geometryGetQuadbinTile(key)), vertex);
}
`;

const QUADKEY_GEOMETRY_WGSL = /* wgsl */ `
fn geometryIsValid(key: vec2u) -> bool {
  let length = key.x >> 26u;
  // Digits fill the low 2 * length bits; every bit between them and the length field is zero.
  let unused = cellMaskLow(58u) & ~cellMaskLow(2u * length);
  return length >= 1u && length <= DGGS_QUADKEY_MAX_LENGTH && all((key & unused) == vec2u(0u));
}
fn geometryGetCenter(key: vec2u) -> vec2f { return geometryGetTileCenter(dggs_quadkey_get_tile(key)); }
fn geometryGetVertexCount(key: vec2u) -> u32 { return 4u; }
fn geometryGetVertex(key: vec2u, vertex: u32) -> vec2f {
  return geometryGetRectangleVertex(dggs_quadkey_get_bounds(key), vertex);
}
`;

const GEOHASH_GEOMETRY_WGSL = /* wgsl */ `
fn geometryIsValid(key: vec2u) -> bool {
  let length = key.x >> 28u;
  let unused = cellMaskLow(60u) & ~cellMaskLow(5u * length);
  return length >= 1u && length <= DGGS_GEOHASH_MAX_LENGTH && all((key & unused) == vec2u(0u));
}
fn geometryGetCenter(key: vec2u) -> vec2f {
  let bounds = dggs_geohash_get_bounds(key);
  return vec2f(0.5 * (bounds.x + bounds.z), 0.5 * (bounds.y + bounds.w));
}
fn geometryGetVertexCount(key: vec2u) -> u32 { return 4u; }
fn geometryGetVertex(key: vec2u, vertex: u32) -> vec2f {
  return dggs_geohash_get_boundary_point(key, vertex);
}
`;

/** S2 geometry shared by outlines; requires dggs.source. @internal */
export const S2_GEOMETRY_WGSL = /* wgsl */ `
// dggs_s2_xyz_to_lnglat calls atan2 with a zero argument at the S2 face centers: atan2(0, 0) at the
// poles (faces 2 and 5) is undefined in WGSL, and on Metal atan2(1, 0) returned -pi/2 for the face
// 1 center (x = -u = -0). Exact zeros are therefore handled without atan2. Upstream
// dggs_s2_xyz_to_lnglat (a plain atan2) is not used: it changes the values on the axes.
fn geometryS2XyzToLngLat(xyz: vec3f) -> vec2f {
  let horizontal = length(xyz.xy);
  var longitude = 0.0;
  if (xyz.x == 0.0) {
    longitude = select(0.0, select(-90.0, 90.0, xyz.y > 0.0), xyz.y != 0.0);
  } else if (xyz.y == 0.0) {
    longitude = select(0.0, 180.0, xyz.x < 0.0);
  } else {
    longitude = atan2(xyz.y, xyz.x) * DGGS_RADIANS_TO_DEGREES;
  }
  return vec2f(longitude, atan2(xyz.z, horizontal) * DGGS_RADIANS_TO_DEGREES);
}
fn geometryS2GetPoint(key: vec2u, offset: vec2f) -> vec2f {
  let st = dggs_s2_ij_to_st(dggs_s2_get_ij(key), dggs_s2_get_level(key), offset);
  return geometryS2XyzToLngLat(dggs_s2_face_uv_to_xyz(dggs_s2_get_face(key), dggs_s2_st_to_uv(st)));
}
fn geometryIsValid(key: vec2u) -> bool { return dggs_s2_is_valid_cell_id(key); }
fn geometryGetCenter(key: vec2u) -> vec2f { return geometryS2GetPoint(key, vec2f(0.5)); }
fn geometryGetVertexCount(key: vec2u) -> u32 { return 4u; }
fn geometryGetVertex(key: vec2u, vertex: u32) -> vec2f {
  return geometryS2GetPoint(key, dggs_s2_get_boundary_offset(vertex));
}
`;

const H3_GEOMETRY_WGSL = /* wgsl */ `
${H3_BOUNDARY_WGSL}
fn geometryIsValid(key: vec2u) -> bool { return dggs_h3_is_valid_cell_id(key); }
fn geometryGetCenter(key: vec2u) -> vec2f { return dggs_h3_get_center_lnglat(key); }
fn geometryFillBoundary(key: vec2u, points: ptr<function, array<vec2f, 16>>) -> u32 {
  let boundary = cellIndexH3GetBoundary(key);
  for (var vertex = 0u; vertex < boundary.count; vertex++) {
    (*points)[vertex] = boundary.points[vertex];
  }
  return boundary.count;
}
`;

/** Fills boundary points one vertex at a time for the families with a closed-form vertex function. */
const GENERIC_FILL_BOUNDARY_WGSL = /* wgsl */ `
fn geometryFillBoundary(key: vec2u, points: ptr<function, array<vec2f, 16>>) -> u32 {
  let count = min(geometryGetVertexCount(key), 16u);
  for (var vertex = 0u; vertex < count; vertex++) {
    (*points)[vertex] = geometryGetVertex(key, vertex);
  }
  return count;
}
`;

const A5_GEOMETRY_WGSL = /* wgsl */ `
fn geometryIsValid(key: vec2u) -> bool {
  return !dggs_u64_is_zero(key) && dggs_a5_deserialize(key).valid != 0u;
}
fn geometryGetCenter(key: vec2u) -> vec2f { return dggs_a5_get_center_lnglat(key); }
fn geometryGetVertexCount(key: vec2u) -> u32 {
  return dggs_a5_get_shape_vertex_count(dggs_a5_deserialize(key));
}
fn geometryGetVertex(key: vec2u, vertex: u32) -> vec2f {
  return dggs_a5_get_boundary_point(key, vertex);
}
`;

const FAMILY_GEOMETRY_WGSL: Record<GPUCellGeometryFamily, string> = {
  quadbin: QUADBIN_GEOMETRY_WGSL,
  quadkey: QUADKEY_GEOMETRY_WGSL,
  geohash: GEOHASH_GEOMETRY_WGSL,
  s2: S2_GEOMETRY_WGSL,
  h3: H3_GEOMETRY_WGSL,
  a5: A5_GEOMETRY_WGSL
};

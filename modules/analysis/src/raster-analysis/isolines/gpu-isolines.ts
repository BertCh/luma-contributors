// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {validateRasterStorageBindings} from '../raster-grid/raster-storage-preflight';
import {
  createTransientView,
  GPUScan,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  createFillNode,
  createWGSLKernelNode,
  createPublishNode,
  type WGSLKernelBinding
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import type {GPUCommandNodeProducer} from '@luma.gl/graph/gpu-core';
import {
  validateGraphViewsBelongToGraph,
  validateGPUStorageBindingByteLength,
  validateRequirement,
  getPackedFloat32Values,
  getPackedUint32Values
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  validateIsolinesTile,
  validateIsolinesMaximumSegmentCount,
  type GPUIsolinesTile
} from './isolines-tile';
import {
  createRasterGatedLoopNodes,
  createRasterIterationResetNode,
  createRasterIterationState
} from '../cost-distance/raster-relaxation';
import {validateRasterGrid} from '../raster-band-utils';
import type {GPURasterBand, GPURasterHaloStage, GPURasterMetadata} from '@luma.gl/graph/gpu-raster';
import {
  getRasterAlgebraInputNodes,
  getRasterAlgebraInputViews,
  getNamedRasterAlgebraInputViews,
  isRasterAlgebraBand,
  validateRasterAlgebraInput,
  validateRasterAlgebraInputBindings
} from '../raster-algebra/raster-algebra-input';
import {GPU_ISOLINES_PARAMETER_LENGTH} from './isolines-parameters';
import {getNamedRasterOutputViews} from '../raster-algebra/raster-algebra-utils';
import {
  getIsolinesCornersWGSL,
  ISOLINES_SADDLE_WGSL,
  validateIsolinesAliasing,
  validateIsolinesView
} from './isolines-utils';

const OPERATION = 'GPUIsolines';

/** Caller-owned, capacity-bounded segment outputs of {@link GPUIsolines}. */
export type GPUIsolinesOutput = {
  /**
   * `[x0, y0, x1, y1]` world coordinates per segment, relative to the parameters' `localOrigin`. The segment capacity is `segments.length`.
   * The high side (`value >= level`) is on the left when walking from p0 to p1 (x right, y up).
   */
  segments: GraphDataView<'float32x4'>;
  /** Level index (row of the `levels` view) per segment. At least `segments.length` rows. */
  segmentLevels: GraphDataView<'uint32'>;
  /**
   * Optional `[startEdge, endEdge]` per segment: the global ids of the raster edges that hold p0
   * and p1. With `tile`, width/height and coordinates below refer to the whole domain. Horizontal edge `(c, r) -> (c + 1, r)` is `r * (width - 1) + c`; vertical edge
   * `(c, r) -> (c, r + 1)` is `height * (width - 1) + r * width + c`.
   */
  segmentEdges?: GraphDataView<'uint32x2'>;
  /** Optional stitched polyline outputs; presence schedules stitching. */
  polylines?: GPUIsolinesPolylineOutput;
  /** One-row status words. */
  status: GPUIsolinesStatus;
};

/** One-row status words of {@link GPUIsolines}. */
export type GPUIsolinesStatus = {
  /** One row receiving `min(requiredCount, segments.length)`. */
  count: GraphDataView<'uint32'>;
  /** Optional one row receiving the unclamped segment count. */
  requiredCount?: GraphDataView<'uint32'>;
  /** One row receiving 1 when segments were dropped for lack of capacity, otherwise 0. */
  overflow: GraphDataView<'uint32'>;
  /**
   * One row receiving 1 when segments or polylines were dropped, otherwise 0. Required with
   * `output.polylines`.
   */
  polylineOverflow?: GraphDataView<'uint32'>;
};

/**
 * Optional stitched polyline outputs of {@link GPUIsolines} (`output.polylines`). Presence of this
 * object schedules the stitching pass.
 *
 * Chains are formed per level. Open chains start at the segment with no predecessor. Closed rings
 * start at their smallest segment index and repeat the first vertex at the end. Polylines are
 * ordered by head segment index. Vertices are copies of segment endpoints, with bit-identical
 * consecutive vertices of a polyline collapsed (zero-length segments, where a corner equals the
 * level exactly, are kept in the raw segment output because they bridge chains in the stitching).
 * The capacity check uses the count before that collapse. Polyline `p` spans
 * vertex rows `[polylineOffsets[p], polylineOffsets[p + 1])` and is complete: a polyline that does
 * not fit into `vertices` is dropped, with all later ones, and `status.polylineOverflow` is set. Worst case needs
 * `2 * segmentCapacity` vertices (every chain one segment long).
 *
 * If the segment output overflowed, no polyline is produced and `status.polylineOverflow` is set.
 */
export type GPUIsolinesPolylineOutput = {
  /** World positions relative to the parameters' `localOrigin`; the capacity is `vertices.length`. */
  vertices: GraphDataView<'float32x2'>;
  /** `segments.length + 1` rows. Rows `0..polylineCount` are the vertex offsets of each polyline. */
  polylineOffsets: GraphDataView<'uint32'>;
  /** Level index per polyline, at least `segments.length` rows. */
  polylineLevels: GraphDataView<'uint32'>;
  /** 1 for a closed ring, 0 for an open chain, at least `segments.length` rows. */
  polylineClosed: GraphDataView<'uint32'>;
  /** One row receiving the number of complete polylines written. */
  polylineCount: GraphDataView<'uint32'>;
  /** One row receiving the number of vertices written (`polylineOffsets[polylineCount]`). */
  vertexCount: GraphDataView<'uint32'>;
};

/** Float32 column readback, including bytes returned by a GPU buffer read. */
export type GPUIsolinesFloatReadback = Float32Array | Uint8Array | ArrayBuffer;
/** Uint32 column readback, including bytes returned by a GPU buffer read. */
export type GPUIsolinesUintReadback = Uint32Array | Uint8Array | ArrayBuffer;

/** Host readbacks of the stitched output and its two loss words. */
export type GPUIsolinesPolylineReadback = {
  polylines: {
    vertices: GPUIsolinesFloatReadback;
    polylineOffsets: GPUIsolinesUintReadback;
    polylineLevels: GPUIsolinesUintReadback;
    polylineClosed: GPUIsolinesUintReadback;
    polylineCount: GPUIsolinesUintReadback;
    vertexCount: GPUIsolinesUintReadback;
  };
  status: {overflow: GPUIsolinesUintReadback; polylineOverflow: GPUIsolinesUintReadback};
};

/** Retained XY geometry; coordinates include the supplied local origin, without reprojection. */
export type GPUIsolinesDecodedPolylines = {
  polylines: {
    coordinates: [number, number][];
    levelIndex: number;
    levelValue: number;
    closed: boolean;
  }[];
  segmentComplete: boolean;
  /** False for either segment loss or stitched polyline loss, matching the combined loss word. */
  polylineComplete: boolean;
  complete: boolean;
};

/** Decode complete retained chains, bounded by validated live counts and CSR offsets. */
export function decodeGPUIsolinesPolylines(
  readback: GPUIsolinesPolylineReadback,
  options: {levels: GPUIsolinesFloatReadback; localOrigin?: readonly [number, number]}
): GPUIsolinesDecodedPolylines {
  const id = 'decodeGPUIsolinesPolylines';
  const read = <ArrayType extends Float32Array | Uint32Array>(
    name: string,
    data: ArrayType | Uint8Array | ArrayBuffer,
    readValues: (bytes: Uint8Array, minimumLength: number) => ArrayType,
    minimumLength: number,
    expectedBrand: 'Uint32Array' | 'Float32Array'
  ): ArrayType => {
    validateRequirement(id, data !== undefined && data !== null, `${name} readback is required`);
    // The intrinsic typed-array brand accepts foreign realms and Node Buffer bytes without
    // reinterpreting unrelated element types (or a DataView) as successful readback words.
    const typedArrayName = Object.getOwnPropertyDescriptor(
      Object.getPrototypeOf(Uint8Array.prototype),
      Symbol.toStringTag
    )!.get!;
    const observedBrand = ArrayBuffer.isView(data) ? typedArrayName.call(data) : undefined;
    validateRequirement(
      id,
      ArrayBuffer.isView(data)
        ? observedBrand === expectedBrand || observedBrand === 'Uint8Array'
        : Object.prototype.toString.call(data) === '[object ArrayBuffer]',
      `${name} must be ${expectedBrand}, Uint8Array or ArrayBuffer; received ${observedBrand ?? Object.prototype.toString.call(data)}`
    );
    validateRequirement(
      id,
      data.byteLength % 4 === 0,
      `${name} byte length must be a multiple of 4; received ${data.byteLength}`
    );
    validateRequirement(
      id,
      data.byteLength / 4 >= minimumLength,
      `${name} must contain at least ${minimumLength} elements; received ${data.byteLength / 4}`
    );
    // Create the logical live prefix directly: Buffer.slice aliases its parent and foreign-realm
    // byte arrays fail instanceof checks. The common reader handles little-endian bytes.
    const isView = ArrayBuffer.isView(data);
    const bytes = new Uint8Array(
      isView ? data.buffer : (data as ArrayBuffer),
      isView ? data.byteOffset : 0,
      minimumLength * 4
    );
    return readValues(bytes, minimumLength);
  };
  const polylineCount = read(
    'polylineCount',
    readback.polylines.polylineCount,
    getPackedUint32Values,
    1,
    'Uint32Array'
  )[0];
  const vertexCount = read(
    'vertexCount',
    readback.polylines.vertexCount,
    getPackedUint32Values,
    1,
    'Uint32Array'
  )[0];
  const overflow = read(
    'status.overflow',
    readback.status.overflow,
    getPackedUint32Values,
    1,
    'Uint32Array'
  )[0];
  const polylineOverflow = read(
    'status.polylineOverflow',
    readback.status.polylineOverflow,
    getPackedUint32Values,
    1,
    'Uint32Array'
  )[0];
  validateRequirement(
    id,
    overflow <= 1 && polylineOverflow <= 1,
    'status loss words must be 0 or 1'
  );
  validateRequirement(
    id,
    overflow === 0 || polylineOverflow === 1,
    'segment loss requires polylineOverflow = 1'
  );
  validateRequirement(
    id,
    overflow === 0 || (polylineCount === 0 && vertexCount === 0),
    'segment loss suppresses stitched polylines'
  );
  validateRequirement(
    id,
    readback.polylines.vertices.byteLength >= 8,
    'polylines.vertices must have an allocated XY row as evidence of stitching; for the CPU adapter supply vertexCapacity >= 1'
  );
  const vertices = read(
    'vertices',
    readback.polylines.vertices,
    getPackedFloat32Values,
    vertexCount * 2,
    'Float32Array'
  );
  const offsets = read(
    'polylineOffsets',
    readback.polylines.polylineOffsets,
    getPackedUint32Values,
    polylineCount + 1,
    'Uint32Array'
  );
  const levelIndices = read(
    'polylineLevels',
    readback.polylines.polylineLevels,
    getPackedUint32Values,
    polylineCount,
    'Uint32Array'
  );
  const closures = read(
    'polylineClosed',
    readback.polylines.polylineClosed,
    getPackedUint32Values,
    polylineCount,
    'Uint32Array'
  );
  const requiredLevelCount = levelIndices.reduce(
    (maximum, levelIndex) => Math.max(maximum, levelIndex + 1),
    0
  );
  const levels = read(
    'levels',
    options.levels,
    getPackedFloat32Values,
    requiredLevelCount,
    'Float32Array'
  );
  const localOrigin = options.localOrigin ?? [0, 0];
  validateRequirement(
    id,
    localOrigin.length === 2 && localOrigin.every(Number.isFinite),
    'localOrigin must contain two finite coordinates'
  );
  validateRequirement(
    id,
    offsets[0] === 0 && offsets[polylineCount] === vertexCount,
    'polylineOffsets must start at 0 and end at vertexCount'
  );
  const polylines: GPUIsolinesDecodedPolylines['polylines'] = [];
  for (let polyline = 0; polyline < polylineCount; polyline++) {
    const begin = offsets[polyline];
    const end = offsets[polyline + 1];
    validateRequirement(
      id,
      begin < end && end <= vertexCount,
      `polylineOffsets must increase within vertexCount; received ${begin}, ${end}`
    );
    const levelIndex = levelIndices[polyline];
    validateRequirement(
      id,
      levelIndex < levels.length && Number.isFinite(levels[levelIndex]),
      `polylineLevels must index a finite level; received ${levelIndex}`
    );
    validateRequirement(
      id,
      closures[polyline] <= 1,
      `polylineClosed must be 0 or 1; received ${closures[polyline]}`
    );
    const coordinates: [number, number][] = [];
    for (let vertex = begin; vertex < end; vertex++) {
      const coordinate: [number, number] = [
        vertices[2 * vertex] + localOrigin[0],
        vertices[2 * vertex + 1] + localOrigin[1]
      ];
      validateRequirement(
        id,
        coordinate.every(Number.isFinite),
        `vertices row ${vertex} must contain finite XY coordinates`
      );
      coordinates.push(coordinate);
    }
    polylines.push({
      coordinates,
      levelIndex,
      levelValue: levels[levelIndex],
      closed: closures[polyline] !== 0
    });
  }
  const segmentComplete = overflow === 0;
  const polylineComplete = polylineOverflow === 0;
  return {
    polylines,
    segmentComplete,
    polylineComplete,
    complete: segmentComplete && polylineComplete
  };
}

/**
 * Properties for {@link GPUIsolines}.
 *
 * Topology: `grid` size, `levels.length` (the maximum level count), the segment and vertex
 * capacities, `tile`, `noDataValue`, band metadata and which optional views exist. Per-frame (no recompile): raster and
 * `levels` contents, and `parameters` (`levelCount`, extent).
 */
export type GPUIsolinesProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'isolines'`. */
  id?: string;
  /** Raster grid: its width and height (each at least 2) size the kernels. */
  grid: GPURasterMetadata;
  /** Optional halo tile placement and half-open cell ownership; fixed for the compiled graph. */
  tile?: GPUIsolinesTile;
  /** Published band or packed row-major float32 raster. Non-finite samples are nodata. */
  values: GraphDataView<'float32'> | GPURasterBand;
  /** Optional input row filter over the raster: nonzero keeps the cell, zero marks no-data. */
  mask?: GraphDataView<'uint32'>;
  /** Optional finite sentinel for legacy values only; bands use their native sentinel. */
  noDataValue?: number;
  /** Level values; rows `[0, levelCount)` are active. Contents may change per frame. */
  levels: GraphDataView<'float32'>;
  /** Per-frame float32 view written with `getGPUIsolinesParameterValues`. */
  parameters: GraphDataView<'float32'>;
  /** Caller-owned outputs: segments, optional polylines, and status words. */
  output: GPUIsolinesOutput;
};

/**
 * Marching-squares isolines of a raster at several levels, as ordered segments and optionally
 * stitched polylines.
 *
 * A cell with a nodata corner emits nothing. Corners with `value >= level` are high; saddles are
 * resolved by the bilinear (asymptotic) decider of the published `GPURasterContours`: the two
 * high corners are joined when the saddle value of the bilinear surface through the four corners,
 * `(v0 * v2 - v1 * v3) / (v0 + v2 - v1 - v3)`, is at or above the level. (Before 2026-11-10 the
 * rule was the mean of the four corners, which differs when the corners are far from symmetric.)
 * Segments are ordered by (cell index, level index, slot), so output is deterministic. The pipeline is count, `GPUScan`, scatter, publish, and then for
 * polylines a link pass, a fixed number of pointer-jumping rounds, scans and a gather. Changing
 * levels, `levelCount`, or the extent never recompiles.
 *
 * Tiling: `tile` owns cells by their lower-left sample in a half-open rectangle. Supply all four
 * samples of every owned cell (a one-sample right/top halo at internal tile boundaries). Counts,
 * capacity status and stitching include only owned cells. Use the SAME domain parameters and
 * localOrigin for every tile: domain sample offsets and the shared sample anchors are combined
 * in integer arithmetic before conversion and interpolation, so seam endpoints use the same
 * float32 expression. `segmentEdges` then holds domain edge IDs;
 * pair them with `segmentLevels` to join tile fragments without coordinate snapping. Polylines
 * are stitched within each owned rectangle; cross-tile chains remain fragments. External vector
 * overlays must use this same origin and coordinate frame (row 0 at minY).
 *
 * Precision: vertices are `anchorX + (gridX - columnAnchor + 0.5) * cellWidth` in float32,
 * relative to the packer's `localOrigin`. The integer sample subtraction precedes conversion to
 * float32; an origin near a large sample index therefore retains sub-cell coordinates. Absolute vertices (the default origin `[0, 0]`) are quantized to the float32 ulp,
 * 0.5 m at a UTM northing of 4.6e6 m (2 m at 2e7 m in EPSG:3857). With `localOrigin` set to the
 * raster or tile corner, a vertex is within about `1.2e-7` of its distance from that corner of the
 * float64 crossing (1.2 cm at 100 km); add the origin back on the CPU or as the layer's coordinate
 * origin.
 *
 * Row direction: row-zero sample centers are half a cell above `minY` of the outer extent; rows advance
 * toward larger y. Placement uses only the dimensions of `grid` (or the domain with `tile`);
 * `grid.affine` is checked for finite values at construction but is not used for placement. A north-up grid (row 0 at the top, the default of
 * `makeGPURasterMetadata`) therefore comes out mirrored north to south unless the caller reverses
 * the row order of the raster before upload. Mirroring the output instead (`y' = minY + maxY - y`)
 * also swaps the documented side: high values are then on the right of a segment, and ring
 * windings are reversed. Following the grid affine is deferred (roadmap 12.4).
 */
export class GPUIsolines implements GPUCommandNodeProducer, GPURasterHaloStage {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUIsolinesProps;
  /** Cells read one sample right and above. */
  readonly requiredHalo = 1;
  /** Segment capacity (`output.segments.length`). */
  readonly segmentCapacity: number;
  /** Maximum number of levels (`levels.length`). */
  readonly maximumLevelCount: number;

  constructor(props: GPUIsolinesProps) {
    this.id = props.id ?? 'isolines';
    this.props = props;
    const {id} = this;
    const {output} = props;
    const polylines = output?.polylines;
    validateRasterGrid(id, props.grid);
    const {width, height} = props.grid;
    validateIsolinesTile(id, props.grid, props.tile);
    validateIsolinesMaximumSegmentCount(id, props.grid, props.tile, props.levels.length);
    if (props.noDataValue !== undefined && !Number.isFinite(props.noDataValue)) {
      throw new Error(`${id} noDataValue must be finite (NaN cells are always nodata)`);
    }
    validateRasterAlgebraInput(id, 'values', props.values, props.grid, width * height);
    validateIsolinesView(id, 'mask', props.mask, 'uint32', width * height);
    validateIsolinesView(id, 'levels', props.levels, 'float32', 1);
    validateIsolinesView(
      id,
      'parameters',
      props.parameters,
      'float32',
      GPU_ISOLINES_PARAMETER_LENGTH
    );
    if (!output?.segments) {
      throw new Error(`${id} needs output.segments`);
    }
    this.maximumLevelCount = props.levels.length;
    this.segmentCapacity = output.segments.length;
    const capacity = this.segmentCapacity;
    if (capacity < 1) {
      throw new Error(`${id} output.segments must hold at least one row`);
    }
    validateIsolinesView(id, 'output.segments', output.segments, 'float32x4', 1);
    validateIsolinesView(id, 'output.segmentLevels', output.segmentLevels, 'uint32', capacity);
    validateIsolinesView(id, 'output.segmentEdges', output.segmentEdges, 'uint32x2', capacity);
    if (!output.status.count) {
      throw new Error(`${id} needs output.status.count`);
    }
    if (!output.status?.overflow) {
      throw new Error(`${id} needs output.status.overflow`);
    }
    for (const [name, scalar] of [
      ['status.count', output.status.count],
      ['status.overflow', output.status.overflow],
      ['status.requiredCount', output.status.requiredCount],
      ['status.polylineOverflow', output.status.polylineOverflow]
    ] as const) {
      validateIsolinesView(id, `output.${name}`, scalar, 'uint32', 1);
    }
    if (polylines) {
      validateIsolinesView(id, 'output.polylines.vertices', polylines.vertices, 'float32x2', 1);
      validateIsolinesView(
        id,
        'output.polylines.polylineOffsets',
        polylines.polylineOffsets,
        'uint32',
        capacity + 1
      );
      validateIsolinesView(
        id,
        'output.polylines.polylineLevels',
        polylines.polylineLevels,
        'uint32',
        capacity
      );
      validateIsolinesView(
        id,
        'output.polylines.polylineClosed',
        polylines.polylineClosed,
        'uint32',
        capacity
      );
      for (const name of ['polylineCount', 'vertexCount'] as const) {
        if (!polylines[name]) {
          throw new Error(`${id} needs output.polylines.${name}`);
        }
        validateIsolinesView(id, `output.polylines.${name}`, polylines[name], 'uint32', 1);
      }
      if (!output.status.polylineOverflow) {
        throw new Error(`${id} needs output.status.polylineOverflow`);
      }
      validateIsolinesView(
        id,
        'output.status.polylineOverflow',
        output.status.polylineOverflow,
        'uint32',
        1
      );
    }
    validateIsolinesAliasing(id, getNamedRasterOutputViews(output), {
      ...getNamedRasterAlgebraInputViews(props.values, 'values'),
      mask: props.mask,
      levels: props.levels,
      parameters: props.parameters
    });
  }

  private getOutputViews(): GraphDataView[] {
    const {output} = this.props;
    const {polylines, status} = output;
    return [
      output.segments,
      output.segmentLevels,
      output.segmentEdges,
      output.status.count,
      output.status.requiredCount,
      status.overflow,
      status.polylineOverflow,
      polylines?.vertices,
      polylines?.polylineOffsets,
      polylines?.polylineLevels,
      polylines?.polylineClosed,
      polylines?.polylineCount,
      polylines?.vertexCount
    ].filter(view => view !== undefined);
  }

  /** Returns count, scan, scatter and publish nodes, plus stitching nodes with `polylines`. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const storageProps = this.props;
    validateRasterAlgebraInputBindings(graph, this.id, [storageProps.values], storageProps.grid);
    validateRasterStorageBindings(
      graph,
      this.id,
      [storageProps.mask, storageProps.levels, storageProps.parameters, ...this.getOutputViews()],
      {
        'cell counts and offsets':
          (storageProps.grid.width - 1) * (storageProps.grid.height - 1) * 4,
        'segment edges':
          storageProps.output.polylines && !storageProps.output.segmentEdges
            ? this.segmentCapacity * 8
            : 0,
        'stitch states': storageProps.output.polylines ? this.segmentCapacity * 16 : 0,
        'staged polyline vertices': (storageProps.output.polylines?.vertices.length ?? 0) * 8,
        'stitch totals': storageProps.output.polylines ? 8 : 4
      }
    );

    const {id, props, segmentCapacity, maximumLevelCount} = this;
    const {output} = props;
    const {polylines} = output;
    const {width, height} = props.grid;
    validateGraphViewsBelongToGraph(id, graph, [
      ...getRasterAlgebraInputViews(props.values),
      props.mask,
      props.levels,
      props.parameters,
      ...this.getOutputViews()
    ]);
    const cellCount = (width - 1) * (height - 1);
    // Check the largest row-proportional scratch binding before creating any command nodes.
    for (const [name, byteLength] of [
      ['cell scratch', cellCount * 4],
      ['segment scratch', segmentCapacity * 8],
      ['polyline scratch', (polylines?.vertices.length ?? 0) * 8]
    ] as const) {
      validateGPUStorageBindingByteLength(
        graph.device,
        id,
        name,
        byteLength,
        'use smaller tiles or capacities'
      );
    }
    const source = getRasterAlgebraInputNodes(graph, `${id}-source`, props.values, props.grid);
    const nodes: GPUCommandNode<Parameters>[] = [...source.nodes];
    const ownedCells = props.tile?.ownedCells ?? [0, 0, width - 1, height - 1];
    const hasValidity = Boolean(props.mask);
    const common = `const WIDTH: u32 = ${width}u;
const HEIGHT: u32 = ${height}u;
const MAXIMUM_LEVELS: u32 = ${maximumLevelCount}u;
const CAPACITY: u32 = ${segmentCapacity}u;
${getIsolinesCornersWGSL(hasValidity, isRasterAlgebraBand(props.values) ? undefined : props.noDataValue)}
fn ownsCell(column: u32, row: u32) -> bool {
  return column >= ${ownedCells[0]}u && column < ${ownedCells[2]}u &&
    row >= ${ownedCells[1]}u && row < ${ownedCells[3]}u;
}`;
    const rasterBindings: WGSLKernelBinding[] = [
      {name: 'params', view: props.parameters, type: 'f32', access: 'read'},
      {name: 'levels', view: props.levels, type: 'f32', access: 'read'},
      {name: 'raster', view: source.values, type: 'f32', access: 'read'},
      ...(props.mask
        ? [{name: 'validity', view: props.mask, type: 'u32', access: 'read'} as const]
        : [])
    ];

    const counts = createTransientView(graph, `${id}-cell-counts`, 'uint32', cellCount);
    const offsets = createTransientView(graph, `${id}-cell-offsets`, 'uint32', cellCount);
    const total = createTransientView(graph, `${id}-total`, 'uint32', 1);
    const needsEdges = Boolean(polylines);
    const edges =
      output.segmentEdges ??
      (needsEdges
        ? createTransientView(graph, `${id}-edges`, 'uint32x2', segmentCapacity)
        : undefined);

    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-count`,
        operation: OPERATION,
        variant: 'count',
        bindings: [
          ...rasterBindings,
          {name: 'counts', view: counts, type: 'u32', access: 'read_write'}
        ],
        invocationCount: cellCount,
        declarations: common,
        body: `let cx = index % (WIDTH - 1u);
  let cy = index / (WIDTH - 1u);
  let corners = loadCorners(cx, cy);
  var cellTotal = 0u;
  if (ownsCell(cx, cy) && corners.valid) {
    let levelCount = min(u32(max(params[paramsOffset], 0.0)), MAXIMUM_LEVELS);
    for (var levelIndex = 0u; levelIndex < levelCount; levelIndex++) {
      let mask = getHighMask(corners.v, levels[levelsOffset + levelIndex]);
      cellTotal += countOneBits(getExitMask(mask));
    }
  }
  counts[countsOffset + index] = cellTotal;`
      }),
      ...new GPUScan({id: `${id}-cell-scan`, input: counts, output: offsets}).getCommandNodes(
        graph
      ),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-total`,
        operation: OPERATION,
        variant: 'total',
        bindings: [
          {name: 'counts', view: counts, type: 'u32', access: 'read'},
          {name: 'offsets', view: offsets, type: 'u32', access: 'read'},
          {name: 'totalOut', view: total, type: 'u32', access: 'read_write'}
        ],
        invocationCount: 1,
        body: `totalOut[totalOutOffset] =
    offsets[offsetsOffset + ${cellCount - 1}u] + counts[countsOffset + ${cellCount - 1}u];`
      })
    );

    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-scatter`,
        operation: OPERATION,
        variant: 'scatter',
        bindings: [
          ...rasterBindings,
          {name: 'offsets', view: offsets, type: 'u32', access: 'read'},
          {name: 'segments', view: output.segments, type: 'f32', access: 'read_write'},
          {name: 'segmentLevels', view: output.segmentLevels, type: 'u32', access: 'read_write'},
          ...(edges
            ? [{name: 'edges', view: edges, type: 'u32', access: 'read_write'} as const]
            : [])
        ],
        invocationCount: cellCount,
        declarations: `${common}
${getCrossingWGSL(props.tile)}`,
        body: `let cx = index % (WIDTH - 1u);
  let cy = index / (WIDTH - 1u);
  let corners = loadCorners(cx, cy);
  if (!ownsCell(cx, cy) || !corners.valid) {
    return;
  }
  let v = corners.v;
  var position = offsets[offsetsOffset + index];
  let levelCount = min(u32(max(params[paramsOffset], 0.0)), MAXIMUM_LEVELS);
  for (var levelIndex = 0u; levelIndex < levelCount; levelIndex++) {
    let level = levels[levelsOffset + levelIndex];
    let mask = getHighMask(v, level);
    let exits = getExitMask(mask);
    if (exits == 0u) {
      continue;
    }
    let entries = getEntryMask(mask);
    // Saddles have two exits. High corners are joined when the bilinear saddle value is high.
    let isJoined = countOneBits(exits) < 2u || isSaddleJoined(v, level);
    for (var exitEdge = 0u; exitEdge < 4u; exitEdge++) {
      if (((exits >> exitEdge) & 1u) == 0u) {
        continue;
      }
      let entryEdge = getPairedEntry(exitEdge, entries, isJoined);
      // A corner exactly at the level gives a zero-length segment. It stays: it is the bridge
      // that links the neighbouring cells' chains in the edge-based stitching.
      if (position < CAPACITY) {
        let start = getEdgeCrossing(exitEdge, level, v, cx, cy);
        let end = getEdgeCrossing(entryEdge, level, v, cx, cy);
        let base = segmentsOffset + 4u * position;
        segments[base] = start.x;
        segments[base + 1u] = start.y;
        segments[base + 2u] = end.x;
        segments[base + 3u] = end.y;
        segmentLevels[segmentLevelsOffset + position] = levelIndex;
        ${
          edges
            ? `edges[edgesOffset + 2u * position] = getEdgeId(exitEdge, cx, cy);
        edges[edgesOffset + 2u * position + 1u] = getEdgeId(entryEdge, cx, cy);`
            : ''
        }
      }
      position++;
    }
  }`
      }),
      // Segment levels double as the compact ids: segments are already in place.
      createPublishNode<Parameters>(graph, {
        id: `${id}-publish`,
        operation: OPERATION,
        requiredCount: total,
        output: {
          ids: graph.createDataView(output.segmentLevels.buffer, {
            format: 'uint32',
            byteOffset: output.segmentLevels.byteOffset,
            length: output.segments.length
          }),
          status: {
            count: output.status.count,
            overflow: output.status.overflow,
            requiredCount: output.status.requiredCount
          }
        }
      })
    );

    if (polylines && edges) {
      nodes.push(
        ...this.getStitchNodes(graph, {
          counts,
          offsets,
          edges,
          polylines,
          common: getStitchWGSL(width, height)
        })
      );
    }
    if (!polylines && output.status.polylineOverflow) {
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-publish-polyline-overflow`,
          operation: OPERATION,
          variant: 'publish-polyline-overflow',
          bindings: [
            {name: 'segmentOverflow', view: output.status.overflow, type: 'u32', access: 'read'},
            {
              name: 'polylineOverflow',
              view: output.status.polylineOverflow,
              type: 'u32',
              access: 'read_write'
            }
          ],
          invocationCount: 1,
          body: 'polylineOverflow[polylineOverflowOffset] = segmentOverflow[segmentOverflowOffset];'
        })
      );
    }
    if (props.tile && output.segmentEdges) {
      const {domainWidth, domainHeight, columnOffset, rowOffset} = props.tile;
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-domain-edges`,
          operation: OPERATION,
          variant: 'domain-edges',
          bindings: [
            {name: 'edges', view: output.segmentEdges, type: 'u32', access: 'read_write'},
            {name: 'count', view: output.status.count, type: 'u32', access: 'read'}
          ],
          invocationCount: segmentCapacity,
          body: `if (index >= count[countOffset]) { return; }
  for (var endpoint = 0u; endpoint < 2u; endpoint++) {
    let position = edgesOffset + 2u * index + endpoint;
    let edge = edges[position];
    var domainEdge = 0u;
    if (edge < ${height * (width - 1)}u) {
      domainEdge = (edge / ${width - 1}u + ${rowOffset}u) * ${domainWidth - 1}u +
        edge % ${width - 1}u + ${columnOffset}u;
    } else {
      let vertical = edge - ${height * (width - 1)}u;
      domainEdge = ${domainHeight * (domainWidth - 1)}u +
        (vertical / ${width}u + ${rowOffset}u) * ${domainWidth}u +
        vertical % ${width}u + ${columnOffset}u;
    }
    edges[position] = domainEdge;
  }`
        })
      );
    }
    return nodes;
  }

  private getStitchNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>,
    context: {
      counts: GraphDataView<'uint32'>;
      offsets: GraphDataView<'uint32'>;
      edges: GraphDataView<'uint32x2'>;
      polylines: GPUIsolinesPolylineOutput;
      common: string;
    }
  ): GPUCommandNode<Parameters>[] {
    const {id, props, segmentCapacity: capacity} = this;
    const {output} = props;
    const {counts, offsets, edges, polylines} = context;
    const nodes: GPUCommandNode<Parameters>[] = [];
    const transient = (name: string, length: number = capacity) =>
      createTransientView(graph, `${id}-${name}`, 'uint32', length);
    const kernel = (
      step: string,
      bindings: WGSLKernelBinding[],
      body: string,
      declarations: string = '',
      invocationCount: number = capacity
    ) =>
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-${step}`,
          operation: OPERATION,
          variant: step,
          bindings,
          invocationCount,
          declarations: `const CAPACITY: u32 = ${capacity}u;
${context.common}
${declarations}`,
          body
        })
      );
    const read = (name: string, view: GraphDataView): WGSLKernelBinding => ({
      name,
      view,
      type: 'u32',
      access: 'read'
    });
    const write = (name: string, view: GraphDataView): WGSLKernelBinding => ({
      name,
      view,
      type: 'u32',
      access: 'read_write'
    });

    // Active segment count: zero when the segment output overflowed (documented behaviour).
    const active = transient('active', 1);
    kernel(
      'stitch-active',
      [
        read('countIn', output.status.count),
        read('overflowIn', output.status.overflow),
        write('activeOut', active)
      ],
      `activeOut[activeOutOffset] =
    select(min(countIn[countInOffset], CAPACITY), 0u, overflowIn[overflowInOffset] != 0u);`,
      '',
      1
    );

    // Predecessor link: the same-level segment in the neighbour cell across the start edge whose
    // end edge is that edge. Successors follow from ranks, so no successor array is needed.
    const previous = transient('previous');
    kernel(
      'links',
      [
        read('edges', edges),
        read('levels', output.segmentLevels),
        read('offsets', offsets),
        read('counts', counts),
        read('activeCount', active),
        write('previousOut', previous)
      ],
      `var previousSegment = NONE;
  if (index < activeCount[activeCountOffset]) {
    let startEdge = edges[edgesOffset + 2u * index];
    let endEdge = edges[edgesOffset + 2u * index + 1u];
    let level = levels[levelsOffset + index];
    previousSegment = findLinkedSegment(getNeighbourCell(endEdge, startEdge), level, 1u, startEdge);
  }
  previousOut[previousOutOffset + index] = previousSegment;`,
      getLinkWGSL()
    );

    // Pointer jumping on the predecessor links. Each node tracks its jump pointer, the smallest key
    // seen (heads of open chains have key 0, every other node index + 1, so a ring converges on its
    // smallest index), the distance to that node, and the number of steps its pointer spans.
    //
    // The four fields of one node live in one 16-byte record, because every round gathers all of
    // them at the jump target: one cache line instead of four.
    //
    // The round budget covers the worst case (one chain of `capacity` segments), but the rounds are
    // gated by a GPU convergence flag: a round that lowers no key proves every window already spans
    // its whole chain or ring (otherwise the node 2^r steps from the minimum would see it through
    // its jump), after which keys and distances are fixed and open-chain pointers sit on their
    // head. Short contour chains therefore need about log2(longest chain) rounds, not
    // log2(capacity). Rounds run in gated pairs, and an even count leaves the ping-pong result in
    // `states[0]` whatever round the gate stops at.
    const roundCount = 2 * Math.ceil((Math.ceil(Math.log2(capacity)) + 1) / 2);
    const states = [0, 1].map(copy => transient(`jump-${copy}-state`, 4 * capacity));
    kernel(
      'jump-init',
      [read('previous', previous), write('stateOut', states[0])],
      `let hasPrevious = previous[previousOffset + index] != NONE;
  let base = stateOutOffset + 4u * index;
  stateOut[base] = select(index, previous[previousOffset + index], hasPrevious);
  stateOut[base + 1u] = select(0u, index + 1u, hasPrevious);
  stateOut[base + 2u] = 0u;
  stateOut[base + 3u] = select(0u, 1u, hasPrevious);`
    );
    const jumpState = createRasterIterationState<Parameters>(
      graph,
      `${id}-jump`,
      OPERATION,
      capacity,
      id
    );
    nodes.push(
      createRasterIterationResetNode<Parameters>(graph, {
        id: `${id}-jump-reset`,
        operation: OPERATION,
        state: jumpState
      }),
      ...createRasterGatedLoopNodes<Parameters>(graph, {
        id: `${id}-jump`,
        operation: OPERATION,
        state: jumpState,
        maxIterations: roundCount,
        groupSize: 2,
        createRound: ({nodeId, iteration, slotBinding, roundSlotWGSL, condition, extraResources}) =>
          createWGSLKernelNode<Parameters>(graph, {
            id: nodeId,
            operation: OPERATION,
            variant: 'jump',
            bindings: [
              read('stateIn', states[iteration % 2]),
              write('stateOut', states[(iteration + 1) % 2]),
              {name: 'status', view: jumpState.status, type: 'atomic<u32>', access: 'read_write'},
              slotBinding
            ],
            invocationCount: jumpState.invocationCount,
            body: `let ownBase = stateInOffset + 4u * index;
  let jump = stateIn[ownBase];
  let jumpBase = stateInOffset + 4u * jump;
  let ownKey = stateIn[ownBase + 1u];
  let jumpKey = stateIn[jumpBase + 1u];
  let ownWins = ownKey <= jumpKey;
  let outBase = stateOutOffset + 4u * index;
  stateOut[outBase] = stateIn[jumpBase];
  stateOut[outBase + 1u] = select(jumpKey, ownKey, ownWins);
  stateOut[outBase + 2u] = select(
    stateIn[jumpBase + 2u] + stateIn[ownBase + 3u],
    stateIn[ownBase + 2u],
    ownWins
  );
  stateOut[outBase + 3u] = stateIn[ownBase + 3u] + stateIn[jumpBase + 3u];
  let roundBit = 1u << ${roundSlotWGSL};
  if (!ownWins && (atomicLoad(&status[statusOffset]) & roundBit) == 0u) {
    atomicOr(&status[statusOffset], roundBit);
  }`,
            condition,
            extraResources
          })
      })
    );
    const final = states[0];

    // Head and rank per segment; bit 31 of the rank marks a closed ring.
    const headOf = transient('head-of');
    const rankOf = transient('rank-of');
    kernel(
      'resolve',
      [read('stateIn', final), write('headOut', headOf), write('rankOut', rankOf)],
      `let base = stateInOffset + 4u * index;
  let key = stateIn[base + 1u];
  let isRing = key != 0u;
  headOut[headOutOffset + index] = select(stateIn[base], key - 1u, isRing);
  rankOut[rankOutOffset + index] =
    stateIn[base + 2u] | select(0u, 0x80000000u, isRing);`
    );

    // Chain lengths via integer atomicMax, then head flags (1 open, 2 closed).
    const chainLength = transient('chain-length');
    const headKind = transient('head-kind');
    nodes.push(
      createFillNode<Parameters>(graph, {
        id: `${id}-chain-length-clear`,
        operation: OPERATION,
        view: chainLength,
        type: 'u32',
        value: '0u'
      })
    );
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-chain-length`,
        operation: OPERATION,
        variant: 'chain-length',
        bindings: [
          read('headOf', headOf),
          read('rankOf', rankOf),
          read('activeCount', active),
          {name: 'chainLength', view: chainLength, type: 'atomic<u32>', access: 'read_write'},
          write('headKindOut', headKind)
        ],
        invocationCount: capacity,
        body: `var kind = 0u;
  if (index < activeCount[activeCountOffset]) {
    let packed = rankOf[rankOfOffset + index];
    let rank = packed & 0x7fffffffu;
    atomicMax(&chainLength[chainLengthOffset + headOf[headOfOffset + index]], rank + 1u);
    kind = select(0u, select(1u, 2u, (packed >> 31u) != 0u), rank == 0u);
  }
  headKindOut[headKindOutOffset + index] = kind;`
      })
    );

    const headFlag = transient('head-flag');
    const vertexCountPerHead = transient('head-vertex-count');
    const headIndex = transient('head-index');
    const vertexOffset = transient('vertex-offset');
    kernel(
      'scan-input',
      [
        read('headKind', headKind),
        read('chainLength', chainLength),
        write('flagOut', headFlag),
        write('vertexCountOut', vertexCountPerHead)
      ],
      `let isHead = headKind[headKindOffset + index] != 0u;
  flagOut[flagOutOffset + index] = select(0u, 1u, isHead);
  vertexCountOut[vertexCountOutOffset + index] =
    select(0u, chainLength[chainLengthOffset + index] + 1u, isHead);`
    );
    nodes.push(
      ...new GPUScan({
        id: `${id}-head-scan`,
        input: headFlag,
        output: headIndex
      }).getCommandNodes(graph),
      ...new GPUScan({
        id: `${id}-vertex-scan`,
        input: vertexCountPerHead,
        output: vertexOffset
      }).getCommandNodes(graph)
    );

    kernel(
      'polyline-records',
      [
        read('headKind', headKind),
        read('headIndex', headIndex),
        read('vertexOffset', vertexOffset),
        read('segmentLevels', output.segmentLevels),
        write('polylineOffsets', polylines.polylineOffsets),
        write('polylineLevels', polylines.polylineLevels),
        write('polylineClosed', polylines.polylineClosed)
      ],
      `let kind = headKind[headKindOffset + index];
  if (kind != 0u) {
    let polyline = headIndex[headIndexOffset + index];
    polylineOffsets[polylineOffsetsOffset + polyline] = vertexOffset[vertexOffsetOffset + index];
    polylineLevels[polylineLevelsOffset + polyline] = segmentLevels[segmentLevelsOffset + index];
    polylineClosed[polylineClosedOffset + polyline] = select(0u, 1u, kind == 2u);
  }`
    );

    const vertexCapacity = polylines.vertices.length;
    // Vertices are first written uncompacted (`vertexOffset + rank`), then consecutive duplicates
    // (bit-identical neighbours, from zero-length bridge segments) are compacted out.
    const stagedVertices = createTransientView(
      graph,
      `${id}-staged-vertices`,
      'float32x2',
      vertexCapacity
    );
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-vertices`,
        operation: OPERATION,
        variant: 'vertices',
        bindings: [
          read('headOf', headOf),
          read('rankOf', rankOf),
          read('activeCount', active),
          read('chainLength', chainLength),
          read('vertexOffset', vertexOffset),
          {name: 'segments', view: output.segments, type: 'f32', access: 'read'},
          {name: 'vertices', view: stagedVertices, type: 'f32', access: 'read_write'}
        ],
        invocationCount: capacity,
        declarations: `const VERTEX_CAPACITY: u32 = ${vertexCapacity}u;`,
        body: `if (index >= activeCount[activeCountOffset]) {
    return;
  }
  let head = headOf[headOfOffset + index];
  let packed = rankOf[rankOfOffset + index];
  let rank = packed & 0x7fffffffu;
  let length = chainLength[chainLengthOffset + head];
  let base = vertexOffset[vertexOffsetOffset + head];
  if (base + length + 1u > VERTEX_CAPACITY) {
    return;
  }
  let own = segmentsOffset + 4u * index;
  let slot = verticesOffset + 2u * (base + rank);
  vertices[slot] = segments[own];
  vertices[slot + 1u] = segments[own + 1u];
  if (rank + 1u == length) {
    // The last vertex of a ring repeats the first one; an open chain ends at its last p1.
    let closing = (packed >> 31u) != 0u;
    let source = select(own + 2u, segmentsOffset + 4u * head, closing);
    vertices[slot + 2u] = segments[source];
    vertices[slot + 3u] = segments[source + 1u];
  }`
      })
    );

    const totals = transient('stitch-totals', 2);
    kernel(
      'stitch-totals',
      [
        read('headIndex', headIndex),
        read('headFlag', headFlag),
        read('vertexOffset', vertexOffset),
        read('vertexCount', vertexCountPerHead),
        write('totalsOut', totals)
      ],
      `totalsOut[totalsOutOffset] =
    headIndex[headIndexOffset + CAPACITY - 1u] + headFlag[headFlagOffset + CAPACITY - 1u];
  totalsOut[totalsOutOffset + 1u] =
    vertexOffset[vertexOffsetOffset + CAPACITY - 1u] +
    vertexCount[vertexCountOffset + CAPACITY - 1u];`,
      '',
      1
    );
    // contract-allow: single-invocation -- binary search over segmentCapacity offsets, at most ceil(log2(segmentCapacity + 1)) steps.
    kernel(
      'stitch-publish',
      [
        read('totals', totals),
        read('segmentOverflow', output.status.overflow),
        write('polylineOffsets', polylines.polylineOffsets),
        write('polylineCountOut', polylines.polylineCount),
        write('vertexCountOut', polylines.vertexCount),
        write('overflowOut', output.status.polylineOverflow!)
      ],
      `let polylineTotal = totals[totalsOffset];
  let vertexTotal = totals[totalsOffset + 1u];
  polylineOffsets[polylineOffsetsOffset + polylineTotal] = vertexTotal;
  // Offsets are nondecreasing: keep the longest prefix of polylines that fits the vertex capacity.
  var low = 0u;
  var high = polylineTotal;
  loop {
    if (low >= high) {
      break;
    }
    let middle = (low + high + 1u) / 2u;
    if (polylineOffsets[polylineOffsetsOffset + middle] <= VERTEX_CAPACITY) {
      low = middle;
    } else {
      high = middle - 1u;
    }
  }
  polylineCountOut[polylineCountOutOffset] = low;
  vertexCountOut[vertexCountOutOffset] = polylineOffsets[polylineOffsetsOffset + low];
  overflowOut[overflowOutOffset] =
    select(0u, 1u, segmentOverflow[segmentOverflowOffset] != 0u || low < polylineTotal);`,
      `const VERTEX_CAPACITY: u32 = ${vertexCapacity}u;`,
      1
    );

    // Compaction of consecutive duplicate vertices within each polyline.
    const startFlag = transient('start-flag', vertexCapacity);
    const keepFlag = transient('keep-flag', vertexCapacity);
    const keepOffset = transient('keep-offset', vertexCapacity);
    const capacityDeclaration = `const VERTEX_CAPACITY: u32 = ${vertexCapacity}u;`;
    const readStaged: WGSLKernelBinding = {
      name: 'staged',
      view: stagedVertices,
      type: 'f32',
      access: 'read'
    };
    nodes.push(
      createFillNode<Parameters>(graph, {
        id: `${id}-start-flag-clear`,
        operation: OPERATION,
        view: startFlag,
        type: 'u32',
        value: '0u'
      })
    );
    kernel(
      'start-flags',
      [
        read('headKind', headKind),
        read('vertexOffset', vertexOffset),
        write('startOut', startFlag)
      ],
      `if (headKind[headKindOffset + index] != 0u) {
    let first = vertexOffset[vertexOffsetOffset + index];
    if (first < VERTEX_CAPACITY) {
      startOut[startOutOffset + first] = 1u;
    }
  }`,
      capacityDeclaration
    );
    kernel(
      'keep-flags',
      [
        readStaged,
        read('startFlag', startFlag),
        read('vertexCount', polylines.vertexCount),
        write('keepOut', keepFlag)
      ],
      `var keep = 0u;
  if (index < vertexCount[vertexCountOffset]) {
    let current = staged[stagedOffset + 2u * index];
    let currentY = staged[stagedOffset + 2u * index + 1u];
    let isStart = startFlag[startFlagOffset + index] != 0u || index == 0u;
    // Bit-identical neighbours inside one polyline collapse; polyline starts always stay.
    let isDuplicate = !isStart &&
      staged[stagedOffset + 2u * index - 2u] == current &&
      staged[stagedOffset + 2u * index - 1u] == currentY;
    keep = select(1u, 0u, isDuplicate);
  }
  keepOut[keepOutOffset + index] = keep;`,
      capacityDeclaration,
      vertexCapacity
    );
    nodes.push(
      ...new GPUScan({id: `${id}-keep-scan`, input: keepFlag, output: keepOffset}).getCommandNodes(
        graph
      )
    );
    kernel(
      'compact-vertices',
      [
        readStaged,
        read('keepFlag', keepFlag),
        read('keepOffset', keepOffset),
        {name: 'vertices', view: polylines.vertices, type: 'f32', access: 'read_write'}
      ],
      `if (keepFlag[keepFlagOffset + index] != 0u) {
    let destination = verticesOffset + 2u * keepOffset[keepOffsetOffset + index];
    vertices[destination] = staged[stagedOffset + 2u * index];
    vertices[destination + 1u] = staged[stagedOffset + 2u * index + 1u];
  }`,
      capacityDeclaration,
      vertexCapacity
    );
    kernel(
      'compact-offsets',
      [
        read('keepFlag', keepFlag),
        read('keepOffset', keepOffset),
        read('polylineCount', polylines.polylineCount),
        read('vertexCount', polylines.vertexCount),
        write('polylineOffsets', polylines.polylineOffsets)
      ],
      `if (index <= polylineCount[polylineCountOffset]) {
    let old = polylineOffsets[polylineOffsetsOffset + index];
    let total = keepOffset[keepOffsetOffset + VERTEX_CAPACITY - 1u] +
      keepFlag[keepFlagOffset + VERTEX_CAPACITY - 1u];
    polylineOffsets[polylineOffsetsOffset + index] =
      select(keepOffset[keepOffsetOffset + old], total, old >= vertexCount[vertexCountOffset]);
  }`,
      capacityDeclaration,
      capacity + 1
    );
    kernel(
      'compact-totals',
      [
        read('keepFlag', keepFlag),
        read('keepOffset', keepOffset),
        write('vertexCountOut', polylines.vertexCount)
      ],
      `vertexCountOut[vertexCountOutOffset] =
    keepOffset[keepOffsetOffset + VERTEX_CAPACITY - 1u] +
    keepFlag[keepFlagOffset + VERTEX_CAPACITY - 1u];`,
      capacityDeclaration,
      1
    );
    return nodes;
  }
}

/** WGSL `getEdgeCrossing(k, level, v, cx, cy)` in world coordinates, canonical edge direction. */
function getCrossingWGSL(tile?: GPUIsolinesTile): string {
  return /* wgsl */ `${ISOLINES_SADDLE_WGSL}

// The entry edge a segment leaving through exitEdge ends on, walking the cell boundary.
fn getPairedEntry(exitEdge: u32, entries: u32, isJoined: bool) -> u32 {
  for (var step = 1u; step < 4u; step++) {
    let candidate = select((exitEdge + 4u - step) % 4u, (exitEdge + step) % 4u, isJoined);
    if (((entries >> candidate) & 1u) != 0u) {
      return candidate;
    }
  }
  return exitEdge;
}

fn getEdgeCrossing(k: u32, level: f32, v: vec4<f32>, cx: u32, cy: u32) -> vec2<f32> {
  let anchorX = params[paramsOffset + 1u];
  let anchorY = params[paramsOffset + 2u];
  let cellWidth = params[paramsOffset + 3u];
  let cellHeight = params[paramsOffset + 4u];
  let columnAnchor = i32(params[paramsOffset + 5u]);
  let rowAnchor = i32(params[paramsOffset + 6u]);
  // Canonical orientation: horizontal edges run +x, vertical edges run +y.
  var a = v.x;
  var b = v.y;
  if (k == 1u) {
    a = v.y;
    b = v.z;
  } else if (k == 2u) {
    a = v.w;
    b = v.z;
  } else if (k == 3u) {
    a = v.x;
    b = v.w;
  }
  let t = (level - a) / (b - a);
  if (k == 0u || k == 2u) {
    let gridX = f32(i32(cx + ${tile?.columnOffset ?? 0}u) - columnAnchor) + t;
    let gridY = f32(i32(cy + ${tile?.rowOffset ?? 0}u + select(0u, 1u, k == 2u)) - rowAnchor);
    return vec2<f32>(anchorX + (gridX + 0.5) * cellWidth, anchorY + (gridY + 0.5) * cellHeight);
  }
  let gridX = f32(i32(cx + ${tile?.columnOffset ?? 0}u + select(0u, 1u, k == 1u)) - columnAnchor);
  let gridY = f32(i32(cy + ${tile?.rowOffset ?? 0}u) - rowAnchor) + t;
  return vec2<f32>(anchorX + (gridX + 0.5) * cellWidth, anchorY + (gridY + 0.5) * cellHeight);
}`;
}

/** WGSL edge/cell topology helpers shared by the stitching kernels. */
function getStitchWGSL(width: number, height: number): string {
  return /* wgsl */ `
const WIDTH: u32 = ${width}u;
const HEIGHT: u32 = ${height}u;
const NONE: u32 = 0xffffffffu;

// The two cells that share edge e: x is below or left of it, y above or right of it.
fn getEdgeCells(e: u32) -> vec2<u32> {
  let cellColumns = WIDTH - 1u;
  let cellRows = HEIGHT - 1u;
  let horizontalCount = HEIGHT * cellColumns;
  var low = NONE;
  var high = NONE;
  if (e < horizontalCount) {
    let r = e / cellColumns;
    let c = e % cellColumns;
    if (r >= 1u) { low = (r - 1u) * cellColumns + c; }
    if (r < cellRows) { high = r * cellColumns + c; }
  } else {
    let i = e - horizontalCount;
    let r = i / WIDTH;
    let c = i % WIDTH;
    if (c >= 1u && r < cellRows) { low = r * cellColumns + c - 1u; }
    if (c < cellColumns && r < cellRows) { high = r * cellColumns + c; }
  }
  return vec2<u32>(low, high);
}

fn cellHasEdge(cell: u32, e: u32) -> bool {
  let cellColumns = WIDTH - 1u;
  let cx = cell % cellColumns;
  let cy = cell / cellColumns;
  let bottom = cy * cellColumns + cx;
  let left = HEIGHT * cellColumns + cy * WIDTH + cx;
  return e == bottom || e == bottom + cellColumns || e == left || e == left + 1u;
}

// Cell on the other side of edge \`across\`, given another edge of the segment's own cell.
fn getNeighbourCell(known: u32, across: u32) -> u32 {
  let cells = getEdgeCells(across);
  if (cells.x != NONE && cellHasEdge(cells.x, known)) {
    return cells.y;
  }
  return cells.x;
}`;
}

function getLinkWGSL(): string {
  return /* wgsl */ `
// Segment of \`level\` in \`cell\` whose start (field 0) or end (field 1) edge is \`edge\`.
fn findLinkedSegment(cell: u32, level: u32, field: u32, edge: u32) -> u32 {
  if (cell == NONE) {
    return NONE;
  }
  let first = offsets[offsetsOffset + cell];
  let segmentCount = counts[countsOffset + cell];
  for (var i = 0u; i < segmentCount; i++) {
    let position = first + i;
    if (position >= CAPACITY) {
      break;
    }
    if (levels[levelsOffset + position] == level &&
        edges[edgesOffset + 2u * position + field] == edge) {
      return position;
    }
  }
  return NONE;
}`;
}

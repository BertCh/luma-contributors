// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPU_GRID_COORDINATE_WGSL,
  GPUGridIndex,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma-contributors/graph/gpu-core';
import {
  GPU_PLANAR_SPATIAL_CONTEXT_SUPPORT,
  resolveGPUSpatialContext,
  type GPUSpatialContext
} from '../../analysis-core/contracts/spatial-context';
import {
  validateCapacity,
  validatePackedView,
  validateGraphViewsBelongToGraph,
  validateGraphOutputsDisjointFromInputs
} from '../../utils/gpu-contributor-utils';
import {
  createWGSLKernelNode,
  getWGSLFloatLiteral,
  type WGSLKernelBinding
} from '../../utils/wgsl-kernel-nodes';
import {createTiledWGSLKernelNodes} from '../../utils/wgsl-kernel-tiles';
import {GET_SPATIAL_NAN_WGSL, IS_SPATIAL_FINITE_WGSL} from '../shared/spatial-wgsl-point-helpers';

const REDUCTION_WORKGROUP_SIZE = 256;
/** Largest neighbor-grid side; the grid has `side^2` cells. */
const MAXIMUM_NEIGHBOR_GRID_SIDE = 1024;
/** Penetration noise floor: 4 f32 ulps (2^-23 each) of the largest coordinate. */
const COORDINATE_ULP_FLOOR = 4.76837158203125e-7;

/** Exact all-pairs relaxation is O(n^2) per pass; row tiling keeps each dispatch bounded. */
export const MAXIMUM_GPU_DORLING_CARTOGRAM_ROWS = 65536;

/** Caller-owned outputs. Rows retain input order, including invalid rows. */
export type GPUDorlingCartogramOutput = {
  positions: GraphDataView<'float32x2'>;
  radii: GraphDataView<'float32'>;
  sourceIds: GraphDataView<'uint32'>;
  /** One float32x2 row: maximum penetration, maximum displacement from initial position. */
  metrics: GraphDataView<'float32x2'>;
  /** Four uint32 words: iterations used, converged, budget exhausted, excluded row count. */
  status: GraphDataView<'uint32'>;
};

export type GPUDorlingCartogramProps = {
  id?: string;
  spatialContext?: GPUSpatialContext;
  positions: GraphDataView<'float32x2'>;
  values: GraphDataView<'float32'>;
  sourceIds?: GraphDataView<'uint32'>;
  /** Circle area per unit value, in squared position units. Default 1. */
  areaScale?: number;
  /** Static GPU iteration budget, 0..256. Default 100. */
  maximumIterations?: number;
  /** Maximum penetration accepted as converged, in position units. Default 0.001. */
  overlapTolerance?: number;
  /** Attraction toward the original center per pass, 0..1. Default 0.01. May leave residual overlap. */
  attraction?: number;
  /** Rows per relax/diagnose dispatch. Results do not depend on it. Default: one dispatch per pass. */
  rowTileSize?: number;
  /**
   * Compile-time. Restricts the overlap search to a per-pass uniform grid over the circle centers
   * (cells found with `GPUGridIndex`, rows sorted inside each cell) instead of testing every pair, so
   * a pass costs roughly the number of nearby circle pairs rather than `rows^2`. Each row visits
   * the cells within its radius plus the largest radius. Penetrations and so convergence are the
   * same as the all-pairs search; the sums of the corrections differ from it only by float32
   * rounding order. Mostly empty or highly clustered layouts (one huge circle among tiny ones,
   * or thousands of circles piled on one point) still approach the all-pairs cost. Default
   * `'auto'`: the grid from {@link DORLING_NEIGHBOR_GRID_MINIMUM_ROWS} rows, where it measured
   * faster than all-pairs (about 2.8x at 65536 rows; slower at 16384 rows).
   */
  neighborGrid?: boolean | 'auto';
  output: GPUDorlingCartogramOutput;
};

/** Row count from which `neighborGrid: 'auto'` uses the neighbor grid. */
export const DORLING_NEIGHBOR_GRID_MINIMUM_ROWS = 32768;

/**
 * Bounded planar Dorling circle relaxation. Uses exact spatial all-pairs queries (O(n²), at
 * most 65536 rows, optionally split into row tiles via `rowTileSize`; `neighborGrid` restricts each
 * pass to nearby grid cells) and synchronous half-penetration corrections averaged over overlapping
 * neighbors. Initial centers are input positions; radii are sqrt(value * areaScale / pi).
 * Origin attraction preserves geography; no GeoDa coordinate parity is claimed. Residual penetration alone
 * determines convergence, measured against `overlapTolerance` or four f32 ulps of the largest
 * coordinate, whichever is larger, because f32 centers cannot resolve smaller penetration. A
 * stationary overlapping layout is never called converged.
 * Zero values are valid, stationary zero-radius points and exert no force. Negative/missing
 * values, nonfinite coordinates, |coordinate| > 1e15, or radius > 1e15 are excluded and emit
 * NaN positions/radii. Source IDs are copied even for excluded rows. No data is read back.
 * Inputs must be packed views; caller chunk ownership is unchanged and never destroyed.
 */
export class GPUDorlingCartogram implements GPUCommandNodeProducer {
  readonly id: string;
  readonly props: GPUDorlingCartogramProps;
  readonly spatialContext: GPUSpatialContext;
  readonly maximumIterations: number;
  readonly areaScale: number;
  readonly overlapTolerance: number;

  constructor(props: GPUDorlingCartogramProps) {
    this.props = props;
    this.id = props.id ?? 'dorling-cartogram';
    this.spatialContext = resolveGPUSpatialContext(
      this.id,
      props.spatialContext,
      GPU_PLANAR_SPATIAL_CONTEXT_SUPPORT
    );
    this.maximumIterations = props.maximumIterations ?? 100;
    this.areaScale = props.areaScale ?? 1;
    this.overlapTolerance = props.overlapTolerance ?? 0.001;
    validateCapacity(
      this.id,
      'rows',
      props.positions.length,
      0,
      MAXIMUM_GPU_DORLING_CARTOGRAM_ROWS
    );
    validateCapacity(this.id, 'maximumIterations', this.maximumIterations, 0, 256);
    for (const value of [this.areaScale, this.overlapTolerance]) {
      if (
        !Number.isFinite(value) ||
        value < 0 ||
        value > 1e15 ||
        (Math.fround(value) !== value && Math.fround(value) === 0)
      ) {
        throw new Error(
          `${this.id} scale and tolerance must be finite nonnegative float32 values at most 1e15`
        );
      }
    }
    if (
      !Number.isFinite(props.attraction ?? 0.01) ||
      (props.attraction ?? 0.01) < 0 ||
      (props.attraction ?? 0.01) > 1
    ) {
      throw new Error(`${this.id} attraction must be in [0,1]`);
    }
    const rows = props.positions.length;
    const entries = [
      [props.positions, 'float32x2', rows],
      [props.values, 'float32', rows],
      [props.sourceIds, 'uint32', rows],
      [props.output.positions, 'float32x2', rows],
      [props.output.radii, 'float32', rows],
      [props.output.sourceIds, 'uint32', rows],
      [props.output.metrics, 'float32x2', 1],
      [props.output.status, 'uint32', 4]
    ] as const;
    for (const [view, format, length] of entries) {
      if (view) {
        validatePackedView(view, [format], this.id);
        if (view.length !== length) {
          throw new Error(`${this.id} view length mismatch`);
        }
      }
    }
    const outputs = Object.values(props.output);
    validateGraphOutputsDisjointFromInputs(this.id, outputs, [
      props.positions,
      props.values,
      props.sourceIds
    ]);
    outputs.forEach((output, index) =>
      validateGraphOutputsDisjointFromInputs(this.id, [output], outputs.slice(index + 1))
    );
  }

  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, maximumIterations} = this;
    const {output} = props;
    validateGraphViewsBelongToGraph(id, graph, [
      props.positions,
      props.values,
      props.sourceIds,
      ...Object.values(output)
    ]);
    const rows = props.positions.length;
    if (rows === 0) {
      return [
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-empty`,
          operation: 'GPUDorlingCartogram',
          variant: 'empty',
          invocationCount: 1,
          bindings: [
            {name: 'metrics', view: output.metrics, type: 'f32', access: 'read_write'},
            {name: 'status', view: output.status, type: 'u32', access: 'read_write'}
          ],
          body: 'metrics[metricsOffset] = 0.0; metrics[metricsOffset+1u] = 0.0; status[statusOffset] = 0u; status[statusOffset+1u] = 1u; status[statusOffset+2u] = 0u; status[statusOffset+3u] = 0u;'
        })
      ];
    }
    const scratch = createTransientView(graph, `${id}-scratch`, 'float32x2', Math.max(rows, 1));
    // One spare row after the per-row metrics holds the largest radius for the neighbor grid.
    const rowMetrics = createTransientView(graph, `${id}-row-metrics`, 'float32x4', rows + 1);
    const neighborGrid =
      props.neighborGrid === 'auto' || props.neighborGrid === undefined
        ? rows >= DORLING_NEIGHBOR_GRID_MINIMUM_ROWS
        : props.neighborGrid;
    const gridSide = Math.max(1, Math.min(Math.ceil(Math.sqrt(rows)), MAXIMUM_NEIGHBOR_GRID_SIDE));
    const gridCellCount = gridSide * gridSide;
    const declarations = `const ROWS: u32 = ${rows}u;
const TOLERANCE: f32 = ${getWGSLFloatLiteral(this.overlapTolerance)};
${IS_SPATIAL_FINITE_WGSL}
fn valid(value: f32) -> bool { return isSpatialFinite(value); }
${GET_SPATIAL_NAN_WGSL}
fn nan(row: u32) -> f32 { _ = row; return getSpatialNaN(); }`;
    const nodes: GPUCommandNode<Parameters>[] = [];
    nodes.push(
      createWGSLKernelNode(graph, {
        id: `${id}-initialize`,
        operation: 'GPUDorlingCartogram',
        variant: 'initialize',
        invocationCount: Math.max(rows, 1),
        declarations,
        bindings: [
          {name: 'initial', view: props.positions, type: 'f32', access: 'read'},
          {name: 'values', view: props.values, type: 'f32', access: 'read'},
          ...(props.sourceIds
            ? [{name: 'ids', view: props.sourceIds, type: 'u32' as const, access: 'read' as const}]
            : []),
          {name: 'centers', view: output.positions, type: 'f32', access: 'read_write'},
          {name: 'radii', view: output.radii, type: 'f32', access: 'read_write'},
          {name: 'sourceIds', view: output.sourceIds, type: 'u32', access: 'read_write'},
          {name: 'status', view: output.status, type: 'u32', access: 'read_write'}
        ],
        body: `if (index == 0u) { status[statusOffset] = 0u; status[statusOffset+1u] = 0u; status[statusOffset+2u] = 0u; status[statusOffset+3u] = 0u; }
if (index >= ROWS) { return; }
let center = vec2f(initial[initialOffset+index*2u], initial[initialOffset+index*2u+1u]);
let value = values[valuesOffset+index];
let radius = sqrt(value) * ${getWGSLFloatLiteral(Math.sqrt(this.areaScale / Math.PI))};
let included = valid(center.x) && valid(center.y) && all(abs(center) <= vec2f(1e15)) && valid(value) && value >= 0.0 && valid(radius) && radius <= 1e15;
centers[centersOffset+index*2u] = select(nan(index), center.x, included);
centers[centersOffset+index*2u+1u] = select(nan(index), center.y, included);
radii[radiiOffset+index] = select(nan(index), radius, included);
sourceIds[sourceIdsOffset+index] = ${props.sourceIds ? 'ids[idsOffset+index]' : 'index'};`
      })
    );
    // One workgroup tree-reduces the per-row metrics of state `step` into metrics and status.
    const reduceMetrics = (step: number) =>
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-diagnose-${step}`,
        operation: 'GPUDorlingCartogram',
        variant: 'diagnose',
        invocationCount: REDUCTION_WORKGROUP_SIZE,
        workgroupSize: REDUCTION_WORKGROUP_SIZE,
        guardIndex: false,
        declarations: `${declarations}
const REDUCTION_SIZE: u32 = ${REDUCTION_WORKGROUP_SIZE}u;
var<workgroup> partial: array<vec4f, ${REDUCTION_WORKGROUP_SIZE}>;`,
        bindings: [
          {name: 'rowMetrics', view: rowMetrics, type: 'f32', access: 'read'},
          {name: 'metrics', view: output.metrics, type: 'f32', access: 'read_write'},
          {name: 'status', view: output.status, type: 'u32', access: 'read_write'}
        ],
        body: `let lane = localInvocationIndex;
var accumulator = vec4f(0.0);
for (var row = lane; row < ROWS; row += REDUCTION_SIZE) {
  let base = rowMetricsOffset + row * 4u;
  accumulator = vec4f(max(accumulator.x, rowMetrics[base]), max(accumulator.y, rowMetrics[base + 1u]), accumulator.z + rowMetrics[base + 2u], max(accumulator.w, rowMetrics[base + 3u]));
}
partial[lane] = accumulator;
workgroupBarrier();
for (var stride = REDUCTION_SIZE / 2u; stride > 0u; stride = stride >> 1u) {
  if (lane < stride) {
    let first = partial[lane]; let second = partial[lane + stride];
    partial[lane] = vec4f(max(first.x, second.x), max(first.y, second.y), first.z + second.z, max(first.w, second.w));
  }
  workgroupBarrier();
}
if (lane == 0u) {
  let overlap = partial[0].x; let displacement = partial[0].y; let excluded = u32(partial[0].z);
  // Penetration cannot be resolved below a few ulps of the largest coordinate, so the tolerance
  // never drops under that floor.
  let tolerance = max(TOLERANCE, partial[0].w * ${COORDINATE_ULP_FLOOR});
  metrics[metricsOffset] = overlap; metrics[metricsOffset+1u] = displacement;
  ${step > 0 ? 'if (status[statusOffset+1u] == 0u) { status[statusOffset] += 1u; }' : ''}
  status[statusOffset+1u] = select(0u,1u,overlap <= tolerance);
  status[statusOffset+2u] = select(0u,1u,overlap > tolerance && status[statusOffset] == ${maximumIterations}u);
  status[statusOffset+3u] = excluded;
}`
      });
    // Neighbor grid: bounds of the centers, a GPUGridIndex over them, and rows sorted by ID inside
    // each cell so that sums run in a fixed order. Reach per row is its radius plus the largest radius.
    const gridBounds = neighborGrid
      ? createTransientView(graph, `${id}-grid-bounds`, 'float32', 4)
      : undefined;
    const gridCellOffsets = neighborGrid
      ? createTransientView(graph, `${id}-grid-cell-offsets`, 'uint32', gridCellCount + 1)
      : undefined;
    const gridObjectIds = neighborGrid
      ? createTransientView(graph, `${id}-grid-object-ids`, 'uint32', Math.max(rows, 1))
      : undefined;
    const gridCount = neighborGrid
      ? createTransientView(graph, `${id}-grid-count`, 'uint32', 1)
      : undefined;
    const gridOverflow = neighborGrid
      ? createTransientView(graph, `${id}-grid-overflow`, 'uint32', 1)
      : undefined;
    const gridDeclarations = `${declarations}
const GRID_SIDE: u32 = ${gridSide}u;
${GPU_GRID_COORDINATE_WGSL}
fn getGridCell(value: f32, minimum: f32, maximum: f32) -> u32 {
  return getGridCoordinate(clamp(value, minimum, maximum), minimum, maximum, GRID_SIDE);
}`;
    const gridBindings: WGSLKernelBinding[] = neighborGrid
      ? [
          {name: 'cellOffsets', view: gridCellOffsets!, type: 'u32', access: 'read'},
          {name: 'objectIds', view: gridObjectIds!, type: 'u32', access: 'read'},
          {name: 'gridBounds', view: gridBounds!, type: 'f32', access: 'read'}
        ]
      : [];
    // Runs `action` for every indexed row `other` in the cells near `center` (one cell of margin
    // absorbs rounding); `action` may `continue`. Needs `center` and `radius` in scope.
    const visitNeighbors = (
      action: string
    ) => `let reach = radius + rowMetrics[rowMetricsOffset + ROWS * 4u];
  let boundsMinimum = vec2f(gridBounds[gridBoundsOffset], gridBounds[gridBoundsOffset + 1u]);
  let boundsMaximum = vec2f(gridBounds[gridBoundsOffset + 2u], gridBounds[gridBoundsOffset + 3u]);
  let firstColumn = max(getGridCell(center.x - reach, boundsMinimum.x, boundsMaximum.x), 1u) - 1u;
  let lastColumn = min(getGridCell(center.x + reach, boundsMinimum.x, boundsMaximum.x) + 1u, GRID_SIDE - 1u);
  let firstRow = max(getGridCell(center.y - reach, boundsMinimum.y, boundsMaximum.y), 1u) - 1u;
  let lastRow = min(getGridCell(center.y + reach, boundsMinimum.y, boundsMaximum.y) + 1u, GRID_SIDE - 1u);
  for (var cellRow = firstRow; cellRow <= lastRow; cellRow++) {
    for (var cellColumn = firstColumn; cellColumn <= lastColumn; cellColumn++) {
      let cell = cellRow * GRID_SIDE + cellColumn;
      let lastSlot = cellOffsets[cellOffsetsOffset + cell + 1u];
      for (var slot = cellOffsets[cellOffsetsOffset + cell]; slot < lastSlot; slot++) {
        let other = objectIds[objectIdsOffset + slot];
        ${action}
      }
    }
  }`;
    const addGridBuildNodes = (centers: GraphDataView<'float32x2'>, step: string) => {
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-grid-bounds-${step}`,
          operation: 'GPUDorlingCartogram',
          variant: 'grid-bounds',
          invocationCount: REDUCTION_WORKGROUP_SIZE,
          workgroupSize: REDUCTION_WORKGROUP_SIZE,
          guardIndex: false,
          declarations: `${declarations}
var<workgroup> partialMinimum: array<vec2f, ${REDUCTION_WORKGROUP_SIZE}>;
var<workgroup> partialMaximum: array<vec2f, ${REDUCTION_WORKGROUP_SIZE}>;`,
          bindings: [
            {name: 'centers', view: centers, type: 'f32', access: 'read'},
            {name: 'gridBounds', view: gridBounds!, type: 'f32', access: 'read_write'}
          ],
          body: `let lane = localInvocationIndex;
var minimum = vec2f(3.4e38);
var maximum = vec2f(-3.4e38);
for (var row = lane; row < ROWS; row += ${REDUCTION_WORKGROUP_SIZE}u) {
  let center = vec2f(centers[centersOffset + row * 2u], centers[centersOffset + row * 2u + 1u]);
  if (valid(center.x) && valid(center.y)) { minimum = min(minimum, center); maximum = max(maximum, center); }
}
partialMinimum[lane] = minimum;
partialMaximum[lane] = maximum;
workgroupBarrier();
for (var stride = ${REDUCTION_WORKGROUP_SIZE / 2}u; stride > 0u; stride = stride >> 1u) {
  if (lane < stride) {
    partialMinimum[lane] = min(partialMinimum[lane], partialMinimum[lane + stride]);
    partialMaximum[lane] = max(partialMaximum[lane], partialMaximum[lane + stride]);
  }
  workgroupBarrier();
}
if (lane == 0u) {
  let empty = partialMinimum[0].x > partialMaximum[0].x;
  let low = select(partialMinimum[0], vec2f(0.0), empty);
  let high = select(partialMaximum[0], vec2f(0.0), empty);
  gridBounds[gridBoundsOffset] = low.x; gridBounds[gridBoundsOffset + 1u] = low.y;
  gridBounds[gridBoundsOffset + 2u] = high.x; gridBounds[gridBoundsOffset + 3u] = high.y;
}`
        }),
        ...new GPUGridIndex({
          id: `${id}-grid-index-${step}`,
          positions: centers,
          gridSize: [gridSide, gridSide],
          bounds: [0, 0, 1, 1],
          boundsBuffer: gridBounds!,
          cellOffsets: gridCellOffsets!,
          objectIds: gridObjectIds!,
          count: gridCount!,
          overflow: gridOverflow!
        }).getCommandNodes(graph),
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-grid-sort-${step}`,
          operation: 'GPUDorlingCartogram',
          variant: 'grid-sort',
          invocationCount: gridCellCount,
          declarations: 'const GAPS = array<u32, 8>(701u, 301u, 132u, 57u, 23u, 10u, 4u, 1u);',
          bindings: [
            {name: 'cellOffsets', view: gridCellOffsets!, type: 'u32', access: 'read'},
            {name: 'objectIds', view: gridObjectIds!, type: 'u32', access: 'read_write'}
          ],
          body: `let first = cellOffsets[cellOffsetsOffset + index];
  let last = cellOffsets[cellOffsetsOffset + index + 1u];
  if (last <= first + 1u) { return; }
  // Shell sort with Ciura gaps; cells hold few rows unless circles pile up.
  for (var gapIndex = 0u; gapIndex < 8u; gapIndex++) {
    let gap = GAPS[gapIndex];
    for (var slot = first + gap; slot < last; slot++) {
      let value = objectIds[objectIdsOffset + slot];
      var hole = slot;
      loop {
        if (hole < first + gap) { break; }
        let earlier = objectIds[objectIdsOffset + hole - gap];
        if (earlier <= value) { break; }
        objectIds[objectIdsOffset + hole] = earlier;
        hole = hole - gap;
      }
      objectIds[objectIdsOffset + hole] = value;
    }
  }`
        })
      );
    };
    // Final state metrics come from a dedicated all-pairs pass; earlier states reuse the
    // penetrations that the following relax pass computes anyway.
    const diagnose = (centers: GraphDataView<'float32x2'>, step: number) => {
      if (neighborGrid) {
        addGridBuildNodes(centers, `diagnose-${step}`);
      }
      const diagnoseRow = `if (index >= ROWS) { return; }
let radius = radii[radiiOffset+index]; var overlap = 0.0; var displacement = 0.0; var coordinate = 0.0;
if (valid(radius)) {
  let center = vec2f(centers[centersOffset+index*2u],centers[centersOffset+index*2u+1u]);
  displacement = distance(center,vec2f(initial[initialOffset+index*2u],initial[initialOffset+index*2u+1u]));
  coordinate = max(abs(center.x), abs(center.y));
  if (radius > 0.0) {
    ${
      neighborGrid
        ? visitNeighbors(`let otherRadius = radii[radiiOffset+other];
        if (other == index || !valid(otherRadius) || otherRadius == 0.0) { continue; }
        let otherCenter = vec2f(centers[centersOffset+other*2u],centers[centersOffset+other*2u+1u]);
        overlap = max(overlap,radius+otherRadius-distance(center,otherCenter));`)
        : `for (var other = index+1u; other < ROWS; other++) {
      let otherRadius = radii[radiiOffset+other];
      if (!valid(otherRadius) || otherRadius == 0.0) { continue; }
      let otherCenter = vec2f(centers[centersOffset+other*2u],centers[centersOffset+other*2u+1u]);
      overlap = max(overlap,radius+otherRadius-distance(center,otherCenter));
    }`
    }
  }
}
rowMetrics[rowMetricsOffset+index*4u] = overlap;
rowMetrics[rowMetricsOffset+index*4u+1u] = displacement;
rowMetrics[rowMetricsOffset+index*4u+2u] = select(1.0,0.0,valid(radius));
rowMetrics[rowMetricsOffset+index*4u+3u] = coordinate;`;
      nodes.push(
        ...createTiledWGSLKernelNodes<Parameters>(
          graph,
          {
            id: `${id}-row-diagnose-${step}`,
            operation: 'GPUDorlingCartogram',
            variant: 'row-diagnose',
            invocationCount: Math.max(rows, 1),
            declarations: neighborGrid ? gridDeclarations : declarations,
            bindings: [
              {name: 'centers', view: centers, type: 'f32', access: 'read'},
              {name: 'initial', view: props.positions, type: 'f32', access: 'read'},
              {name: 'radii', view: output.radii, type: 'f32', access: 'read'},
              {name: 'rowMetrics', view: rowMetrics, type: 'f32', access: 'read_write'},
              ...gridBindings
            ],
            body: diagnoseRow
          },
          props.rowTileSize
        )
      );
      nodes.push(reduceMetrics(step));
    };
    if (neighborGrid) {
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-maximum-radius`,
          operation: 'GPUDorlingCartogram',
          variant: 'maximum-radius',
          invocationCount: REDUCTION_WORKGROUP_SIZE,
          workgroupSize: REDUCTION_WORKGROUP_SIZE,
          guardIndex: false,
          declarations: `${declarations}
var<workgroup> partial: array<f32, ${REDUCTION_WORKGROUP_SIZE}>;`,
          bindings: [
            {name: 'radii', view: output.radii, type: 'f32', access: 'read'},
            {name: 'rowMetrics', view: rowMetrics, type: 'f32', access: 'read_write'}
          ],
          body: `let lane = localInvocationIndex;
var maximum = 0.0;
for (var row = lane; row < ROWS; row += ${REDUCTION_WORKGROUP_SIZE}u) {
  let radius = radii[radiiOffset + row];
  if (valid(radius)) { maximum = max(maximum, radius); }
}
partial[lane] = maximum;
workgroupBarrier();
for (var stride = ${REDUCTION_WORKGROUP_SIZE / 2}u; stride > 0u; stride = stride >> 1u) {
  if (lane < stride) { partial[lane] = max(partial[lane], partial[lane + stride]); }
  workgroupBarrier();
}
if (lane == 0u) { rowMetrics[rowMetricsOffset + ROWS * 4u] = partial[0]; }`
        })
      );
    }
    if (maximumIterations === 0) {
      diagnose(output.positions, 0);
    }
    for (let step = 1; step <= maximumIterations; step++) {
      const source = step % 2 ? output.positions : scratch;
      const target = step % 2 ? scratch : output.positions;
      // Relax moves state step-1 to a tentative state step and records the metrics of state
      // step-1 (the penetrations are already computed). The reduction then decides convergence
      // of state step-1; a converged state is held by the commit pass, as if relax never moved it.
      if (neighborGrid) {
        addGridBuildNodes(source, `relax-${step}`);
      }
      nodes.push(
        ...createTiledWGSLKernelNodes(
          graph,
          {
            id: `${id}-relax-${step}`,
            operation: 'GPUDorlingCartogram',
            variant: 'relax',
            invocationCount: Math.max(rows, 1),
            declarations: neighborGrid ? gridDeclarations : declarations,
            bindings: [
              {name: 'centers', view: source, type: 'f32', access: 'read'},
              {name: 'destinations', view: target, type: 'f32', access: 'read_write'},
              {name: 'radii', view: output.radii, type: 'f32', access: 'read'},
              // The grid kernel is at the 8-binding limit, so it drops the converged check; the
              // commit pass below repeats it (a converged state is restored from its source).
              ...(neighborGrid
                ? []
                : [
                    {
                      name: 'status',
                      view: output.status,
                      type: 'u32' as const,
                      access: 'read' as const
                    }
                  ]),
              {name: 'initial', view: props.positions, type: 'f32', access: 'read'},
              {name: 'rowMetrics', view: rowMetrics, type: 'f32', access: 'read_write'},
              ...gridBindings
            ],
            body: `if (index >= ROWS) { return; }
let center = vec2f(centers[centersOffset+index*2u],centers[centersOffset+index*2u+1u]);
let radius = radii[radiiOffset+index]; var correction = vec2f(0.0); var neighbors = 0u;
var overlap = 0.0; var displacement = 0.0; var coordinate = 0.0;
let origin = vec2f(initial[initialOffset+index*2u],initial[initialOffset+index*2u+1u]);
if (valid(radius)) { displacement = distance(center, origin); coordinate = max(abs(center.x), abs(center.y)); }
if (valid(radius) && radius > 0.0) {
  ${
    neighborGrid
      ? visitNeighbors(`let otherRadius = radii[radiiOffset+other];
    if (other == index || !valid(otherRadius) || otherRadius <= 0.0) { continue; }
    let delta = center-vec2f(centers[centersOffset+other*2u],centers[centersOffset+other*2u+1u]);
    let separation = length(delta); let penetration = radius+otherRadius-separation;
    if (penetration > 0.0) {
      overlap = max(overlap, penetration);
      var direction = vec2f(select(-1.0,1.0,index > other),0.0);
      if (separation > 0.0) { direction = delta/separation; }
      correction += direction*(penetration*0.5); neighbors++;
    }`)
      : `for (var other = 0u; other < ROWS; other++) {
    let otherRadius = radii[radiiOffset+other];
    if (other == index || !valid(otherRadius) || otherRadius <= 0.0) { continue; }
    let delta = center-vec2f(centers[centersOffset+other*2u],centers[centersOffset+other*2u+1u]);
    let separation = length(delta); let penetration = radius+otherRadius-separation;
    if (penetration > 0.0) {
      overlap = max(overlap, penetration);
      var direction = vec2f(select(-1.0,1.0,index > other),0.0);
      if (separation > 0.0) { direction = delta/separation; }
      correction += direction*(penetration*0.5); neighbors++;
    }
  }`
  }
}
rowMetrics[rowMetricsOffset+index*4u] = overlap;
rowMetrics[rowMetricsOffset+index*4u+1u] = displacement;
rowMetrics[rowMetricsOffset+index*4u+2u] = select(1.0,0.0,valid(radius));
rowMetrics[rowMetricsOffset+index*4u+3u] = coordinate;
var moved = center;
if (valid(radius) && radius > 0.0${neighborGrid ? '' : ' && status[statusOffset+1u] == 0u'}) {
  moved = center+correction/f32(max(neighbors,1u))+(origin-center)*${getWGSLFloatLiteral(props.attraction ?? 0.01)};
}
destinations[destinationsOffset+index*2u] = moved.x; destinations[destinationsOffset+index*2u+1u] = moved.y;`
          },
          props.rowTileSize
        )
      );
      nodes.push(
        reduceMetrics(step - 1),
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-commit-${step}`,
          operation: 'GPUDorlingCartogram',
          variant: 'commit',
          invocationCount: Math.max(rows * 2, 1),
          bindings: [
            {name: 'source', view: source, type: 'f32', access: 'read'},
            {name: 'destinations', view: target, type: 'f32', access: 'read_write'},
            {name: 'status', view: output.status, type: 'u32', access: 'read'}
          ],
          body: `if (index < ${rows * 2}u && status[statusOffset+1u] != 0u) { destinations[destinationsOffset+index] = source[sourceOffset+index]; }`
        })
      );
      if (step === maximumIterations) {
        diagnose(target, step);
      }
    }
    if (maximumIterations % 2) {
      nodes.push(
        createWGSLKernelNode(graph, {
          id: `${id}-copy`,
          operation: 'GPUDorlingCartogram',
          variant: 'copy',
          invocationCount: Math.max(rows * 2, 1),
          bindings: [
            {name: 'source', view: scratch, type: 'f32', access: 'read'},
            {name: 'destinations', view: output.positions, type: 'f32', access: 'read_write'}
          ],
          body: `if (index < ${rows * 2}u) { destinations[destinationsOffset+index] = source[sourceOffset+index]; }`
        })
      );
    }
    return nodes;
  }
}

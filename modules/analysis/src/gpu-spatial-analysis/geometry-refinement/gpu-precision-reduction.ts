// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {
  GPUCommandGraph,
  GPUCommandNode,
  GPUCommandNodeProducer,
  GraphDataView
} from '@luma.gl/graph/gpu-core';
import {createTransientView, GPUScan} from '@luma.gl/graph/gpu-core';
import {GPUMakeValid, type GPUMakeValidProps} from '../geometry-topology/gpu-make-valid';
import {
  validateSinglePackedView,
  getGraphOutputViews,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  createWGSLKernelNode,
  getWGSLFloatLiteral
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  resolveGPUSpatialContext,
  type GPUSpatialContext
} from '../../analysis-core/contracts/spatial-context';
import {getReadPointWGSL} from '../shared/spatial-wgsl-point-helpers';

/** Polygon precision uses the existing bounded topology output and diagnostics. */
export type GPUPolygonPrecisionReductionProps = Omit<
  GPUMakeValidProps,
  'precision' | 'snapRounding'
> & {
  /** Positive finite grid step. Coordinates are transformed to grid units before exact lattice noding. */
  gridSize: number;
  /** [originX, originY, actualGridSize, 0], always inspect the actual step. */
  lattice: GraphDataView<'float32'>;
  /** Collapsed polygons are empty in both policies, as in GEOS set_precision. */
  mode?: 'valid-output' | 'keep-collapsed';
};

/**
 * Hot-pixel noding plus even-odd polygon reconstruction. This is topology precision
 * reduction, not independent coordinate rounding. Crossing and surviving original
 * vertices share an explicit lattice; collapsed rings are dropped, remnants remain
 * in topology diagnostics. Implicit ring closures are materialized before noding. The supported grid
 * is any positive float32 grid size; grid_size=0 (double-precision identity) is excluded.
 *
 * Coordinates are divided by `gridSize` and multiplied back in f32. For power-of-two grid sizes this
 * is exact; for other sizes the result is the nearest f32 to the grid point, which may differ by one
 * ulp from GEOS `set_precision`, and when `|x| / gridSize` exceeds 2^22 (the lattice limit of
 * `GPUSnapRoundedNoding`) the requested grid is no longer resolvable in f32.
 */
export class GPUPolygonPrecisionReduction implements GPUCommandNodeProducer {
  readonly id: string;
  readonly props: GPUPolygonPrecisionReductionProps;
  constructor(props: GPUPolygonPrecisionReductionProps) {
    this.id = props.id ?? 'polygon-precision-reduction';
    this.props = props;
    if (!(Number.isFinite(Math.fround(props.gridSize)) && Math.fround(props.gridSize) > 0)) {
      throw new Error('Precision grid must be finite and positive');
    }
    validateSinglePackedView(props.lattice, ['float32'], this.id);
    if (props.lattice.length !== 4) {
      throw new Error('Precision lattice needs four scalars');
    }
    validateGraphOutputsDisjointFromInputs(
      this.id,
      [props.lattice, ...getGraphOutputViews(props.output)],
      [
        props.polygons.positions,
        props.polygons.ringOffsets,
        props.polygons.polygonOffsets,
        props.polygons.featureOffsets,
        props.polygons.sourceIds
      ]
    );
  }
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {props, id} = this;
    validateGraphViewsBelongToGraph(id, graph, [
      props.polygons.positions,
      props.polygons.ringOffsets,
      props.polygons.polygonOffsets,
      props.polygons.featureOffsets,
      props.polygons.sourceIds,
      props.lattice,
      ...getGraphOutputViews(props.output)
    ]);
    const ringCount = props.polygons.ringOffsets.length - 1;
    const positions = createTransientView(
      graph,
      `${id}-closed-positions`,
      'float32x2',
      props.polygons.positions.length + ringCount
    );
    const ringOffsets = createTransientView(graph, `${id}-closed-offsets`, 'uint32', ringCount + 1);
    const closedLengths = createTransientView(
      graph,
      `${id}-closed-lengths`,
      'uint32',
      ringCount + 1
    );
    const gridLiteral = getWGSLFloatLiteral(props.gridSize);
    // Closed length per ring, a prefix sum for ring offsets, then a parallel per-ring copy.
    const closeRings = [
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-close-lengths`,
        operation: 'GPUPolygonPrecisionReduction',
        variant: 'closed-lengths',
        invocationCount: ringCount + 1,
        bindings: [
          {name: 'positions', view: props.polygons.positions, type: 'f32', access: 'read'},
          {name: 'offsets', view: props.polygons.ringOffsets, type: 'u32', access: 'read'},
          {name: 'lengths', view: closedLengths, type: 'u32', access: 'read_write'}
        ],
        declarations: `${getReadPointWGSL('point', 'positions')}`,
        body: `var length = 0u;
if (index < ${ringCount}u) {
  let first = offsets[offsetsOffset + index]; let last = offsets[offsetsOffset + index + 1u];
  length = last - first;
  if (last > first && any(point(first) != point(last - 1u))) { length++; }
}
lengths[lengthsOffset + index] = length;`
      }),
      ...new GPUScan({
        id: `${id}-close-scan`,
        input: closedLengths,
        output: ringOffsets,
        mode: 'exclusive'
      }).getCommandNodes(graph),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-close-rings`,
        operation: 'GPUPolygonPrecisionReduction',
        variant: 'close-rings',
        invocationCount: ringCount,
        bindings: [
          {name: 'positions', view: props.polygons.positions, type: 'f32', access: 'read'},
          {name: 'offsets', view: props.polygons.ringOffsets, type: 'u32', access: 'read'},
          {name: 'outPositions', view: positions, type: 'f32', access: 'read_write'},
          {name: 'outOffsets', view: ringOffsets, type: 'u32', access: 'read'}
        ],
        declarations: `${getReadPointWGSL('point', 'positions')}
fn writePoint(vertex: u32, value: vec2f) { outPositions[outPositionsOffset + vertex * 2u] = value.x / ${gridLiteral}; outPositions[outPositionsOffset + vertex * 2u + 1u] = value.y / ${gridLiteral}; }`,
        body: `var written = outOffsets[outOffsetsOffset + index];
let first = offsets[offsetsOffset + index]; let last = offsets[offsetsOffset + index + 1u];
for (var vertex = first; vertex < last; vertex++) { writePoint(written, point(vertex)); written++; }
if (last > first && any(point(first) != point(last - 1u))) { writePoint(written, point(first)); }`
      })
    ];
    return [
      ...closeRings,
      ...new GPUMakeValid({
        ...props,
        id,
        polygons: {...props.polygons, positions, ringOffsets},
        precision: {
          vertexTolerance: 0.25,
          storage: 'float32',
          predicates: 'exact-or-uncertain'
        },
        snapRounding: {gridSize: 1, lattice: props.lattice}
      }).getCommandNodes(graph),
      ...[
        props.output.status,
        props.output.nodedSegments.status,
        props.output.directedEdges.status
      ].map((status, statusIndex) => {
        const views = getGraphOutputViews(status);
        return createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-empty-status-${statusIndex}`,
          operation: 'GPUPolygonPrecisionReduction',
          invocationCount: 1,
          bindings: [
            {name: 'offsets', view: props.polygons.ringOffsets, type: 'u32', access: 'read'},
            {name: 'emptyLattice', view: props.lattice, type: 'f32', access: 'read_write'},
            ...views.map((view, viewIndex) => ({
              name: `status${viewIndex}`,
              view,
              type: 'u32' as const,
              access: 'read_write' as const
            }))
          ],
          body: `for (var ring = 0u; ring < ${ringCount}u; ring++) { if (offsets[offsetsOffset + ring] != offsets[offsetsOffset + ring + 1u]) { return; } }
emptyLattice[emptyLatticeOffset] = 0.0; emptyLattice[emptyLatticeOffset + 1u] = 0.0; emptyLattice[emptyLatticeOffset + 2u] = 1.0; emptyLattice[emptyLatticeOffset + 3u] = 0.0;
${views.map((_, viewIndex) => `status${viewIndex}[status${viewIndex}Offset] = 0u;`).join('\n')}`
        });
      }),
      ...[
        props.output.polygons.positions,
        props.output.nodedSegments.endpoints,
        props.output.directedEdges.endpoints
      ].map((view, outputIndex) =>
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-restore-grid-${outputIndex}`,
          operation: 'GPUPolygonPrecisionReduction',
          invocationCount: view.length * (view.format === 'float32x2' ? 2 : 4),
          bindings: [{name: 'values', view, type: 'f32', access: 'read_write'}],
          body: `values[valuesOffset + index] *= ${getWGSLFloatLiteral(props.gridSize)};`
        })
      ),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-restore-lattice`,
        operation: 'GPUPolygonPrecisionReduction',
        invocationCount: 3,
        bindings: [{name: 'lattice', view: props.lattice, type: 'f32', access: 'read_write'}],
        body: `lattice[latticeOffset + index] *= ${getWGSLFloatLiteral(props.gridSize)};`
      })
    ];
  }
}

export type GPULinePrecisionReductionProps = {
  id?: string;
  spatialContext?: GPUSpatialContext;
  positions: GraphDataView<'float32x2'>;
  lineOffsets: GraphDataView<'uint32'>;
  gridSize: number;
  mode: 'valid-output' | 'keep-collapsed';
  output: {
    positions: GraphDataView<'float32x2'>;
    lineOffsets: GraphDataView<'uint32'>;
    /** Per line: 0 regular/empty, 1 removed collapse, 2 retained collapse, 3 nonfinite. */
    status: GraphDataView<'uint32'>;
  };
};

/** Grid-rounded lines with explicit collapse policy and per-source line status. */
export class GPULinePrecisionReduction implements GPUCommandNodeProducer {
  readonly id: string;
  readonly props: GPULinePrecisionReductionProps;
  readonly spatialContext: GPUSpatialContext;
  constructor(props: GPULinePrecisionReductionProps) {
    this.id = props.id ?? 'line-precision-reduction';
    this.props = props;
    this.spatialContext = resolveGPUSpatialContext(this.id, props.spatialContext);
    if (!(Number.isFinite(Math.fround(props.gridSize)) && Math.fround(props.gridSize) > 0)) {
      throw new Error('Invalid precision grid');
    }
    for (const view of [props.positions, props.output.positions]) {
      validateSinglePackedView(view, ['float32x2'], this.id);
    }
    for (const view of [props.lineOffsets, props.output.lineOffsets, props.output.status]) {
      validateSinglePackedView(view, ['uint32'], this.id);
    }
    if (
      props.output.positions.length < props.positions.length ||
      props.output.lineOffsets.length !== props.lineOffsets.length ||
      props.output.status.length !== props.lineOffsets.length - 1
    ) {
      throw new Error('Precision output length mismatch');
    }
    validateGraphOutputsDisjointFromInputs(this.id, Object.values(props.output), [
      props.positions,
      props.lineOffsets
    ]);
  }
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {props, id} = this;
    validateGraphViewsBelongToGraph(id, graph, [
      props.positions,
      props.lineOffsets,
      ...Object.values(props.output)
    ]);
    const lineCount = props.lineOffsets.length - 1;
    // Rounded lines are staged at their input offsets (a line never grows past its input length),
    // then compacted by a prefix sum of kept counts.
    const staged = createTransientView(
      graph,
      `${id}-staged`,
      'float32x2',
      Math.max(props.positions.length, 1)
    );
    const keptCounts = createTransientView(graph, `${id}-kept-counts`, 'uint32', lineCount + 1);
    return [
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-reduce`,
        operation: 'GPULinePrecisionReduction',
        variant: 'round-lines',
        invocationCount: lineCount + 1,
        bindings: [
          {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
          {name: 'offsets', view: props.lineOffsets, type: 'u32', access: 'read'},
          {name: 'outPositions', view: staged, type: 'f32', access: 'read_write'},
          {name: 'keptCounts', view: keptCounts, type: 'u32', access: 'read_write'},
          {name: 'status', view: props.output.status, type: 'u32', access: 'read_write'}
        ],
        declarations: `const GRID: f32 = ${getWGSLFloatLiteral(props.gridSize)};
${getReadPointWGSL('point', 'positions')}
fn writePoint(vertex: u32, value: vec2f) { outPositions[outPositionsOffset + vertex * 2u] = value.x; outPositions[outPositionsOffset + vertex * 2u + 1u] = value.y; }`,
        body: `if (index == ${lineCount}u) { keptCounts[keptCountsOffset + index] = 0u; return; }
let line = index;
let first = min(offsets[offsetsOffset + line], ${props.positions.length}u);
let last = min(offsets[offsetsOffset + line + 1u], ${props.positions.length}u);
var kept = 0u; var previous = vec2f(0.0); var invalid = false;
for (var vertex = first; vertex < last; vertex++) {
  let value = point(vertex);
  if (any((bitcast<vec2u>(value) & vec2u(0x7f800000u)) == vec2u(0x7f800000u))) { invalid = true; }
  // Round half up from the exact fraction. floor(scaled + 0.5) is wrong in f32 wherever the sum
  // is inexact: odd integers from 2^23 move to the even neighbour, and 0.49999997 rounds to 1.
  let scaled = value / GRID;
  let lower = floor(scaled);
  let rounded = (lower + step(vec2f(0.5), scaled - lower)) * GRID;
  if (any((bitcast<vec2u>(rounded) & vec2u(0x7f800000u)) == vec2u(0x7f800000u))) { invalid = true; }
  if (kept == 0u || any(rounded != previous)) { writePoint(first + kept, rounded); previous = rounded; kept++; }
}
var resultStatus = 0u;
if (invalid) { kept = 0u; resultStatus = 3u; }
else if (kept == 1u) {
  ${props.mode === 'keep-collapsed' ? 'if (last - first >= 2u) { writePoint(first + 1u, previous); kept = 2u; resultStatus = 2u; } else { kept = 0u; resultStatus = 1u; }' : 'kept = 0u; resultStatus = 1u;'}
}
keptCounts[keptCountsOffset + line] = kept; status[statusOffset + line] = resultStatus;`
      }),
      ...new GPUScan({
        id: `${id}-scan`,
        input: keptCounts,
        output: props.output.lineOffsets,
        mode: 'exclusive'
      }).getCommandNodes(graph),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-compact`,
        operation: 'GPULinePrecisionReduction',
        variant: 'compact-lines',
        invocationCount: lineCount,
        bindings: [
          {name: 'offsets', view: props.lineOffsets, type: 'u32', access: 'read'},
          {name: 'staged', view: staged, type: 'f32', access: 'read'},
          {name: 'keptCounts', view: keptCounts, type: 'u32', access: 'read'},
          {name: 'outOffsets', view: props.output.lineOffsets, type: 'u32', access: 'read'},
          {name: 'outPositions', view: props.output.positions, type: 'f32', access: 'read_write'}
        ],
        body: `let first = min(offsets[offsetsOffset + index], ${props.positions.length}u);
let destination = outOffsets[outOffsetsOffset + index];
for (var vertex = 0u; vertex < keptCounts[keptCountsOffset + index]; vertex++) {
  outPositions[outPositionsOffset + (destination + vertex) * 2u] = staged[stagedOffset + (first + vertex) * 2u];
  outPositions[outPositionsOffset + (destination + vertex) * 2u + 1u] = staged[stagedOffset + (first + vertex) * 2u + 1u];
}`
      })
    ];
  }
}

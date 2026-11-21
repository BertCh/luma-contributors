// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {validateGeometryReadWriteViews} from '../geometry-edit/geometry-storage-validation';

import {
  validateRefinementOutputLength,
  validateRefinementNumber
} from './refinement-output-validation';
import {validateGraphChunkBindingRanges} from '../../analysis-core/kernels/graph-chunk-dispatch';
import {validateOutlineOutputPasses} from '../outline-geometry/validate-outline-output-passes';
import {validateRefinementScratchByteLengths} from './refinement-storage-validation';
import type {
  GPUCommandGraph,
  GPUCommandNode,
  GPUCommandNodeProducer,
  GraphDataView
} from '@luma.gl/graph/gpu-core';
import {getGPUSnapRoundedNodingMaximumVertexCount} from '../geometry-topology/gpu-snap-rounded-noding';
import {createTransientView, GPUScan} from '@luma.gl/graph/gpu-core';
import {GPUMakeValid, type GPUMakeValidProps} from '../geometry-topology/gpu-make-valid';
import {
  snapshotTopologyStatus,
  validatePolygonizeStatus,
  validateSupportedTopologyStatus
} from '../geometry-topology/topology-status';
import {
  validateSinglePackedView,
  validateOption,
  getGraphOutputViews,
  validateGraphOutputsDisjointFromInputs,
  validateMaximumCount,
  validateRequirement,
  getNextPowerOfTwo
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
import {GRID_HALF_UP_ROUNDING_WGSL} from '../geometry-edit/half-up-rounding';
import {snapshotConstructionFields} from '../../analysis-core/kernels/construction-snapshot';
import {snapshotRefinementSpatialContext} from './refinement-construction-snapshot';

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
 * Quotient of an original coordinate by a grid step whose half-up cell is certified from the
 * original operands. The continuous quotient is kept when division chose the right cell; otherwise
 * it moves just inside the certified cell, or onto the certified integer when that boundary is not
 * representable (cells of 2^23 and above). @internal
 */
export const POLYGON_PRECISION_COORDINATE_WGSL = /* wgsl */ `${GRID_HALF_UP_ROUNDING_WGSL}
fn normalizePrecisionCoordinate(value: f32, step: f32) -> f32 {
  let quotient = bitcast<f32>(bitcast<u32>(value / step));
  let certified = roundGridHalfUp(value, 0.0, step);
  let candidate = roundHalfUp(quotient);
  if (candidate == certified) { return quotient; }
  let increase = certified > candidate;
  let boundary = select(certified + 0.5, certified - 0.5, increase);
  let bits = bitcast<u32>(boundary);
  let nudged = bitcast<f32>(select(bits - 1u, bits + 1u, increase == (boundary > 0.0)));
  // Where the half boundary rounds onto an integer, one step past it leaves the certified cell.
  if (roundHalfUp(nudged) != certified) { return certified; }
  return nudged;
}
`;

/**
 * Hot-pixel noding plus even-odd polygon reconstruction. This is topology precision
 * reduction, not independent coordinate rounding. Crossing and surviving original
 * vertices share an explicit lattice; collapsed rings are dropped, remnants remain
 * in topology diagnostics. Implicit ring closures are materialized before noding. The supported grid
 * is any positive float32 grid size; grid_size=0 (double-precision identity) is excluded.
 *
 * Endpoint cells are certified against the original float32 coordinate and grid step with
 * exact dyadic comparisons, including positive and negative half ties. Coordinates are divided
 * by `gridSize` and multiplied back in f32. For power-of-two grid sizes this
 * is exact; for other sizes the result is the nearest f32 to the grid point, which may differ by one
 * ulp from GEOS `set_precision`, and when `|x| / gridSize` exceeds 2^22 (the lattice limit of
 * `GPUSnapRoundedNoding`) the requested grid is no longer resolvable in f32.
 */
export class GPUPolygonPrecisionReduction implements GPUCommandNodeProducer {
  readonly id: string;
  readonly props: GPUPolygonPrecisionReductionProps;
  constructor(props: GPUPolygonPrecisionReductionProps) {
    const snapshot = snapshotConstructionFields(props, [
      'id',
      'spatialContext',
      'polygons',
      'intersectionCapacity',
      'gridSize',
      'lattice',
      'mode',
      'spatialSort',
      'leafCapacity',
      'output'
    ]);
    this.id = snapshot.id ?? 'polygon-precision-reduction';
    const output = snapshotConstructionFields(snapshot.output, [
      'nodedSegments',
      'directedEdges',
      'polygons',
      'diagnostics',
      'status'
    ]);
    const noded = snapshotConstructionFields(output.nodedSegments, [
      'endpoints',
      'sourceFeatureIds',
      'sourceRingIds',
      'sourceEdgeIds',
      'sourceStartParameters',
      'sourceEndParameters',
      'status'
    ]);
    const directed = snapshotConstructionFields(output.directedEdges, [
      'endpoints',
      'sourceSegmentIds',
      'sourceFeatureIds',
      'twinIds',
      'ringIds',
      'flags',
      'status'
    ]);
    // Validate the captured records: a snapshot reads every declared field once, including
    // inherited, non-enumerable and proxy-supplied destinations that report no membership.
    const nodedStatus = snapshotTopologyStatus(noded.status);
    const directedStatus = snapshotTopologyStatus(directed.status);
    for (const [name, status] of [
      ['nodedSegments.status', nodedStatus],
      ['directedEdges.status', directedStatus]
    ] as const) {
      validateRequirement(
        this.id,
        !('requiredRingCount' in status),
        `output.${name}.requiredRingCount is final-only; use output.${name}.requiredCount for segments or output.status.requiredRingCount for final rings`
      );
    }
    const finalStatus = snapshotTopologyStatus(output.status);
    for (const [name, status] of [
      ['nodedSegments.status', nodedStatus],
      ['directedEdges.status', directedStatus],
      ['status', finalStatus]
    ] as const) {
      validateSupportedTopologyStatus(this.id, status, `output.${name}`);
    }
    validatePolygonizeStatus(this.id, finalStatus);
    props = Object.freeze({
      ...snapshot,
      ...(snapshot.spatialContext
        ? {spatialContext: snapshotRefinementSpatialContext(snapshot.spatialContext)}
        : {}),
      polygons: snapshotConstructionFields(snapshot.polygons, [
        'kind',
        'positions',
        'ringOffsets',
        'polygonOffsets',
        'featureOffsets',
        'sourceIds'
      ]),
      output: Object.freeze({
        ...output,
        nodedSegments: Object.freeze({
          ...noded,
          status: nodedStatus
        }),
        directedEdges: Object.freeze({
          ...directed,
          status: directedStatus
        }),
        polygons: snapshotConstructionFields(output.polygons, [
          'kind',
          'positions',
          'ringOffsets',
          'polygonOffsets',
          'featureOffsets',
          'sourceIds'
        ]),
        diagnostics: snapshotConstructionFields(output.diagnostics, [
          'edgeClasses',
          'closedRingCount',
          'openChainCount',
          'cutEdgeCount',
          'dangleCount',
          'unusedEdgeCount'
        ]),
        status: finalStatus
      })
    });
    this.props = props;
    validateRefinementNumber(
      {id: this.id, contributor: 'GPUPolygonPrecisionReduction'},
      'gridSize',
      props.gridSize,
      true
    );
    validateOption(
      {id: this.id, contributor: 'GPUPolygonPrecisionReduction'},
      'mode',
      props.mode === undefined ? 'valid-output' : props.mode,
      ['valid-output', 'keep-collapsed']
    );

    validateSinglePackedView(props.lattice, ['float32'], `${this.id} lattice`, {
      id: this.id,
      contributor: 'GPUPolygonPrecisionReduction'
    });
    validateRefinementOutputLength(
      {id: this.id, contributor: 'GPUPolygonPrecisionReduction'},
      'lattice',
      props.lattice,
      4,
      '[originX, originY, actualGridSize, 0]'
    );
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
    validateGraphChunkBindingRanges(
      graph,
      id,
      Object.fromEntries(
        [
          props.polygons.positions,
          props.polygons.ringOffsets,
          props.polygons.polygonOffsets,
          props.polygons.featureOffsets,
          props.polygons.sourceIds,
          props.lattice,
          ...getGraphOutputViews(props.output.polygons),
          ...getGraphOutputViews(props.output.nodedSegments),
          ...getGraphOutputViews(props.output.directedEdges),
          ...getGraphOutputViews(props.output.diagnostics),
          ...getGraphOutputViews(props.output.status)
        ].map((view, index) => [`binding.${index}`, view])
      )
    );
    // Precision reduction closes rings once; make-valid materializes its implicit closures once
    // more before snap-rounding. Preflight that actual downstream row count and all larger tables.
    const inputRingCount = props.polygons.ringOffsets.length - 1;
    const nestedVertexCount = props.polygons.positions.length + 2 * inputRingCount;
    const hotCapacity = nestedVertexCount + props.intersectionCapacity;
    const segmentCapacity = props.output.nodedSegments.endpoints.length;
    const incidenceCapacity = Math.max(segmentCapacity + nestedVertexCount, 2 * hotCapacity);
    const ringCapacity = props.output.polygons.ringOffsets.length - 1;
    validateMaximumCount(
      id,
      'closed positions rows + intersectionCapacity',
      hotCapacity,
      getGPUSnapRoundedNodingMaximumVertexCount(graph.device, 0)
    );
    validateRefinementScratchByteLengths(graph, id, {
      closedPositions: nestedVertexCount * 8,
      closedOffsets: (inputRingCount + 1) * 4,
      hotPixelSegmentTables: hotCapacity * 128,
      crossingTerms: props.intersectionCapacity * 8 * 4,
      candidateListsAndScan: incidenceCapacity * 2 * 4,
      nodedLatticeEndpoints: segmentCapacity * 16,
      parityRecords: segmentCapacity * 5 * 4,
      parityHierarchy: (2 * getNextPowerOfTwo(segmentCapacity) - 1) * 8,
      ringBoundsAndProbes: ringCapacity * 16,
      ringShellRecords: ringCapacity * 3 * 4,
      ringShellHierarchy: (2 * getNextPowerOfTwo(ringCapacity) - 1) * 8,
      ...(props.leafCapacity ? {requestedHierarchy: (2 * props.leafCapacity - 1) * 8} : {})
    });
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
    const nonempty = createTransientView(graph, `${id}-nonempty`, 'uint32', 1);
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
        // Bounds and the lattice origin see certified quotients at the requested grid, so the
        // minimum certified cell is inside the lattice even when division rounds above it.
        declarations: `${getReadPointWGSL('point', 'positions')}
${POLYGON_PRECISION_COORDINATE_WGSL}
fn writePoint(vertex: u32, value: vec2f) { outPositions[outPositionsOffset + vertex * 2u] = normalizePrecisionCoordinate(value.x, ${gridLiteral}); outPositions[outPositionsOffset + vertex * 2u + 1u] = normalizePrecisionCoordinate(value.y, ${gridLiteral}); }`,
        body: `var written = outOffsets[outOffsetsOffset + index];
let first = offsets[offsetsOffset + index]; let last = offsets[offsetsOffset + index + 1u];
for (var vertex = first; vertex < last; vertex++) { writePoint(written, point(vertex)); written++; }
if (last > first && any(point(first) != point(last - 1u))) { writePoint(written, point(first)); }`
      })
    ];
    const topologyNodes = new GPUMakeValid({
      ...props,
      id,
      polygons: {...props.polygons, positions, ringOffsets},
      precision: {vertexTolerance: 0.25, storage: 'float32', predicates: 'exact-or-uncertain'},
      snapRounding: {gridSize: 1, lattice: props.lattice}
    }).getCommandNodes(graph);
    // Bounds and lattice sizing see the requested-grid quotient, corrected only when division
    // put it in the wrong cell. Correct the snap input again after sizing, using the original
    // stored coordinate and the actual step (the lattice cell may be coarser than the grid).
    const snapIndex = topologyNodes.findIndex(
      node => node.id === `${id}-polygonize-noding-snap-vertices`
    );
    const snapPositions = topologyNodes[snapIndex]?.resources?.find(
      resource =>
        'buffer' in resource &&
        'format' in resource.buffer &&
        resource.buffer.buffer.id === `${id}-ring-closure-positions`
    );
    if (!snapPositions || !('buffer' in snapPositions) || !('format' in snapPositions.buffer)) {
      throw new Error(`${id} precision snap input is missing from make-valid topology`);
    }
    const correctSnapInput = createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-correct-snap-input`,
      operation: 'GPUPolygonPrecisionReduction',
      variant: 'correct-snap-input',
      invocationCount: nestedVertexCount,
      bindings: [
        {name: 'positions', view: props.polygons.positions, type: 'f32', access: 'read'},
        {name: 'originalOffsets', view: props.polygons.ringOffsets, type: 'u32', access: 'read'},
        {name: 'closedOffsets', view: ringOffsets, type: 'u32', access: 'read'},
        {name: 'lattice', view: props.lattice, type: 'f32', access: 'read'},
        {name: 'snapPositions', view: snapPositions.buffer, type: 'f32', access: 'read_write'}
      ],
      declarations: `${getReadPointWGSL('point', 'positions')}
${POLYGON_PRECISION_COORDINATE_WGSL}`,
      body: `var ring = 0u; var high = ${ringCount}u;
while (ring + 1u < high) {
  let middle = (ring + high) / 2u;
  if (closedOffsets[closedOffsetsOffset + middle] + middle <= index) { ring = middle; } else { high = middle; }
}
let first = closedOffsets[closedOffsetsOffset + ring]; let last = closedOffsets[closedOffsetsOffset + ring + 1u];
var closedRow = index - ring;
if (index < first + ring || closedRow >= last) { closedRow = first; }
let originalFirst = originalOffsets[originalOffsetsOffset + ring]; let originalLast = originalOffsets[originalOffsetsOffset + ring + 1u];
var originalRow = originalFirst + closedRow - first;
if (originalRow >= originalLast) { originalRow = originalFirst; }
if (originalLast <= originalFirst) { return; }
let value = point(originalRow);
let cell = lattice[latticeOffset + 2u]; let step = ${gridLiteral} * cell;
snapPositions[snapPositionsOffset + index * 2u] = normalizePrecisionCoordinate(value.x, step) * cell;
snapPositions[snapPositionsOffset + index * 2u + 1u] = normalizePrecisionCoordinate(value.y, step) * cell;`
    });
    return [
      ...closeRings,
      ...topologyNodes.slice(0, snapIndex),
      correctSnapInput,
      ...topologyNodes.slice(snapIndex),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-original-edge-rows`,
        operation: 'GPUPolygonPrecisionReduction',
        variant: 'original-edge-rows',
        invocationCount: segmentCapacity,
        bindings: [
          {
            name: 'count',
            view: props.output.nodedSegments.status.count,
            type: 'u32',
            access: 'read'
          },
          {
            name: 'sourceRings',
            view: props.output.nodedSegments.sourceRingIds,
            type: 'u32',
            access: 'read'
          },
          {
            name: 'sourceEdges',
            view: props.output.nodedSegments.sourceEdgeIds,
            type: 'u32',
            access: 'read_write'
          },
          {name: 'closedOffsets', view: ringOffsets, type: 'u32', access: 'read'},
          {name: 'originalOffsets', view: props.polygons.ringOffsets, type: 'u32', access: 'read'}
        ],
        // Make-valid maps its own closure copy back to the precision wrapper's closed input.
        // Undo this wrapper's packing too, including arbitrary unused input prefix rows.
        body: `if (index >= count[countOffset]) { return; }
let ring = sourceRings[sourceRingsOffset + index];
if (ring >= ${ringCount}u) { return; }
sourceEdges[sourceEdgesOffset + index] = sourceEdges[sourceEdgesOffset + index] - closedOffsets[closedOffsetsOffset + ring] + originalOffsets[originalOffsetsOffset + ring];`
      }),
      createWGSLKernelNode(graph, {
        id: `${id}-clear-nonempty`,
        operation: 'GPUPolygonPrecisionReduction',
        invocationCount: 1,
        bindings: [{name: 'nonempty', view: nonempty, type: 'u32', access: 'read_write'}],
        body: 'nonempty[nonemptyOffset]=0u;'
      }),
      createWGSLKernelNode(graph, {
        id: `${id}-flag-nonempty`,
        operation: 'GPUPolygonPrecisionReduction',
        invocationCount: ringCount,
        bindings: [
          {name: 'offsets', view: props.polygons.ringOffsets, type: 'u32', access: 'read'},
          {name: 'nonempty', view: nonempty, type: 'atomic<u32>', access: 'read_write'}
        ],
        body: 'if(offsets[offsetsOffset+index]!=offsets[offsetsOffset+index+1u]) {atomicOr(&nonempty[nonemptyOffset],1u);}'
      }),
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
            {name: 'nonempty', view: nonempty, type: 'u32', access: 'read'},
            {name: 'emptyLattice', view: props.lattice, type: 'f32', access: 'read_write'},
            ...views.map((view, viewIndex) => ({
              name: `status${viewIndex}`,
              view,
              type: 'u32' as const,
              access: 'read_write' as const
            }))
          ],
          body: `if(nonempty[nonemptyOffset]!=0u) {return;}
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

/**
 * Line-route grid rounding. Below 2^24 cells the half-up cell is certified from the original
 * operands. From 2^24 cells, where the certification helper falls back to the division candidate,
 * an input that is already an exact grid multiple is returned unchanged: with odd significands
 * `mv`, `mg`, `value / step = (mv / mg) * 2^(ev - eg)` is an integer exactly when `mg` divides `mv`
 * and `ev >= eg`. Other inputs beyond 2^24 cells keep the uncertified candidate. @internal
 */
export const LINE_PRECISION_COORDINATE_WGSL = /* wgsl */ `${GRID_HALF_UP_ROUNDING_WGSL}
fn oddFloatSignificand(value: f32) -> vec2i {
  let bits = bitcast<u32>(value);
  let biased = (bits >> 23u) & 255u;
  var significand = (bits & 8388607u) | select(8388608u, 0u, biased == 0u);
  var exponent = select(i32(biased) - 150, -149, biased == 0u);
  for (var shift = 0u; shift < 24u && significand != 0u && (significand & 1u) == 0u; shift++) {
    significand = significand >> 1u; exponent++;
  }
  return vec2i(i32(significand), exponent);
}
fn isExactGridMultiple(value: f32, step: f32) -> bool {
  if (value == 0.0) { return true; }
  let numerator = oddFloatSignificand(value);
  let denominator = oddFloatSignificand(step);
  return u32(numerator.x) % u32(denominator.x) == 0u && numerator.y >= denominator.y;
}
fn roundLineGridCoordinate(value: f32, step: f32) -> f32 {
  let large = (bitcast<u32>(value / step) & 0x7fffffffu) >= 0x4b800000u;
  if (large && (bitcast<u32>(value) & 0x7f800000u) != 0x7f800000u && isExactGridMultiple(value, step)) {
    return value;
  }
  return roundGridHalfUp(value, 0.0, step) * step;
}
`;

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
    props = {
      ...props,
      ...(props.spatialContext
        ? {spatialContext: snapshotRefinementSpatialContext(props.spatialContext)}
        : {}),
      output: {...props.output}
    };
    this.props = props;
    validateRefinementNumber(
      {id: this.id, contributor: 'GPULinePrecisionReduction'},
      'gridSize',
      props.gridSize,
      true
    );
    validateOption({id: this.id, contributor: 'GPULinePrecisionReduction'}, 'mode', props.mode, [
      'valid-output',
      'keep-collapsed'
    ]);
    this.spatialContext = resolveGPUSpatialContext(this.id, props.spatialContext);

    for (const [path, view] of [
      ['positions', props.positions],
      ['output.positions', props.output.positions]
    ] as const) {
      validateSinglePackedView(view, ['float32x2'], `${this.id} ${path}`, {
        id: this.id,
        contributor: 'GPULinePrecisionReduction'
      });
    }
    for (const [path, view] of [
      ['lineOffsets', props.lineOffsets],
      ['output.lineOffsets', props.output.lineOffsets],
      ['output.status', props.output.status]
    ] as const) {
      validateSinglePackedView(view, ['uint32'], `${this.id} ${path}`, {
        id: this.id,
        contributor: 'GPULinePrecisionReduction'
      });
    }
    validateRefinementOutputLength(
      {id: this.id, contributor: 'GPULinePrecisionReduction'},
      'output.positions',
      props.output.positions,
      props.positions.length,
      'positions.length',
      true
    );
    validateRefinementOutputLength(
      {id: this.id, contributor: 'GPULinePrecisionReduction'},
      'output.lineOffsets',
      props.output.lineOffsets,
      props.lineOffsets.length,
      'lineOffsets.length'
    );
    validateRefinementOutputLength(
      {id: this.id, contributor: 'GPULinePrecisionReduction'},
      'output.status',
      props.output.status,
      props.lineOffsets.length - 1,
      'lineOffsets.length - 1'
    );
    validateOutlineOutputPasses(this.id, props.output, [['positions', 'lineOffsets'], ['status']]);
    validateGeometryReadWriteViews(this.id, [props.output.lineOffsets], [props.output.positions]);
    validateGraphOutputsDisjointFromInputs(this.id, Object.values(props.output), [
      props.positions,
      props.lineOffsets
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
        [props.positions, props.lineOffsets, ...Object.values(props.output)].map((view, index) => [
          `binding.${index}`,
          view
        ])
      )
    );
    validateRefinementScratchByteLengths(graph, id, {
      staged: props.positions.length * 8,
      keptCountsAndScan: props.lineOffsets.length * 4
    });
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
        declarations: `${LINE_PRECISION_COORDINATE_WGSL}
const GRID: f32 = ${getWGSLFloatLiteral(props.gridSize)};
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
  let rounded = vec2f(roundLineGridCoordinate(value.x, GRID), roundLineGridCoordinate(value.y, GRID));
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

// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import type {GPUPolygonStudyWindow} from '../../analysis-core/contracts/study-window';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  resolveGPUSpatialContext,
  type GPUSpatialContext
} from '../../analysis-core/contracts/spatial-context';
import {
  validateCapacity,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {EXACT_ORIENTATION_WGSL} from '../../analysis-core/kernels/exact-orientation-wgsl';
import {EXACT_GEOMETRY_WGSL} from '../../analysis-core/kernels/exact-geometry-wgsl';
import {
  getStudyWindowBinding,
  getStudyWindowWGSL,
  getStudyWindowWGSLOptions
} from '../../analysis-core/kernels/study-window-wgsl';
import {IS_FINITE_FLOAT_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {createTranslationPointPatternNodes} from './translation-point-pattern';
import {validatePointProcessPositive, validatePointProcessView} from './point-process-utils';

/** Largest event count: the intensity and pair kernels are all-pairs O(n²) on one device. */
const MAXIMUM_EVENT_COUNT = 4096;
/** Largest radius count: radii are evaluated in a single resident curve. */
const MAXIMUM_RADIUS_COUNT = 256;
/** Largest per-pair triangle-intersection budget, bounding the overlap loop in one invocation. */
const MAXIMUM_OVERLAP_EDGE_PAIRS = 1048576;

/** Intensity-reweighted reduced-sample K in a prepared polygon window. */
export type GPUInhomogeneousPointPatternProps = {
  id?: string;
  /** Planar only; longitude-latitude is rejected. Default planar. */
  spatialContext?: GPUSpatialContext;
  positions: GraphDataView<'float32x2'>;
  /** Add the preparation to this graph before this consumer; holes/components are preserved. */
  studyWindow: GPUPolygonStudyWindow;
  /** Supplied positive intensities, or a leave-one-out Epanechnikov estimate. */
  suppliedIntensity?: GraphDataView<'float32'>;
  /** Required for estimated intensity; planar coordinate units. */
  bandwidth?: number;
  /** Default exclude. Only applies to estimated intensity at event locations. */
  selfContribution?: 'include' | 'exclude';
  /** Default border; translation uses the exact polygon intersection composition. */
  edgeCorrection?: 'border' | 'translation';
  /** Omit for annular K differences. Kernel options require translation correction. */
  pairCorrelation?: {
    kernel: 'epanechnikov';
    /** Compact support half-width in coordinate units, not the kernel standard deviation. */
    bandwidth: number;
    /** Default radius; distance rejects coincident distinct events. */
    divisor?: 'radius' | 'distance';
    /** Default none; reflection adds the reflected kernel at the distance origin. */
    zeroCorrection?: 'none' | 'reflection';
  };
  /** Optional resident unordered-pair allocation cap; insufficient capacity throws. */
  maximumPairCount?: number;
  /** Translation triangle-intersection budget per event pair; default 65536. */
  maximumOverlapEdgePairs?: number;
  /** Strictly increasing positive radii, validated on the GPU. */
  radii: GraphDataView<'float32'>;
  output: {
    intensity: GraphDataView<'float32'>;
    k: GraphDataView<'float32'>;
    /** Annular K difference by default, or the explicitly selected smoothed estimator. */
    pairCorrelation: GraphDataView<'float32'>;
    /** 0 valid, 1 empty erosion, 2 invalid input, 3 overlap work exhausted, 4 unresolved overlap. */
    status: GraphDataView<'uint32'>;
  };
};

/**
 * GPU polygon-window intensity estimation and inhomogeneous border K.
 * K(r) = sum_i I(boundary_i > r)/lambda_i * sum_{j != i,dij <= r} 1/lambda_j
 *        / sum_i I(boundary_i > r)/lambda_i.
 * The estimate uses the Epanechnikov kernel divided by its analytic polygon-window mass at the
 * evaluation point. Supplied intensities bypass estimation. Zero/nonfinite included intensity
 * invalidates inference instead of silently dropping events. Exact all-pairs small-frame path,
 * O(n² radiusCount), at most 4096 events; estimated intensity is a plug-in estimate, not an
 * unbiased known-intensity estimator. Translation reuses GPU pair overlap weights across radii.
 * Polygon overlap uses oriented triangle composition; work/numeric limits return explicit status.
 */
export class GPUInhomogeneousPointPattern implements GPUCommandNodeProducer {
  readonly id: string;
  readonly props: GPUInhomogeneousPointPatternProps;
  /** Resolved spatial context (planar). */
  readonly spatialContext: GPUSpatialContext;
  constructor(props: GPUInhomogeneousPointPatternProps) {
    const id = props.id ?? 'inhomogeneous-point-pattern';
    this.id = id;
    this.props = props;
    this.spatialContext = resolveGPUSpatialContext(id, props.spatialContext);
    const rows = props.positions.length;
    const radiusCount = props.radii.length;
    validateCapacity(id, 'positions rows', rows, 1, MAXIMUM_EVENT_COUNT);
    validateCapacity(id, 'radii rows', radiusCount, 1, MAXIMUM_RADIUS_COUNT);
    validateRequirement(
      id,
      props.studyWindow.kind === 'polygon' &&
        props.studyWindow.spatialContext.coordinateSpace === 'planar',
      'studyWindow must be a prepared planar polygon window'
    );
    validateRequirement(
      id,
      props.selfContribution === undefined ||
        ['include', 'exclude'].includes(props.selfContribution),
      "selfContribution must be 'include' or 'exclude'"
    );
    validateRequirement(
      id,
      props.edgeCorrection === undefined ||
        ['border', 'translation'].includes(props.edgeCorrection),
      "edgeCorrection must be 'border' or 'translation'"
    );
    if (props.pairCorrelation) {
      validateRequirement(
        id,
        props.edgeCorrection === 'translation' && props.pairCorrelation.kernel === 'epanechnikov',
        "pairCorrelation needs edgeCorrection 'translation' and the 'epanechnikov' kernel"
      );
      validatePointProcessPositive(
        props.pairCorrelation.bandwidth,
        id,
        'pairCorrelation.bandwidth'
      );
      validateRequirement(
        id,
        props.pairCorrelation.divisor === undefined ||
          ['radius', 'distance'].includes(props.pairCorrelation.divisor),
        "pairCorrelation.divisor must be 'radius' or 'distance'"
      );
      validateRequirement(
        id,
        props.pairCorrelation.zeroCorrection === undefined ||
          ['none', 'reflection'].includes(props.pairCorrelation.zeroCorrection),
        "pairCorrelation.zeroCorrection must be 'none' or 'reflection'"
      );
    }
    if (props.maximumPairCount !== undefined) {
      validateRequirement(
        id,
        Number.isSafeInteger(props.maximumPairCount) &&
          props.maximumPairCount >= (rows * (rows - 1)) / 2,
        'maximumPairCount must be an integer of at least rows * (rows - 1) / 2'
      );
    }
    if (props.maximumOverlapEdgePairs !== undefined) {
      validateCapacity(
        id,
        'maximumOverlapEdgePairs',
        props.maximumOverlapEdgePairs,
        1,
        MAXIMUM_OVERLAP_EDGE_PAIRS
      );
    }
    if (!props.suppliedIntensity) {
      validatePointProcessPositive(props.bandwidth ?? NaN, id, 'bandwidth');
    }
    validatePointProcessView(props.positions, 'float32x2', rows, id);
    validatePointProcessView(props.radii, 'float32', radiusCount, id);
    if (props.suppliedIntensity) {
      validatePointProcessView(props.suppliedIntensity, 'float32', rows, id);
    }
    validatePointProcessView(props.output.intensity, 'float32', rows, id);
    validatePointProcessView(props.output.k, 'float32', radiusCount, id);
    validatePointProcessView(props.output.pairCorrelation, 'float32', radiusCount, id);
    validatePointProcessView(props.output.status, 'uint32', radiusCount, id);
    validateGraphOutputsDisjointFromInputs(this.id, Object.values(props.output), [
      props.positions,
      props.radii,
      props.suppliedIntensity
    ]);
  }
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const rows = props.positions.length;
    const windowViews = props.studyWindow.getViews(graph as unknown as GPUCommandGraph<unknown>);
    validateGraphViewsBelongToGraph(id, graph, [
      props.positions,
      props.radii,
      props.suppliedIntensity,
      windowViews.packed,
      ...Object.values(props.output)
    ]);
    validateGraphOutputsDisjointFromInputs(id, Object.values(props.output), [windowViews.packed]);
    const boundary = createTransientView(graph, `${id}-boundary`, 'float32', rows);
    const windowDeclarations = `${EXACT_ORIENTATION_WGSL}\n${EXACT_GEOMETRY_WGSL}\n${getStudyWindowWGSL(getStudyWindowWGSLOptions(windowViews, 'epanechnikov'))}`;
    const pointDeclaration = `fn point(row: u32) -> vec2<f32> {
  return vec2<f32>(positions[positionsOffset + 2u * row], positions[positionsOffset + 2u * row + 1u]);
}`;
    const bandwidth = Math.fround(props.bandwidth ?? 1);
    const intensityNode = createWGSLKernelNode(graph, {
      id: `${id}-intensity`,
      operation: 'GPUInhomogeneousPointPattern',
      variant: props.suppliedIntensity ? 'supplied' : 'estimated',
      invocationCount: rows,
      bindings: [
        {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
        getStudyWindowBinding(windowViews),
        ...(props.suppliedIntensity
          ? [
              {
                name: 'supplied',
                view: props.suppliedIntensity,
                type: 'f32' as const,
                access: 'read' as const
              }
            ]
          : []),
        {name: 'intensity', view: props.output.intensity, type: 'f32', access: 'read_write'},
        {name: 'boundary', view: boundary, type: 'f32', access: 'read_write'}
      ],
      declarations: `${IS_FINITE_FLOAT_WGSL}\n${windowDeclarations}\n${pointDeclaration}`,
      body: `let focus = point(index);
  let invalid = bitcast<f32>(0x7fc00000u | (index & 0u));
  var estimate = invalid;
  var boundaryDistance = -1.0;
  if (isStudyWindowUsable() && isInsideStudyWindow(focus)) {
    boundaryDistance = getStudyWindowBoundaryDistance(focus, 3.402823e38);
    ${
      props.suppliedIntensity
        ? 'estimate = supplied[suppliedOffset + index];'
        : `
    var total = 0.0;
    for (var row = 0u; row < ${rows}u; row++) {
      ${props.selfContribution === 'include' ? '' : 'if (row == index) { continue; }'}
      let source = point(row);
      if (!isInsideStudyWindow(source)) { continue; }
      let delta = (focus - source) / ${bandwidth}f;
      total += max(0.0, 1.0 - dot(delta, delta));
    }
    let mass = getStudyWindowKernelMass(focus, ${bandwidth}f);
    if (mass > 0.0) { estimate = total * (2.0 / 3.141592653589793) / ${bandwidth}f / ${bandwidth}f / mass; }`
    }
  }
  intensity[intensityOffset + index] = estimate;
  boundary[boundaryOffset + index] = boundaryDistance;`
    });
    if (props.edgeCorrection === 'translation') {
      return [
        intensityNode,
        ...createTranslationPointPatternNodes(graph, props, id, boundary, windowViews)
      ];
    }
    const radiusCount = props.radii.length;
    const terms = createTransientView(
      graph,
      `${id}-border-k-values`,
      'float32',
      radiusCount * rows
    );
    return [
      intensityNode,
      createWGSLKernelNode(graph, {
        id: `${id}-border-k-terms`,
        operation: 'GPUInhomogeneousPointPattern',
        variant: 'border-k-terms',
        // One invocation per (radius, focus event); each loops over the events once.
        invocationCount: radiusCount * rows,
        bindings: [
          {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
          {name: 'intensity', view: props.output.intensity, type: 'f32', access: 'read'},
          {name: 'boundary', view: boundary, type: 'f32', access: 'read'},
          {name: 'radii', view: props.radii, type: 'f32', access: 'read'},
          {name: 'terms', view: terms, type: 'f32', access: 'read_write'}
        ],
        declarations: `${IS_FINITE_FLOAT_WGSL}\n${pointDeclaration}`,
        body: `let radiusIndex = index / ${rows}u;
  let focus = index % ${rows}u;
  let radius = radii[radiiOffset + radiusIndex];
  let focusIntensity = intensity[intensityOffset + focus];
  var term = 0.0;
  if (boundary[boundaryOffset + focus] >= 0.0 && isFiniteFloat(focusIntensity) && focusIntensity > 0.0 && boundary[boundaryOffset + focus] > radius) {
    var neighbors = 0.0;
    for (var row = 0u; row < ${rows}u; row++) {
      if (row == focus || boundary[boundaryOffset + row] < 0.0) { continue; }
      let sourceIntensity = intensity[intensityOffset + row];
      if (sourceIntensity > 0.0 && distance(point(focus), point(row)) <= radius) { neighbors += 1.0 / sourceIntensity; }
    }
    term = neighbors / focusIntensity;
  }
  terms[termsOffset + index] = term;`
      }),
      createWGSLKernelNode(graph, {
        id: `${id}-border-k`,
        operation: 'GPUInhomogeneousPointPattern',
        variant: 'border-k',
        invocationCount: radiusCount,
        // Fixed focus order per radius keeps the float sums independent of scheduling. The loop is
        // O(rows) per radius (at most 4096 reads); the O(rows²) neighbor sums are in the terms node.
        bindings: [
          {name: 'intensity', view: props.output.intensity, type: 'f32', access: 'read'},
          {name: 'boundary', view: boundary, type: 'f32', access: 'read'},
          {name: 'radii', view: props.radii, type: 'f32', access: 'read'},
          {name: 'terms', view: terms, type: 'f32', access: 'read'},
          {name: 'result', view: props.output.k, type: 'f32', access: 'read_write'},
          {name: 'status', view: props.output.status, type: 'u32', access: 'read_write'}
        ],
        declarations: IS_FINITE_FLOAT_WGSL,
        body: `let radius = radii[radiiOffset + index];
  var valid = isFiniteFloat(radius) && radius > 0.0;
  if (index > 0u) { valid = valid && radius > radii[radiiOffset + index - 1u]; }
  var denominator = 0.0;
  var numerator = 0.0;
  var included = 0u;
  for (var focus = 0u; focus < ${rows}u; focus++) {
    if (boundary[boundaryOffset + focus] < 0.0) { continue; }
    included++;
    let focusIntensity = intensity[intensityOffset + focus];
    if (!isFiniteFloat(focusIntensity) || focusIntensity <= 0.0) { valid = false; continue; }
    if (boundary[boundaryOffset + focus] <= radius) { continue; }
    denominator += 1.0 / focusIntensity;
    numerator += terms[termsOffset + index * ${rows}u + focus];
  }
  valid = valid && included > 0u && isFiniteFloat(numerator) && isFiniteFloat(denominator);
  var code = select(2u, select(1u, 0u, denominator > 0.0), valid);
  var value = bitcast<f32>(0x7fc00000u | (index & 0u));
  if (code == 0u) { value = numerator / denominator; }
  result[resultOffset + index] = value;
  status[statusOffset + index] = code;`
      }),
      createWGSLKernelNode(graph, {
        id: `${id}-pair-correlation`,
        operation: 'GPUInhomogeneousPointPattern',
        variant: 'annular-pcf',
        invocationCount: props.radii.length,
        bindings: [
          {name: 'radii', view: props.radii, type: 'f32', access: 'read'},
          {name: 'values', view: props.output.k, type: 'f32', access: 'read'},
          {name: 'result', view: props.output.pairCorrelation, type: 'f32', access: 'read_write'}
        ],
        body: `var previousRadius = 0.0;
  var previousValue = 0.0;
  if (index > 0u) { previousRadius = radii[radiiOffset + index - 1u]; previousValue = values[valuesOffset + index - 1u]; }
  let radius = radii[radiiOffset + index];
  result[resultOffset + index] = (values[valuesOffset + index] - previousValue) / (3.141592653589793 * (radius * radius - previousRadius * previousRadius));`
      })
    ];
  }
}

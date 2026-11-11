// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
// SPDX-FileComment: Independently implemented for WebGPU; inspired by NVIDIA RAPIDS cuProj.
// This datum-grid extension is independently authored; no cuProj grid implementation is copied.

import type {
  GPUCommandGraph,
  GPUCommandNode,
  GPUCommandNodeProducer,
  GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  createWGSLKernelNode,
  type WGSLKernelBinding
} from '../analysis-core/kernels/wgsl-kernel-nodes';
import {
  validateCapacity,
  validateFiniteNumber,
  validateMaximumCount,
  validateRequirement,
  validateViewLength,
  validateSinglePackedView,
  validateGraphViewsBelongToGraph,
  validateGraphOutputsDisjointFromInputs
} from '../analysis-core/kernels/gpu-contributor-utils';

/**
 * Bit values written to the `status` output of {@link GPUDatumGridTransform}. Zero is success;
 * non-zero rows write NaN positions.
 */
export const GPU_DATUM_GRID_TRANSFORM_STATUS = {
  /** The row was transformed. */
  success: 0,
  /** Non-finite input or result, or a latitude outside +/-90 degrees. */
  invalid: 1,
  /** No grid covers the position. The first covering grid wins; there is no fall through. */
  noCoverage: 2,
  /** A covering grid has a nodata corner. */
  nodata: 4,
  /** The inverse iteration did not reach the residual tolerance. */
  inverseExhausted: 8
} as const;

/**
 * Default inverse residual tolerance in degrees: 5e-9 degrees of latitude is 0.56 mm. The earlier
 * default of 1e-5 degrees (1.1 m) accepted the unshifted input for any shift below a meter.
 */
const DEFAULT_INVERSE_TOLERANCE_DEGREES = 5e-9;

/**
 * Caller-decoded regular grid. Longitude must use the same continuous branch as positions.
 *
 * Antimeridian limitation: grid and position longitudes are compared as plain numbers without
 * wrapping, so a grid spanning +/-180 must be expressed on a continuous branch (for example 170 to
 * 190) and positions on that branch. Positions on the other branch report `noCoverage`.
 */
export type GPUDatumGrid = {
  width: number;
  height: number;
  sampleOffset: number;
  origin: readonly [number, number];
  spacing: readonly [number, number];
};
export type GPUDatumGridTransformProps = {
  id?: string;
  /** Longitude/latitude in degrees and height in declared units. */
  positions: GraphDataView<'float32x3'>;
  /** Optional residual words; positions + positionsLow retain sub-f32 coordinate detail. */
  positionsLow?: GraphDataView<'float32x3'>;
  /** Packed shifts: longitude degrees, latitude degrees, signed height meters. */
  samples: GraphDataView<'float32x3'>;
  /** Row mask of `samples`: zero marks nodata. All four interpolation corners must be valid, including at edges. */
  mask: GraphDataView<'uint32'>;
  /** First covering grid wins. Nodata in that grid fails; it does not fall through. */
  grids: readonly GPUDatumGrid[];
  operation: 'horizontal' | 'vertical' | 'combined';
  direction?: 'forward' | 'inverse';
  /** Explicit metadata; signed samples express target minus source height in meters. */
  sourceVerticalReference?: string;
  targetVerticalReference?: string;
  heightUnitMeters?: number;
  maximumIterations?: number;
  /**
   * Inverse residual tolerance in degrees of longitude and latitude (the larger of the two
   * absolute residuals of `forward(result) - input`). Defaults to 5e-9 degrees, about 0.56 mm
   * on the ground. The residual is formed from the low words, so it resolves far below the f32
   * spacing of a coordinate; the inverse always takes at least one update before testing it.
   * The test never asks for less than four f32 spacings of the shift being inverted (4.8e-7 of
   * it), the resolution of the residual itself.
   */
  tolerance?: number;
  output: {
    positions: GraphDataView<'float32x3'>;
    /** Required with positionsLow; residual relative to the unchanged input high words. */
    positionsLow?: GraphDataView<'float32x3'>;
    /** {@link GPU_DATUM_GRID_TRANSFORM_STATUS} values. */
    status: GraphDataView<'uint32'>;
  };
};

/** GPU-resident bilinear datum shift with ordered grids and bounded inverse iteration. */
export class GPUDatumGridTransform implements GPUCommandNodeProducer {
  readonly id: string;
  /** The sample mask, `props.mask`. */
  readonly mask: GraphDataView<'uint32'>;
  constructor(readonly props: GPUDatumGridTransformProps) {
    this.id = props.id ?? 'datum-grid-transform';
    const {mask} = props;
    validateRequirement(this.id, mask !== undefined, 'mask must be given');
    this.mask = mask;
    validateSinglePackedView(props.positions, ['float32x3'], this.id);
    if (props.positionsLow) validateSinglePackedView(props.positionsLow, ['float32x3'], this.id);
    if (props.output.positionsLow)
      validateSinglePackedView(props.output.positionsLow, ['float32x3'], this.id);
    validateRequirement(
      this.id,
      Boolean(props.positionsLow) === Boolean(props.output.positionsLow),
      'positionsLow and output.positionsLow must be given together'
    );
    if (props.positionsLow) {
      validateViewLength(this.id, 'positionsLow', props.positionsLow, props.positions.length);
      validateViewLength(
        this.id,
        'output.positionsLow',
        props.output.positionsLow!,
        props.positions.length
      );
    }
    validateSinglePackedView(props.samples, ['float32x3'], this.id);
    validateSinglePackedView(this.mask, ['uint32'], this.id);
    validateSinglePackedView(props.output.positions, ['float32x3'], this.id);
    validateSinglePackedView(props.output.status, ['uint32'], this.id);
    // Output rows are aligned, grid indices are exact u32 offsets, and interpolation needs 2x2 cells.
    validateViewLength(this.id, 'output.positions', props.output.positions, props.positions.length);
    validateViewLength(this.id, 'output.status', props.output.status, props.positions.length);
    validateViewLength(this.id, 'mask', this.mask, props.samples.length);
    for (const [gridIndex, grid] of props.grids.entries()) {
      const name = `grids[${gridIndex}]`;
      validateCapacity(this.id, `${name}.width`, grid.width, 2);
      validateCapacity(this.id, `${name}.height`, grid.height, 2);
      validateCapacity(this.id, `${name}.sampleOffset`, grid.sampleOffset, 0);
      validateMaximumCount(
        this.id,
        `${name} sampleOffset + width * height`,
        grid.sampleOffset + grid.width * grid.height,
        props.samples.length
      );
      for (const [axis, value] of grid.origin.entries()) {
        validateFiniteNumber(this.id, `${name}.origin[${axis}]`, value);
      }
      for (const [axis, value] of grid.spacing.entries()) {
        validateFiniteNumber(this.id, `${name}.spacing[${axis}]`, value);
        validateRequirement(
          this.id,
          Math.fround(value) !== 0,
          `${name}.spacing[${axis}] must be nonzero in float32`
        );
      }
    }
    validateFiniteNumber(this.id, 'heightUnitMeters', props.heightUnitMeters ?? 1);
    validateRequirement(
      this.id,
      Math.fround(props.heightUnitMeters ?? 1) > 0,
      'heightUnitMeters must be positive in float32'
    );
    validateFiniteNumber(
      this.id,
      'tolerance',
      props.tolerance ?? DEFAULT_INVERSE_TOLERANCE_DEGREES
    );
    validateRequirement(
      this.id,
      Math.fround(props.tolerance ?? DEFAULT_INVERSE_TOLERANCE_DEGREES) > 0,
      'tolerance must be positive in float32'
    );
    validateCapacity(this.id, 'maximumIterations', props.maximumIterations ?? 16, 1, 1024);
    if (props.operation !== 'horizontal') {
      // Vertical source and target references are mandatory, even when only height units differ.
      validateRequirement(
        this.id,
        Boolean(props.sourceVerticalReference) && Boolean(props.targetVerticalReference),
        `${props.operation} operation requires sourceVerticalReference and targetVerticalReference`
      );
    }
    validateGraphOutputsDisjointFromInputs(this.id, Object.values(props.output), [
      props.positions,
      ...(props.positionsLow ? [props.positionsLow] : []),
      props.samples,
      this.mask
    ]);
  }
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {props, id} = this;
    validateGraphViewsBelongToGraph(id, graph, [
      props.positions,
      ...(props.positionsLow ? [props.positionsLow] : []),
      props.samples,
      this.mask,
      ...Object.values(props.output)
    ]);
    const horizontal = props.operation !== 'vertical';
    const vertical = props.operation !== 'horizontal';
    const inverse = props.direction === 'inverse';
    return [
      createWGSLKernelNode(graph, {
        id,
        operation: 'GPUDatumGridTransform',
        variant: `${props.operation}-${props.direction ?? 'forward'}`,
        invocationCount: props.positions.length,
        bindings: [
          {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
          ...(props.positionsLow
            ? [
                {
                  name: 'positionsLow',
                  view: props.positionsLow,
                  type: 'f32',
                  access: 'read'
                } satisfies WGSLKernelBinding,
                {
                  name: 'outputLow',
                  view: props.output.positionsLow!,
                  type: 'f32',
                  access: 'read_write'
                } satisfies WGSLKernelBinding
              ]
            : []),
          {name: 'samples', view: props.samples, type: 'f32', access: 'read'},
          {name: 'validity', view: this.mask, type: 'u32', access: 'read'},
          {name: 'output', view: props.output.positions, type: 'f32', access: 'read_write'},
          {name: 'status', view: props.output.status, type: 'u32', access: 'read_write'}
        ],
        declarations: `struct GridSample { shift: vec3f, status: u32 }
fn datumFinite(value: f32) -> bool { return (bitcast<u32>(value) & 0x7f800000u) != 0x7f800000u; }
// Compare the residual against the latitude limits before adding it to the high word.
// Summing first rounds sub-ulp excursions beyond either pole back onto the valid boundary.
fn datumLatitudeValid(high: f32, low: f32) -> bool { return low <= 90.0 - high && low >= -90.0 - high; }
// Residual limit of the inverse: the tolerance, but never less than four f32 spacings of the
// terms the residual is formed from (4 * 2^-23 of the largest), below which it is rounding.
fn datumInverseLimit(displacement: vec2f, shift: vec2f, low: vec2f) -> f32 {
  let magnitude = max(max(abs(displacement), abs(shift)), abs(low));
  return max(${props.tolerance ?? DEFAULT_INVERSE_TOLERANCE_DEGREES}, 4.8e-7 * max(magnitude.x, magnitude.y));
}
fn loadDatumSample(row: u32) -> vec3f { let offset = samplesOffset + row * 3u; return vec3f(samples[offset], samples[offset + 1u], samples[offset + 2u]); }
fn sampleDatum(position: vec2f, low: vec2f) -> GridSample {
${props.grids
  .map(
    grid => `{
  let highDifference = fma(position, vec2f(1.0), -vec2f(${grid.origin.map(value => `${Math.fround(value)}`).join(',')}));
  let lowDifference = fma(low, vec2f(1.0), -vec2f(${grid.origin.map(value => `${value - Math.fround(value)}`).join(',')}));
  if (!datumFinite(lowDifference.x) || !datumFinite(lowDifference.y) || !datumFinite(highDifference.x) || !datumFinite(highDifference.y)) { return GridSample(vec3f(0.0), ${GPU_DATUM_GRID_TRANSFORM_STATUS.invalid}u); }
  let rawCoordinate = (highDifference + lowDifference) / vec2f(${grid.spacing.join(',')});
  // A tiny grid-coordinate margin absorbs f32 quotient error at closed outer boundaries.
  if (all(rawCoordinate >= vec2f(-1e-5)) && all(rawCoordinate <= vec2f(${grid.width - 1}, ${grid.height - 1}) + vec2f(1e-5))) {
    let coordinate = clamp(rawCoordinate, vec2f(0.0), vec2f(${grid.width - 1}, ${grid.height - 1}));
    let column = min(u32(coordinate.x), ${grid.width - 2}u);
    let row = min(u32(coordinate.y), ${grid.height - 2}u);
    let corner = ${grid.sampleOffset}u + row * ${grid.width}u + column;
    let corners = vec4u(corner, corner + 1u, corner + ${grid.width}u, corner + ${grid.width + 1}u);
    for (var slot = 0u; slot < 4u; slot++) {
      let shift = loadDatumSample(corners[slot]);
      if (validity[validityOffset + corners[slot]] == 0u || !datumFinite(shift.x) || !datumFinite(shift.y) || !datumFinite(shift.z)) { return GridSample(vec3f(0.0), ${GPU_DATUM_GRID_TRANSFORM_STATUS.nodata}u); }
    }
    let fraction = coordinate - vec2f(f32(column), f32(row));
    return GridSample(mix(mix(loadDatumSample(corners.x), loadDatumSample(corners.y), fraction.x), mix(loadDatumSample(corners.z), loadDatumSample(corners.w), fraction.x), fraction.y), 0u);
  }
}`
  )
  .join('\n')}
  return GridSample(vec3f(0.0), ${GPU_DATUM_GRID_TRANSFORM_STATUS.noCoverage}u);
}`,
        body: `let offset = positionsOffset + index * 3u;
let source = vec3f(positions[offset], positions[offset + 1u], positions[offset + 2u]);
let sourceLow = ${props.positionsLow ? 'vec3f(positionsLow[positionsLowOffset + index * 3u], positionsLow[positionsLowOffset + index * 3u + 1u], positionsLow[positionsLowOffset + index * 3u + 2u])' : 'vec3f(0.0)'};
var displacement = sourceLow;
var result = source;
var failure = 0u;
if (!datumFinite(source.x) || !datumFinite(source.y) || !datumFinite(source.z) || !datumFinite(sourceLow.x) || !datumFinite(sourceLow.y) || !datumFinite(sourceLow.z) || !datumLatitudeValid(source.y, sourceLow.y)) { failure = ${GPU_DATUM_GRID_TRANSFORM_STATUS.invalid}u; }
if (failure == 0u) {
  var sampled = sampleDatum(source.xy, displacement.xy);
  failure = sampled.status;
  ${
    inverse && horizontal
      ? `var converged = false;
  for (var iteration = 0u; iteration < ${props.maximumIterations ?? 16}u && failure == 0u; iteration++) {
    let residual = displacement.xy + sampled.shift.xy - sourceLow.xy;
    if (!datumFinite(residual.x) || !datumFinite(residual.y)) { failure = ${GPU_DATUM_GRID_TRANSFORM_STATUS.invalid}u; break; }
    // The first residual is the grid shift at the input, not an error of the estimate: always
    // take one update before testing, or every shift below the tolerance is skipped.
    if (iteration > 0u && max(abs(residual.x), abs(residual.y)) <= datumInverseLimit(displacement.xy, sampled.shift.xy, sourceLow.xy)) { converged = true; break; }
    displacement = vec3f(displacement.xy - residual, displacement.z);
    sampled = sampleDatum(source.xy, displacement.xy);
    failure = sampled.status;
  }
  if (!converged && failure == 0u) {
    let residual = displacement.xy + sampled.shift.xy - sourceLow.xy;
    if (!datumFinite(residual.x) || !datumFinite(residual.y)) { failure = ${GPU_DATUM_GRID_TRANSFORM_STATUS.invalid}u; }
    else if (max(abs(residual.x), abs(residual.y)) > datumInverseLimit(displacement.xy, sampled.shift.xy, sourceLow.xy)) { failure = ${GPU_DATUM_GRID_TRANSFORM_STATUS.inverseExhausted}u; }
  }`
      : horizontal
        ? 'displacement = vec3f(sourceLow.xy + sampled.shift.xy, sourceLow.z);'
        : ''
  }
  ${vertical ? `displacement.z = sourceLow.z + ${inverse ? '-' : ''}sampled.shift.z / ${props.heightUnitMeters ?? 1};` : ''}
}
result = source + displacement;
var resultLow = displacement;
if (failure == 0u && (!datumFinite(result.x) || !datumFinite(result.y) || !datumFinite(result.z) || !datumLatitudeValid(source.y, displacement.y))) { failure = ${GPU_DATUM_GRID_TRANSFORM_STATUS.invalid}u; }
${props.output.positionsLow ? 'if (failure == 0u) { result = source; }' : ''}
if (failure != 0u) { result = vec3f(bitcast<f32>(0x7fc00000u | (index & 0u))); resultLow = result; }
output[outputOffset + index * 3u] = result.x;
output[outputOffset + index * 3u + 1u] = result.y;
output[outputOffset + index * 3u + 2u] = result.z;
${props.output.positionsLow ? 'outputLow[outputLowOffset + index * 3u] = resultLow.x; outputLow[outputLowOffset + index * 3u + 1u] = resultLow.y; outputLow[outputLowOffset + index * 3u + 2u] = resultLow.z;' : ''}
status[statusOffset + index] = failure;`
      })
    ];
  }
}

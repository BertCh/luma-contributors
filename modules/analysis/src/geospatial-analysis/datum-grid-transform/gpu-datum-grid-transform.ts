// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
// SPDX-FileComment: Independently implemented for WebGPU; inspired by NVIDIA RAPIDS cuProj.
// This datum-grid extension is independently authored; no cuProj grid implementation is copied.

import {
  createGPUComputeCommandNode,
  getBoundedDispatchLayout,
  getViewBinding,
  getViewElementOffset,
  validatePackedView,
  type GraphVectorView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {Kernel} from '@luma.gl/graph';
import {
  createWGSLKernelNode,
  getWGSLFloatLiteral,
  type WGSLKernelProps,
  type WGSLKernelBinding
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  validateCapacity,
  validateFiniteNumber,
  validateMaximumCount,
  validateRequirement,
  validateViewLength,
  validateSinglePackedView,
  getGraphViewChunks,
  validateCommandNodeBudget,
  validateGraphOutputsDisjointFromInputs
} from '../../analysis-core/kernels/gpu-contributor-utils';

import {
  createGraphChunkDispatchParameters,
  planGraphChunkDispatch,
  validateDisjointWritableViews,
  validateGraphChunkBindingRanges,
  validateGraphChunkDispatchBindings,
  type GraphChunkDispatchPlan
} from '../../analysis-core/kernels/graph-chunk-dispatch';

// Fixed workgroups and the pre-multiplication guard preserve uint32 indices at large dispatches.
const DATUM_CHUNK_WORKGROUP_SIZE = 256;
// A 256-thread workgroup must be guarded before its index multiplication can wrap uint32.
const DATUM_MAXIMUM_LINEAR_WORKGROUP_COUNT = 0x1000000;
// Each packed sample needs three scalar u32 indices, including an aligned binding prefix.
const DATUM_MAXIMUM_SAMPLE_COUNT = Math.floor(0xffffffff / 3) - 64;
// Bound inverse shader work and its u32 loop-bound literal.
const DATUM_MAXIMUM_ITERATIONS = 1024;

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

/** Preserves the source view kind for each row-aligned output format. */
type GPUDatumGridTransformRowView<
  Format extends 'float32' | 'float32x2' | 'float32x3' | 'uint32',
  Rows extends GraphDataView | GraphVectorView
> = Rows extends GraphDataView ? GraphDataView<Format> : GraphVectorView<Format>;

export type GPUDatumGridTransformProps<
  Rows extends
    | GraphDataView<'float32x3'>
    | GraphVectorView<'float32x3'> = GraphDataView<'float32x3'>
> = {
  id?: string;
  /** Longitude/latitude in degrees and height in declared units. */
  positions: Rows;
  /** Optional residual words; positions + positionsLow retain sub-f32 coordinate detail. */
  positionsLow?: Rows;
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
    positions: Rows;
    /** Required with positionsLow; residual relative to the unchanged input high words. */
    positionsLow?: Rows;
    /** {@link GPU_DATUM_GRID_TRANSFORM_STATUS} values. */
    status: GPUDatumGridTransformRowView<'uint32', Rows>;
  };
};

/**
 * GPU-resident bilinear datum shift with ordered grids and bounded inverse iteration.
 * Positions, optional low words and all outputs accept identical ordered chunk lengths; the
 * correction samples and sample mask remain packed and resident across every dispatch.
 */
export class GPUDatumGridTransform<
  Rows extends
    | GraphDataView<'float32x3'>
    | GraphVectorView<'float32x3'> = GraphDataView<'float32x3'>
> implements GPUCommandNodeProducer
{
  readonly id: string;
  /** Upper bound including the chunk parameter upload. */
  readonly estimatedCommandNodeCount: number;
  private readonly chunkPlan: GraphChunkDispatchPlan;
  /** The sample mask, `props.mask`. */
  readonly mask: GraphDataView<'uint32'>;
  readonly props: GPUDatumGridTransformProps<Rows>;
  constructor(props: GPUDatumGridTransformProps<Rows>) {
    this.id = props.id ?? 'datum-grid-transform';
    validateRequirement(
      this.id,
      props.operation === 'horizontal' ||
        props.operation === 'vertical' ||
        props.operation === 'combined',
      `operation received ${String(props.operation)}; accepted values are horizontal, vertical, combined`
    );
    validateRequirement(
      this.id,
      props.direction === undefined ||
        props.direction === 'forward' ||
        props.direction === 'inverse',
      `direction received ${String(props.direction)}; accepted values are forward, inverse`
    );
    this.props = {
      ...props,
      output: {...props.output},
      grids: props.grids.map(grid => ({
        ...grid,
        origin: [...grid.origin] as [number, number],
        spacing: [...grid.spacing] as [number, number]
      }))
    };
    const {mask} = props;
    validateRequirement(this.id, mask !== undefined, 'mask must be given');
    this.mask = mask;
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
    validateSinglePackedView(props.samples, ['float32x3'], `${this.id} samples`);
    validateSinglePackedView(this.mask, ['uint32'], `${this.id} mask`);
    for (const [name, view, format] of [
      ['positions', props.positions, 'float32x3'],
      ['positionsLow', props.positionsLow, 'float32x3'],
      ['output.positions', props.output.positions, 'float32x3'],
      ['output.positionsLow', props.output.positionsLow, 'float32x3'],
      ['output.status', props.output.status, 'uint32']
    ] as const) {
      for (const chunk of view ? getGraphViewChunks(view) : [])
        validatePackedView(chunk, [format], `${this.id} ${name}`);
    }
    // Output rows are aligned, grid indices are exact u32 offsets, and interpolation needs 2x2 cells.
    validateViewLength(this.id, 'output.positions', props.output.positions, props.positions.length);
    validateViewLength(this.id, 'output.status', props.output.status, props.positions.length);
    validateViewLength(this.id, 'mask', this.mask, props.samples.length);
    validateCapacity(
      this.id,
      'samples.length',
      props.samples.length,
      0,
      DATUM_MAXIMUM_SAMPLE_COUNT
    );
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
    validateCapacity(
      this.id,
      'maximumIterations',
      props.maximumIterations ?? 16,
      1,
      DATUM_MAXIMUM_ITERATIONS
    );
    if (props.operation !== 'horizontal') {
      // Vertical source and target references are mandatory, even when only height units differ.
      validateRequirement(
        this.id,
        Boolean(props.sourceVerticalReference) && Boolean(props.targetVerticalReference),
        `${props.operation} operation requires sourceVerticalReference and targetVerticalReference`
      );
    }
    this.chunkPlan = planGraphChunkDispatch(this.id, {
      positions: props.positions,
      positionsLow: props.positionsLow,
      'output.positions': props.output.positions,
      'output.positionsLow': props.output.positionsLow,
      'output.status': props.output.status
    });
    validateDisjointWritableViews(this.id, props.output);
    this.estimatedCommandNodeCount = 1 + this.chunkPlan.nonemptyChunkCount;
    validateCommandNodeBudget(this.id, this.estimatedCommandNodeCount);
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
    validateGraphChunkDispatchBindings(graph, id, this.chunkPlan);
    validateGraphChunkBindingRanges(graph, id, {samples: props.samples, mask: this.mask});
    if (this.chunkPlan.nonemptyChunkCount === 0) return [];
    const chunked = [props.positions, props.positionsLow, ...Object.values(props.output)].some(
      view => view && !('buffer' in view)
    );
    const horizontal = props.operation !== 'vertical';
    const vertical = props.operation !== 'horizontal';
    const inverse = props.direction === 'inverse';
    const kernelProps: WGSLKernelProps<Parameters> = {
      id,
      operation: 'GPUDatumGridTransform',
      variant: `${props.operation}-${props.direction ?? 'forward'}`,
      invocationCount: props.positions.length,
      bindings: [
        {
          name: 'positions',
          view: getGraphViewChunks(props.positions)[0],
          type: 'f32',
          access: 'read'
        },
        ...(props.positionsLow
          ? [
              {
                name: 'positionsLow',
                view: getGraphViewChunks(props.positionsLow)[0],
                type: 'f32',
                access: 'read'
              } satisfies WGSLKernelBinding,
              {
                name: 'outputLow',
                view: getGraphViewChunks(props.output.positionsLow!)[0],
                type: 'f32',
                access: 'read_write'
              } satisfies WGSLKernelBinding
            ]
          : []),
        {name: 'samples', view: props.samples, type: 'f32', access: 'read'},
        {name: 'validity', view: this.mask, type: 'u32', access: 'read'},
        {
          name: 'output',
          view: getGraphViewChunks(props.output.positions)[0],
          type: 'f32',
          access: 'read_write'
        },
        {
          name: 'status',
          view: getGraphViewChunks(props.output.status)[0],
          type: 'u32',
          access: 'read_write'
        }
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
  return max(${getWGSLFloatLiteral(props.tolerance ?? DEFAULT_INVERSE_TOLERANCE_DEGREES)}, 4.8e-7 * max(magnitude.x, magnitude.y));
}
fn loadDatumSample(row: u32) -> vec3f { let offset = samplesOffset + row * 3u; return vec3f(samples[offset], samples[offset + 1u], samples[offset + 2u]); }
fn sampleDatum(position: vec2f, low: vec2f) -> GridSample {
${props.grids
  .map(
    grid => `{
  let highDifference = fma(position, vec2f(1.0), -vec2f(${grid.origin.map(value => getWGSLFloatLiteral(value)).join(',')}));
  let lowDifference = fma(low, vec2f(1.0), -vec2f(${grid.origin.map(value => getWGSLFloatLiteral(value - Math.fround(value))).join(',')}));
  if (!datumFinite(lowDifference.x) || !datumFinite(lowDifference.y) || !datumFinite(highDifference.x) || !datumFinite(highDifference.y)) { return GridSample(vec3f(0.0), ${GPU_DATUM_GRID_TRANSFORM_STATUS.invalid}u); }
  let rawCoordinate = (highDifference + lowDifference) / vec2f(${grid.spacing.map(getWGSLFloatLiteral).join(',')});
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
  ${vertical ? `displacement.z = sourceLow.z + ${inverse ? '-' : ''}sampled.shift.z / ${getWGSLFloatLiteral(props.heightUnitMeters ?? 1)};` : ''}
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
    };
    if (!chunked) return [createWGSLKernelNode(graph, kernelProps)];
    const layouts = this.chunkPlan.chunks.map(chunk =>
      getBoundedDispatchLayout(
        id,
        chunk.length,
        DATUM_CHUNK_WORKGROUP_SIZE,
        graph.device.limits.maxComputeWorkgroupsPerDimension
      )
    );
    const source = `
@group(0) @binding(0) var<storage, read> chunkParameters: array<u32>;
${kernelProps.bindings.map((binding, bindingIndex) => `@group(0) @binding(${bindingIndex + 1}) var<storage, ${binding.access}> ${binding.name}: array<${binding.type}>;`).join('\n')}
const samplesOffset: u32 = ${getViewElementOffset(props.samples)}u;
const validityOffset: u32 = ${getViewElementOffset(this.mask)}u;
${kernelProps.declarations}
@compute @workgroup_size(${DATUM_CHUNK_WORKGROUP_SIZE})
fn main(@builtin(workgroup_id) workgroupId: vec3u,
        @builtin(num_workgroups) workgroupCount: vec3u,
        @builtin(local_invocation_index) localInvocationIndex: u32) {
  ${kernelProps.bindings.map(binding => `_ = &${binding.name};`).join('\n  ')}
  let workgroupIndex = (workgroupId.z * workgroupCount.y + workgroupId.y) * workgroupCount.x + workgroupId.x;
  if (workgroupIndex >= ${DATUM_MAXIMUM_LINEAR_WORKGROUP_COUNT}u) { return; }
  let index = workgroupIndex * ${DATUM_CHUNK_WORKGROUP_SIZE}u + localInvocationIndex;
  let rowCount = chunkParameters[1];
  if (index >= rowCount) { return; }
  ${kernelProps.bindings
    .filter(binding => binding.name !== 'samples' && binding.name !== 'validity')
    .map(
      binding =>
        `let ${binding.name}Offset = arrayLength(&${binding.name}) - ${binding.view.rowByteLength / 4}u * rowCount;`
    )
    .join('\n  ')}
  ${kernelProps.body}
}`;
    const {slots, nodes: uploadNodes} = createGraphChunkDispatchParameters(graph, {
      id: `${id}-chunks`,
      plan: this.chunkPlan
    });
    return [
      ...uploadNodes,
      ...this.chunkPlan.chunks
        .filter(chunk => chunk.length > 0)
        .map(chunk => {
          const parameters = slots[chunk.chunkIndex];
          const chunkBindings = kernelProps.bindings.map(binding => ({
            ...binding,
            view:
              binding.name === 'samples' || binding.name === 'validity'
                ? binding.view
                : chunk.views[
                    binding.name === 'output'
                      ? 'output.positions'
                      : binding.name === 'outputLow'
                        ? 'output.positionsLow'
                        : binding.name === 'status'
                          ? 'output.status'
                          : binding.name
                  ]
          }));
          return createGPUComputeCommandNode<Parameters>({
            id: `${id}-chunk-${chunk.chunkIndex}`,
            workload: {
              operation: 'GPUDatumGridTransform',
              variant: `${props.operation}-${props.direction ?? 'forward'}`,
              commandCount: 1,
              maximumInvocationCount: chunk.length,
              maximumWorkgroupCount:
                layouts[chunk.chunkIndex].x *
                layouts[chunk.chunkIndex].y *
                layouts[chunk.chunkIndex].z
            },
            resources: [
              {buffer: parameters, usage: 'storage-read'},
              ...chunkBindings.map(binding => ({
                buffer: binding.view,
                usage:
                  binding.access === 'read'
                    ? ('storage-read' as const)
                    : ('storage-read-write' as const)
              }))
            ],
            compile: ({device}) => {
              const kernel = new Kernel(device, {
                id: 'datum-grid-transform-chunk',
                source,
                shaderLayout: {
                  bindings: [
                    {name: 'chunkParameters', group: 0, location: 0, type: 'read-only-storage'},
                    ...chunkBindings.map((binding, bindingIndex) => ({
                      name: binding.name,
                      group: 0,
                      location: bindingIndex + 1,
                      type:
                        binding.access === 'read'
                          ? ('read-only-storage' as const)
                          : ('storage' as const)
                    }))
                  ]
                }
              });
              return {
                encode: ({computePass, getBuffer}) =>
                  kernel.dispatch(computePass, {
                    bindings: {
                      chunkParameters: getViewBinding(parameters, getBuffer),
                      ...Object.fromEntries(
                        chunkBindings.map(binding => [
                          binding.name,
                          getViewBinding(binding.view, getBuffer)
                        ])
                      )
                    },
                    ...layouts[chunk.chunkIndex]
                  }),
                destroy: () => kernel.destroy()
              };
            }
          });
        })
    ];
  }
}

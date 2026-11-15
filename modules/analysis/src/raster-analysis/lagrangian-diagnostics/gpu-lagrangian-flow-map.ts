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
import {
  validateCapacity,
  validateFiniteNumber,
  validateGPUStorageBindingByteLength,
  validateGraphViewsBelongToGraph,
  validateMaximumCount,
  validateSinglePackedView,
  validateRequirement,
  validateViewLength
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {GET_NAN_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {getFieldSamplingWGSL} from '../vector-field-common/vector-field-sampling';

import type {GPUSpatialContext} from '../../analysis-core/contracts/spatial-context';
import {
  createLagrangianProjectionNodes,
  resolveLagrangianSpatialContext
} from './lagrangian-geographic-projection';

import {validateRasterBuffersDistinct} from '../raster-band-utils';

const OPERATION = 'GPULagrangianFlowMap';

/** Largest axis length: keeps row-major grid addressing inside uint32. */
export const GPU_LAGRANGIAN_FLOW_MAP_MAXIMUM_FIELD_AXIS = 65535;

/** Number of float32 elements in a {@link GPULagrangianFlowMap} parameter view. */
export const GPU_LAGRANGIAN_FLOW_MAP_PARAMETER_LENGTH = 8;

/**
 * Largest accepted `stepCount`. Every particle integrates all of its steps inside one invocation,
 * so the cap bounds that loop (a bounded per-thread loop, contract section 2), not memory.
 */
export const GPU_LAGRANGIAN_FLOW_MAP_MAXIMUM_STEP_COUNT = 8192;

/** Region value of a cell or particle outside every region. */
export const GPU_LAGRANGIAN_NO_REGION = 0xffffffff;

/** Per-frame settings of {@link GPULagrangianFlowMap}. */
export type GPULagrangianFlowMapSettings = {
  /**
   * Field raster placement `[originX, originY, cellWidth, cellHeight]`, the placement of
   * `GPUParticleAdvection`: row 0 has the smallest y and cell `(i, j)` is centred at
   * `origin + (i + 0.5, j + 0.5) * cellSize`.
   */
  fieldExtent: readonly [number, number, number, number];
  /** Time at which the particles are released, in the time unit of the field velocities. */
  startTime: number;
  /**
   * Signed time step. A negative step integrates backward in time, which is what backward FTLE
   * and backward connectivity need. The integration time is `stepCount * |timeStep|`.
   */
  timeStep: number;
  /** Time at which the field `velocitiesStart` is valid. */
  sliceStartTime: number;
  /** Time at which the field `velocitiesEnd` is valid. */
  sliceEndTime: number;
};

/**
 * Packs the float parameters of {@link GPULagrangianFlowMap}.
 *
 * Layout: `[originX, originY, cellWidth, cellHeight, startTime, timeStep, sliceStartTime,
 * sliceEndTime]` as float32. Times are float32 on the GPU, so release times far from the slice
 * times should be given relative to `sliceStartTime`.
 *
 * @param settings Per-frame settings.
 * @param target Optional destination of at least {@link GPU_LAGRANGIAN_FLOW_MAP_PARAMETER_LENGTH} elements.
 */
export function getGPULagrangianFlowMapParameterValues(
  settings: GPULagrangianFlowMapSettings,
  target: Float32Array = new Float32Array(GPU_LAGRANGIAN_FLOW_MAP_PARAMETER_LENGTH)
): Float32Array {
  if (target.length < GPU_LAGRANGIAN_FLOW_MAP_PARAMETER_LENGTH) {
    throw new Error(
      `${OPERATION} parameter target must hold ${GPU_LAGRANGIAN_FLOW_MAP_PARAMETER_LENGTH} elements`
    );
  }
  const [originX, originY, cellWidth, cellHeight] = settings.fieldExtent;
  for (const [name, value] of [
    ['cellWidth', cellWidth],
    ['cellHeight', cellHeight]
  ] as const) {
    validateFiniteNumber(OPERATION, name, value, Number.MIN_VALUE);
  }
  for (const [name, value] of [
    ['originX', originX],
    ['originY', originY],
    ['startTime', settings.startTime],
    ['timeStep', settings.timeStep],
    ['sliceStartTime', settings.sliceStartTime],
    ['sliceEndTime', settings.sliceEndTime]
  ] as const) {
    validateFiniteNumber(OPERATION, name, value);
  }
  target.set([
    originX,
    originY,
    cellWidth,
    cellHeight,
    settings.startTime,
    settings.timeStep,
    settings.sliceStartTime,
    settings.sliceEndTime
  ]);
  return target;
}

/** Caller-owned outputs of {@link GPULagrangianFlowMap}. */
export type GPULagrangianFlowMapOutput = {
  /** One uint32 per seed: 1 when the final position is valid, 0 when lost or projection-rejected. */
  validity?: GraphDataView<'uint32'>;
  /**
   * Final position of every seed, `seeds.length` rows. A particle that leaves the field or reaches
   * a cell with a NaN velocity ends with a NaN position and no later step is taken.
   */
  positions: GraphDataView<'float32x2'>;
  /**
   * Region of every seed, then the region of every final position, period-major: rows
   * `0..n-1` hold the start regions and rows `n..2n-1` the end regions (`GPU_LAGRANGIAN_NO_REGION`
   * for a lost particle or a cell outside every region). This is the `classes` layout of
   * `GPUTransitionMatrix` with `periodCount: 2`, whose counts are the connectivity matrix and
   * whose probabilities are its row normalization. Requires `regions`.
   */
  regionClasses?: GraphDataView<'uint32'>;
  /**
   * Time in the start region before the first exit, one row per seed, as `|timeStep|` times the
   * number of steps that began inside the start region. Its resolution is one time step. A particle
   * that never leaves reports `stepCount * |timeStep|` (censored), a lost particle counts as an
   * exit, and a seed outside every region reports 0. Requires `regions`.
   */
  residenceTimes?: GraphDataView<'float32'>;
};

/**
 * Properties for {@link GPULagrangianFlowMap}.
 *
 * Per-frame: the contents of `parameters`, `velocitiesStart`, `velocitiesEnd`, `seeds` and
 * `regions`. Compile-time: `fieldWidth`, `fieldHeight`, `stepCount`, the view lengths and which
 * optional outputs exist.
 */
export type GPULagrangianFlowMapProps = {
  /** Prefix for generated node IDs. Defaults to `'lagrangian-flow-map'`. */
  id?: string;
  /** Planar (default), or longitude-latitude with metric 'none' for internally projected integration. */
  spatialContext?: GPUSpatialContext;
  /** Seed positions in the field's frame; degrees in longitude-latitude mode. */
  seeds: GraphDataView<'float32x2'>;
  /** Packed row-major `(u, v)` field valid at `sliceStartTime`, `fieldWidth * fieldHeight` rows. */
  velocitiesStart: GraphDataView<'float32x2'>;
  /** The field at `sliceEndTime`, same layout. */
  velocitiesEnd: GraphDataView<'float32x2'>;
  /** Field width in cells (1 to 65535). */
  fieldWidth: number;
  /** Field height in cells (1 to 65535). */
  fieldHeight: number;
  /** Number of RK2 steps every particle takes (1 to {@link GPU_LAGRANGIAN_FLOW_MAP_MAXIMUM_STEP_COUNT}). */
  stepCount: number;
  /**
   * Per-frame parameters, at least {@link GPU_LAGRANGIAN_FLOW_MAP_PARAMETER_LENGTH} rows, from
   * {@link getGPULagrangianFlowMapParameterValues}.
   */
  parameters: GraphDataView<'float32'>;
  /**
   * Optional region of every field cell, `fieldWidth * fieldHeight` rows, row-major like the
   * field. `GPU_LAGRANGIAN_NO_REGION` marks a cell outside every region. A position takes the
   * region of the cell that contains it (no interpolation).
   */
  regions?: GraphDataView<'uint32'>;
  output: GPULagrangianFlowMapOutput;
};

/**
 * Renames the identifiers of the shared field sampler so that a second copy can sit in the same
 * module, bound to another storage array.
 */
function getSecondFieldSamplingWGSL(bindingName: string, width: number, height: number): string {
  return getFieldSamplingWGSL(bindingName, width, height).replace(
    /\b(FIELD_WIDTH|FIELD_HEIGHT|isNanFloat|readFieldVelocity|isNanVelocity|sampleField)\b/g,
    '$1End'
  );
}

/**
 * Integrates a particle for each seed through a time-varying planar vector field and writes its
 * final position: the Lagrangian flow map, which is the input of FTLE
 * ({@link GPUFiniteTimeLyapunovExponent}), of the connectivity matrix (with `GPUTransitionMatrix`)
 * and of residence times.
 *
 * Method: second-order Runge-Kutta (midpoint) steps `p' = p + h v(p + h/2 v(p, t), t + h/2)` with
 * bilinear sampling in space (the sampler of `GPUParticleAdvection`) and linear interpolation in
 * time between two slices: `v(p, t) = (1 - w) velocitiesStart(p) + w velocitiesEnd(p)` with
 * `w = (t - sliceStartTime) / (sliceEndTime - sliceStartTime)` clamped to `[0, 1]`, so the field is
 * held constant outside the slice interval. The caller advances a longer record by swapping the
 * two slices and the slice times between encodings. There is no respawn, no randomness and no
 * atomic: every particle is independent and a run is bitwise reproducible.
 *
 * Planar velocities are field units per time unit. In longitude-latitude mode, seeds, output
 * positions and fieldExtent are degrees; velocities are Web Mercator projected metres per time
 * unit, NOT ground east/north velocities. Published projection programs project seeds internally
 * and invert each RK2 sampling position into the geographic raster. No wrapping or clamping is
 * performed; outside the Web Mercator square-world domain a particle is lost. Float32 absolute
 * projected coordinates have metre-scale precision far from the projection origin. This mode
 * implements projected advection, not geodesic motion. spatialContext is compile-time.
 *
 * Per-thread loop: each invocation takes all `stepCount` steps of its particle (a bounded loop,
 * capped at {@link GPU_LAGRANGIAN_FLOW_MAP_MAXIMUM_STEP_COUNT}); the particles are the parallel
 * dimension.
 *
 * Reference: the definitions of Lagrangian diagnostics in lcs-parcels (MIT) and the OceanParcels
 * Lagrangian_diags tutorial; tests use a binary64 CPU reference and linear flows with a closed form.
 *
 * Row direction: every field raster places row 0 at the smallest y (`originY` of its extent) and
 * rows advance toward larger y; `v` is the velocity toward larger y. No grid affine is read and
 * nothing is checked at construction. A north-up field (row 0 at the top, the usual raster order)
 * must have its row order reversed by the caller before upload, or the flow is mirrored north to
 * south. Following a grid affine is deferred (roadmap 12.4).
 */
export class GPULagrangianFlowMap implements GPUCommandNodeProducer {
  /** Prefix for graph node IDs. */
  readonly id: string;
  /** Number of seeds. */
  readonly seedCount: number;
  /** Validated properties. */
  readonly props: Readonly<GPULagrangianFlowMapProps>;

  constructor(props: GPULagrangianFlowMapProps) {
    const id = props.id ?? 'lagrangian-flow-map';
    resolveLagrangianSpatialContext(id, props.spatialContext);
    const {fieldWidth, fieldHeight, stepCount, output} = props;
    validateCapacity(id, 'fieldWidth', fieldWidth, 1, GPU_LAGRANGIAN_FLOW_MAP_MAXIMUM_FIELD_AXIS);
    validateCapacity(id, 'fieldHeight', fieldHeight, 1, GPU_LAGRANGIAN_FLOW_MAP_MAXIMUM_FIELD_AXIS);
    validateCapacity(id, 'stepCount', stepCount);
    validateMaximumCount(id, 'stepCount', stepCount, GPU_LAGRANGIAN_FLOW_MAP_MAXIMUM_STEP_COUNT);
    validateSinglePackedView(props.seeds, ['float32x2'], `${id} seeds`);
    validateSinglePackedView(props.velocitiesStart, ['float32x2'], `${id} velocitiesStart`);
    validateSinglePackedView(props.velocitiesEnd, ['float32x2'], `${id} velocitiesEnd`);
    validateViewLength(id, 'velocitiesStart', props.velocitiesStart, fieldWidth * fieldHeight);
    validateViewLength(id, 'velocitiesEnd', props.velocitiesEnd, fieldWidth * fieldHeight);
    validateSinglePackedView(props.parameters, ['float32'], `${id} parameters`);
    validateRequirement(
      id,
      props.parameters.length >= GPU_LAGRANGIAN_FLOW_MAP_PARAMETER_LENGTH,
      `parameters must contain ${GPU_LAGRANGIAN_FLOW_MAP_PARAMETER_LENGTH} float32 rows`
    );
    this.seedCount = props.seeds.length;
    validateCapacity(id, 'seeds', this.seedCount);
    validateSinglePackedView(output.positions, ['float32x2'], `${id} output.positions`);
    validateViewLength(id, 'output.positions', output.positions, this.seedCount);
    if (output.validity) {
      validateSinglePackedView(output.validity, ['uint32'], `${id} output.validity`);
      validateViewLength(id, 'output.validity', output.validity, this.seedCount);
    }
    if (props.regions) {
      validateSinglePackedView(props.regions, ['uint32'], `${id} regions`);
      validateViewLength(id, 'regions', props.regions, fieldWidth * fieldHeight);
    }
    validateRequirement(
      id,
      !(output.regionClasses || output.residenceTimes) || props.regions !== undefined,
      'output.regionClasses and output.residenceTimes require regions'
    );
    if (output.regionClasses) {
      validateSinglePackedView(output.regionClasses, ['uint32'], `${id} output.regionClasses`);
      validateViewLength(id, 'output.regionClasses', output.regionClasses, 2 * this.seedCount);
    }
    if (output.residenceTimes) {
      validateSinglePackedView(output.residenceTimes, ['float32'], `${id} output.residenceTimes`);
      validateViewLength(id, 'output.residenceTimes', output.residenceTimes, this.seedCount);
    }
    validateRasterBuffersDistinct(
      id,
      [output.positions, output.regionClasses, output.residenceTimes, output.validity],
      [props.seeds, props.velocitiesStart, props.velocitiesEnd, props.parameters, props.regions]
    );
    this.id = id;
    this.props = props;
  }

  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, seedCount} = this;
    const {output, regions, fieldWidth, fieldHeight, stepCount} = props;
    validateGraphViewsBelongToGraph(id, graph, [
      props.seeds,
      props.velocitiesStart,
      props.velocitiesEnd,
      props.parameters,
      regions,
      output.positions,
      output.validity,
      output.regionClasses,
      output.residenceTimes
    ]);
    const hasRegions = regions !== undefined;
    validateGPUStorageBindingByteLength(
      graph.device,
      id,
      'seeds',
      seedCount * 8,
      'release fewer particles per graph'
    );
    validateGPUStorageBindingByteLength(
      graph.device,
      id,
      'velocitiesStart/velocitiesEnd',
      fieldWidth * fieldHeight * 8,
      'use a smaller field'
    );
    const geographic = props.spatialContext?.coordinateSpace === 'longitude-latitude';
    const hasDiagnostics = Boolean(
      output.regionClasses || output.residenceTimes || output.validity
    );
    // Pack diagnostics so the integration kernel stays within the core eight-storage-binding limit,
    // including the geographic projection and regions. Region IDs retain their uint32 bits.
    if (hasDiagnostics) {
      validateGPUStorageBindingByteLength(
        graph.device,
        id,
        'diagnostics',
        seedCount * 16,
        'release fewer particles per graph'
      );
    }
    const projection = geographic ? createLagrangianProjectionNodes(graph, id) : undefined;
    const diagnostics = hasDiagnostics
      ? createTransientView(graph, `${id}-diagnostics`, 'uint32x4', seedCount)
      : undefined;
    return [
      ...(projection?.nodes ?? []),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-integrate`,
        operation: OPERATION,
        variant: hasRegions ? 'regions' : 'positions',
        invocationCount: seedCount,
        bindings: [
          {name: 'seeds', view: props.seeds, type: 'f32', access: 'read'},
          {name: 'velocitiesStart', view: props.velocitiesStart, type: 'f32', access: 'read'},
          {name: 'velocitiesEnd', view: props.velocitiesEnd, type: 'f32', access: 'read'},
          {name: 'parameters', view: props.parameters, type: 'f32', access: 'read'},
          {name: 'positions', view: output.positions, type: 'f32', access: 'read_write'},
          ...(hasRegions
            ? [{name: 'regions', view: regions, type: 'u32', access: 'read'} as const]
            : []),
          ...(diagnostics
            ? [{name: 'diagnostics', view: diagnostics, type: 'u32', access: 'read_write'} as const]
            : []),
          ...(projection ? [projection.binding] : [])
        ],
        modules: projection?.modules,
        defines: projection?.defines,
        declarations: /* wgsl */ `
${projection?.declarations ?? ''}
${getFieldSamplingWGSL('velocitiesStart', fieldWidth, fieldHeight)}
${getSecondFieldSamplingWGSL('velocitiesEnd', fieldWidth, fieldHeight)}
${GET_NAN_WGSL}
const STEP_COUNT: u32 = ${stepCount}u;
const NO_REGION: u32 = 0xffffffffu;

fn readParameter(slot: u32) -> f32 { return parameters[parametersOffset + slot]; }

// Time-interpolated velocity: (u, v, valid).
fn sampleVelocity(position: vec2<f32>, extent: vec4<f32>, weight: f32) -> vec3<f32> {
  ${
    projection
      ? `let geographicPosition = ${projection.inverse}(position, 1u);
  if (geographicPosition.valid == 0u) { return vec3<f32>(0.0); }
  let samplePosition = geographicPosition.position;`
      : 'let samplePosition = position;'
  }
  let first = sampleField(samplePosition, extent);
  let second = sampleFieldEnd(samplePosition, extent);
  if (first.z == 0.0 || second.z == 0.0) {
    return vec3<f32>(0.0);
  }
  return vec3<f32>(mix(first.xy, second.xy, weight), 1.0);
}

fn isInsideField(position: vec2<f32>, extent: vec4<f32>) -> bool {
  ${
    projection
      ? `let geographicPosition = ${projection.inverse}(position, 1u);
  if (geographicPosition.valid == 0u) { return false; }
  let samplePosition = geographicPosition.position;`
      : 'let samplePosition = position;'
  }
  let local = (samplePosition - extent.xy) / extent.zw;
  return !isNanVelocity(local) && local.x >= 0.0 && local.y >= 0.0 &&
    local.x < f32(FIELD_WIDTH) && local.y < f32(FIELD_HEIGHT);
}

${
  hasRegions
    ? /* wgsl */ `fn getRegion(position: vec2<f32>, extent: vec4<f32>) -> u32 {
  if (!isInsideField(position, extent)) {
    return NO_REGION;
  }
  ${projection ? `let samplePosition = ${projection.inverse}(position, 1u).position;` : 'let samplePosition = position;'}
  let local = (samplePosition - extent.xy) / extent.zw;
  let column = min(u32(local.x), FIELD_WIDTH - 1u);
  let row = min(u32(local.y), FIELD_HEIGHT - 1u);
  return regions[regionsOffset + row * FIELD_WIDTH + column];
}`
    : ''
}`,
        body: /* wgsl */ `
  let extent = vec4<f32>(readParameter(0u), readParameter(1u), readParameter(2u), readParameter(3u));
  let startTime = readParameter(4u);
  let timeStep = readParameter(5u);
  let sliceStart = readParameter(6u);
  let sliceEnd = readParameter(7u);
  let inverseSpan = select(0.0, 1.0 / (sliceEnd - sliceStart), sliceEnd != sliceStart);
  var position = vec2<f32>(seeds[seedsOffset + 2u * index], seeds[seedsOffset + 2u * index + 1u]);
  ${
    projection
      ? `let projectedSeed = ${projection.forward}(position, 1u);
  position = projectedSeed.position;
  var lost = projectedSeed.valid == 0u || !isInsideField(position, extent);`
      : 'var lost = !isInsideField(position, extent);'
  }
${
  hasRegions
    ? /* wgsl */ `  let startRegion = select(getRegion(position, extent), NO_REGION, lost);
  var exited = startRegion == NO_REGION;
  var insideSteps = 0u;`
    : ''
}
  for (var step = 0u; step < STEP_COUNT; step = step + 1u) {
    if (lost) {
      break;
    }
${
  hasRegions
    ? /* wgsl */ `    if (!exited) {
      if (getRegion(position, extent) == startRegion) {
        insideSteps = insideSteps + 1u;
      } else {
        exited = true;
      }
    }`
    : ''
}
    let time = startTime + f32(step) * timeStep;
    let weightStart = clamp((time - sliceStart) * inverseSpan, 0.0, 1.0);
    let weightMiddle = clamp((time + 0.5 * timeStep - sliceStart) * inverseSpan, 0.0, 1.0);
    let first = sampleVelocity(position, extent, weightStart);
    if (first.z == 0.0) {
      lost = true;
      break;
    }
    let second = sampleVelocity(position + 0.5 * timeStep * first.xy, extent, weightMiddle);
    if (second.z == 0.0) {
      lost = true;
      break;
    }
    position = position + timeStep * second.xy;
    if (!isInsideField(position, extent)) {
      lost = true;
    }
  }
  ${
    projection
      ? `let finalPosition = ${projection.inverse}(position, select(1u, 0u, lost));
  lost = lost || finalPosition.valid == 0u;
  let endPosition = position;
  position = finalPosition.position;`
      : 'let endPosition = position;'
  }
  if (lost) {
    position = vec2<f32>(getNaN());
  }
  positions[positionsOffset + 2u * index] = position.x;
  positions[positionsOffset + 2u * index + 1u] = position.y;
${
  diagnostics
    ? `
  diagnostics[diagnosticsOffset + 4u * index] = ${hasRegions ? 'startRegion' : 'NO_REGION'};
  diagnostics[diagnosticsOffset + 4u * index + 1u] = ${hasRegions ? 'select(getRegion(endPosition, extent), NO_REGION, lost)' : 'NO_REGION'};
  diagnostics[diagnosticsOffset + 4u * index + 2u] = bitcast<u32>(${hasRegions ? 'f32(insideSteps) * abs(timeStep)' : '0.0'});
  diagnostics[diagnosticsOffset + 4u * index + 3u] = select(1u, 0u, lost);`
    : ''
}`
      }),
      ...(diagnostics
        ? [
            createWGSLKernelNode<Parameters>(graph, {
              id: `${id}-publish-diagnostics`,
              operation: OPERATION,
              invocationCount: seedCount,
              bindings: [
                {name: 'diagnostics', view: diagnostics, type: 'u32', access: 'read'},
                ...(output.regionClasses
                  ? [
                      {
                        name: 'regionClasses',
                        view: output.regionClasses,
                        type: 'u32',
                        access: 'read_write'
                      } as const
                    ]
                  : []),
                ...(output.residenceTimes
                  ? [
                      {
                        name: 'residenceTimes',
                        view: output.residenceTimes,
                        type: 'f32',
                        access: 'read_write'
                      } as const
                    ]
                  : []),
                ...(output.validity
                  ? [
                      {
                        name: 'validity',
                        view: output.validity,
                        type: 'u32',
                        access: 'read_write'
                      } as const
                    ]
                  : [])
              ],
              body: `
${
  output.regionClasses
    ? `regionClasses[regionClassesOffset + index] = diagnostics[diagnosticsOffset + 4u * index];
regionClasses[regionClassesOffset + ${seedCount}u + index] = diagnostics[diagnosticsOffset + 4u * index + 1u];`
    : ''
}
${output.residenceTimes ? 'residenceTimes[residenceTimesOffset + index] = bitcast<f32>(diagnostics[diagnosticsOffset + 4u * index + 2u]);' : ''}
${output.validity ? 'validity[validityOffset + index] = diagnostics[diagnosticsOffset + 4u * index + 3u];' : ''}`
            })
          ]
        : [])
    ];
  }
}

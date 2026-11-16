// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  getGPUEllipsoid,
  getGPUSphereRadius,
  type GPUSpatialEllipsoid
} from '../../analysis-core/contracts/spatial-context';
import {
  createGPUComputeCommandNode,
  getBoundedDispatchLayout,
  getViewBinding,
  validatePackedView,
  type GraphVectorView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {Kernel} from '@luma.gl/graph';
import type {GPUCommandNodeProducer} from '@luma.gl/graph/gpu-core';
import {
  createWGSLKernelNode,
  getWGSLFloatLiteral,
  type WGSLKernelProps,
  type WGSLKernelBinding
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  validateCapacity,
  validateFiniteNumber,
  validateRequirement,
  validateViewLength,
  validateGraphOutputsDisjointFromInputs,
  getGraphViewChunks,
  validateCommandNodeBudget
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  resolveGPUSpatialContext,
  type GPUSpatialContext,
  type GPUSpatialContextSupport
} from '../../analysis-core/contracts/index';
import {RHUMB_WGSL} from './rhumb-wgsl';
import {GEODESIC_WGSL} from './geodesic-wgsl';
import {
  getVincentyWGSL,
  GPU_GEODESIC_DEFAULT_ITERATIONS,
  GPU_GEODESIC_MAXIMUM_ITERATIONS,
  GPU_GEODESIC_INVERSE_ANTIPODAL_MARGIN,
  SPHERE_PAIR_WGSL,
  type GPUGeodesicModel
} from './geodesic-kernels';

import {
  createGraphChunkDispatchParameters,
  planGraphChunkDispatch,
  validateDisjointWritableViews,
  validateGraphChunkDispatchBindings,
  type GraphChunkDispatchPlan
} from '../../analysis-core/kernels/graph-chunk-dispatch';

const OPERATION = 'GPUGeodesicPairs';
// Chunk parameters consume one of WebGPU's eight guaranteed storage bindings.
const MAXIMUM_CHUNK_OUTPUT_BINDINGS = 5;
// Fixed workgroups and the pre-multiplication guard preserve uint32 indices at large dispatches.
const GEODESIC_CHUNK_WORKGROUP_SIZE = 256;
// A 256-thread workgroup must be guarded before its index multiplication can wrap uint32.
const GEODESIC_MAXIMUM_LINEAR_WORKGROUP_COUNT = 0x1000000;
// WebGPU guarantees eight storage bindings per stage; reserve two for the input columns.
const MAXIMUM_OUTPUT_BINDINGS = 6;

/** Preserves the source view kind for each row-aligned output format. */
type GPUGeodesicPairsRowView<
  Format extends 'float32' | 'float32x2' | 'float32x3' | 'uint32',
  Rows extends GraphDataView | GraphVectorView
> = Rows extends GraphDataView ? GraphDataView<Format> : GraphVectorView<Format>;

/** Per-pair outputs of {@link GPUGeodesicPairs}; each optional, one row per pair. */
export type GPUGeodesicPairsOutput<
  Rows extends
    | GraphDataView<'float32x2'>
    | GraphVectorView<'float32x2'> = GraphDataView<'float32x2'>
> = {
  /** Geodesic distance in the context's length unit. */
  distances?: GPUGeodesicPairsRowView<'float32', Rows>;
  /** Initial bearing at the origin, degrees clockwise from north in `(-180, 180]` (turf `bearing`). */
  initialBearings?: GPUGeodesicPairsRowView<'float32', Rows>;
  /** Bearing on arrival at the target, degrees clockwise from north in `(-180, 180]`. */
  finalBearings?: GPUGeodesicPairsRowView<'float32', Rows>;
  /**
   * Great-circle midpoint (turf `midpoint`), longitude continuous with the origin. Spherical in
   * every model. For `'rhumb'` it is the midpoint of the rhumb line by distance: the mean
   * latitude, and the longitude the loxodrome has at that latitude, which is linear in the
   * Mercator latitude and so is not the mean longitude unless the line runs east-west.
   */
  midpoints?: Rows;
  /** Per-pair status scalars. */
  status?: {
    /** `'wgs84'` only: `1` when Vincenty resolved outside the antipodal exclusion;
     * `0` for a sphere fallback, including a numerically stopped but uncertifiable inverse. */
    converged?: GPUGeodesicPairsRowView<'uint32', Rows>;
    /** One for an ellipsoidal sphere fallback or a spherical bearing/midpoint within
     * 0.02 radians of an antipode. Zero does not certify arbitrary custom ellipsoids. */
    approximation?: GPUGeodesicPairsRowView<'uint32', Rows>;
    /** One if Vincenty used its entire iteration budget without stopping.
     * A singular inverse rejected before iteration reports zero. */
    iterationLimitReached?: GPUGeodesicPairsRowView<'uint32', Rows>;
  };
};

/**
 * Properties for {@link GPUGeodesicPairs}.
 *
 * Per-frame (no recompile): the contents of `origins` and `targets`. Compile-time: the pair count,
 * `spatialContext`, `maximumIterations`, and which outputs are present.
 */
export type GPUGeodesicPairsProps<
  Rows extends
    | GraphDataView<'float32x2'>
    | GraphVectorView<'float32x2'> = GraphDataView<'float32x2'>
> = {
  /** Prefix for generated node IDs. Defaults to `'geodesic-pairs'`. */
  id?: string;
  /** Longitude/latitude metric and output-unit contract. */
  spatialContext: GPUSpatialContext;
  /** Origins, longitude/latitude degrees. */
  origins: Rows;
  /** Targets aligned with `origins`. */
  targets: Rows;
  /** Compile-time Vincenty iteration cap for `'wgs84'`. Default 16. */
  maximumIterations?: number;
  /** Per-pair outputs; at least one. */
  output: GPUGeodesicPairsOutput<Rows>;
};

/**
 * Geodesic distance, initial and final bearing, and midpoint between many origin/target pairs
 * (turf `distance`, `bearing`, `midpoint`; PostGIS `ST_Distance(geography)`, `ST_Azimuth`).
 *
 * `'sphere'` uses the shared haversine helpers, which subtract coordinates in degrees before any
 * trigonometry, so meter-scale pairs keep their precision. `'wgs84'` runs Vincenty's inverse in f32
 * with the short-line rewrites described on `getVincentyWGSL` and a compile-time iteration cap;
 * pairs that do not converge, or lie within 0.02 radians of an antipode, fall back to the sphere
 * result with `converged = 0` and `approximation = 1`. `'rhumb'` evaluates the Mercator-stretched closed form in `rhumb-wgsl.ts`
 * (east-west limit, antimeridian short way, latitudes clamped at 89.9999 degrees).
 * One invocation per pair; no reductions, so results are deterministic. Packed and chunked inputs
 * are supported; every input and output column must have identical ordered chunk lengths.
 *
 * Every model takes longitude differences with `geodesicLongitudeDelta`, so a pair a few
 * kilometers apart across the antimeridian keeps its f32 precision.
 *
 * **Relation to luma.gl.** `GPUHaversineDistance` (`@luma.gl/graph/geospatial`) computes the same
 * great-circle distance as the `'sphere'` model. It defaults to a radius of 6371 and so returns
 * kilometers, where this class takes the radius and unit from the spatial context (6371008.8 m
 * by default: 8.1 m more over New York to Paris). The sphere model stays local because the
 * official f32 kernel subtracts raw longitudes (1.7 m lost on a 6 km pair across the
 * antimeridian) and has no bearing or midpoint output.
 */
export class GPUGeodesicPairs<
  Rows extends
    | GraphDataView<'float32x2'>
    | GraphVectorView<'float32x2'> = GraphDataView<'float32x2'>
> implements GPUCommandNodeProducer
{
  /** Prefix for every node ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUGeodesicPairsProps<Rows>;
  /** Canonical coordinate and metric contract. */
  readonly spatialContext: GPUSpatialContext;
  /** Resolved model. */
  readonly model: GPUGeodesicModel;
  /** Resolved sphere radius in the context's length unit. */
  readonly radius: number;
  /** Ellipsoid in the context's length unit, used by the 'wgs84'/ellipsoidal model. */
  readonly ellipsoid: GPUSpatialEllipsoid;
  /** Resolved Vincenty iteration cap. */
  readonly maximumIterations: number;
  /** Upper bound including upload and output groups for all nonempty chunks. */
  readonly estimatedCommandNodeCount: number;
  private readonly chunkPlan: GraphChunkDispatchPlan;

  constructor(props: GPUGeodesicPairsProps<Rows>) {
    this.id = props.id ?? 'geodesic-pairs';
    this.props = props;
    this.spatialContext = props.spatialContext;
    const metric = resolveGeodesicSpatialContext(this.id, this.spatialContext);
    this.model = metric.model;
    this.radius = metric.radius;
    this.ellipsoid = metric.ellipsoid;
    this.maximumIterations = props.maximumIterations ?? GPU_GEODESIC_DEFAULT_ITERATIONS;
    validateGeodesicOptions(
      this.id,
      this.model,
      this.radius,
      this.maximumIterations,
      this.ellipsoid
    );
    const {id} = this;
    const {output} = props;
    for (const [name, view] of [
      ['origins', props.origins],
      ['targets', props.targets]
    ] as const) {
      for (const chunk of getGraphViewChunks(view))
        validatePackedView(chunk, ['float32x2'], `${id} ${name}`);
    }
    const pairCount = props.origins.length;
    validateCapacity(id, 'origins.length', pairCount);
    validateViewLength(id, 'targets', props.targets, pairCount);
    const formats = {
      distances: 'float32',
      initialBearings: 'float32',
      finalBearings: 'float32',
      midpoints: 'float32x2',
      'status.converged': 'uint32',
      'status.approximation': 'uint32',
      'status.iterationLimitReached': 'uint32'
    } as const;
    validateGeodesicColumns(id, output, formats, pairCount);
    validateRequirement(
      id,
      !output.status?.converged || this.model === 'wgs84',
      "output.status.converged requires model 'wgs84'"
    );
    const outputFields = flattenGeodesicOutput(output);
    this.chunkPlan = planGraphChunkDispatch(id, {
      origins: props.origins,
      targets: props.targets,
      ...outputFields
    });
    validateDisjointWritableViews(id, outputFields);
    this.estimatedCommandNodeCount =
      1 +
      this.chunkPlan.nonemptyChunkCount *
        Math.ceil(
          Object.values(outputFields).filter(Boolean).length / MAXIMUM_CHUNK_OUTPUT_BINDINGS
        );
    validateCommandNodeBudget(id, this.estimatedCommandNodeCount);
    validateGraphOutputsDisjointFromInputs(id, Object.values(outputFields), [
      props.origins,
      props.targets
    ]);
  }

  /** Returns bounded per-pair nodes, split when optional outputs exceed eight bindings. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {props, id} = this;
    const {output} = props;
    validateGraphChunkDispatchBindings(graph, id, this.chunkPlan);
    const chunked =
      !('buffer' in props.origins) ||
      !('buffer' in props.targets) ||
      Object.values(flattenGeodesicOutput(output)).some(view => view && !('buffer' in view));
    const bindings: WGSLKernelBinding[] = [
      {name: 'origins', view: getGraphViewChunks(props.origins)[0], type: 'f32', access: 'read'},
      {name: 'targets', view: getGraphViewChunks(props.targets)[0], type: 'f32', access: 'read'}
    ];
    const statements: string[] = [];
    const rhumb = this.model === 'rhumb';
    const add = (
      name:
        | Exclude<keyof GPUGeodesicPairsOutput, 'status'>
        | 'converged'
        | 'approximation'
        | 'iterationLimitReached',
      type: 'f32' | 'u32',
      statement: string
    ) => {
      const view =
        name === 'converged' || name === 'approximation' || name === 'iterationLimitReached'
          ? output.status?.[name]
          : output[name];
      if (view) {
        bindings.push({name, view: getGraphViewChunks(view)[0], type, access: 'read_write'});
        statements.push(statement);
      }
    };
    add('distances', 'f32', 'distances[distancesOffset + index] = distance;');
    add(
      'initialBearings',
      'f32',
      'initialBearings[initialBearingsOffset + index] = initialBearing;'
    );
    add('finalBearings', 'f32', 'finalBearings[finalBearingsOffset + index] = finalBearing;');
    add(
      'midpoints',
      'f32',
      `let midpoint = ${
        rhumb
          ? 'rhumbMidpoint(origin, targetPosition)'
          : 'geodesicInterpolate(origin, targetPosition, geodesicCentralAngle(origin, targetPosition), 0.5)'
      };
  midpoints[midpointsOffset + 2u * index] = midpoint.x;
  midpoints[midpointsOffset + 2u * index + 1u] = midpoint.y;`
    );
    add('converged', 'u32', 'converged[convergedOffset + index] = select(0u, 1u, isConverged);');
    add(
      'approximation',
      'u32',
      'approximation[approximationOffset + index] = select(0u, 1u, isApproximation);'
    );
    add(
      'iterationLimitReached',
      'u32',
      'iterationLimitReached[iterationLimitReachedOffset + index] = select(0u, 1u, isIterationLimitReached);'
    );
    const wgs84 = this.model === 'wgs84';
    const outputBindings = bindings.slice(2);
    const maximumOutputs = chunked ? MAXIMUM_CHUNK_OUTPUT_BINDINGS : MAXIMUM_OUTPUT_BINDINGS;
    const outputGroups = Math.ceil(outputBindings.length / maximumOutputs);
    // Resolve every bounded layout before the parameter upload allocates scratch.
    const layouts = this.chunkPlan.chunks.map(chunk =>
      getBoundedDispatchLayout(
        id,
        chunk.length,
        GEODESIC_CHUNK_WORKGROUP_SIZE,
        graph.device.limits.maxComputeWorkgroupsPerDimension
      )
    );
    const dispatchParameters = chunked
      ? createGraphChunkDispatchParameters(graph, {id: `${id}-chunks`, plan: this.chunkPlan})
      : undefined;
    const nodes: GPUCommandNode<Parameters>[] = [...(dispatchParameters?.nodes ?? [])];
    for (let groupIndex = 0; groupIndex < outputGroups; groupIndex++) {
      const firstOutput = groupIndex * maximumOutputs;
      const kernelProps: WGSLKernelProps<Parameters> = {
        id: `${id}-pairs${groupIndex === 0 ? '' : `-${groupIndex}`}`,
        operation: OPERATION,
        variant: this.model,
        bindings: [
          ...bindings.slice(0, 2),
          ...outputBindings.slice(firstOutput, firstOutput + maximumOutputs)
        ],
        invocationCount: props.origins.length,
        declarations: `${GEODESIC_WGSL}
${wgs84 ? getVincentyWGSL(this.maximumIterations, this.ellipsoid) : ''}
${SPHERE_PAIR_WGSL}
${rhumb ? RHUMB_WGSL : ''}
const RADIUS: f32 = ${getWGSLFloatLiteral(this.radius)};`,
        body: /* wgsl */ `let origin = vec2<f32>(origins[originsOffset + 2u * index], origins[originsOffset + 2u * index + 1u]);
  let targetPosition = vec2<f32>(targets[targetsOffset + 2u * index], targets[targetsOffset + 2u * index + 1u]);
  var distance = ${
    rhumb ? 'rhumbAngle(origin, targetPosition)' : 'geodesicCentralAngle(origin, targetPosition)'
  } * RADIUS;
  var initialBearing = ${
    rhumb
      ? 'rhumbBearingDegrees(origin, targetPosition)'
      : 'geodesicInitialBearingDegrees(origin, targetPosition)'
  };
  var finalBearing = ${
    rhumb ? 'initialBearing' : 'sphereFinalBearingDegrees(origin, targetPosition)'
  };
  let antipodal = GEODESIC_PI - geodesicCentralAngle(origin, targetPosition) < ${getWGSLFloatLiteral(GPU_GEODESIC_INVERSE_ANTIPODAL_MARGIN)};
  var isApproximation = ${rhumb ? 'false' : 'antipodal'};
  var isConverged = true;
  var isIterationLimitReached = false;
  ${
    wgs84
      ? `let inverse = vincentyInverse(origin, targetPosition);
  isConverged = inverse.converged && !antipodal;
  isApproximation = !isConverged;
  isIterationLimitReached = inverse.iterationLimitReached;
  if (isConverged) {
    distance = inverse.distance;
    initialBearing = inverse.initialBearing;
    finalBearing = inverse.finalBearing;
  }`
      : ''
  }
  ${statements.slice(firstOutput, firstOutput + maximumOutputs).join('\n  ')}`
      };
      if (!dispatchParameters) {
        nodes.push(createWGSLKernelNode(graph, kernelProps));
        continue;
      }
      const source = `
@group(0) @binding(0) var<storage, read> chunkParameters: array<u32>;
${kernelProps.bindings.map((binding, bindingIndex) => `@group(0) @binding(${bindingIndex + 1}) var<storage, ${binding.access}> ${binding.name}: array<${binding.type}>;`).join('\n')}
${kernelProps.declarations}
@compute @workgroup_size(${GEODESIC_CHUNK_WORKGROUP_SIZE})
fn main(@builtin(workgroup_id) workgroupId: vec3u,
        @builtin(num_workgroups) workgroupCount: vec3u,
        @builtin(local_invocation_index) localInvocationIndex: u32) {
  ${kernelProps.bindings.map(binding => `_ = &${binding.name};`).join('\n  ')}
  let workgroupIndex = (workgroupId.z * workgroupCount.y + workgroupId.y) * workgroupCount.x + workgroupId.x;
  if (workgroupIndex >= ${GEODESIC_MAXIMUM_LINEAR_WORKGROUP_COUNT}u) { return; }
  let index = workgroupIndex * ${GEODESIC_CHUNK_WORKGROUP_SIZE}u + localInvocationIndex;
  let rowCount = chunkParameters[1];
  if (index >= rowCount) { return; }
  ${kernelProps.bindings.map(binding => `let ${binding.name}Offset = arrayLength(&${binding.name}) - ${binding.view.rowByteLength / 4}u * rowCount;`).join('\n  ')}
  ${kernelProps.body}
}`;
      for (const chunk of this.chunkPlan.chunks) {
        if (chunk.length === 0) continue;
        const parameters = dispatchParameters.slots[chunk.chunkIndex];
        const chunkBindings = kernelProps.bindings.map(binding => ({
          ...binding,
          view: chunk.views[
            binding.name === 'converged' ||
            binding.name === 'approximation' ||
            binding.name === 'iterationLimitReached'
              ? `status.${binding.name}`
              : binding.name
          ]
        }));
        nodes.push(
          createGPUComputeCommandNode<Parameters>({
            id: `${id}-pairs-${groupIndex}-chunk-${chunk.chunkIndex}`,
            workload: {
              operation: OPERATION,
              variant: this.model,
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
                id: `geodesic-pairs-chunk-${this.model}-${groupIndex}`,
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
          })
        );
      }
    }
    return nodes;
  }
}

/** Longitude/latitude with a great-circle, rhumb or ellipsoidal metric. */
const GEODESIC_SPATIAL_CONTEXT_SUPPORT: GPUSpatialContextSupport = Object.freeze({
  measure: 'distance',
  coordinateSpaces: Object.freeze(['longitude-latitude'] as const),
  metricsByCoordinateSpace: Object.freeze({
    'longitude-latitude': Object.freeze(['great-circle', 'rhumb', 'ellipsoidal'] as const)
  })
});

const GEODESIC_MODELS = {
  'great-circle': 'sphere',
  rhumb: 'rhumb',
  ellipsoidal: 'wgs84'
} as const;

/**
 * Resolves a canonical spatial context to the kernel model shared by geodesic contributors. A thin
 * wrapper over `resolveGPUSpatialContext`: the support table checks the space and metric, and the
 * wrapper maps the metric to its kernel model. Radius and ellipsoid use the context's length unit.
 */
export function resolveGeodesicSpatialContext(
  id: string,
  spatialContext: GPUSpatialContext
): {model: GPUGeodesicModel; radius: number; ellipsoid: GPUSpatialEllipsoid} {
  const resolved = resolveGPUSpatialContext(id, spatialContext, GEODESIC_SPATIAL_CONTEXT_SUPPORT);
  const model: GPUGeodesicModel = GEODESIC_MODELS[resolved.metric as keyof typeof GEODESIC_MODELS];
  const ellipsoid = getGPUEllipsoid(resolved);
  return {
    model,
    radius: getGPUSphereRadius(resolved),
    ellipsoid:
      resolved.units === 'kilometers'
        ? {...ellipsoid, semiMajorAxis: ellipsoid.semiMajorAxis / 1000}
        : ellipsoid
  };
}

/** Validates the shared model options of the geodesic column contributors. @internal */
export function validateGeodesicOptions(
  id: string,
  model: GPUGeodesicModel,
  radius: number,
  maximumIterations: number,
  ellipsoid?: GPUSpatialEllipsoid
): void {
  validateRequirement(
    id,
    model === 'sphere' || model === 'wgs84' || model === 'rhumb',
    "model must be 'sphere', 'wgs84' or 'rhumb'"
  );
  validateFiniteNumber(id, 'radius', radius);
  validateRequirement(id, Math.fround(radius) > 0, 'radius must be positive in float32');
  if (model === 'wgs84' && ellipsoid) {
    const {semiMajorAxis, flattening} = ellipsoid;
    const semiMinorAxis = semiMajorAxis * (1 - flattening);
    // Match the derived constants emitted by getVincentyWGSL, including JS underflow/overflow.
    const secondEccentricitySquared =
      (semiMajorAxis * semiMajorAxis - semiMinorAxis * semiMinorAxis) /
      (semiMinorAxis * semiMinorAxis);
    for (const [name, value] of Object.entries({
      semiMajorAxis,
      semiMinorAxis,
      flattening,
      secondEccentricitySquared
    })) {
      validateFiniteNumber(id, `spatialContext.ellipsoid.${name}`, value);
    }
    validateRequirement(
      id,
      Math.fround(semiMajorAxis) > 0 && Math.fround(semiMinorAxis) > 0,
      'spatialContext.ellipsoid axes must be positive in float32'
    );
    validateRequirement(
      id,
      Math.fround(flattening) < 1,
      'spatialContext.ellipsoid.flattening must be less than 1 in float32'
    );
  }
  validateCapacity(id, 'maximumIterations', maximumIterations, 1, GPU_GEODESIC_MAXIMUM_ITERATIONS);
}

/**
 * Returns the writable views of a geodesic output keyed by their `output` path (`status.converged`
 * for the per-row convergence flags). @internal
 */
export function flattenGeodesicOutput(output: {
  status?: {
    converged?: GraphDataView | GraphVectorView;
    approximation?: GraphDataView | GraphVectorView;
    iterationLimitReached?: GraphDataView | GraphVectorView;
  };
  [name: string]: unknown;
}): Record<string, GraphDataView | GraphVectorView | undefined> {
  const {status, ...columns} = output;
  return {
    ...(columns as Record<string, GraphDataView | GraphVectorView | undefined>),
    'status.converged': status?.converged,
    ...(status?.approximation ? {'status.approximation': status.approximation} : {}),
    ...(status?.iterationLimitReached
      ? {'status.iterationLimitReached': status.iterationLimitReached}
      : {})
  };
}

/** Validates optional per-row output columns against their formats and row count. @internal */
export function validateGeodesicColumns(
  id: string,
  output: {
    status?: {
      converged?: GraphDataView | GraphVectorView;
      approximation?: GraphDataView | GraphVectorView;
      iterationLimitReached?: GraphDataView | GraphVectorView;
    };
  } & Record<string, unknown>,
  formats: Record<string, 'float32' | 'float32x2' | 'uint32'>,
  rowCount: number
): void {
  let columnCount = 0;
  for (const [name, view] of Object.entries(flattenGeodesicOutput(output))) {
    if (!view) {
      continue;
    }
    const format = formats[name];
    validateRequirement(id, Boolean(format), `output.${name} is not a known column`);
    for (const chunk of getGraphViewChunks(view))
      validatePackedView(chunk, [format], `${id} output.${name}`);
    validateViewLength(id, `output.${name}`, view, rowCount);
    columnCount++;
  }
  validateRequirement(id, columnCount > 0, 'output requires at least one column');
}

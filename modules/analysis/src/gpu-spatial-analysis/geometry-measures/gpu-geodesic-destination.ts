// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

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
  validateRequirement,
  validateViewLength,
  validateGraphOutputsDisjointFromInputs,
  getGraphViewChunks,
  validateCommandNodeBudget
} from '../../analysis-core/kernels/gpu-contributor-utils';
import type {GPUSpatialContext, GPUSpatialEllipsoid} from '../../analysis-core/contracts/index';
import {RHUMB_WGSL} from './rhumb-wgsl';
import {GEODESIC_WGSL} from './geodesic-wgsl';
import {
  getVincentyWGSL,
  GPU_GEODESIC_DEFAULT_ITERATIONS,
  GPU_GEODESIC_DIRECT_MINIMUM_AZIMUTH_COSINE,
  SPHERE_PAIR_WGSL,
  type GPUGeodesicModel
} from './geodesic-kernels';
import {
  resolveGeodesicSpatialContext,
  flattenGeodesicOutput,
  validateGeodesicColumns,
  validateGeodesicOptions
} from './gpu-geodesic-pairs';

import {
  createGraphChunkDispatchParameters,
  planGraphChunkDispatch,
  validateDisjointWritableViews,
  validateGraphChunkDispatchBindings,
  type GraphChunkDispatchPlan
} from '../../analysis-core/kernels/graph-chunk-dispatch';

const OPERATION = 'GPUGeodesicDestination';
// Three inputs plus chunk parameters leave four outputs at WebGPU's eight-binding minimum.
const MAXIMUM_CHUNK_OUTPUT_BINDINGS = 4;
// Fixed workgroups and a guard before multiplication preserve uint32 indices at large dispatches.
const GEODESIC_CHUNK_WORKGROUP_SIZE = 256;
// A 256-thread workgroup must be guarded before its index multiplication can wrap uint32.
const GEODESIC_MAXIMUM_LINEAR_WORKGROUP_COUNT = 0x1000000;

/** Preserves the source view kind for each row-aligned output format. */
type GPUGeodesicDestinationRowView<
  Format extends 'float32' | 'float32x2' | 'float32x3' | 'uint32',
  Rows extends GraphDataView | GraphVectorView
> = Rows extends GraphDataView ? GraphDataView<Format> : GraphVectorView<Format>;

/** Per-row outputs of {@link GPUGeodesicDestination}; each optional, one row per origin. */
export type GPUGeodesicDestinationOutput<
  Rows extends
    | GraphDataView<'float32x2'>
    | GraphVectorView<'float32x2'> = GraphDataView<'float32x2'>
> = {
  /** Destination longitude/latitude, longitude continuous with the origin (may leave ±180). */
  destinations?: Rows;
  /** Bearing on arrival, degrees clockwise from north in `(-180, 180]`. */
  finalBearings?: GPUGeodesicDestinationRowView<'float32', Rows>;
  /** Per-row status scalars. */
  status?: {
    /** `'wgs84'` only: `1` when Vincenty's direct iteration converged. */
    converged?: GPUGeodesicDestinationRowView<'uint32', Rows>;
    /** One when the direct iteration exhausted maximumIterations. */
    iterationLimitReached?: GPUGeodesicDestinationRowView<'uint32', Rows>;
    /** One for an unconverged result, or a nonzero-distance arrival with
     * abs(cos(latitude)) <= 0.03, where arrival azimuth conditioning is excluded. */
    approximation?: GPUGeodesicDestinationRowView<'uint32', Rows>;
  };
};

/**
 * Properties for {@link GPUGeodesicDestination}.
 *
 * Per-frame (no recompile): the contents of every input column. Compile-time: the row count,
 * `spatialContext`, `maximumIterations`, and which outputs are present.
 */
export type GPUGeodesicDestinationProps<
  Rows extends
    | GraphDataView<'float32x2'>
    | GraphVectorView<'float32x2'> = GraphDataView<'float32x2'>
> = {
  /** Prefix for generated node IDs. Defaults to `'geodesic-destination'`. */
  id?: string;
  /** Longitude/latitude metric and output-unit contract. */
  spatialContext: GPUSpatialContext;
  /** Origins, longitude/latitude degrees. */
  origins: Rows;
  /** Initial bearings in degrees clockwise from north, aligned with `origins`. */
  bearings: GPUGeodesicDestinationRowView<'float32', Rows>;
  /** Distances in the context's length unit, aligned with `origins`. */
  distances: GPUGeodesicDestinationRowView<'float32', Rows>;
  /** Compile-time Vincenty iteration cap for `'wgs84'`. Default 16. */
  maximumIterations?: number;
  /** Per-row outputs; at least one. */
  output: GPUGeodesicDestinationOutput<Rows>;
};

/**
 * Destination points from origins, bearings and distances (turf `destination`, PostGIS
 * `ST_Project`), for example range rings, sector wedges or projected vessel positions.
 *
 * `'sphere'` uses the shared spherical destination formula; `'wgs84'` runs Vincenty's direct
 * solution in f32 with a compile-time iteration cap (it converges in a few iterations for every
 * distance shorter than half the meridian). One invocation per row; deterministic. Accepts packed
 * or chunked rows, with every input and output column sharing identical ordered chunk lengths.
 */
export class GPUGeodesicDestination<
  Rows extends
    | GraphDataView<'float32x2'>
    | GraphVectorView<'float32x2'> = GraphDataView<'float32x2'>
> implements GPUCommandNodeProducer
{
  /** Prefix for every node ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUGeodesicDestinationProps<Rows>;
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
  /** Upper bound including upload and optional output splitting for every nonempty chunk. */
  readonly estimatedCommandNodeCount: number;
  private readonly chunkPlan: GraphChunkDispatchPlan;

  constructor(props: GPUGeodesicDestinationProps<Rows>) {
    this.id = props.id ?? 'geodesic-destination';
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
    for (const [name, view, format] of [
      ['origins', props.origins, 'float32x2'],
      ['bearings', props.bearings, 'float32'],
      ['distances', props.distances, 'float32']
    ] as const) {
      for (const chunk of getGraphViewChunks(view))
        validatePackedView(chunk, [format], `${id} ${name}`);
    }
    const rowCount = props.origins.length;
    validateCapacity(id, 'origins.length', rowCount);
    validateViewLength(id, 'bearings', props.bearings, rowCount);
    validateViewLength(id, 'distances', props.distances, rowCount);
    validateGeodesicColumns(
      id,
      props.output,
      {
        destinations: 'float32x2',
        finalBearings: 'float32',
        'status.converged': 'uint32',
        'status.iterationLimitReached': 'uint32',
        'status.approximation': 'uint32'
      },
      rowCount
    );
    validateRequirement(
      id,
      !props.output.status?.converged || this.model === 'wgs84',
      "output.status.converged requires model 'wgs84'"
    );
    const outputFields = flattenGeodesicOutput(props.output);
    this.chunkPlan = planGraphChunkDispatch(id, {
      origins: props.origins,
      bearings: props.bearings,
      distances: props.distances,
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
    validateGraphOutputsDisjointFromInputs(id, Object.values(flattenGeodesicOutput(props.output)), [
      props.origins,
      props.bearings,
      props.distances
    ]);
  }

  /** Returns the single per-row node. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {props, id} = this;
    const {output} = props;
    validateGraphChunkDispatchBindings(graph, id, this.chunkPlan);
    const chunked =
      !('buffer' in props.origins) ||
      !('buffer' in props.bearings) ||
      !('buffer' in props.distances) ||
      Object.values(flattenGeodesicOutput(output)).some(view => view && !('buffer' in view));
    const bindings: WGSLKernelBinding[] = [
      {name: 'origins', view: getGraphViewChunks(props.origins)[0], type: 'f32', access: 'read'},
      {name: 'bearings', view: getGraphViewChunks(props.bearings)[0], type: 'f32', access: 'read'},
      {name: 'distances', view: getGraphViewChunks(props.distances)[0], type: 'f32', access: 'read'}
    ];
    const statements: string[] = [];
    if (output.destinations) {
      bindings.push({
        name: 'destinations',
        view: getGraphViewChunks(output.destinations)[0],
        type: 'f32',
        access: 'read_write'
      });
      statements.push(`destinations[destinationsOffset + 2u * index] = destination.x;
  destinations[destinationsOffset + 2u * index + 1u] = destination.y;`);
    }
    if (output.finalBearings) {
      bindings.push({
        name: 'finalBearings',
        view: getGraphViewChunks(output.finalBearings)[0],
        type: 'f32',
        access: 'read_write'
      });
      statements.push('finalBearings[finalBearingsOffset + index] = finalBearing;');
    }
    if (output.status?.converged) {
      bindings.push({
        name: 'converged',
        view: getGraphViewChunks(output.status.converged)[0],
        type: 'u32',
        access: 'read_write'
      });
      statements.push('converged[convergedOffset + index] = select(0u, 1u, isConverged);');
    }
    if (output.status?.iterationLimitReached) {
      bindings.push({
        name: 'iterationLimitReached',
        view: getGraphViewChunks(output.status.iterationLimitReached)[0],
        type: 'u32',
        access: 'read_write'
      });
      statements.push(
        'iterationLimitReached[iterationLimitReachedOffset + index] = select(1u, 0u, isConverged);'
      );
    }
    if (output.status?.approximation) {
      bindings.push({
        name: 'approximation',
        view: getGraphViewChunks(output.status.approximation)[0],
        type: 'u32',
        access: 'read_write'
      });
      statements.push(
        `approximation[approximationOffset + index] = select(0u, 1u, !isConverged || (distance != 0.0 && abs(geodesicCosLatitude(destination.y)) <= ${getWGSLFloatLiteral(GPU_GEODESIC_DIRECT_MINIMUM_AZIMUTH_COSINE)}));`
      );
    }
    const wgs84 = this.model === 'wgs84';
    const rhumb = this.model === 'rhumb';
    const computeBody = /* wgsl */ `let origin = vec2<f32>(origins[originsOffset + 2u * index], origins[originsOffset + 2u * index + 1u]);
  let bearing = bearings[bearingsOffset + index];
  let distance = distances[distancesOffset + index];
  ${
    wgs84
      ? `let direct = vincentyDirect(origin, bearing, distance);
  let destination = direct.destination;
  let finalBearing = direct.finalBearing;
  let isConverged = direct.converged;`
      : rhumb
        ? `let rhumbResult = rhumbDestination(origin, bearing, distance / RADIUS);
  let destination = rhumbResult.destination;
  let finalBearing = rhumbResult.finalBearing;
  let isConverged = true;`
        : `let destination = geodesicDestination(origin, bearing, distance / RADIUS);
  let finalBearing = sphereDirectFinalBearingDegrees(origin, bearing, distance / RADIUS);
  let isConverged = true;`
  }
`;
    const kernelProps: WGSLKernelProps<Parameters> = {
      id: `${id}-destination`,
      operation: OPERATION,
      variant: this.model,
      bindings,
      invocationCount: props.origins.length,
      declarations: `${GEODESIC_WGSL}
${wgs84 ? getVincentyWGSL(this.maximumIterations, this.ellipsoid) : ''}
${SPHERE_PAIR_WGSL}
${this.model === 'rhumb' ? RHUMB_WGSL : ''}
const RADIUS: f32 = ${getWGSLFloatLiteral(this.radius)};`,
      body: `${computeBody}\n  ${statements.join('\n  ')}`
    };
    // Packed callers retain the original one-node route, which uses at most eight bindings.
    if (!chunked) return [createWGSLKernelNode(graph, kernelProps)];
    const layouts = this.chunkPlan.chunks.map(chunk =>
      getBoundedDispatchLayout(
        id,
        chunk.length,
        GEODESIC_CHUNK_WORKGROUP_SIZE,
        graph.device.limits.maxComputeWorkgroupsPerDimension
      )
    );
    const {slots, nodes: uploadNodes} = createGraphChunkDispatchParameters(graph, {
      id: `${id}-chunks`,
      plan: this.chunkPlan
    });
    const nodes: GPUCommandNode<Parameters>[] = [...uploadNodes];
    const outputBindings = bindings.slice(3);
    const outputGroups = Math.ceil(outputBindings.length / MAXIMUM_CHUNK_OUTPUT_BINDINGS);
    for (let groupIndex = 0; groupIndex < outputGroups; groupIndex++) {
      const firstOutput = groupIndex * MAXIMUM_CHUNK_OUTPUT_BINDINGS;
      const groupBindings = [
        ...bindings.slice(0, 3),
        ...outputBindings.slice(firstOutput, firstOutput + MAXIMUM_CHUNK_OUTPUT_BINDINGS)
      ];
      // Each statement corresponds to one output binding; keep destinations' two stores together.
      const groupBody = `${computeBody}\n  ${statements.slice(firstOutput, firstOutput + MAXIMUM_CHUNK_OUTPUT_BINDINGS).join('\n  ')}`;
      const source = `
@group(0) @binding(0) var<storage, read> chunkParameters: array<u32>;
${groupBindings.map((binding, bindingIndex) => `@group(0) @binding(${bindingIndex + 1}) var<storage, ${binding.access}> ${binding.name}: array<${binding.type}>;`).join('\n')}
${kernelProps.declarations}
@compute @workgroup_size(${GEODESIC_CHUNK_WORKGROUP_SIZE})
fn main(@builtin(workgroup_id) workgroupId: vec3u,
        @builtin(num_workgroups) workgroupCount: vec3u,
        @builtin(local_invocation_index) localInvocationIndex: u32) {
  ${groupBindings.map(binding => `_ = &${binding.name};`).join('\n  ')}
  let workgroupIndex = (workgroupId.z * workgroupCount.y + workgroupId.y) * workgroupCount.x + workgroupId.x;
  if (workgroupIndex >= ${GEODESIC_MAXIMUM_LINEAR_WORKGROUP_COUNT}u) { return; }
  let index = workgroupIndex * ${GEODESIC_CHUNK_WORKGROUP_SIZE}u + localInvocationIndex;
  let rowCount = chunkParameters[1];
  if (index >= rowCount) { return; }
  ${groupBindings.map(binding => `let ${binding.name}Offset = arrayLength(&${binding.name}) - ${binding.view.rowByteLength / 4}u * rowCount;`).join('\n  ')}
  ${groupBody}
}`;
      for (const chunk of this.chunkPlan.chunks) {
        if (chunk.length === 0) continue;
        const parameters = slots[chunk.chunkIndex];
        const chunkBindings = groupBindings.map(binding => ({
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
            id: `${id}-destination-${groupIndex}-chunk-${chunk.chunkIndex}`,
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
                id: `geodesic-destination-chunk-${this.model}-${groupIndex}`,
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

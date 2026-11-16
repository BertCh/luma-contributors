// luma.gl contributors
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createGraphChunkGroupKernelNode,
  validateGraphChunkGroupKernelSupport
} from './graph-chunk-groups';

import {
  getBoundedDispatchLayout,
  GraphDataView,
  GraphVectorView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer
} from '@luma.gl/graph/gpu-core';
import {resolveGPUSpatialContext, type GPUSpatialContext} from '../contracts/spatial-context';
import {
  validateCapacity,
  validateCommandNodeBudget,
  validateMaximumCount,
  validateRequirement,
  validateViewLength,
  validateSinglePackedView,
  getGraphViewChunks,
  validateGraphOutputsDisjointFromInputs
} from './gpu-contributor-utils';
import {
  GPU_FITTING_MAXIMUM_ROW_COUNT,
  GPU_FITTING_MAXIMUM_WORK,
  validateFittingView
} from './fitting-utils';
import {
  planGraphChunkDispatch,
  validateDisjointWritableViews,
  validateGraphChunkBindingRanges,
  validateGraphChunkDispatchBindings,
  type GraphChunkDispatchPlan
} from './graph-chunk-dispatch';
import {type WGSLKernelBinding} from './wgsl-kernel-nodes';
import {IS_FINITE_VALUE_WGSL} from './wgsl-common';
import {FITTING_SPHERICAL_HARMONIC_WGSL} from './fitting-coordinate-encoding-wgsl';

/** Flat scalar indexing within each chunk uses uint32, even on larger-binding devices. */
const MAXIMUM_ELEMENT_COUNT = 0xffffffff;
/** Fixed workgroup size keeps all row chunks on the same pipeline specialization. */
const WORKGROUP_SIZE = 256;

/** SatCLIP's L=40 has degrees 0 through 39 (1600 real basis values). */
export const GPU_FITTING_MAXIMUM_HARMONIC_DEGREE = 39;
/** Fourier feature width is at most 512 floats; bounds each coordinate's frequency work. */
export const GPU_FITTING_MAXIMUM_FREQUENCY_COUNT = 256;
/** Bounds float32 trigonometric argument reduction error for normalized Fourier coordinates. */
export const GPU_FITTING_MAXIMUM_FOURIER_PHASE = 8192;
/** Two spatial coordinates, with an optional normalized time/height coordinate. */
export const GPU_FITTING_MAXIMUM_COORDINATE_DIMENSIONS = 3;

/** Preserves packed row types by default and the caller's chunked view kind when requested. */
type GPUFittingCoordinateEncoderRowView<
  Format extends GraphDataView['format'],
  Rows extends GraphDataView | GraphVectorView
> = Rows extends GraphVectorView ? GraphVectorView<Format> : GraphDataView<Format>;

export type GPUFittingCoordinateEncoderOutput<
  Rows extends GraphDataView | GraphVectorView = GraphDataView
> = {
  /** Row-major real features; invalid rows are zero-filled. */
  features: GPUFittingCoordinateEncoderRowView<'float32', Rows>;
  /** One per source row, nonzero when retained and finite. */
  validity: GPUFittingCoordinateEncoderRowView<'uint32', Rows>;
};
type GPUFittingCoordinateEncoderCommonProps<
  Rows extends GraphDataView | GraphVectorView = GraphDataView
> = {
  id?: string;
  rowCount: number;
  /** Scalar-packed row-major normalized coordinates (Fourier), or lon/lat degrees (harmonics). */
  coordinates: GPUFittingCoordinateEncoderRowView<'float32', Rows>;
  spatialContext?: GPUSpatialContext;
  mask?: GPUFittingCoordinateEncoderRowView<'uint32', Rows>;
  output: GPUFittingCoordinateEncoderOutput<Rows>;
};
export type GPUFittingCoordinateEncoderProps<
  Rows extends GraphDataView | GraphVectorView = GraphDataView
> = GPUFittingCoordinateEncoderCommonProps<Rows> &
  (
    | {
        encoding: 'fourier';
        dimensions: number;
        /** Caller-supplied frequency vectors, frequencyCount * dimensions scalars, cycles/unit. */
        frequencies: GraphDataView<'float32'>;
        frequencyCount: number;
      }
    | {encoding: 'spherical-harmonics'; maximumDegree: number}
  );

/**
 * Fourier features [cos(2 pi Bx), sin(2 pi Bx)] interleaved per frequency (Tancik et al. 2020), or
 * orthonormal real spherical harmonics through maximumDegree. Harmonics are ordered l*l+l+m,
 * m=-l..l; negative m is sqrt(2) Im(Y_l^|m|), positive m is sqrt(2) Re(Y_l^m), m=0 is Y_l^0.
 * Condon-Shortley phase is included. This explicit convention requires a basis conversion for
 * models using another real-harmonic convention. Input longitude/latitude is in degrees.
 * Compile-time: shapes, encoding, maximumDegree, mask presence and spatialContext. Per-frame:
 * coordinates, frequencies and mask. Normalize/project Fourier coordinates before uploading;
 * phase magnitudes beyond 8192 radians are invalid, protecting float32 trig argument reduction.
 * Coordinates, mask and outputs may be chunked at identical source-row boundaries. Coordinate
 * chunks hold rows * dimensions scalars; feature chunks hold rows * featureCount scalars.
 * Every chunk must fit one device binding. Frequencies stay packed. The fitting row/work budgets
 * are unchanged; they apply to the total source rows.
 */
export class GPUFittingCoordinateEncoder<
  Props extends GPUFittingCoordinateEncoderProps<
    GraphDataView | GraphVectorView
  > = GPUFittingCoordinateEncoderProps
> implements GPUCommandNodeProducer
{
  readonly id: string;
  readonly props: Props & GPUFittingCoordinateEncoderProps<Props['coordinates']>;
  readonly spatialContext: GPUSpatialContext;
  readonly featureCount: number;
  readonly estimatedCommandNodeCount: number;
  private readonly chunkPlan: GraphChunkDispatchPlan;

  constructor(props: Props) {
    this.id = props.id ?? 'fitting-coordinate-encoder';
    this.props = props as Props & GPUFittingCoordinateEncoderProps<Props['coordinates']>;
    const {id} = this;
    validateCapacity(id, 'rowCount', props.rowCount, 0, GPU_FITTING_MAXIMUM_ROW_COUNT);
    validateRequirement(
      id,
      props.encoding === 'fourier' || props.encoding === 'spherical-harmonics',
      'encoding must be fourier or spherical-harmonics'
    );
    this.spatialContext = resolveGPUSpatialContext(id, props.spatialContext, {
      coordinateSpaces: [props.encoding === 'fourier' ? 'planar' : 'longitude-latitude'],
      metrics: props.encoding === 'fourier' ? ['none', 'native'] : ['none', 'great-circle']
    });
    let dimensions: number;
    if (props.encoding === 'fourier') {
      dimensions = props.dimensions;
      validateCapacity(id, 'dimensions', dimensions, 2, GPU_FITTING_MAXIMUM_COORDINATE_DIMENSIONS);
      validateCapacity(
        id,
        'frequencyCount',
        props.frequencyCount,
        1,
        GPU_FITTING_MAXIMUM_FREQUENCY_COUNT
      );
      this.featureCount = 2 * props.frequencyCount;
      validateFittingView(
        id,
        'frequencies',
        props.frequencies,
        'float32',
        props.frequencyCount * dimensions
      );
    } else {
      dimensions = 2;
      validateCapacity(
        id,
        'maximumDegree',
        props.maximumDegree,
        0,
        GPU_FITTING_MAXIMUM_HARMONIC_DEGREE
      );
      this.featureCount = (props.maximumDegree + 1) ** 2;
    }
    validateMaximumCount(
      id,
      'encoding work',
      props.rowCount *
        this.featureCount *
        (props.encoding === 'fourier' ? dimensions : props.maximumDegree + 1),
      GPU_FITTING_MAXIMUM_WORK
    );
    const rowFields = [
      ['coordinates', props.coordinates, dimensions, 'float32'],
      ['mask', props.mask, 1, 'uint32'],
      ['features', props.output.features, this.featureCount, 'float32'],
      ['validity', props.output.validity, 1, 'uint32']
    ] as const;
    const rowViews = Object.fromEntries(
      rowFields.map(([name, view, elementsPerRow, format]) => {
        if (!view) return [name, undefined];
        validateViewLength(id, name, view, props.rowCount * elementsPerRow);
        // Metadata-only row prefixes let the shared planner compare row topology while the actual
        // scalar-packed views keep their public layout and full binding ranges.
        const data = getGraphViewChunks(view).map(chunk => {
          validateSinglePackedView(chunk, [format], `${id} ${name}`);
          validateCapacity(id, `${name} chunk elements`, chunk.length, 0, MAXIMUM_ELEMENT_COUNT);
          validateRequirement(
            id,
            chunk.length % elementsPerRow === 0,
            `${name} chunks must contain whole rows of ${elementsPerRow} elements`
          );
          return new GraphDataView(chunk.buffer, {...chunk, length: chunk.length / elementsPerRow});
        });
        return [
          name,
          new GraphVectorView({
            id: `${id}-${name}-rows`,
            name,
            format: view.format,
            length: props.rowCount,
            valueLength: props.rowCount,
            stride: 1,
            byteStride: 4,
            rowByteLength: 4,
            data
          })
        ];
      })
    );
    this.chunkPlan = planGraphChunkDispatch(id, rowViews);
    validateDisjointWritableViews(id, props.output);
    validateGraphOutputsDisjointFromInputs(
      id,
      [props.output.features, props.output.validity],
      this.getInputViews()
    );
    this.estimatedCommandNodeCount = this.chunkPlan.nonemptyChunkCount;
    validateCommandNodeBudget(id, this.estimatedCommandNodeCount);
  }

  private getInputViews(): (GraphDataView | GraphVectorView)[] {
    const {props} = this;
    return [
      props.coordinates,
      ...(props.mask ? [props.mask] : []),
      ...(props.encoding === 'fourier' ? [props.frequencies] : [])
    ];
  }

  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, featureCount, chunkPlan} = this;
    validateGraphChunkBindingRanges(graph, id, {
      coordinates: props.coordinates,
      mask: props.mask,
      features: props.output.features,
      validity: props.output.validity,
      frequencies: props.encoding === 'fourier' ? props.frequencies : undefined
    });
    validateGraphChunkDispatchBindings(graph, id, chunkPlan);
    validateGraphChunkGroupKernelSupport(
      graph,
      id,
      3 + Number(Boolean(props.mask)) + Number(props.encoding === 'fourier'),
      WORKGROUP_SIZE
    );
    for (const chunk of chunkPlan.chunks) {
      if (chunk.length > 0)
        getBoundedDispatchLayout(
          id,
          chunk.length,
          WORKGROUP_SIZE,
          graph.device.limits.maxComputeWorkgroupsPerDimension
        );
    }

    const coordinates = getGraphViewChunks(props.coordinates);
    const mask = props.mask ? getGraphViewChunks(props.mask) : undefined;
    const features = getGraphViewChunks(props.output.features);
    const validity = getGraphViewChunks(props.output.validity);
    return [
      ...chunkPlan.chunks
        .filter(chunk => chunk.length > 0)
        .map(chunk =>
          createGraphChunkGroupKernelNode(graph, {
            id: `${id}-chunk-${chunk.chunkIndex}`,
            operation: 'GPUFittingCoordinateEncoder',
            variant: props.encoding,
            invocationCount: chunk.length,
            workgroupSize: WORKGROUP_SIZE,
            bindings: [
              {
                name: 'coordinates',
                view: coordinates[chunk.chunkIndex],
                type: 'f32',
                access: 'read'
              },
              ...(props.mask
                ? [
                    {
                      name: 'mask',
                      view: mask![chunk.chunkIndex],
                      type: 'u32' as const,
                      access: 'read' as const
                    }
                  ]
                : []),
              ...(props.encoding === 'fourier'
                ? [
                    {
                      name: 'frequencies',
                      view: props.frequencies,
                      type: 'f32' as const,
                      access: 'read' as const
                    }
                  ]
                : []),
              {
                name: 'features',
                view: features[chunk.chunkIndex],
                type: 'f32',
                access: 'read_write'
              },
              {
                name: 'validity',
                view: validity[chunk.chunkIndex],
                type: 'u32',
                access: 'read_write'
              }
            ] as WGSLKernelBinding[],
            declarations:
              IS_FINITE_VALUE_WGSL +
              (props.encoding === 'spherical-harmonics' ? FITTING_SPHERICAL_HARMONIC_WGSL : ''),
            body: `if (index >= getChunkParameter(1u)) { return; }
  let coordinatesBase = arrayLength(&coordinates) - getChunkParameter(1u) * ${props.encoding === 'fourier' ? props.dimensions : 2}u;
  let featuresBase = arrayLength(&features) - getChunkParameter(1u) * ${featureCount}u;
  let validityBase = arrayLength(&validity) - getChunkParameter(1u);
  ${props.mask ? 'let maskBase = arrayLength(&mask) - getChunkParameter(1u);' : ''}
  var valid = ${props.mask ? 'mask[maskBase + index] != 0u' : 'true'};
  ${
    props.encoding === 'fourier'
      ? `
  for (var coordinate = 0u; coordinate < ${props.dimensions}u; coordinate++) {
    valid = valid && isFiniteValue(coordinates[coordinatesBase + index * ${props.dimensions}u + coordinate]);
  }
  for (var frequency = 0u; frequency < ${props.frequencyCount}u; frequency++) {
    var phase = 0.0;
    for (var coordinate = 0u; coordinate < ${props.dimensions}u; coordinate++) {
      phase += coordinates[coordinatesBase + index * ${props.dimensions}u + coordinate]
        * frequencies[frequenciesOffset + frequency * ${props.dimensions}u + coordinate];
    }
    phase *= 6.283185307179586;
    valid = valid && isFiniteValue(phase) && abs(phase) <= ${GPU_FITTING_MAXIMUM_FOURIER_PHASE}.0;
    features[featuresBase + index * ${featureCount}u + 2u * frequency] = cos(phase);
    features[featuresBase + index * ${featureCount}u + 2u * frequency + 1u] = sin(phase);
  }`
      : `
  let longitude = coordinates[coordinatesBase + 2u * index];
  let latitude = coordinates[coordinatesBase + 2u * index + 1u];
  valid = valid && isFiniteValue(longitude) && isFiniteValue(latitude)
    && abs(longitude) <= 180.0 && abs(latitude) <= 90.0;
  let latitudeRadians = latitude * 0.017453292519943295;
  let longitudeRadians = longitude * 0.017453292519943295;
  for (var degree = 0u; degree <= ${props.maximumDegree}u; degree++) {
    for (var order = 0u; order <= degree; order++) {
      let legendre = normalizedLegendre(degree, order, latitudeRadians);
      let center = degree * degree + degree;
      if (order == 0u) {
        features[featuresBase + index * ${featureCount}u + center] = legendre;
      } else {
        let angle = f32(order) * longitudeRadians;
        features[featuresBase + index * ${featureCount}u + center + order] = 1.4142135623730951 * legendre * cos(angle);
        features[featuresBase + index * ${featureCount}u + center - order] = 1.4142135623730951 * legendre * sin(angle);
      }
    }
  }`
  }
  if (!valid) {
    for (var feature = 0u; feature < ${featureCount}u; feature++) {
      features[featuresBase + index * ${featureCount}u + feature] = 0.0;
    }
  }
  validity[validityBase + index] = select(0u, 1u, valid);`
          })
        )
    ];
  }
}

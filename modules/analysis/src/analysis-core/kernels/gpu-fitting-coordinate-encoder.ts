// luma.gl contributors
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {
  GPUCommandGraph,
  GPUCommandNode,
  GPUCommandNodeProducer,
  GraphDataView
} from '@luma.gl/graph/gpu-core';
import {resolveGPUSpatialContext, type GPUSpatialContext} from '../contracts/spatial-context';
import {
  validateCapacity,
  validateCommandNodeBudget,
  validateMaximumCount,
  validateRequirement
} from './gpu-contributor-utils';
import {
  GPU_FITTING_MAXIMUM_ROW_COUNT,
  GPU_FITTING_MAXIMUM_WORK,
  validateFittingGraph,
  validateFittingView,
  validateFittingWritableViews
} from './fitting-utils';
import {createWGSLKernelNode} from './wgsl-kernel-nodes';
import {IS_FINITE_VALUE_WGSL} from './wgsl-common';
import {FITTING_SPHERICAL_HARMONIC_WGSL} from './fitting-coordinate-encoding-wgsl';

/** SatCLIP's L=40 has degrees 0 through 39 (1600 real basis values). */
export const GPU_FITTING_MAXIMUM_HARMONIC_DEGREE = 39;
/** Fourier feature width is at most 512 floats; bounds each coordinate's frequency work. */
export const GPU_FITTING_MAXIMUM_FREQUENCY_COUNT = 256;
/** Bounds float32 trigonometric argument reduction error for normalized Fourier coordinates. */
export const GPU_FITTING_MAXIMUM_FOURIER_PHASE = 8192;
/** Two spatial coordinates, with an optional normalized time/height coordinate. */
export const GPU_FITTING_MAXIMUM_COORDINATE_DIMENSIONS = 3;

export type GPUFittingCoordinateEncoderOutput = {
  /** Row-major real features; invalid rows are zero-filled. */
  features: GraphDataView<'float32'>;
  /** One per source row, nonzero when retained and finite. */
  validity: GraphDataView<'uint32'>;
};
type GPUFittingCoordinateEncoderCommonProps = {
  id?: string;
  rowCount: number;
  /** Scalar-packed row-major normalized coordinates (Fourier), or lon/lat degrees (harmonics). */
  coordinates: GraphDataView<'float32'>;
  spatialContext?: GPUSpatialContext;
  mask?: GraphDataView<'uint32'>;
  output: GPUFittingCoordinateEncoderOutput;
};
export type GPUFittingCoordinateEncoderProps = GPUFittingCoordinateEncoderCommonProps &
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
 */
export class GPUFittingCoordinateEncoder implements GPUCommandNodeProducer {
  readonly id: string;
  readonly props: GPUFittingCoordinateEncoderProps;
  readonly spatialContext: GPUSpatialContext;
  readonly featureCount: number;
  readonly estimatedCommandNodeCount = 1;

  constructor(props: GPUFittingCoordinateEncoderProps) {
    this.id = props.id ?? 'fitting-coordinate-encoder';
    this.props = props;
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
    validateFittingView(
      id,
      'coordinates',
      props.coordinates,
      'float32',
      props.rowCount * dimensions
    );
    if (props.mask) validateFittingView(id, 'mask', props.mask, 'uint32', props.rowCount);
    validateFittingView(
      id,
      'output.features',
      props.output.features,
      'float32',
      props.rowCount * this.featureCount
    );
    validateFittingView(id, 'output.validity', props.output.validity, 'uint32', props.rowCount);
    validateFittingWritableViews(
      id,
      [props.output.features, props.output.validity],
      this.getInputViews()
    );
    validateCommandNodeBudget(id, this.estimatedCommandNodeCount);
  }

  private getInputViews(): GraphDataView[] {
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
    const {id, props, featureCount} = this;
    validateFittingGraph(id, graph, [
      ...this.getInputViews(),
      props.output.features,
      props.output.validity
    ]);
    return [
      createWGSLKernelNode(graph, {
        id,
        operation: 'GPUFittingCoordinateEncoder',
        variant: props.encoding,
        invocationCount: props.rowCount,
        bindings: [
          {name: 'coordinates', view: props.coordinates, type: 'f32', access: 'read'},
          ...(props.mask
            ? [{name: 'mask', view: props.mask, type: 'u32' as const, access: 'read' as const}]
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
          {name: 'features', view: props.output.features, type: 'f32', access: 'read_write'},
          {name: 'validity', view: props.output.validity, type: 'u32', access: 'read_write'}
        ],
        declarations:
          IS_FINITE_VALUE_WGSL +
          (props.encoding === 'spherical-harmonics' ? FITTING_SPHERICAL_HARMONIC_WGSL : ''),
        body: `var valid = ${props.mask ? 'mask[maskOffset + index] != 0u' : 'true'};
  ${
    props.encoding === 'fourier'
      ? `
  for (var coordinate = 0u; coordinate < ${props.dimensions}u; coordinate++) {
    valid = valid && isFiniteValue(coordinates[coordinatesOffset + index * ${props.dimensions}u + coordinate]);
  }
  for (var frequency = 0u; frequency < ${props.frequencyCount}u; frequency++) {
    var phase = 0.0;
    for (var coordinate = 0u; coordinate < ${props.dimensions}u; coordinate++) {
      phase += coordinates[coordinatesOffset + index * ${props.dimensions}u + coordinate]
        * frequencies[frequenciesOffset + frequency * ${props.dimensions}u + coordinate];
    }
    phase *= 6.283185307179586;
    valid = valid && isFiniteValue(phase) && abs(phase) <= ${GPU_FITTING_MAXIMUM_FOURIER_PHASE}.0;
    features[featuresOffset + index * ${featureCount}u + 2u * frequency] = cos(phase);
    features[featuresOffset + index * ${featureCount}u + 2u * frequency + 1u] = sin(phase);
  }`
      : `
  let longitude = coordinates[coordinatesOffset + 2u * index];
  let latitude = coordinates[coordinatesOffset + 2u * index + 1u];
  valid = valid && isFiniteValue(longitude) && isFiniteValue(latitude)
    && abs(longitude) <= 180.0 && abs(latitude) <= 90.0;
  let latitudeRadians = latitude * 0.017453292519943295;
  let longitudeRadians = longitude * 0.017453292519943295;
  for (var degree = 0u; degree <= ${props.maximumDegree}u; degree++) {
    for (var order = 0u; order <= degree; order++) {
      let legendre = normalizedLegendre(degree, order, latitudeRadians);
      let center = degree * degree + degree;
      if (order == 0u) {
        features[featuresOffset + index * ${featureCount}u + center] = legendre;
      } else {
        let angle = f32(order) * longitudeRadians;
        features[featuresOffset + index * ${featureCount}u + center + order] = 1.4142135623730951 * legendre * cos(angle);
        features[featuresOffset + index * ${featureCount}u + center - order] = 1.4142135623730951 * legendre * sin(angle);
      }
    }
  }`
  }
  if (!valid) {
    for (var feature = 0u; feature < ${featureCount}u; feature++) {
      features[featuresOffset + index * ${featureCount}u + feature] = 0.0;
    }
  }
  validity[validityOffset + index] = select(0u, 1u, valid);`
      })
    ];
  }
}

// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {snapshotConstructionFields} from '../../analysis-core/kernels/construction-snapshot';
import {snapshotRasterGrid} from '../raster-grid/raster-construction-snapshot';

import {validateRasterStorageBindings} from '../raster-grid/raster-storage-preflight';
import {
  createTransientView,
  validatePackedUint32View,
  validatePackedView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView,
  type GraphTextureView
} from '@luma.gl/graph/gpu-core';
import {GPURasterBufferToTexture} from '@luma.gl/graph/gpu-raster';
import {
  createWGSLKernelNode,
  type WGSLKernelBinding
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import type {GPUCommandNodeProducer} from '@luma.gl/graph/gpu-core';
import {
  captureGraphCommandNodes,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {getFieldSamplingWGSL} from '../vector-field-common/vector-field-sampling';
import {
  GPU_LINE_INTEGRAL_CONVOLUTION_PARAMETER_LENGTH,
  GPU_LINE_INTEGRAL_CONVOLUTION_UINT32_PARAMETER_LENGTH,
  LINE_INTEGRAL_CONVOLUTION_NOISE_PURPOSE
} from './line-integral-convolution-parameters';
import {PHILOX_WGSL} from '../../analysis-core/kernels/philox-random';
import {validateRasterGrid} from '../raster-band-utils';
import type {GPURasterMetadata} from '@luma.gl/graph/gpu-raster';

const OPERATION = 'GPULineIntegralConvolution';
const MAXIMUM_STEP_COUNT = 256;

/** Caller-owned outputs of {@link GPULineIntegralConvolution}. */
export type GPULineIntegralConvolutionOutput = {
  /**
   * Row-major LIC values in `[0, 1]`, `width * height` rows, row 0 with the smallest y. NaN where
   * the pixel centre has no field data.
   */
  values: GraphDataView<'float32'>;
  /** Optional field speed at each pixel centre (NaN without data), for colour modulation. */
  speeds?: GraphDataView<'float32'>;
  /** Optional `r32float` storage texture of `width x height` receiving `values`. */
  texture?: GraphTextureView<'r32float'>;
};

/**
 * Properties for {@link GPULineIntegralConvolution}.
 *
 * Per-frame (no recompile): the contents of `parameters` (extents, step length, minimum speed,
 * phase, period), `uint32Parameters` (seed) and `velocities`. Topology: field and output sizes,
 * `stepCount`, which optional outputs are present.
 */
export type GPULineIntegralConvolutionProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'line-integral-convolution'`. */
  id?: string;
  /** Packed row-major `(u, v)` field, `fieldWidth * fieldHeight` rows; NaN marks no data. */
  velocities: GraphDataView<'float32x2'>;
  /** Field width in cells. Compile-time. */
  fieldWidth: number;
  /** Field height in cells. Compile-time. */
  fieldHeight: number;
  /** Output raster grid: its width and height size the kernels. Compile-time. */
  grid: GPURasterMetadata;
  /** Streamline steps in each direction (`L`), at most 256. Compile-time. Defaults to 20. */
  stepCount?: number;
  /** Per-frame float32 parameters from `getGPULineIntegralConvolutionParameterValues`. */
  parameters: GraphDataView<'float32'>;
  /** Per-frame uint32 parameters from `getGPULineIntegralConvolutionUint32ParameterValues`. */
  uint32Parameters: GraphDataView<'uint32'>;
  /** Caller-owned outputs. */
  output: GPULineIntegralConvolutionOutput;
};

/**
 * Line integral convolution (Cabral and Leedom 1993): a texture that smears white noise along the
 * streamlines of a vector field, so the flow direction is visible everywhere at once.
 *
 * The first node fills a `width x height` white-noise raster with
 * `philox((column, row, 0, 0), (seed, 1)).x`. The second traces, from each pixel centre, `L` RK2
 * midpoint steps forward and `L` backward along the normalised field, stopping at the field or
 * output edge, at NaN data, or where the speed is zero or below `minimumSpeed`. Every landing
 * must have usable velocity support before its noise sample is accumulated. One step is the
 * unit world direction `d` times the scalar `stepLength / |(d.x / cellWidth, d.y / cellHeight)|`:
 * it is tangent to the field in world space and `stepLength` output pixels long, also when the
 * output cells are not square. The value is the weighted mean of the noise at the visited pixels with a
 * Hann window `0.5 * (1 + cos(pi * s / (L + 1)))` over the signed step `s`. When `period > 0` the
 * window is multiplied by the ripple `0.5 * (1 + cos(2 * pi * (s / period - phase)))`; advancing
 * `phase` each frame makes the texture flow along the field without recomputing anything else
 * (animated LIC, Forssell 1994). The output extent is independent of the field extent, so a
 * viewport can be rendered at screen resolution from a coarse field.
 *
 * Determinism: no atomics and counter-based noise, so encodings are bitwise reproducible on one
 * adapter. Across adapters, `cos` and `normalize` rounding can move a sample into a neighbouring
 * pixel near a pixel edge.
 *
 * Integration stays in output-cell coordinates. The packer optionally subtracts localOrigin
 * in binary64 (use the output origin) before narrowing placement origins. A landing contributes only
 * after it has usable field support, nonzero speed and an in-bounds output pixel.
 *
 * Row direction: with positive cellHeight, the field and output row-zero centers lie half a cell above originY and
 * rows advance toward larger y; `v` is the velocity toward larger y. Each lattice uses its own
 * signed cell sizes, which explicitly set index direction; the origin is an outer corner. No grid affine is read and
 * nothing is checked at construction. A north-up field (row 0 at the top) can be uploaded as is
 * with a top-edge origin and negative cellHeight. Callers choosing positive cellHeight must
 * reverse those rows and use a bottom-edge origin. Following a grid affine is deferred (roadmap 12.4).
 */
export class GPULineIntegralConvolution implements GPUCommandNodeProducer {
  /** Prefix for graph node and transient IDs. */
  readonly id: string;
  /** Streamline steps in each direction. */
  readonly stepCount: number;
  /** Validated properties. */
  readonly props: Readonly<GPULineIntegralConvolutionProps>;

  constructor(props: GPULineIntegralConvolutionProps) {
    props = snapshotConstructionFields(props, [
      'id',
      'velocities',
      'fieldWidth',
      'fieldHeight',
      'grid',
      'stepCount',
      'parameters',
      'uint32Parameters',
      'output'
    ]);
    props = Object.freeze({
      ...props,
      grid: snapshotRasterGrid(props.grid),
      output: snapshotConstructionFields(props.output, ['values', 'speeds', 'texture'])
    });
    const id = props.id ?? 'line-integral-convolution';
    for (const [name, value] of [
      ['fieldWidth', props.fieldWidth],
      ['fieldHeight', props.fieldHeight],
      ['grid width', props.grid.width],
      ['grid height', props.grid.height]
    ] as const) {
      if (!Number.isInteger(value) || value < 1 || value > 65535) {
        throw new Error(`${id} ${name} must be an integer in [1, 65535]`);
      }
    }
    const stepCount = props.stepCount ?? 20;
    if (!Number.isInteger(stepCount) || stepCount < 0 || stepCount > MAXIMUM_STEP_COUNT) {
      throw new Error(`${id} stepCount must be an integer in [0, ${MAXIMUM_STEP_COUNT}]`);
    }
    validatePackedView(props.velocities, ['float32x2'], `${id} velocities`);
    if (props.velocities.length !== props.fieldWidth * props.fieldHeight) {
      throw new Error(`${id} velocities must have fieldWidth * fieldHeight rows`);
    }
    validatePackedView(props.parameters, ['float32'], `${id} parameters`);
    if (props.parameters.length < GPU_LINE_INTEGRAL_CONVOLUTION_PARAMETER_LENGTH) {
      throw new Error(
        `${id} parameters must contain ${GPU_LINE_INTEGRAL_CONVOLUTION_PARAMETER_LENGTH} float32 rows`
      );
    }
    validatePackedUint32View(props.uint32Parameters, `${id} uint32Parameters`);
    if (props.uint32Parameters.length < GPU_LINE_INTEGRAL_CONVOLUTION_UINT32_PARAMETER_LENGTH) {
      throw new Error(
        `${id} uint32Parameters must contain ${GPU_LINE_INTEGRAL_CONVOLUTION_UINT32_PARAMETER_LENGTH} uint32 rows`
      );
    }
    const pixelCount = validateRasterGrid(id, props.grid);
    const {output} = props;
    validatePackedView(output.values, ['float32'], `${id} output.values`);
    if (output.values.length !== pixelCount) {
      throw new Error(`${id} output.values must have grid width * height rows`);
    }
    if (output.speeds) {
      validatePackedView(output.speeds, ['float32'], `${id} output.speeds`);
      if (output.speeds.length !== pixelCount) {
        throw new Error(`${id} output.speeds must have grid width * height rows`);
      }
    }
    if (
      output.texture &&
      (output.texture.format !== 'r32float' ||
        output.texture.width !== props.grid.width ||
        output.texture.height !== props.grid.height)
    ) {
      throw new Error(`${id} output.texture must be r32float with width x height`);
    }
    validateGraphOutputsDisjointFromInputs(
      id,
      [output.values, output.speeds],
      [props.velocities, props.parameters, props.uint32Parameters]
    );
    this.id = id;
    this.stepCount = stepCount;
    this.props = props;
  }

  /** Returns the noise, convolution and optional texture nodes. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const storageProps = this.props;
    const storageCellCount = storageProps.grid.width * storageProps.grid.height;
    validateRasterStorageBindings(
      graph,
      this.id,
      [
        storageProps.velocities,
        storageProps.parameters,
        storageProps.uint32Parameters,
        storageProps.output.values,
        storageProps.output.speeds
      ],
      {'noise plane': storageCellCount * 4}
    );

    const {id, props, stepCount} = this;
    const {output} = props;
    const {width, height} = props.grid;
    validateGraphViewsBelongToGraph(id, graph, [
      props.velocities,
      props.parameters,
      props.uint32Parameters,
      output.values,
      output.speeds
    ]);
    if (output.texture && output.texture.texture.graph !== graph) {
      throw new Error(`${id} views must belong to the target graph`);
    }
    const pixelCount = width * height;
    const noise = createTransientView(graph, `${id}-noise`, 'float32', pixelCount);
    const nodes: GPUCommandNode<Parameters>[] = [];
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-noise`,
        operation: OPERATION,
        variant: 'noise',
        bindings: [
          {
            name: 'words',
            view: props.uint32Parameters,
            type: 'u32',
            access: 'read'
          },
          {name: 'noise', view: noise, type: 'f32', access: 'read_write'}
        ],
        invocationCount: pixelCount,
        declarations: `${PHILOX_WGSL}
const WIDTH: u32 = ${width}u;`,
        body: `let seed = words[wordsOffset];
  let counter = vec4<u32>(index % WIDTH, index / WIDTH, 0u, 0u);
  noise[noiseOffset + index] = philoxUnitFloat(
    philox4x32(counter, vec2<u32>(seed, ${LINE_INTEGRAL_CONVOLUTION_NOISE_PURPOSE}u)).x
  );`
      })
    );
    const bindings: WGSLKernelBinding[] = [
      {
        name: 'velocities',
        view: props.velocities,
        type: 'f32',
        access: 'read'
      },
      {
        name: 'parameters',
        view: props.parameters,
        type: 'f32',
        access: 'read'
      },
      {name: 'noise', view: noise, type: 'f32', access: 'read'},
      {
        name: 'values',
        view: output.values,
        type: 'f32',
        access: 'read_write'
      }
    ];
    if (output.speeds) {
      bindings.push({
        name: 'speeds',
        view: output.speeds,
        type: 'f32',
        access: 'read_write'
      });
    }
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-convolve`,
        operation: OPERATION,
        variant: 'convolve',
        bindings,
        invocationCount: pixelCount,
        declarations: `${getFieldSamplingWGSL('velocities', props.fieldWidth, props.fieldHeight)}
const WIDTH: u32 = ${width}u;
const HEIGHT: u32 = ${height}u;
const STEP_COUNT: u32 = ${stepCount}u;
const PI: f32 = 3.14159265358979;
fn readParameter(slot: u32) -> f32 { return parameters[parametersOffset + slot]; }
fn getWeight(signedStep: f32, phase: f32, period: f32) -> f32 {
  let window = 0.5 * (1.0 + cos(PI * signedStep / f32(STEP_COUNT + 1u)));
  var ripple = 1.0;
  if (period > 0.0) {
    ripple = 0.5 * (1.0 + cos(2.0 * PI * (signedStep / period - phase)));
  }
  return window * ripple;
}
// Unit flow direction times sign, or (0, 0, 0) where the trace must stop.
fn getDirection(position: vec2<f32>, fieldExtent: vec4<f32>, directionSign: f32, minimumSpeed: f32) -> vec3<f32> {
  let cell = vec2<f32>(readParameter(6u), readParameter(7u));
  let origin = vec2<f32>(readParameter(4u), readParameter(5u));
  let sample = sampleField(position * cell, vec4<f32>(fieldExtent.xy - origin, fieldExtent.zw));
  if (sample.z == 0.0) {
    return vec3<f32>(0.0);
  }
  let speed = length(sample.xy);
  if (!(speed > 0.0) || speed < minimumSpeed) {
    return vec3<f32>(0.0);
  }
  return vec3<f32>(directionSign * sample.xy / speed, 1.0);
}
// Output-cell displacement of a world direction, normalized in pixel space. Mapping back by
// the signed cell size preserves the world direction even for non-square or north-up cells.
fn getStepVector(direction: vec2<f32>, cell: vec2<f32>, stepLength: f32) -> vec2<f32> {
  let pixelDirection = direction / cell;
  return pixelDirection * (stepLength / length(pixelDirection));
}`,
        body: /* wgsl */ `
  let fieldExtent = vec4<f32>(readParameter(0u), readParameter(1u), readParameter(2u), readParameter(3u));
  let origin = vec2<f32>(readParameter(4u), readParameter(5u));
  let cell = vec2<f32>(readParameter(6u), readParameter(7u));
  let stepLength = readParameter(8u);
  let minimumSpeed = readParameter(9u);
  let phase = readParameter(10u);
  let period = readParameter(11u);
  let column = index % WIDTH;
  let row = index / WIDTH;
  // Integration state is in output-cell coordinates, including on the legacy packed route.
  let centre = vec2<f32>(f32(column), f32(row)) + vec2<f32>(0.5);
  let centreSample = sampleField(centre * cell, vec4<f32>(fieldExtent.xy - origin, fieldExtent.zw));
  if (centreSample.z == 0.0) {
    var nanBits = 0x7fc00000u;
    values[valuesOffset + index] = bitcast<f32>(nanBits);
    ${output.speeds ? 'speeds[speedsOffset + index] = bitcast<f32>(nanBits);' : ''}
    return;
  }
  ${output.speeds ? 'speeds[speedsOffset + index] = length(centreSample.xy);' : ''}
  var weightSum = getWeight(0.0, phase, period);
  var sum = weightSum * noise[noiseOffset + index];
  for (var direction = 0u; direction < 2u; direction = direction + 1u) {
    let directionSign = select(1.0, -1.0, direction == 1u);
    var position = centre;
    for (var stepIndex = 1u; stepIndex <= STEP_COUNT; stepIndex = stepIndex + 1u) {
      let first = getDirection(position, fieldExtent, directionSign, minimumSpeed);
      if (first.z == 0.0) {
        break;
      }
      let middle = getDirection(position + 0.5 * getStepVector(first.xy, cell, stepLength), fieldExtent, directionSign, minimumSpeed);
      if (middle.z == 0.0) {
        break;
      }
      position = position + getStepVector(middle.xy, cell, stepLength);
      if (getDirection(position, fieldExtent, directionSign, minimumSpeed).z == 0.0) {
        break;
      }
      let pixel = floor(position);
      if (pixel.x < 0.0 || pixel.y < 0.0 || pixel.x >= f32(WIDTH) || pixel.y >= f32(HEIGHT)) {
        break;
      }
      let weight = getWeight(directionSign * f32(stepIndex), phase, period);
      sum = sum + weight * noise[noiseOffset + u32(pixel.y) * WIDTH + u32(pixel.x)];
      weightSum = weightSum + weight;
    }
  }
  values[valuesOffset + index] = select(noise[noiseOffset + index], sum / weightSum, weightSum > 0.0);`
      })
    );
    if (output.texture) {
      const texture = output.texture;
      nodes.push(
        ...captureGraphCommandNodes(graph, () =>
          new GPURasterBufferToTexture({
            id: `${id}-texture`,
            input: {
              id: `${id}-texture-band`,
              format: 'float32',
              storage: {kind: 'buffer', values: output.values}
            },
            output: texture
          }).addToGraph(graph)
        )
      );
    }
    return nodes;
  }
}

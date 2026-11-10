// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {
  GPUCommandGraph,
  GPUCommandNode,
  GPUCommandNodeProducer,
  GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  validateCapacity,
  validateFiniteNumber,
  validateGPUStorageBindingByteLength,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validatePackedView,
  validateRequirement,
  validateViewLength
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {GET_NAN_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';

const OPERATION = 'GPUFiniteTimeLyapunovExponent';

/** Number of float32 elements in a {@link GPUFiniteTimeLyapunovExponent} parameter view. */
export const GPU_FINITE_TIME_LYAPUNOV_EXPONENT_PARAMETER_LENGTH = 4;

/** Per-frame settings of {@link GPUFiniteTimeLyapunovExponent}. */
export type GPUFiniteTimeLyapunovExponentSettings = {
  /** Distance between neighbouring seeds along x, in field units. */
  seedSpacingX: number;
  /** Distance between neighbouring seeds along y, in field units. */
  seedSpacingY: number;
  /**
   * Integration time `T` of the flow map. The exponent is divided by `|T|`, so a backward flow map
   * (negative time step) gives the backward FTLE, whose ridges are attracting structures.
   */
  integrationTime: number;
};

/**
 * Packs the float parameters of {@link GPUFiniteTimeLyapunovExponent}.
 *
 * Layout: `[seedSpacingX, seedSpacingY, integrationTime, 0]` as float32.
 *
 * @param settings Per-frame settings.
 * @param target Optional destination of at least 4 elements.
 */
export function getGPUFiniteTimeLyapunovExponentParameterValues(
  settings: GPUFiniteTimeLyapunovExponentSettings,
  target: Float32Array = new Float32Array(GPU_FINITE_TIME_LYAPUNOV_EXPONENT_PARAMETER_LENGTH)
): Float32Array {
  if (target.length < GPU_FINITE_TIME_LYAPUNOV_EXPONENT_PARAMETER_LENGTH) {
    throw new Error(
      `${OPERATION} parameter target must hold ${GPU_FINITE_TIME_LYAPUNOV_EXPONENT_PARAMETER_LENGTH} elements`
    );
  }
  validateFiniteNumber(OPERATION, 'seedSpacingX', settings.seedSpacingX, Number.MIN_VALUE);
  validateFiniteNumber(OPERATION, 'seedSpacingY', settings.seedSpacingY, Number.MIN_VALUE);
  validateFiniteNumber(OPERATION, 'integrationTime', settings.integrationTime);
  validateRequirement(OPERATION, settings.integrationTime !== 0, 'integrationTime must not be 0');
  target.set([settings.seedSpacingX, settings.seedSpacingY, settings.integrationTime, 0]);
  return target;
}

/** Properties for {@link GPUFiniteTimeLyapunovExponent}. */
export type GPUFiniteTimeLyapunovExponentProps = {
  /** Prefix for generated node IDs. Defaults to `'finite-time-lyapunov-exponent'`. */
  id?: string;
  /**
   * Final positions of a regular grid of seeds, row-major (`seedWidth * seedHeight` rows, row 0
   * the smallest y), for example `GPULagrangianFlowMap` output. NaN marks a lost particle.
   */
  positions: GraphDataView<'float32x2'>;
  /** Seed grid width in seeds (2 to 65535). */
  seedWidth: number;
  /** Seed grid height in seeds (2 to 65535). */
  seedHeight: number;
  /** Per-frame parameters, at least 4 float32 rows, from {@link getGPUFiniteTimeLyapunovExponentParameterValues}. */
  parameters: GraphDataView<'float32'>;
  /** One float32 per seed. NaN where the particle or a neighbour used by the difference is lost. */
  output: {ftle: GraphDataView<'float32'>};
};

/**
 * Finite-time Lyapunov exponent of a flow map on a regular seed grid:
 * `FTLE = ln(sqrt(lambdaMax(J^T J))) / |T|` where `J` is the gradient of the final positions with
 * respect to the seeds.
 *
 * `J` comes from finite differences of the neighbouring final positions: central differences in
 * the interior and one-sided differences on the grid border. `lambdaMax` is the closed form of the
 * larger eigenvalue of the symmetric 2 by 2 matrix `J^T J`. The result is NaN where any particle
 * in the stencil is lost.
 *
 * Reference: the FTLE definition of lcs-parcels (MIT) and the OceanParcels Lagrangian_diags
 * tutorial; tests check a linear strain field with the known exponent and a binary64 CPU
 * reference. Ridge or curve extraction (hyperbolic and elliptic LCS) is not provided.
 */
export class GPUFiniteTimeLyapunovExponent implements GPUCommandNodeProducer {
  /** Prefix for graph node IDs. */
  readonly id: string;
  /** Validated properties. */
  readonly props: Readonly<GPUFiniteTimeLyapunovExponentProps>;

  constructor(props: GPUFiniteTimeLyapunovExponentProps) {
    const id = props.id ?? 'finite-time-lyapunov-exponent';
    validateCapacity(id, 'seedWidth', props.seedWidth, 2, 65535);
    validateCapacity(id, 'seedHeight', props.seedHeight, 2, 65535);
    validatePackedView(props.positions, ['float32x2'], `${id} positions`);
    validateViewLength(id, 'positions', props.positions, props.seedWidth * props.seedHeight);
    validatePackedView(props.parameters, ['float32'], `${id} parameters`);
    validateRequirement(
      id,
      props.parameters.length >= GPU_FINITE_TIME_LYAPUNOV_EXPONENT_PARAMETER_LENGTH,
      `parameters must contain ${GPU_FINITE_TIME_LYAPUNOV_EXPONENT_PARAMETER_LENGTH} float32 rows`
    );
    validatePackedView(props.output.ftle, ['float32'], `${id} output.ftle`);
    validateViewLength(id, 'output.ftle', props.output.ftle, props.seedWidth * props.seedHeight);
    validateGraphOutputsDisjointFromInputs(
      id,
      [props.output.ftle],
      [props.positions, props.parameters]
    );
    this.id = id;
    this.props = props;
  }

  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const {seedWidth, seedHeight} = props;
    validateGraphViewsBelongToGraph(id, graph, [
      props.positions,
      props.parameters,
      props.output.ftle
    ]);
    validateGPUStorageBindingByteLength(
      graph.device,
      id,
      'positions',
      seedWidth * seedHeight * 8,
      'use a smaller seed grid'
    );
    return [
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-exponent`,
        operation: OPERATION,
        invocationCount: seedWidth * seedHeight,
        bindings: [
          {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
          {name: 'parameters', view: props.parameters, type: 'f32', access: 'read'},
          {name: 'ftle', view: props.output.ftle, type: 'f32', access: 'read_write'}
        ],
        declarations: /* wgsl */ `${GET_NAN_WGSL}
const SEED_WIDTH: u32 = ${seedWidth}u;
const SEED_HEIGHT: u32 = ${seedHeight}u;
fn readPosition(column: u32, row: u32) -> vec2<f32> {
  let base = positionsOffset + 2u * (row * SEED_WIDTH + column);
  return vec2<f32>(positions[base], positions[base + 1u]);
}
fn isNanPosition(value: vec2<f32>) -> bool {
  return (bitcast<u32>(value.x) & 0x7fffffffu) > 0x7f800000u ||
    (bitcast<u32>(value.y) & 0x7fffffffu) > 0x7f800000u;
}`,
        body: /* wgsl */ `
  let column = index % SEED_WIDTH;
  let row = index / SEED_WIDTH;
  let column0 = select(column, column - 1u, column > 0u);
  let column1 = select(column, column + 1u, column + 1u < SEED_WIDTH);
  let row0 = select(row, row - 1u, row > 0u);
  let row1 = select(row, row + 1u, row + 1u < SEED_HEIGHT);
  let west = readPosition(column0, row);
  let east = readPosition(column1, row);
  let south = readPosition(column, row0);
  let north = readPosition(column, row1);
  var exponent = getNaN();
  if (!(isNanPosition(west) || isNanPosition(east) || isNanPosition(south) || isNanPosition(north))) {
    let spacingX = parameters[parametersOffset];
    let spacingY = parameters[parametersOffset + 1u];
    let time = abs(parameters[parametersOffset + 2u]);
    let gradientX = (east - west) / (f32(column1 - column0) * spacingX);
    let gradientY = (north - south) / (f32(row1 - row0) * spacingY);
    let a = dot(gradientX, gradientX);
    let b = dot(gradientX, gradientY);
    let d = dot(gradientY, gradientY);
    let largest = 0.5 * (a + d) + sqrt(0.25 * (a - d) * (a - d) + b * b);
    exponent = 0.5 * log(largest) / time;
  }
  ftle[ftleOffset + index] = exponent;`
      })
    ];
  }
}

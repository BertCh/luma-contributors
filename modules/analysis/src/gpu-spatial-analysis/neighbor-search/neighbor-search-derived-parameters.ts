// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  createWGSLKernelNode,
  getWGSLFloatLiteral
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {IS_FINITE_FLOAT_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {getBoundsReductionNodes} from '../hilbert-keys/bounds-reduction';
import {
  GPU_NEIGHBOR_SEARCH_PARAMETER_LENGTH,
  GPU_NEIGHBOR_SEARCH_TEMPORAL_PARAMETER_LENGTH
} from './neighbor-search-parameters';

/** Output of {@link createNeighborSearchDerivedParameterNodes}. @internal */
export type NeighborSearchDerivedParameters<Parameters> = {
  /** Fill and reduction nodes; they must run before the search. */
  nodes: readonly GPUCommandNode<Parameters>[];
  /** The packed parameter view to hand to `GPUNeighborSearch`. */
  parameters: GraphDataView<'float32'>;
  /** About one target per two cells, a lattice for contributors that expose no `gridSize`. */
  gridSize: [number, number];
};

function getFloatLiteralOrInfinity(value: number): string {
  // A const-expression overflow is a WGSL compile error, so infinity is made at run time.
  return Number.isFinite(value)
    ? getWGSLFloatLiteral(Math.fround(value))
    : 'bitcast<f32>(0x7f800000u | (index & 0u))';
}

/**
 * Writes `GPUNeighborSearch` parameters on the GPU for a self join by a contributor that has no
 * caller-supplied bounds: binary weights, a compile-time `radius` and, for planar positions, the
 * extent of the finite rows. Longitude/latitude positions use the whole sphere, so the search is
 * periodic across +-180. With `times`, the time window and the extent of the finite times follow
 * in the temporal layout.
 *
 * @internal
 */
export function createNeighborSearchDerivedParameterNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    operation: string;
    positions: GraphDataView<'float32x2'>;
    /** Compile-time distance limit, or `Infinity` for unbounded. */
    radius: number;
    geographic?: boolean;
    times?: GraphDataView<'float32'>;
    /** Required with `times`. */
    timeRadius?: number;
  }
): NeighborSearchDerivedParameters<Parameters> {
  const {id, operation, positions, radius, geographic, times, timeRadius} = props;
  const rowCount = positions.length;
  const parameterLength = times
    ? GPU_NEIGHBOR_SEARCH_TEMPORAL_PARAMETER_LENGTH
    : GPU_NEIGHBOR_SEARCH_PARAMETER_LENGTH;
  const parameters = createTransientView(graph, `${id}-parameters`, 'float32', parameterLength);
  const nodes: GPUCommandNode<Parameters>[] = [];
  const bounds = geographic
    ? undefined
    : createTransientView(graph, `${id}-point-bounds`, 'float32', 4);
  if (bounds) {
    nodes.push(
      ...getBoundsReductionNodes<Parameters>(graph, {
        id: `${id}-point-bounds`,
        operation,
        variant: 'neighbor-parameters-point-bounds',
        bindings: [{name: 'positions', view: positions, type: 'f32', access: 'read'}],
        declarations: `${IS_FINITE_FLOAT_WGSL}
fn readItem(row: u32) -> vec4f {
  let x = positions[positionsOffset + row * 2u];
  let y = positions[positionsOffset + row * 2u + 1u];
  return vec4f(x, y, select(0.0, 1.0, isFiniteFloat(x) && isFiniteFloat(y)), 0.0);
}`,
        itemCount: rowCount,
        output: bounds
      })
    );
  }
  const timeBounds = times
    ? createTransientView(graph, `${id}-time-bounds`, 'float32', 4)
    : undefined;
  if (times && timeBounds) {
    nodes.push(
      ...getBoundsReductionNodes<Parameters>(graph, {
        id: `${id}-time-bounds`,
        operation,
        variant: 'neighbor-parameters-time-bounds',
        bindings: [{name: 'times', view: times, type: 'f32', access: 'read'}],
        declarations: `${IS_FINITE_FLOAT_WGSL}
fn readItem(row: u32) -> vec4f {
  let time = times[timesOffset + row];
  return vec4f(time, 0.0, select(0.0, 1.0, isFiniteFloat(time)), 0.0);
}`,
        itemCount: rowCount,
        output: timeBounds
      })
    );
  }
  const boundsWGSL = bounds
    ? `for (var word = 0u; word < 4u; word++) {
    parameters[parametersOffset + word] = bounds[boundsOffset + word];
  }`
    : `parameters[parametersOffset] = -180.0;
  parameters[parametersOffset + 1u] = -90.0;
  parameters[parametersOffset + 2u] = 180.0;
  parameters[parametersOffset + 3u] = 90.0;`;
  nodes.push(
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-parameters`,
      operation,
      variant: 'neighbor-parameters',
      bindings: [
        ...(bounds
          ? [{name: 'bounds', view: bounds, type: 'f32' as const, access: 'read' as const}]
          : []),
        ...(timeBounds
          ? [{name: 'timeBounds', view: timeBounds, type: 'f32' as const, access: 'read' as const}]
          : []),
        {name: 'parameters', view: parameters, type: 'f32', access: 'read_write'}
      ],
      invocationCount: 1,
      body: `${boundsWGSL}
  parameters[parametersOffset + 4u] = ${getFloatLiteralOrInfinity(radius)};
  parameters[parametersOffset + 5u] = 0.0;
  parameters[parametersOffset + 6u] = 1.0;
  parameters[parametersOffset + 7u] = 0.0;
  parameters[parametersOffset + 8u] = 1.0;
  parameters[parametersOffset + 9u] = 0.0;
  parameters[parametersOffset + 10u] = 0.0;
  parameters[parametersOffset + 11u] = 0.0;${
    timeBounds
      ? `
  parameters[parametersOffset + 12u] = ${getFloatLiteralOrInfinity(timeRadius ?? Number.NaN)};
  parameters[parametersOffset + 13u] = timeBounds[timeBoundsOffset];
  parameters[parametersOffset + 14u] = timeBounds[timeBoundsOffset + 2u];`
      : ''
  }`
    })
  );
  const columns = Math.max(1, Math.ceil(Math.sqrt(rowCount / 2)));
  return {nodes, parameters, gridSize: [columns, columns]};
}

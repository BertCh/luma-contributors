// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  getGraphOutputViews,
  validateCommandNodeBudget,
  validateFiniteNumber,
  validateRequirement
} from '../analysis-core/kernels/gpu-contributor-utils';
import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import type {GPURasterBufferBand, GPURasterMetadata} from '@luma.gl/graph/gpu-raster';
import {getRasterBufferBandNodes, getRasterBandViews} from './raster-band-utils';
import {
  createWGSLKernelNode,
  getWGSLFloatLiteral
} from '../analysis-core/kernels/wgsl-kernel-nodes';
import {IS_FINITE_VALUE_WGSL} from '../analysis-core/kernels/wgsl-common';
import {
  RASTER_BAND_CANONICALIZATION_NODE_COUNT,
  createRasterJacobiLoopNodes,
  getRasterJacobiLoopNodeCount,
  requireRasterLength,
  validateRasterExpansionViews,
  validateRasterExpansionGraph,
  type RasterIterationStatusOutputs
} from './raster-expansion-utils';

/** Default number of relaxation rounds, each moving the cost front by one cell. */
const DEFAULT_MAXIMUM_ITERATIONS = 128;

export type GPURasterWalkingCostProps = {
  id?: string;
  grid: GPURasterMetadata;
  elevation: GPURasterBufferBand;
  friction: GPURasterBufferBand;
  sources: GraphDataView<'uint32'>;
  /** Coefficient multiplying additive average-friction * ground-distance cost. Default 1. */
  frictionWeight?: number;
  model?:
    | {kind: 'tobler'}
    | {
        kind: 'grass' /** a*distance + b*uphill + c*moderate-downhill + d*steep-downhill. */;
        coefficients: readonly [number, number, number, number];
        downhillSlopeThreshold: number;
      };
  /** Relaxation rounds before the run stops unconverged. Default 128; at most 26,196 (the command node budget). */
  maximumIterations?: number;
  output: {
    costs: GraphDataView<'float32'>;
    validity: GraphDataView<'uint32'>;
    /**
     * `converged` is 1 when a round changed nothing before the iteration limit. The optional
     * `iterationLimitReached` (1 when the limit stopped an unconverged run) and `iterationCount`
     * (rounds that ran, including the final verifying round) are published when present.
     */
    status: RasterIterationStatusOutputs;
  };
};

/**
 * Directed 8-connected least-cost movement. Elevations and affine distances must use the same
 * linear unit (meters for Tobler seconds). Uphill and downhill transitions are evaluated separately.
 * Tobler uses 6*exp(-3.5*abs(slope+0.05)) km/hour. GRASS-style energy coefficients are explicit;
 * negative/nonfinite transition costs are impassable. Whole-grid support is required.
 *
 * Jacobi relaxation runs at most `maximumIterations + 1` rounds. Later rounds are gated on the GPU
 * and dispatch nothing once a round changes no cell. The current costs are always published.
 * `status.converged` is 1 only when a round changed nothing; otherwise cells that the front has
 * not reached yet hold `+Infinity` and the other costs are upper bounds.
 */
export class GPURasterWalkingCost implements GPUCommandNodeProducer {
  readonly id: string;
  readonly props: GPURasterWalkingCostProps;
  /** Command nodes `getCommandNodes` returns, at most; checked against the node budget. */
  readonly estimatedCommandNodeCount: number;
  constructor(props: GPURasterWalkingCostProps) {
    this.id = props.id ?? 'raster-walking-cost';
    this.props = props;
    const {id} = this;
    const count = validateRasterExpansionViews(
      id,
      props.grid,
      [
        ...getRasterBandViews(props.elevation),
        ...getRasterBandViews(props.friction),
        props.sources
      ],
      getGraphOutputViews(props.output)
    );
    requireRasterLength(props.sources, count, id, 'sources');
    requireRasterLength(props.output.costs, count, id, 'output.costs');
    requireRasterLength(props.output.validity, count, id, 'output.validity');
    requireRasterLength(props.output.status.converged, 1, id, 'output.status.converged');
    requireRasterLength(
      props.output.status.iterationLimitReached,
      1,
      id,
      'output.status.iterationLimitReached'
    );
    requireRasterLength(props.output.status.iterationCount, 1, id, 'output.status.iterationCount');
    const maximumIterations = props.maximumIterations ?? DEFAULT_MAXIMUM_ITERATIONS;
    validateRequirement(
      id,
      Number.isInteger(maximumIterations) && maximumIterations > 0,
      'maximumIterations must be a positive integer'
    );
    validateFiniteNumber(id, 'frictionWeight', props.frictionWeight ?? 1, 0);
    if (props.model?.kind === 'grass') {
      validateRequirement(
        id,
        props.model.coefficients.every(Number.isFinite),
        'model.coefficients must be four finite numbers'
      );
      validateRequirement(
        id,
        Number.isFinite(props.model.downhillSlopeThreshold) &&
          props.model.downhillSlopeThreshold < 0,
        'model.downhillSlopeThreshold must be a negative finite number'
      );
    }
    this.estimatedCommandNodeCount =
      2 * RASTER_BAND_CANONICALIZATION_NODE_COUNT +
      2 +
      getRasterJacobiLoopNodeCount(maximumIterations + 1);
    validateCommandNodeBudget(id, this.estimatedCommandNodeCount);
  }
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const {grid, output} = props;
    const count = grid.width * grid.height;
    validateRasterExpansionGraph(id, graph, [
      ...getRasterBandViews(props.elevation),
      ...getRasterBandViews(props.friction),
      props.sources,
      ...getGraphOutputViews(output)
    ]);
    const elevation = getRasterBufferBandNodes(
      graph,
      `${id}-elevation`,
      props.elevation,
      grid,
      true
    );
    const friction = getRasterBufferBandNodes(graph, `${id}-friction`, props.friction, grid, true);
    const surface = createTransientView(graph, `${id}-surface`, 'float32x2', count);
    const first = createTransientView(graph, `${id}-first`, 'float32', count);
    const second = createTransientView(graph, `${id}-second`, 'float32', count);
    const nodes: GPUCommandNode<Parameters>[] = [
      ...elevation.nodes,
      ...friction.nodes,
      createWGSLKernelNode(graph, {
        id: `${id}-prepare`,
        operation: 'GPURasterWalkingCost',
        invocationCount: count,
        bindings: [
          {name: 'elevation', view: elevation.band.storage.values, type: 'f32', access: 'read'},
          {name: 'elevationValidity', view: elevation.band.validity!, type: 'u32', access: 'read'},
          {name: 'friction', view: friction.band.storage.values, type: 'f32', access: 'read'},
          {name: 'frictionValidity', view: friction.band.validity!, type: 'u32', access: 'read'},
          {name: 'sources', view: props.sources, type: 'u32', access: 'read'},
          {name: 'surface', view: surface, type: 'f32', access: 'read_write'},
          {name: 'first', view: first, type: 'f32', access: 'read_write'}
        ],
        body: `let valid = elevationValidity[elevationValidityOffset + index] != 0u
    && frictionValidity[frictionValidityOffset + index] != 0u
    && friction[frictionOffset + index] >= 0.0;
  // Invalid cells carry NaN elevation, so no neighbor relaxes through them.
  surface[surfaceOffset + index * 2u] = select(
    bitcast<f32>(0x7fc00000u | (index & 0u)),
    elevation[elevationOffset + index],
    valid
  );
  surface[surfaceOffset + index * 2u + 1u] = friction[frictionOffset + index];
  first[firstOffset + index] = select(
    bitcast<f32>(0x7f800000u | (index & 0u)),
    0.0,
    valid && sources[sourcesOffset + index] != 0u
  );`
      })
    ];
    const model = props.model;
    const affine = grid.affine.map(getWGSLFloatLiteral);
    const energyWGSL =
      model?.kind === 'grass'
        ? `let energy = ${getWGSLFloatLiteral(model.coefficients[0])} * groundDistance
    + select(
      select(
        ${getWGSLFloatLiteral(model.coefficients[2])} * rise,
        ${getWGSLFloatLiteral(model.coefficients[3])} * rise,
        slope < ${getWGSLFloatLiteral(model.downhillSlopeThreshold)}
      ),
      ${getWGSLFloatLiteral(model.coefficients[1])} * rise,
      rise > 0.0
    );`
        : 'let energy = groundDistance * 0.6 * exp(3.5 * abs(slope + 0.05));';
    const declarations = `${IS_FINITE_VALUE_WGSL}
const WIDTH: u32 = ${grid.width}u;
const HEIGHT: u32 = ${grid.height}u;
const FRICTION_WEIGHT: f32 = ${getWGSLFloatLiteral(props.frictionWeight ?? 1)};

// Ground length of the step to the neighbor at (columnOffset, rowOffset).
fn getStepDistance(columnOffset: i32, rowOffset: i32) -> f32 {
  let columnStep = f32(columnOffset);
  let rowStep = f32(rowOffset);
  return length(vec2<f32>(
    ${affine[0]} * columnStep + ${affine[1]} * rowStep,
    ${affine[3]} * columnStep + ${affine[4]} * rowStep
  ));
}

// Cost of stepping from the neighbor to this cell: movement energy plus averaged friction.
fn getTransitionCost(
  elevation: f32,
  friction: f32,
  neighborElevation: f32,
  neighborFriction: f32,
  groundDistance: f32
) -> f32 {
  let rise = elevation - neighborElevation;
  let slope = rise / groundDistance;
  ${energyWGSL}
  return energy + FRICTION_WEIGHT * 0.5 * (friction + neighborFriction) * groundDistance;
}`;
    let previous = first;
    let next = second;
    // One extra round tests whether the final iterate is a true fixed point.
    nodes.push(
      ...createRasterJacobiLoopNodes(graph, {
        id: `${id}-iterate`,
        operation: 'GPURasterWalkingCost',
        invocationCount: count,
        roundCount: (props.maximumIterations ?? DEFAULT_MAXIMUM_ITERATIONS) + 1,
        status: output.status,
        createRound: (
          {nodeId, slotBinding, markChangedWGSL, condition, extraResources},
          status
        ) => {
          const round = createWGSLKernelNode(graph, {
            id: nodeId,
            operation: 'GPURasterWalkingCost',
            variant: 'relaxation-round',
            invocationCount: count,
            bindings: [
              {name: 'surface', view: surface, type: 'f32', access: 'read'},
              {name: 'previous', view: previous, type: 'f32', access: 'read'},
              {name: 'next', view: next, type: 'f32', access: 'read_write'},
              {name: 'status', view: status, type: 'atomic<u32>', access: 'read_write'},
              slotBinding
            ],
            declarations,
            body: `let elevation = surface[surfaceOffset + index * 2u];
  let friction = surface[surfaceOffset + index * 2u + 1u];
  let original = previous[previousOffset + index];
  var best = original;
  if (isFiniteValue(elevation)) {
    let column = i32(index % WIDTH);
    let row = i32(index / WIDTH);
    for (var rowOffset = -1; rowOffset <= 1; rowOffset++) {
      for (var columnOffset = -1; columnOffset <= 1; columnOffset++) {
        if (columnOffset == 0 && rowOffset == 0) { continue; }
        let neighborColumn = column + columnOffset;
        let neighborRow = row + rowOffset;
        if (neighborColumn < 0 || neighborRow < 0 || neighborColumn >= i32(WIDTH) || neighborRow >= i32(HEIGHT)) { continue; }
        let neighbor = u32(neighborRow) * WIDTH + u32(neighborColumn);
        let neighborElevation = surface[surfaceOffset + neighbor * 2u];
        if (!isFiniteValue(neighborElevation)) { continue; }
        let groundDistance = getStepDistance(columnOffset, rowOffset);
        if (groundDistance <= 0.0) { continue; }
        let cost = getTransitionCost(
          elevation,
          friction,
          neighborElevation,
          surface[surfaceOffset + neighbor * 2u + 1u],
          groundDistance
        );
        if (cost >= 0.0 && isFiniteValue(cost)) {
          best = min(best, previous[previousOffset + neighbor] + cost);
        }
      }
    }
  }
  next[nextOffset + index] = best;
  if (best < original) { ${markChangedWGSL} }`,
            condition,
            extraResources
          });
          [previous, next] = [next, previous];
          return round;
        }
      }),
      createWGSLKernelNode(graph, {
        id: `${id}-publish`,
        operation: 'GPURasterWalkingCost',
        invocationCount: count,
        bindings: [
          {name: 'previous', view: previous, type: 'f32', access: 'read'},
          {name: 'surface', view: surface, type: 'f32', access: 'read'},
          {name: 'costs', view: output.costs, type: 'f32', access: 'read_write'},
          {name: 'validity', view: output.validity, type: 'u32', access: 'read_write'}
        ],
        declarations: IS_FINITE_VALUE_WGSL,
        body: `// The current iterate is always published; status.converged says whether it is final.
  let valid = isFiniteValue(surface[surfaceOffset + index * 2u]);
  costs[costsOffset + index] = select(
    bitcast<f32>(0x7fc00000u | (index & 0u)),
    previous[previousOffset + index],
    valid
  );
  validity[validityOffset + index] = u32(valid);`
      })
    );
    return nodes;
  }
}

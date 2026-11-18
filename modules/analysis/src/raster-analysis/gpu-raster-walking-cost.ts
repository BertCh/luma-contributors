// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  validateRasterStorageBindings,
  getRasterIterationSlotByteLength
} from './raster-grid/raster-storage-preflight';
import {
  getGraphOutputViews,
  validateCommandNodeBudget,
  validateFiniteNumber,
  validateRequirement,
  getGraphOutputViews as getRasterPreflightOutputViews
} from '../analysis-core/kernels/gpu-contributor-utils';
import {validateGPUStatusPort, type GPUInvalidStatusPort} from '../analysis-core/contracts/status';
import {
  getRasterGridGeoreference,
  getRasterGroundScaleWGSL,
  validateRasterCellSizeMode,
  type GPURasterCellSizeMode
} from '../analysis-core/contracts/raster-grid';
import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import type {GPURasterBufferBand, GPURasterMetadata} from '@luma.gl/graph/gpu-raster';
import {
  getRasterBufferBandNodes,
  getRasterBandViews,
  validateParameterTarget,
  validateRasterGridAxisAligned,
  validateRasterParameters
} from './raster-band-utils';
import {createWGSLKernelNode} from '../analysis-core/kernels/wgsl-kernel-nodes';
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

/** Number of float32 values read from {@link GPURasterWalkingCostProps.parameters}. */
export const GPU_RASTER_WALKING_COST_PARAMETER_LENGTH = 16;

/** Movement model of {@link GPURasterWalkingCost}. The kind is compile-time; numbers are per-frame. */
export type GPURasterWalkingCostModel =
  | {kind: 'tobler'}
  | {
      kind: 'grass' /** a*distance + b*uphill + c*moderate-downhill + d*steep-downhill. */;
      coefficients: readonly [number, number, number, number];
      downhillSlopeThreshold: number;
    };

/** Per-frame settings packed by {@link getGPURasterWalkingCostParameterValues}. */
export type GPURasterWalkingCostSettings = {
  /** Coefficient multiplying additive average-friction * ground-distance cost. Default 1. */
  frictionWeight?: number;
  /** Movement model; only the GRASS coefficients and threshold are packed. Default Tobler. */
  model?: GPURasterWalkingCostModel;
};

/**
 * Packs the per-frame parameters of {@link GPURasterWalkingCost}: the grid's linear affine terms,
 * its row edges (for `cellSizeMode`), the friction weight and the GRASS coefficients.
 *
 * Layout (float32): `[a, b, d, e, northEdge, southEdge, frictionWeight, 0, c0, c1, c2, c3,
 * downhillSlopeThreshold, 0, 0, 0]`, where `a, b, d, e` are `grid.affine[0, 1, 3, 4]` and the edges
 * are the world y of the top of row 0 and the bottom of the last row. Slots 8 to 12 also identify
 * the model: all zero for Tobler, a negative threshold for GRASS. The values must be packed for the
 * same `model.kind` the contributor was compiled with; the kernel checks that on the GPU and
 * reports a mismatch in `status.invalidCount` (see {@link GPURasterWalkingCost}). Write the result into a
 * `GPUParameterBuffer` between encodings to change the georeference, friction weight or GRASS
 * coefficients without recompiling. Only differences of the affine are used, so the origin needs
 * no float64 subtraction; the edges are used only by the `'web-mercator'` and `'geographic'`
 * cell size modes.
 *
 * @param settings Per-frame settings. A Tobler model packs zero coefficients.
 * @param grid Grid whose georeference is packed. Its width and height must match the compiled one.
 * @param target Optional destination of at least 16 values. A new array is returned when omitted.
 * @throws If a setting is out of range or `target` is too short.
 */
export function getGPURasterWalkingCostParameterValues(
  settings: GPURasterWalkingCostSettings,
  grid: GPURasterMetadata,
  target: Float32Array = new Float32Array(GPU_RASTER_WALKING_COST_PARAMETER_LENGTH)
): Float32Array {
  const id = 'getGPURasterWalkingCostParameterValues';
  validateParameterTarget(id, target, GPU_RASTER_WALKING_COST_PARAMETER_LENGTH);
  validateWalkingCostSettings(id, settings);
  const [a, b, , d, e] = grid.affine;
  const {northEdge, southEdge} = getRasterGridGeoreference(grid);
  target.fill(0, 0, GPU_RASTER_WALKING_COST_PARAMETER_LENGTH);
  target.set([a, b, d, e, northEdge, southEdge, settings.frictionWeight ?? 1], 0);
  if (settings.model?.kind === 'grass') {
    target.set([...settings.model.coefficients, settings.model.downhillSlopeThreshold], 8);
  }
  return target;
}

/** Validates the per-frame settings shared by the props and the packer. */
function validateWalkingCostSettings(id: string, settings: GPURasterWalkingCostSettings): void {
  validateFiniteNumber(id, 'frictionWeight', settings.frictionWeight ?? 1, 0);
  if (settings.model?.kind === 'grass') {
    validateRequirement(
      id,
      settings.model.coefficients.length === 4 &&
        settings.model.coefficients.every(Number.isFinite),
      'model.coefficients must be four finite numbers'
    );
    validateRequirement(
      id,
      Number.isFinite(settings.model.downhillSlopeThreshold) &&
        settings.model.downhillSlopeThreshold < 0,
      'model.downhillSlopeThreshold must be a negative finite number'
    );
  }
}

export type GPURasterWalkingCostProps = {
  id?: string;
  grid: GPURasterMetadata;
  elevation: GPURasterBufferBand;
  friction: GPURasterBufferBand;
  sources: GraphDataView<'uint32'>;
  /**
   * Per-frame parameters with at least {@link GPU_RASTER_WALKING_COST_PARAMETER_LENGTH} float32
   * values, written with {@link getGPURasterWalkingCostParameterValues}. The affine, friction
   * weight and GRASS coefficients are read from it (the numbers in `model` are ignored; `model.kind`
   * still selects the kernel), so a changed georeference does not recompile. Pack it for the same
   * `model.kind`: values packed for the other model are detected on the GPU, reported in
   * `output.status.invalidCount` and move nothing.
   */
  parameters: GraphDataView<'float32'>;
  /**
   * How affine units become ground meters for step lengths and slopes. Default `'uniform'` (affine
   * units are meters). `'geographic'` (degrees) and `'web-mercator'` (EPSG:3857 meters) scale each
   * step on the WGS84 ellipsoid at the latitude between the two cell centres, and need an
   * axis-aligned grid. Compile-time.
   */
  cellSizeMode?: GPURasterCellSizeMode;
  /** Movement model. Default Tobler. The kind is compile-time; the numbers are packed into `parameters`. */
  model?: GPURasterWalkingCostModel;
  /** Relaxation rounds before the run stops unconverged. Default 128; at most 26,196 (the command node budget). */
  maximumIterations?: number;
  output: {
    costs: GraphDataView<'float32'>;
    validity: GraphDataView<'uint32'>;
    /**
     * `converged` is 1 when a round changed nothing before the iteration limit. The optional
     * `iterationLimitReached` (1 when the limit stopped an unconverged run) and `iterationCount`
     * (rounds that ran, including the final verifying round) are published when present. The
     * optional `invalidCount` is 1 when `parameters` were not packed for the compiled `model.kind`
     * (slots 8 to 12 not all zero for Tobler; a threshold that is not negative, or a non-finite
     * coefficient, for GRASS), else 0. Such a run moves nothing: sources cost 0 and every other
     * cell stays `+Infinity`, whether or not this view is given.
     */
    status: RasterIterationStatusOutputs & GPUInvalidStatusPort;
  };
};

/**
 * Directed 8-connected least-cost movement. Elevations and ground step lengths must use the same
 * linear unit (meters for Tobler seconds): with the default `cellSizeMode: 'uniform'` the affine
 * is already in that unit, and `'geographic'` or `'web-mercator'` grids are converted to ground
 * meters per row. Uphill and downhill transitions are evaluated separately.
 * Tobler uses 6*exp(-3.5*abs(slope+0.05)) km/hour. GRASS-style energy coefficients are explicit;
 * negative/nonfinite transition costs are impassable. Whole-grid support is required.
 *
 * Jacobi relaxation runs at most `maximumIterations + 1` rounds. Later rounds are gated on the GPU
 * and dispatch nothing once a round changes no cell. The current costs are always published.
 * `status.converged` is 1 only when a round changed nothing; otherwise cells that the front has
 * not reached yet hold `+Infinity` and the other costs are upper bounds.
 *
 * No-data follows the published band contract: a cell is impassable where either band's validity
 * is 0, its raw value equals the band's `noDataValue`, its value is not finite, or its friction is
 * negative. Output `validity` is 1 exactly where the cell was passable.
 *
 * Per-frame (no recompile): the contents of `parameters` (affine linear terms and row edges,
 * `frictionWeight`, GRASS coefficients and threshold), the band values and `sources`.
 * Compile-time: `grid.width`, `grid.height`, `cellSizeMode`, `model.kind` and `maximumIterations`.
 * Because `model.kind` is compiled in while the numbers arrive per frame, parameters packed for the
 * other model are detected on the GPU: nothing moves and `status.invalidCount` is 1.
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
        props.sources,
        props.parameters
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
    requireRasterLength(props.output.status.invalidCount, 1, id, 'output.status.invalidCount');
    validateGPUStatusPort(id, props.output.status);
    validateRasterCellSizeMode(id, props.cellSizeMode);
    validateRasterGridAxisAligned(id, props.grid, props.cellSizeMode);
    validateRasterParameters(id, props.parameters, GPU_RASTER_WALKING_COST_PARAMETER_LENGTH);
    const maximumIterations = props.maximumIterations ?? DEFAULT_MAXIMUM_ITERATIONS;
    validateRequirement(
      id,
      Number.isInteger(maximumIterations) && maximumIterations > 0,
      'maximumIterations must be a positive integer'
    );
    validateWalkingCostSettings(id, {model: props.model});
    this.estimatedCommandNodeCount =
      2 * RASTER_BAND_CANONICALIZATION_NODE_COUNT +
      2 +
      (props.output.status.invalidCount ? 1 : 0) +
      getRasterJacobiLoopNodeCount(maximumIterations + 1);
    validateCommandNodeBudget(id, this.estimatedCommandNodeCount);
  }
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const storageProps = this.props;
    const storageCellCount = storageProps.grid.width * storageProps.grid.height;
    validateRasterStorageBindings(
      graph,
      this.id,
      [
        ...getRasterBandViews(storageProps.elevation),
        ...getRasterBandViews(storageProps.friction),
        storageProps.sources,
        storageProps.parameters,
        ...getRasterPreflightOutputViews(storageProps.output)
      ],
      {
        'walking surface': storageCellCount * 8,
        'canonical band and cost planes': storageCellCount * 4,
        'iteration state': 16,
        'iteration gates': getRasterIterationSlotByteLength(
          (storageProps.maximumIterations ?? DEFAULT_MAXIMUM_ITERATIONS) + 1
        )
      }
    );

    const {id, props} = this;
    const {grid, output} = props;
    const count = grid.width * grid.height;
    validateRasterExpansionGraph(id, graph, [
      ...getRasterBandViews(props.elevation),
      ...getRasterBandViews(props.friction),
      props.sources,
      props.parameters,
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
    const settings = props.parameters;
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
    const cellSizeMode = props.cellSizeMode ?? 'uniform';
    const setting = (slot: number): string => `settings[settingsOffset + ${slot}u]`;
    const energyWGSL =
      props.model?.kind === 'grass'
        ? `let energy = ${setting(8)} * groundDistance
    + select(
      select(${setting(10)} * rise, ${setting(11)} * rise, slope < ${setting(12)}),
      ${setting(9)} * rise,
      rise > 0.0
    );`
        : 'let energy = groundDistance * 0.6 * exp(3.5 * abs(slope + 0.05));';
    // Slots 8 to 12 identify the model the parameters were packed for: Tobler packs zeros, GRASS
    // a negative threshold (validated by the packer). The CPU cannot see the buffer, so the kernel
    // checks it; a mismatch relaxes nothing and is reported in status.invalidCount.
    const parametersMatchModelWGSL =
      props.model?.kind === 'grass'
        ? `fn parametersMatchModel() -> bool {
  return ${setting(12)} < 0.0 && isFiniteValue(${setting(8)}) && isFiniteValue(${setting(9)}) &&
    isFiniteValue(${setting(10)}) && isFiniteValue(${setting(11)}) && isFiniteValue(${setting(12)});
}`
        : `fn parametersMatchModel() -> bool {
  return ${setting(8)} == 0.0 && ${setting(9)} == 0.0 && ${setting(10)} == 0.0 &&
    ${setting(11)} == 0.0 && ${setting(12)} == 0.0;
}`;
    if (output.status.invalidCount) {
      nodes.push(
        createWGSLKernelNode(graph, {
          id: `${id}-check-parameters`,
          operation: 'GPURasterWalkingCost',
          variant: 'check-parameters',
          invocationCount: 1,
          bindings: [
            {name: 'settings', view: settings, type: 'f32', access: 'read'},
            {
              name: 'invalidCount',
              view: output.status.invalidCount,
              type: 'u32',
              access: 'read_write'
            }
          ],
          declarations: `${IS_FINITE_VALUE_WGSL}
${parametersMatchModelWGSL}`,
          body: 'invalidCount[invalidCountOffset] = select(1u, 0u, parametersMatchModel());'
        })
      );
    }
    // Ground meters per world unit at the latitude between the two cell centres.
    const groundScaleWGSL =
      cellSizeMode === 'uniform'
        ? ''
        : `let rowFraction = (f32(row) + 0.5 + 0.5 * rowStep) / f32(HEIGHT);
  let edge = mix(${setting(4)}, ${setting(5)}, rowFraction);
  ${getRasterGroundScaleWGSL(cellSizeMode)}
  worldStep = worldStep * groundScale;`;
    const declarations = `${IS_FINITE_VALUE_WGSL}
const WIDTH: u32 = ${grid.width}u;
const HEIGHT: u32 = ${grid.height}u;
${parametersMatchModelWGSL}

// Ground length of the step from a cell in this row to the neighbor at (columnOffset, rowOffset).
fn getStepDistance(row: i32, columnOffset: i32, rowOffset: i32) -> f32 {
  let columnStep = f32(columnOffset);
  let rowStep = f32(rowOffset);
  var worldStep = vec2<f32>(
    ${setting(0)} * columnStep + ${setting(1)} * rowStep,
    ${setting(2)} * columnStep + ${setting(3)} * rowStep
  );
  ${groundScaleWGSL}
  return length(worldStep);
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
  return energy + ${setting(6)} * 0.5 * (friction + neighborFriction) * groundDistance;
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
              {name: 'settings', view: settings, type: 'f32', access: 'read'},
              slotBinding
            ],
            declarations,
            body: `let elevation = surface[surfaceOffset + index * 2u];
  let friction = surface[surfaceOffset + index * 2u + 1u];
  let original = previous[previousOffset + index];
  var best = original;
  if (isFiniteValue(elevation) && parametersMatchModel()) {
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
        let groundDistance = getStepDistance(row, columnOffset, rowOffset);
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

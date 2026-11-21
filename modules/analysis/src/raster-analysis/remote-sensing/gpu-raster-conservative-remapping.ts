// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GraphDataView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer
} from '@luma.gl/graph/gpu-core';
import {
  createWGSLKernelNode,
  getWGSLFloatLiteral
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  validateGraphPorts,
  validateCapacity,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  RASTER_FINITE_WGSL,
  rasterBinding,
  validateRasterViews,
  validateRasterGraph
} from './raster-kernel-utils';

import {validateRasterStorageBindings} from './raster-binding-validation';

/** Source cells each destination cell may overlap: one invocation loops over them and the prepared-weight stride stores them, so the loop and the storage stay small. */
export const MAXIMUM_CONSERVATIVE_CANDIDATE_CELLS = 256;

/** Largest denominator tried when looking for a rational spacing ratio and origin offset. */
const MAXIMUM_COMMENSURATE_DENOMINATOR = 4096;
/** Units of the binary fixed-point fraction used for grids that are not commensurate. */
const FIXED_POINT_UNITS = 2 ** 32;
/** Signed 32-bit bound for the cell and unit positions formed on the GPU. */
const MAXIMUM_SIGNED_POSITION = 2 ** 31 - 1;

/**
 * How one axis places destination cell edges, in source cells, without float32 coordinates.
 *
 * `rational`: the spacing ratio is `numerator / denominator` and the origin offset is
 * `offsetUnits / denominator` source cells, so every edge is an integer number of units and
 * overlaps are exact. `fixed`: whole cells plus a `2^-32` fraction; an edge is within
 * `destination width * 2^-33` source cells of its binary64 position, and overlaps of at most
 * `minimumOverlap` source cells are dropped.
 */
type ConservativeAxisPlan =
  | {kind: 'rational'; denominator: number; numerator: number; offsetUnits: number}
  | {
      kind: 'fixed';
      ratioWhole: number;
      ratioFraction: number;
      offsetWhole: number;
      offsetFraction: number;
      minimumOverlap: number;
    };

/** Splits `value` into a floor and a `2^-32` fraction in `[0, 2^32)`. */
function getFixedPointParts(value: number): {whole: number; fraction: number} {
  let whole = Math.floor(value);
  let fraction = Math.round((value - whole) * FIXED_POINT_UNITS);
  if (fraction >= FIXED_POINT_UNITS) {
    whole += 1;
    fraction = 0;
  }
  return {whole, fraction};
}

/** Chooses the exact rational plan when the grids are commensurate on this axis. */
function getConservativeAxisPlan(
  id: string,
  axisName: string,
  sourceOrigin: number,
  sourceSpacing: number,
  destinationOrigin: number,
  destinationSpacing: number,
  destinationCellCount: number
): ConservativeAxisPlan {
  const ratio = destinationSpacing / sourceSpacing;
  const offset = (destinationOrigin - sourceOrigin) / sourceSpacing;
  // Rounding of the binary64 origin difference, in source cells.
  const offsetTolerance =
    (4 * Number.EPSILON * (Math.abs(destinationOrigin) + Math.abs(sourceOrigin))) / sourceSpacing +
    1e-12;
  for (let denominator = 1; denominator <= MAXIMUM_COMMENSURATE_DENOMINATOR; denominator++) {
    const scaledRatio = ratio * denominator;
    const numerator = Math.round(scaledRatio);
    const scaledOffset = offset * denominator;
    const offsetUnits = Math.round(scaledOffset);
    if (
      numerator >= 1 &&
      Math.abs(scaledRatio - numerator) <= 1e-12 * scaledRatio &&
      Math.abs(scaledOffset - offsetUnits) <=
        denominator * offsetTolerance + 1e-12 * Math.abs(scaledOffset)
    ) {
      if (
        Math.abs(offsetUnits) + (destinationCellCount + 1) * numerator <=
        MAXIMUM_SIGNED_POSITION
      ) {
        return {kind: 'rational', denominator, numerator, offsetUnits: offsetUnits + 0};
      }
      break;
    }
  }
  const ratioParts = getFixedPointParts(ratio);
  const offsetParts = getFixedPointParts(offset);
  validateRequirement(
    id,
    Math.abs(offsetParts.whole) + (destinationCellCount + 1) * (ratioParts.whole + 1) <=
      MAXIMUM_SIGNED_POSITION,
    `the destination ${axisName} extent, measured in source cells from the source origin, must stay below 2^31`
  );
  return {
    kind: 'fixed',
    ratioWhole: ratioParts.whole,
    ratioFraction: ratioParts.fraction,
    offsetWhole: offsetParts.whole,
    offsetFraction: offsetParts.fraction,
    // Four times the largest edge error, and at least 1e-6 of the smaller of the two cells.
    minimumOverlap: Math.max(1e-6 * Math.min(1, ratio), destinationCellCount * 2 ** -31)
  };
}

/**
 * WGSL statements that define, for destination cell `${indexName}` on one axis, the lower and
 * upper edge in source cells as a whole cell (`i32`) and a fraction in `[0, 1]` (`f32`).
 */
function getConservativeAxisEdgesWGSL(
  prefix: string,
  indexName: string,
  plan: ConservativeAxisPlan
): string {
  if (plan.kind === 'rational') {
    return `let ${prefix}LowerUnits = ${plan.offsetUnits} + i32(${indexName}) * ${plan.numerator};
let ${prefix}UpperUnits = ${prefix}LowerUnits + ${plan.numerator};
let ${prefix}LowerWhole = getFloorQuotient(${prefix}LowerUnits, ${plan.denominator});
let ${prefix}UpperWhole = getFloorQuotient(${prefix}UpperUnits, ${plan.denominator});
let ${prefix}LowerFraction = f32(${prefix}LowerUnits - ${prefix}LowerWhole * ${plan.denominator}) / ${plan.denominator}.0;
let ${prefix}UpperFraction = f32(${prefix}UpperUnits - ${prefix}UpperWhole * ${plan.denominator}) / ${plan.denominator}.0;
let ${prefix}MinimumOverlap = 0.0;`;
  }
  return `let ${prefix}Product = getUnsignedProduct(${indexName}, ${plan.ratioFraction}u);
let ${prefix}LowerUnits = ${prefix}Product.y + ${plan.offsetFraction}u;
let ${prefix}LowerCarry = select(0u, 1u, ${prefix}LowerUnits < ${prefix}Product.y);
let ${prefix}LowerWhole = ${plan.offsetWhole} + i32(${indexName}) * ${plan.ratioWhole} + i32(${prefix}Product.x + ${prefix}LowerCarry);
let ${prefix}UpperUnits = ${prefix}LowerUnits + ${plan.ratioFraction}u;
let ${prefix}UpperCarry = select(0, 1, ${prefix}UpperUnits < ${prefix}LowerUnits);
let ${prefix}UpperWhole = ${prefix}LowerWhole + ${plan.ratioWhole} + ${prefix}UpperCarry;
let ${prefix}LowerFraction = f32(${prefix}LowerUnits) * ${1 / FIXED_POINT_UNITS};
let ${prefix}UpperFraction = f32(${prefix}UpperUnits) * ${1 / FIXED_POINT_UNITS};
let ${prefix}MinimumOverlap = ${getWGSLFloatLiteral(plan.minimumOverlap)};`;
}

const CONSERVATIVE_OVERLAP_WGSL = /* wgsl */ `
// Floor of numerator / denominator for a positive denominator (WGSL division truncates).
fn getFloorQuotient(numerator: i32, denominator: i32) -> i32 {
  let quotient = numerator / denominator;
  return select(quotient, quotient - 1, numerator - quotient * denominator < 0);
}
// Exact 64-bit product of two u32 values as (high, low).
fn getUnsignedProduct(left: u32, right: u32) -> vec2<u32> {
  let leftLow = left & 0xffffu;
  let leftHigh = left >> 16u;
  let rightLow = right & 0xffffu;
  let rightHigh = right >> 16u;
  let lowLow = leftLow * rightLow;
  let lowHigh = leftLow * rightHigh;
  let highLow = leftHigh * rightLow;
  let highHigh = leftHigh * rightHigh;
  let middle = (lowLow >> 16u) + (lowHigh & 0xffffu) + (highLow & 0xffffu);
  let low = (lowLow & 0xffffu) | (middle << 16u);
  let high = highHigh + (lowHigh >> 16u) + (highLow >> 16u) + (middle >> 16u);
  return vec2<u32>(high, low);
}
// Length, in source cells, of the part of [lower, upper] inside the unit cell [0, 1]. Edges are
// given relative to the candidate cell as whole cells plus a fraction in [0, 1], so an edge that
// coincides with a cell boundary is exactly 0 or 1 and a cell that is only touched gets 0.
fn getUnitCellOverlap(lowerWhole: i32, lowerFraction: f32, upperWhole: i32, upperFraction: f32, minimumOverlap: f32) -> f32 {
  let lower = max(f32(lowerWhole) + lowerFraction, 0.0);
  let upper = min(f32(upperWhole) + upperFraction, 1.0);
  let overlap = upper - lower;
  return select(0.0, overlap, overlap > minimumOverlap);
}`;

/** Axis-aligned Cartesian cells; origin is the lower-left corner and spacing is positive. */
export type GPURasterRectilinearGrid = {
  width: number;
  height: number;
  origin: readonly [number, number];
  spacing: readonly [number, number];
};
export type GPURasterConservativeRemappingProps = {
  id?: string;
  source: GPURasterRectilinearGrid;
  destination: GPURasterRectilinearGrid;
  values: GraphDataView<'float32'>;
  mask?: GraphDataView<'uint32'>;
  /** Caller-owned persistent overlap storage, sized destination cells * candidate stride. */
  preparedWeights?: {
    indices: GraphDataView<'uint32'>;
    areas: GraphDataView<'float32'>;
  };
  /** prepare writes supplied storage (default); reuse reads weights from a prior submitted preparation. */
  weightMode?: 'prepare' | 'reuse';
  /** Destination area includes uncovered area; covered area renormalizes valid overlaps. */
  normalization: 'destination-area' | 'covered-area' | 'extensive';
  output: {
    values: GraphDataView<'float32'>;
    coveredArea: GraphDataView<'float32'>;
    status: GraphDataView<'uint32'>;
  };
};

/** Rectangle overlap integration, bounded to 256 candidate source cells per destination.
 *
 * Cell edges are placed from integers, never from float32 coordinates. On an axis where the
 * spacing ratio and the origin offset (in source cells) are rational with a denominator of at
 * most 4096, edges are exact, so a destination cell gets no weight from a source cell it only
 * touches, at any distance from the origin (nested grids, `0.1` into `0.3`, half-cell shifts).
 * Otherwise edges are whole cells plus a `2^-32` fraction, within `destination width * 2^-33`
 * source cells of their binary64 position, and an overlap of at most
 * `max(1e-6 of the smaller cell, destination width * 2^-31)` source cells is dropped. Overlap
 * areas and the accumulated value are float32.
 *
 * Compact fixed-stride weights can be retained in explicit caller-owned storage; observation
 * masks are applied during accumulation. Extensive values are source-cell totals. Status 1
 * means no valid overlap; 2 means arithmetic overflow. No spherical or curvilinear geometry.
 */
export class GPURasterConservativeRemapping implements GPUCommandNodeProducer {
  readonly id: string;
  readonly candidateWidth: number;
  readonly candidateHeight: number;
  private readonly axisPlans: readonly [ConservativeAxisPlan, ConservativeAxisPlan];
  constructor(readonly props: GPURasterConservativeRemappingProps) {
    validateGraphPorts(props.id ?? 'raster-conservative-remapping', props, {
      values: {required: true, viewType: 'data', formats: ['float32']},
      mask: {required: false, viewType: 'data', formats: ['uint32']},
      'preparedWeights.indices': {
        required: Boolean(props.preparedWeights),
        viewType: 'data',
        formats: ['uint32']
      },
      'preparedWeights.areas': {
        required: Boolean(props.preparedWeights),
        viewType: 'data',
        formats: ['float32']
      },
      'output.values': {required: true, viewType: 'data', formats: ['float32']},
      'output.coveredArea': {required: true, viewType: 'data', formats: ['float32']},
      'output.status': {required: true, viewType: 'data', formats: ['uint32']}
    });
    this.id = props.id ?? 'raster-conservative-remapping';
    for (const [gridName, grid] of [
      ['source', props.source],
      ['destination', props.destination]
    ] as const) {
      validateCapacity(this.id, `${gridName}.width`, grid.width);
      validateCapacity(this.id, `${gridName}.height`, grid.height);
      validateCapacity(this.id, `${gridName} cell count`, grid.width * grid.height);
      validateRequirement(
        this.id,
        grid.origin.every(Number.isFinite) &&
          grid.spacing.every(
            value => Number.isFinite(Math.fround(value)) && Math.fround(value) > 0
          ),
        `${gridName}.origin must be finite and ${gridName}.spacing must be positive and finite in float32`
      );
    }
    validateRequirement(
      this.id,
      ['destination-area', 'covered-area', 'extensive'].includes(props.normalization),
      "normalization must be 'destination-area', 'covered-area' or 'extensive'"
    );
    validateRequirement(
      this.id,
      [props.source, props.destination].every(
        grid =>
          Number.isFinite(Math.fround(grid.spacing[0] * grid.spacing[1])) &&
          Math.fround(grid.spacing[0] * grid.spacing[1]) > 0
      ),
      'cell areas (spacing[0] times spacing[1]) must be positive and finite in float32'
    );
    validateRequirement(
      this.id,
      props.destination.origin.every((value, dimension) =>
        Number.isFinite(Math.fround(value - props.source.origin[dimension]))
      ),
      'destination.origin minus source.origin must be finite in float32'
    );
    this.axisPlans = [
      getConservativeAxisPlan(
        this.id,
        'x',
        props.source.origin[0],
        props.source.spacing[0],
        props.destination.origin[0],
        props.destination.spacing[0],
        props.destination.width
      ),
      getConservativeAxisPlan(
        this.id,
        'y',
        props.source.origin[1],
        props.source.spacing[1],
        props.destination.origin[1],
        props.destination.spacing[1],
        props.destination.height
      )
    ];
    this.candidateWidth = Math.ceil(props.destination.spacing[0] / props.source.spacing[0]) + 1;
    this.candidateHeight = Math.ceil(props.destination.spacing[1] / props.source.spacing[1]) + 1;
    validateRequirement(
      this.id,
      this.candidateWidth * this.candidateHeight <= MAXIMUM_CONSERVATIVE_CANDIDATE_CELLS,
      `a destination cell may overlap at most ${MAXIMUM_CONSERVATIVE_CANDIDATE_CELLS} source cells (candidate window ${this.candidateWidth} by ${this.candidateHeight}); use a finer destination or coarser source spacing ratio`
    );
    validateRequirement(
      this.id,
      props.weightMode === undefined || ['prepare', 'reuse'].includes(props.weightMode),
      "weightMode must be 'prepare' or 'reuse'"
    );
    validateRequirement(
      this.id,
      props.weightMode !== 'reuse' || props.preparedWeights !== undefined,
      "weightMode 'reuse' requires preparedWeights"
    );
    if (props.preparedWeights) {
      const capacity =
        props.destination.width *
        props.destination.height *
        this.candidateWidth *
        this.candidateHeight;
      validateRequirement(
        this.id,
        props.preparedWeights.indices.format === 'uint32' &&
          props.preparedWeights.areas.format === 'float32',
        'preparedWeights.indices must be uint32 and preparedWeights.areas float32'
      );
      validateRequirement(
        this.id,
        props.preparedWeights.indices.length === capacity &&
          props.preparedWeights.areas.length === capacity,
        `preparedWeights must hold ${capacity} entries (destination cells times candidate stride)`
      );
    }
    const sourceCount = props.source.width * props.source.height;
    const destinationCount = props.destination.width * props.destination.height;
    validateRequirement(
      this.id,
      props.values.length >= sourceCount && (!props.mask || props.mask.length >= sourceCount),
      'values and mask must hold at least one entry per source cell'
    );
    validateRequirement(
      this.id,
      Object.values(props.output).every(view => view.length >= destinationCount),
      'output views must hold at least one entry per destination cell'
    );
    validateCapacity(
      this.id,
      'destination cells times candidate stride',
      destinationCount * this.candidateWidth * this.candidateHeight
    );
    const inputs = [props.values, ...(props.mask ? [props.mask] : [])];
    const prepared = props.preparedWeights ? Object.values(props.preparedWeights) : [];
    const outputs = Object.values(props.output);
    const reuse = props.weightMode === 'reuse';
    validateRasterViews(
      this.id,
      [...inputs, ...(reuse ? prepared : [])],
      [...outputs, ...(!reuse ? prepared : [])],
      [
        ...(!reuse && prepared.length ? [{inputs: [], outputs: prepared}] : []),
        {inputs: [...inputs, ...prepared], outputs}
      ]
    );
  }
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {props, id} = this;
    const inputs = [props.values, ...(props.mask ? [props.mask] : [])];
    const preparedViews = props.preparedWeights ? Object.values(props.preparedWeights) : [];
    validateRasterGraph(
      id,
      graph,
      [...inputs, ...(props.weightMode === 'reuse' ? preparedViews : [])],
      [...Object.values(props.output), ...(props.weightMode !== 'reuse' ? preparedViews : [])]
    );
    const source = props.source;
    const destination = props.destination;
    const count = destination.width * destination.height;
    const stride = this.candidateWidth * this.candidateHeight;
    validateRasterStorageBindings(
      graph,
      id,
      [],
      props.preparedWeights ? {} : {'prepared indices and areas scratch': count * stride * 4}
    );
    const indices =
      props.preparedWeights?.indices ??
      createTransientView(graph, `${id}-indices`, 'uint32', count * stride);
    const weights =
      props.preparedWeights?.areas ??
      createTransientView(graph, `${id}-weights`, 'float32', count * stride);
    // Edges are placed in source cells from integers (see ConservativeAxisPlan), never from
    // float32 coordinates: f32(index) * f32(spacing) puts coinciding edges one float32 step
    // apart at large indices, which gave a destination cell a sliver of a cell it only touches.
    const [planX, planY] = this.axisPlans;
    const prepare =
      props.weightMode === 'reuse'
        ? undefined
        : createWGSLKernelNode<Parameters>(graph, {
            id: `${id}-prepare`,
            operation: 'GPURasterConservativeRemapping',
            invocationCount: count,
            bindings: [
              rasterBinding('indices', indices, true),
              rasterBinding('weights', weights, true)
            ],
            declarations: CONSERVATIVE_OVERLAP_WGSL,
            body: `let destinationColumn = index % ${destination.width}u;
let destinationRow = index / ${destination.width}u;
${getConservativeAxisEdgesWGSL('x', 'destinationColumn', planX)}
${getConservativeAxisEdgesWGSL('y', 'destinationRow', planY)}
let first = vec2<i32>(xLowerWhole, yLowerWhole);
for (var candidate = 0u; candidate < ${stride}u; candidate++) {
  let coordinate = first + vec2<i32>(i32(candidate % ${this.candidateWidth}u), i32(candidate / ${this.candidateWidth}u));
  let slot = index * ${stride}u + candidate;
  indices[indicesOffset + slot] = 0xffffffffu; weights[weightsOffset + slot] = 0.0;
  if (any(coordinate < vec2<i32>(0)) || any(coordinate >= vec2<i32>(${source.width}, ${source.height}))) { continue; }
  let overlapX = getUnitCellOverlap(xLowerWhole - coordinate.x, xLowerFraction, xUpperWhole - coordinate.x, xUpperFraction, xMinimumOverlap);
  let overlapY = getUnitCellOverlap(yLowerWhole - coordinate.y, yLowerFraction, yUpperWhole - coordinate.y, yUpperFraction, yMinimumOverlap);
  if (!(overlapX > 0.0 && overlapY > 0.0)) { continue; }
  indices[indicesOffset + slot] = u32(coordinate.y) * ${source.width}u + u32(coordinate.x);
  weights[weightsOffset + slot] = (overlapX * overlapY) * ${getWGSLFloatLiteral(source.spacing[0] * source.spacing[1])};
}`
          });
    const apply = createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-apply`,
      operation: 'GPURasterConservativeRemapping',
      invocationCount: count,
      bindings: [
        rasterBinding('values', props.values),
        ...(props.mask ? [rasterBinding('mask', props.mask)] : []),
        rasterBinding('indices', indices),
        rasterBinding('weights', weights),
        rasterBinding('result', props.output.values, true),
        rasterBinding('coveredArea', props.output.coveredArea, true),
        rasterBinding('status', props.output.status, true)
      ],
      declarations: RASTER_FINITE_WGSL,
      body: `var numerator = 0.0; var support = 0.0;
for (var candidate = 0u; candidate < ${stride}u; candidate++) {
  let slot = index * ${stride}u + candidate; let sourceIndex = indices[indicesOffset + slot];
  if (sourceIndex == 0xffffffffu) { continue; }
  let value = values[valuesOffset + sourceIndex];
  if (!finite(value) ${props.mask ? '|| mask[maskOffset + sourceIndex] == 0u' : ''}) { continue; }
  let area = weights[weightsOffset + slot]; numerator += value * area; support += area;
}
coveredArea[coveredAreaOffset + index] = support;
var value = missing(); var code = 1u;
if (support > 0.0) {
  value = numerator / ${props.normalization === 'covered-area' ? 'support' : getWGSLFloatLiteral(props.normalization === 'extensive' ? source.spacing[0] * source.spacing[1] : destination.spacing[0] * destination.spacing[1])};
  code = select(2u, 0u, finite(value));
}
result[resultOffset + index] = value; status[statusOffset + index] = code;`
    });
    return prepare ? [prepare, apply] : [apply];
  }
}

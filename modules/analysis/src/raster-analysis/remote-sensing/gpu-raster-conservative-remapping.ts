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
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  validateCapacity,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  RASTER_FINITE_WGSL,
  rasterBinding,
  validateRasterViews,
  validateRasterGraph
} from './raster-kernel-utils';

/** Source cells each destination cell may overlap: one invocation loops over them and the prepared-weight stride stores them, so the loop and the storage stay small. */
export const MAXIMUM_CONSERVATIVE_CANDIDATE_CELLS = 256;

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
  validity?: GraphDataView<'uint32'>;
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

/** Exact rectangle overlap integration, bounded to 256 candidate source cells per destination.
 * Compact fixed-stride weights can be retained in explicit caller-owned storage; observation
 * masks are applied during accumulation. Extensive values are source-cell totals. Status 1
 * means no valid overlap; 2 means arithmetic overflow. No spherical or curvilinear geometry.
 */
export class GPURasterConservativeRemapping implements GPUCommandNodeProducer {
  readonly id: string;
  readonly candidateWidth: number;
  readonly candidateHeight: number;
  constructor(readonly props: GPURasterConservativeRemappingProps) {
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
      props.values.length >= sourceCount &&
        (!props.validity || props.validity.length >= sourceCount),
      'values and validity must hold at least one entry per source cell'
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
    validateRasterViews(
      this.id,
      [props.values, ...(props.validity ? [props.validity] : [])],
      Object.values(props.output)
    );
  }
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {props, id} = this;
    const inputs = [props.values, ...(props.validity ? [props.validity] : [])];
    const preparedViews = props.preparedWeights ? Object.values(props.preparedWeights) : [];
    validateRasterGraph(
      id,
      graph,
      [...inputs, ...(props.weightMode === 'reuse' ? preparedViews : [])],
      [...Object.values(props.output), ...(props.weightMode !== 'reuse' ? preparedViews : [])]
    );
    validateRasterViews(id, inputs, [...Object.values(props.output), ...preparedViews]);
    const source = props.source;
    const destination = props.destination;
    const count = destination.width * destination.height;
    const stride = this.candidateWidth * this.candidateHeight;
    const indices =
      props.preparedWeights?.indices ??
      createTransientView(graph, `${id}-indices`, 'uint32', count * stride);
    const weights =
      props.preparedWeights?.areas ??
      createTransientView(graph, `${id}-weights`, 'float32', count * stride);
    // Origin differences are formed in host double precision before conversion to local float32.
    const origin = destination.origin.map((value, dimension) => value - source.origin[dimension]);
    const prepare = createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-prepare`,
      operation: 'GPURasterConservativeRemapping',
      invocationCount: count,
      bindings: [rasterBinding('indices', indices, true), rasterBinding('weights', weights, true)],
      body: `let lower = vec2<f32>(${origin[0]}, ${origin[1]}) + vec2<f32>(f32(index % ${destination.width}u), f32(index / ${destination.width}u)) * vec2<f32>(${destination.spacing[0]}, ${destination.spacing[1]});
let upper = lower + vec2<f32>(${destination.spacing[0]}, ${destination.spacing[1]});
let first = vec2<i32>(floor(lower / vec2<f32>(${source.spacing[0]}, ${source.spacing[1]})));
for (var candidate = 0u; candidate < ${stride}u; candidate++) {
  let coordinate = first + vec2<i32>(i32(candidate % ${this.candidateWidth}u), i32(candidate / ${this.candidateWidth}u));
  let slot = index * ${stride}u + candidate;
  indices[indicesOffset + slot] = 0xffffffffu; weights[weightsOffset + slot] = 0.0;
  if (any(coordinate < vec2<i32>(0)) || any(coordinate >= vec2<i32>(${source.width}, ${source.height}))) { continue; }
  let sourceLower = vec2<f32>(coordinate) * vec2<f32>(${source.spacing[0]}, ${source.spacing[1]});
  let overlap = max(vec2<f32>(0.0), min(upper, sourceLower + vec2<f32>(${source.spacing[0]}, ${source.spacing[1]})) - max(lower, sourceLower));
  indices[indicesOffset + slot] = u32(coordinate.y) * ${source.width}u + u32(coordinate.x);
  weights[weightsOffset + slot] = overlap.x * overlap.y;
}`
    });
    const apply = createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-apply`,
      operation: 'GPURasterConservativeRemapping',
      invocationCount: count,
      bindings: [
        rasterBinding('values', props.values),
        ...(props.validity ? [rasterBinding('validity', props.validity)] : []),
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
  if (!finite(value) ${props.validity ? '|| validity[validityOffset + sourceIndex] == 0u' : ''}) { continue; }
  let area = weights[weightsOffset + slot]; numerator += value * area; support += area;
}
coveredArea[coveredAreaOffset + index] = support;
var value = missing(); var code = 1u;
if (support > 0.0) {
  value = numerator / ${props.normalization === 'covered-area' ? 'support' : String(props.normalization === 'extensive' ? source.spacing[0] * source.spacing[1] : destination.spacing[0] * destination.spacing[1])};
  code = select(2u, 0u, finite(value));
}
result[resultOffset + index] = value; status[statusOffset + index] = code;`
    });
    return props.weightMode === 'reuse' ? [apply] : [prepare, apply];
  }
}

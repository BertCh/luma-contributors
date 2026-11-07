// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma-contributors/graph/gpu-core';
import {createWGSLKernelNode, type WGSLKernelBinding} from '../../utils/wgsl-kernel-nodes';
import {GET_NAN_WGSL, IS_FINITE_BITS_WGSL} from '../../utils/wgsl-common';
import {
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validatePackedView
} from '../../utils/gpu-contributor-utils';
import {
  GPU_PLANAR_SPATIAL_CONTEXT_SUPPORT,
  resolveGPUSpatialContext,
  type GPUSpatialContext
} from '../../analysis-core/contracts/spatial-context';

/** Output columns for each radius bin. */
export const GPU_MARKED_POINT_STATISTICS_COLUMNS = {
  K: 0,
  L: 1,
  PAIR_CORRELATION: 2,
  MARK_CORRELATION: 3
} as const;
/** A rectangular-window point-pattern estimator with caller-supplied intensities and marks. */
export type GPUMarkedPointStatisticsProps = {
  id?: string;
  spatialContext?: GPUSpatialContext;
  positions: GraphDataView<'float32x2'>;
  /** Optional categories. Cross functions require both this view and typePair. */
  types?: GraphDataView<'uint32'>;
  typePair?: readonly [number, number];
  /** Optional finite scalar mark. The mark correlation uses product marks. */
  marks?: GraphDataView<'float32'>;
  /** Positive point intensity, in events per planar area. Values below minimumIntensity are excluded. */
  intensities?: GraphDataView<'float32'>;
  mask?: GraphDataView<'uint32'>;
  /** `[minimumX, minimumY, maximumX, maximumY, maximumRadius, minimumIntensity]`. */
  parameters: GraphDataView<'float32'>;
  radiusCount: number;
  /** Rectangle translation correction (default), or explicitly uncorrected estimates. */
  edgeCorrection?: 'translation' | 'none';
  output: {
    /** Row-major `[K(r), sqrt(K(r)/pi), annular pair correlation, annular mark correlation]`. */
    statistics: GraphDataView<'float32'>;
    /** Directed pair counts in each cumulative radius, excluding self pairs. */
    pairCounts: GraphDataView<'uint32'>;
    /** Per bin: 0 valid, 1 invalid window/empty category, 2 nonfinite accumulation. Empty shells give NaN correlations. */
    status: GraphDataView<'uint32'>;
  };
};

/**
 * Cross-type K/L, inhomogeneous K and annular pair/mark correlation on a rectangle. Radius bins
 * end at maximumRadius * (bin+1)/radiusCount. Ordered non-self pairs use translation weights
 * area / ((width-|dx|)(height-|dy|)); zero-overlap pairs are excluded. Supplied intensity changes
 * normalization to sum(1/(lambda_i lambda_j overlap)), without an estimated-intensity correction.
 * Without intensity, K = area * weightedPairs / (n_focus * n_neighbor), matching the plug-in
 * intensity estimator (including same-type n² normalization). Pair correlation uses annular area,
 * not a kernel smoother. Mark correlation divides weighted mean pair products by the product of
 * the two category means. Zero mean marks and empty shells are undefined. These estimator choices
 * are explicit; they are not interchangeable with every spatstat smoothing/normalization option.
 * Dense deterministic GPU traversal, <= 16,384 rows and 128 radii; O(rows²*radii) work.
 */
export class GPUMarkedPointStatistics implements GPUCommandNodeProducer {
  readonly id: string;
  readonly spatialContext: GPUSpatialContext;
  constructor(readonly props: GPUMarkedPointStatisticsProps) {
    this.id = props.id ?? 'marked-point-statistics';
    this.spatialContext = resolveGPUSpatialContext(this.id, props.spatialContext, {
      ...GPU_PLANAR_SPATIAL_CONTEXT_SUPPORT,
      measure: 'distance'
    });
    const rows = props.positions.length;
    if (
      rows < 1 ||
      rows > 16384 ||
      !Number.isInteger(props.radiusCount) ||
      props.radiusCount < 1 ||
      props.radiusCount > 128
    )
      throw new Error(`${this.id}: invalid bounded pair shape`);
    if (Boolean(props.types) !== Boolean(props.typePair))
      throw new Error(`${this.id}: types and typePair must be supplied together`);
    if (props.typePair?.some(value => !Number.isInteger(value) || value < 0 || value > 0xffffffff))
      throw new Error(`${this.id}: invalid category identifier`);
    if (props.edgeCorrection && !['translation', 'none'].includes(props.edgeCorrection))
      throw new Error(`${this.id}: unknown edge correction`);
    for (const binding of getBindings(props)) {
      validatePackedView(
        binding.view,
        [
          binding.name === 'positions' ? 'float32x2' : binding.type === 'u32' ? 'uint32' : 'float32'
        ],
        this.id
      );
      if (binding.name !== 'parameters' && binding.view.length !== rows)
        throw new Error(`${this.id}: row count mismatch`);
    }
    if (props.parameters.length < 6)
      throw new Error(`${this.id}: six parameter values are required`);
    for (const [name, view] of Object.entries(props.output)) {
      validatePackedView(view, [name === 'statistics' ? 'float32' : 'uint32'], this.id);
      if (view.length !== props.radiusCount * (name === 'statistics' ? 4 : 1))
        throw new Error(`${this.id}: output size mismatch`);
    }
    validateGraphOutputsDisjointFromInputs(
      this.id,
      Object.values(props.output),
      getBindings(props).map(binding => binding.view)
    );
  }
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const rows = props.positions.length;
    const bins = props.radiusCount;
    const partials = createTransientView(graph, `${id}-partials`, 'float32', rows * bins * 6);
    const moments = createTransientView(graph, `${id}-moments`, 'float32', 4);
    const bindings = getBindings(props);
    validateGraphViewsBelongToGraph(id, graph, [
      ...bindings.map(binding => binding.view),
      ...Object.values(props.output)
    ]);
    const common = `${GET_NAN_WGSL}\n${IS_FINITE_BITS_WGSL}
fn included(row: u32) -> bool {
  if (!isFiniteBits(parameters[parametersOffset + 5u]) || parameters[parametersOffset + 5u] < 0.0) { return false; }
  ${props.mask ? 'if (mask[maskOffset + row] == 0u) { return false; }' : ''}
  ${props.marks ? 'if (!isFiniteBits(marks[marksOffset + row])) { return false; }' : ''}
  ${props.intensities ? 'let intensity = intensities[intensitiesOffset + row]; if (!isFiniteBits(intensity) || !(intensity > 0.0) || intensity < max(0.0, parameters[parametersOffset + 5u])) { return false; }' : ''}
  let x = positions[positionsOffset + row * 2u]; let y = positions[positionsOffset + row * 2u + 1u];
  return isFiniteBits(x) && isFiniteBits(y) && x >= parameters[parametersOffset] && y >= parameters[parametersOffset + 1u] && x <= parameters[parametersOffset + 2u] && y <= parameters[parametersOffset + 3u];
}
fn isFocus(row: u32) -> bool { return ${props.types ? `types[typesOffset + row] == ${props.typePair![0]}u` : 'true'}; }
fn isNeighbor(row: u32) -> bool { return ${props.types ? `types[typesOffset + row] == ${props.typePair![1]}u` : 'true'}; }
fn mark(row: u32) -> f32 { return ${props.marks ? 'marks[marksOffset + row]' : '1.0'}; }`;
    return [
      createWGSLKernelNode(graph, {
        id: `${id}-moments`,
        operation: 'GPUMarkedPointStatistics',
        variant: 'moments',
        invocationCount: 1,
        bindings: [
          ...bindings,
          {name: 'moments', view: moments, type: 'f32', access: 'read_write'}
        ],
        declarations: common,
        body: `var focusCount = 0.0; var neighborCount = 0.0; var focusMean = 0.0; var neighborMean = 0.0;
for (var row = 0u; row < ${rows}u; row++) {
  if (!included(row)) { continue; }
  if (isFocus(row)) { focusCount += 1.0; focusMean += (mark(row) - focusMean) / focusCount; }
  if (isNeighbor(row)) { neighborCount += 1.0; neighborMean += (mark(row) - neighborMean) / neighborCount; }
}
moments[momentsOffset] = focusCount; moments[momentsOffset + 1u] = neighborCount;
moments[momentsOffset + 2u] = focusMean; moments[momentsOffset + 3u] = neighborMean;`
      }),
      createWGSLKernelNode(graph, {
        id: `${id}-pairs`,
        operation: 'GPUMarkedPointStatistics',
        variant: 'pairs',
        invocationCount: rows * bins,
        bindings: [
          ...bindings,
          {name: 'partials', view: partials, type: 'f32', access: 'read_write'}
        ],
        declarations: common,
        body: `let focus = index / ${bins}u; let bin = index % ${bins}u;
let width = parameters[parametersOffset + 2u] - parameters[parametersOffset]; let height = parameters[parametersOffset + 3u] - parameters[parametersOffset + 1u]; let area = width * height;
let upper = parameters[parametersOffset + 4u] * f32(bin + 1u) / ${bins}.0; let lower = parameters[parametersOffset + 4u] * f32(bin) / ${bins}.0;
var cumulative = 0.0; var shell = 0.0; var products = 0.0; var shellWeight = 0.0; var count = 0.0;
if (included(focus) && isFocus(focus) && width > 0.0 && height > 0.0 && upper > 0.0 && isFiniteBits(upper) && isFiniteBits(area)) {
  for (var neighbor = 0u; neighbor < ${rows}u; neighbor++) {
    if (neighbor == focus || !included(neighbor) || !isNeighbor(neighbor)) { continue; }
    let delta = abs(vec2<f32>(positions[positionsOffset + focus * 2u] - positions[positionsOffset + neighbor * 2u], positions[positionsOffset + focus * 2u + 1u] - positions[positionsOffset + neighbor * 2u + 1u]));
    let distance = length(delta); if (distance > upper || !isFiniteBits(distance)) { continue; }
    let overlap = ${props.edgeCorrection === 'none' ? 'area' : '(width - delta.x) * (height - delta.y)'};
    if (!(overlap > 0.0)) { continue; }
    let correction = area / overlap;
    let intensityWeight = ${props.intensities ? '1.0 / intensities[intensitiesOffset + focus] / intensities[intensitiesOffset + neighbor]' : '1.0'};
    let weight = correction * intensityWeight;
    cumulative += weight; count += 1.0;
    if (distance > lower || bin == 0u) { shell += weight; products += weight * mark(focus) * mark(neighbor); shellWeight += weight; }
  }
}
let base = partialsOffset + index * 6u;
partials[base] = cumulative; partials[base + 1u] = shell; partials[base + 2u] = products; partials[base + 3u] = shellWeight; partials[base + 4u] = count; partials[base + 5u] = 0.0;`
      }),
      createWGSLKernelNode(graph, {
        id: `${id}-finish`,
        operation: 'GPUMarkedPointStatistics',
        variant: 'finish',
        invocationCount: bins,
        bindings: [
          {name: 'partials', view: partials, type: 'f32', access: 'read'},
          {name: 'moments', view: moments, type: 'f32', access: 'read'},
          {name: 'parameters', view: props.parameters, type: 'f32', access: 'read'},
          {name: 'result', view: props.output.statistics, type: 'f32', access: 'read_write'},
          {name: 'counts', view: props.output.pairCounts, type: 'u32', access: 'read_write'},
          {name: 'status', view: props.output.status, type: 'u32', access: 'read_write'}
        ],
        declarations: `${GET_NAN_WGSL}\n${IS_FINITE_BITS_WGSL}`,
        body: `var cumulative = 0.0; var shell = 0.0; var products = 0.0; var shellWeight = 0.0; var count = 0u;
for (var row = 0u; row < ${rows}u; row++) { let base = partialsOffset + (row * ${bins}u + index) * 6u; cumulative += partials[base]; shell += partials[base + 1u]; products += partials[base + 2u]; shellWeight += partials[base + 3u]; count += u32(partials[base + 4u]); }
let width = parameters[parametersOffset + 2u] - parameters[parametersOffset]; let height = parameters[parametersOffset + 3u] - parameters[parametersOffset + 1u]; let area = width * height;
let upper = parameters[parametersOffset + 4u] * f32(index + 1u) / ${bins}.0; let lower = parameters[parametersOffset + 4u] * f32(index) / ${bins}.0;
let valid = width > 0.0 && height > 0.0 && upper > 0.0 && isFiniteBits(upper) && isFiniteBits(area) && moments[momentsOffset] > 0.0 && moments[momentsOffset + 1u] > 0.0;
let scale = ${props.intensities ? '1.0 / area' : 'area / moments[momentsOffset] / moments[momentsOffset + 1u]'};
let estimate = cumulative * scale; let pairCorrelation = shell * scale / (3.141592653589793 * (upper * upper - lower * lower));
let markMean = moments[momentsOffset + 2u] * moments[momentsOffset + 3u];
let base = resultOffset + index * 4u;
result[base] = select(getNaN(), estimate, valid && isFiniteBits(estimate)); result[base + 1u] = select(getNaN(), sqrt(estimate / 3.141592653589793), valid && isFiniteBits(estimate));
result[base + 2u] = select(getNaN(), pairCorrelation, valid && shellWeight > 0.0 && isFiniteBits(pairCorrelation));
let markCorrelation = products / shellWeight / markMean;
result[base + 3u] = select(getNaN(), markCorrelation, valid && shellWeight > 0.0 && markMean != 0.0 && isFiniteBits(markCorrelation));
counts[countsOffset + index] = count; status[statusOffset + index] = select(1u, select(2u, 0u, isFiniteBits(estimate) && isFiniteBits(products)), valid);`
      })
    ];
  }
}
function getBindings(props: GPUMarkedPointStatisticsProps): WGSLKernelBinding[] {
  return [
    {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
    {name: 'parameters', view: props.parameters, type: 'f32', access: 'read'},
    ...(props.types
      ? [{name: 'types', view: props.types, type: 'u32' as const, access: 'read' as const}]
      : []),
    ...(props.marks
      ? [{name: 'marks', view: props.marks, type: 'f32' as const, access: 'read' as const}]
      : []),
    ...(props.intensities
      ? [
          {
            name: 'intensities',
            view: props.intensities,
            type: 'f32' as const,
            access: 'read' as const
          }
        ]
      : []),
    ...(props.mask
      ? [{name: 'mask', view: props.mask, type: 'u32' as const, access: 'read' as const}]
      : [])
  ];
}

// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUReduction,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  createWGSLKernelNode,
  type WGSLKernelBinding
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {GET_NAN_WGSL, IS_FINITE_BITS_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {
  validateCapacity,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validateMaximumCount,
  validateSinglePackedView
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  getPairStatisticsInputNodes,
  getPairStatisticsSharedWGSL,
  validatePairStatisticsRowCount
} from './pair-statistics-grid';
import {
  getPairStatisticsCellNeighborLoopWGSL,
  getPairStatisticsSegmentTotalNodes,
  PAIR_STATISTICS_PARTIAL_ENTRY_MAXIMUM,
  validatePairStatisticsGridSize
} from './pair-statistics-cell-neighbors';
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
  /**
   * Optional compile-time `[columns, rows]` cell lattice over the window of `parameters`, with
   * cells at least `maximumRadius` wide, so pairs come from the 3 x 3 cells around each focus
   * instead of every row and up to `2^24 / (3 * radiusCount)` rows are accepted. Pair counts are
   * unchanged; sums are added in cell order, so statistics can differ in the last float32 bits.
   * A `maximumRadius` whose square overflows float32 gives no pairs. Omitted keeps the dense
   * all-pairs traversal (at most 16,384 rows).
   */
  gridSize?: readonly [number, number];
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
 * Without intensity, K = area * weightedPairs / (n_focus * n_neighbor) for two different types
 * (spatstat Kcross) and area * weightedPairs / (n (n - 1)) when focus and neighbor are the same
 * set (no types, or equal typePair), the normalization of GPURipley and spatstat Kest; a
 * same-type set of one point has no K (NaN, status 1). The pair correlation shares that
 * normalization. With intensity the n² form stays. Pair correlation uses annular area,
 * not a kernel smoother. Mark correlation divides weighted mean pair products by the product of
 * the two category means. Zero mean marks and empty shells are undefined. These estimator choices
 * are explicit; they are not interchangeable with every spatstat smoothing/normalization option.
 * Without `gridSize`: dense deterministic GPU traversal, <= 16,384 rows and 128 radii,
 * O(rows²*radii) work. With `gridSize`: one pass per focus over a cell index of the window.
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
    validateCapacity(this.id, 'positions length', rows, 1, props.gridSize ? 2 ** 24 - 1 : 16384);
    if (!Number.isInteger(props.radiusCount) || props.radiusCount < 1 || props.radiusCount > 128)
      throw new Error(`${this.id}: invalid bounded pair shape`);
    if (props.gridSize) {
      validatePairStatisticsGridSize(this.id, props.gridSize);
      // Six float32 partials per row and radius; the limit is counted in pairs of float32.
      validateMaximumCount(
        this.id,
        'positions length x radiusCount x 3',
        rows * props.radiusCount * 3,
        PAIR_STATISTICS_PARTIAL_ENTRY_MAXIMUM
      );
    }
    if (Boolean(props.types) !== Boolean(props.typePair))
      throw new Error(`${this.id}: types and typePair must be supplied together`);
    if (props.typePair?.some(value => !Number.isInteger(value) || value < 0 || value > 0xffffffff))
      throw new Error(`${this.id}: invalid category identifier`);
    if (props.edgeCorrection && !['translation', 'none'].includes(props.edgeCorrection))
      throw new Error(`${this.id}: unknown edge correction`);
    for (const binding of getBindings(props)) {
      validateSinglePackedView(
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
      validateSinglePackedView(view, [name === 'statistics' ? 'float32' : 'uint32'], this.id);
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
    if (props.gridSize) {
      return getGridNodes(graph, id, props, props.gridSize, common, moments);
    }
    const partials = createTransientView(graph, `${id}-partials`, 'float32', rows * bins * 6);
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
      ...[
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
        })
      ],
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
${getFinishWGSL(props)}`
      })
    ];
  }
}

/** Writes one radius bin's statistics, count and status from its totals and the moments. */
function getFinishWGSL(props: GPUMarkedPointStatisticsProps): string {
  const bins = props.radiusCount;
  // Without intensities the plug-in normalisation is area / (n_focus * n_neighbor) for two
  // different types and area / (n (n - 1)) when focus and neighbor are the same set (no types, or
  // equal typePair): the n (n - 1) ordered pairs of GPURipley and spatstat Kest. One point of a
  // type then has no same-type K.
  const isSameType = !props.typePair || props.typePair[0] === props.typePair[1];
  const neighborDivisor =
    isSameType && !props.intensities
      ? '(moments[momentsOffset + 1u] - 1.0)'
      : 'moments[momentsOffset + 1u]';
  return `let width = parameters[parametersOffset + 2u] - parameters[parametersOffset]; let height = parameters[parametersOffset + 3u] - parameters[parametersOffset + 1u]; let area = width * height;
let upper = parameters[parametersOffset + 4u] * f32(index + 1u) / ${bins}.0; let lower = parameters[parametersOffset + 4u] * f32(index) / ${bins}.0;
let valid = width > 0.0 && height > 0.0 && upper > 0.0 && isFiniteBits(upper) && isFiniteBits(area) && moments[momentsOffset] > 0.0 && ${neighborDivisor} > 0.0;
let scale = ${props.intensities ? '1.0 / area' : `area / moments[momentsOffset] / ${neighborDivisor}`};
let estimate = cumulative * scale; let pairCorrelation = shell * scale / (3.141592653589793 * (upper * upper - lower * lower));
let markMean = moments[momentsOffset + 2u] * moments[momentsOffset + 3u];
let base = resultOffset + index * 4u;
result[base] = select(getNaN(), estimate, valid && isFiniteBits(estimate)); result[base + 1u] = select(getNaN(), sqrt(estimate / 3.141592653589793), valid && isFiniteBits(estimate));
result[base + 2u] = select(getNaN(), pairCorrelation, valid && shellWeight > 0.0 && isFiniteBits(pairCorrelation));
let markCorrelation = products / shellWeight / markMean;
result[base + 3u] = select(getNaN(), markCorrelation, valid && shellWeight > 0.0 && markMean != 0.0 && isFiniteBits(markCorrelation));
counts[countsOffset + index] = count; status[statusOffset + index] = select(1u, select(2u, 0u, isFiniteBits(estimate) && isFiniteBits(products)), valid);`;
}
/**
 * Grid path: per-row roles (focus, neighbor), marks and intensities in one view, a cell index over
 * the window with cells at least `maximumRadius` wide, then one invocation per focus row that
 * visits the 3 x 3 neighboring cells. Each pair lands in the first radius bin whose upper edge
 * holds it (the dense pass's shell); cumulative sums and counts are prefix sums over the shells,
 * written to bin-major tables. Segment totals (at most 256 entries per invocation) reduce the
 * tables per bin; the moments come from `GPUReduction` sums (means are sums over counts, not the
 * dense pass's running means). No pass loops over every row in one invocation.
 */
function getGridNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  props: GPUMarkedPointStatisticsProps,
  gridSize: readonly [number, number],
  common: string,
  moments: GraphDataView<'float32'>
): GPUCommandNode<Parameters>[] {
  validatePairStatisticsRowCount(id, graph.device, props.positions.length);
  const rows = props.positions.length;
  const bins = props.radiusCount;
  // Bin-major tables: sums are [cumulative, shell, products] x bin x row; counts are bin x row.
  const binSums = createTransientView(graph, `${id}-bin-sums`, 'float32', 3 * bins * rows);
  const binCounts = createTransientView(graph, `${id}-bin-counts`, 'uint32', bins * rows);
  const totalSums = createTransientView(graph, `${id}-total-sums`, 'float32', 3 * bins);
  const totalCounts = createTransientView(graph, `${id}-total-counts`, 'uint32', bins);
  // Per row: focus and neighbor indicators, and their marks (0 when the role is absent).
  const momentTerms = {
    focusCount: createTransientView(graph, `${id}-focus-indicators`, 'uint32', rows),
    neighborCount: createTransientView(graph, `${id}-neighbor-indicators`, 'uint32', rows),
    focusSum: createTransientView(graph, `${id}-focus-marks`, 'float32', rows),
    neighborSum: createTransientView(graph, `${id}-neighbor-marks`, 'float32', rows)
  };
  const momentTotals = {
    focusCount: createTransientView(graph, `${id}-focus-count`, 'uint32', 1),
    neighborCount: createTransientView(graph, `${id}-neighbor-count`, 'uint32', 1),
    focusSum: createTransientView(graph, `${id}-focus-mark-sum`, 'float32', 1),
    neighborSum: createTransientView(graph, `${id}-neighbor-mark-sum`, 'float32', 1)
  };
  const inclusion = createTransientView(graph, `${id}-inclusion`, 'uint32', rows);
  // Per row: role bits (1 focus, 2 neighbor) as float32, mark and intensity.
  const attributes = createTransientView(graph, `${id}-attributes`, 'float32', rows * 3);
  const inputs = getPairStatisticsInputNodes<Parameters>(graph, {
    id: `${id}-grid`,
    operation: 'GPUMarkedPointStatistics',
    positions: props.positions,
    parameters: props.parameters,
    gridSize,
    mask: inclusion
  });
  return [
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-inclusion`,
      operation: 'GPUMarkedPointStatistics',
      variant: 'inclusion',
      invocationCount: rows,
      bindings: [
        ...getBindings(props),
        {name: 'inclusion', view: inclusion, type: 'u32', access: 'read_write'},
        {name: 'attributes', view: attributes, type: 'f32', access: 'read_write'}
      ],
      declarations: common,
      body: `let isIncluded = included(index);
let roles = select(0u, select(0u, 1u, isFocus(index)) | select(0u, 2u, isNeighbor(index)), isIncluded);
inclusion[inclusionOffset + index] = select(0u, 1u, roles != 0u);
attributes[attributesOffset + index * 3u] = f32(roles);
attributes[attributesOffset + index * 3u + 1u] = select(0.0, mark(index), isIncluded);
attributes[attributesOffset + index * 3u + 2u] = ${props.intensities ? 'intensities[intensitiesOffset + index]' : '1.0'};`
    }),
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-moment-terms`,
      operation: 'GPUMarkedPointStatistics',
      variant: 'moment-terms',
      invocationCount: rows,
      bindings: [
        {name: 'attributes', view: attributes, type: 'f32', access: 'read'},
        {name: 'focusCounts', view: momentTerms.focusCount, type: 'u32', access: 'read_write'},
        {
          name: 'neighborCounts',
          view: momentTerms.neighborCount,
          type: 'u32',
          access: 'read_write'
        },
        {name: 'focusMarks', view: momentTerms.focusSum, type: 'f32', access: 'read_write'},
        {name: 'neighborMarks', view: momentTerms.neighborSum, type: 'f32', access: 'read_write'}
      ],
      body: `let roles = u32(attributes[attributesOffset + index * 3u]); let rowMark = attributes[attributesOffset + index * 3u + 1u];
let isFocusRow = (roles & 1u) != 0u; let isNeighborRow = (roles & 2u) != 0u;
focusCounts[focusCountsOffset + index] = select(0u, 1u, isFocusRow); neighborCounts[neighborCountsOffset + index] = select(0u, 1u, isNeighborRow);
focusMarks[focusMarksOffset + index] = select(0.0, rowMark, isFocusRow); neighborMarks[neighborMarksOffset + index] = select(0.0, rowMark, isNeighborRow);`
    }),
    ...[
      new GPUReduction({
        id: `${id}-focus-count-reduction`,
        input: momentTerms.focusCount,
        output: momentTotals.focusCount,
        operation: 'sum'
      }),
      new GPUReduction({
        id: `${id}-neighbor-count-reduction`,
        input: momentTerms.neighborCount,
        output: momentTotals.neighborCount,
        operation: 'sum'
      }),
      new GPUReduction({
        id: `${id}-focus-mark-reduction`,
        input: momentTerms.focusSum,
        output: momentTotals.focusSum,
        operation: 'sum'
      }),
      new GPUReduction({
        id: `${id}-neighbor-mark-reduction`,
        input: momentTerms.neighborSum,
        output: momentTotals.neighborSum,
        operation: 'sum'
      })
    ].flatMap(reduction => reduction.getCommandNodes(graph)),
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-moments`,
      operation: 'GPUMarkedPointStatistics',
      variant: 'grid-moments',
      invocationCount: 1,
      bindings: [
        {name: 'focusCount', view: momentTotals.focusCount, type: 'u32', access: 'read'},
        {name: 'neighborCount', view: momentTotals.neighborCount, type: 'u32', access: 'read'},
        {name: 'focusSum', view: momentTotals.focusSum, type: 'f32', access: 'read'},
        {name: 'neighborSum', view: momentTotals.neighborSum, type: 'f32', access: 'read'},
        {name: 'moments', view: moments, type: 'f32', access: 'read_write'}
      ],
      body: `let focusTotal = f32(focusCount[focusCountOffset]); let neighborTotal = f32(neighborCount[neighborCountOffset]);
moments[momentsOffset] = focusTotal; moments[momentsOffset + 1u] = neighborTotal;
moments[momentsOffset + 2u] = select(0.0, focusSum[focusSumOffset] / focusTotal, focusTotal > 0.0);
moments[momentsOffset + 3u] = select(0.0, neighborSum[neighborSumOffset] / neighborTotal, neighborTotal > 0.0);`
    }),
    ...inputs.nodes,
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-grid-pairs`,
      operation: 'GPUMarkedPointStatistics',
      variant: 'grid-pairs',
      invocationCount: rows,
      bindings: [
        {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
        {name: 'attributes', view: attributes, type: 'f32', access: 'read'},
        {name: 'parameters', view: props.parameters, type: 'f32', access: 'read'},
        {name: 'cellKeys', view: inputs.cellKeys, type: 'u32', access: 'read'},
        {name: 'cellOffsets', view: inputs.cellOffsets, type: 'u32', access: 'read'},
        {name: 'sortedRows', view: inputs.sortedRows, type: 'u32', access: 'read'},
        {name: 'binSums', view: binSums, type: 'f32', access: 'read_write'},
        {name: 'binCounts', view: binCounts, type: 'u32', access: 'read_write'}
      ],
      declarations: `${getPairStatisticsSharedWGSL(gridSize)}
${IS_FINITE_BITS_WGSL}
const BINS: u32 = ${bins}u;
fn getUpper(bin: u32) -> f32 { return parameters[parametersOffset + 4u] * f32(bin + 1u) / ${bins}.0; }`,
      body: `var shells: array<f32, BINS>; var products: array<f32, BINS>; var counts: array<u32, BINS>;
let lattice = readLattice();
let focus = index; let focusKey = cellKeys[cellKeysOffset + focus];
let width = parameters[parametersOffset + 2u] - parameters[parametersOffset]; let height = parameters[parametersOffset + 3u] - parameters[parametersOffset + 1u]; let area = width * height;
let maximumRadius = parameters[parametersOffset + 4u];
let focusRoles = u32(attributes[attributesOffset + focus * 3u]);
if (lattice.valid && focusKey < CELL_COUNT && (focusRoles & 1u) != 0u && width > 0.0 && height > 0.0 && maximumRadius > 0.0 && isFiniteBits(area)) ${getPairStatisticsCellNeighborLoopWGSL(
        'focusKey',
        `if (neighbor == focus || (u32(attributes[attributesOffset + neighbor * 3u]) & 2u) == 0u) { continue; }
    let delta = abs(vec2<f32>(positions[positionsOffset + focus * 2u] - positions[positionsOffset + neighbor * 2u], positions[positionsOffset + focus * 2u + 1u] - positions[positionsOffset + neighbor * 2u + 1u]));
    let distance = length(delta);
    if (!isFiniteBits(distance) || distance > getUpper(BINS - 1u)) { continue; }
    var shell = min(u32(distance / maximumRadius * ${bins}.0), BINS - 1u);
    loop { if (shell == 0u || distance > getUpper(shell - 1u)) { break; } shell -= 1u; }
    loop { if (distance <= getUpper(shell)) { break; } shell += 1u; }
    let overlap = ${props.edgeCorrection === 'none' ? 'area' : '(width - delta.x) * (height - delta.y)'};
    if (!(overlap > 0.0)) { continue; }
    let correction = area / overlap;
    let intensityWeight = ${props.intensities ? '1.0 / attributes[attributesOffset + focus * 3u + 2u] / attributes[attributesOffset + neighbor * 3u + 2u]' : '1.0'};
    let weight = correction * intensityWeight;
    shells[shell] += weight; counts[shell] += 1u;
    products[shell] += weight * attributes[attributesOffset + focus * 3u + 1u] * attributes[attributesOffset + neighbor * 3u + 1u];`
      )}
var cumulative = 0.0; var count = 0u;
for (var bin = 0u; bin < BINS; bin++) {
  cumulative += shells[bin]; count += counts[bin];
  let upper = getUpper(bin);
  let usable = upper > 0.0 && isFiniteBits(upper);
  let entry = bin * ${rows}u + focus;
  binSums[binSumsOffset + entry] = select(0.0, cumulative, usable);
  binSums[binSumsOffset + ${bins * rows}u + entry] = select(0.0, shells[bin], usable);
  binSums[binSumsOffset + ${2 * bins * rows}u + entry] = select(0.0, products[bin], usable);
  binCounts[binCountsOffset + entry] = select(0u, count, usable);
}`
    }),
    ...getPairStatisticsSegmentTotalNodes<Parameters>(graph, {
      id: `${id}-total-sums`,
      operation: 'GPUMarkedPointStatistics',
      segmentCount: 3 * bins,
      rowCount: rows,
      input: binSums,
      output: totalSums
    }),
    ...getPairStatisticsSegmentTotalNodes<Parameters>(graph, {
      id: `${id}-total-counts`,
      operation: 'GPUMarkedPointStatistics',
      segmentCount: bins,
      rowCount: rows,
      input: binCounts,
      output: totalCounts
    }),
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-finish`,
      operation: 'GPUMarkedPointStatistics',
      variant: 'grid-finish',
      invocationCount: bins,
      bindings: [
        {name: 'totalSums', view: totalSums, type: 'f32', access: 'read'},
        {name: 'totalCounts', view: totalCounts, type: 'u32', access: 'read'},
        {name: 'moments', view: moments, type: 'f32', access: 'read'},
        {name: 'parameters', view: props.parameters, type: 'f32', access: 'read'},
        {name: 'result', view: props.output.statistics, type: 'f32', access: 'read_write'},
        {name: 'counts', view: props.output.pairCounts, type: 'u32', access: 'read_write'},
        {name: 'status', view: props.output.status, type: 'u32', access: 'read_write'}
      ],
      declarations: `${GET_NAN_WGSL}\n${IS_FINITE_BITS_WGSL}`,
      body: `let cumulative = totalSums[totalSumsOffset + index]; let shell = totalSums[totalSumsOffset + ${bins}u + index];
let products = totalSums[totalSumsOffset + ${2 * bins}u + index]; let shellWeight = shell;
let count = totalCounts[totalCountsOffset + index];
${getFinishWGSL(props)}`
    })
  ];
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

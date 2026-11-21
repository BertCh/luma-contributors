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
  validateRequirement,
  validateGPUStorageBindingByteLength,
  validateSinglePackedView
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  getPairStatisticsInputNodes,
  getPairStatisticsSharedWGSL,
  validatePairStatisticsRowCount
} from './pair-statistics-grid';
import {
  validateGraphChunkBindingRanges,
  validateDisjointWritableViews
} from '../../analysis-core/kernels/graph-chunk-dispatch';
import {validateGraphChunkGroupKernelSupport} from '../../analysis-core/kernels/graph-chunk-groups';
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
/** Dense reference work is quadratic in rows for every radius; keep it explicitly bounded. */
const MAXIMUM_BRUTE_FORCE_ROWS = 16384;
/** Per-invocation shell arrays in the indexed path and per-radius reference accumulators. */
const MAXIMUM_RADIUS_COUNT = 128;
/** Default upper dimensions; the per-frame radius can reduce the active lattice. */
const DEFAULT_GRID_SIZE = [256, 256] as const;
/** Unscaled small-mark sums stay below 2^124 at the 2^24 row-ID limit. */
const MAXIMUM_UNSCALED_MARK_MAGNITUDE = 2 ** 100;
/** The partial-table budget keeps rows below 2^23, so large-mark sums have at least 2x headroom. */
const LARGE_MARK_SUM_SCALE_EXPONENT = 24;

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
   * Defaults to `[256, 256]`. Finite radii whose square overflows float32 use a single cell,
   * retaining the finite-distance predicate. Requires `searchStructure: 'grid'` (the default).
   */
  gridSize?: readonly [number, number];
  /**
   * Compile-time candidate search: `'grid'` (default), or `'brute-force'` for the dense reference
   * traversal, limited to 16,384 rows. Both use the same estimators and radius edges; fixed-tree
   * grid sums can differ from the serial reference sums within float32 rounding tolerance.
   */
  searchStructure?: 'grid' | 'brute-force';
  output: {
    /** Row-major `[K(r), sqrt(K(r)/pi), annular pair correlation, annular mark correlation]`. */
    statistics: GraphDataView<'float32'>;
    /** Directed non-self pairs per radius, clamped to 0xffffffff on overflow (status 3, NaN statistics). */
    pairCounts: GraphDataView<'uint32'>;
    /** Per bin: 0 valid, 1 invalid window/empty category, 2 nonfinite accumulation, 3 pair-count overflow. Empty shells give NaN correlations. */
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
 * the two category means. Indexed means average large marks at a power-of-two scale before
 * restoring their magnitude; an unnormalized category mark sum need not fit float32. Small marks
 * remain unscaled so they survive cancellation of large opposite marks.
 * Zero mean marks and empty shells are undefined. These estimator choices
 * are explicit; they are not interchangeable with every spatstat smoothing/normalization option.
 * Defaults to one pass per focus over a cell index of the window, with `[256, 256]` maximum
 * lattice dimensions and at most 128 radii. The partial-table budget is `rows * radiusCount * 3`
 * <= 2^24 floats, further bounded by the device. `searchStructure: 'brute-force'` explicitly
 * selects the dense reference: <= 16,384 rows and O(rows²*radii) work. A broad radius or dense
 * cell still requires quadratic pair work in the indexed path.
 * Per-frame: positions, types, marks, intensities, mask and parameters. Compile-time: view
 * lengths, optional views, typePair, radiusCount, gridSize, searchStructure and edgeCorrection.
 */
export class GPUMarkedPointStatistics implements GPUCommandNodeProducer {
  readonly id: string;
  readonly spatialContext: GPUSpatialContext;
  private readonly searchStructure: 'grid' | 'brute-force';
  constructor(readonly props: GPUMarkedPointStatisticsProps) {
    this.id = props.id ?? 'marked-point-statistics';
    this.spatialContext = resolveGPUSpatialContext(this.id, props.spatialContext, {
      ...GPU_PLANAR_SPATIAL_CONTEXT_SUPPORT,
      measure: 'distance'
    });
    const rows = props.positions.length;
    this.searchStructure = props.searchStructure ?? 'grid';
    validateRequirement(
      this.id,
      this.searchStructure === 'grid' || this.searchStructure === 'brute-force',
      'searchStructure must be grid or brute-force'
    );
    validateRequirement(
      this.id,
      this.searchStructure === 'grid' || props.gridSize === undefined,
      'gridSize requires searchStructure grid'
    );
    validateCapacity(
      this.id,
      'positions length',
      rows,
      1,
      this.searchStructure === 'brute-force' ? MAXIMUM_BRUTE_FORCE_ROWS : 2 ** 24 - 1
    );
    validateCapacity(this.id, 'radiusCount', props.radiusCount, 1, MAXIMUM_RADIUS_COUNT);
    if (this.searchStructure === 'grid') {
      validatePairStatisticsGridSize(this.id, props.gridSize ?? DEFAULT_GRID_SIZE);
      validateMaximumCount(
        this.id,
        'positions length x radiusCount x 3',
        rows * props.radiusCount * 3,
        PAIR_STATISTICS_PARTIAL_ENTRY_MAXIMUM
      );
    }
    validateRequirement(
      `${this.id}:`,
      Boolean(props.types) === Boolean(props.typePair),
      'types and typePair must be supplied together'
    );
    if (props.typePair) {
      validateRequirement(
        `${this.id}:`,
        props.typePair.length === 2,
        `typePair must hold exactly 2 category identifiers, got ${props.typePair.length}`
      );
      for (const index of [0, 1] as const)
        validateCapacity(`${this.id}:`, `typePair[${index}]`, props.typePair[index], 0, 0xffffffff);
    }
    validateRequirement(
      `${this.id}:`,
      !props.edgeCorrection || ['translation', 'none'].includes(props.edgeCorrection),
      "edgeCorrection must be 'translation' or 'none'"
    );
    for (const binding of getBindings(props)) {
      validateSinglePackedView(
        binding.view,
        [
          binding.name === 'positions' ? 'float32x2' : binding.type === 'u32' ? 'uint32' : 'float32'
        ],
        `${this.id}: ${binding.name}`
      );
      if (binding.name !== 'parameters')
        validateRequirement(
          `${this.id}:`,
          binding.view.length === rows,
          `${binding.name} length ${binding.view.length} must equal positions length (${rows})`
        );
    }
    validateRequirement(
      `${this.id}:`,
      props.parameters.length >= 6,
      `parameters length ${props.parameters.length} must be at least 6`
    );
    for (const [name, view] of Object.entries(props.output)) {
      validateSinglePackedView(
        view,
        [name === 'statistics' ? 'float32' : 'uint32'],
        `${this.id}: output.${name}`
      );
      const columns = name === 'statistics' ? 4 : 1;
      validateRequirement(
        `${this.id}:`,
        view.length === props.radiusCount * columns,
        `output.${name} length ${view.length} must equal radiusCount x ${columns} (${props.radiusCount * columns})`
      );
    }
    validateDisjointWritableViews(this.id, props.output);
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
    const gridSize = props.gridSize ?? DEFAULT_GRID_SIZE;
    validatePairStatisticsRowCount(id, graph.device, rows);
    validateMaximumCount(
      id,
      'pair partials byteLength',
      rows * bins * (this.searchStructure === 'brute-force' ? 6 : 3) * 4,
      graph.device.limits.maxBufferSize
    );
    validateGPUStorageBindingByteLength(
      graph.device,
      id,
      'pair partials',
      rows * bins * (this.searchStructure === 'brute-force' ? 6 : 3) * 4,
      'reduce positions length or radiusCount'
    );
    if (this.searchStructure === 'grid') {
      validateMaximumCount(
        id,
        'cell offsets byteLength',
        (gridSize[0] * gridSize[1] + 1) * 4,
        graph.device.limits.maxBufferSize
      );
      validateGPUStorageBindingByteLength(
        graph.device,
        id,
        'cell offsets',
        (gridSize[0] * gridSize[1] + 1) * 4,
        'reduce gridSize'
      );
    }
    const bindings = getBindings(props);
    validateGraphChunkBindingRanges(graph, id, {
      ...Object.fromEntries(bindings.map(binding => [binding.name, binding.view])),
      ...props.output
    });
    if (this.searchStructure === 'grid') validateGraphChunkGroupKernelSupport(graph, id, 2);
    validateGraphViewsBelongToGraph(id, graph, [
      ...bindings.map(binding => binding.view),
      ...Object.values(props.output)
    ]);
    const moments = createTransientView(graph, `${id}-moments`, 'float32', 4);
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
    if (this.searchStructure === 'grid') {
      return getGridNodes(graph, id, props, gridSize, common, moments);
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
function getFinishWGSL(props: GPUMarkedPointStatisticsProps, overflow = 'false'): string {
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
let overflow = ${overflow};
let valid = !overflow && width > 0.0 && height > 0.0 && upper > 0.0 && isFiniteBits(upper) && isFiniteBits(area) && moments[momentsOffset] > 0.0 && ${neighborDivisor} > 0.0;
let scale = ${props.intensities ? '1.0 / area' : `area / moments[momentsOffset] / ${neighborDivisor}`};
let estimate = cumulative * scale; let pairCorrelation = shell * scale / (3.141592653589793 * (upper * upper - lower * lower));
let markMean = moments[momentsOffset + 2u] * moments[momentsOffset + 3u];
let base = resultOffset + index * 4u;
result[base] = select(getNaN(), estimate, valid && isFiniteBits(estimate)); result[base + 1u] = select(getNaN(), sqrt(estimate / 3.141592653589793), valid && isFiniteBits(estimate));
result[base + 2u] = select(getNaN(), pairCorrelation, valid && shellWeight > 0.0 && isFiniteBits(pairCorrelation));
let markCorrelation = products / shellWeight / markMean;
result[base + 3u] = select(getNaN(), markCorrelation, valid && shellWeight > 0.0 && markMean != 0.0 && isFiniteBits(markCorrelation));
counts[countsOffset + index] = count; status[statusOffset + index] = select(select(1u, select(2u, 0u, isFiniteBits(estimate) && isFiniteBits(products)), valid), 3u, overflow);`;
}
/**
 * Grid path: per-row roles (focus, neighbor), marks and intensities in one view, a cell index over
 * the window with cells at least `maximumRadius` wide, then one invocation per focus row that
 * visits the 3 x 3 neighboring cells. Each pair lands in the first radius bin whose upper edge
 * holds it (the dense pass's shell); cumulative sums and counts are prefix sums over the shells,
 * written to bin-major tables. Segment totals (at most 256 entries per invocation) reduce the
 * tables per bin; the moments come from `GPUReduction` sums (means are sums over counts, not the
 * dense pass's running means). Outside candidate walks, reductions have bounded work per invocation.
 */
function getGridNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  props: GPUMarkedPointStatisticsProps,
  gridSize: readonly [number, number],
  common: string,
  moments: GraphDataView<'float32'>
): GPUCommandNode<Parameters>[] {
  const rows = props.positions.length;
  const bins = props.radiusCount;
  // Bin-major tables: sums are [cumulative, shell, products] x bin x row; counts are bin x row.
  const binSums = createTransientView(graph, `${id}-bin-sums`, 'float32', 3 * bins * rows);
  const binCounts = createTransientView(graph, `${id}-bin-counts`, 'uint32', bins * rows);
  const totalSums = createTransientView(graph, `${id}-total-sums`, 'float32', 3 * bins);
  const totalCounts = createTransientView(graph, `${id}-total-counts`, 'uint32', bins * 2);
  // Keep small marks unscaled so they survive cancellation of large opposite marks. Large marks
  // are scaled by 2^-24 before summation; divide by the category count before restoring the scale.
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
    neighborSum: createTransientView(graph, `${id}-neighbor-mark-sum`, 'float32', 1),
    focusScaledSum: createTransientView(graph, `${id}-focus-scaled-mark-sum`, 'float32', 1),
    neighborScaledSum: createTransientView(graph, `${id}-neighbor-scaled-mark-sum`, 'float32', 1)
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
    mask: inclusion,
    legacyLattice: true
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
let unscaledMark = select(0.0, rowMark, abs(rowMark) <= ${MAXIMUM_UNSCALED_MARK_MAGNITUDE.toExponential(17)});
focusMarks[focusMarksOffset + index] = select(0.0, unscaledMark, isFocusRow); neighborMarks[neighborMarksOffset + index] = select(0.0, unscaledMark, isNeighborRow);`
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
    // Reuse the two row-sized mark buffers after their unscaled sums have been reduced.
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-scaled-mark-terms`,
      operation: 'GPUMarkedPointStatistics',
      variant: 'scaled-mark-terms',
      invocationCount: rows,
      bindings: [
        {name: 'attributes', view: attributes, type: 'f32', access: 'read'},
        {name: 'focusMarks', view: momentTerms.focusSum, type: 'f32', access: 'read_write'},
        {name: 'neighborMarks', view: momentTerms.neighborSum, type: 'f32', access: 'read_write'}
      ],
      body: `let roles = u32(attributes[attributesOffset + index * 3u]);
let rowMark = attributes[attributesOffset + index * 3u + 1u];
var scaledMark = 0.0;
if (abs(rowMark) > ${MAXIMUM_UNSCALED_MARK_MAGNITUDE.toExponential(17)}) {
  scaledMark = ldexp(rowMark, -${LARGE_MARK_SUM_SCALE_EXPONENT});
}
focusMarks[focusMarksOffset + index] = select(0.0, scaledMark, (roles & 1u) != 0u);
neighborMarks[neighborMarksOffset + index] = select(0.0, scaledMark, (roles & 2u) != 0u);`
    }),
    ...[
      new GPUReduction({
        id: `${id}-focus-scaled-mark-reduction`,
        input: momentTerms.focusSum,
        output: momentTotals.focusScaledSum,
        operation: 'sum'
      }),
      new GPUReduction({
        id: `${id}-neighbor-scaled-mark-reduction`,
        input: momentTerms.neighborSum,
        output: momentTotals.neighborScaledSum,
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
        {name: 'focusScaledSum', view: momentTotals.focusScaledSum, type: 'f32', access: 'read'},
        {
          name: 'neighborScaledSum',
          view: momentTotals.neighborScaledSum,
          type: 'f32',
          access: 'read'
        },
        {name: 'moments', view: moments, type: 'f32', access: 'read_write'}
      ],
      declarations: `fn getCategoryMeanMark(unscaledSum: f32, scaledSum: f32, count: f32) -> f32 {
  if (count == 0.0) { return 0.0; }
  // Division may round just beyond the representable mean at the finite f32 endpoints.
  // Clamp before restoring the power-of-two scale, keeping the restored mean finite.
  let maximumScaledMean = ldexp(bitcast<f32>(0x7f7fffffu), -${LARGE_MARK_SUM_SCALE_EXPONENT});
  let scaledMean = clamp(scaledSum / count, -maximumScaledMean, maximumScaledMean);
  return unscaledSum / count + ldexp(scaledMean, ${LARGE_MARK_SUM_SCALE_EXPONENT});
}`,
      body: `let focusTotal = f32(focusCount[focusCountOffset]); let neighborTotal = f32(neighborCount[neighborCountOffset]);
moments[momentsOffset] = focusTotal; moments[momentsOffset + 1u] = neighborTotal;
moments[momentsOffset + 2u] = getCategoryMeanMark(focusSum[focusSumOffset], focusScaledSum[focusScaledSumOffset], focusTotal);
moments[momentsOffset + 3u] = getCategoryMeanMark(neighborSum[neighborSumOffset], neighborScaledSum[neighborScaledSumOffset], neighborTotal);`
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
      declarations: `${getPairStatisticsSharedWGSL(gridSize, true)}
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
      output: totalCounts,
      detectOverflow: true
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
${getFinishWGSL(props, `totalCounts[totalCountsOffset + ${bins}u + index] != 0u`)}`
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

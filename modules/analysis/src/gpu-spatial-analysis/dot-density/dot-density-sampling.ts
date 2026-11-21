// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {snapshotConstructionFields} from '../../analysis-core/kernels/construction-snapshot';

import type {Device} from '@luma.gl/core';
import {
  createTransientView,
  GPUScan,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  createFillNode,
  createWGSLKernelNode,
  createPublishNode,
  type WGSLKernelBinding
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import type {GPUCompactOutput} from '../../analysis-core/kernels/gpu-contributor-types';
import {
  getStorageBindingElementLimit,
  validateCapacity,
  validateGraphOutputsDisjointFromInputs,
  validateGPUStorageBindingByteLength,
  validateCompactOutput,
  validateMaximumCount,
  validateRequirement,
  validateSinglePackedView
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  DOT_DENSITY_CANDIDATE_PURPOSE,
  DOT_DENSITY_REMAINDER_PURPOSE,
  getDotSlotCountLimit
} from './dot-density-constants';
import {
  getPolygonTriangulationIndexCount,
  GPUPolygonTriangulation,
  validatePolygonTriangulationScratch
} from '../polygon-triangulation/gpu-polygon-triangulation';
import {createSegmentSumNode} from '../../analysis-core/kernels/sorted-segment-sums';
import {IS_NAN_BITS_WGSL} from '../../analysis-core/kernels/wgsl-common';
import type {GPUPolygonGeometryFields} from '../../analysis-core/contracts/index';
import {GPU_DOT_DENSITY_PARAMETER_LENGTH} from './dot-density-parameters';
import {PHILOX_WGSL} from '../../analysis-core/kernels/philox-random';
import {getDotTriangleCumulativeAreaNodes} from './dot-density-triangle-cdf';
import {getCoreDefaultBuffer} from '../../analysis-core/contracts/network-topology';
import {
  validateDisjointWritableViews,
  validateGraphChunkBindingRanges
} from '../../analysis-core/kernels/graph-chunk-dispatch';
import {getReadPointWGSL} from '../shared/spatial-wgsl-point-helpers';

/** Logical writes in separate passes may share an aligned prefix, but may never alias. @internal */
export function validateDotSamplingLogicalOutputs(
  id: string,
  outputs: Readonly<Record<string, GraphDataView | undefined>>
): void {
  const fields = Object.entries(outputs).filter((entry): entry is [string, GraphDataView] =>
    Boolean(entry[1])
  );
  for (let firstIndex = 0; firstIndex < fields.length; firstIndex++) {
    const [firstName, first] = fields[firstIndex];
    if (first.length === 0) continue;
    const firstBuffer = first.buffer.defaultBuffer
      ? getCoreDefaultBuffer(first.buffer.defaultBuffer)
      : first.buffer;
    for (let secondIndex = firstIndex + 1; secondIndex < fields.length; secondIndex++) {
      const [secondName, second] = fields[secondIndex];
      if (second.length === 0) continue;
      const secondBuffer = second.buffer.defaultBuffer
        ? getCoreDefaultBuffer(second.buffer.defaultBuffer)
        : second.buffer;
      validateRequirement(
        id,
        firstBuffer !== secondBuffer ||
          first.byteOffset + first.length * first.rowByteLength <= second.byteOffset ||
          second.byteOffset + second.length * second.rowByteLength <= first.byteOffset,
        `${firstName} and ${secondName} logical byte ranges must not overlap`
      );
    }
  }
}

/** Feature ID written past the published count. */
const NO_FEATURE = '0xffffffffu';
/** Words per dot in the transient dot table: slot, rank, first triangle and end triangle. */
const DOT_WORDS = 4;

/**
 * Largest dot capacity (`output.positions.length`) of `GPUDotDensity` and the shared dot sampler on
 * `device`: the per-dot slot table holds 16 bytes per dot in one storage binding, so 8388608 on a
 * default device (128 MiB bindings).
 *
 * @param device Device whose `maxStorageBufferBindingSize` applies.
 */
export function getGPUDotDensityMaximumCapacity(device: Device): number {
  return getStorageBindingElementLimit(device, DOT_WORDS * 4);
}

/** Lanes per feature of the mask-maximum pass. */
const MASK_MAXIMUM_WORKGROUP_SIZE = 256;
/** Default mask-rejection attempts per dot. */
export const DEFAULT_DOT_MAXIMUM_ATTEMPTS = 32;
/** One workgroup per normalized feature uses 256 invocations within the uint32 dispatch count. */
export const DOT_SAMPLING_MAXIMUM_NORMALIZED_FEATURE_COUNT = Math.floor(
  0xffffffff / MASK_MAXIMUM_WORKGROUP_SIZE
);

/** Optional dasymetric constraint shared by {@link GPUDotDensity} and {@link GPURandomPointsInPolygon}. */
export type GPUDotDensityMask = {
  /**
   * Packed row-major weights in `[0, 1]`, `width * height` rows, placed by the `maskPlacement`
   * parameter. A candidate is kept with probability equal to the weight of its (nearest) cell;
   * outside the raster or on NaN the weight is 0.
   *
   * Weights are absolute acceptance probabilities, not per-feature relative densities. A feature
   * with mean weight `w` places a dot with probability `1 - (1 - w)^maximumAttempts`, so features
   * whose weights are all small (for example mostly water) lose dots and report them in
   * `status.failedCount`, which biases counts downward there. To conserve counts, rescale the
   * weights so each feature's maximum is near 1, raise `maximumAttempts`, or set
   * `normalizeFeatures`.
   */
  weights: GraphDataView<'float32'>;
  /**
   * Compile-time. Rescales the acceptance probability of every feature by the largest mask weight
   * over the mask cells under the feature's bounding box: a candidate is kept when its draw is
   * below `weight / featureMaximum`. Dots are still distributed in proportion to the weights inside
   * each feature, but a feature whose weights are all small no longer loses dots (only a feature
   * whose box covers no positive weight does, and it reports them in `status.failedCount`).
   * Costs one extra pass over the mask cells of every feature box. Defaults to `false`, which
   * keeps the weights as absolute acceptance probabilities.
   */
  normalizeFeatures?: boolean;
  /** Mask width in cells. Compile-time. */
  width: number;
  /** Mask height in cells. Compile-time. */
  height: number;
};

/** Caller-owned outputs shared by {@link GPUDotDensity} and {@link GPURandomPointsInPolygon}. */
export type GPUDotDensityOutput = {
  /**
   * Dot positions, capacity rows. Rows past `dots.count` are NaN. Zero-area features get no slots.
   * A dot of a feature that failed to triangulate, or whose mask attempts all failed, is NaN
   * (and counted in `status.failedCount`), so it occupies its slot but is not drawn.
   */
  positions: GraphDataView<'float32x2'>;
  /**
   * Compact dot list: `ids` holds the feature row of each dot (capacity rows, `0xffffffff` past the
   * count), `count = min(total, capacity)` is safe as an instance count, and `overflow` is 1 when
   * the retained total exceeds final capacity. Optional candidateOverflow reports slot clamping;
   * requiredCount counts retained requests and is then a lower bound.
   */
  dots: GPUCompactOutput;
  /** Optional category per dot (dot density only), capacity rows. */
  categories?: GraphDataView<'uint32'>;
  /** Optional per-slot dot counts, one row per slot. */
  slotCounts?: GraphDataView<'uint32'>;
  /**
   * Per-slot pre-clamp starts, without a terminal. Read dot rows
   * `[min(start, count), min(start + slotCounts[slot], count))`; double bounds for flat XY.
   */
  slotStarts?: GraphDataView<'uint32'>;
  /** Optional completeness scalars. */
  status?: {
    /**
     * Optional one-row count of dots that could not be placed: dots of features whose polygons
     * failed to triangulate (or overflowed the triangulation work budget) or whose area is zero or
     * not finite, plus dots whose mask attempts all failed. Zero-area features contribute their
     * clamped requested count here, but consume no output slots. Zero means every dot was placed.
     */
    failedCount?: GraphDataView<'uint32'>;
  };
};

/** Captures output descriptors and resolves the slot-start alias once at construction. @internal */
export function snapshotDotSamplingOutput(
  id: string,
  output: GPUDotDensityOutput
): GPUDotDensityOutput {
  validateRequirement(
    id,
    Boolean(output && typeof output === 'object'),
    'output must be an output descriptor'
  );
  validateRequirement(
    id,
    !('slotOffsets' in output),
    'output.slotOffsets was removed; use output.slotStarts'
  );
  const captured = snapshotConstructionFields(output, [
    'positions',
    'dots',
    'categories',
    'slotCounts',
    'slotStarts',
    'status'
  ]);
  validateRequirement(
    id,
    Boolean(captured.dots && typeof captured.dots === 'object'),
    'output.dots must be a compact output descriptor'
  );
  const dots = snapshotConstructionFields(captured.dots, ['ids', 'status']);
  validateRequirement(
    id,
    Boolean(dots.status && typeof dots.status === 'object'),
    'output.dots.status must provide count and overflow views'
  );
  return Object.freeze({
    ...captured,
    dots: Object.freeze({
      ...dots,
      status: snapshotConstructionFields(dots.status, [
        'count',
        'overflow',
        'candidateOverflow',
        'requiredCount'
      ])
    }),
    status: captured.status && snapshotConstructionFields(captured.status, ['failedCount'])
  });
}

/** Internal configuration of the shared dot sampling graph. @internal */
export type DotSamplingConfig = GPUPolygonGeometryFields & {
  id: string;
  operation: string;
  /** Float values per slot (dot density) or integer counts per feature (random points). */
  source:
    | {kind: 'values'; values: GraphDataView<'float32'>}
    | {kind: 'counts'; counts: GraphDataView<'uint32'>};
  categoryCount: number;
  parameters: GraphDataView<'uint32'>;
  maximumAttempts: number;
  mask?: GPUDotDensityMask;
  output: GPUDotDensityOutput;
};

/** Validates a shared dot sampling configuration and returns the feature count. @internal */
export function validateDotSamplingConfig(config: DotSamplingConfig): number {
  const {id, output, mask} = config;
  validateSinglePackedView(config.positions, ['float32x2'], `${id} positions`);
  validateSinglePackedView(config.featureOffsets, ['uint32'], `${id} featureOffsets`);
  validateSinglePackedView(config.polygonOffsets, ['uint32'], `${id} polygonOffsets`);
  validateSinglePackedView(config.ringOffsets, ['uint32'], `${id} ringOffsets`);
  validateCapacity(id, 'polygonOffsets rows', config.polygonOffsets.length, 1);
  validateCapacity(id, 'ringOffsets rows', config.ringOffsets.length, 1);
  validateRequirement(
    id,
    config.polygonOffsets.length > 1 || config.positions.length === 0,
    'positions must be empty when polygonOffsets describes no polygons'
  );
  const featureCount = config.featureOffsets.length - 1;
  if (featureCount < 1) {
    throw new Error(`${id} featureOffsets must describe at least one feature`);
  }
  validateCapacity(id, 'categoryCount', config.categoryCount);
  const slotCount = featureCount * config.categoryCount;
  validateCapacity(id, 'featureCount * categoryCount', slotCount);
  if (config.source.kind === 'values') {
    validateSinglePackedView(config.source.values, ['float32'], `${id} values`);
    if (config.source.values.length !== slotCount) {
      throw new Error(`${id} values must have featureCount * categoryCount rows`);
    }
  } else {
    validateSinglePackedView(config.source.counts, ['uint32'], `${id} counts`);
    if (config.source.counts.length !== featureCount) {
      throw new Error(`${id} counts must have one row per feature`);
    }
  }
  validateSinglePackedView(config.parameters, ['uint32'], `${id} parameters`);
  if (config.parameters.length < GPU_DOT_DENSITY_PARAMETER_LENGTH) {
    throw new Error(
      `${id} parameters must contain ${GPU_DOT_DENSITY_PARAMETER_LENGTH} uint32 rows`
    );
  }
  validateCapacity(id, 'maximumAttempts', config.maximumAttempts);
  if (mask?.normalizeFeatures) {
    validateMaximumCount(
      id,
      'normalized feature count',
      featureCount,
      DOT_SAMPLING_MAXIMUM_NORMALIZED_FEATURE_COUNT
    );
  }
  if (mask) {
    validateSinglePackedView(mask.weights, ['float32'], `${id} mask.weights`);
    if (
      !Number.isInteger(mask.width) ||
      !Number.isInteger(mask.height) ||
      mask.width < 1 ||
      mask.height < 1 ||
      mask.weights.length !== mask.width * mask.height
    ) {
      throw new Error(`${id} mask.weights must have mask.width * mask.height rows`);
    }
  }
  validateSinglePackedView(output.positions, ['float32x2'], `${id} output.positions`);
  validateCompactOutput(id, output.dots);
  const capacity = output.positions.length;
  validateRequirement(
    id,
    getDotSlotCountLimit(slotCount, capacity) > capacity ||
      Boolean(output.dots.status.candidateOverflow),
    'output.dots.status.candidateOverflow is required when the per-slot limit can hide loss below final capacity'
  );
  if (capacity < 1 || output.dots.ids.length !== capacity) {
    throw new Error(`${id} output.positions and output.dots.ids must share a positive capacity`);
  }
  if (output.categories) {
    validateSinglePackedView(output.categories, ['uint32'], `${id} output.categories`);
    if (output.categories.length !== capacity) {
      throw new Error(`${id} output.categories must have capacity rows`);
    }
  }
  const failedCountView = output.status?.failedCount;
  if (failedCountView) {
    validateSinglePackedView(failedCountView, ['uint32'], `${id} output.status.failedCount`);
    if (failedCountView.length < 1) {
      throw new Error(`${id} output.status.failedCount must contain one uint32 row`);
    }
  }
  for (const [name, view] of [
    ['slotCounts', output.slotCounts],
    ['slotStarts', output.slotStarts]
  ] as const) {
    if (view) {
      validateSinglePackedView(view, ['uint32'], `${id} output.${name}`);
      if (view.length !== slotCount) {
        throw new Error(`${id} output.${name} must have one row per slot`);
      }
    }
  }
  const outputViews = getDotSamplingOutputs(config);
  validateDotSamplingLogicalOutputs(id, outputViews as Record<string, GraphDataView | undefined>);
  validateDisjointWritableViews(id, {
    ids: output.dots.ids,
    categories: output.categories,
    slotCounts: output.slotCounts,
    slotStarts: output.slotStarts
  });
  validateDisjointWritableViews(id, {
    slotCounts: output.slotCounts,
    failedCount: output.status?.failedCount
  });
  validateDisjointWritableViews(id, {
    positions: output.positions,
    failedCount: output.status?.failedCount
  });
  validateDisjointWritableViews(id, {
    count: output.dots.status.count,
    overflow: output.dots.status.overflow,
    candidateOverflow: output.dots.status.candidateOverflow,
    requiredCount: output.dots.status.requiredCount
  });
  validateGraphOutputsDisjointFromInputs(
    id,
    Object.values(outputViews),
    getDotSamplingInputs(config)
  );
  return featureCount;
}

function getDotSamplingOutputs(config: DotSamplingConfig) {
  const {output} = config;
  return {
    'output.positions': output.positions,
    'output.dots.ids': output.dots.ids,
    'output.dots.status.count': output.dots.status.count,
    'output.dots.status.overflow': output.dots.status.overflow,
    'output.dots.status.candidateOverflow': output.dots.status.candidateOverflow,
    'output.dots.status.requiredCount': output.dots.status.requiredCount,
    'output.categories': output.categories,
    'output.status.failedCount': output.status?.failedCount,
    'output.slotCounts': output.slotCounts,
    'output.slotStarts': output.slotStarts
  };
}

function getDotSamplingInputs(config: DotSamplingConfig): GraphDataView[] {
  return [
    config.positions,
    config.featureOffsets,
    config.polygonOffsets,
    config.ringOffsets,
    config.source.kind === 'values' ? config.source.values : config.source.counts,
    config.parameters,
    ...(config.mask ? [config.mask.weights] : [])
  ];
}

/**
 * Builds the shared dot sampling nodes:
 * `triangulation` (GPUPolygonTriangulation) -> `triangle-areas` -> `feature-triangles` (triangle
 * range and validity per feature) -> feature area sums -> `cumulative-areas` (per-feature inclusive
 * prefix of fixed-point areas) -> `clear-failed` -> `counts` (dots per slot) -> slot prefix scan ->
 * `total` -> `locate` (slot, rank and triangle range per dot, feature IDs, categories) ->
 * `sample` (area-weighted triangle, then uniform point in the triangle) -> `publish`.
 *
 * @internal
 */
export function getDotSamplingNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  config: DotSamplingConfig,
  featureCount: number
): GPUCommandNode<Parameters>[] {
  const {id, operation, output, mask, categoryCount} = config;
  validateGraphChunkBindingRanges(graph, id, {
    ...Object.fromEntries(
      getDotSamplingInputs(config).map((view, index) => [`input ${index}`, view])
    ),
    ...getDotSamplingOutputs(config)
  });
  const slotCount = featureCount * categoryCount;
  const capacity = output.positions.length;
  validateMaximumCount(
    id,
    'output.positions rows',
    capacity,
    getGPUDotDensityMaximumCapacity(graph.device)
  );
  const slotLimit = getDotSlotCountLimit(slotCount, capacity);
  const polygonCount = config.polygonOffsets.length - 1;
  const triangleCount = Math.floor(
    getPolygonTriangulationIndexCount(
      config.positions.length,
      config.ringOffsets.length - 1,
      polygonCount
    ) / 3
  );
  // Check all derived bindings, including the nested triangulator's layout and scratch,
  // before declaring any resource. The CDF scans have four uint32 triangle-sized scratch views;
  // their hierarchical summaries are smaller than those views.
  validatePolygonTriangulationScratch(
    graph.device,
    id,
    config.positions.length,
    config.ringOffsets.length - 1,
    polygonCount
  );
  const normalizeFeatures = Boolean(mask?.normalizeFeatures);
  for (const [name, wordCount] of [
    ['triangle indices', Math.max(3 * triangleCount, 1)],
    ['triangle CDF scratch', Math.max(triangleCount, 1)],
    ['cumulative areas', Math.max(triangleCount, 1) + (normalizeFeatures ? featureCount : 0)],
    ['feature offsets', featureCount + 1],
    ['slot scratch', slotCount],
    ['dot slots', capacity * DOT_WORDS],
    ['triangulation polygon layout', 4 * polygonCount + 2],
    ['triangulation ring layout', 2 * (config.ringOffsets.length - 1) + 1]
  ] as const) {
    validateGPUStorageBindingByteLength(
      graph.device,
      id,
      name,
      Math.max(wordCount, 1) * 4,
      'reduce the geometry, category count or output capacity'
    );
  }
  const nodes: GPUCommandNode<Parameters>[] = [];

  const triangleIndices = createTransientView(
    graph,
    `${id}-triangle-indices`,
    'uint32',
    Math.max(3 * triangleCount, 1)
  );
  const polygonValid = createTransientView(graph, `${id}-polygon-valid`, 'uint32', polygonCount);
  const polygonTriangleOffsets = createTransientView(
    graph,
    `${id}-polygon-triangle-offsets`,
    'uint32',
    polygonCount + 1
  );
  const triangleAreas = createTransientView(
    graph,
    `${id}-triangle-areas`,
    'float32',
    Math.max(triangleCount, 1)
  );
  const triangleSlotCount = Math.max(triangleCount, 1);
  // With feature normalization the per-feature maximum mask weight (f32 bits) follows the triangles.
  const cumulativeAreas = createTransientView(
    graph,
    `${id}-cumulative-areas`,
    'uint32',
    triangleSlotCount + (normalizeFeatures ? featureCount : 0)
  );
  const featureTriangleOffsets = createTransientView(
    graph,
    `${id}-feature-triangle-offsets`,
    'uint32',
    featureCount + 1
  );
  const featureValid = createTransientView(graph, `${id}-feature-valid`, 'uint32', featureCount);
  const featureAreas = createTransientView(graph, `${id}-feature-areas`, 'float32', featureCount);
  const slotCounts =
    output.slotCounts ?? createTransientView(graph, `${id}-slot-counts`, 'uint32', slotCount);
  const slotOffsets =
    output.slotStarts ?? createTransientView(graph, `${id}-slot-offsets`, 'uint32', slotCount);
  const clampFlag = createTransientView(graph, `${id}-clamped`, 'uint32', 1);
  const total = createTransientView(graph, `${id}-total`, 'uint32', 1);
  const dotSlots = createTransientView(graph, `${id}-dot-slots`, 'uint32', capacity * DOT_WORDS);
  const failedCount =
    output.status?.failedCount ?? createTransientView(graph, `${id}-failed`, 'uint32', 1);

  // 1. Triangulate every polygon (shell first, later rings are holes, as GeoArrow positions say).
  nodes.push(
    ...new GPUPolygonTriangulation({
      id: `${id}-triangulation`,
      positions: config.positions,
      featureOffsets: config.featureOffsets,
      polygonOffsets: config.polygonOffsets,
      ringOffsets: config.ringOffsets,
      output: {
        indices: triangleIndices,
        valid: polygonValid,
        triangleOffsets: polygonTriangleOffsets
      }
    }).getCommandNodes(graph)
  );

  // 2. Area of every triangle slot; padding slots are degenerate and have area 0.
  nodes.push(
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-triangle-areas`,
      operation,
      variant: 'triangle-areas',
      bindings: [
        {name: 'polygonPositions', view: config.positions, type: 'f32', access: 'read'},
        {name: 'triangleIndices', view: triangleIndices, type: 'u32', access: 'read'},
        {name: 'polygonTriangleOffsets', view: polygonTriangleOffsets, type: 'u32', access: 'read'},
        {name: 'triangleAreas', view: triangleAreas, type: 'f32', access: 'read_write'}
      ],
      invocationCount: Math.max(triangleCount, 1),
      declarations: /* wgsl */ `
${getReadPointWGSL('readVertex', 'polygonPositions')}`,
      body: /* wgsl */ `
  var area = 0.0;
  if (index < polygonTriangleOffsets[polygonTriangleOffsetsOffset + ${polygonCount}u]) {
    let base = triangleIndicesOffset + 3u * index;
    let firstIndex = triangleIndices[base];
    let secondIndex = triangleIndices[base + 1u];
    let thirdIndex = triangleIndices[base + 2u];
    if (firstIndex == secondIndex && firstIndex == thirdIndex) {
      triangleAreas[triangleAreasOffset + index] = 0.0;
      return;
    }
    let first = readVertex(firstIndex);
    // Edge vectors from one vertex keep the cross product accurate far from the origin.
    let edgeOne = readVertex(triangleIndices[base + 1u]) - first;
    let edgeTwo = readVertex(triangleIndices[base + 2u]) - first;
    area = 0.5 * abs(edgeOne.x * edgeTwo.y - edgeOne.y * edgeTwo.x);
  }
  triangleAreas[triangleAreasOffset + index] = area;`
    })
  );

  // 3. Triangle range [offsets[f], offsets[f + 1]) of every feature and whether all of its polygons
  // triangulated. GPU-computed offsets exclude degenerate rings and empty polygons, preserving
  // the same area-reduction order as the geometry with those entries removed.
  nodes.push(
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-feature-triangles`,
      operation,
      variant: 'feature-triangles',
      bindings: [
        {name: 'featureOffsets', view: config.featureOffsets, type: 'u32', access: 'read'},
        {name: 'polygonTriangleOffsets', view: polygonTriangleOffsets, type: 'u32', access: 'read'},
        {name: 'polygonValid', view: polygonValid, type: 'u32', access: 'read'},
        {
          name: 'featureTriangleOffsets',
          view: featureTriangleOffsets,
          type: 'u32',
          access: 'read_write'
        },
        {name: 'featureValid', view: featureValid, type: 'u32', access: 'read_write'}
      ],
      invocationCount: featureCount + 1,
      declarations: /* wgsl */ `
const FEATURE_COUNT: u32 = ${featureCount}u;
const POLYGON_COUNT: u32 = ${polygonCount}u;
const TRIANGLE_COUNT: u32 = ${triangleCount}u;
fn getTriangleStart(polygon: u32) -> u32 {
  return min(polygonTriangleOffsets[polygonTriangleOffsetsOffset + polygon], TRIANGLE_COUNT);
}`,
      body: /* wgsl */ `
  let firstPolygon = min(featureOffsets[featureOffsetsOffset + index], POLYGON_COUNT);
  featureTriangleOffsets[featureTriangleOffsetsOffset + index] = getTriangleStart(firstPolygon);
  if (index < FEATURE_COUNT) {
    let endPolygon = min(featureOffsets[featureOffsetsOffset + index + 1u], POLYGON_COUNT);
    var valid = 1u;
    for (var polygon = firstPolygon; polygon < endPolygon; polygon = polygon + 1u) {
      valid = valid & polygonValid[polygonValidOffset + polygon];
    }
    featureValid[featureValidOffset + index] = valid;
  }`
    })
  );

  // 4. Total area per feature (deterministic segmented sum).
  nodes.push(
    createSegmentSumNode<Parameters>(graph, {
      id: `${id}-feature-areas`,
      operation,
      segmentCount: featureCount,
      input: triangleAreas,
      segmentOffsets: featureTriangleOffsets,
      output: featureAreas
    })
  );

  // 5. Parallel inclusive fixed-point CDF, restarting at each feature's contiguous slice.
  nodes.push(
    ...getDotTriangleCumulativeAreaNodes(graph, {
      id: `${id}-cumulative-areas`,
      operation,
      triangleAreas,
      featureTriangleOffsets,
      featureValid,
      featureAreas,
      cumulativeAreas
    })
  );

  // 6. Optional per-feature maximum mask weight over the cells under the feature's bounding box,
  // one 256-lane workgroup per feature, stored after the triangles in `cumulativeAreas`.
  if (mask && normalizeFeatures) {
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-feature-mask-maximum`,
        operation,
        variant: 'feature-mask-maximum',
        bindings: [
          {name: 'polygonPositions', view: config.positions, type: 'f32', access: 'read'},
          {name: 'featureOffsets', view: config.featureOffsets, type: 'u32', access: 'read'},
          {name: 'polygonOffsets', view: config.polygonOffsets, type: 'u32', access: 'read'},
          {name: 'ringOffsets', view: config.ringOffsets, type: 'u32', access: 'read'},
          {name: 'parameters', view: config.parameters, type: 'u32', access: 'read'},
          {name: 'maskWeights', view: mask.weights, type: 'f32', access: 'read'},
          {name: 'cumulativeAreas', view: cumulativeAreas, type: 'u32', access: 'read_write'}
        ],
        invocationCount: featureCount * MASK_MAXIMUM_WORKGROUP_SIZE,
        workgroupSize: MASK_MAXIMUM_WORKGROUP_SIZE,
        guardIndex: false,
        declarations: /* wgsl */ `${IS_NAN_BITS_WGSL}
${getReadPointWGSL('readVertex', 'polygonPositions')}
const WORKGROUP_SIZE: u32 = ${MASK_MAXIMUM_WORKGROUP_SIZE}u;
const MASK_WIDTH: u32 = ${mask.width}u;
const MASK_HEIGHT: u32 = ${mask.height}u;
const POLYGON_COUNT: u32 = ${polygonCount}u;
const RING_COUNT: u32 = ${config.ringOffsets.length - 1}u;
const VERTEX_COUNT: u32 = ${config.positions.length}u;
const TRIANGLE_SLOTS: u32 = ${triangleSlotCount}u;
var<workgroup> partialMinimum: array<vec2<f32>, ${MASK_MAXIMUM_WORKGROUP_SIZE}>;
var<workgroup> partialMaximum: array<vec2<f32>, ${MASK_MAXIMUM_WORKGROUP_SIZE}>;
var<workgroup> partialWeight: array<f32, ${MASK_MAXIMUM_WORKGROUP_SIZE}>;`,
        body: /* wgsl */ `
  let lane = localInvocationIndex;
  // Keep the feature independent of localInvocationIndex in WGSL's uniformity analysis.
  // Dividing index by WORKGROUP_SIZE is mathematically equivalent, but remains non-uniform.
  let feature = workgroupIndex;
  // A bounded multidimensional dispatch may pad whole workgroups. This guard is uniform.
  if (feature >= arrayLength(&featureOffsets) - featureOffsetsOffset - 1u) { return; }
  // Vertex range of the feature: first vertex of its first ring to the end of its last polygon.
  let firstPolygon = min(featureOffsets[featureOffsetsOffset + feature], POLYGON_COUNT);
  let endPolygon = min(featureOffsets[featureOffsetsOffset + feature + 1u], POLYGON_COUNT);
  let firstVertex = min(ringOffsets[ringOffsetsOffset + min(polygonOffsets[polygonOffsetsOffset + firstPolygon], RING_COUNT)], VERTEX_COUNT);
  let endVertex = min(ringOffsets[ringOffsetsOffset + min(polygonOffsets[polygonOffsetsOffset + endPolygon], RING_COUNT)], VERTEX_COUNT);
  var minimum = vec2<f32>(3.4e38);
  var maximum = vec2<f32>(-3.4e38);
  for (var vertex = firstVertex + lane; vertex < endVertex; vertex = vertex + WORKGROUP_SIZE) {
    let point = readVertex(vertex);
    if (!isNanBits(point.x) && !isNanBits(point.y) && abs(point.x) < 3.4e38 && abs(point.y) < 3.4e38) {
      minimum = min(minimum, point);
      maximum = max(maximum, point);
    }
  }
  partialMinimum[lane] = minimum;
  partialMaximum[lane] = maximum;
  workgroupBarrier();
  for (var stride = WORKGROUP_SIZE / 2u; stride > 0u; stride = stride >> 1u) {
    if (lane < stride) {
      partialMinimum[lane] = min(partialMinimum[lane], partialMinimum[lane + stride]);
      partialMaximum[lane] = max(partialMaximum[lane], partialMaximum[lane + stride]);
    }
    workgroupBarrier();
  }
  // Cell box of the feature (one cell of margin absorbs rounding of sampled points).
  let origin = vec2<f32>(bitcast<f32>(parameters[parametersOffset + 2u]), bitcast<f32>(parameters[parametersOffset + 3u]));
  let cell = vec2<f32>(bitcast<f32>(parameters[parametersOffset + 4u]), bitcast<f32>(parameters[parametersOffset + 5u]));
  let firstCorner = (partialMinimum[0] - origin) / cell;
  let oppositeCorner = (partialMaximum[0] - origin) / cell;
  let low = min(firstCorner, oppositeCorner);
  let high = max(firstCorner, oppositeCorner);
  var firstColumn = 0;
  var lastColumn = -1;
  var firstRow = 0;
  var lastRow = -1;
  if (partialMinimum[0].x <= partialMaximum[0].x && !isNanBits(low.x) && !isNanBits(low.y) &&
      !isNanBits(high.x) && !isNanBits(high.y) && high.x >= 0.0 && high.y >= 0.0 &&
      low.x < f32(MASK_WIDTH) && low.y < f32(MASK_HEIGHT)) {
    firstColumn = max(i32(floor(max(low.x, 0.0))) - 1, 0);
    lastColumn = min(i32(floor(min(high.x, f32(MASK_WIDTH)))) + 1, i32(MASK_WIDTH) - 1);
    firstRow = max(i32(floor(max(low.y, 0.0))) - 1, 0);
    lastRow = min(i32(floor(min(high.y, f32(MASK_HEIGHT)))) + 1, i32(MASK_HEIGHT) - 1);
  }
  let columnCount = u32(max(lastColumn - firstColumn + 1, 0));
  let cellCount = columnCount * u32(max(lastRow - firstRow + 1, 0));
  var best = 0.0;
  for (var item = lane; item < cellCount; item = item + WORKGROUP_SIZE) {
    let column = u32(firstColumn) + item % columnCount;
    let row = u32(firstRow) + item / columnCount;
    let weight = maskWeights[maskWeightsOffset + row * MASK_WIDTH + column];
    if (!isNanBits(weight)) {
      best = max(best, weight);
    }
  }
  partialWeight[lane] = best;
  workgroupBarrier();
  for (var stride = WORKGROUP_SIZE / 2u; stride > 0u; stride = stride >> 1u) {
    if (lane < stride) {
      partialWeight[lane] = max(partialWeight[lane], partialWeight[lane + stride]);
    }
    workgroupBarrier();
  }
  if (lane == 0u) {
    cumulativeAreas[cumulativeAreasOffset + TRIANGLE_SLOTS + feature] = bitcast<u32>(partialWeight[0]);
  }`
      })
    );
  }

  nodes.push(
    createFillNode<Parameters>(graph, {
      id: `${id}-clear-failed`,
      operation,
      view: failedCount,
      type: 'u32',
      value: '0u',
      componentCount: 1
    })
  );

  // 7. Dots per slot: ceil(value * dotsPerUnit - u) (dot density) or the integer count.
  nodes.push(
    createFillNode<Parameters>(graph, {
      id: `${id}-clear-clamped`,
      operation,
      view: clampFlag,
      type: 'u32',
      value: '0u'
    })
  );
  const sourceBinding: WGSLKernelBinding =
    config.source.kind === 'values'
      ? {
          name: 'sourceValues',
          view: config.source.values,
          type: 'f32',
          access: 'read'
        }
      : {
          name: 'sourceCounts',
          view: config.source.counts,
          type: 'u32',
          access: 'read'
        };
  nodes.push(
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-counts`,
      operation,
      variant: 'counts',
      bindings: [
        sourceBinding,
        {name: 'featureAreas', view: featureAreas, type: 'f32', access: 'read'},
        {name: 'failed', view: failedCount, type: 'atomic<u32>', access: 'read_write'},
        {
          name: 'parameters',
          view: config.parameters,
          type: 'u32',
          access: 'read'
        },
        {
          name: 'slotCounts',
          view: slotCounts,
          type: 'u32',
          access: 'read_write'
        },
        {
          name: 'clampFlag',
          view: clampFlag,
          type: 'atomic<u32>',
          access: 'read_write'
        }
      ],
      invocationCount: slotCount,
      declarations: `${PHILOX_WGSL}
${IS_NAN_BITS_WGSL}
const SLOT_LIMIT: u32 = ${slotLimit}u;
const CATEGORY_COUNT: u32 = ${categoryCount}u;`,
      body:
        config.source.kind === 'values'
          ? /* wgsl */ `
  let seed = parameters[parametersOffset];
  let dotsPerUnit = bitcast<f32>(parameters[parametersOffset + 1u]);
  let scaled = sourceValues[sourceValuesOffset + index] * dotsPerUnit;
  var dots = 0u;
  var clamped = false;
  if (!isNanBits(scaled) && scaled > 0.0) {
    let remainder = philoxUnitFloat(
      philox4x32(vec4<u32>(index, 0u, 0u, 0u), vec2<u32>(seed, ${DOT_DENSITY_REMAINDER_PURPOSE}u)).x
    );
    // Subtract only the integral part: subtracting the uniform from a large integer can
    // round down before ceil. Preserve positive infinity's existing clamp path.
    var ceiling = scaled;
    if ((bitcast<u32>(scaled) & 0x7f800000u) != 0x7f800000u) {
      let integral = floor(scaled);
      ceiling = integral + select(0.0, 1.0, remainder < scaled - integral);
    }
    if (ceiling > f32(SLOT_LIMIT)) {
      dots = SLOT_LIMIT;
      clamped = true;
    } else if (ceiling > 0.0) {
      dots = u32(ceiling);
      if (dots > SLOT_LIMIT) {
        dots = SLOT_LIMIT;
        clamped = true;
      }
    }
  }
  if (featureAreas[featureAreasOffset + index / CATEGORY_COUNT] == 0.0) {
    atomicAdd(&failed[failedOffset], dots);
    dots = 0u;
  }
  slotCounts[slotCountsOffset + index] = dots;
  if (clamped) {
    atomicStore(&clampFlag[clampFlagOffset], 1u);
  }`
          : /* wgsl */ `
  _ = parameters[parametersOffset];
  var dots = sourceCounts[sourceCountsOffset + index];
  if (dots > SLOT_LIMIT) {
    dots = SLOT_LIMIT;
    atomicStore(&clampFlag[clampFlagOffset], 1u);
  }
  if (featureAreas[featureAreasOffset + index / CATEGORY_COUNT] == 0.0) {
    atomicAdd(&failed[failedOffset], dots);
    dots = 0u;
  }
  slotCounts[slotCountsOffset + index] = dots;`
    })
  );

  // 8. Exclusive offsets and the total.
  nodes.push(
    ...new GPUScan({
      id: `${id}-slot-scan`,
      input: slotCounts,
      output: slotOffsets,
      mode: 'exclusive'
    }).getCommandNodes(graph)
  );
  nodes.push(
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-total`,
      operation,
      variant: 'total',
      bindings: [
        {name: 'slotCounts', view: slotCounts, type: 'u32', access: 'read'},
        {name: 'slotOffsets', view: slotOffsets, type: 'u32', access: 'read'},
        {name: 'totalOut', view: total, type: 'u32', access: 'read_write'}
      ],
      invocationCount: 1,
      body: `let last = ${slotCount - 1}u;
  totalOut[totalOutOffset] = slotOffsets[slotOffsetsOffset + last] + slotCounts[slotCountsOffset + last];`
    })
  );

  // 9. Slot, rank and triangle range of every published dot (binary search on the inclusive prefix).
  const locateBindings: WGSLKernelBinding[] = [
    {name: 'slotCounts', view: slotCounts, type: 'u32', access: 'read'},
    {name: 'slotOffsets', view: slotOffsets, type: 'u32', access: 'read'},
    {name: 'total', view: total, type: 'u32', access: 'read'},
    {name: 'featureTriangleOffsets', view: featureTriangleOffsets, type: 'u32', access: 'read'},
    {name: 'dotSlots', view: dotSlots, type: 'u32', access: 'read_write'},
    {
      name: 'featureIds',
      view: output.dots.ids,
      type: 'u32',
      access: 'read_write'
    }
  ];
  if (output.categories) {
    locateBindings.push({
      name: 'categories',
      view: output.categories,
      type: 'u32',
      access: 'read_write'
    });
  }
  nodes.push(
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-locate`,
      operation,
      variant: 'locate',
      bindings: locateBindings,
      invocationCount: capacity,
      declarations: `const SLOT_COUNT: u32 = ${slotCount}u;
const CATEGORY_COUNT: u32 = ${categoryCount}u;`,
      body: /* wgsl */ `
  let count = min(total[totalOffset], ${capacity}u);
  let table = dotSlotsOffset + ${DOT_WORDS}u * index;
  if (index >= count) {
    dotSlots[table] = ${NO_FEATURE};
    dotSlots[table + 1u] = 0u;
    dotSlots[table + 2u] = 0u;
    dotSlots[table + 3u] = 0u;
    featureIds[featureIdsOffset + index] = ${NO_FEATURE};
    ${output.categories ? `categories[categoriesOffset + index] = ${NO_FEATURE};` : ''}
    return;
  }
  // First slot whose inclusive prefix exceeds the dot index; empty slots never match.
  var low = 0u;
  var high = SLOT_COUNT - 1u;
  while (low < high) {
    let middle = (low + high) / 2u;
    if (slotOffsets[slotOffsetsOffset + middle] + slotCounts[slotCountsOffset + middle] > index) {
      high = middle;
    } else {
      low = middle + 1u;
    }
  }
  let feature = low / CATEGORY_COUNT;
  dotSlots[table] = low;
  dotSlots[table + 1u] = index - slotOffsets[slotOffsetsOffset + low];
  dotSlots[table + 2u] = featureTriangleOffsets[featureTriangleOffsetsOffset + feature];
  dotSlots[table + 3u] = featureTriangleOffsets[featureTriangleOffsetsOffset + feature + 1u];
  featureIds[featureIdsOffset + index] = feature;
  ${output.categories ? 'categories[categoriesOffset + index] = low % CATEGORY_COUNT;' : ''}`
    })
  );

  // 10. Area-weighted triangle, then a uniform point inside it.
  const sampleBindings: WGSLKernelBinding[] = [
    {
      name: 'polygonPositions',
      view: config.positions,
      type: 'f32',
      access: 'read'
    },
    {name: 'triangleIndices', view: triangleIndices, type: 'u32', access: 'read'},
    {name: 'cumulativeAreas', view: cumulativeAreas, type: 'u32', access: 'read'},
    {name: 'dotSlots', view: dotSlots, type: 'u32', access: 'read'},
    {
      name: 'parameters',
      view: config.parameters,
      type: 'u32',
      access: 'read'
    },
    {
      name: 'positions',
      view: output.positions,
      type: 'f32',
      access: 'read_write'
    },
    {
      name: 'failed',
      view: failedCount,
      type: 'atomic<u32>',
      access: 'read_write'
    }
  ];
  if (mask) {
    sampleBindings.push({
      name: 'maskWeights',
      view: mask.weights,
      type: 'f32',
      access: 'read'
    });
  }
  nodes.push(
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-sample`,
      operation,
      variant: mask ? 'sample-masked' : 'sample',
      bindings: sampleBindings,
      invocationCount: capacity,
      declarations: `${PHILOX_WGSL}
const CATEGORY_COUNT: u32 = ${categoryCount}u;
// Without a mask the first candidate is always kept; with one, candidates are rejected by weight.
const MAXIMUM_ATTEMPTS: u32 = ${mask ? config.maximumAttempts : 1}u;
${getReadPointWGSL('readVertex', 'polygonPositions')}
// Uniform point of a triangle: r1 = sqrt(u1), P = (1 - r1) A + r1 (1 - u2) B + r1 u2 C, written
// relative to A (the same point) for accuracy far from the origin.
fn getTrianglePoint(triangle: u32, u1: f32, u2: f32) -> vec2<f32> {
  let base = triangleIndicesOffset + 3u * triangle;
  let a = readVertex(triangleIndices[base]);
  let edgeB = readVertex(triangleIndices[base + 1u]) - a;
  let edgeC = readVertex(triangleIndices[base + 2u]) - a;
  return a + sqrt(u1) * ((1.0 - u2) * edgeB + u2 * edgeC);
}
${
  mask
    ? `${IS_NAN_BITS_WGSL}
const MASK_WIDTH: u32 = ${mask.width}u;
const MASK_HEIGHT: u32 = ${mask.height}u;
fn getMaskWeight(point: vec2<f32>) -> f32 {
  let origin = vec2<f32>(bitcast<f32>(parameters[parametersOffset + 2u]), bitcast<f32>(parameters[parametersOffset + 3u]));
  let cell = vec2<f32>(bitcast<f32>(parameters[parametersOffset + 4u]), bitcast<f32>(parameters[parametersOffset + 5u]));
  let local = (point - origin) / cell;
  if (isNanBits(local.x) || isNanBits(local.y) || local.x < 0.0 || local.y < 0.0 ||
      local.x >= f32(MASK_WIDTH) || local.y >= f32(MASK_HEIGHT)) {
    return 0.0;
  }
  let weight = maskWeights[maskWeightsOffset + u32(local.y) * MASK_WIDTH + u32(local.x)];
  return select(weight, 0.0, isNanBits(weight));
}`
    : ''
}`,
      body: /* wgsl */ `
  let table = dotSlotsOffset + ${DOT_WORDS}u * index;
  let slot = dotSlots[table];
  var nanBits = 0x7fc00000u;
  var position = vec2<f32>(bitcast<f32>(nanBits));
  if (slot != ${NO_FEATURE}) {
    let rank = dotSlots[table + 1u];
    let firstTriangle = dotSlots[table + 2u];
    let endTriangle = dotSlots[table + 3u];
    let seed = parameters[parametersOffset];
    // Fixed-point area of the whole feature; 0 when it failed to triangulate or has no area.
    var featureArea = 0u;
    if (endTriangle > firstTriangle) {
      featureArea = cumulativeAreas[cumulativeAreasOffset + endTriangle - 1u];
    }
    ${normalizeFeatures ? `let featureMaskMaximum = bitcast<f32>(cumulativeAreas[cumulativeAreasOffset + ${triangleSlotCount}u + slot / CATEGORY_COUNT]);` : ''}
    var accepted = false;
    if (featureArea > 0u) {
      for (var attempt = 0u; attempt < MAXIMUM_ATTEMPTS; attempt = attempt + 1u) {
        let random = philox4x32(vec4<u32>(slot, rank, attempt, 0u), vec2<u32>(seed, ${DOT_DENSITY_CANDIDATE_PURPOSE}u));
        let areaTarget = min(u32(philoxUnitFloat(random.x) * f32(featureArea)), featureArea - 1u);
        // First triangle whose cumulative area exceeds the target; zero-area triangles never match.
        var low = firstTriangle;
        var high = endTriangle - 1u;
        while (low < high) {
          let middle = (low + high) / 2u;
          if (cumulativeAreas[cumulativeAreasOffset + middle] > areaTarget) {
            high = middle;
          } else {
            low = middle + 1u;
          }
        }
        let candidate = getTrianglePoint(low, philoxUnitFloat(random.y), philoxUnitFloat(random.z));
        ${mask ? `accepted = philoxUnitFloat(random.w)${normalizeFeatures ? ' * featureMaskMaximum' : ''} < getMaskWeight(candidate);` : 'accepted = true;'}
        if (accepted) {
          position = candidate;
          break;
        }
      }
    }
    if (!accepted) {
      atomicAdd(&failed[failedOffset], 1u);
    }
  }
  positions[positionsOffset + 2u * index] = position.x;
  positions[positionsOffset + 2u * index + 1u] = position.y;`
    })
  );

  // 11. Retained total, final capacity overflow and separate slot-clamping diagnostics.
  nodes.push(
    createPublishNode<Parameters>(graph, {
      id: `${id}-publish`,
      operation,
      requiredCount: total,
      output: output.dots,
      candidateOverflowSources: output.dots.status.candidateOverflow ? [clampFlag] : []
    })
  );
  return nodes;
}

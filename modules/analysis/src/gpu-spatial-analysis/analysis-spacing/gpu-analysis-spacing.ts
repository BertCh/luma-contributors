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
  resolveGPUSpatialContext,
  type GPUSpatialContext,
  type GPUCandidateStatusPort,
  type GPUOverflowStatusPort,
  type GPUInvalidStatusPort,
  GPU_EARTH_MEAN_RADIUS_METERS,
  validateCapacity,
  validateCommandNodeBudget,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validateGPUStorageBindingByteLength,
  validateMaximumCount,
  validateRequirement,
  validateViewLength
} from '../../analysis-core/index';
import {validateSinglePackedView} from '../../analysis-core/kernels/gpu-contributor-utils';
import {GPU_NEIGHBOR_SEARCH_GREAT_CIRCLE_SUPPORT} from '../neighbor-search/index';
import type {GPUAnalysisSpacingStatistics} from '../ontology/analysis-experience/index';
import {
  GPU_ANALYSIS_DEFAULT_NEIGHBOR_RANK,
  GPU_ANALYSIS_MAXIMUM_NEIGHBOR_RANK,
  getGPUAnalysisExtentSizes
} from '../ontology/analysis-experience/index';
import {createAnalysisSpacingStatisticsNodes} from './analysis-spacing-statistics-nodes';

/** One million keeps row scratch bounded and counts exactly representable in float32. */
export const GPU_ANALYSIS_SPACING_MAXIMUM_POINT_COUNT = 1_000_000;
/** 32 summary/status words followed by at most 512 k-distance quantiles: 2176 bytes. */
export const GPU_ANALYSIS_SPACING_SUMMARY_LENGTH = 544;
/** 128 MiB, WebGPU's guaranteed maximum storage binding size, in 32-bit slots. */
const MAXIMUM_NEIGHBOR_CAPACITY = 33_554_432;
/** Grid build cell offsets must fit the guaranteed storage binding size. */
const MAXIMUM_GRID_CELL_COUNT = 1_048_576;

/**
 * Caller-owned packed words. Integer words: 0 count, 1 invalidCount, 2 overflow,
 * 3 requiredCount (full-row search slots), 4 duplicateLocationCount, 5 neighborRank,
 * 6 curve length, 7 layout version, 23 aboveUpperFenceCount, 25 distinct count,
 * 26 geographic flag, 27 candidateOverflow (neighbour CSR scratch). All other populated words are float32 bits. Words 8..11 are
 * extent, 12..13 centre, 14 standard distance, 15 median centre distance, 16..22
 * nearest minimum/Q1/median/Q3/mean/maximum/fence, 24 distinct-location median,
 * 28..29 the planar centre relative to the extent minimum (zero for geography),
 * 32..543 the curve. Reserved words are zero. Empty input has count zero.
 */
export type GPUAnalysisSpacingOutput = {summary: GraphDataView<'uint32'>};

export type GPUAnalysisSpacingProps = {
  id?: string;
  positions: GraphDataView<'float32x2'>;
  /** Nonzero keeps a row. Nonfinite coordinates and geographic latitudes outside ±90 are invalid. */
  mask?: GraphDataView<'uint32'>;
  /** Stable caller identities break duplicate-representative ties; uniqueness is caller-declared. */
  identities?: GraphDataView<'uint32'>;
  /** Required. Geographic results use the CPU reference's mean Earth sphere, in metres. */
  spatialContext: GPUSpatialContext;
  neighborRank?: number;
  gridSize?: readonly [number, number];
  /** Slots in the full k-neighbour CSR. Defaults to rows * rank. candidateOverflow refuses decoding. */
  neighborCapacity?: number;
  output: GPUAnalysisSpacingOutput;
};

/**
 * Full-population spacing through graph-owned sort, scan, reduction and indexed neighbour search.
 * GPUNeighborSearch composes the published GPUGridIndex; no CPU position or status readback occurs.
 * Distinct locations get a second indexed k=1 search, so duplicate-heavy data retains its spacing.
 *
 * Per-frame: position, mask and identity contents. Compile-time: context, lengths, rank, grid
 * and capacities. Input positions are single packed f32 pairs; geographic precision is bounded
 * by their quantization (use a declared planar local frame for sub-metre work). No sampling.
 * All scratch belongs to the graph; destroying its compiled form releases resources. Callers
 * read only output.summary once and pass its bytes to decodeGPUAnalysisSpacingStatistics.
 *
 * Reductions use the published parallel tree. Extreme planar magnitudes whose squared distances
 * exceed f32 are rejected at decoding rather than published as usable defaults. Geographic
 * covering arcs follow the CPU rule; gaps indistinguishable at f32 precision may choose a tie.
 */
export class GPUAnalysisSpacing implements GPUCommandNodeProducer {
  readonly id: string;
  readonly props: GPUAnalysisSpacingProps;
  readonly spatialContext: GPUSpatialContext;
  readonly neighborRank: number;
  readonly neighborCapacity: number;
  readonly gridSize: readonly [number, number];
  /** Canonical status ports alias words in the same caller-owned bounded summary. */
  readonly output: GPUAnalysisSpacingOutput & {
    status: Required<GPUCandidateStatusPort> &
      GPUOverflowStatusPort &
      Required<GPUInvalidStatusPort>;
  };
  /** Upper allowance: eight radix sorts/scans, reductions, two tiled grid searches at 1M rows. */
  readonly estimatedCommandNodeCount = 8192;

  constructor(props: GPUAnalysisSpacingProps) {
    this.id = props.id ?? 'analysis-spacing';
    const {id} = this;
    this.spatialContext = resolveGPUSpatialContext(
      id,
      props.spatialContext,
      GPU_NEIGHBOR_SEARCH_GREAT_CIRCLE_SUPPORT
    );
    validateRequirement(id, Boolean(props.spatialContext), 'spatialContext must be declared');
    if (this.spatialContext.coordinateSpace === 'longitude-latitude') {
      validateRequirement(
        id,
        (this.spatialContext.units ?? 'meters') === 'meters' &&
          (this.spatialContext.sphereRadius ?? GPU_EARTH_MEAN_RADIUS_METERS) ===
            GPU_EARTH_MEAN_RADIUS_METERS,
        'spatialContext must use meters and the mean Earth sphere (6371008.8)'
      );
    }
    validateSinglePackedView(props.positions, ['float32x2'], `${id} positions`);
    validateCapacity(
      id,
      'positions.length',
      props.positions.length,
      1,
      GPU_ANALYSIS_SPACING_MAXIMUM_POINT_COUNT
    );
    this.neighborRank = props.neighborRank ?? GPU_ANALYSIS_DEFAULT_NEIGHBOR_RANK;
    validateCapacity(id, 'neighborRank', this.neighborRank, 1, GPU_ANALYSIS_MAXIMUM_NEIGHBOR_RANK);
    this.neighborCapacity = props.neighborCapacity ?? props.positions.length * this.neighborRank;
    validateCapacity(id, 'neighborCapacity', this.neighborCapacity, 1, MAXIMUM_NEIGHBOR_CAPACITY);
    this.gridSize = props.gridSize ? [...props.gridSize] : [128, 128];
    validateViewLength(id, 'gridSize', this.gridSize, 2);
    for (const [axis, size] of this.gridSize.entries())
      validateCapacity(id, `gridSize[${axis}]`, size, 1, MAXIMUM_GRID_CELL_COUNT);
    validateMaximumCount(
      id,
      'gridSize cells',
      this.gridSize[0] * this.gridSize[1],
      MAXIMUM_GRID_CELL_COUNT
    );
    for (const name of ['mask', 'identities'] as const) {
      const view = props[name];
      if (view) {
        validateSinglePackedView(view, ['uint32'], `${id} ${name}`);
        validateViewLength(id, name, view, props.positions.length);
      }
    }
    validateSinglePackedView(props.output.summary, ['uint32'], `${id} output.summary`);
    validateViewLength(
      id,
      'output.summary',
      props.output.summary,
      GPU_ANALYSIS_SPACING_SUMMARY_LENGTH
    );
    validateGraphOutputsDisjointFromInputs(
      id,
      [props.output.summary],
      [props.positions, props.mask, props.identities]
    );
    validateCommandNodeBudget(id, this.estimatedCommandNodeCount);
    this.props = {...props, gridSize: [...this.gridSize] as [number, number]};
    // Handles expose only the owner's identity in their type; these packed views belong to
    // the command graph that creates/imports them. Use its public view factory for aliases.
    const summaryGraph = props.output.summary.buffer.graph as GPUCommandGraph;
    const scalar = (word: number) =>
      summaryGraph.createDataView(props.output.summary.buffer, {
        format: 'uint32',
        length: 1,
        byteOffset: props.output.summary.byteOffset + word * 4
      });
    this.output = {
      ...props.output,
      status: {
        candidateOverflow: scalar(27),
        overflow: scalar(2),
        invalidCount: scalar(1)
      }
    };
  }

  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    validateGraphViewsBelongToGraph(this.id, graph, [
      this.props.positions,
      this.props.mask,
      this.props.identities,
      this.props.output.summary
    ]);
    validateGPUStorageBindingByteLength(
      graph.device,
      this.id,
      'neighborCapacity scratch',
      this.neighborCapacity * 4,
      'lower neighborRank or positions.length'
    );
    validateGPUStorageBindingByteLength(
      graph.device,
      this.id,
      'positions.length * neighborRank scratch',
      this.props.positions.length * this.neighborRank * 4,
      'lower neighborRank or positions.length'
    );
    validateGPUStorageBindingByteLength(
      graph.device,
      this.id,
      'gridSize cell offsets',
      (this.gridSize[0] * this.gridSize[1] + 1) * 4,
      'lower gridSize'
    );
    validateGPUStorageBindingByteLength(
      graph.device,
      this.id,
      'positions scratch',
      this.props.positions.length * 8,
      'lower positions.length'
    );
    const nodes = createAnalysisSpacingStatisticsNodes(graph, this);
    validateMaximumCount(this.id, 'command nodes', nodes.length, this.estimatedCommandNodeCount);
    return nodes;
  }
}

/** Decodes the single bounded readback, refusing overflow and nonfinite numerical results. */
export function decodeGPUAnalysisSpacingStatistics(
  words: Uint32Array
): GPUAnalysisSpacingStatistics | undefined {
  const id = 'decodeGPUAnalysisSpacingStatistics';
  validateCapacity(
    id,
    'summary.length',
    words.length,
    GPU_ANALYSIS_SPACING_SUMMARY_LENGTH,
    GPU_ANALYSIS_SPACING_SUMMARY_LENGTH
  );
  validateRequirement(id, words[7] === 1, 'summary version must be 1');
  validateRequirement(id, words[2] === 0, `summary overflow must be 0 (requiredCount ${words[3]})`);
  validateRequirement(
    id,
    words[27] === 0,
    `summary candidateOverflow must be 0 (requiredCount ${words[3]})`
  );
  const pointCount = words[0];
  if (pointCount === 0) return undefined;
  validateCapacity(id, 'pointCount', pointCount, 1, GPU_ANALYSIS_SPACING_MAXIMUM_POINT_COUNT);
  const values = new Float32Array(words.buffer, words.byteOffset, words.length);
  const coordinateModel = words[26] === 1 ? 'geographic' : 'planar';
  const extent = Array.from(values.slice(8, 12)) as [number, number, number, number];
  const sizes = getGPUAnalysisExtentSizes(extent, coordinateModel);
  const statistics: GPUAnalysisSpacingStatistics = {
    pointCount,
    sampleSize: pointCount,
    sampled: false,
    coordinateModel,
    unit: coordinateModel === 'geographic' ? 'metre' : 'coordinate-unit',
    extent,
    extentWidth: sizes.width,
    extentHeight: sizes.height,
    extentDiagonal: Math.hypot(sizes.width, sizes.height),
    center:
      coordinateModel === 'planar'
        ? [extent[0] + values[28], extent[1] + values[29]]
        : [values[12], values[13]],
    standardDistance: values[14],
    medianDistanceFromCenter: values[15],
    duplicateLocationCount: words[4]
  };
  const usedValues = [...extent, ...statistics.center, values[14], values[15]];
  if (pointCount >= 2) {
    statistics.nearestNeighbor = {
      minimum: values[16],
      lowerQuartile: values[17],
      median: values[18],
      upperQuartile: values[19],
      mean: values[20],
      maximum: values[21],
      upperFence: values[22],
      aboveUpperFenceCount: words[23],
      ...(words[25] >= 2 ? {nonZeroMedian: values[24]} : {})
    };
    usedValues.push(...Array.from(values.slice(16, 23)));
    if (words[25] >= 2) usedValues.push(values[24]);
  }
  validateCapacity(id, 'neighborRank', words[5], 1, GPU_ANALYSIS_MAXIMUM_NEIGHBOR_RANK);
  const curveLength = pointCount > words[5] ? Math.min(pointCount, 512) : 0;
  validateRequirement(id, words[6] === curveLength, `curve length must be ${curveLength}`);
  if (curveLength > 0) {
    const distances = Array.from(values.slice(32, 32 + curveLength));
    statistics.kDistance = {k: words[5], distances};
    usedValues.push(...distances);
  }
  validateRequirement(
    id,
    usedValues.every(Number.isFinite),
    'summary distances and coordinates must be finite (use a local planar frame for extreme magnitudes)'
  );
  return statistics;
}

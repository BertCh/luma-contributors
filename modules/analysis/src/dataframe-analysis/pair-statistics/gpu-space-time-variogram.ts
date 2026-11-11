// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  validateCapacity,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validateSinglePackedView
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  GPU_PLANAR_SPATIAL_CONTEXT_SUPPORT,
  resolveGPUSpatialContext,
  type GPUSpatialContext
} from '../../analysis-core/contracts/spatial-context';
import {
  getPairHistogramNodes,
  getPairHistogramReadWGSL,
  PAIR_HISTOGRAM_LARGE_TERM_SHIFT,
  validatePairHistogramShape
} from './pair-histogram';
import {
  getPairStatisticsInputNodes,
  PAIR_STATISTICS_FLOAT_WGSL,
  validatePairStatisticsInputs,
  validatePairStatisticsRowCount
} from './pair-statistics-grid';
import {getPairStatisticsUnitVarianceNodes} from './pair-statistics-fixed-point-unit';
import {
  validatePairStatisticsBounds,
  type GPUPairStatisticsBounds
} from './pair-statistics-parameters';

const OPERATION = 'GPUSpaceTimeVariogram';

/** Number of float32 elements in a `GPUSpaceTimeVariogram` parameter buffer. */
export const GPU_SPACE_TIME_VARIOGRAM_PARAMETER_LENGTH = 8;

/** Fixed-point units per unit of the typical cross term, as for the spatial variogram. */
const UNITS_PER_TYPICAL_TERM = 16777216;

/** Per-frame parameters of {@link GPUSpaceTimeVariogram}. */
export type GPUSpaceTimeVariogramSettings = {
  /** Inclusive `[minX, minY, maxX, maxY]` extent; rows outside it are excluded. */
  bounds: Exclude<GPUPairStatisticsBounds, 'study-window'>;
  /** Largest spatial lag, in the units of the positions. */
  maximumDistance: number;
  /** Largest time lag, in the unit of the relative `times`. */
  maximumTimeLag: number;
};

/**
 * Packs the 8-element float32 layout
 * `[minX, minY, maxX, maxY, maximumDistance, maximumTimeLag, 0, 0]`.
 *
 * @throws If a value is not finite, the bounds are inverted, or a maximum lag is not positive.
 */
export function getGPUSpaceTimeVariogramParameterValues(
  settings: GPUSpaceTimeVariogramSettings,
  target: Float32Array = new Float32Array(GPU_SPACE_TIME_VARIOGRAM_PARAMETER_LENGTH)
): Float32Array {
  if (target.length < GPU_SPACE_TIME_VARIOGRAM_PARAMETER_LENGTH) {
    throw new Error(
      `${OPERATION} target must hold ${GPU_SPACE_TIME_VARIOGRAM_PARAMETER_LENGTH} elements`
    );
  }
  validatePairStatisticsBounds(OPERATION, settings.bounds, settings.maximumDistance);
  if (!Number.isFinite(settings.maximumTimeLag) || settings.maximumTimeLag <= 0) {
    throw new Error(`${OPERATION} maximumTimeLag must be positive and finite`);
  }
  target.set([...settings.bounds, settings.maximumDistance, settings.maximumTimeLag, 0, 0]);
  return target;
}

/** Properties of {@link GPUSpaceTimeVariogram}. */
export type GPUSpaceTimeVariogramProps = {
  id?: string;
  spatialContext?: GPUSpatialContext;
  positions: GraphDataView<'float32x2'>;
  values: GraphDataView<'float32'>;
  /** Defaults to values, giving the ordinary Matheron space-time semivariogram. */
  secondValues?: GraphDataView<'float32'>;
  /**
   * Event times relative to a float64 origin chosen on the CPU, in one unit. Never upload raw
   * epoch seconds or milliseconds: float32 spaces them 128 s (or about 2 minutes) apart. Build
   * them with `getGPUSpaceTimeRelativeTimes` from `gpu-spatial-analysis`, which also reports the
   * remaining float32 resolution.
   */
  times: GraphDataView<'float32'>;
  mask?: GraphDataView<'uint32'>;
  /** Written with {@link getGPUSpaceTimeVariogramParameterValues}. */
  parameters: GraphDataView<'float32'>;
  /** Maximum `[columns, rows]` of the neighbor-search lattice; affects speed, not results. */
  gridSize: readonly [number, number];
  /** Positive spatial classes over `(0, maximumDistance]`, in addition to the zero-lag class. */
  lagCount: number;
  /** Positive temporal classes over `(0, maximumTimeLag]`, in addition to the zero-lag class. */
  temporalLagCount: number;
  output: {
    /**
     * `(lagCount + 1) * (temporalLagCount + 1)` bins, spatial-major:
     * `spatialClass * (temporalLagCount + 1) + temporalClass`. Class 0 of each axis is the exact
     * zero lag, so bin 0 holds coincident points at the same time, the first row the purely
     * temporal variogram and the first column the purely spatial one.
     */
    semivariances: GraphDataView<'float32'>;
    pairCounts: GraphDataView<'uint32'>;
    /**
     * Per bin: 0 valid, 1 empty or invalid parameters, 2 nonfinite arithmetic, 3 the exact
     * integer accumulation overflowed.
     */
    status: GraphDataView<'uint32'>;
  };
};

/**
 * Empirical space-time semivariogram on the indexed pair histogram: signed
 * `sum (x_i-x_j)(y_i-y_j) / (2 N)` per bin, equal to the Matheron semivariogram when
 * `secondValues` is omitted. Each unordered pair of included rows with `d <= maximumDistance` and
 * `|t_i - t_j| <= maximumTimeLag` is counted once.
 *
 * Classes: class 0 of each axis holds exactly zero lag (identical float32 coordinates, identical
 * float32 relative times); positive lags fall in `1 + min(floor(lag / maximum * count),
 * count - 1)`, left-closed and right-open with the final edge included. A row is included when its
 * mask is nonzero and its coordinates, values and time are finite, with the position inside the
 * parameter bounds. Terms are accumulated as exact integers (positive and negative parts apart),
 * so results are reproducible; the fixed-point unit is the typical cross term, as in
 * `GPUVariogram`. Cost is the number of pairs in neighboring lattice cells, not `rows^2`.
 */
export class GPUSpaceTimeVariogram implements GPUCommandNodeProducer {
  readonly id: string;
  readonly spatialContext: GPUSpatialContext;
  constructor(readonly props: GPUSpaceTimeVariogramProps) {
    this.id = props.id ?? 'space-time-variogram';
    const id = this.id;
    this.spatialContext = resolveGPUSpatialContext(id, props.spatialContext, {
      ...GPU_PLANAR_SPATIAL_CONTEXT_SUPPORT,
      measure: 'distance'
    });
    const rows = validatePairStatisticsInputs(id, {
      ...props,
      parameterLength: GPU_SPACE_TIME_VARIOGRAM_PARAMETER_LENGTH
    });
    validateCapacity(id, 'lagCount', props.lagCount);
    validateCapacity(id, 'temporalLagCount', props.temporalLagCount);
    // Count, positive and negative sums, and the large-term channel of each.
    validatePairHistogramShape(id, getBinCount(props), 5);
    for (const view of [props.secondValues, props.times])
      if (view) {
        validateSinglePackedView(view, ['float32'], `${id} input`);
        if (view.length !== rows) throw new Error(`${id} values must match positions`);
      }
    const bins = getBinCount(props);
    for (const [name, view] of Object.entries(props.output)) {
      validateSinglePackedView(
        view,
        [name === 'semivariances' ? 'float32' : 'uint32'],
        `${id} ${name}`
      );
      if (view.length < bins) throw new Error(`${id} output.${name} must hold ${bins} bins`);
    }
    validateGraphOutputsDisjointFromInputs(id, Object.values(props.output), [
      props.positions,
      props.values,
      props.secondValues,
      props.times,
      props.mask,
      props.parameters
    ]);
  }

  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const {positions, values, secondValues, times, mask, parameters, gridSize, output} = props;
    validatePairStatisticsRowCount(id, graph.device, positions.length);
    validateGraphViewsBelongToGraph(id, graph, [
      positions,
      values,
      secondValues,
      times,
      mask,
      parameters,
      ...Object.values(output)
    ]);
    const rows = positions.length;
    const bins = getBinCount(props);
    const temporalClasses = props.temporalLagCount + 1;
    const inputs = getPairStatisticsInputNodes<Parameters>(graph, {
      id,
      operation: OPERATION,
      positions,
      parameters,
      gridSize,
      values,
      mask
    });
    const nodes = [...inputs.nodes];
    const unit = getPairStatisticsUnitVarianceNodes<Parameters>(graph, {
      id,
      operation: OPERATION,
      values,
      mask,
      statistics: inputs.statistics!
    });
    nodes.push(...unit.nodes);
    let secondStatistics: GraphDataView<'float32'> | undefined;
    let secondUnit: GraphDataView<'float32'> | undefined;
    if (secondValues) {
      // Only the centered second column and its unit are used; the cell index is the first one's.
      const second = getPairStatisticsInputNodes<Parameters>(graph, {
        id: `${id}-second`,
        operation: OPERATION,
        positions,
        parameters,
        gridSize,
        values: secondValues,
        mask
      });
      const secondUnitNodes = getPairStatisticsUnitVarianceNodes<Parameters>(graph, {
        id: `${id}-second`,
        operation: OPERATION,
        values: secondValues,
        mask,
        statistics: second.statistics!
      });
      nodes.push(...second.nodes, ...secondUnitNodes.nodes);
      secondStatistics = second.statistics;
      secondUnit = secondUnitNodes.unitVariance;
    }

    // Fixed-point scale of the cross term: 2^24 units per sqrt(u1 * u2); 0 for a constant column.
    const scale = createTransientView(graph, `${id}-scale`, 'float32', 1);
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-scale`,
        operation: OPERATION,
        variant: 'scale',
        invocationCount: 1,
        bindings: [
          {name: 'firstUnit', view: unit.unitVariance, type: 'f32', access: 'read'},
          ...(secondUnit
            ? [
                {
                  name: 'secondUnit',
                  view: secondUnit,
                  type: 'f32' as const,
                  access: 'read' as const
                }
              ]
            : []),
          {name: 'scale', view: scale, type: 'f32', access: 'read_write'}
        ],
        body: `let first = firstUnit[firstUnitOffset];
  let second = ${secondUnit ? 'secondUnit[secondUnitOffset]' : 'first'};
  scale[scaleOffset] = select(0.0, ${UNITS_PER_TYPICAL_TERM}.0 / (sqrt(first) * sqrt(second)), first > 0.0 && second > 0.0);`
      })
    );

    // Per row [centered first, centered second, time]; the first is NaN when any input is not
    // finite, which drops the row from every pair (complete-case rule).
    const rowTable = createTransientView(graph, `${id}-row-table`, 'float32', rows * 3);
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-row-table`,
        operation: OPERATION,
        variant: 'row-table',
        invocationCount: rows,
        bindings: [
          {name: 'firstStatistics', view: inputs.statistics!, type: 'f32', access: 'read'},
          ...(secondStatistics
            ? [
                {
                  name: 'secondStatistics',
                  view: secondStatistics,
                  type: 'f32' as const,
                  access: 'read' as const
                }
              ]
            : []),
          {name: 'times', view: times, type: 'f32', access: 'read'},
          {name: 'rowTable', view: rowTable, type: 'f32', access: 'read_write'}
        ],
        declarations: PAIR_STATISTICS_FLOAT_WGSL,
        body: `let first = firstStatistics[firstStatisticsOffset + index];
  let second = ${secondStatistics ? 'secondStatistics[secondStatisticsOffset + index]' : 'first'};
  let time = times[timesOffset + index];
  let usable = isFiniteValue(first) && isFiniteValue(second) && isFiniteValue(time);
  rowTable[rowTableOffset + index * 3u] = select(getQuietNaN(index), first, usable);
  rowTable[rowTableOffset + index * 3u + 1u] = second;
  rowTable[rowTableOffset + index * 3u + 2u] = time;`
      })
    );

    // Channels: 0 count, 1 positive terms, 2 negative terms, 3 and 4 their large-term channels.
    const channelCount = 5;
    const declarations = `
const LAG_COUNT: u32 = ${props.lagCount}u;
const TEMPORAL_LAG_COUNT: u32 = ${props.temporalLagCount}u;
const TEMPORAL_CLASSES: u32 = ${temporalClasses}u;
const LARGE_TERM_UNIT: f32 = ${2 ** PAIR_HISTOGRAM_LARGE_TERM_SHIFT}.0;`;
    const histogram = getPairHistogramNodes<Parameters>(graph, {
      id: `${id}-histogram`,
      operation: OPERATION,
      positions,
      parameters,
      gridSize,
      sortedRows: inputs.sortedRows,
      cellOffsets: inputs.cellOffsets,
      slotCount: bins,
      channelCount,
      pairOrder: 'unordered',
      extraBindings: [
        {name: 'rowTable', view: rowTable, type: 'f32', access: 'read'},
        {name: 'scaleView', view: scale, type: 'f32', access: 'read'}
      ],
      declarations,
      focusPrologue: `let focusBase = rowTableOffset + focus * 3u;
    let focusFirst = rowTable[focusBase];
    let focusSecond = rowTable[focusBase + 1u];
    let focusTime = rowTable[focusBase + 2u];
    let maximumTime = readParameter(5u);
    let scale = scaleView[scaleViewOffset];
    let focusUsable = isFiniteValue(focusFirst) && isFiniteValue(maximumTime) && maximumTime > 0.0;`,
      pairAction: `let otherBase = rowTableOffset + neighbor * 3u;
            let otherFirst = rowTable[otherBase];
            if (!focusUsable || !isFiniteValue(otherFirst)) { continue; }
            let timeLag = abs(rowTable[otherBase + 2u] - focusTime);
            if (timeLag > maximumTime) { continue; }
            let spatialClass = select(
              1u + min(u32(pairDistance / lattice.maximumDistance * f32(LAG_COUNT)), LAG_COUNT - 1u),
              0u, deltaX == 0.0 && deltaY == 0.0);
            let temporalClass = select(
              1u + min(u32(timeLag / maximumTime * f32(TEMPORAL_LAG_COUNT)), TEMPORAL_LAG_COUNT - 1u),
              0u, timeLag == 0.0);
            let bin = spatialClass * TEMPORAL_CLASSES + temporalClass;
            let term = 0.5 * (rowTable[otherBase] - focusFirst) * (rowTable[otherBase + 1u] - focusSecond) * scale;
            accumulate(bin, 0u, 1u);
            accumulateScaled(bin, 1u, 3u, term);
            accumulateScaled(bin, 2u, 4u, -term);`
    });
    nodes.push(...histogram.nodes);

    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-finish`,
        operation: OPERATION,
        variant: 'finish',
        invocationCount: bins,
        bindings: [
          {name: 'accumulators', view: histogram.accumulators, type: 'u32', access: 'read'},
          {name: 'scaleView', view: scale, type: 'f32', access: 'read'},
          {name: 'semivariances', view: output.semivariances, type: 'f32', access: 'read_write'},
          {name: 'pairCounts', view: output.pairCounts, type: 'u32', access: 'read_write'},
          {name: 'status', view: output.status, type: 'u32', access: 'read_write'}
        ],
        declarations: `${PAIR_STATISTICS_FLOAT_WGSL}
${declarations}
${getPairHistogramReadWGSL('accumulators', channelCount, '', bins)}`,
        body: `let scale = scaleView[scaleViewOffset];
  let count = readPairAccumulator(index, 0u);
  let positive = readPairAccumulator(index, 1u) + readPairAccumulator(index, 3u) * LARGE_TERM_UNIT;
  let negative = readPairAccumulator(index, 2u) + readPairAccumulator(index, 4u) * LARGE_TERM_UNIT;
  // A zero scale means a constant column: every term is 0.
  let value = select(0.0, (positive - negative) / scale / count, scale > 0.0);
  let finite = isFiniteValue(value);
  semivariances[semivariancesOffset + index] = select(getQuietNaN(index), value, count > 0.0 && finite);
  pairCounts[pairCountsOffset + index] = readPairCount(index, 0u);
  status[statusOffset + index] = select(select(select(2u, 0u, finite), 3u, readPairOverflow()), 1u, count == 0.0);`
      })
    );
    return nodes;
  }
}

function getBinCount(props: {lagCount: number; temporalLagCount: number}): number {
  return (props.lagCount + 1) * (props.temporalLagCount + 1);
}

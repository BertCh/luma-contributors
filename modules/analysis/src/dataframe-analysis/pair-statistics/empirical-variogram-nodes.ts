// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {validateGraphViewsBelongToGraph} from '../../analysis-core/kernels/gpu-contributor-utils';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  getPairHistogramNodes,
  getPairHistogramReadWGSL,
  PAIR_HISTOGRAM_LARGE_TERM_SHIFT
} from './pair-histogram';
import {
  getPairStatisticsInputNodes,
  PAIR_STATISTICS_FLOAT_WGSL,
  validatePairStatisticsRowCount
} from './pair-statistics-grid';
import {getPairStatisticsUnitVarianceNodes} from './pair-statistics-fixed-point-unit';
import {getLegacyVariogramFallbackNodes} from './legacy-variogram-fallback';
import type {GPUSpaceTimeVariogramProps} from './gpu-space-time-variogram';

const OPERATION = 'GPUSpaceTimeVariogram';
/** Fixed-point units per typical cross term, shared with the spatial variogram. */
const UNITS_PER_TYPICAL_TERM = 16777216;

/** Indexed empirical variogram inputs; the legacy cross layout permits an absent time column. */
type EmpiricalVariogramProps = Omit<GPUSpaceTimeVariogramProps, 'times'> &
  Pick<Partial<GPUSpaceTimeVariogramProps>, 'times'>;

/**
 * Shared indexed signed cross-semivariance kernels. `legacyBins` folds exact zero lags into the
 * first positive bin, preserving GPUCrossVariogram's layout and raw floating-point arithmetic.
 * @internal
 */
export function getEmpiricalVariogramNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  props: EmpiricalVariogramProps,
  legacyBins: boolean = false
): GPUCommandNode<Parameters>[] {
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
  if (legacyBins) {
    // The caller supplies a complete-case mask. Build only the shared index: global moments
    // can erase small raw differences, and fixed-point scaling can quantize valid terms to zero.
    const inputs = getPairStatisticsInputNodes<Parameters>(graph, {
      id,
      operation: OPERATION,
      positions,
      parameters,
      gridSize,
      legacyLattice: true,
      mask
    });
    return [...inputs.nodes, ...getLegacyVariogramFallbackNodes(graph, id, props, inputs)];
  }
  const rows = positions.length;
  const bins = (props.lagCount + 1) * (props.temporalLagCount + 1);
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
        ...(times
          ? [{name: 'times', view: times, type: 'f32' as const, access: 'read' as const}]
          : []),
        {name: 'rowTable', view: rowTable, type: 'f32', access: 'read_write'}
      ],
      declarations: PAIR_STATISTICS_FLOAT_WGSL,
      body: `let first = firstStatistics[firstStatisticsOffset + index];
  let second = ${secondStatistics ? 'secondStatistics[secondStatisticsOffset + index]' : 'first'};
  let time = ${times ? 'times[timesOffset + index]' : '0.0'};
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
            let spatialClass = select(1u + min(u32(pairDistance / lattice.maximumDistance * f32(LAG_COUNT)), LAG_COUNT - 1u), 0u, deltaX == 0.0 && deltaY == 0.0);
            let temporalClass = select(1u + min(u32(timeLag / maximumTime * f32(TEMPORAL_LAG_COUNT)), TEMPORAL_LAG_COUNT - 1u), 0u, timeLag == 0.0);
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

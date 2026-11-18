// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GraphDataView} from '@luma.gl/graph/gpu-core';
import {
  getGPUSpatialWeightsChunks,
  type GPUSpatialWeights,
  type GPUChunkedSpatialWeights
} from '../spatial-weights/index';
import {getOrdinaryLeastSquaresAllocationLengths} from './gpu-ordinary-least-squares';
import {getRegressionTileLayout} from './regression-row-tiles';
import {
  getSpatialProbeAllocationLengths,
  type RegressionAllocationLengths
} from './spatial-likelihood-allocations';
import {
  GPU_SPATIAL_WEIGHTS_TRACES_DEFAULT_PROBE_COUNT,
  GPU_SPATIAL_WEIGHTS_TRACES_MAXIMUM_EXACT_ROW_COUNT,
  type GPUSpatialWeightsTracesMode
} from './spatial-weights-traces-parameters';
import {padToAlignedElementCount} from './spatial-maximum-likelihood-nodes';
import {getRegressionLossViews} from './regression-weights-completeness';

/** CSR and completeness bindings; optional distances and descriptive metadata are not read. */
export function getRegressionWeightBindings(
  weights: GPUSpatialWeights | GPUChunkedSpatialWeights
): Record<string, GraphDataView> {
  return Object.fromEntries([
    ...getGPUSpatialWeightsChunks(weights).flatMap((chunk, index) => [
      [`offsets ${index}`, chunk.weights.offsets] as const,
      [`neighbors ${index}`, chunk.weights.neighbors] as const,
      [`weights ${index}`, chunk.weights.weights] as const
    ]),
    ...getRegressionLossViews(weights).map(
      (view, index) => [`weights loss ${index}`, view] as const
    )
  ]);
}

/** Largest chunk capacity, since no CSR-reading dispatch binds all chunks together. */
export function getRegressionMaximumEdgeCount(
  weights: GPUSpatialWeights | GPUChunkedSpatialWeights
): number {
  return getGPUSpatialWeightsChunks(weights).reduce(
    (maximum, chunk) => Math.max(maximum, chunk.weights.neighbors.length),
    0
  );
}

/** Complete power/Gram-trace allocation plan, including padded probe tables and transpose/sort/scan. */
export function getRegressionTraceAllocationLengths(props: {
  rowCount: number;
  edgeCount: number;
  maximumPower: number;
  tileCount?: number;
  mode?: GPUSpatialWeightsTracesMode;
  probeCount?: number;
  probeBatchSize?: number;
  includeGram?: boolean;
}): RegressionAllocationLengths {
  const {rowCount, edgeCount, maximumPower} = props;
  const tileCount =
    props.tileCount ?? Math.ceil(rowCount / Math.max(32, Math.ceil(rowCount / 1024)));
  const mode =
    props.mode ??
    (rowCount <= GPU_SPATIAL_WEIGHTS_TRACES_MAXIMUM_EXACT_ROW_COUNT ? 'exact' : 'stochastic');
  const probeTotal =
    mode === 'exact'
      ? rowCount
      : (props.probeCount ?? GPU_SPATIAL_WEIGHTS_TRACES_DEFAULT_PROBE_COUNT);
  const dotRowCount = props.includeGram ? Math.max(maximumPower, 2) + 2 : maximumPower;
  return {
    ...getSpatialProbeAllocationLengths(
      rowCount,
      tileCount,
      probeTotal,
      dotRowCount,
      props.probeBatchSize
    ),
    'trace norm partials': tileCount,
    'trace scaled weights': edgeCount,
    'trace sums partials': maximumPower * padToAlignedElementCount(tileCount),
    'trace ones vectors': rowCount,
    'trace infinity norm': 1,
    ...(props.includeGram
      ? {
          'Gram partials': tileCount * 2,
          'transpose offsets': rowCount + 1,
          // Published auto sort uses local bitonic <=256 rows, otherwise row-sized ping-pong
          // and <=16 histogram words per 256 rows. Scan scratch only shrinks its input.
          'transpose slots and sort scratch': edgeCount,
          'transpose tolerance': 1
        }
      : {})
  };
}

/** Nested OLS and both GM variance routes, before a parent creates its first transient. */
export function getRegressionErrorGMAllocationLengths(props: {
  rowCount: number;
  predictorCount: number;
  edgeCount: number;
  tileRowCount?: number;
  tileCount?: number;
  homoskedastic?: boolean;
  traceMode?: GPUSpatialWeightsTracesMode;
  traceProbeCount?: number;
}): RegressionAllocationLengths {
  const {rowCount, predictorCount} = props;
  const tileRows = props.tileRowCount ?? getRegressionTileLayout(rowCount).tileRowCount;
  const tileCount = props.tileCount ?? Math.ceil(rowCount / tileRows);
  const columns = predictorCount + 1;
  const filteredCount = columns * columns + columns + 1;
  return {
    ...getOrdinaryLeastSquaresAllocationLengths(rowCount, predictorCount, tileRows),
    'GM OLS outputs': Math.max(columns, 22),
    'GM row scratch': rowCount,
    'GM workspace': predictorCount + 4 + columns + 16,
    'GM mean partials': tileCount * columns,
    'GM moment partials': tileCount * 7,
    'GM moments': 7,
    'GM filtered partials': tileCount * filteredCount,
    'GM filtered moments': filteredCount,
    'GM residual partials': tileCount * 5,
    'GM residual totals': 5,
    'GM trace outputs': 4,
    ...(props.homoskedastic
      ? {
          ...getRegressionTraceAllocationLengths({
            rowCount,
            edgeCount: props.edgeCount,
            maximumPower: 2,
            mode: props.traceMode,
            probeCount: props.traceProbeCount,
            includeGram: true
          }),
          'GM power partials': tileCount * 3,
          'GM power totals': 3
        }
      : {})
  };
}

/** 2SLS moments and transpose; optional robust covariance scratch is added by its shared planner. */
export function getRegressionTwoStageAllocationLengths(
  rowCount: number,
  predictorCount: number,
  edgeCount: number,
  instrumentOrder: number,
  tileRowCount?: number
): RegressionAllocationLengths {
  const tileRows = tileRowCount ?? getRegressionTileLayout(rowCount).tileRowCount;
  const tileCount = Math.ceil(rowCount / tileRows);
  const instrumentCount = (instrumentOrder + 1) * predictorCount;
  const columns = predictorCount + 1;
  const meanCount = instrumentCount + 2;
  const momentCount = instrumentCount ** 2 + instrumentCount * columns + instrumentCount + columns;
  const momentStride = momentCount + columns + 1 + instrumentCount - predictorCount;
  return {
    '2SLS mean partials': tileCount * meanCount,
    '2SLS moment partials': tileCount * momentStride,
    '2SLS moments': momentStride,
    '2SLS workspace': meanCount + 2 * columns + 6,
    '2SLS residual partials': tileCount * 4,
    '2SLS residual totals': 4,
    '2SLS refinement partials': tileCount * (instrumentCount + 1),
    '2SLS refinement totals': instrumentCount + 1,
    '2SLS norm partials': tileCount * 3,
    '2SLS AK partials': tileCount * (columns + 5),
    '2SLS AK totals': columns + 5,
    '2SLS row scratch': rowCount,
    '2SLS transpose offsets': rowCount + 1,
    '2SLS transpose slots and sort scratch': edgeCount,
    '2SLS transpose tolerance': 1
  };
}

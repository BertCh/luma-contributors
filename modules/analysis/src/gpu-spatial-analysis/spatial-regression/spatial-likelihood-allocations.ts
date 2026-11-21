// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {
  validateGPUStorageBindingByteLength,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {padToAlignedElementCount} from './spatial-maximum-likelihood-nodes';
import {getRegressionTileLayout, getRowTileMergeScratchLength} from './regression-row-tiles';
import {
  GPU_SPATIAL_WEIGHTS_TRACES_DEFAULT_PROBE_COUNT,
  GPU_SPATIAL_WEIGHTS_TRACES_AUTO_EXACT_MAXIMUM_ROWS,
  GPU_SPATIAL_WEIGHTS_TRACES_SCRATCH_ELEMENT_LIMIT,
  type GPUSpatialWeightsTracesMode
} from './spatial-weights-traces-parameters';

/** Binding-sized float32/uint32 arrays. Plans contain no graph objects or allocations. */
export type RegressionAllocationLengths = Readonly<Record<string, number>>;

/** Rejects every derived binding before a composite contributor declares its first resource. */
export function validateRegressionAllocations(
  device: Device,
  id: string,
  lengths: RegressionAllocationLengths
): void {
  for (const [name, length] of Object.entries(lengths)) {
    validateGPUStorageBindingByteLength(
      device,
      id,
      name,
      Math.max(1, length) * 4,
      'reduce rows, probes or powers'
    );
    validateRequirement(
      id,
      Math.max(1, length) * 4 <= device.limits.maxBufferSize,
      `${name} allocation must fit maxBufferSize (${device.limits.maxBufferSize})`
    );
  }
}

/** Shapes shared by power-trace and likelihood-information probe chains. */
export function getSpatialProbeAllocationLengths(
  rowCount: number,
  tileCount: number,
  probeTotal: number,
  powerCount: number,
  maximumBatchProbeCount = Number.POSITIVE_INFINITY
): RegressionAllocationLengths {
  const batchProbeCount = Math.min(
    probeTotal,
    maximumBatchProbeCount,
    Math.max(1, Math.floor(GPU_SPATIAL_WEIGHTS_TRACES_SCRATCH_ELEMENT_LIMIT / rowCount))
  );
  const batchCount = Math.ceil(probeTotal / batchProbeCount);
  const dotsStride = batchCount * padToAlignedElementCount(batchProbeCount);
  return {
    'probe vectors': batchProbeCount * rowCount,
    'probe partials': batchProbeCount * tileCount,
    'probe merge groups': getRowTileMergeScratchLength(tileCount, batchProbeCount),
    'probe batch information': batchCount * 64,
    'probe dots': powerCount * dotsStride
  };
}

/** Complete non-Gram power-trace and error-likelihood allocation plan, including nested probes. */
export function getSpatialErrorLikelihoodAllocationLengths(props: {
  rowCount: number;
  edgeCount: number;
  predictorCount: number;
  maximumPower?: number;
  logDeterminant?: 'series' | 'exact';
  traceMode?: GPUSpatialWeightsTracesMode;
  traceProbeCount?: number;
  tileRowCount?: number;
  /** A chunked CSR has a separate rounded tile count per chunk. */
  tileCount?: number;
  traceTileCount?: number;
}): RegressionAllocationLengths {
  const {rowCount, edgeCount, predictorCount} = props;
  const maximumPower = props.maximumPower ?? 128;
  const mode =
    props.traceMode ??
    (rowCount <= GPU_SPATIAL_WEIGHTS_TRACES_AUTO_EXACT_MAXIMUM_ROWS ? 'exact' : 'stochastic');
  const probeTotal =
    mode === 'exact'
      ? rowCount
      : (props.traceProbeCount ?? GPU_SPATIAL_WEIGHTS_TRACES_DEFAULT_PROBE_COUNT);
  const layout = getRegressionTileLayout(rowCount, props.tileRowCount);
  const tileCount = props.tileCount ?? layout.tileCount;
  const traceTileRows = Math.max(32, Math.ceil(rowCount / 1024));
  const traceTileCount = props.traceTileCount ?? Math.ceil(rowCount / traceTileRows);
  const columns = predictorCount + 2;
  const packedCount = (columns * (columns + 1)) / 2;
  const ordinaryLeastSquaresCount = (predictorCount * (predictorCount + 1)) / 2 + predictorCount;
  const traceProbes = getSpatialProbeAllocationLengths(
    rowCount,
    traceTileCount,
    probeTotal,
    maximumPower
  );
  const informationProbes = getSpatialProbeAllocationLengths(rowCount, tileCount, probeTotal, 1);
  return {
    'likelihood workspace': 22 + 4 * predictorCount + Number(props.logDeterminant === 'exact'),
    'likelihood scalar certificate': props.logDeterminant === 'exact' ? rowCount : 0,
    'likelihood scalar certificate partials':
      props.logDeterminant === 'exact' ? getRegressionTileLayout(rowCount).tileCount : 0,
    'likelihood scalar certificate total': props.logDeterminant === 'exact' ? 1 : 0,
    'power traces/sums': maximumPower,
    'likelihood norm partials': tileCount,
    'likelihood merge groups': getRowTileMergeScratchLength(
      tileCount,
      Math.max(3 * packedCount, ordinaryLeastSquaresCount, predictorCount + 1, 6)
    ),
    'likelihood mean partials': tileCount * (predictorCount + 1),
    'likelihood OLS partials': tileCount * ordinaryLeastSquaresCount,
    'likelihood OLS totals': ordinaryLeastSquaresCount,
    'likelihood Gram partials': 3 * tileCount * packedCount,
    'likelihood Gram totals': 3 * packedCount,
    'likelihood residuals': rowCount,
    'likelihood residual partials': tileCount * 6,
    'likelihood residual totals': 6,
    'trace infinity norm': 1,
    'trace norm partials': traceTileCount,
    'trace merge groups': getRowTileMergeScratchLength(traceTileCount, maximumPower),
    'trace scaled weights': edgeCount,
    'trace sums partials': maximumPower * padToAlignedElementCount(traceTileCount),
    'trace ones vectors': rowCount,
    ...Object.fromEntries(
      Object.entries(traceProbes).map(([name, length]) => [`trace ${name}`, length])
    ),
    ...Object.fromEntries(
      Object.entries(informationProbes).map(([name, length]) => [`information ${name}`, length])
    )
  };
}

/** Upper bound for one whole-CSR error likelihood, including both nested probe chains. */
export function getSpatialErrorLikelihoodNodeCount(
  rowCount: number,
  maximumPower: number,
  traceMode?: GPUSpatialWeightsTracesMode,
  traceProbeCount = GPU_SPATIAL_WEIGHTS_TRACES_DEFAULT_PROBE_COUNT
): number {
  const mode =
    traceMode ??
    (rowCount <= GPU_SPATIAL_WEIGHTS_TRACES_AUTO_EXACT_MAXIMUM_ROWS ? 'exact' : 'stochastic');
  const probeTotal = mode === 'exact' ? rowCount : traceProbeCount;
  const batchSize = Math.min(
    probeTotal,
    Math.max(1, Math.floor(GPU_SPATIAL_WEIGHTS_TRACES_SCRATCH_ELEMENT_LIMIT / rowCount))
  );
  const batches = Math.ceil(probeTotal / batchSize);
  // 128 covers the fixed fit, publication and optional two OLS diagnostic graphs. Each stochastic
  // power has a matvec and a merge; the information chain has a matvec per power and a final merge.
  return 128 + 2 * maximumPower + batches * ((mode === 'stochastic' ? 3 : 2) * maximumPower + 1);
}

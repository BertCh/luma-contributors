// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  getGPUSpatialWeightsChunks,
  type GPUSpatialWeights,
  type GPUChunkedSpatialWeights
} from '../spatial-weights/index';
import {getRegressionLossFoldCount} from './regression-weights-completeness';
import {REGRESSION_TILE_MERGE_GROUP_SIZE} from './regression-row-tiles';
import {getGPUSpatialWeightsTilePlan} from './chunked-weights-kernels';
import {GPU_SPATIAL_WEIGHTS_TRACES_SCRATCH_ELEMENT_LIMIT} from './spatial-weights-traces-parameters';

/** Published transpose/sort/scan primitives have no node-count API; tested upper allowance. */
const TRACE_TRANSPOSE_NODE_ALLOWANCE = 512;
/** Fixed fit/publication stages, both nested OLS/BP fits and their hierarchical stages. */
const LIKELIHOOD_FIXED_NODE_ALLOWANCE = 128;

export type RegressionTraceCommandPlan = {
  weights: GPUSpatialWeights | GPUChunkedSpatialWeights;
  tileCount: number;
  maximumPower: number;
  probeTotal: number;
  batchProbeCount: number;
  stochastic: boolean;
  includeGram?: boolean;
  approximation?: boolean;
};

/** Counts all trace chunk passes, group passes, probe batches and optional publication stages. */
export function getRegressionTraceCommandNodeCount(plan: RegressionTraceCommandPlan): number {
  const chunks = getGPUSpatialWeightsChunks(plan.weights).length;
  const hierarchical = Number(plan.tileCount > REGRESSION_TILE_MERGE_GROUP_SIZE);
  const merge = 1 + hierarchical;
  const steps = plan.includeGram ? Math.max(plan.maximumPower, 2) : plan.maximumPower;
  const batches = Math.ceil(plan.probeTotal / plan.batchProbeCount);
  const losses = getRegressionLossFoldCount(plan.weights);
  return (
    chunks * (2 + plan.maximumPower) +
    2 * merge +
    1 +
    batches *
      (steps * chunks +
        (plan.stochastic ? steps * merge : 0) +
        (plan.includeGram ? 1 + 2 * merge : 0)) +
    1 +
    (plan.includeGram ? TRACE_TRANSPOSE_NODE_ALLOWANCE + 1 + hierarchical : 0) +
    Number(Boolean(plan.approximation)) +
    (losses ? losses + 8 : 0)
  );
}

/** Conservative composite bound including trace children, information probes and chunk row passes. */
export function getRegressionLikelihoodCommandNodeCount(plan: {
  weights: GPUSpatialWeights | GPUChunkedSpatialWeights;
  rowCount: number;
  tileCount: number;
  traceTileCount: number;
  maximumPower: number;
  probeTotal: number;
  stochastic: boolean;
  kind: 'lag' | 'error';
}): number {
  const chunks = getGPUSpatialWeightsChunks(plan.weights).length;
  const batchProbeCount = Math.min(
    plan.probeTotal,
    Math.max(1, Math.floor(GPU_SPATIAL_WEIGHTS_TRACES_SCRATCH_ELEMENT_LIMIT / plan.rowCount))
  );
  const batches = Math.ceil(plan.probeTotal / batchProbeCount);
  const losses = getRegressionLossFoldCount(plan.weights);
  return (
    getRegressionTraceCommandNodeCount({...plan, tileCount: plan.traceTileCount, batchProbeCount}) +
    LIKELIHOOD_FIXED_NODE_ALLOWANCE +
    chunks * (plan.kind === 'lag' ? plan.maximumPower + 1 : 7) +
    batches *
      (plan.maximumPower * chunks + 1 + Number(plan.tileCount > REGRESSION_TILE_MERGE_GROUP_SIZE)) +
    (losses ? losses + 16 : 0)
  );
}

/** GM chunk passes, nested OLS/BP and optional homoskedastic trace/transpose child. */
export function getRegressionGMCommandNodeCount(plan: {
  weights: GPUSpatialWeights | GPUChunkedSpatialWeights;
  rowCount: number;
  homoskedastic: boolean;
  traceMode?: 'exact' | 'stochastic';
  traceProbeCount?: number;
}): number {
  const chunks = getGPUSpatialWeightsChunks(plan.weights).length;
  const losses = getRegressionLossFoldCount(plan.weights);
  let count = LIKELIHOOD_FIXED_NODE_ALLOWANCE + 6 * chunks + (losses ? losses + 16 : 0);
  if (plan.homoskedastic) {
    const stochastic =
      (plan.traceMode ?? (plan.rowCount <= 4096 ? 'exact' : 'stochastic')) === 'stochastic';
    const probeTotal = stochastic ? (plan.traceProbeCount ?? 64) : plan.rowCount;
    count += getRegressionTraceCommandNodeCount({
      weights: plan.weights,
      tileCount: getGPUSpatialWeightsTilePlan(
        plan.weights,
        Math.max(32, Math.ceil(plan.rowCount / 1024))
      ).tileCount,
      maximumPower: 2,
      probeTotal,
      batchProbeCount: Math.min(
        probeTotal,
        Math.max(1, Math.floor(GPU_SPATIAL_WEIGHTS_TRACES_SCRATCH_ELEMENT_LIMIT / plan.rowCount))
      ),
      stochastic,
      includeGram: true,
      approximation: true
    });
  }
  return count;
}

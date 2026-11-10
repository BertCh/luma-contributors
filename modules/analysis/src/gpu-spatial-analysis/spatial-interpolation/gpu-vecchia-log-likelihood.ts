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
  validateCapacity,
  validateViewLength
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {validatePredictionViews} from '../../analysis-core/kernels/gpu-prediction-utils';
import {GPU_VECCHIA_PARAMETER_LENGTH} from './vecchia-parameters';
import {GPULocalKriging} from './gpu-local-kriging';
import {
  createVecchiaLogLikelihoodSumNodes,
  type GPUVecchiaLikelihoodOutput
} from './vecchia-likelihood-nodes';

const OPERATION = 'GPUVecchiaLogLikelihood';
/** Every conditioning-set solve goes through the `GPULocalKriging` private arrays. */
const MAXIMUM_NEIGHBOR_COUNT = 32;

/** Outputs of {@link GPUVecchiaLogLikelihood}, all caller-owned. */
export type GPUVecchiaLogLikelihoodOutput = GPUVecchiaLikelihoodOutput;

/** Props of {@link GPUVecchiaLogLikelihood}. Counts and which views exist are compile-time. */
export type GPUVecchiaLogLikelihoodProps = {
  id?: string;
  /** One packed row per site, in any row order the conditioning sets refer to. */
  positions: GraphDataView<'float32x2'>;
  /**
   * Rows `[value, observationErrorVariance, reserved, validity (0 excludes)]`. Values are residuals
   * from any known mean (simple kriging, mean zero); subtract an estimated trend first.
   */
  observations: GraphDataView<'float32x4'>;
  /** CSR offsets (`rowCount + 1`) of `buildVecchiaConditioningSets`. */
  neighborhoodOffsets: GraphDataView<'uint32'>;
  /** Earlier rows of the ordering; an empty set makes a site's term the marginal one. */
  neighborhoods: GraphDataView<'uint32'>;
  /** Largest conditioning set, 1..32. Compile time. */
  neighborCount: number;
  /**
   * Written by `getGPUVecchiaLogLikelihoodParameterValues`. Per-frame: model, nugget, sill and range
   * change without a recompile, the point of a likelihood evaluation.
   */
  parameters: GraphDataView<'float32'>;
  output: GPUVecchiaLogLikelihoodOutput;
};

/**
 * Log-likelihood of a zero-mean Gaussian process under the Vecchia approximation
 * (Vecchia 1988; Datta et al. 2016): `log L = sum_i -1/2 [log 2 pi + log d_i + r_i^2 / d_i]`, where
 * `d_i` and the mean of `r_i = y_i - mean_i` are the kriging conditional variance and mean of site
 * `i` given its earlier conditioning set.
 *
 * Each conditioning-set solve is the `simple` mode of `GPULocalKriging` over the caller's ordered
 * CSR neighbourhoods, whose per-frame `parameters` make this a likelihood evaluation with no
 * recompile. A site's term reads `d_i` from `values.y` and adds the observation error variance of
 * that site. Sites with an empty conditioning set use the marginal variance `sill + nugget`.
 *
 * Failure semantics (float32 experiment, dev-docs/research/spatiotemporal-review-2026-11-09): a site
 * whose solve reports failure (status not 0), or whose variance is nonpositive, counts in `status[0]`,
 * and any such site makes `logLikelihood` NaN. A caller reads that word, not a partial sum.
 * Excluded sites (validity 0) count in `status[1]` and drop out.
 *
 * The total is a fixed-point sum, so it is exact for the float32 per-site terms (order independent),
 * and is published as a double-single pair so a CPU optimizer can resolve differences near 1e-3 at
 * a magnitude of 1e6.
 *
 * Compile-time: row count, conditioning-set width, the wiring of optional views. Per-frame: the
 * covariance `parameters`.
 */
export class GPUVecchiaLogLikelihood implements GPUCommandNodeProducer {
  readonly id: string;
  readonly props: GPUVecchiaLogLikelihoodProps;
  readonly rowCount: number;
  private readonly kriging: GPULocalKriging;

  constructor(props: GPUVecchiaLogLikelihoodProps) {
    this.props = props;
    this.id = props.id ?? 'vecchia-log-likelihood';
    const {id} = this;
    const {positions, observations, parameters, neighborhoods, neighborhoodOffsets, output} = props;
    this.rowCount = positions.length;
    validateCapacity(id, 'neighborCount', props.neighborCount, 1, MAXIMUM_NEIGHBOR_COUNT);
    validateViewLength(id, 'observations', observations, this.rowCount);
    validateViewLength(id, 'neighborhoodOffsets', neighborhoodOffsets, this.rowCount + 1);
    validateViewLength(id, 'output.conditionals', output.conditionals, this.rowCount);
    validateViewLength(id, 'output.conditionalStatus', output.conditionalStatus, this.rowCount);
    validateViewLength(id, 'output.siteTerms', output.siteTerms, this.rowCount);
    validateViewLength(id, 'output.siteStatus', output.siteStatus, this.rowCount);
    validateViewLength(id, 'output.logLikelihood', output.logLikelihood, 2);
    validateViewLength(id, 'output.status', output.status, 3);
    validateViewLength(id, 'parameters', parameters, GPU_VECCHIA_PARAMETER_LENGTH);
    validatePredictionViews(
      id,
      [positions, observations, neighborhoodOffsets, neighborhoods, parameters],
      [
        output.conditionals,
        output.conditionalStatus,
        output.siteTerms,
        output.siteStatus,
        output.logLikelihood,
        output.status
      ]
    );
    this.kriging = new GPULocalKriging({
      id: `${id}-solve`,
      positions,
      observations,
      supports: positions,
      mode: 'simple',
      parameters,
      neighborCount: props.neighborCount,
      neighborhoods,
      neighborhoodOffsets,
      output: {values: output.conditionals, status: output.conditionalStatus}
    });
  }

  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, rowCount} = this;
    const {observations, neighborhoodOffsets, parameters, output} = props;
    return [
      ...this.kriging.getCommandNodes(graph),
      ...createVecchiaLogLikelihoodSumNodes(graph, {
        id,
        operation: OPERATION,
        rowCount,
        observations,
        neighborhoodOffsets,
        parameters,
        output
      })
    ];
  }
}

/**
 * Reads back the outputs of {@link GPUVecchiaLogLikelihood}: the log-likelihood total in binary64,
 * or NaN if any site failed.
 *
 * @param logLikelihood The two-element `[high, low]` output.
 * @param status The three-word status output.
 */
export function getVecchiaLogLikelihoodResult(
  logLikelihood: ArrayLike<number>,
  status: ArrayLike<number>
): {
  logLikelihood: number;
  failedSiteCount: number;
  excludedSiteCount: number;
  usedSiteCount: number;
} {
  const failedSiteCount = status[0];
  return {
    logLikelihood: failedSiteCount > 0 ? Number.NaN : logLikelihood[0] + logLikelihood[1],
    failedSiteCount,
    excludedSiteCount: status[1],
    usedSiteCount: status[2]
  };
}

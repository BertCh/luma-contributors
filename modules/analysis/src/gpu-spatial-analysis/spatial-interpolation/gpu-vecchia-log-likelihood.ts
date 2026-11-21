// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUPackedReadbackValues} from '../../analysis-core/kernels/packed-readback';
import {getVecchiaReadbackValues, decodeVecchiaStatus} from './vecchia-readback';
import {
  snapshotVecchiaLossProps,
  validateVecchiaPorts,
  createVecchiaNeighborhoodStatusNodes,
  type VecchiaNeighborhoodStatus
} from './vecchia-neighborhood-status';
import {MAXIMUM_VECCHIA_NEIGHBOR_COUNT} from './vecchia-conditioning-sets';
import {getInterpolationNeighborhoodLossViews} from './interpolation-neighborhood-loss';

import {
  validateInterpolationPredictionViews,
  validateInterpolationOutputPasses,
  validateInterpolationOutputsDisjointFromInputs
} from './interpolation-view-validation';

import {validateGraphChunkBindingRanges} from '../../analysis-core/kernels/graph-chunk-dispatch';
import {
  snapshotConstructionFields,
  snapshotConstructionSpatialContext
} from '../../analysis-core/kernels/construction-snapshot';
import {validateGPUStorageBindingByteLength} from '../../analysis-core/kernels/gpu-contributor-utils';

import type {
  GPUCommandGraph,
  GPUCommandNode,
  GPUCommandNodeProducer,
  GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  GPU_PLANAR_SPATIAL_CONTEXT_SUPPORT,
  resolveGPUSpatialContext,
  type GPUSpatialContext
} from '../../analysis-core/contracts/spatial-context';
import {
  validateCapacity,
  validateViewLength
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {GPU_VECCHIA_PARAMETER_LENGTH} from './vecchia-parameters';
import {GPULocalKriging} from './gpu-local-kriging';
import {
  createVecchiaLogLikelihoodSumNodes,
  type GPUVecchiaLikelihoodOutput
} from './vecchia-likelihood-nodes';

const OPERATION = 'GPUVecchiaLogLikelihood';
/** Every conditioning-set solve goes through the `GPULocalKriging` private arrays. */
const MAXIMUM_NEIGHBOR_COUNT = MAXIMUM_VECCHIA_NEIGHBOR_COUNT;

/** Outputs of {@link GPUVecchiaLogLikelihood}, all caller-owned. */
export type GPUVecchiaLogLikelihoodOutput = GPUVecchiaLikelihoodOutput;

/** Distances are planar Euclidean in the native units of the positions. */
const PLANAR_DISTANCE_SPATIAL_CONTEXT_SUPPORT = Object.freeze({
  ...GPU_PLANAR_SPATIAL_CONTEXT_SUPPORT,
  measure: 'distance' as const
});

/** Props of {@link GPUVecchiaLogLikelihood}. Counts and which views exist are compile-time. */
export type GPUVecchiaLogLikelihoodProps = {
  id?: string;
  /**
   * Coordinate space of the positions. Compile-time. Omitted means native planar coordinates with
   * Euclidean distance (unchanged behavior); only that is accepted, so longitude/latitude throws:
   * project to a planar frame first.
   */
  spatialContext?: GPUSpatialContext;
  /** One packed row per site, in any row order the conditioning sets refer to. */
  positions: GraphDataView<'float32x2'>;
  /**
   * Rows `[value, observationErrorVariance, reserved, validity (> 0 includes; zero, negative and NaN exclude)]`. Values are residuals
   * from any known mean (simple kriging, mean zero); subtract an estimated trend first.
   */
  observations: GraphDataView<'float32x4'>;
  /** CSR offsets (`rowCount + 1`) of `buildVecchiaConditioningSets`. */
  neighborhoodOffsets: GraphDataView<'uint32'>;
  /** Earlier rows of the ordering; an empty set makes a site's term the marginal one. */
  neighborhoods: GraphDataView<'uint32'>;
  /** Largest conditioning set, 1..32. Compile time. */
  neighborCount: number;
  /** Optional producer loss flags; any supplied loss suppresses dependent totals. */
  neighborhoodStatus?: VecchiaNeighborhoodStatus;
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
 * that site. Sites with an empty conditioning set, and sites whose conditioning rows are all
 * unusable (validity not strictly positive, or a non-finite value, error variance or position), use the marginal
 * variance `sill + nugget`, as `GPUVecchiaFisherScoring` does; `conditionalStatus` is 0 for them.
 *
 * Failure semantics (float32 experiment, dev-docs/research/spatiotemporal-review-2026-11-09): a site
 * whose solve reports failure (status not 0), or whose variance is nonpositive, counts in `status[0]`,
 * and any such site makes `logLikelihood` NaN. A caller reads that word, not a partial sum.
 * Excluded sites (validity not strictly positive) count in `status[1]` and drop out.
 *
 * The total is a fixed-point sum, so it is exact for the float32 per-site terms (order independent),
 * and is published as a double-single pair so a CPU optimizer can resolve differences near 1e-3 at
 * a magnitude of 1e6.
 *
 * Compile-time: row count, conditioning-set width, the wiring of optional views. Per-frame: the
 * covariance `parameters`.
 * The GPU spatial route retains the nearest at most 32 usable conditioning rows per site
 * (MAXIMUM_VECCHIA_NEIGHBOR_COUNT); CPU evaluation uses all usable CSR slots.
 */
export class GPUVecchiaLogLikelihood implements GPUCommandNodeProducer {
  readonly id: string;
  /** Resolved coordinate space (native planar when `spatialContext` was omitted). */
  readonly spatialContext: GPUSpatialContext;
  readonly props: GPUVecchiaLogLikelihoodProps;
  readonly rowCount: number;
  private readonly kriging: GPULocalKriging;

  constructor(props: GPUVecchiaLogLikelihoodProps) {
    validateVecchiaPorts(props.id ?? 'vecchia-log-likelihood', props, false, false);
    const originalProps = props;
    const captured = snapshotConstructionFields(props, [
      'id',
      'neighborCount',
      'neighborhoodOffsets',
      'neighborhoods',
      'observations',
      'output',
      'parameters',
      'positions',
      'spatialContext'
    ]);
    this.props = props = Object.freeze({
      ...captured,
      spatialContext:
        captured.spatialContext && snapshotConstructionSpatialContext(captured.spatialContext),
      output: snapshotConstructionFields(captured.output, [
        'conditionals',
        'conditionalStatus',
        'siteTerms',
        'siteStatus',
        'logLikelihood',
        'status'
      ])
    });
    this.props = props = snapshotVecchiaLossProps(originalProps, () => props);
    const lossViews = getInterpolationNeighborhoodLossViews(
      props.id ?? 'vecchia-log-likelihood',
      props
    );
    this.id = props.id ?? 'vecchia-log-likelihood';
    this.spatialContext = resolveGPUSpatialContext(
      this.id,
      props.spatialContext,
      PLANAR_DISTANCE_SPATIAL_CONTEXT_SUPPORT
    );
    const {id} = this;
    const {positions, observations, parameters, neighborhoods, neighborhoodOffsets, output} = props;
    this.rowCount = positions.length;
    if (output.neighborhoodStatus)
      validateViewLength(id, 'output.neighborhoodStatus', output.neighborhoodStatus, 3);
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
    validateInterpolationOutputPasses(id, output, [
      {write: {conditionals: output.conditionals, conditionalStatus: output.conditionalStatus}},
      {
        read: {conditionals: output.conditionals, conditionalStatus: output.conditionalStatus},
        write: {siteTerms: output.siteTerms, siteStatus: output.siteStatus, status: output.status}
      },
      {read: {status: output.status}, write: {logLikelihood: output.logLikelihood}},
      {write: {logLikelihood: output.logLikelihood, neighborhoodStatus: output.neighborhoodStatus}}
    ]);
    validateInterpolationOutputsDisjointFromInputs(id, Object.values(output), [
      positions,
      observations,
      neighborhoodOffsets,
      neighborhoods,
      parameters,
      ...lossViews
    ]);
    validateInterpolationPredictionViews(
      id,
      [positions, observations, neighborhoodOffsets, neighborhoods, parameters, ...lossViews],
      [
        output.conditionals,
        output.conditionalStatus,
        output.siteTerms,
        output.siteStatus,
        output.logLikelihood,
        output.status,
        ...(output.neighborhoodStatus ? [output.neighborhoodStatus] : [])
      ]
    );
    this.kriging = new GPULocalKriging({
      id: `${id}-solve`,
      spatialContext: this.spatialContext,
      positions,
      observations,
      supports: positions,
      mode: 'simple',
      parameters,
      neighborCount: props.neighborCount,
      neighborhoods,
      neighborhoodOffsets,
      // A site whose conditioning rows are all unusable is marginal, as in GPUVecchiaFisherScoring.
      emptyNeighborhood: 'prior',
      output: {values: output.conditionals, status: output.conditionalStatus}
    });
  }

  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, rowCount} = this;
    validateGraphChunkBindingRanges(graph, id, {
      positions: props.positions,
      observations: props.observations,
      neighborhoodOffsets: props.neighborhoodOffsets,
      neighborhoods: props.neighborhoods,
      parameters: props.parameters,
      ...Object.fromEntries(
        getInterpolationNeighborhoodLossViews(id, props).map((view, index) => [
          `neighborhoodLoss${index}`,
          view
        ])
      ),
      ...props.output
    });
    validateGPUStorageBindingByteLength(
      graph.device,
      id,
      'fixed-point words',
      3 * 4,
      'reduce the likelihood cell count'
    );
    validateGPUStorageBindingByteLength(
      graph.device,
      id,
      'fixed-point scale',
      4,
      'use a device with storage bindings'
    );
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
      }),
      ...createVecchiaNeighborhoodStatusNodes(
        graph,
        id,
        props.neighborhoodStatus,
        output.logLikelihood,
        output.neighborhoodStatus
      )
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
  logLikelihood: GPUPackedReadbackValues,
  status: GPUPackedReadbackValues
): {
  logLikelihood: number;
  failedSiteCount: number;
  excludedSiteCount: number;
  usedSiteCount: number;
} {
  logLikelihood = getVecchiaReadbackValues(
    'getVecchiaLogLikelihoodResult',
    'logLikelihood',
    logLikelihood,
    2,
    'float32'
  );
  status = getVecchiaReadbackValues('getVecchiaLogLikelihoodResult', 'status', status, 3, 'uint32');
  const failedSiteCount = status[0];
  return {
    logLikelihood: failedSiteCount > 0 ? Number.NaN : logLikelihood[0] + logLikelihood[1],
    failedSiteCount,
    excludedSiteCount: status[1],
    usedSiteCount: status[2]
  };
}

/** Decodes the three solve counts and optional input-neighborhood loss evidence. */
export function decodeGPUVecchiaLogLikelihoodStatus(
  status: GPUPackedReadbackValues,
  neighborhoodStatus?: GPUPackedReadbackValues
) {
  return decodeVecchiaStatus('decodeGPUVecchiaLogLikelihoodStatus', status, neighborhoodStatus);
}

/** Reads each port with its declared scalar type, including raw buffer readback byte ranges. */
export function decodeGPUVecchiaLogLikelihoodResult(ports: {
  logLikelihood: Float32Array | Uint8Array | ArrayBuffer;
  status: Uint32Array | Uint8Array | ArrayBuffer;
  neighborhoodStatus?: Uint32Array | Uint8Array | ArrayBuffer;
}) {
  const totals = getVecchiaReadbackValues(
    'decodeGPUVecchiaLogLikelihoodResult',
    'logLikelihood',
    ports.logLikelihood,
    2,
    'float32',
    true
  );
  const result = getVecchiaLogLikelihoodResult(totals, ports.status);
  const status = decodeGPUVecchiaLogLikelihoodStatus(ports.status, ports.neighborhoodStatus);
  status.complete = status.complete && Number.isFinite(result.logLikelihood);
  return {...result, logLikelihood: status.complete ? result.logLikelihood : NaN, status};
}

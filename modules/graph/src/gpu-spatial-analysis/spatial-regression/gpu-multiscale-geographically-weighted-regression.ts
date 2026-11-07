// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma-contributors/graph/gpu-core';
import {createMultiscaleInference} from './multiscale-inference';
import {createWGSLKernelNode} from '../../utils/wgsl-kernel-nodes';
import {GET_NAN_WGSL, IS_FINITE_BITS_WGSL} from '../../utils/wgsl-common';
import {
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validatePackedView
} from '../../utils/gpu-contributor-utils';
import {
  GPU_PLANAR_OR_GREAT_CIRCLE_SPATIAL_CONTEXT_SUPPORT,
  getGPUSphereRadius,
  resolveGPUSpatialContext,
  type GPUSpatialContext
} from '../../analysis-core/contracts/spatial-context';
import {getCandidateLoopWGSL} from './local-weighted-least-squares-wgsl';
import {
  GPU_LOCAL_REGRESSION_MAXIMUM_PREDICTOR_COUNT,
  createLocalRegressionGrid,
  getRegressionBindings,
  getRegressionWGSL,
  validateRegressionInputs,
  type LocalRegressionInputs
} from './gpu-generalized-geographically-weighted-regression';
import {getCholeskyWGSL} from './spatial-regression-solve';

const CONVERGENCE_WORKGROUP_SIZE = 256;
const CONVERGENCE_MAXIMUM_GROUP_COUNT = 256;
const CONVERGENCE_ELEMENTS_PER_GROUP = 1024;

export type GPUMultiscaleGeographicallyWeightedRegressionProps = Omit<
  LocalRegressionInputs,
  | 'offset'
  | 'parameters'
  | 'inference'
  | 'bandwidthSelection'
  | 'leaveOneOut'
  | 'predictExcluded'
  | 'bandwidthMode'
  | 'maximumNeighborCount'
> & {
  /** `[relativeContributionTolerance, interceptBandwidth, predictorBandwidth0, ...]`. */
  parameters: GraphDataView<'float32'>;
  /** Optional column-major candidate bandwidths, including intercept, each with candidateCount entries. */
  bandwidthCandidates?: GraphDataView<'float32'>;
  bandwidthCandidateCount?: number;
  /** Known homoskedastic error variance, required for conditionalStandardErrors. */
  noiseVariance?: GraphDataView<'float32'>;
  /** Full backfitting response-smoother propagation, conditional on selected bandwidth history. */
  inference?: {
    varianceConvention?: 'residual-df' | 'smoother-df';
    output: {
      standardErrors: GraphDataView<'float32'>;
      effectiveParameters: GraphDataView<'float32'>;
      /** trace(S), trace(S' S), estimated variance, residual degrees of freedom. */
      summary: GraphDataView<'float32'>;
      /** 0 valid; 1 unsuccessful fit; 2 nonpositive degrees/variance or nonfinite smoother. */
      status: GraphDataView<'uint32'>;
    };
  };
  output: {
    /** Local coefficients, intercept first. NaN unless the entire backfit converged. */
    coefficients: GraphDataView<'float32'>;
    fitted: GraphDataView<'float32'>;
    /** Per row GPU_ITERATIVE_REGRESSION_STATUS. Global backfit failure propagates to every included row. */
    status: {
      /** One status code per observation. */
      codes: GraphDataView<'uint32'>;
      /** One scalar: number of completed backfit sweeps. */
      iterationCount: GraphDataView<'uint32'>;
    };
    selectedBandwidths?: GraphDataView<'float32'>;
    /** Conditional on all other fitted component surfaces and selected bandwidths; not full MGWR covariance. */
    conditionalStandardErrors?: GraphDataView<'float32'>;
  };
};

/**
 * Gaussian-response multiscale GWR with per-variable distance bandwidths.
 * Cyclic additive backfitting starts all coefficient surfaces at the global least squares fit (zero with full inference). For each variable, regress
 * the partial residual on that variable using its own spatial kernel, then replace that coefficient
 * surface simultaneously at every location. An optional finite candidate grid selects each bandwidth
 * by leave-one-out partial-residual SSE during every sweep (ties choose the first candidate).
 * No valid candidate invalidates the fit. Stop when the largest relative change in any fitted
 * variable contribution is at most tolerance. Fixed graph depth bounds work and propagates singular
 * univariate fits/non-convergence. Conditional coefficient standard errors, when requested, use the
 * caller-supplied noise variance and the univariate linear-smoother weights, holding all other
 * surfaces and selected bandwidths fixed. They do not include backfit or bandwidth-selection
 * uncertainty. Optional inference instead propagates the full coupled response Jacobian,
 * estimates residual variance and reports bandwidth-history-conditioned MGWR standard errors
 * and effective parameters. No readback or regularization.
 * Dense O(iterations * predictors * rows²) traversal is intended for bounded workloads.
 */
export class GPUMultiscaleGeographicallyWeightedRegression implements GPUCommandNodeProducer {
  readonly id: string;
  readonly spatialContext: GPUSpatialContext;
  /** Sphere radius of great-circle distances, or null for planar distances. */
  readonly sphereRadius: number | null;
  constructor(readonly props: GPUMultiscaleGeographicallyWeightedRegressionProps) {
    this.id = props.id ?? 'multiscale-geographically-weighted-regression';
    this.spatialContext = resolveGPUSpatialContext(this.id, props.spatialContext, {
      ...GPU_PLANAR_OR_GREAT_CIRCLE_SPATIAL_CONTEXT_SUPPORT,
      measure: 'distance'
    });
    this.sphereRadius =
      this.spatialContext.metric === 'great-circle'
        ? getGPUSphereRadius(this.spatialContext)
        : null;
    validateRegressionInputs(this.id, props, this.sphereRadius !== null);
    if (props.parameters.length < props.predictorCount + 2)
      throw new Error(`${this.id}: supply tolerance and one bandwidth per coefficient`);
    if (props.bandwidthCandidates) {
      const count = props.bandwidthCandidateCount;
      if (!count || !Number.isInteger(count) || count > 32)
        throw new Error(`${this.id}: candidate count must be 1 to 32`);
      validatePackedView(props.bandwidthCandidates, ['float32'], this.id);
      if (props.bandwidthCandidates.length !== (props.predictorCount + 1) * count)
        throw new Error(`${this.id}: candidate grid shape mismatch`);
    } else if (props.bandwidthCandidateCount !== undefined)
      throw new Error(`${this.id}: candidate count requires candidates`);
    if (props.output.conditionalStandardErrors && !props.noiseVariance)
      throw new Error(`${this.id}: conditional inference requires known noise variance`);
    if (props.noiseVariance) {
      validatePackedView(props.noiseVariance, ['float32'], this.id);
      if (props.noiseVariance.length !== 1)
        throw new Error(`${this.id}: noise variance must be scalar`);
    }
    for (const [name, view] of Object.entries(getOutputColumns(props.output))) {
      if (!view) continue;
      validatePackedView(
        view,
        [
          ['coefficients', 'fitted', 'selectedBandwidths', 'conditionalStandardErrors'].includes(
            name
          )
            ? 'float32'
            : 'uint32'
        ],
        this.id
      );
      const length =
        name === 'iterations'
          ? 1
          : name === 'selectedBandwidths'
            ? props.predictorCount + 1
            : props.response.length *
              (['coefficients', 'conditionalStandardErrors'].includes(name)
                ? props.predictorCount + 1
                : 1);
      if (view.length !== length) throw new Error(`${this.id}: output length mismatch`);
    }
    if (props.inference) {
      if (
        props.response.length > 128 ||
        props.response.length ** 3 * (props.predictorCount + 1) * (props.maximumIterations ?? 32) >
          268435456
      )
        throw new Error(`${this.id}: full inference work budget exceeded`);
      if (
        !['residual-df', 'smoother-df'].includes(
          props.inference.varianceConvention ?? 'residual-df'
        )
      )
        throw new Error(`${this.id}: unsupported variance convention`);
      for (const [name, view] of Object.entries(props.inference.output)) {
        validatePackedView(view, [name === 'status' ? 'uint32' : 'float32'], this.id);
        const length =
          name === 'standardErrors'
            ? props.response.length * (props.predictorCount + 1)
            : name === 'effectiveParameters'
              ? props.predictorCount + 1
              : name === 'summary'
                ? 4
                : 1;
        if (view.length !== length) throw new Error(`${this.id}: inference shape mismatch`);
      }
    }
    validateGraphOutputsDisjointFromInputs(
      this.id,
      [
        ...Object.values(getOutputColumns(props.output)),
        ...Object.values(props.inference?.output ?? {})
      ],
      [
        ...getRegressionBindings(props).map(binding => binding.view),
        props.bandwidthCandidates,
        props.noiseVariance
      ]
    );
  }
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const rows = props.response.length;
    const columns = props.predictorCount + 1;
    const maximumIterations = props.maximumIterations ?? 32;
    const first = createTransientView(graph, `${id}-first`, 'float32', rows * columns);
    const second = createTransientView(graph, `${id}-second`, 'float32', rows * columns);
    const previous = createTransientView(graph, `${id}-previous`, 'float32', rows * columns);
    const state = createTransientView(graph, `${id}-state`, 'uint32', 2);
    const workingParameters = createTransientView(
      graph,
      `${id}-working-parameters`,
      'float32',
      columns + 1
    );
    // Per-workgroup partials (largest change, first non-finite element key, included row count).
    const convergenceGroupCount = Math.min(
      CONVERGENCE_MAXIMUM_GROUP_COUNT,
      Math.max(1, Math.ceil((rows * columns) / CONVERGENCE_ELEMENTS_PER_GROUP))
    );
    const convergencePartials = createTransientView(
      graph,
      `${id}-convergence-partials`,
      'uint32',
      3 * convergenceGroupCount
    );
    const candidateCount = props.bandwidthCandidateCount ?? 1;
    const candidateScores = props.bandwidthCandidates
      ? createTransientView(graph, `${id}-candidate-scores`, 'float32', rows * candidateCount)
      : undefined;
    const gridSetup = createLocalRegressionGrid(graph, id, props, this.sphereRadius);
    const effective = gridSetup.props;
    const isGrid = gridSetup.grid !== undefined;
    const bindings = getRegressionBindings({...effective, parameters: workingParameters});
    const common = getRegressionWGSL(effective, {sphereRadius: this.sphereRadius});
    // Grid kernels bind positions and a per-row (partial response, predictor value) pair instead
    // of the predictors and response, so they stay within eight storage buffers.
    const gridBindings = gridSetup.grid
      ? [
          {
            name: 'cellOffsets',
            view: gridSetup.grid.cellOffsets,
            type: 'u32' as const,
            access: 'read' as const
          },
          {
            name: 'sortedIds',
            view: gridSetup.grid.sortedIds,
            type: 'u32' as const,
            access: 'read' as const
          }
        ]
      : [];
    const gridCommon = `${getRegressionWGSL(effective, {
      sphereRadius: this.sphereRadius,
      grid: props.indexGridSize,
      neighborhoodOnly: true
    })}
fn isActive(row: u32) -> bool { return isFiniteBits(pairs[pairsOffset + 2u * row]); }`;
    const pairs = isGrid
      ? createTransientView(graph, `${id}-pairs`, 'float32', 2 * rows)
      : undefined;
    const loop = (body: string) => `{\n${getCandidateLoopWGSL(body, true, 'isActive(row)')}\n}`;
    validateGraphViewsBelongToGraph(id, graph, [
      ...bindings.map(binding => binding.view),
      ...Object.values(getOutputColumns(props.output)),
      props.parameters,
      props.bandwidthCandidates,
      props.noiseVariance
    ]);
    // Full-inference propagation starts from zero surfaces, so only fits without it start every
    // surface at the global ordinary least squares coefficients, as mgwr does.
    const globalFit = props.inference
      ? undefined
      : createTransientView(graph, `${id}-global-fit`, 'float32', columns);
    const nodes: GPUCommandNode<Parameters>[] = [...gridSetup.nodes];
    if (globalFit) {
      nodes.push(
        createWGSLKernelNode(graph, {
          id: `${id}-global-fit`,
          operation: 'GPUMultiscaleGeographicallyWeightedRegression',
          variant: 'global-fit',
          invocationCount: 1,
          bindings: [
            ...bindings.filter(binding => binding.name !== 'parameters'),
            {name: 'globalFit', view: globalFit, type: 'f32', access: 'read_write'}
          ],
          declarations: `${common}\n${getCholeskyWGSL(props.predictorCount, GPU_LOCAL_REGRESSION_MAXIMUM_PREDICTOR_COUNT + 1)}`,
          body: `// Centered, equilibrated normal equations of the included rows; zeros when singular.
for (var column = 0u; column < ${columns}u; column++) { globalFit[globalFitOffset + column] = 0.0; }
var means: array<f32, ${props.predictorCount}>;
var responseMean = 0.0;
var count = 0u;
for (var row = 0u; row < ${rows}u; row++) {
  if (!included(row)) { continue; }
  count++;
  let weight = 1.0 / f32(count);
  responseMean += weight * (response[responseOffset + row] - responseMean);
  for (var column = 0u; column < ${props.predictorCount}u; column++) {
    means[column] += weight * (predictor(row, column + 1u) - means[column]);
  }
}
if (count < ${columns}u) { return; }
var gram: array<f32, ${props.predictorCount * props.predictorCount}>;
var cross: array<f32, ${props.predictorCount}>;
for (var row = 0u; row < ${rows}u; row++) {
  if (!included(row)) { continue; }
  let centeredResponse = response[responseOffset + row] - responseMean;
  for (var column = 0u; column < ${props.predictorCount}u; column++) {
    let first = predictor(row, column + 1u) - means[column];
    cross[column] += first * centeredResponse;
    for (var other = 0u; other <= column; other++) {
      gram[column * ${props.predictorCount}u + other] += first * (predictor(row, other + 1u) - means[other]);
    }
  }
}
var equilibration: array<f32, ${props.predictorCount}>;
for (var column = 0u; column < ${props.predictorCount}u; column++) {
  let diagonal = gram[column * ${props.predictorCount + 1}u];
  if (!(diagonal > 0.0) || !isFiniteBits(diagonal)) { return; }
  equilibration[column] = inverseSqrt(diagonal);
}
for (var column = 0u; column < ${props.predictorCount}u; column++) {
  cross[column] *= equilibration[column];
  for (var other = 0u; other <= column; other++) { gram[column * ${props.predictorCount}u + other] *= equilibration[column] * equilibration[other]; }
}
if (!choleskyFactor_${props.predictorCount}(&gram)) { return; }
choleskySolve_${props.predictorCount}(&gram, &cross);
var intercept = responseMean;
for (var column = 0u; column < ${props.predictorCount}u; column++) {
  let slope = cross[column] * equilibration[column];
  if (!isFiniteBits(slope)) { return; }
  globalFit[globalFitOffset + column + 1u] = slope;
  intercept -= slope * means[column];
}
if (isFiniteBits(intercept)) { globalFit[globalFitOffset] = intercept; }`
        })
      );
    }
    nodes.push(
      createWGSLKernelNode(graph, {
        id: `${id}-initialize`,
        operation: 'GPUMultiscaleGeographicallyWeightedRegression',
        variant: 'initialize',
        invocationCount: rows * columns,
        bindings: [
          {name: 'coefficients', view: first, type: 'f32', access: 'read_write'},
          {name: 'state', view: state, type: 'u32', access: 'read_write'},
          ...(globalFit
            ? [{name: 'globalFit', view: globalFit, type: 'f32' as const, access: 'read' as const}]
            : [])
        ],
        body: `coefficients[coefficientsOffset + index] = ${globalFit ? `globalFit[globalFitOffset + index % ${columns}u]` : '0.0'};
if (index == 0u) { state[stateOffset] = 3u; state[stateOffset + 1u] = 0u; }`
      })
    );
    nodes.push(
      createWGSLKernelNode(graph, {
        id: `${id}-parameters`,
        operation: 'GPUMultiscaleGeographicallyWeightedRegression',
        variant: 'parameters',
        invocationCount: columns + 1,
        bindings: [
          {name: 'source', view: props.parameters, type: 'f32', access: 'read'},
          {name: 'parameters', view: workingParameters, type: 'f32', access: 'read_write'}
        ],
        body: 'parameters[parametersOffset + index] = source[sourceOffset + index];'
      })
    );
    const inference = props.inference
      ? createMultiscaleInference(graph, this, workingParameters, state)
      : undefined;
    if (inference) nodes.push(inference.initialize());
    let current = first;
    let next = second;
    for (let iteration = 0; iteration < maximumIterations; iteration++) {
      nodes.push(
        createWGSLKernelNode(graph, {
          id: `${id}-snapshot-${iteration}`,
          operation: 'GPUMultiscaleGeographicallyWeightedRegression',
          variant: 'snapshot',
          invocationCount: rows * columns,
          bindings: [
            {name: 'source', view: current, type: 'f32', access: 'read'},
            {name: 'destination', view: previous, type: 'f32', access: 'read_write'}
          ],
          body: 'destination[destinationOffset + index] = source[sourceOffset + index];'
        })
      );
      for (let column = 0; column < columns; column++) {
        if (isGrid && pairs) {
          nodes.push(
            createWGSLKernelNode(graph, {
              id: `${id}-prepare-${iteration}-${column}`,
              operation: 'GPUMultiscaleGeographicallyWeightedRegression',
              variant: `prepare-${column}`,
              invocationCount: rows,
              bindings: [
                ...bindings.filter(binding => binding.name !== 'parameters'),
                {name: 'source', view: current, type: 'f32', access: 'read'},
                {name: 'destination', view: next, type: 'f32', access: 'read_write'},
                {name: 'pairs', view: pairs, type: 'f32', access: 'read_write'}
              ],
              declarations: common,
              body: `var partial = response[responseOffset + index];
for (var other = 0u; other < ${columns}u; other++) {
  destination[destinationOffset + index * ${columns}u + other] = source[sourceOffset + index * ${columns}u + other];
  if (other != ${column}u) { partial -= predictor(index, other) * source[sourceOffset + index * ${columns}u + other]; }
}
pairs[pairsOffset + 2u * index] = select(getNaN(), partial, included(index));
pairs[pairsOffset + 2u * index + 1u] = predictor(index, ${column}u);`
            })
          );
        }
        if (props.bandwidthCandidates && candidateScores) {
          nodes.push(
            isGrid && pairs
              ? createWGSLKernelNode(graph, {
                  id: `${id}-cv-${iteration}-${column}`,
                  operation: 'GPUMultiscaleGeographicallyWeightedRegression',
                  variant: `cv-${column}`,
                  invocationCount: rows * candidateCount,
                  bindings: [
                    {name: 'positions', view: effective.positions, type: 'f32', access: 'read'},
                    {name: 'pairs', view: pairs, type: 'f32', access: 'read'},
                    {
                      name: 'candidates',
                      view: props.bandwidthCandidates,
                      type: 'f32',
                      access: 'read'
                    },
                    {name: 'scores', view: candidateScores, type: 'f32', access: 'read_write'},
                    {name: 'state', view: state, type: 'u32', access: 'read'},
                    ...gridBindings
                  ],
                  declarations: gridCommon,
                  body: `if (state[stateOffset] != 3u) { scores[scoresOffset + index] = 0.0; return; }
let focus = index / ${candidateCount}u; let candidate = index % ${candidateCount}u;
var score = 0.0;
if (isActive(focus)) {
  score = getNaN(); let bandwidth = candidates[candidatesOffset + ${column * candidateCount}u + candidate];
  if (bandwidth > 0.0 && isFiniteBits(bandwidth)) {
    var numerator = 0.0; var denominator = 0.0;
    let origin = getPosition(focus);
    ${loop(`if (row == focus) { continue; }
      let weight = spatialWeight(focus, row, bandwidth); if (!(weight > 0.0)) { continue; }
      let value = pairs[pairsOffset + 2u * row + 1u]; let partial = pairs[pairsOffset + 2u * row];
      numerator += weight * value * partial; denominator += weight * value * value;`)}
    if (denominator > 0.0 && isFiniteBits(denominator)) { let residual = pairs[pairsOffset + 2u * focus] - pairs[pairsOffset + 2u * focus + 1u] * numerator / denominator; score = residual * residual; }
  }
}
scores[scoresOffset + index] = score;`
                })
              : createWGSLKernelNode(graph, {
                  id: `${id}-cv-${iteration}-${column}`,
                  operation: 'GPUMultiscaleGeographicallyWeightedRegression',
                  variant: `cv-${column}`,
                  invocationCount: rows * candidateCount,
                  bindings: [
                    ...bindings.filter(binding => binding.name !== 'parameters'),
                    {name: 'source', view: current, type: 'f32', access: 'read'},
                    {
                      name: 'candidates',
                      view: props.bandwidthCandidates,
                      type: 'f32',
                      access: 'read'
                    },
                    {name: 'scores', view: candidateScores, type: 'f32', access: 'read_write'},
                    {name: 'state', view: state, type: 'u32', access: 'read'}
                  ],
                  declarations: `${common}
fn partialResponse(row: u32) -> f32 {
  var partial = response[responseOffset + row];
  for (var other = 0u; other < ${columns}u; other++) { if (other != ${column}u) { partial -= predictor(row, other) * source[sourceOffset + row * ${columns}u + other]; } }
  return partial;
}`,
                  body: `if (state[stateOffset] != 3u) { scores[scoresOffset + index] = 0.0; return; }
let focus = index / ${candidateCount}u; let candidate = index % ${candidateCount}u;
var score = 0.0;
if (included(focus)) {
  score = getNaN(); let bandwidth = candidates[candidatesOffset + ${column * candidateCount}u + candidate];
  if (bandwidth > 0.0 && isFiniteBits(bandwidth)) {
    var numerator = 0.0; var denominator = 0.0;
    for (var row = 0u; row < ${rows}u; row++) {
      if (row == focus || !included(row)) { continue; }
      let weight = spatialWeight(focus, row, bandwidth); if (!(weight > 0.0)) { continue; }
      let value = predictor(row, ${column}u); numerator += weight * value * partialResponse(row); denominator += weight * value * value;
    }
    if (denominator > 0.0 && isFiniteBits(denominator)) { let residual = partialResponse(focus) - predictor(focus, ${column}u) * numerator / denominator; score = residual * residual; }
  }
}
scores[scoresOffset + index] = score;`
                })
          );
          nodes.push(
            createWGSLKernelNode(graph, {
              id: `${id}-select-${iteration}-${column}`,
              operation: 'GPUMultiscaleGeographicallyWeightedRegression',
              variant: `select-${column}`,
              invocationCount: 1,
              bindings: [
                {name: 'scores', view: candidateScores, type: 'f32', access: 'read'},
                {name: 'candidates', view: props.bandwidthCandidates, type: 'f32', access: 'read'},
                {name: 'parameters', view: workingParameters, type: 'f32', access: 'read_write'},
                {name: 'state', view: state, type: 'u32', access: 'read'}
              ],
              declarations: `${GET_NAN_WGSL}\n${IS_FINITE_BITS_WGSL}`,
              body: `if (state[stateOffset] != 3u) { return; }
var best = getNaN(); var bestScore = 3.402823e38;
for (var candidate = 0u; candidate < ${candidateCount}u; candidate++) {
  var sum = 0.0;
  for (var row = 0u; row < ${rows}u; row++) { sum += scores[scoresOffset + row * ${candidateCount}u + candidate]; }
  if (isFiniteBits(sum) && sum < bestScore) { bestScore = sum; best = candidates[candidatesOffset + ${column * candidateCount}u + candidate]; }
}
parameters[parametersOffset + ${column + 1}u] = best;`
            })
          );
        }
        nodes.push(
          isGrid && pairs
            ? createWGSLKernelNode(graph, {
                id: `${id}-sweep-${iteration}-column-${column}`,
                operation: 'GPUMultiscaleGeographicallyWeightedRegression',
                variant: `column-${column}`,
                invocationCount: rows,
                bindings: [
                  {name: 'positions', view: effective.positions, type: 'f32', access: 'read'},
                  {name: 'pairs', view: pairs, type: 'f32', access: 'read'},
                  {name: 'parameters', view: workingParameters, type: 'f32', access: 'read'},
                  {name: 'destination', view: next, type: 'f32', access: 'read_write'},
                  {name: 'state', view: state, type: 'u32', access: 'read'},
                  ...gridBindings
                ],
                declarations: gridCommon,
                body: `if (state[stateOffset] != 3u) { return; }
var coefficient = getNaN(); let bandwidth = parameters[parametersOffset + ${column + 1}u];
if (isActive(index) && bandwidth > 0.0 && isFiniteBits(bandwidth)) {
  var numerator = 0.0; var denominator = 0.0;
  let origin = getPosition(index);
  ${loop(`let weight = spatialWeight(index, row, bandwidth); if (!(weight > 0.0)) { continue; }
    let value = pairs[pairsOffset + 2u * row + 1u]; let partial = pairs[pairsOffset + 2u * row];
    numerator += weight * value * partial; denominator += weight * value * value;`)}
  if (denominator > 0.0 && isFiniteBits(denominator)) { coefficient = numerator / denominator; }
}
destination[destinationOffset + index * ${columns}u + ${column}u] = coefficient;`
              })
            : createWGSLKernelNode(graph, {
                id: `${id}-sweep-${iteration}-column-${column}`,
                operation: 'GPUMultiscaleGeographicallyWeightedRegression',
                variant: `column-${column}`,
                invocationCount: rows,
                bindings: [
                  ...bindings,
                  {name: 'source', view: current, type: 'f32', access: 'read'},
                  {name: 'destination', view: next, type: 'f32', access: 'read_write'},
                  {name: 'state', view: state, type: 'u32', access: 'read'}
                ],
                declarations: common,
                body: `for (var column = 0u; column < ${columns}u; column++) { destination[destinationOffset + index * ${columns}u + column] = source[sourceOffset + index * ${columns}u + column]; }
if (state[stateOffset] != 3u) { return; }
var coefficient = getNaN(); let bandwidth = parameters[parametersOffset + ${column + 1}u];
if (included(index) && bandwidth > 0.0 && isFiniteBits(bandwidth)) {
  var numerator = 0.0; var denominator = 0.0;
  for (var row = 0u; row < ${rows}u; row++) {
    if (!included(row)) { continue; }
    let weight = spatialWeight(index, row, bandwidth); if (!(weight > 0.0)) { continue; }
    var partial = response[responseOffset + row];
    for (var other = 0u; other < ${columns}u; other++) {
      if (other != ${column}u) { partial -= predictor(row, other) * source[sourceOffset + row * ${columns}u + other]; }
    }
    let value = predictor(row, ${column}u); numerator += weight * value * partial; denominator += weight * value * value;
  }
  if (denominator > 0.0 && isFiniteBits(denominator)) { coefficient = numerator / denominator; }
}
destination[destinationOffset + index * ${columns}u + ${column}u] = coefficient;`
              })
        );
        if (inference) nodes.push(inference.advance(iteration, column));
        [current, next] = [next, current];
      }
      nodes.push(
        createWGSLKernelNode(graph, {
          id: `${id}-convergence-${iteration}`,
          operation: 'GPUMultiscaleGeographicallyWeightedRegression',
          variant: 'convergence',
          invocationCount: convergenceGroupCount * CONVERGENCE_WORKGROUP_SIZE,
          workgroupSize: CONVERGENCE_WORKGROUP_SIZE,
          guardIndex: false,
          bindings: [
            ...bindings.filter(binding => binding.name !== 'parameters'),
            {name: 'source', view: current, type: 'f32', access: 'read'},
            {name: 'previous', view: previous, type: 'f32', access: 'read'},
            {name: 'partials', view: convergencePartials, type: 'u32', access: 'read_write'},
            {name: 'state', view: state, type: 'u32', access: 'read'}
          ],
          declarations: `${common}
var<workgroup> sharedChange: array<f32, ${CONVERGENCE_WORKGROUP_SIZE}>;
var<workgroup> sharedKey: array<u32, ${CONVERGENCE_WORKGROUP_SIZE}>;
var<workgroup> sharedCount: array<u32, ${CONVERGENCE_WORKGROUP_SIZE}>;`,
          // Each thread strides over the flattened (row, column) elements and keeps the largest
          // relative change, the lowest flat element key of a non-finite value, and its included
          // row count; a workgroup tree merges them into one partial. Maximum, minimum and
          // integer sums are order independent, so the decision equals the serial scan exactly.
          body: `let isRunning = state[stateOffset] == 3u;
var localChange = 0.0; var localKey = 0xffffffffu; var localCount = 0u;
let elementCount = select(0u, ${rows * columns}u, isRunning);
for (var element = index; element < elementCount; element += INVOCATION_COUNT) {
  let row = element / ${columns}u; let column = element % ${columns}u;
  if (!included(row)) { continue; }
  if (column == 0u) { localCount++; }
  let coefficient = source[sourceOffset + element];
  // Key bit 0 distinguishes a non-finite contribution (state 4) from a coefficient (state 2).
  if (!isFiniteBits(coefficient)) { localKey = element * 2u; break; }
  let contribution = predictor(row, column) * coefficient;
  let previousContribution = predictor(row, column) * previous[previousOffset + element];
  if (!isFiniteBits(contribution) || !isFiniteBits(previousContribution)) { localKey = element * 2u + 1u; break; }
  localChange = max(localChange, abs(contribution - previousContribution) / max(1.0, abs(previousContribution)));
}
sharedChange[localInvocationIndex] = localChange; sharedKey[localInvocationIndex] = localKey; sharedCount[localInvocationIndex] = localCount;
workgroupBarrier();
for (var stride = ${CONVERGENCE_WORKGROUP_SIZE / 2}u; stride > 0u; stride = stride >> 1u) {
  if (localInvocationIndex < stride) {
    sharedChange[localInvocationIndex] = max(sharedChange[localInvocationIndex], sharedChange[localInvocationIndex + stride]);
    sharedKey[localInvocationIndex] = min(sharedKey[localInvocationIndex], sharedKey[localInvocationIndex + stride]);
    sharedCount[localInvocationIndex] += sharedCount[localInvocationIndex + stride];
  }
  workgroupBarrier();
}
if (localInvocationIndex == 0u) {
  let group = index / ${CONVERGENCE_WORKGROUP_SIZE}u;
  partials[partialsOffset + 3u * group] = bitcast<u32>(sharedChange[0]);
  partials[partialsOffset + 3u * group + 1u] = sharedKey[0];
  partials[partialsOffset + 3u * group + 2u] = sharedCount[0];
}`
        }),
        createWGSLKernelNode(graph, {
          id: `${id}-convergence-decide-${iteration}`,
          operation: 'GPUMultiscaleGeographicallyWeightedRegression',
          variant: 'convergence-decide',
          invocationCount: 1,
          bindings: [
            {name: 'parameters', view: workingParameters, type: 'f32', access: 'read'},
            {name: 'partials', view: convergencePartials, type: 'u32', access: 'read'},
            {name: 'state', view: state, type: 'u32', access: 'read_write'}
          ],
          declarations: IS_FINITE_BITS_WGSL,
          body: `if (state[stateOffset] != 3u) { return; }
state[stateOffset + 1u] += 1u;
let tolerance = parameters[parametersOffset]; if (!(tolerance > 0.0) || !isFiniteBits(tolerance)) { state[stateOffset] = 4u; return; }
var maximumChange = 0.0; var firstKey = 0xffffffffu; var count = 0u;
for (var group = 0u; group < ${convergenceGroupCount}u; group++) {
  maximumChange = max(maximumChange, bitcast<f32>(partials[partialsOffset + 3u * group]));
  firstKey = min(firstKey, partials[partialsOffset + 3u * group + 1u]);
  count += partials[partialsOffset + 3u * group + 2u];
}
if (firstKey != 0xffffffffu) { state[stateOffset] = select(2u, 4u, (firstKey & 1u) == 1u); return; }
if (count == 0u) { state[stateOffset] = 1u; } else if (maximumChange <= tolerance) { state[stateOffset] = 0u; }`
        })
      );
    }
    // Publish separately to keep every kernel within the default eight-storage-buffer limit.
    const outputState = createTransientView(graph, `${id}-output-state`, 'uint32', rows);
    nodes.push(
      createWGSLKernelNode(graph, {
        id: `${id}-row-status`,
        operation: 'GPUMultiscaleGeographicallyWeightedRegression',
        variant: 'row-status',
        invocationCount: rows,
        bindings: [
          ...bindings,
          {name: 'state', view: state, type: 'u32', access: 'read'},
          {name: 'rowStatus', view: outputState, type: 'u32', access: 'read_write'}
        ],
        declarations: common,
        body: 'rowStatus[rowStatusOffset + index] = select(1u, state[stateOffset], included(index));'
      })
    );
    nodes.push(
      createWGSLKernelNode(graph, {
        id: `${id}-publish`,
        operation: 'GPUMultiscaleGeographicallyWeightedRegression',
        variant: 'publish',
        invocationCount: rows,
        bindings: [
          {name: 'source', view: current, type: 'f32', access: 'read'},
          {name: 'rowStatus', view: outputState, type: 'u32', access: 'read'},
          {name: 'predictors', view: props.predictors, type: 'f32', access: 'read'},
          {name: 'state', view: state, type: 'u32', access: 'read'},
          {
            name: 'coefficients',
            view: props.output.coefficients,
            type: 'f32',
            access: 'read_write'
          },
          {name: 'fitted', view: props.output.fitted, type: 'f32', access: 'read_write'},
          {name: 'status', view: props.output.status.codes, type: 'u32', access: 'read_write'},
          {
            name: 'iterations',
            view: props.output.status.iterationCount,
            type: 'u32',
            access: 'read_write'
          }
        ],
        declarations: GET_NAN_WGSL,
        body: `let code = rowStatus[rowStatusOffset + index]; var prediction = 0.0;
for (var column = 0u; column < ${columns}u; column++) {
  let coefficient = select(getNaN(), source[sourceOffset + index * ${columns}u + column], code == 0u);
  coefficients[coefficientsOffset + index * ${columns}u + column] = coefficient;
  var value = 1.0; if (column > 0u) { value = predictors[predictorsOffset + index * ${props.predictorCount}u + column - 1u]; }
  prediction += value * coefficient;
}
fitted[fittedOffset + index] = prediction; status[statusOffset + index] = code;
if (index == 0u) { iterations[iterationsOffset] = state[stateOffset + 1u]; }`
      })
    );
    if (props.output.selectedBandwidths)
      nodes.push(
        createWGSLKernelNode(graph, {
          id: `${id}-publish-bandwidths`,
          operation: 'GPUMultiscaleGeographicallyWeightedRegression',
          variant: 'publish-bandwidths',
          invocationCount: columns,
          bindings: [
            {name: 'parameters', view: workingParameters, type: 'f32', access: 'read'},
            {
              name: 'bandwidths',
              view: props.output.selectedBandwidths,
              type: 'f32',
              access: 'read_write'
            }
          ],
          body: 'bandwidths[bandwidthsOffset + index] = parameters[parametersOffset + index + 1u];'
        })
      );
    if (isGrid && pairs && props.output.conditionalStandardErrors && props.noiseVariance) {
      for (let column = 0; column < columns; column++) {
        nodes.push(
          createWGSLKernelNode(graph, {
            id: `${id}-conditional-prepare-${column}`,
            operation: 'GPUMultiscaleGeographicallyWeightedRegression',
            variant: `conditional-prepare-${column}`,
            invocationCount: rows,
            bindings: [
              ...bindings.filter(binding => binding.name !== 'parameters'),
              {name: 'pairs', view: pairs, type: 'f32', access: 'read_write'}
            ],
            declarations: common,
            body: `pairs[pairsOffset + 2u * index] = select(getNaN(), 0.0, included(index));
pairs[pairsOffset + 2u * index + 1u] = predictor(index, ${column}u);`
          }),
          createWGSLKernelNode(graph, {
            id: `${id}-conditional-inference-${column}`,
            operation: 'GPUMultiscaleGeographicallyWeightedRegression',
            variant: `conditional-inference-${column}`,
            invocationCount: rows,
            bindings: [
              {name: 'positions', view: effective.positions, type: 'f32', access: 'read'},
              {name: 'pairs', view: pairs, type: 'f32', access: 'read'},
              {name: 'parameters', view: workingParameters, type: 'f32', access: 'read'},
              {name: 'noiseVariance', view: props.noiseVariance, type: 'f32', access: 'read'},
              {name: 'state', view: state, type: 'u32', access: 'read'},
              {
                name: 'standardErrors',
                view: props.output.conditionalStandardErrors,
                type: 'f32',
                access: 'read_write'
              },
              ...gridBindings
            ],
            declarations: gridCommon,
            body: `var standardError = getNaN(); let noise = noiseVariance[noiseVarianceOffset];
if (state[stateOffset] == 0u && isActive(index) && noise >= 0.0 && isFiniteBits(noise)) {
  let bandwidth = parameters[parametersOffset + ${column + 1}u]; var numerator = 0.0; var denominator = 0.0;
  let origin = getPosition(index);
  ${loop(`let weight = spatialWeight(index, row, bandwidth); if (!(weight > 0.0)) { continue; }
    let value = pairs[pairsOffset + 2u * row + 1u];
    denominator += weight * value * value; numerator += weight * weight * value * value;`)}
  if (denominator > 0.0) { let estimate = sqrt(noise * numerator) / denominator; if (isFiniteBits(estimate)) { standardError = estimate; } }
}
standardErrors[standardErrorsOffset + index * ${columns}u + ${column}u] = standardError;`
          })
        );
      }
    } else if (props.output.conditionalStandardErrors && props.noiseVariance)
      nodes.push(
        createWGSLKernelNode(graph, {
          id: `${id}-conditional-inference`,
          operation: 'GPUMultiscaleGeographicallyWeightedRegression',
          variant: 'conditional-inference',
          invocationCount: rows * columns,
          bindings: [
            ...bindings,
            {name: 'noiseVariance', view: props.noiseVariance, type: 'f32', access: 'read'},
            {name: 'state', view: state, type: 'u32', access: 'read'},
            {
              name: 'standardErrors',
              view: props.output.conditionalStandardErrors,
              type: 'f32',
              access: 'read_write'
            }
          ],
          declarations: common,
          body: `let focus = index / ${columns}u; let column = index % ${columns}u;
var standardError = getNaN(); let noise = noiseVariance[noiseVarianceOffset];
if (state[stateOffset] == 0u && included(focus) && noise >= 0.0 && isFiniteBits(noise)) {
  let bandwidth = parameters[parametersOffset + column + 1u]; var numerator = 0.0; var denominator = 0.0;
  for (var row = 0u; row < ${rows}u; row++) {
    if (!included(row)) { continue; }
    let weight = spatialWeight(focus, row, bandwidth); let value = predictor(row, column);
    denominator += weight * value * value; numerator += weight * weight * value * value;
  }
  if (denominator > 0.0) { let estimate = sqrt(noise * numerator) / denominator; if (isFiniteBits(estimate)) { standardError = estimate; } }
}
standardErrors[standardErrorsOffset + index] = standardError;`
        })
      );
    if (inference) nodes.push(...inference.finish());
    return nodes;
  }
}

function getOutputColumns(output: GPUMultiscaleGeographicallyWeightedRegressionProps['output']) {
  const {status, ...columns} = output;
  return {...columns, status: status.codes, iterations: status.iterationCount};
}

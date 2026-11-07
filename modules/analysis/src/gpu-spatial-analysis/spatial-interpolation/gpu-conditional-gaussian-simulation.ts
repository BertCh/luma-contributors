// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, assert} from '@luma.gl/core';
import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  createWGSLKernelNode,
  getWGSLFloatLiteral
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {GET_NAN_WGSL, IS_FINITE_VALUE_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {PHILOX_PURPOSES, PHILOX_WGSL} from '../../analysis-core/kernels/philox-random';
import {
  importGraphBuffer,
  validateCapacity,
  validateGPUStorageBindingByteLength,
  validateGraphViewsBelongToGraph
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  getPackedPredictionAccessors,
  validatePredictionViews
} from '../../analysis-core/kernels/gpu-prediction-utils';
import {
  getPredictionCovarianceWGSL,
  validatePredictionCovariance,
  type GPUPredictionCovariance
} from './gpu-local-kriging';
import {getBlockedConditionalSimulationNodes} from './conditional-simulation-blocked';

/** Exact bounded joint conditional Gaussian simulation, with explicit known mean. */
export type GPUConditionalGaussianSimulationProps = {
  id?: string;
  positions: GraphDataView<'float32x2'>;
  /** [value, measurement-error variance, reserved, validity]; zero validity omits the sample. */
  observations: GraphDataView<'float32x4'>;
  queries: GraphDataView<'float32x2'>;
  covariance: GPUPredictionCovariance;
  mean?: number;
  seed: number;
  replicateCount: number;
  /** Stable replicate identity across explicit batches; defaults 0. */
  replicateOffset?: number;
  /**
   * Retain factors and moments between encodings. Invalidate after changing input buffers;
   * destroy compiled graphs before calling destroy() to release the retained storage.
   * The constructor's replicateCount is the maximum mutable draw-batch capacity.
   */
  reusePreparation?: boolean;
  output: {
    /** Replicate-major simulated latent query values. */
    values: GraphDataView<'float32'>;
    /** One [conditional mean, conditional variance] row per query. */
    moments: GraphDataView<'float32x2'>;
    /** One status per replicate: 0 success, 1 invalid input/inactive retained-batch row, 2 numerical failure. */
    status: GraphDataView<'uint32'>;
  };
};

/**
 * Exact dense conditioning for at most 4096 observation and query rows combined.
 * A GPU Cholesky factor of the joint observation/query covariance yields conditional means and
 * a joint conditional square root. Semidefinite query blocks (including exact data hits) are
 * supported; singular observation blocks fail explicitly. Up to 128 observations and 128 queries
 * use one cooperative workgroup; larger domains use blocked multiworkgroup Cholesky and tiled
 * realizations. The dense factor and realization batch must fit individual device storage bindings.
 *
 * Capacity: the joint factor is one dense (n + q)^2 float32 binding (n observations, q queries),
 * 64 MiB at the 4096-row maximum, plus O(n + q) vectors. Both the cooperative and the blocked
 * (`conditional-simulation-blocked`) kernels use that same dense binding; blocking only tiles the
 * factorization across workgroups and does not lower the memory. The constructor row cap is 4096
 * and getCommandNodes asserts the factor and the replicate batch (q * replicateCount floats) fit
 * the device's maxStorageBufferBindingSize. Domains beyond that need a sparse or
 * low-rank approximation, not this class.
 * The shared factor is recomputed each encoding by default. Explicit
 * reusePreparation retains factors/moments until invalidated; validity is committed on the GPU,
 * so an encoded-but-discarded command buffer cannot make a future encoding trust an unbuilt factor.
 * Philox counters identify (replicate, latent query component), making stream identities independent
 * of dispatch and batch partitioning.
 * f32 Box-Muller is distributionally reproducible within a backend; no claim of cross-driver
 * transcendental bit equality or of matching another library's random sequence is made.
 */
export class GPUConditionalGaussianSimulation implements GPUCommandNodeProducer {
  readonly id: string;
  readonly props: GPUConditionalGaussianSimulationProps;
  private preparationControl: Uint32Array | undefined;
  private preparationControlBuffer: Buffer | undefined;
  private preparationStatisticsView: GraphDataView<'uint32x2'> | undefined;
  private ownedBuffers: Buffer[] = [];
  private destroyed = false;
  constructor(props: GPUConditionalGaussianSimulationProps) {
    this.props = props;
    this.id = props.id ?? 'conditional-gaussian-simulation';
    validatePredictionCovariance(props.covariance);
    validateCapacity(this.id, 'replicateCount', props.replicateCount, 1);
    validateCapacity(this.id, 'replicateOffset', props.replicateOffset ?? 0, 0);
    if (
      props.queries.length < 1 ||
      props.positions.length + props.queries.length > 4096 ||
      props.positions.length !== props.observations.length ||
      props.output.values.length < props.queries.length * props.replicateCount ||
      props.output.moments.length < props.queries.length ||
      props.output.status.length < props.replicateCount ||
      !Number.isInteger(props.seed) ||
      props.seed < 0 ||
      props.seed > 0xffffffff ||
      (props.replicateOffset ?? 0) + props.replicateCount > 0x100000000 ||
      !Number.isFinite(props.mean ?? 0)
    ) {
      throw new Error(`${this.id} invalid bounded simulation topology`);
    }
    validatePredictionViews(
      this.id,
      [props.positions, props.observations, props.queries],
      Object.values(props.output)
    );
    if (props.reusePreparation) {
      // Mutable batch capacity must fit the uint32 control buffer.
      assert(props.replicateCount <= 0xffffffff);
      this.preparationControl = Uint32Array.of(
        1,
        1,
        0,
        props.seed,
        props.replicateOffset ?? 0,
        props.replicateCount
      );
      new Float32Array(this.preparationControl.buffer)[2] = props.mean ?? 0;
    }
  }

  /** GPU counts of attempted factorizations and conditional-moment solves, after graph.add. */
  get preparationStatistics(): GraphDataView<'uint32x2'> | undefined {
    return this.preparationStatisticsView;
  }

  /**
   * Rebuild after positions, queries, validity or measurement errors change.
   * Covariance parameters and view shapes remain graph specializations; recreate for those changes.
   */
  invalidatePreparation(): void {
    const control = this.getPreparationControl();
    // Create a new producer rather than wrapping generation identities.
    assert(control[0] < 0xffffffff && control[1] < 0xffffffff);
    control[0]++;
    control[1]++;
    this.writePreparationControl();
  }

  /** Refresh finite observation values. Changed geometry detected during validation also rebuilds. */
  invalidateObservations(): void {
    const control = this.getPreparationControl();
    // Create a new producer rather than wrapping generation identities.
    assert(control[1] < 0xffffffff);
    control[1]++;
    this.writePreparationControl();
  }

  /** Update the known mean and conditional moments without changing covariance geometry. */
  setMean(mean: number): void {
    const control = this.getPreparationControl();
    // The known mean is finite in f32 and generation identities must not wrap.
    assert(Number.isFinite(Math.fround(mean)) && control[1] < 0xffffffff);
    new Float32Array(control.buffer)[2] = mean;
    control[1]++;
    this.writePreparationControl();
  }

  /**
   * Update a retained draw batch. replicateCount is bounded by the constructor's capacity.
   * Inactive output rows are NaN with status 1; only the selected prefix is a draw batch.
   * These setters enqueue buffer writes; submit preceding encodings before changing parameters.
   */
  setDrawParameters(parameters: {
    seed?: number;
    replicateOffset?: number;
    replicateCount?: number;
  }): void {
    const control = this.getPreparationControl();
    const seed = parameters.seed ?? control[3];
    const replicateOffset = parameters.replicateOffset ?? control[4];
    const replicateCount = parameters.replicateCount ?? control[5];
    // The Philox seed is a uint32 key.
    assert(Number.isInteger(seed) && seed >= 0 && seed <= 0xffffffff);
    // Replicate identities start at a nonnegative exact integer.
    assert(Number.isSafeInteger(replicateOffset) && replicateOffset >= 0);
    // A batch is nonempty and fits the constructor's output capacity.
    assert(
      Number.isSafeInteger(replicateCount) &&
        replicateCount >= 1 &&
        replicateCount <= this.props.replicateCount
    );
    // The last replicate identity remains within uint32.
    assert(replicateOffset + replicateCount <= 0x100000000);
    control[3] = seed;
    control[4] = replicateOffset;
    control[5] = replicateCount;
    this.writePreparationControl();
  }

  /** Destroy compiled graphs first. Only internally allocated preparation buffers are freed. */
  destroy(): void {
    for (const buffer of this.ownedBuffers) buffer.destroy();
    this.ownedBuffers = [];
    this.preparationControlBuffer = undefined;
    this.preparationStatisticsView = undefined;
    this.destroyed = true;
  }

  private getPreparationControl(): Uint32Array {
    // Mutable preparation is available only for a live producer with reusePreparation enabled.
    assert(this.preparationControl && !this.destroyed);
    return this.preparationControl;
  }

  private writePreparationControl(): void {
    if (this.preparationControlBuffer)
      this.preparationControlBuffer.write(this.getPreparationControl());
  }

  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const capacity = Math.max(1, props.positions.length + props.queries.length);
    const queryCount = props.queries.length;
    validateGraphViewsBelongToGraph(id, graph, [
      props.positions,
      props.observations,
      props.queries,
      ...Object.values(props.output)
    ]);
    const factorSize = capacity * capacity;
    const scratchLength = 2 + factorSize + capacity * 5;
    // Capacity: the dense (n + q)^2 factor must fit one storage binding. The blocked variant
    // (conditional-simulation-blocked, used above 128 rows) tiles the work but keeps this same
    // dense binding, so shrink n + q (or split queries across instances) when this fires.
    validateGPUStorageBindingByteLength(
      graph.device,
      id,
      'joint covariance factor',
      scratchLength * 4,
      'reduce observations plus queries'
    );
    validateGPUStorageBindingByteLength(
      graph.device,
      id,
      'realization batch',
      queryCount * props.replicateCount * 4,
      'reduce replicateCount and use replicateOffset for explicit batches'
    );
    const createPersistentView = <Format extends 'float32' | 'float32x2' | 'uint32'>(
      name: string,
      format: Format,
      length: number,
      values?: Uint32Array
    ) => {
      const buffer = graph.device.createBuffer({
        id: `${id}-${name}`,
        byteLength: length * (format === 'float32x2' ? 8 : 4),
        usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST,
        ...(values ? {data: values} : {})
      });
      this.ownedBuffers.push(buffer);
      return {buffer, view: importGraphBuffer(graph, `${id}-${name}`, buffer, format, length)};
    };
    // Imported storage retains data across encodings; transient allocations may be aliased.
    const retained = Boolean(this.preparationControl);
    if (retained) {
      // A retained producer belongs to one graph and is declared once.
      assert(!this.destroyed && this.ownedBuffers.length === 0);
    }
    const scratch = retained
      ? createPersistentView('factor', 'float32', scratchLength).view
      : createTransientView(graph, `${id}-factor`, 'float32', scratchLength);
    const moments = retained
      ? createPersistentView('preparation-moments', 'float32x2', queryCount).view
      : props.output.moments;
    const controlStorage = retained
      ? createPersistentView('preparation-control', 'uint32', 6, this.preparationControl)
      : undefined;
    const preparationState = retained
      ? createPersistentView('preparation-state', 'uint32', 7, new Uint32Array(7)).view
      : undefined;
    if (controlStorage && preparationState) {
      this.preparationControlBuffer = controlStorage.buffer;
      this.preparationStatisticsView = graph.createDataView(preparationState.buffer, {
        format: 'uint32x2',
        length: 1,
        byteOffset: 16
      });
    }
    const meanExpression = retained
      ? 'bitcast<f32>(preparationControl[preparationControlOffset + 2u])'
      : getWGSLFloatLiteral(props.mean ?? 0);
    const pointStart = 2 + factorSize;
    const observationStart = pointStart + capacity * 2;
    const errorStart = observationStart + capacity;
    const whitenedStart = errorStart + capacity;
    const conditionBindings = [
      {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
      {name: 'observations', view: props.observations, type: 'f32', access: 'read'},
      {name: 'queries', view: props.queries, type: 'f32', access: 'read'},
      {name: 'scratch', view: scratch, type: 'f32', access: 'read_write'},
      {name: 'moments', view: moments, type: 'f32', access: 'read_write'},
      ...(controlStorage && preparationState
        ? [
            {
              name: 'preparationControl',
              view: controlStorage.view,
              type: 'u32',
              access: 'read'
            } as const,
            {
              name: 'preparationState',
              view: preparationState,
              type: 'u32',
              access: 'read_write'
            } as const
          ]
        : [])
    ] as const;
    const conditionDeclarations = `${getPackedPredictionAccessors({positions: 2, observations: 4, queries: 2})}\n${GET_NAN_WGSL}\n${IS_FINITE_VALUE_WGSL}\n${getPredictionCovarianceWGSL(props.covariance)}
fn point(row: u32) -> vec2<f32> { return vec2<f32>(scratch[scratchOffset + ${pointStart}u + row * 2u], scratch[scratchOffset + ${pointStart + 1}u + row * 2u]); }
fn factor(row: u32, column: u32) -> f32 { return scratch[scratchOffset + 2u + row * ${capacity}u + column]; }
fn writeFactor(row: u32, column: u32, value: f32) { scratch[scratchOffset + 2u + row * ${capacity}u + column] = value; }`;
    const makeConditionNode = (variant: string, body: string, cooperative = false) =>
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-${variant}`,
        operation: 'GPUConditionalGaussianSimulation',
        variant,
        invocationCount: cooperative ? 64 : 1,
        workgroupSize: cooperative ? 64 : 2,
        guardIndex: !cooperative,
        bindings: conditionBindings,
        declarations:
          conditionDeclarations +
          (cooperative
            ? '\nvar<workgroup> failed: atomic<u32>;' +
              (retained ? '\nvar<workgroup> shouldFactor: u32;' : '')
            : ''),
        body
      });
    const prepareNode = makeConditionNode(
      'prepare-conditioning',
      `${
        retained
          ? `
// State: applied geometry/value generations, factor/moment gates, execution counts, valid factor.
preparationState[preparationStateOffset + 2u] = 0u;
preparationState[preparationStateOffset + 3u] = 0u;
var rebuildFactor = preparationState[preparationStateOffset + 6u] != 1u || preparationState[preparationStateOffset] != preparationControl[preparationControlOffset];
if (!rebuildFactor && preparationState[preparationStateOffset + 1u] == preparationControl[preparationControlOffset + 1u] && scratch[scratchOffset] == 0.0) { return; }
let previousObservedCount = u32(scratch[scratchOffset + 1u]);
// Any failed validation invalidates reuse until a complete preparation succeeds.
preparationState[preparationStateOffset + 6u] = 0u;
preparationState[preparationStateOffset + 3u] = 1u;
`
          : ''
      }
scratch[scratchOffset] = 1.0; scratch[scratchOffset + 1u] = 0.0;
for (var query = 0u; query < ${queryCount}u; query++) { moments[momentsOffset + query * 2u] = getNaN(); moments[momentsOffset + query * 2u + 1u] = getNaN(); }
var observedCount = 0u;
for (var sample = 0u; sample < ${props.positions.length}u; sample++) {
  let observation = read_observations(sample); if (!(observation.w > 0.0)) { continue; }
  let position = read_positions(sample);
  if (!isFiniteValue(observation.x) || !isFiniteValue(observation.y) || observation.y < 0.0 || !isFiniteValue(position.x) || !isFiniteValue(position.y)) { return; }
  ${retained ? `if (!rebuildFactor && (any(point(observedCount) != position) || scratch[scratchOffset + ${errorStart}u + observedCount] != observation.y)) { rebuildFactor = true; }` : ''}
  scratch[scratchOffset + ${pointStart}u + observedCount * 2u] = position.x;
  scratch[scratchOffset + ${pointStart + 1}u + observedCount * 2u] = position.y;
  let centered = observation.x - ${meanExpression};
  if (!isFiniteValue(centered)) { scratch[scratchOffset] = 2.0; return; }
  scratch[scratchOffset + ${observationStart}u + observedCount] = centered;
  scratch[scratchOffset + ${errorStart}u + observedCount] = observation.y;
  observedCount++;
}
scratch[scratchOffset + 1u] = f32(observedCount);
${retained ? 'if (observedCount != previousObservedCount) { rebuildFactor = true; }' : ''}
for (var query = 0u; query < ${queryCount}u; query++) {
  let position = read_queries(query); if (!isFiniteValue(position.x) || !isFiniteValue(position.y)) { return; }
  let row = observedCount + query;
  ${retained ? 'if (!rebuildFactor && any(point(row) != position)) { rebuildFactor = true; }' : ''}
  scratch[scratchOffset + ${pointStart}u + row * 2u] = position.x;
  scratch[scratchOffset + ${pointStart + 1}u + row * 2u] = position.y;
  scratch[scratchOffset + ${errorStart}u + row] = 0.0;
}
scratch[scratchOffset] = 0.0;
${
  retained
    ? `preparationState[preparationStateOffset + 2u] = u32(rebuildFactor);
preparationState[preparationStateOffset + 6u] = u32(!rebuildFactor);`
    : ''
}`
    );
    if (props.positions.length > 128 || queryCount > 128) {
      return [
        prepareNode,
        ...getBlockedConditionalSimulationNodes(graph, {
          id,
          props,
          capacity,
          scratch,
          moments,
          control: controlStorage?.view,
          preparationState,
          declarations: conditionDeclarations,
          conditionBindings,
          observationStart,
          errorStart,
          whitenedStart,
          meanExpression
        })
      ];
    }
    return [
      prepareNode,
      makeConditionNode(
        'cooperative-factor',
        `
${
  retained
    ? `
if (localInvocationIndex == 0u) { shouldFactor = u32(preparationState[preparationStateOffset + 2u] != 0u && scratch[scratchOffset] == 0.0); }
if (workgroupUniformLoad(&shouldFactor) == 0u) { return; }
if (localInvocationIndex == 0u) { preparationState[preparationStateOffset + 4u]++; }
`
    : ''
}
if (localInvocationIndex == 0u) { atomicStore(&failed, select(1u, 0u, scratch[scratchOffset] == 0.0)); }
workgroupBarrier();
let observedCount = u32(scratch[scratchOffset + 1u]);
let size = observedCount + ${queryCount}u;
let tolerance = ${getWGSLFloatLiteral((props.covariance.sill + (props.covariance.nugget ?? 0)) * 1e-6)};
// Every lane reaches every barrier, including inactive rows and failed factorizations.
for (var column = 0u; column < ${capacity}u; column++) {
  if (localInvocationIndex == 0u && column < size && atomicLoad(&failed) == 0u) {
    var value = covariance(point(column), point(column)) + scratch[scratchOffset + ${errorStart}u + column];
    for (var component = 0u; component < column; component++) { let loading = factor(column, component); value -= loading * loading; }
    if (!isFiniteValue(value) || value < -tolerance || (column < observedCount && value <= tolerance)) { atomicStore(&failed, 1u); }
    else { writeFactor(column, column, sqrt(max(value, 0.0))); }
  }
  storageBarrier();
  workgroupBarrier();
  if (atomicLoad(&failed) == 0u) {
    for (var row = column + 1u + localInvocationIndex; row < size; row += 64u) {
      var value = covariance(point(row), point(column));
      for (var component = 0u; component < column; component++) { value -= factor(row, component) * factor(column, component); }
      let diagonal = factor(column, column);
      if (diagonal == 0.0) {
        if (abs(value) > tolerance) { atomicStore(&failed, 1u); }
        else { writeFactor(row, column, 0.0); }
      } else {
        let loading = value / diagonal;
        if (!isFiniteValue(loading)) { atomicStore(&failed, 1u); }
        else { writeFactor(row, column, loading); }
      }
    }
  }
  // Storage factors and the workgroup failure flag have separate memory classes.
  storageBarrier();
  workgroupBarrier();
}
if (localInvocationIndex == 0u && atomicLoad(&failed) != 0u && scratch[scratchOffset] == 0.0) {
  scratch[scratchOffset] = 2.0;
}
${
  retained
    ? `if (localInvocationIndex == 0u && scratch[scratchOffset] == 0.0) {
preparationState[preparationStateOffset + 6u] = 1u;
preparationState[preparationStateOffset] = preparationControl[preparationControlOffset];
}`
    : ''
}
`,
        true
      ),
      makeConditionNode(
        'conditional-moments',
        `${retained ? 'if (preparationState[preparationStateOffset + 3u] == 0u) { return; }' : ''}
if (scratch[scratchOffset] != 0.0) { return; }
${retained ? 'preparationState[preparationStateOffset + 5u]++;' : ''}
scratch[scratchOffset] = 2.0;
let observedCount = u32(scratch[scratchOffset + 1u]);
for (var row = 0u; row < observedCount; row++) {
  var value = scratch[scratchOffset + ${observationStart}u + row];
  for (var column = 0u; column < row; column++) { value -= factor(row, column) * scratch[scratchOffset + ${whitenedStart}u + column]; }
  let whitened = value / factor(row, row);
  if (!isFiniteValue(whitened)) { return; }
  scratch[scratchOffset + ${whitenedStart}u + row] = whitened;
}
var expectations: array<f32, ${queryCount}>;
var variances: array<f32, ${queryCount}>;
for (var query = 0u; query < ${queryCount}u; query++) {
  let row = observedCount + query; var expectation = ${meanExpression}; var variance = 0.0;
  for (var column = 0u; column < observedCount; column++) { expectation += factor(row, column) * scratch[scratchOffset + ${whitenedStart}u + column]; }
  for (var component = 0u; component <= query; component++) { let loading = factor(row, observedCount + component); variance += loading * loading; }
  if (!isFiniteValue(expectation) || !isFiniteValue(variance)) { return; }
  expectations[query] = expectation; variances[query] = variance;
}
for (var query = 0u; query < ${queryCount}u; query++) {
  moments[momentsOffset + query * 2u] = expectations[query]; moments[momentsOffset + query * 2u + 1u] = variances[query];
}
scratch[scratchOffset] = 0.0;
${retained ? 'preparationState[preparationStateOffset + 1u] = preparationControl[preparationControlOffset + 1u];' : ''}`
      ),
      createWGSLKernelNode(graph, {
        id,
        operation: 'GPUConditionalGaussianSimulation',
        variant: 'joint-realizations',
        invocationCount: props.replicateCount,
        workgroupSize: 32,
        bindings: [
          {name: 'scratch', view: scratch, type: 'f32', access: 'read'},
          {name: 'moments', view: moments, type: 'f32', access: 'read'},
          {name: 'values', view: props.output.values, type: 'f32', access: 'read_write'},
          {name: 'statuses', view: props.output.status, type: 'u32', access: 'read_write'},
          ...(controlStorage
            ? [
                {
                  name: 'preparationControl',
                  view: controlStorage.view,
                  type: 'u32',
                  access: 'read'
                } as const,
                {
                  name: 'outputMoments',
                  view: props.output.moments,
                  type: 'f32',
                  access: 'read_write'
                } as const
              ]
            : [])
        ],
        declarations: `${GET_NAN_WGSL}\n${IS_FINITE_VALUE_WGSL}\n${PHILOX_WGSL}`,
        body: `
${
  retained
    ? `
if (index == 0u) {
  for (var query = 0u; query < ${queryCount}u; query++) {
    outputMoments[outputMomentsOffset + query * 2u] = moments[momentsOffset + query * 2u];
    outputMoments[outputMomentsOffset + query * 2u + 1u] = moments[momentsOffset + query * 2u + 1u];
  }
}
if (index >= preparationControl[preparationControlOffset + 5u]) {
  statuses[statusesOffset + index] = 1u;
  for (var query = 0u; query < ${queryCount}u; query++) { values[valuesOffset + index * ${queryCount}u + query] = getNaN(); }
  return;
}
`
    : ''
}
statuses[statusesOffset + index] = u32(scratch[scratchOffset]);
if (scratch[scratchOffset] != 0.0) {
  for (var query = 0u; query < ${queryCount}u; query++) { values[valuesOffset + index * ${queryCount}u + query] = getNaN(); }
  return;
}
let observedCount = u32(scratch[scratchOffset + 1u]);
var normal: array<f32, ${queryCount}>;
for (var query = 0u; query < ${queryCount}u; query++) {
  let random = philox4x32(vec4<u32>(index + ${retained ? 'preparationControl[preparationControlOffset + 4u]' : `${props.replicateOffset ?? 0}u`}, query, 0u, 0u), vec2<u32>(${retained ? 'preparationControl[preparationControlOffset + 3u]' : `${props.seed}u`}, ${PHILOX_PURPOSES.conditionalGaussianSimulation.id}u));
  normal[query] = sqrt(-2.0 * log((f32(random.x >> 8u) + 0.5) / 16777216.0)) * cos(6.28318530718 * (f32(random.y >> 8u) + 0.5) / 16777216.0);
}
for (var query = 0u; query < ${queryCount}u; query++) {
  var realization = moments[momentsOffset + query * 2u];
  for (var component = 0u; component <= query; component++) { realization += scratch[scratchOffset + 2u + (observedCount + query) * ${capacity}u + observedCount + component] * normal[component]; }
  values[valuesOffset + index * ${queryCount}u + query] = realization;
  if (!isFiniteValue(realization)) { statuses[statusesOffset + index] = 2u; }
}
if (statuses[statusesOffset + index] != 0u) {
  for (var query = 0u; query < ${queryCount}u; query++) { values[valuesOffset + index * ${queryCount}u + query] = getNaN(); }
}`
      })
    ];
  }
}

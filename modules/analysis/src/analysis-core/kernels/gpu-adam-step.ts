// luma.gl contributors
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {
  GPUCommandGraph,
  GPUCommandNode,
  GPUCommandNodeProducer,
  GraphDataView
} from '@luma.gl/graph/gpu-core';
import type {GPUIterationStatusPort, GPUInvalidStatusPort} from '../contracts/status';
import {validateCapacity, validateCommandNodeBudget} from './gpu-contributor-utils';
import {
  GPU_FITTING_MAXIMUM_PARAMETER_COUNT,
  validateFittingGraph,
  validateFittingView,
  validateFittingWritableViews
} from './fitting-utils';
import {createWGSLDispatchGate, createWGSLKernelNode} from './wgsl-kernel-nodes';
import {GPU_ADAM_STEP_PARAMETER_LENGTH} from './adam-step-parameters';
import {ADAM_MOMENT_RANGE_WGSL, getAdamBiasCorrectionWGSL} from './adam-step-wgsl';
import {validateDisjointWritableViews} from './graph-chunk-dispatch';
import {snapshotConstructionFields} from './construction-snapshot';

/** Keeps history small (40 kB) and bias correction bounded to 14 binary-composition rounds. */
export const GPU_ADAM_STEP_MAXIMUM_ITERATIONS = 10_000;

export type GPUAdamStepOutput = {
  /** Caller-owned persistent state; initial weights are supplied by the caller. */
  weights: GraphDataView<'float32'>;
  /** Initialize both moments to zero for a new fit. */
  firstMoment: GraphDataView<'float32'>;
  secondMoment: GraphDataView<'float32'>;
  /** maximumIterations + 1 losses, indexed by the number of completed updates. */
  lossHistory: GraphDataView<'float32'>;
  /** Initialize all status scalars to zero for a new fit. */
  status: Required<GPUIterationStatusPort & GPUInvalidStatusPort>;
};
export type GPUAdamStepProps = {
  id?: string;
  /** Finite gradient at the current weights; normalize in the caller for mean losses. */
  gradients: GraphDataView<'float32'>;
  /** Scalar loss evaluated at the current weights before this step. */
  loss: GraphDataView<'float32'>;
  /** Packed with getGPUAdamStepParameterValues; usable with GPUParameterBuffer. */
  parameters: GraphDataView<'float32'>;
  maximumIterations: number;
  output: GPUAdamStepOutput;
};

/**
 * One bias-corrected Adam update (Kingma and Ba, Algorithm 1), with a persistent GPU iteration
 * budget. Compile-time: parameter count and maximumIterations. Per-frame: parameters, gradients,
 * loss and optimizer state. Emits two nodes: loss/status/gate control and a gated elementwise step.
 * Loss plateau means |loss - previous| <= tolerance * max(1, |previous|); tolerance zero disables
 * it. The caller evaluates loss/gradients between steps and calls once more after the last update
 * to record lossHistory[iterationCount], including the final iterate at the budget. That final call
 * performs no update. Finished calls preserve weights and moments; a converged call is a no-op.
 * No model, derivative, weights or datasets are bundled. Skip-zero leaves moments unchanged and
 * uses the global update count for bias correction (not a per-parameter sparse Adam clock).
 */
export class GPUAdamStep implements GPUCommandNodeProducer {
  readonly id: string;
  readonly props: GPUAdamStepProps;
  readonly estimatedCommandNodeCount = 2;

  constructor(props: GPUAdamStepProps) {
    props = Object.freeze({
      ...snapshotConstructionFields(props, [
        'id',
        'gradients',
        'loss',
        'parameters',
        'maximumIterations',
        'output'
      ]),
      output: Object.freeze({
        ...snapshotConstructionFields(props.output, [
          'weights',
          'firstMoment',
          'secondMoment',
          'lossHistory',
          'status'
        ]),
        status: snapshotConstructionFields(props.output.status, [
          'iterationCount',
          'converged',
          'iterationLimitReached',
          'invalidCount'
        ])
      })
    });
    this.id = props.id ?? 'adam-step';
    this.props = props;
    const {id} = this;
    const {output} = props;
    const parameterCount = props.gradients.length;
    validateCapacity(id, 'parameterCount', parameterCount, 1, GPU_FITTING_MAXIMUM_PARAMETER_COUNT);
    validateCapacity(
      id,
      'maximumIterations',
      props.maximumIterations,
      1,
      GPU_ADAM_STEP_MAXIMUM_ITERATIONS
    );
    validateFittingView(id, 'gradients', props.gradients, 'float32', parameterCount);
    validateFittingView(id, 'loss', props.loss, 'float32', 1);
    validateFittingView(
      id,
      'parameters',
      props.parameters,
      'float32',
      GPU_ADAM_STEP_PARAMETER_LENGTH
    );
    for (const name of ['weights', 'firstMoment', 'secondMoment'] as const) {
      validateFittingView(id, `output.${name}`, output[name], 'float32', parameterCount);
    }
    validateFittingView(
      id,
      'output.lossHistory',
      output.lossHistory,
      'float32',
      props.maximumIterations + 1
    );
    for (const [name, view] of Object.entries(output.status))
      validateFittingView(id, `output.status.${name}`, view, 'uint32', 1);
    const writable = {
      weights: output.weights,
      firstMoment: output.firstMoment,
      secondMoment: output.secondMoment,
      lossHistory: output.lossHistory,
      ...Object.fromEntries(
        Object.entries(output.status).map(([name, view]) => [`status.${name}`, view])
      )
    };
    validateDisjointWritableViews(id, writable, {separateDispatches: true});
    // External inputs may share disjoint bytes with a result used in another dispatch.
    for (const [index, input] of [props.gradients, props.loss, props.parameters].entries()) {
      validateDisjointWritableViews(
        id,
        {...writable, [`input ${index}`]: input},
        {separateDispatches: true}
      );
    }
    validateFittingWritableViews(
      id,
      [output.lossHistory, ...Object.values(output.status)],
      [props.loss, props.parameters]
    );
    // iterationCount is writable in control, then read-only beside the model update.
    // WebGPU forbids that read-only binding sharing a physical buffer with the writable model,
    // even when all logical/aligned ranges are disjoint.
    validateFittingWritableViews(
      id,
      [output.weights, output.firstMoment, output.secondMoment],
      [props.gradients, props.parameters, output.status.iterationCount]
    );
    validateCommandNodeBudget(id, this.estimatedCommandNodeCount);
  }

  private getWritableViews(): GraphDataView[] {
    const {output} = this.props;
    return [
      output.weights,
      output.firstMoment,
      output.secondMoment,
      output.lossHistory,
      ...Object.values(output.status)
    ];
  }

  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const {output, maximumIterations} = props;
    const {status} = output;
    validateFittingGraph(id, graph, [
      props.gradients,
      props.loss,
      props.parameters,
      ...this.getWritableViews()
    ]);
    const gate = createWGSLDispatchGate(graph, {
      id: `${id}-gate`,
      operation: 'GPUAdamStep',
      shapes: [[props.gradients.length]]
    });
    return [
      createWGSLKernelNode(graph, {
        id: `${id}-control`,
        operation: 'GPUAdamStep',
        variant: 'loss-budget-status',
        invocationCount: 1,
        bindings: [
          {name: 'loss', view: props.loss, type: 'f32', access: 'read'},
          {name: 'history', view: output.lossHistory, type: 'f32', access: 'read_write'},
          {name: 'iterations', view: status.iterationCount, type: 'u32', access: 'read_write'},
          {name: 'converged', view: status.converged, type: 'u32', access: 'read_write'},
          {name: 'limited', view: status.iterationLimitReached, type: 'u32', access: 'read_write'},
          {name: 'invalid', view: status.invalidCount, type: 'u32', access: 'read_write'},
          {name: 'parameters', view: props.parameters, type: 'f32', access: 'read'},
          gate.binding
        ],
        declarations: ADAM_MOMENT_RANGE_WGSL,
        body: `${gate.disarmWGSL}
  if (converged[convergedOffset] != 0u || invalid[invalidOffset] != 0u) { return; }
  let completed = iterations[iterationsOffset];
  let currentLoss = loss[lossOffset];
  var validParameters = isFiniteValue(currentLoss);
  ${Array.from(
    {length: GPU_ADAM_STEP_PARAMETER_LENGTH},
    (_, parameter) =>
      `validParameters = validParameters && isFiniteValue(parameters[parametersOffset + ${parameter}u]);`
  ).join('\n  ')}
  validParameters = validParameters && parameters[parametersOffset] > 0.0
    && parameters[parametersOffset + 1u] >= 0.0 && parameters[parametersOffset + 1u] < 1.0
    && parameters[parametersOffset + 2u] >= 0.0 && parameters[parametersOffset + 2u] < 1.0
    && parameters[parametersOffset + 3u] >= 1e-30 && parameters[parametersOffset + 4u] >= 0.0;
  if (!validParameters || completed > ${maximumIterations}u) {
    invalid[invalidOffset] = 1u;
    return;
  }
  history[historyOffset + completed] = currentLoss;
  if (completed > 0u && parameters[parametersOffset + 4u] > 0.0) {
    let previous = history[historyOffset + completed - 1u];
    if (isAdamLossPlateau(currentLoss, previous, parameters[parametersOffset + 4u])) {
      converged[convergedOffset] = 1u;
      limited[limitedOffset] = 0u;
      return;
    }
  }
  if (completed >= ${maximumIterations}u) {
    limited[limitedOffset] = 1u;
    return;
  }
  iterations[iterationsOffset] = completed + 1u;
  limited[limitedOffset] = select(0u, 1u, completed + 1u == ${maximumIterations}u);
  ${gate.armWGSL}`
      }),
      createWGSLKernelNode(graph, {
        id: `${id}-update`,
        operation: 'GPUAdamStep',
        variant: 'bias-corrected-adam',
        invocationCount: props.gradients.length,
        ...gate.getGate(0),
        bindings: [
          {name: 'gradients', view: props.gradients, type: 'f32', access: 'read'},
          {name: 'weights', view: output.weights, type: 'f32', access: 'read_write'},
          {name: 'firstMoment', view: output.firstMoment, type: 'f32', access: 'read_write'},
          {name: 'secondMoment', view: output.secondMoment, type: 'f32', access: 'read_write'},
          {name: 'parameters', view: props.parameters, type: 'f32', access: 'read'},
          {name: 'iterations', view: status.iterationCount, type: 'u32', access: 'read'}
        ],
        declarations:
          getAdamBiasCorrectionWGSL(GPU_ADAM_STEP_MAXIMUM_ITERATIONS) + ADAM_MOMENT_RANGE_WGSL,
        body: `let gradient = gradients[gradientsOffset + index];
  if (parameters[parametersOffset + 5u] != 0.0 && gradient == 0.0) { return; }
  let beta1 = parameters[parametersOffset + 1u];
  let beta2 = parameters[parametersOffset + 2u];
  let first = beta1 * firstMoment[firstMomentOffset + index] + (1.0 - beta1) * gradient;
  let second = beta2 * secondMoment[secondMomentOffset + index] + getAdamSecondMomentContribution(gradient, beta2);
  let iteration = iterations[iterationsOffset];
  let firstCorrection = getAdamBiasCorrection(beta1, iteration);
  let secondCorrection = getAdamBiasCorrection(beta2, iteration);
  // Normalize before the square root so reassociation cannot form an overflowing quotient.
  let denominator = getAdamCorrectedSecondMomentRoot(second, secondCorrection) + parameters[parametersOffset + 3u];
  firstMoment[firstMomentOffset + index] = first;
  secondMoment[secondMomentOffset + index] = second;
  weights[weightsOffset + index] -= getAdamWeightDelta(
    parameters[parametersOffset], first, firstCorrection, denominator);`
      })
    ];
  }
}

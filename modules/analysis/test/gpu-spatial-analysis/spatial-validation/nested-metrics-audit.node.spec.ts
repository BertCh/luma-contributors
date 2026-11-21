// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it, vi} from 'vitest';
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {
  GPUNestedInterpolationValidation,
  type GPUNestedInterpolationValidationProps
} from '../../../src/gpu-spatial-analysis/spatial-validation/gpu-nested-interpolation-validation';
import {GPUNestedInterpolationValidationOperation} from '../../../src/gpu-spatial-analysis/spatial-validation/gpu-nested-interpolation-validation-operation';
import {
  CPUNestedInterpolationValidationOperation,
  computeNestedInterpolationValidationOnCPU
} from '../../../src/gpu-spatial-analysis/spatial-validation/cpu-nested-interpolation-validation-operation';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {createSamplingCPUInputs} from '../point-process/validation-sampling-operation-test-utils';
import {
  watchConstructionFields,
  createNestedCaptureInput,
  createNestedCaptureProps,
  getNestedCaptureExpected
} from './construction-capture-audit.test-utils';
import {WgslExec, WgslParser, WgslReflect} from 'wgsl_reflect';
import {
  createNestedMetricAuditValues,
  getNestedMetricAuditKernels
} from './nested-metrics-audit.test-utils';

it('captures every pure CPU RBF candidate option and reproduces its constant polynomial oracle', () => {
  const candidate = watchConstructionFields(
    {
      mode: 'rbf' as const,
      kernel: 'gaussian' as const,
      degree: 0 as const,
      epsilon: 0.5,
      smoothing: 0.25
    },
    'candidate'
  );
  const options = watchConstructionFields(
    {foldCount: 2, innerFoldCount: 2, neighborCount: 2, candidates: [candidate.props]},
    'options'
  );
  const actual = computeNestedInterpolationValidationOnCPU(
    createNestedCaptureInput(),
    options.props
  );
  options.expectReadOnce();
  candidate.expectReadOnce();
  // Degree-zero augmentation reproduces constant training values exactly in real arithmetic.
  for (const [port, expected] of Object.entries(getNestedCaptureExpected()))
    expected.forEach((value, index) => {
      const measured = actual[port as keyof typeof actual][index];
      if (Number.isNaN(value)) expect(measured).toBeNaN();
      else {
        expect(Number.isFinite(measured)).toBe(true);
        expect(Math.abs(measured - value)).toBeLessThan(1e-7);
      }
    });
});

it('reads pure CPU nested settings and covariance getters once before evaluating every hand-oracle port', () => {
  const covariance = watchConstructionFields(
    {model: 'spherical' as const, range: 0.5, sill: 1, nugget: 0},
    'covariance'
  );
  const candidate = watchConstructionFields(
    {mode: 'ordinary-kriging' as const, covariance: covariance.props},
    'candidate'
  );
  const options = watchConstructionFields(
    {
      id: 'pure-nested-capture',
      foldCount: 2,
      innerFoldCount: 2,
      innerBufferDistance: 0,
      seed: 0,
      neighborCount: 2,
      candidates: [candidate.props]
    },
    'options'
  );
  const actual = computeNestedInterpolationValidationOnCPU(
    createNestedCaptureInput(),
    options.props
  );
  options.expectReadOnce();
  candidate.expectReadOnce();
  covariance.expectReadOnce();
  for (const [port, expected] of Object.entries(getNestedCaptureExpected()))
    expect(Array.from(actual[port as keyof typeof actual]), port).toEqual(expected);
});

it.each([
  'contributor',
  'cpu',
  'gpu'
] as const)('captures %s nested candidate/covariance/output maps once and preserves the hand oracle', async form => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let nextView = 0;
  const view = <Format extends 'float32' | 'float32x2' | 'uint32'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `nested-capture-${nextView++}`, format, length);
  try {
    const covarianceValues = {model: 'spherical' as const, range: 0.5, sill: 1, nugget: 0};
    const covariance = watchConstructionFields(covarianceValues, 'covariance');
    const candidateValues = {mode: 'ordinary-kriging' as const, covariance: covariance.props};
    const candidate = watchConstructionFields(candidateValues, 'candidate');
    const candidates = [candidate.props];
    const namedValues = {...createNestedCaptureProps(), candidates};
    const namedOutput = watchConstructionFields(namedValues.output, 'output');
    namedValues.output = namedOutput.props;
    const liveOutputValues = {
      selected: view('uint32', 2),
      scores: view('float32', 6),
      predictions: view('float32', 16),
      metrics: view('float32', 12),
      status: view('uint32', 2)
    };
    const liveOutput = watchConstructionFields(liveOutputValues, 'output');
    const originalLiveOutputs = {...liveOutputValues};
    const liveValues: GPUNestedInterpolationValidationProps = {
      id: 'nested-capture',
      mode: 'tiled',
      positions: view('float32x2', 8),
      response: view('float32', 8),
      training: view('uint32', 16),
      testing: view('uint32', 16),
      innerBlocks: view('uint32', 8),
      foldCount: 2,
      innerFoldCount: 2,
      innerBufferDistance: 0,
      seed: 0,
      neighborCount: 2,
      candidates,
      output: liveOutput.props
    };
    const live = watchConstructionFields(liveValues, 'props');
    const named = watchConstructionFields(namedValues, 'props');
    const operation =
      form === 'contributor'
        ? new GPUNestedInterpolationValidation(live.props)
        : form === 'cpu'
          ? new CPUNestedInterpolationValidationOperation(named.props)
          : new GPUNestedInterpolationValidationOperation(named.props);
    const watchers = [
      form === 'contributor' ? live : named,
      form === 'contributor' ? liveOutput : namedOutput,
      candidate,
      covariance
    ];
    for (const watcher of watchers) watcher.expectReadOnce();
    covarianceValues.range = -1;
    candidates.push({
      mode: 'ordinary-kriging',
      covariance: {model: 'spherical', range: -1, sill: 1, nugget: 0}
    });
    namedValues.rowCount = 1;
    namedValues.neighborCount = 1;
    namedValues.innerBufferDistance = -1;
    liveValues.neighborCount = 1;
    liveValues.innerBufferDistance = -1;
    liveOutputValues.predictions = view('float32', 1);
    if (operation instanceof CPUNestedInterpolationValidationOperation) {
      for (const scale of [1, 2]) {
        const input = createNestedCaptureInput(scale);
        const actual = await operation.execute({
          inputs: createSamplingCPUInputs({
            positions: {format: 'float32x2', values: input.positions},
            response: {format: 'float32', values: input.response},
            training: {format: 'uint32', values: input.training},
            testing: {format: 'uint32', values: input.testing},
            innerBlocks: {format: 'uint32', values: input.innerBlocks}
          })
        });
        for (const [port, expected] of Object.entries(getNestedCaptureExpected(scale))) {
          expect(ArrayBuffer.isView(actual[port]), port).toBe(true);
          expect(Array.from(actual[port] as ArrayLike<number>), port).toEqual(expected);
        }
      }
    } else {
      const contributor =
        operation instanceof GPUNestedInterpolationValidation
          ? operation
          : (operation.createContributor(port => {
              const inputViews = {
                positions: liveValues.positions,
                response: liveValues.response,
                training: liveValues.training,
                testing: liveValues.testing,
                innerBlocks: liveValues.innerBlocks
              };
              return (
                port in inputViews
                  ? inputViews[port as keyof typeof inputViews]
                  : originalLiveOutputs[port as keyof typeof originalLiveOutputs]
              ) as never;
            }) as GPUNestedInterpolationValidation);
      expect(contributor.props.neighborCount).toBe(2);
      expect(contributor.props.candidates).toHaveLength(1);
      expect(contributor.props.candidates[0]).toMatchObject({covariance: {range: 0.5}});
      expect(() => contributor.getCommandNodes(graph)).not.toThrow();
    }
    for (const watcher of watchers) watcher.expectReadOnce();
  } finally {
    device.destroy();
  }
});

it.each([
  8, 257
])('audit-validation nested held-out R² retains half-step means after response translation (%s rows)', rows => {
  const kernels = getNestedMetricAuditKernels(rows);
  expect(kernels).toHaveLength(2);
  const errors = vi.spyOn(console, 'error').mockImplementation(message => {
    if (!String(message).startsWith('Variable _ not found.')) throw new Error(String(message));
  });
  try {
    for (const shift of [0, 10_000_000]) {
      const values = createNestedMetricAuditValues(rows, shift);
      for (const kernel of kernels) {
        const bindings = Object.fromEntries(
          new WgslReflect(kernel.source).storage.map(binding => [
            binding.binding,
            values[binding.name]
          ])
        );
        new WgslExec(new WgslParser().parse(kernel.source)).dispatchWorkgroups(
          kernel.entryPoint,
          [1, 1, 1],
          {0: bindings},
          {constants: kernel.constants}
        );
      }
      // Hand oracle: errors [1,1], SSE=2, SST=(0-.5)^2+(1-.5)^2=.5.
      expect(Array.from(values['metrics'])).toEqual([2, 0, 1, 1, 1, -3]);
      expect(Array.from(values['status'])).toEqual([0]);
    }
  } finally {
    errors.mockRestore();
  }
});

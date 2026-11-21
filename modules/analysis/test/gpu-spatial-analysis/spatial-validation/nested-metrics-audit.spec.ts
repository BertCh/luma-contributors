// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {ShaderLayout} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {Kernel} from '@luma.gl/graph';
import {expect, it} from 'vitest';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {
  GPUNestedInterpolationValidation,
  type GPUNestedInterpolationValidationProps
} from '../../../src/gpu-spatial-analysis/spatial-validation/gpu-nested-interpolation-validation';
import {GPUNestedInterpolationValidationOperation} from '../../../src/gpu-spatial-analysis/spatial-validation/gpu-nested-interpolation-validation-operation';
import {CPUNestedInterpolationValidationOperation} from '../../../src/gpu-spatial-analysis/spatial-validation/cpu-nested-interpolation-validation-operation';
import {AnalysisRig} from '../catchment-accessibility/rig';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {
  createNestedCaptureInput,
  createNestedCaptureProps,
  getNestedCaptureExpected
} from './construction-capture-audit.test-utils';
import {
  createNestedMetricAuditValues,
  getNestedMetricAuditKernels
} from './nested-metrics-audit.test-utils';

for (const candidateMode of ['ordinary-kriging', 'rbf'] as const)
  it.each([
    'contributor',
    'cpu',
    'gpu'
  ] as const)(`retains all nested ${candidateMode} hand-oracle outputs after %s configuration mutation`, async form => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const input = createNestedCaptureInput();
    const props = createNestedCaptureProps(candidateMode);
    const rig = new AnalysisRig(device);
    try {
      let actual: Record<string, ArrayLike<number>>;
      if (form === 'contributor') {
        const output = {
          selected: rig.output('uint32', 2),
          scores: rig.output('float32', 6),
          predictions: rig.output('float32', 16),
          metrics: rig.output('float32', 12),
          status: rig.output('uint32', 2)
        };
        const originalOutput = {...output};
        const liveProps: GPUNestedInterpolationValidationProps = {
          ...props,
          positions: rig.input(input.positions, 'float32x2', 8),
          response: rig.input(input.response, 'float32'),
          training: rig.input(input.training, 'uint32'),
          testing: rig.input(input.testing, 'uint32'),
          innerBlocks: rig.input(input.innerBlocks, 'uint32'),
          output
        };
        const contributor = new GPUNestedInterpolationValidation(liveProps);
        liveProps.innerBufferDistance = -1;
        liveProps.neighborCount = 1;
        if (props.candidates[0].mode === 'ordinary-kriging')
          props.candidates[0].covariance.range = -1;
        else {
          props.candidates[0].epsilon = 0;
          props.candidates[0].smoothing = -1;
        }
        output.predictions = rig.output('float32', 1);
        rig.run(contributor);
        actual = {
          selected: await rig.readUint(originalOutput.selected),
          scores: await rig.readFloat(originalOutput.scores),
          predictions: await rig.readFloat(originalOutput.predictions),
          metrics: await rig.readFloat(originalOutput.metrics),
          status: await rig.readUint(originalOutput.status)
        };
      } else {
        const operation =
          form === 'cpu'
            ? new CPUNestedInterpolationValidationOperation(props)
            : new GPUNestedInterpolationValidationOperation(props);
        props.rowCount = 1;
        props.neighborCount = 1;
        props.innerBufferDistance = -1;
        props.output.metrics = 'changed';
        if (props.candidates[0].mode === 'ordinary-kriging')
          props.candidates[0].covariance.range = -1;
        else {
          props.candidates[0].epsilon = 0;
          props.candidates[0].smoothing = -1;
        }
        actual = await runOperationGraph(device, {
          inputs: {
            positions: {format: 'float32x2', values: input.positions},
            response: {format: 'float32', values: input.response},
            training: {format: 'uint32', values: input.training},
            testing: {format: 'uint32', values: input.testing},
            innerBlocks: {format: 'uint32', values: input.innerBlocks}
          },
          operations: [operation],
          outputs: ['selected', 'scores', 'predictions', 'metrics', 'status']
        });
      }
      // Independent hand derivation in construction-capture-audit.test-utils.ts:
      // weights 1/2, prediction 1, inner MSE 0, outer [MSE,MAE,bias,R²]=[5,2,-2,-4].
      // 3e-6 relative/absolute covers the float32 bordered solve and metric arithmetic.
      for (const [port, expected] of Object.entries(getNestedCaptureExpected())) {
        expect(actual[port].length).toBe(expected.length);
        expected.forEach((value, index) => {
          if (Number.isNaN(value)) expect(actual[port][index]).toBeNaN();
          else {
            expect(Number.isFinite(actual[port][index])).toBe(true);
            expect(Math.abs(actual[port][index] - value)).toBeLessThanOrEqual(
              3e-6 * Math.max(1, Math.abs(value))
            );
          }
        });
      }
    } finally {
      rig.destroy();
    }
  });

it.each([
  8, 257
])('audit-validation GPU nested R² retains translated half-step means (%s rows)', async rows => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const sources = getNestedMetricAuditKernels(rows);
  const kernels = sources.map(
    source =>
      new Kernel(device, {
        source: source.source,
        constants: source.constants,
        shaderLayout: source.shaderLayout as ShaderLayout
      })
  );
  try {
    for (const shift of [0, 10_000_000]) {
      const buffers = Object.fromEntries(
        Object.entries(createNestedMetricAuditValues(rows, shift)).map(([name, values]) => {
          if (name === 'metrics' || name === 'status') {
            // Readback needs COPY_SRC in addition to the kernel's storage usage.
            const buffer = createOutputBuffer(device, values.length);
            buffer.write(values);
            return [name, buffer];
          }
          return [name, createInputBuffer(device, values)];
        })
      );
      try {
        const encoder = device.createCommandEncoder();
        for (const [index, kernel] of kernels.entries()) {
          const pass = encoder.beginComputePass();
          const names =
            index === 0
              ? ['response', 'testing', 'predictions', 'status', 'partial']
              : ['partial', 'metrics', 'status'];
          kernel.dispatch(pass, {
            bindings: Object.fromEntries(names.map(name => [name, buffers[name]])),
            x: 1
          });
          pass.end();
        }
        device.submit(encoder.finish());
        // SSE=2; SST=.5 independently of the response offset. All arithmetic is dyadic.
        expect(await readFloat32(buffers['metrics'])).toEqual([2, 0, 1, 1, 1, -3]);
        expect(await readUint32(buffers['status'])).toEqual([0]);
      } finally {
        for (const buffer of Object.values(buffers)) buffer.destroy();
      }
    }
  } finally {
    for (const kernel of kernels) kernel.destroy();
  }
});

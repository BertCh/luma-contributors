// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUPointPatternEnvelopeOperation} from '../../../src/gpu-spatial-analysis/point-process/gpu-point-pattern-envelope-operation';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {runPoisonedOperation} from './output-oracle-harness';
import {getForwardedEnvelopeCase, MAXIMUM_DEVIATION_CASES} from './forwarded-envelope-fixtures';
import {
  getBoundaryEnvelopeCase,
  expectBoundaryEnvelope,
  getResidualEnvelopeCase,
  expectResidualEnvelope
} from './final-wave-envelope-fixtures';

for (const {reference, probability} of MAXIMUM_DEVIATION_CASES) {
  it(`TWN-03 GPU maximum deviation with reference ${reference}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const {props, inputs} = getForwardedEnvelopeCase(false);
    inputs.reference.values = Float32Array.of(reference);
    const result = await runPoisonedOperation(
      device,
      new GPUPointPatternEnvelopeOperation(props),
      inputs,
      Object.values(props.output) as string[]
    );
    expect(Array.from(result['status'])).toEqual([0]);
    expect(Array.from(result['statistics'])).toEqual([1, probability, 0.125, 7]);
  });
}

it('W5-11 GPU includes the attainable 1/49 boundary but excludes its predecessor', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const below of [false, true]) {
    const {props, inputs} = getBoundaryEnvelopeCase(below);
    expectBoundaryEnvelope(
      await runPoisonedOperation(
        device,
        new GPUPointPatternEnvelopeOperation(props),
        inputs,
        Object.values(props.output) as string[]
      ),
      below
    );
  }
});

it('TWN-03 GPU retains sub-binary64 residuals through the inference decision', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const exponent of [-13, -27]) {
    const {props, inputs} = getResidualEnvelopeCase(exponent);
    expectResidualEnvelope(
      await runPoisonedOperation(
        device,
        new GPUPointPatternEnvelopeOperation(props),
        inputs,
        Object.values(props.output) as string[]
      ),
      exponent
    );
  }
});

it('TWN-03 GPU retains exact integrated ties despite different ordinate order', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const {props, inputs} = getResidualEnvelopeCase(-27);
  inputs.simulations.values = Float32Array.from({length: 38}, (_, index) =>
    index % 2 ? 1 : 2 ** -27
  );
  const result = await runPoisonedOperation(
    device,
    new GPUPointPatternEnvelopeOperation(props),
    inputs,
    Object.values(props.output) as string[]
  );
  // Both exact sums equal 1 + 2^-54, hence all 19 draws are inclusive ties.
  expect(Array.from(result['statistics'])).toEqual([1, 1, Math.fround(1 / 20), 19]);
  expect(Array.from(result['status'])).toEqual([0]);
});

for (const integrated of [false, true]) {
  it(`forwarded GPU envelope preserves ${integrated ? 'N2 positive subnormal weights' : 'B2 maximum-deviation residuals'}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const {props, inputs} = getForwardedEnvelopeCase(integrated);
    const result = await runPoisonedOperation(
      device,
      new GPUPointPatternEnvelopeOperation(props),
      inputs,
      Object.values(props.output) as string[]
    );
    expect(Array.from(result['status'])).toEqual([0]);
    // Exact inference: zero exceedances; displayed statistics remain rounded device arithmetic.
    expect(Number.isFinite(result['statistics'][0])).toBe(true);
    expect(Array.from(result['statistics'].slice(1))).toEqual(
      integrated ? [0.25, 0.25, 3] : [0.125, 0.125, 7]
    );
    if (!integrated) expect(result['statistics'][0]).toBe(1);
  });
}

// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {it} from 'vitest';
import {CPUPointPatternEnvelopeOperation} from '../../../src/gpu-spatial-analysis/point-process/cpu-point-pattern-envelope-operation';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {
  getBoundaryEnvelopeCase,
  expectBoundaryEnvelope,
  getResidualEnvelopeCase,
  expectResidualEnvelope
} from './final-wave-envelope-fixtures';

it('W5-11 includes the attainable 1/49 boundary but excludes its predecessor', async () => {
  const device = createNullWebGPUDevice();
  try {
    for (const below of [false, true]) {
      const {props, inputs} = getBoundaryEnvelopeCase(below);
      expectBoundaryEnvelope(
        await runOperationGraph(device, {
          inputs,
          operations: [new CPUPointPatternEnvelopeOperation(props)],
          outputs: Object.values(props.output) as string[]
        }),
        below
      );
    }
  } finally {
    device.destroy();
  }
});

it('TWN-03 retains sub-binary64 residuals through the inference decision', async () => {
  const device = createNullWebGPUDevice();
  try {
    for (const exponent of [-13, -27]) {
      const {props, inputs} = getResidualEnvelopeCase(exponent);
      expectResidualEnvelope(
        await runOperationGraph(device, {
          inputs,
          operations: [new CPUPointPatternEnvelopeOperation(props)],
          outputs: Object.values(props.output) as string[]
        }),
        exponent
      );
    }
  } finally {
    device.destroy();
  }
});

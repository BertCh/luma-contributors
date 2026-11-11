// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPUCommandGraph,
  createTransientView,
  type GPUGraphVectorFormat
} from '@luma.gl/graph/gpu-core';
import {expect} from 'vitest';
import type {GPUContributorOperation} from '../../../src/analysis-core/operations/gpu-contributor-operation';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

/** Constructs the contributor against null-device views without compiling or executing shaders. */
export function constructRegressionOperationContributor(
  operation: GPUContributorOperation,
  inputs: Readonly<Record<string, {format: GPUGraphVectorFormat; length: number}>>
) {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const views = Object.fromEntries(
      Object.entries({...inputs, ...operation.outputs}).map(([port, shape]) => [
        port,
        createTransientView(graph, port, shape.format, shape.length)
      ])
    );
    return operation.createContributor(port => views[port] as never);
  } finally {
    operation.destroy();
    device.destroy();
  }
}

/** Relative error for nonzero values; zero and non-finite values compare numerically. */
export function expectRegressionRowsClose(
  actual: ArrayLike<number>,
  expected: ArrayLike<number>,
  relativeTolerance: number
): void {
  expect(actual.length).toBe(expected.length);
  for (let index = 0; index < expected.length; index++) {
    const value = expected[index];
    if (Number.isNaN(value)) {
      expect(actual[index], `row ${index}`).toBeNaN();
    } else if (value === 0 || !Number.isFinite(value)) {
      // Numeric equality deliberately does not compare the sign of zero.
      expect(actual[index] === value, `row ${index}`).toBe(true);
    } else {
      expect(Math.abs(actual[index] - value) / Math.abs(value), `row ${index}`).toBeLessThanOrEqual(
        relativeTolerance
      );
    }
  }
}

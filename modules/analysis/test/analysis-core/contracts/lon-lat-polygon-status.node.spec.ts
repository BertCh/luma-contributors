// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, expectTypeOf, it} from 'vitest';
import {AnalysisValidationError} from '../../../src/analysis-core/kernels/analysis-validation-error';
import {
  validateGPUStatusPort,
  type GPULonLatPolygonStatusPort,
  type GPUStatusScalar
} from '../../../src/analysis-core/contracts/status';

function createStatusScalar(format: string): GPUStatusScalar {
  return {format, length: 1} as GPUStatusScalar;
}

describe('longitude-latitude polygon status contract', () => {
  it('exposes exactly the antimeridian and pole uncertainty keys', () => {
    expectTypeOf<keyof GPULonLatPolygonStatusPort>().toEqualTypeOf<
      'antimeridianUncertainCount' | 'poleUncertainCount'
    >();
    const keys: Array<keyof GPULonLatPolygonStatusPort> = [
      'antimeridianUncertainCount',
      'poleUncertainCount'
    ];
    expect(keys).toEqual(['antimeridianUncertainCount', 'poleUncertainCount']);
  });

  it('accepts uint32 scalar uncertainty counts', () => {
    const status: GPULonLatPolygonStatusPort = {
      antimeridianUncertainCount: createStatusScalar('uint32'),
      poleUncertainCount: createStatusScalar('uint32')
    };
    expect(() => validateGPUStatusPort('X', status)).not.toThrow();
  });

  it('rejects a float32 pole uncertainty count with the exact diagnostic', () => {
    let failure: unknown;
    try {
      validateGPUStatusPort('X', {
        antimeridianUncertainCount: createStatusScalar('uint32'),
        poleUncertainCount: createStatusScalar('float32')
      });
    } catch (error) {
      failure = error;
    }
    expect(failure).toBeInstanceOf(AnalysisValidationError);
    expect(failure).toMatchObject({
      name: 'AnalysisValidationError',
      message: 'X poleUncertainCount must contain one uint32 row',
      id: 'X',
      contributor: 'X',
      code: 'REQUIREMENT',
      path: 'poleUncertainCount',
      received: {format: 'float32', length: 1},
      expected: {format: 'uint32', length: 1}
    });
  });
});

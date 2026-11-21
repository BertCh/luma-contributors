// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {getGPUNetworkKFunctionParameterValues} from '../../../src/gpu-network/network-k-function/network-k-function-parameters';

it('PCK-12 rejects uint32 overflow before writing and preserves the split seed and zero count', () => {
  const target = Uint32Array.of(9, 9, 9, 9);
  expect(() =>
    getGPUNetworkKFunctionParameterValues(
      {seed: 2 ** 32 + 1, activeSimulations: 4_294_967_296},
      target
    )
  ).toThrow(/activeSimulations/);
  expect(Array.from(target)).toEqual([9, 9, 9, 9]);
  // Euclidean base-2^32 decomposition of 2^32+1 is low=1, high=1.
  expect(
    Array.from(getGPUNetworkKFunctionParameterValues({seed: 2 ** 32 + 1, activeSimulations: 0}))
  ).toEqual([1, 1, 0, 0]);
  expect(getGPUNetworkKFunctionParameterValues({seed: 0, activeSimulations: 0xffffffff})[2]).toBe(
    0xffffffff
  );
});

it('PCK-12 enforces the optional compiled simulation limit', () => {
  expect(() =>
    getGPUNetworkKFunctionParameterValues({seed: 1, activeSimulations: 2}, undefined, 1, 'my-k')
  ).toThrow(/my-k.*activeSimulations/);
  expect(
    getGPUNetworkKFunctionParameterValues({seed: 1, activeSimulations: 1}, undefined, 1)[2]
  ).toBe(1);
});

for (const [name, seed, target, path, received] of [
  ['short target', 1, new Uint32Array(3), 'target.length', 3],
  ['invalid seed', -1, new Uint32Array(4), 'seed', -1]
] as const) {
  it(`S4 preserves the instance identity and received value for ${name}`, () => {
    let error: unknown;
    try {
      getGPUNetworkKFunctionParameterValues({seed, activeSimulations: 0}, target, 1, 'custom-k');
    } catch (caught) {
      error = caught;
    }
    expect(error).toMatchObject({id: 'custom-k', path, received});
    expect(String(error)).toMatch(/custom-k/);
  });
}

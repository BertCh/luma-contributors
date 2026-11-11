// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {
  getPhiloxOpenUnitFloat,
  getPhiloxUnitFloat,
  PHILOX_WGSL
} from '../../src/analysis-core/kernels/philox-random';
import {requireTestDevice} from './require-test-device';
import {createSeededRandom, getFloatWords, runWGSLModuleKernel} from './wgsl-module-harness';

// WGSL `philoxOpenUnitFloat` and `philoxUnitFloat` against their CPU forms, bit for bit, at the
// ends of the word range and on random words, and the two ends against hand-written values.

it('WGSL philoxOpenUnitFloat stays inside (0, 1) and agrees with the CPU form bit for bit', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const random = createSeededRandom(905);
  const words = Uint32Array.from([
    0,
    1,
    511,
    512,
    0x7fffffff,
    0x80000000,
    0xfffffdff,
    0xfffffe00,
    0xffffffff,
    ...Array.from({length: 2000}, () => Math.floor(random() * 4294967296))
  ]);
  const output = await runWGSLModuleKernel(device, {
    id: 'philox-open-unit-float',
    inputs: {words},
    outputLength: words.length * 3,
    invocationCount: words.length,
    declarations: PHILOX_WGSL,
    body: `let word = words[wordsOffset + index];
  let open = philoxOpenUnitFloat(word);
  output[outputOffset + index * 3u] = bitcast<u32>(open);
  output[outputOffset + index * 3u + 1u] = bitcast<u32>(philoxUnitFloat(word));
  output[outputOffset + index * 3u + 2u] = bitcast<u32>(log(open));`
  });
  const values = getFloatWords(output);
  // Smallest and largest word: 2^-24 and 1 - 2^-24.
  expect(values[0]).toBe(2 ** -24);
  expect(values[8 * 3]).toBe(1 - 2 ** -24);
  for (let index = 0; index < words.length; index++) {
    const open = values[index * 3];
    expect(open, `word ${words[index]}`).toBe(getPhiloxOpenUnitFloat(words[index]));
    expect(values[index * 3 + 1], `word ${words[index]}`).toBe(getPhiloxUnitFloat(words[index]));
    expect(open > 0 && open < 1).toBe(true);
    // The draw can feed log(): finite and strictly negative on the GPU too.
    const logarithm = values[index * 3 + 2];
    expect(Number.isFinite(logarithm) && logarithm < 0, `log of word ${words[index]}`).toBe(true);
  }
});

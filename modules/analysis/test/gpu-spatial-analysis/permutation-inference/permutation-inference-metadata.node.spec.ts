// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {expect, it} from 'vitest';
import {runInNewContext} from 'node:vm';
import {
  decodeGPUGlobalPermutationTestInferenceMetadata,
  decodeGPULocalPermutationTestInferenceMetadata,
  type GPUPermutationInferenceMetadata
} from '../../../src/gpu-spatial-analysis/permutation-inference/permutation-inference-metadata';

const metadata: GPUPermutationInferenceMetadata = {
  alternative: 'greater',
  requestedPermutationCount: 9,
  multipleTesting: 'none',
  testedRowsDefinition: 'fixture',
  stoppingRule: {method: 'besag-clifford', stopAfterExceedances: 2}
};

it('SSR-114 decodes recorded full, early-stopped and unavailable global draws', () => {
  expect(() =>
    decodeGPUGlobalPermutationTestInferenceMetadata(
      {...metadata, ...{permutationCount: 9}},
      new Float32Array(12)
    )
  ).toThrow(/permutationCount was removed.*requestedPermutationCount/);
  for (const count of [9, 2]) {
    const summary = new Float32Array(12);
    summary[3] = count;
    expect(decodeGPUGlobalPermutationTestInferenceMetadata(metadata, summary)).toMatchObject({
      requestedPermutationCount: 9,
      actualPermutationCount: count,
      stoppedEarly: count < 9,
      available: true
    });
    const offset = new Uint8Array(52);
    offset.set(new Uint8Array(summary.buffer), 4);
    expect(
      decodeGPUGlobalPermutationTestInferenceMetadata(metadata, offset.subarray(4))
        .actualPermutationCount
    ).toBe(count);
  }
  const summary = new Float32Array(12).fill(NaN);
  expect(decodeGPUGlobalPermutationTestInferenceMetadata(metadata, summary)).toMatchObject({
    actualPermutationCount: 0,
    available: false
  });
  expect(() =>
    decodeGPUGlobalPermutationTestInferenceMetadata(metadata, new Float32Array(3))
  ).toThrow(/length/);
  expect(() =>
    decodeGPUGlobalPermutationTestInferenceMetadata(metadata, new Uint8Array(49))
  ).toThrow(/whole number/);
});

it('SSR-114 retains actual local draws and the untested sentinel without mutating readback', () => {
  const counts = Uint32Array.of(9, 2, 0xffffffff);
  expect(
    Array.from(
      decodeGPULocalPermutationTestInferenceMetadata(metadata, counts, 3).actualPermutationCounts
    )
  ).toEqual([9, 2, 0xffffffff]);
  const offset = new Uint8Array(13);
  offset.set(new Uint8Array(counts.buffer), 1);
  expect(
    Array.from(
      decodeGPULocalPermutationTestInferenceMetadata(metadata, offset.subarray(1), 3)
        .actualPermutationCounts
    )
  ).toEqual([9, 2, 0xffffffff]);
  expect(() =>
    decodeGPULocalPermutationTestInferenceMetadata(metadata, Uint32Array.of(0), 1)
  ).toThrow(/permutationCount/);
  expect(() => decodeGPULocalPermutationTestInferenceMetadata(metadata, counts.buffer, 4)).toThrow(
    /length/
  );
});

it('S2 rejects impossible recorded draw counts and keeps the legacy count planned in both decoders', () => {
  const fixed: GPUPermutationInferenceMetadata = {
    ...metadata,
    requestedPermutationCount: 13,
    stoppingRule: {method: 'fixed-count'}
  };
  const stopped: GPUPermutationInferenceMetadata = {
    ...fixed,
    stoppingRule: {method: 'besag-clifford', stopAfterExceedances: 5}
  };
  const summary = new Float32Array(12);
  summary[3] = 2;
  for (const plan of [fixed, stopped]) {
    expect(() => decodeGPUGlobalPermutationTestInferenceMetadata(plan, summary)).toThrow(
      /permutationCount/
    );
    expect(() =>
      decodeGPULocalPermutationTestInferenceMetadata(plan, Uint32Array.of(2), 1)
    ).toThrow(/permutationCount/);
  }
  summary[3] = 5;
  expect(decodeGPUGlobalPermutationTestInferenceMetadata(stopped, summary)).toMatchObject({
    requestedPermutationCount: 13,
    actualPermutationCount: 5
  });
  expect(
    decodeGPULocalPermutationTestInferenceMetadata(stopped, Uint32Array.of(5), 1)
      .requestedPermutationCount
  ).toBe(13);
});

it('SSR-114 accepts foreign buffers and rejects the wrong typed element kind', () => {
  // Twelve float32 summary words, with draw count 3 in word 3, and local count 3.
  for (const expression of [
    'Float32Array.of(0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0)',
    'Float32Array.of(0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0).buffer',
    `(() => { const bytes = new Uint8Array(49); bytes.set(new Uint8Array(Float32Array.of(0,0,0,3,0,0,0,0,0,0,0,0).buffer), 1); return bytes.subarray(1); })()`
  ]) {
    expect(
      decodeGPUGlobalPermutationTestInferenceMetadata(metadata, runInNewContext(expression))
        .actualPermutationCount
    ).toBe(3);
  }
  for (const expression of [
    'Uint32Array.of(3)',
    'Uint32Array.of(3).buffer',
    `(() => { const bytes = new Uint8Array(5); bytes.set(new Uint8Array(Uint32Array.of(3).buffer), 1); return bytes.subarray(1); })()`
  ]) {
    expect(
      Array.from(
        decodeGPULocalPermutationTestInferenceMetadata(metadata, runInNewContext(expression), 1)
          .actualPermutationCounts
      )
    ).toEqual([3]);
  }
  expect(() =>
    decodeGPULocalPermutationTestInferenceMetadata(
      metadata,
      new Float32Array(Uint32Array.of(3).buffer) as unknown as Uint32Array,
      1
    )
  ).toThrow(/Uint32Array/);
  expect(() =>
    decodeGPUGlobalPermutationTestInferenceMetadata(
      metadata,
      new Uint32Array(
        Float32Array.of(0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0).buffer
      ) as unknown as Float32Array
    )
  ).toThrow(/Float32Array/);
});

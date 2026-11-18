// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {getWGSLModuleUsage} from '../../src/testing/wgsl-memory-usage';
import {getPrivateMemoryBytes} from '../../src/testing/gpu-contributor-test-utils';
import {auditWGSLModule} from '../analysis-core/kernel-audit/kernel-audit-wgsl';

it.each([
  'UnknownLayout',
  'LayoutAlias'
])('reports an unresolved immutable %s layout instead of silently treating it as a register', type => {
  const source = `alias LayoutAlias = UnknownLayout;
      fn probe(input: ${type}) { let copied: ${type} = input; }`;
  expect(getWGSLModuleUsage(source).unresolvedTypes).toEqual(['UnknownLayout']);
  expect(() => getPrivateMemoryBytes(source)).toThrow(/unresolved.*UnknownLayout/);
  expect(auditWGSLModule(source)).toContainEqual({
    check: 'unresolved memory layout',
    detail: 'UnknownLayout'
  });
});

it('keeps scalar, vector, matrix, pointer and resource let bindings as registers', () => {
  const source = `alias Matrix = mat2x2f;
    @group(0) @binding(0) var image: texture_2d<f32>;
    @group(0) @binding(1) var imageSampler: sampler;
    fn probe() {
      var storage: array<u32, 4>;
      let scalar: u32 = 1u;
      let vector: vec3f = vec3f(1.0);
      let matrix: Matrix = Matrix();
      let pointer: ptr<function, array<u32, 4>> = &storage;
      let textureHandle: texture_2d<f32> = image;
      let samplerHandle: sampler = imageSampler;
    }`;
  const usage = getWGSLModuleUsage(source);
  expect(usage.unresolvedTypes).toEqual([]);
  expect(usage.unresolvedLengths).toEqual([]);
  expect(usage.functionBytes).toBe(16);
  expect(usage.localArrayBytes).toBe(16);
  expect(getPrivateMemoryBytes(source)).toBe(16);
});

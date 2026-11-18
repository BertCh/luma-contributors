// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  getPrivateMemoryBytes,
  getWorkgroupMemoryBytes
} from '../../src/testing/gpu-contributor-test-utils';
import {getParsedKernelSources} from '../../src/testing/gpu-contributor-test-utils';
import type {Device} from '@luma.gl/core';
import {auditWGSLModule} from '../analysis-core/kernel-audit/kernel-audit-wgsl';

it('parsed-source budgets retain recorded override constants', () => {
  const source = 'override N: u32 = 1u; var<workgroup> scratch: array<u32, N>;';
  const nodes = [
    {
      compile: ({device}: {device: Device}) => {
        const shader = device.createShader({stage: 'compute', source});
        device.createComputePipeline({
          id: 'budget',
          shader,
          constants: {N: 8192}
        });
        return {destroy: () => shader.destroy()};
      }
    }
  ];
  const sources = getParsedKernelSources(nodes);
  for (const kernel of sources.getConfigurations('budget')) {
    expect(getWorkgroupMemoryBytes(kernel.source, kernel.constants)).toBe(32768);
  }
});

it('resolves local constants only in their lexical block and keeps struct lengths at module scope', () => {
  const source = `const N = 1u;
    struct ModuleSized { values: array<u32, N> }
    fn helper() {
      { const N = 8192u; var scratch: array<u32, N>; }
      var later: array<u32, N>;
      { const N = 2u; var structure: ModuleSized; }
    }`;
  // 8192 words in the nested block, one later word, one word in the module-sized struct.
  expect(getPrivateMemoryBytes(source)).toBe(32776);
  expect(auditWGSLModule(source).map(finding => finding.check)).toContain('function bytes');
});

it.each([
  [
    'numeric override ID',
    '@id(7) override N: u32 = 1u; var<workgroup> scratch: array<u32, N>;',
    {'7': 8192}
  ],
  [
    'local constant shadow',
    'const N = 8192u; var<workgroup> scratch: array<u32, N>; fn helper() { const N = 1u; }',
    {}
  ],
  [
    'integer division at each operation',
    'const N = 16384u - (3u / 2u) * 8192u; var<workgroup> scratch: array<u32, N>;',
    {}
  ],
  ['whitespace before address space', 'var <workgroup> scratch: array<u32, 8192>;', {}]
] as const)('rejects an oversized workgroup array: %s', (_name, source, constants) => {
  expect(getWorkgroupMemoryBytes(source, constants)).toBe(32768);
  expect(auditWGSLModule(source, constants).map(finding => finding.check)).toContain(
    'workgroup bytes'
  );
});

it.each([
  [
    'initializer-derived private array',
    'var<private> scratch = array<u32, 8192>();',
    'private bytes'
  ],
  [
    'explicit function address space',
    'fn helper() { var<function> scratch: array<u32, 8192>; }',
    'function bytes'
  ],
  [
    'initializer function returning an array',
    'fn makeScratch() -> array<u32, 8192> { return array<u32, 8192>(); } fn helper() { var scratch = makeScratch(); }',
    'function bytes'
  ],
  [
    'initializer function return size stays at module scope',
    'const N = 8192u; fn makeScratch() -> array<u32, N> { return array<u32, N>(); } fn helper() { const N = 1u; var scratch = makeScratch(); }',
    'function bytes'
  ]
])('rejects an oversized array: %s', (_name, source, check) => {
  expect(getPrivateMemoryBytes(source)).toBe(32768);
  expect(auditWGSLModule(source).map(finding => finding.check)).toContain(check);
});

it('counts arrays sized by named constants instead of one scalar', () => {
  const source = 'const N = 8192u; var<workgroup> scratch: array<u32, N>;';
  expect(getWorkgroupMemoryBytes(source)).toBe(32768);
  expect(getPrivateMemoryBytes('const N = 2048u; var<private> scratch: array<vec4f, N>;')).toBe(
    32768
  );
});

it('resolves override expressions, aliases and padded structs', () => {
  const source = `override count: u32 = 2u;
    struct Item { position: vec3f, value: f32, }
    alias Items = array<Item, count * 2u>;
    var<workgroup> items: Items;`;
  expect(getWorkgroupMemoryBytes(source, {count: 4})).toBe(128);
});

it('refuses to certify unresolved memory as a small numeric bound', () => {
  expect(() =>
    getWorkgroupMemoryBytes('override count: u32; var<workgroup> items: array<u32, count>;')
  ).toThrow(/unresolved/i);
  expect(() => getPrivateMemoryBytes('var<private> items: UnknownType;')).toThrow(/unresolved/i);
});

it('honors explicit struct alignment and member size', () => {
  expect(
    getWorkgroupMemoryBytes(
      'struct Item { @align(64) @size(128) value: f32 } var<workgroup> items: array<Item, 2>;'
    )
  ).toBe(256);
});

it('counts function locals on a single line and ignores commented declarations', () => {
  expect(
    getPrivateMemoryBytes('fn helper() { var scratch: array<u32, 64>; }')
  ).toBeGreaterThanOrEqual(256);
  expect(
    getWorkgroupMemoryBytes('// var<workgroup> fake: array<u32, 8192>;\nvar<workgroup> real: u32;')
  ).toBe(16);
});

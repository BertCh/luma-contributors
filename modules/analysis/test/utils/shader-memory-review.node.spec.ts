// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {getWGSLModuleUsage} from '../../src/testing/wgsl-memory-usage';
import {
  getPrivateMemoryBytes,
  getWorkgroupMemoryBytes,
  getStorageBindingCount
} from '../../src/testing/gpu-contributor-test-utils';
import {auditWGSLModule} from '../analysis-core/kernel-audit/kernel-audit-wgsl';
import type {Device} from '@luma.gl/core';
import {getGPUCommandNodeKernelSources} from '../../src/testing/gpu-contributor-test-utils';

const STORAGE =
  '@group(0) @binding(0) var<storage, read> data: array<u32,4096>; @group(0) @binding(1) var<storage,read_write> sink: array<u32,1>;';

it.each([
  false,
  true
])('R1 preserves identical WGSL with oversized overrides in either order, reversed=%s', reversed => {
  const counts = reversed ? [1, 8192] : [8192, 1];
  const nodes = counts.map(count => ({
    compile: ({device}: {device: Device}) => {
      const shader = device.createShader({
        stage: 'compute',
        source: 'override N: u32 = 1u; var<workgroup> scratch: array<u32,N>;'
      });
      device.createComputePipeline({id: 'shared', shader, constants: {N: count}});
      return {destroy: () => shader.destroy()};
    }
  }));
  const configurations = getGPUCommandNodeKernelSources(nodes);
  expect(
    configurations.map(configuration =>
      getWorkgroupMemoryBytes(configuration.source, configuration.constants)
    )
  ).toEqual(counts.map(count => Math.ceil((count * 4) / 16) * 16));
  expect(() => {
    for (const configuration of configurations)
      expect(
        getWorkgroupMemoryBytes(configuration.source, configuration.constants)
      ).toBeLessThanOrEqual(16384);
  }).toThrow();
});

it.each([false, true])('R2 ends the for initializer scope after the body, wrapped=%s', wrapped => {
  const loop = 'for (var data = 0u; data < 1u; data++) { sink[0] = data; }';
  const source = `${STORAGE} @compute @workgroup_size(1) fn main() {
    ${wrapped ? `{ ${loop} }` : loop} let copy = data; sink[0] = copy[1];
  }`;
  const usage = getWGSLModuleUsage(source);
  expect(usage.localArrayBytes).toBe(16384);
  expect(usage.unresolvedTypes).toEqual([]);
  expect(auditWGSLModule(source).map(finding => finding.check)).toContain('function bytes');
});

it.each([
  'let copy',
  'let/* copy */copy',
  'let/* outer /* nested */ comment */copy'
])('R3 preserves declaration tokens around block comments: %s', declaration => {
  const source = `${STORAGE} @compute @workgroup_size(1) fn main() { ${declaration} = data; sink[0] = copy[1]; }`;
  expect(getWGSLModuleUsage(source).localArrayBytes).toBe(16384);
  expect(getPrivateMemoryBytes(source)).toBe(16384);
  expect(auditWGSLModule(source).map(finding => finding.check)).toContain('function bytes');
});

it.each([
  'vec4<UnknownLayout>',
  'mat4x4<UnknownLayout>'
])('R4 refuses an unknown component of %s', type => {
  const source = `fn probe(input: ${type}) { let copied: ${type} = input; }`;
  expect(getWGSLModuleUsage(source).unresolvedTypes.length).toBeGreaterThan(0);
  expect(() => getPrivateMemoryBytes(source)).toThrow(/unresolved/);
  expect(auditWGSLModule(source).map(finding => finding.check)).toContain(
    'unresolved memory layout'
  );
});

it.each([
  ['vec4<Half>', 2048],
  ['mat4x4<Half>', 512]
] as const)('R4 resolves aliased f16 components at the exact workgroup limit: %s', (type, count) => {
  const source = `enable f16; alias Half = f16; var<workgroup> scratch: array<${type},${count}>; @compute @workgroup_size(1) fn main() { scratch[0] = ${type}(); }`;
  expect(getWorkgroupMemoryBytes(source)).toBe(16384);
  expect(auditWGSLModule(source)).toEqual([]);
});

it('R5 counts default-read and explicit-access storage declarations identically', () => {
  const source =
    Array.from(
      {length: 9},
      (_, index) =>
        `@group(0) @binding(${index}) var<storage${index === 8 ? ', read_write' : ''}> data${index}: array<u32>;`
    ).join('\n') +
    ` @compute @workgroup_size(1) fn main() { data8[0] = ${Array.from({length: 8}, (_, index) => `data${index}[0]`).join(' + ')}; }`;
  expect(getStorageBindingCount(source)).toBe(9);
  expect(getWGSLModuleUsage(source).storageBuffers).toBe(9);
  expect(auditWGSLModule(source)).toContainEqual({
    check: 'storage buffers per stage',
    detail: '9 > 8'
  });
});

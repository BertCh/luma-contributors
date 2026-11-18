// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {getWGSLModuleUsage} from '../../src/testing/wgsl-memory-usage';
import {
  getPrivateMemoryBytes,
  getWorkgroupMemoryBytes
} from '../../src/testing/gpu-contributor-test-utils';
import {auditWGSLModule} from '../analysis-core/kernel-audit/kernel-audit-wgsl';

const TWO_WORKGROUP_VARIABLES = `
  var<workgroup> a: array<u32,4093>;
  var<workgroup> b: u32;
  @compute @workgroup_size(1) fn main() { a[0] = b; }
`;

it('X3 rejects per-variable rounded workgroup storage above 16 KiB', () => {
  expect(getWGSLModuleUsage(TWO_WORKGROUP_VARIABLES).workgroupBytes).toBe(16376);
  expect(getWGSLModuleUsage(TWO_WORKGROUP_VARIABLES).workgroupStorageBytes).toBe(16400);
  expect(auditWGSLModule(TWO_WORKGROUP_VARIABLES)).toContainEqual({
    check: 'workgroup bytes',
    detail: '16400 > 16384'
  });
});

it('X3 makes the shared workgroup budget helper use per-variable rounding', () => {
  expect(getWorkgroupMemoryBytes(TWO_WORKGROUP_VARIABLES)).toBe(16400);
});

it('X3 accepts a single workgroup variable at exactly 16 KiB', () => {
  const source = `var<workgroup> a: array<u32,4096>;
    @compute @workgroup_size(1) fn main() { a[0] = 1u; }`;
  expect(getWGSLModuleUsage(source).workgroupBytes).toBe(16384);
  expect(getWGSLModuleUsage(source).workgroupStorageBytes).toBe(16384);
  expect(getWorkgroupMemoryBytes(source)).toBe(16384);
  expect(auditWGSLModule(source)).toEqual([]);
});

it.each(
  ['let', 'var'].flatMap(kind =>
    ['copy', 'τ', 'copyτ', 't\u0301', '𐐀', 'τvar', 'varτ'].map(name => [kind, name])
  )
)('X8 counts a live %s array copy named %s', (kind, name) => {
  const source = `
      @group(0) @binding(0) var<storage> data: array<u32,4096>;
      @group(0) @binding(1) var<storage,read_write> sink: array<u32,1>;
      @compute @workgroup_size(1) fn main() {
        ${kind} ${name} = data; sink[0] = ${name}[1];
      }`;
  const usage = getWGSLModuleUsage(source);
  expect(usage.localArrayBytes).toBe(16384);
  expect(usage.unresolvedTypes).toEqual([]);
  expect(getPrivateMemoryBytes(source)).toBe(16384);
  expect(auditWGSLModule(source).map(finding => finding.check)).toContain('function bytes');
});

it('X8 resolves Unicode names through expressions, functions, parameters, members and constants', () => {
  const source = `
    const μήκος = 4096u;
    alias Δεδομένα = array<u32,μήκος>;
    struct Σχήμα { πεδίο: Δεδομένα, }
    @group(0) @binding(0) var<storage> είσοδος: Σχήμα;
    @group(0) @binding(1) var<storage,read_write> sink: array<u32,1>;
    fn αντίγραφο(τιμή: Δεδομένα) -> Δεδομένα { let τοπικό = τιμή; return τοπικό; }
    @compute @workgroup_size(1) fn κύριο() {
      let τ = αντίγραφο(είσοδος.πεδίο); let δεύτερο = τ; sink[0] = δεύτερο[1];
    }`;
  const usage = getWGSLModuleUsage(source);
  expect(usage.localArrayBytes).toBe(49152);
  expect(usage.unresolvedTypes).toEqual([]);
  expect(usage.unresolvedLengths).toEqual([]);
  expect(usage.functionName).toBe('κύριο');
});

it.each([
  'vec4<Μισό>',
  'mat2x2<Μισό>'
])('X8 resolves Unicode scalar aliases in %s workgroup layouts', type => {
  const source = `enable f16; alias Μισό = f16; const μήκος = 2048u;
      var<workgroup> χώρος: array<${type},μήκος>;
      @compute @workgroup_size(1) fn main() { χώρος[0] = ${type}(); }`;
  expect(getWorkgroupMemoryBytes(source)).toBe(16384);
  expect(getWGSLModuleUsage(source).unresolvedTypes).toEqual([]);
});

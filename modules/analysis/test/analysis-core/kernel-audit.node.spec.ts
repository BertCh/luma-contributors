// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readdirSync, readFileSync} from 'node:fs';
import {dirname, join, resolve} from 'node:path';
import {fileURLToPath} from 'node:url';
import {describe, expect, it} from 'vitest';
import * as publishedDataframe from '@luma.gl/graph/gpu-dataframe';
import * as publishedGeospatial from '@luma.gl/graph/geospatial';
import * as publishedRaster from '@luma.gl/graph/gpu-raster';
import {
  ANALYSIS_BARRELS,
  getAnalysisContributors,
  getContributorNames
} from './kernel-audit/analysis-contributors';
import {KERNEL_AUDIT_CASES} from './kernel-audit/kernel-audit-cases';
import {KERNEL_AUDIT_REJECTIONS} from './kernel-audit/kernel-audit-rejections';
import {auditWGSLModule, findUnusedBindings} from './kernel-audit/kernel-audit-wgsl';
import {auditKernelCase} from './kernel-audit/kernel-audit-runner';
import {createWGSLKernelNode} from '../../src/analysis-core/kernels/wgsl-kernel-nodes';

// Library-wide kernel audit (quality rule 16 and the BRIEF rule 6 compile traps). Every public
// contributor of the six analysis entries is built at maximal documented options on a fake 'core'
// device; every generated WGSL module is checked for core limits (8 storage buffers per stage,
// 8192 B private and per-function memory, 16384 B workgroup memory, workgroup size), binding and
// buffer sizes, `capacity x lanes <= 2^32 - 1`, WGSL reserved words used as identifiers, NaN
// self-compares, const-expression non-finite bitcasts, undeclared calls and unused bindings. These
// are exactly the shader-creation errors that `wgsl_reflect` accepts and Dawn rejects.

/**
 * Contributors the audit cannot build, with the reason. Keep it empty: a contributor that needs
 * real inputs gets a fake view in its case instead.
 */
const CHUNKED_WEIGHTS_AUDIT =
  'chunked weights take a caller-built chunk list per chunk count; weights7-chunked-weights-operations.node.spec.ts parses every kernel and checks the storage-binding limit across 1 to 16 chunks';
const SKIPPED_CONTRIBUTORS: Readonly<Record<string, string>> = {
  GPUChunkedSpatialWeightsAlgebra: CHUNKED_WEIGHTS_AUDIT,
  GPUChunkedSpatialWeightsTranspose: CHUNKED_WEIGHTS_AUDIT,
  GPUChunkedSpatialWeightsTransform: CHUNKED_WEIGHTS_AUDIT,
  GPUChunkedSpatialWeightsValidation: CHUNKED_WEIGHTS_AUDIT
};

/**
 * Contributors owned by published `@luma.gl/graph`. Legacy audit cases that name them are skipped:
 * they audit luma.gl code, not this package.
 */
const PUBLISHED_CONTRIBUTORS = getContributorNames({
  geospatial: publishedGeospatial,
  'gpu-dataframe': publishedDataframe,
  'gpu-raster': publishedRaster
});

const contributors = getAnalysisContributors();
const cases = Object.values(KERNEL_AUDIT_CASES).flat();

describe('kernel audit checks', () => {
  const checks = (source: string) => auditWGSLModule(source).map(finding => finding.check);

  it('rejects reserved words and keywords as identifiers', () => {
    expect(checks('fn f() { let target = 1u; }')).toEqual(['reserved word']);
    expect(checks('var<private> active: u32;')).toEqual(['reserved word']);
    expect(checks('struct S { filter: u32 }')).toEqual(['reserved word']);
    expect(checks('fn f(loop: u32) {}')).toEqual(['keyword identifier']);
    expect(checks('fn f() { switch (1u) { default: {} } }')).toEqual([]);
  });

  it('rejects NaN self-compares only when the operands are the whole comparison', () => {
    expect(checks('fn f(x: f32) -> bool { return x != x; }')).toEqual(['NaN self-compare']);
    expect(checks('fn f(v: vec2f) -> bool { return all(v.x == v.x); }')).toEqual([
      'NaN self-compare'
    ]);
    expect(checks('fn f(a: f32, x: f32) -> bool { return a + x == x; }')).toEqual([]);
    expect(checks('fn f(a: f32, b: f32) -> bool { return a == b; }')).toEqual([]);
  });

  it('rejects const-expression non-finite bitcasts and divisions by zero', () => {
    expect(checks('fn f() -> f32 { return bitcast<f32>(0x7f800000u); }')).toEqual([
      'const non-finite bitcast'
    ]);
    expect(
      checks('const NAN_BITS = 0x7fc00000u; fn f() -> f32 { return bitcast<f32>(NAN_BITS); }')
    ).toEqual(['const non-finite bitcast']);
    expect(checks('fn f() -> f32 { return 1.0 / 0.0; }')).toEqual(['const division by zero']);
    expect(checks('fn f(bits: u32) -> f32 { return bitcast<f32>(bits); }')).toEqual([]);
    expect(checks('fn f() -> f32 { return bitcast<f32>(0x3f800000u); }')).toEqual([]);
  });

  it('rejects calls to undeclared functions', () => {
    expect(checks('fn main() { let value = getSpatialAnalysisQuotient(1.0, 2.0); }')).toEqual([
      'undeclared call',
      'unresolved memory layout'
    ]);
    expect(checks('fn main() { let value: f32 = getSpatialAnalysisQuotient(1.0, 2.0); }')).toEqual([
      'undeclared call'
    ]);
    expect(
      checks(
        'fn getSpatialAnalysisQuotient(a: f32, b: f32) -> f32 { return a / b; } fn main() { let value = getSpatialAnalysisQuotient(1.0, 2.0); }'
      )
    ).toEqual([]);

    expect(checks('fn f(a: f32) -> f32 { return orientExact(a); }')).toEqual(['undeclared call']);
    expect(
      checks('fn g() -> f32 { return 1.0; } fn f() -> f32 { return max(g(), f32(1)); }')
    ).toEqual([]);
    expect(checks('fn f() -> vec2<f32> { return vec2<f32>(array<f32, 2>(1.0, 2.0)[0]); }')).toEqual(
      []
    );
  });

  it('accounts struct, override-sized and per-function memory', () => {
    expect(checks('var<private> big: array<vec4<f32>, 600>;')).toEqual(['private bytes']);
    const workgroup = `struct Pair { a: vec3<f32>, b: f32 }
override GROUP: u32 = 64u;
var<workgroup> table: array<Pair, GROUP * 32u>;`;
    expect(checks(workgroup)).toEqual(['workgroup bytes']);
    expect(auditWGSLModule(workgroup, {GROUP: 16})).toEqual([]);
    expect(checks('fn f() { var scratch: array<u32, 2049>; }')).toEqual(['function bytes']);
    expect(checks('@compute @workgroup_size(512) fn main() {}')).toEqual([
      'workgroup invocations',
      'workgroup size x'
    ]);
    const storage = Array.from(
      {length: 9},
      (_, index) =>
        `@group(0) @binding(${index}) var<storage, ${index === 0 ? 'read_write' : 'read'}> b${index}: array<u32>;`
    ).join('\n');
    const references = Array.from({length: 9}, (_, index) => `b${index}[0]`).join(' + ');
    expect(
      checks(`${storage} @compute @workgroup_size(1) fn main() { b0[0] = ${references}; }`)
    ).toEqual(['storage buffers per stage']);
  });

  it('rejects unused bindings through the common module audit', () => {
    const source = '@group(0) @binding(3) var<storage, read> counts: array<u32>; fn main() {}';
    expect(auditWGSLModule(source)).toEqual([
      {check: 'unused binding', detail: 'counts (@group(0) @binding(3))'}
    ]);
    expect(
      checks(source.replace('fn main() {}', 'fn main() { let length = arrayLength(&counts); }'))
    ).toEqual([]);
  });

  it('reports the declared node and binding through the shared case runner', () => {
    const result = auditKernelCase({
      exportName: 'GPUFittingGradientSums',
      name: 'unused-binding-audit-regression',
      build: ({graph, view}) => {
        const values = view('uint32', 1);
        for (const declared of [true, false]) {
          graph.add({
            id: declared ? 'unused-binding-node' : 'omitted-binding-node',
            type: 'compute',
            resources: [{buffer: values, usage: 'storage-read'}],
            compile: ({device}) => {
              const shader = device.createShader({
                stage: 'compute',
                source:
                  '@group(0) @binding(0) var<storage, read> unusedValues: array<u32>; @compute @workgroup_size(1) fn main() {}'
              });
              device.createComputePipeline({
                shader,
                shaderLayout: {
                  bindings: declared
                    ? [
                        {
                          name: 'hostAliasForValues',
                          type: 'read-only-storage',
                          group: 0,
                          location: 0
                        }
                      ]
                    : []
                }
              });
              return {encode: () => {}, destroy: () => shader.destroy()};
            }
          });
        }
      }
    });
    expect(result.findings).toHaveLength(1);
    expect(result.findings[0]).toEqual({
      check: 'unused binding',
      detail: 'unusedValues (@group(0) @binding(0))',
      location: expect.stringMatching(/^unused-binding-node /)
    });
  });

  it('accepts unused module uniforms that the node does not bind, while retaining application bindings', () => {
    const result = auditKernelCase({
      exportName: 'GPUFittingGradientSums',
      name: 'unused-module-binding-audit-regression',
      build: ({graph, view}) => {
        graph.add(
          createWGSLKernelNode(graph, {
            id: 'unused-module-uniform-node',
            operation: 'UnusedModuleBindingAudit',
            invocationCount: 1,
            bindings: [{name: 'values', view: view('uint32', 1), type: 'u32', access: 'read'}],
            declarations: '@group(0) @binding(100) var<uniform> moduleParameters: vec4u;',
            body: ''
          })
        );
      }
    });
    // The core builder's address reference keeps the application binding in the auto layout.
    expect(result.findings).toEqual([]);
  });

  it('ignores comments, offset names and member names when looking for binding uses', () => {
    expect(
      findUnusedBindings(`
@binding(0) @group(1) var<storage, read> values: array<u32>;
// values[0] is not a real use.
/* values /* nested values */ values */
struct Row { values: u32 }
fn main() { let valuesOffset = 0u; let row = Row(1u); let result = row . values; }
`)
    ).toEqual([{check: 'unused binding', detail: 'values (@binding(0) @group(1))'}]);
  });

  it('checks uniforms, textures and samplers with multiline and reversed attributes', () => {
    expect(
      findUnusedBindings(`
struct Parameters { scale: f32 }
@binding(2)
@group(0) var<uniform> parameters: Parameters;
@group(0) @binding(3) var image: texture_2d<f32>;
@group(0) @binding(4) var imageSampler: sampler;
fn main() { let result = textureSampleLevel(image, imageSampler, vec2f(0.0), 0.0); }
`)
    ).toEqual([{check: 'unused binding', detail: 'parameters (@binding(2) @group(0))'}]);
  });

  it('recognizes Unicode identifiers, helper references and constant-expression attributes', () => {
    expect(
      findUnusedBindings(`
@group(u32(0)) @binding(0) var<storage, read> réflexion: array<u32>;
@group(0) @binding(1) var<storage, read> 朝焼け: array<u32>;
fn getValue() -> u32 { return réflexion[0]; }
fn main() { let result = getValue() + 朝焼け[0]; }
`)
    ).toEqual([]);
    expect(
      findUnusedBindings('@group(0) @binding(0) var<storage, read> réflexion: array<u32>;')
    ).toHaveLength(1);
    expect(findUnusedBindings('var<private> unused: u32;')).toEqual([]);
  });

  it('matches node binding slots through constants and overrides rather than relying on host names', () => {
    const source = `const GROUP = 1u;
override SLOT: u32 = 2u;
@group(u32(GROUP)) @binding(SLOT) var<uniform> parameters: vec4u;`;
    const declared = [{name: 'hostParameters', group: 1, location: 2}];
    expect(findUnusedBindings(source, declared)).toEqual([
      {check: 'unused binding', detail: 'parameters (@group(u32(GROUP)) @binding(SLOT))'}
    ]);
    expect(
      findUnusedBindings(source, [{name: 'hostParameters', group: 1, location: 3}], {SLOT: 3})
    ).toEqual([
      {check: 'unused binding', detail: 'parameters (@group(u32(GROUP)) @binding(SLOT))'}
    ]);
  });
});

const SOURCE_ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '../../src');
const AUDITED_SOURCE_DIRECTORIES = [...Object.keys(ANALYSIS_BARRELS), 'analysis-core'];

function listSourceFiles(directory: string): string[] {
  return readdirSync(directory, {withFileTypes: true}).flatMap(entry => {
    const path = join(directory, entry.name);
    return entry.isDirectory() ? listSourceFiles(path) : entry.name.endsWith('.ts') ? [path] : [];
  });
}

describe('kernel audit sources', () => {
  it('analysis sources write no NaN self-compare (upstream idioms are filtered by text)', () => {
    // TypeScript compares with `===`, so `a == a` / `${a} != ${a}` in source text is WGSL.
    const selfCompare = /(?<![=!<>\w.\]])([A-Za-z_][\w.]*|\$\{[^}]+\})\s*(==|!=)\s*\1(?![\w(.[=])/g;
    const found: string[] = [];
    for (const directory of AUDITED_SOURCE_DIRECTORIES) {
      for (const file of listSourceFiles(join(SOURCE_ROOT, directory))) {
        readFileSync(file, 'utf8')
          .split('\n')
          .forEach((line, index) => {
            const code = line.replace(/\/\/.*$/, '');
            if (/^\s*(\*|\/\*)/.test(code)) return;
            for (const match of code.matchAll(selfCompare)) {
              found.push(`${file.slice(SOURCE_ROOT.length + 1)}:${index + 1} ${match[0]}`);
            }
          });
      }
    }
    expect(found).toEqual([]);
  });
});

describe('kernel audit coverage', () => {
  it('discovers inherited contributors while excluding named core helpers', () => {
    class BaseContributor {
      getCommandNodes() {
        return [];
      }
    }
    class InheritedContributor extends BaseContributor {}
    class Helper {}
    expect([...getContributorNames({fixture: {InheritedContributor, Helper}})]).toEqual([
      ['InheritedContributor', 'fixture']
    ]);
    const coreHelpers: Record<string, string> = {
      GPUParameterBuffer: 'owns parameter storage; does not produce command nodes',
      GPUContributorOperation: 'operation lowering adapter; not a command-node contributor'
    };
    const unclassified = Object.entries(ANALYSIS_BARRELS['analysis-core'])
      .filter(([name, value]) => /^GPU[A-Z]/.test(name) && typeof value === 'function')
      .map(([name]) => name)
      .filter(name => !contributors.has(name) && !(name in coreHelpers));
    expect(
      unclassified,
      'core classes need a kernel case or an explicit helper classification'
    ).toEqual([]);
    expect(
      [...contributors]
        .filter(([, entry]) => entry === 'analysis-core')
        .map(([name]) => name)
        .sort()
    ).toEqual([
      'GPUAdamStep',
      'GPUFittingCoordinateEncoder',
      'GPUFittingGradientSums',
      'GPUFittingMinibatchOrder'
    ]);
  });

  it('builds every public contributor or records why not', () => {
    expect(contributors.size, 'discovered public contributors').toBeGreaterThan(0);
    expect(cases.length, 'kernel-audit cases').toBeGreaterThan(0);
    const covered = new Set(cases.map(auditCase => auditCase.exportName));
    const missing = [...contributors.keys()].filter(
      name => !covered.has(name) && !(name in SKIPPED_CONTRIBUTORS)
    );
    expect(missing, 'contributors without a kernel-audit case').toEqual([]);
  });

  it('has no stale cases or skips', () => {
    const stale = [
      ...cases.map(auditCase => auditCase.exportName),
      ...Object.keys(SKIPPED_CONTRIBUTORS)
    ].filter(name => !contributors.has(name) && !PUBLISHED_CONTRIBUTORS.has(name));
    expect([...new Set(stale)], 'cases or skips naming no public contributor').toEqual([]);
    const labels = cases.map(auditCase => `${auditCase.exportName} ${auditCase.name}`);
    expect(labels.length, 'duplicate case labels').toBe(new Set(labels).size);
    const staleRejections = KERNEL_AUDIT_REJECTIONS.map(
      ({exportName, name}) => `${exportName} ${name}`
    ).filter(label => !labels.includes(label));
    expect(staleRejections, 'rejections naming no case').toEqual([]);
  });
});

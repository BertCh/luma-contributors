// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readFileSync} from 'node:fs';
import {expect, it} from 'vitest';
import {getWGSLModuleUsage} from '../../src/testing/wgsl-memory-usage';
import {getPrivateMemoryBytes} from '../../src/testing/gpu-contributor-test-utils';
import {auditWGSLModule} from '../analysis-core/kernel-audit/kernel-audit-wgsl';
import {getDenseCholeskyWGSL} from '../../src/analysis-core/kernels/dense-solve-wgsl';

const MONOTONE = 'analysis-core/polygon-triangulation/polygon-triangulation-monotone-wgsl.ts';
const TRACK = 'gpu-spatial-analysis/track-similarity/gpu-track-similarity.ts';
const DENSE = 'analysis-core/kernels/dense-solve-wgsl.ts';

// Each expression is checked against its production shader. The surrounding declarations are
// reduced fixed fixtures; a second local copy forces the initializer's type to be established.
const CASES = [
  {shader: MONOTONE, expression: '1u', fragment: 'var leftPosition = 1u;', bytes: 4},
  {
    shader: MONOTONE,
    expression: 'leftAvailable',
    fragment: 'var takeLeft = leftAvailable;',
    parameters: 'leftAvailable: bool',
    bytes: 4
  },
  {
    shader: MONOTONE,
    expression: 'leftPosition <= bottom',
    fragment: 'let leftAvailable = leftPosition <= bottom;',
    parameters: 'leftPosition: u32, bottom: u32',
    bytes: 4
  },
  {
    shader: MONOTONE,
    expression: 'select(row - 1u, ringEnd - 1u, row == ringBegin)',
    parameters: 'row: u32, ringEnd: u32, ringBegin: u32',
    bytes: 4
  },
  {
    shader: MONOTONE,
    expression: 'u32(sliceEnd - sliceStart) / 3u',
    parameters: 'sliceEnd: i32, sliceStart: i32',
    bytes: 4
  },
  {
    shader: TRACK,
    expression: 'min(lane * chunk, pointCount)',
    parameters: 'lane: u32, chunk: u32, pointCount: u32',
    bytes: 4
  },
  // The scaled metric still produces one f32: the copied mutable scalar occupies four bytes.
  {
    shader: TRACK,
    expression: 'dot(normalizedEdge, normalizedEdge)',
    parameters: 'normalizedEdge: vec2<f32>',
    bytes: 4
  },
  {
    shader: TRACK,
    expression:
      'clamp(dot((point - from0) / scale, normalizedEdge) / dot(normalizedEdge, normalizedEdge), 0.0, 1.0)',
    parameters: 'point: vec2f, from0: vec2f, normalizedEdge: vec2f, scale: f32',
    bytes: 4
  },
  {
    shader: DENSE,
    expression: '(*matrix)[column * 8u + column]',
    fragment: 'let original = (*matrix)[column * ${n}u + column];',
    parameters: 'matrix: ptr<function, array<f32, 64>>, column: u32',
    bytes: 4
  },
  {
    shader: DENSE,
    expression: 'result.size',
    fragment: 'let count = result.size;',
    prefix: 'struct Factor {size: u32, values: array<f32, 4096>}',
    parameters: 'result: Factor',
    bytes: 4
  },
  {
    shader: DENSE,
    expression: 'result.scale[row]',
    prefix: 'struct Factor {scale: array<f32, 4096>}',
    parameters: 'result: Factor, row: u32',
    bytes: 4
  },
  {
    shader: 'dataframe-analysis/column-classification/natural-breaks-nodes.ts',
    expression: 'local / teamSize',
    parameters: 'local: u32',
    body: 'let teamSize = max(WORKGROUP_SIZE >> level, 1u);',
    prefix: 'const WORKGROUP_SIZE = 256u; const level = 2u;',
    bytes: 4
  },
  {
    shader: 'gpu-spatial-analysis/spatial-interpolation/conditional-simulation-blocked.ts',
    expression: 'workgroupIndex / TILES_PER_AXIS',
    prefix: 'override TILES_PER_AXIS: u32;',
    parameters: 'workgroupIndex: u32',
    bytes: 4
  },
  {
    shader: 'gpu-spatial-analysis/delaunay-tessellation/gpu-delaunay-tessellation.ts',
    expression: '(first >> 31u) == 1u',
    parameters: 'first: u32',
    bytes: 4
  },
  {
    shader: 'gpu-terrain/terrain-features/gpu-terrain-summits.ts',
    expression: '-currentHalfWidth',
    parameters: 'currentHalfWidth: i32',
    bytes: 4
  },
  {
    shader: 'gpu-network/network-reachability/network-reachability-passes.ts',
    expression: 'atomicLoad(&counterHigh)',
    prefix: 'var<workgroup> counterHigh: atomic<u32>;',
    bytes: 4
  },
  {
    shader: 'gpu-spatial-analysis/catchment-accessibility/gpu-huff-trade-areas.ts',
    expression: 'first.fract',
    body: 'let first = frexp(value);',
    parameters: 'value: f32',
    bytes: 12
  },
  {
    shader: 'gpu-spatial-analysis/catchment-accessibility/gpu-huff-trade-areas.ts',
    expression: 'first.exp - second.exp',
    body: 'let first = frexp(value); let second = frexp(reference);',
    parameters: 'value: f32, reference: f32',
    bytes: 20
  },
  {
    shader: 'gpu-spatial-analysis/spatial-regionalization/gpu-ward-regions.ts',
    expression: 'workgroupUniformLoad(&uniformValue)',
    prefix: 'var<workgroup> uniformValue: u32;',
    bytes: 4
  },
  {
    shader: 'gpu-spatial-analysis/polygon-triangulation/constrained-delaunay-parallel.ts',
    expression: 'arrayLength(&offsets) - offsetsOffset - 1u',
    prefix: '@group(0) @binding(0) var<storage, read> offsets: array<u32>;',
    parameters: 'offsetsOffset: u32',
    bytes: 4
  },
  {
    shader: 'gpu-spatial-analysis/grid-generators/gpu-shape-generator.ts',
    expression: 'length(vec2<f32>(rotatedEast, rotatedNorth))',
    parameters: 'rotatedEast: f32, rotatedNorth: f32',
    bytes: 4
  },
  {
    shader: 'gpu-spatial-analysis/change-points/gpu-pettitt-test.ts',
    expression: 'abs(running)',
    parameters: 'running: f32',
    bytes: 4
  },
  {
    shader: 'gpu-spatial-analysis/geographic-distribution/gpu-rose-statistic.ts',
    expression: 'f32(sharedCount[0])',
    prefix: 'var<workgroup> sharedCount: array<u32, 1>;',
    bytes: 4
  },
  {
    shader: 'gpu-spatial-analysis/geographic-distribution/gpu-rose-statistic.ts',
    expression: 'vec2f(starts[startsOffset + row * 2u], starts[startsOffset + row * 2u + 1u])',
    prefix: '@group(0) @binding(0) var<storage, read> starts: array<f32>;',
    parameters: 'startsOffset: u32, row: u32',
    bytes: 8
  },
  {
    shader: 'gpu-spatial-analysis/scan-statistics/gpu-spatial-scan-statistic.ts',
    expression: 'interval.first',
    prefix: 'struct ScanDrawInterval {first: u32, stop: u32}',
    parameters: 'interval: ScanDrawInterval',
    bytes: 4
  },
  {
    shader: 'gpu-spatial-analysis/geometry-topology/gpu-make-valid.ts',
    expression: 'keep[keepOffset + index] != 0u',
    prefix: '@group(0) @binding(0) var<storage, read> keep: array<u32>;',
    parameters: 'keepOffset: u32, index: u32',
    bytes: 4
  },
  {
    shader: 'gpu-spatial-analysis/spatial-interpolation/gpu-conditional-gaussian-simulation.ts',
    expression:
      'preparationState[preparationStateOffset + 6u] != 1u || preparationState[preparationStateOffset] != preparationControl[preparationControlOffset]',
    prefix:
      '@group(0) @binding(0) var<storage, read> preparationState: array<u32>; @group(0) @binding(1) var<storage, read> preparationControl: array<u32>;',
    parameters: 'preparationStateOffset: u32, preparationControlOffset: u32',
    bytes: 4
  },
  {
    shader: 'pointcloud-analysis/operations/pointcloud-icp-reduction.ts',
    expression: 'mat3x3f(vec3f(0.0),vec3f(0.0),vec3f(0.0))',
    bytes: 48
  }
] satisfies {
  shader: string;
  expression: string;
  fragment?: string;
  prefix?: string;
  parameters?: string;
  body?: string;
  bytes: number;
}[];

it.each(CASES)('resolves $expression from $shader', fixture => {
  const production = readFileSync(new URL(`../../src/${fixture.shader}`, import.meta.url), 'utf8');
  expect(production).toContain('fragment' in fixture ? fixture.fragment : fixture.expression);
  const source = `${'prefix' in fixture ? fixture.prefix : ''}
    fn probe(${'parameters' in fixture ? fixture.parameters : ''}) {
      ${'body' in fixture ? fixture.body : ''}
      let value = ${fixture.expression}; var copied = value;
    }`;
  const usage = getWGSLModuleUsage(source);
  expect(usage.unresolvedTypes).toEqual([]);
  expect(usage.unresolvedLengths).toEqual([]);
  expect(usage.functionBytes).toBe(fixture.bytes);
  expect(usage.localArrayBytes).toBe(0);
  expect(auditWGSLModule(source)).toEqual([]);
});

it('follows a declared user-function return and counts aggregate copies, including nested array elements', () => {
  // DenseCholesky's declared return struct contains a scale array in dense-solve-wgsl.ts.
  const source = `struct Factor {scale: array<f32, 4096>, size: u32}
    fn factor() -> Factor { return Factor(); }
    @group(0) @binding(0) var<storage, read> factors: array<Factor>;
    fn probe() { let input = factors[0]; var copied = input; }
    fn other() { let input = factor(); var copied = input; }`;
  const usage = getWGSLModuleUsage(source);
  expect(usage.functionBytes).toBe(32776);
  expect(usage.localArrayBytes).toBe(65552);
  expect(usage.unresolvedTypes).toEqual([]);
  expect(getPrivateMemoryBytes(source)).toBe(65552);
  expect(auditWGSLModule(source).map(finding => finding.check)).toContain('function bytes');
});

it('counts inferred array constructors and refuses genuinely unknown copied layouts', () => {
  expect(
    getWGSLModuleUsage('fn probe() { let values = array(1u, 2u); var copied = values; }')
      .localArrayBytes
  ).toBe(16);
  const unknown = 'fn probe() { let value = missingType(); var copied = value; }';
  expect(() => getPrivateMemoryBytes(unknown)).toThrow(/unresolved/);
  expect(auditWGSLModule(unknown).map(finding => finding.check)).toContain(
    'unresolved memory layout'
  );
});

it('counts a large struct copy even when it contains no array', () => {
  const source = `struct Large { @size(16384) value: u32 }
    @group(0) @binding(0) var<storage, read> data: array<Large>;
    fn probe() { let value = data[0]; var copied = value; }`;
  expect(getWGSLModuleUsage(source).functionBytes).toBe(32768);
  expect(getPrivateMemoryBytes(source)).toBe(32768);
  expect(auditWGSLModule(source).map(finding => finding.check)).toContain('function bytes');
});

it('resolves the declared return of the production DenseCholesky function', () => {
  const shader = getDenseCholeskyWGSL(4);
  const original = getWGSLModuleUsage(shader);
  const copied = getWGSLModuleUsage(`${shader}
    fn probe(matrix: ptr<function, array<f32, 16>>) {
      var copied = factorDenseCholesky_4(matrix, 4u, false, 0.0);
    }`);
  expect(copied.unresolvedTypes).toEqual([]);
  // Four scale values followed by three scalar members: 4 * 4 + 3 * 4 = 28 bytes.
  expect(copied.totalFunctionBytes - original.totalFunctionBytes).toBe(28);
  expect(copied.localArrayBytes - original.localArrayBytes).toBe(28);
});

it('counts an immutable array copy as declared function storage', () => {
  const source = `@group(0) @binding(0) var<storage, read> data: array<u32, 4096>;
    fn probe() { let copied = data; }`;
  expect(getWGSLModuleUsage(source).functionBytes).toBe(16384);
  expect(getPrivateMemoryBytes(source)).toBe(16384);
  expect(auditWGSLModule(source).map(finding => finding.check)).toContain('function bytes');
});

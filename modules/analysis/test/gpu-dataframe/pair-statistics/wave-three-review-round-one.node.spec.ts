// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readFileSync} from 'node:fs';
import ts from 'typescript';
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, expectTypeOf, it} from 'vitest';
import {WgslExec, WgslParser} from 'wgsl_reflect';
import {QUADRAT_INDEX_WGSL} from '../../../src/dataframe-analysis/pair-statistics/quadrat-index-wgsl';
import {
  getGPUPointPatternIndicesParameterValues,
  type GPUPointPatternIndicesSettings
} from '../../../src/dataframe-analysis/pair-statistics/point-pattern-indices-parameters';
import {getGPUVariogramParameterValues} from '../../../src/dataframe-analysis/pair-statistics/variogram-parameters';
import {getGPUSpaceTimeKFunctionParameterValues} from '../../../src/dataframe-analysis/pair-statistics/space-time-k-function-nodes';
import {GPUPointPatternIndices} from '../../../src/dataframe-analysis/pair-statistics/gpu-point-pattern-indices';
import {GPUVariogram} from '../../../src/dataframe-analysis/pair-statistics/gpu-variogram';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';

it('B1 compares product words in order without mixing logical operator groups', () => {
  const comparator = /fn isQuadratProductBelow\([^]*?\n\}/.exec(QUADRAT_INDEX_WGSL)![0];
  // WGSL does not permit unparenthesized mixtures of && and ||. Sequential word
  // comparisons also make zero handling and equality explicit.
  expect(comparator).not.toMatch(/&&|\|\|/);
  const result = new Uint32Array(1);
  new WgslExec(
    new WgslParser().parse(`${QUADRAT_INDEX_WGSL}
@group(0) @binding(0) var<storage,read_write> result:array<u32>;
@compute @workgroup_size(1) fn main(){result[0]=getQuadratIndex(5.0,8.0,12u);}`)
  ).dispatchWorkgroups('main', 1, {0: {0: result}});
  // floor((5/8)*12) = floor(7.5) = 7, independently of the helper.
  expect([...result]).toEqual([7]);
});

it('B3 validates canonical settings and rejects removed settings', () => {
  const bounds = [0, 0, 1, 1] as const;
  for (const bad of [1e-50, 1e39]) {
    expect(() =>
      getGPUPointPatternIndicesParameterValues({
        bounds,
        ...{maximumDistance: bad},
        indexCellSizeHint: 1
      })
    ).toThrow(/^GPUPointPatternIndices maximumDistance was removed.*indexCellSizeHint/);
    expect(() =>
      getGPUPointPatternIndicesParameterValues({bounds, indexCellSizeHint: bad})
    ).toThrow(/^GPUPointPatternIndices indexCellSizeHint .*float32/);
  }
  expect(() =>
    getGPUVariogramParameterValues({bounds, maximumDistance: 1, ...{azimuthOffset: 1e39}})
  ).toThrow(/^GPUVariogram azimuthOffset was removed.*sectorOffsetRadians/);
  expect(() =>
    getGPUVariogramParameterValues({bounds, maximumDistance: 1, sectorOffsetRadians: 1e39})
  ).toThrow(/^GPUVariogram sectorOffsetRadians .*float32/);
});

it('B4 retains the required numeric index cell size and an extendable settings object type', () => {
  // The AST assertion runs without a compiler project. The reader and extension below
  // are also checked by the lead's test typecheck.
  const source = readFileSync(
    new URL(
      '../../../src/dataframe-analysis/pair-statistics/point-pattern-indices-parameters.ts',
      import.meta.url
    ),
    'utf8'
  );
  const parsed = ts.createSourceFile('settings.ts', source, ts.ScriptTarget.Latest, true);
  const declaration = parsed.statements.find(
    statement =>
      ts.isTypeAliasDeclaration(statement) &&
      statement.name.text === 'GPUPointPatternIndicesSettings'
  );
  if (!declaration || !ts.isTypeAliasDeclaration(declaration))
    throw new Error('Missing settings type');
  expect(ts.isTypeLiteralNode(declaration.type)).toBe(true);
  if (!ts.isTypeLiteralNode(declaration.type)) return;
  const property = declaration.type.members.find(
    member => ts.isPropertySignature(member) && member.name.getText(parsed) === 'indexCellSizeHint'
  );
  if (!property || !ts.isPropertySignature(property)) throw new Error('Missing index cell size');
  expect(property.questionToken).toBeUndefined();
  expect(property.type?.kind).toBe(ts.SyntaxKind.NumberKeyword);
  function readIndexCellSizeHint(settings: GPUPointPatternIndicesSettings): number {
    return settings.indexCellSizeHint;
  }
  expectTypeOf<GPUPointPatternIndicesSettings['indexCellSizeHint']>().toEqualTypeOf<number>();
  interface ExtendedSettings extends GPUPointPatternIndicesSettings {
    description: string;
  }
  const extended: ExtendedSettings = {
    bounds: [0, 0, 1, 1],
    indexCellSizeHint: 2,
    description: 'canonical'
  };
  expect(readIndexCellSizeHint(extended)).toBe(2);
});

const SPACE_TIME_SETTINGS = {
  bounds: [0, 0, 1, 1] as const,
  timeWindow: [0, 1] as const,
  maximumDistance: 1,
  maximumTimeLag: 1
};
it('S1 rejects positive subnormal area even when volume would be normal', () => {
  // Exact powers: width*height=2^-140 is positive subnormal, while area*duration=2^-90.
  expect(Math.fround(2 ** -70 * 2 ** -70)).toBe(2 ** -140);
  expect(() =>
    getGPUSpaceTimeKFunctionParameterValues({
      ...SPACE_TIME_SETTINGS,
      bounds: [0, 0, 2 ** -63, 2 ** -63]
    })
  ).not.toThrow();
  expect(() =>
    getGPUSpaceTimeKFunctionParameterValues({
      ...SPACE_TIME_SETTINGS,
      bounds: [0, 0, 2 ** -70, 2 ** -70],
      timeWindow: [0, 2 ** 50]
    })
  ).toThrow(/area .*normal/);
});
it('S1 rejects positive subnormal volume after normal area and duration', () => {
  expect(Math.fround(2 ** -100 * 2 ** -30)).toBe(2 ** -130);
  expect(() =>
    getGPUSpaceTimeKFunctionParameterValues({
      ...SPACE_TIME_SETTINGS,
      bounds: [0, 0, 2 ** -50, 2 ** -50],
      timeWindow: [0, 2 ** -30]
    })
  ).toThrow(/volume .*normal/);
});

it('S2 retains unit-frame points when a finite cell-size hint has an overflowing square', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const nodes = new GPUPointPatternIndices({
      id: 'hint',
      positions: createTransientView(graph, 'points', 'float32x2', 2),
      parameters: createTransientView(graph, 'parameters', 'float32', 8),
      gridSize: [2, 2],
      output: {nearestNeighborDistances: createTransientView(graph, 'distances', 'float32', 2)}
    }).getCommandNodes(graph);
    const kernels = getGPUCommandNodeKernelSources(nodes);
    for (const id of ['hint-cell-keys', 'hint-nearest']) {
      const source = kernels.find(entry => entry.id === id)!.source;
      const arithmetic = /let radiusSquared =[^;]+;\s*let cellRadius =[^;]+;/.exec(source)![0];
      const finite = /fn isFiniteValue\([^]*?\}/.exec(source)![0];
      const result = new Uint32Array(2);
      // Probe the production numeric prerequisites directly: wgsl_reflect cannot read the
      // bool member of a returned Lattice struct. The retained GPU case covers full wiring.
      const probe = `${finite}
@group(0) @binding(0) var<storage,read> parameters:array<f32>;
@group(0) @binding(1) var<storage,read_write> result:array<u32>;
const CELL_MARGIN:f32=1.0009765625;
@compute @workgroup_size(1) fn main() {
  let radius = parameters[0];
  ${arithmetic}
  result[0] = select(0u,1u,isFiniteValue(radius * radius));
  result[1] = select(0u,1u,isFiniteValue(radiusSquared) && isFiniteValue(cellRadius));
}`;
      new WgslExec(new WgslParser().parse(probe)).dispatchWorkgroups('main', 1, {
        0: {0: Float32Array.of(2 ** 66), 1: result}
      });
      // The old square is 2^132 (not finite f32); a finite performance hint needs no square.
      expect([...result], id).toEqual([0, 1]);
    }
  } finally {
    device.destroy();
  }
});

it('S3 actual overflow publisher clears a previous one on a clean encoding', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const nodes = new GPUVariogram({
      id: 'clear',
      positions: createTransientView(graph, 'points', 'float32x2', 2),
      values: createTransientView(graph, 'values', 'float32', 2),
      parameters: createTransientView(graph, 'parameters', 'float32', 8),
      gridSize: [1, 1],
      lagCount: 1,
      output: {
        semivariances: createTransientView(graph, 'result', 'float32', 1),
        status: {overflow: createTransientView(graph, 'overflow', 'uint32', 1)}
      }
    }).getCommandNodes(graph);
    const kernel = getGPUCommandNodeKernelSources(nodes).find(
      entry => entry.id === 'clear-overflow'
    )!;
    const overflow = Uint32Array.of(1);
    new WgslExec(
      new WgslParser().parse(kernel.source.replace(/^\s*_ =[^;]+;/gm, ''))
    ).dispatchWorkgroups(
      'main',
      1,
      {0: {0: new Uint32Array(9), 1: Float32Array.of(0), 2: overflow}},
      {constants: kernel.constants}
    );
    expect([...overflow]).toEqual([0]);
  } finally {
    device.destroy();
  }
});

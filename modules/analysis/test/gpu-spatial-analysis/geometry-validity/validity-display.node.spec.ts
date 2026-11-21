// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readFileSync} from 'node:fs';
import {runInNewContext} from 'node:vm';
import {transpileModule, ModuleKind} from 'typescript';
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {createWGSLKernelNode} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';
import {GPU_GEOMETRY_VALIDITY_STRUCTURAL_MASK} from '../../../src/gpu-spatial-analysis/geometry-validity/geometry-validity-types';
import {GEOMETRY_VALIDITY_CLASSES_WGSL} from '../../../../../website/src/gpu/runs/entries/geometry-processing/shared/geometry-validity-classes';

it('website classes keep an unchecked zero code incomplete and retain observed defects', () => {
  const body = GEOMETRY_VALIDITY_CLASSES_WGSL;
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const mask = createTransientView(graph, 'mask', 'uint32', 5);
  const validity = createTransientView(graph, 'validity', 'uint32', 5);
  const classes = createTransientView(graph, 'classes', 'uint32', 5);
  const node = createWGSLKernelNode(graph, {
    id: 'display',
    operation: 'test',
    invocationCount: 5,
    bindings: [
      {name: 'mask', view: mask, type: 'u32', access: 'read'},
      {name: 'validity', view: validity, type: 'u32', access: 'read'},
      {name: 'classes', view: classes, type: 'u32', access: 'read_write'}
    ],
    body
  });
  const memory = createCPUGraphMemory();
  memory.write(mask, Uint32Array.of(0, 0, 16, 128, 256));
  memory.write(validity, Uint32Array.of(0, 1, 0, 1, 0));
  executeCommandNodesOnCPU([node], memory);
  expect(Array.from(memory.readUint32(classes))).toEqual([5, 0, 1, 3, 6]);
});

it('website hover preserves the uncertain diagnostic and its class label', () => {
  const source = readFileSync(
    new URL(
      '../../../../../website/src/gpu/runs/entries/geometry-processing/geometry-validity.ts',
      import.meta.url
    ),
    'utf8'
  );
  const exported: Record<string, unknown> = {};
  const helpers = {
    defineOperationRun: (run: unknown) => run,
    chooseSticky: () => 0,
    injectDefects: () => Float32Array.of(0, 0, 1, 0, 0, 1),
    getShellRows: () => ({first: 0, count: 3}),
    findRingCrossings: () => []
  };
  runInNewContext(
    transpileModule(source, {compilerOptions: {module: ModuleKind.CommonJS}}).outputText +
      '\nexports.classNames = CLASS_NAMES;',
    {exports: exported, require: () => helpers}
  );
  const mask = {};
  const validity = {};
  const run = exported['default'] as {
    interaction: {resolve(pointer: unknown, state: unknown): {readout: string}};
  };
  const result = run.interaction.resolve(
    {x: 0, y: 0},
    {
      inputs: {index: {}, focus: {feature: -1}, set: {}},
      outputs: {mask, validity},
      values: {defects: 0},
      peek: (output: unknown) => Uint32Array.of(output === mask ? 256 : 0)
    }
  );
  expect(result.readout).toContain('uncertain');
  expect((exported['classNames'] as readonly string[])[6]).toBe('uncertain predicate');
});

it('showcase categories distinguish unchecked rows from usable and observed structural rows', () => {
  const source = readFileSync(
    new URL(
      '../../../../../examples/deck/spatial-analysis-showcase/src/scenes/geometry/polygon-measures.compute.ts',
      import.meta.url
    ),
    'utf8'
  );
  const body = source
    .slice(source.indexOf("id: 'validity-category'"))
    .match(/body: \/\* wgsl \*\/ `([\s\S]*?)`/u)![1];
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const mask = createTransientView(graph, 'mask', 'uint32', 4);
  const validity = createTransientView(graph, 'validity', 'uint32', 4);
  const category = createTransientView(graph, 'category', 'float32', 4);
  const node = createWGSLKernelNode(graph, {
    id: 'display',
    operation: 'test',
    invocationCount: 4,
    bindings: [
      {name: 'mask', view: mask, type: 'u32', access: 'read'},
      {name: 'validity', view: validity, type: 'u32', access: 'read'},
      {name: 'category', view: category, type: 'f32', access: 'read_write'}
    ],
    declarations: `const STRUCTURAL: u32 = ${GPU_GEOMETRY_VALIDITY_STRUCTURAL_MASK}u; const REPEATED: u32 = 8u;`,
    body
  });
  const memory = createCPUGraphMemory();
  memory.write(mask, Uint32Array.of(0, 0, 16, 128));
  memory.write(validity, Uint32Array.of(0, 1, 0, 1));
  executeCommandNodesOnCPU([node], memory);
  expect(Array.from(memory.readFloat32(category))).toEqual([3, 0, 2, 1]);
});

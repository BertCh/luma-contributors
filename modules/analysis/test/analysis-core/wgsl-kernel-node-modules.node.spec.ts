// luma.gl contributors
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// createWGSLKernelNode without shader modules generates exactly the WGSL it generated before the
// `modules` prop existed (the literal below was captured from the factory before the change), and
// with modules it assembles them through the published WGSLShaderAssembler, always on the
// integer-controlled fp64 variant, and rejects what it cannot bind.

import type {Device} from '@luma.gl/core';
import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import {fp64, fp64arithmetic, type ShaderModule} from '@luma.gl/shaders';
import {describe, expect, test} from 'vitest';
import {
  createWGSLKernelNode,
  getWGSLKernelNodeRecord,
  recordWGSLKernelNodes,
  type WGSLKernelProps
} from '../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {createNullWebGPUDevice, getGPUCommandNodeKernelSources} from '../../src/testing/index';

/** Generated WGSL of the representative kernel, captured before `modules` was added. */
const REPRESENTATIVE_SOURCE =
  '\noverride INVOCATION_COUNT: u32;\noverride WGSL_KERNEL_DISPATCH_X: u32;\noverride WGSL_KERNEL_DISPATCH_Y: u32;\nconst valuesOffset: u32 = 0u;\n@group(0) @binding(0) var<storage, read> values: array<f32>;\nconst outputOffset: u32 = 0u;\n@group(0) @binding(1) var<storage, read_write> output: array<u32>;\n@group(0) @binding(2) var lookup: texture_2d<f32>;\noverride SCALE: f32;\nfn twice(value: f32) -> f32 { return value * 2.0; }\n\n@compute @workgroup_size(256)\nfn main(\n  @builtin(local_invocation_index) localInvocationIndex: u32,\n  @builtin(workgroup_id) workgroupId: vec3<u32>\n) {\n  _ = &values;\n  _ = &output;\n  _ = lookup;\n  _ = INVOCATION_COUNT;\n  let workgroupIndex =\n    (workgroupId.z * WGSL_KERNEL_DISPATCH_Y + workgroupId.y) * WGSL_KERNEL_DISPATCH_X +\n    workgroupId.x;\n  if (workgroupIndex >= 16777216u) { return; }\n  let index = workgroupIndex * 256u + localInvocationIndex;\n  if (index >= INVOCATION_COUNT) {\n    return;\n  }\n  output[outputOffset + index] = u32(twice(values[valuesOffset + index]) * SCALE + textureLoad(lookup, vec2<u32>(0u, 0u), 0).x);\n}';

/** Shader layout of the representative kernel, captured with the source. */
const REPRESENTATIVE_SHADER_LAYOUT = {
  bindings: [
    {name: 'values', type: 'read-only-storage', group: 0, location: 0},
    {name: 'output', type: 'storage', group: 0, location: 1},
    {
      name: 'lookup',
      type: 'texture',
      group: 0,
      location: 2,
      viewDimension: '2d',
      sampleType: 'unfilterable-float'
    }
  ]
};

/** Null WebGPU device that reports a WGSL platform on a GPU the assembler does not treat as Apple. */
function createNonAppleDevice(): Device {
  const device = createNullWebGPUDevice();
  Object.defineProperty(device, 'info', {
    value: {...device.info, gpu: 'nvidia', shadingLanguage: 'wgsl'}
  });
  return device;
}

function createRepresentativeFixture(device: Device = createNullWebGPUDevice()) {
  const graph = new GPUCommandGraph(device, {id: 'representative-kernel'});
  recordWGSLKernelNodes(graph);
  const values = createTransientView(graph, 'values', 'float32', 1000);
  const output = createTransientView(graph, 'output', 'uint32', 1000);
  const texture = graph.createTransientTexture({
    id: 'lookup-texture',
    format: 'r32float',
    width: 4,
    height: 4,
    usage: 0x4 /* TEXTURE_BINDING */
  } as never);
  const props: WGSLKernelProps<unknown> = {
    id: 'representative',
    operation: 'WGSLKernelTest',
    variant: 'pinned',
    bindings: [
      {name: 'values', view: values, type: 'f32', access: 'read'},
      {name: 'output', view: output, type: 'u32', access: 'read_write'}
    ],
    textures: [
      {
        name: 'lookup',
        texture: graph.createTextureView(texture),
        sampleType: 'unfilterable-float'
      }
    ],
    invocationCount: 1000,
    declarations: 'override SCALE: f32;\nfn twice(value: f32) -> f32 { return value * 2.0; }',
    constants: {SCALE: 3},
    body: 'output[outputOffset + index] = u32(twice(values[valuesOffset + index]) * SCALE + textureLoad(lookup, vec2<u32>(0u, 0u), 0).x);'
  };
  return {device, graph, values, output, props};
}

describe('createWGSLKernelNode without modules', () => {
  for (const [label, modules] of [
    ['absent', undefined],
    ['empty', []]
  ] as const) {
    test(`generates the pinned WGSL, layout, constants and record (modules ${label})`, () => {
      const {device, graph, props} = createRepresentativeFixture();
      const node = createWGSLKernelNode(graph, modules ? {...props, modules} : props);
      const [compiled] = getGPUCommandNodeKernelSources([node]);
      expect(compiled!.source).toBe(REPRESENTATIVE_SOURCE);
      expect(compiled!.shaderLayout).toEqual(REPRESENTATIVE_SHADER_LAYOUT);
      expect(compiled!.constants).toEqual({
        SCALE: 3,
        INVOCATION_COUNT: 1000,
        WGSL_KERNEL_DISPATCH_X: 4,
        WGSL_KERNEL_DISPATCH_Y: 1
      });
      expect(node.workload).toEqual({
        operation: 'WGSLKernelTest',
        variant: 'pinned',
        commandCount: 1,
        maximumWorkgroupCount: 4,
        maximumInvocationCount: 1000,
        readByteLength: 4000,
        writeByteLength: 4000
      });
      const record = getWGSLKernelNodeRecord(graph, 'representative');
      expect(record!.source).toBe(REPRESENTATIVE_SOURCE);
      expect(record!.workgroups).toEqual([4, 1, 1]);
      expect(record).toEqual({
        id: 'representative',
        operation: 'WGSLKernelTest',
        variant: 'pinned',
        source: REPRESENTATIVE_SOURCE,
        body: props.body,
        declarations: props.declarations,
        entryPoint: 'main',
        workgroupSize: [256, 1, 1],
        workgroups: [4, 1, 1],
        invocationCount: 1000,
        constants: {
          SCALE: 3,
          INVOCATION_COUNT: 1000,
          WGSL_KERNEL_DISPATCH_X: 4,
          WGSL_KERNEL_DISPATCH_Y: 1
        },
        bindings: props.bindings,
        textures: props.textures,
        indirect: false
      });
      device.destroy();
    });
  }

  test('rejects defines without modules', () => {
    const {device, graph, props} = createRepresentativeFixture();
    expect(() =>
      createWGSLKernelNode(graph, {...props, defines: {LUMA_FP64_INTEGER_ARITHMETIC: true}})
    ).toThrow(/^representative defines need modules/);
    device.destroy();
  });
});

/** One fp64 kernel over a float32 pair input and a float32 pair output. */
function createFP64Props(
  graph: GPUCommandGraph<unknown>,
  modules: NonNullable<WGSLKernelProps<unknown>['modules']>
): WGSLKernelProps<unknown> {
  const input = createTransientView(graph, 'fp64-input', 'float32', 4);
  const output = createTransientView(graph, 'fp64-output', 'float32', 2);
  return {
    id: 'fp64-kernel',
    operation: 'WGSLKernelTest',
    bindings: [
      {name: 'input', view: input, type: 'f32', access: 'read'},
      {name: 'result', view: output, type: 'f32', access: 'read_write'}
    ],
    invocationCount: 1,
    modules,
    body: `let sum = sum_fp64(vec2f(input[inputOffset], input[inputOffset + 1u]),
    vec2f(input[inputOffset + 2u], input[inputOffset + 3u]));
  result[resultOffset] = sum.x;
  result[resultOffset + 1u] = sum.y;`
  };
}

describe('createWGSLKernelNode with modules', () => {
  test('assembles fp64 on the integer variant on a non-Apple GPU and records the assembled text', () => {
    const device = createNonAppleDevice();
    const graph = new GPUCommandGraph(device, {id: 'fp64-record'});
    recordWGSLKernelNodes(graph);
    const props = createFP64Props(graph, [fp64]);
    const node = createWGSLKernelNode(graph, props);
    const [compiled] = getGPUCommandNodeKernelSources([node]);
    const record = getWGSLKernelNodeRecord(graph, 'fp64-kernel');

    expect(record!.source, 'the record holds what runs').toBe(compiled!.source);
    expect(record).toEqual({
      id: 'fp64-kernel',
      operation: 'WGSLKernelTest',
      source: compiled!.source,
      body: props.body,
      entryPoint: 'main',
      workgroupSize: [32, 1, 1],
      workgroups: [1, 1, 1],
      invocationCount: 1,
      constants: {
        INVOCATION_COUNT: 1,
        WGSL_KERNEL_DISPATCH_X: 1,
        WGSL_KERNEL_DISPATCH_Y: 1
      },
      bindings: props.bindings,
      textures: [],
      indirect: false
    });
    expect(compiled!.source).toContain('fn fp64_two_sum_integer_bits(');
    expect(compiled!.source).toContain('fn sin_fp64(');
    // The classic split reads the uniform; the integer variant does not.
    expect(compiled!.source, 'no classic variant').not.toContain('fp64arithmetic.SPLIT');
    expect(compiled!.source).not.toMatch(/^\s*#(?:if|ifdef|ifndef|else|endif)/m);
    // Application bindings keep their locations; the module uniform goes to 100 and is not in the
    // pipeline's shader layout, because no function the kernel calls reads it.
    expect(compiled!.source).toContain(
      '@group(0) @binding(0) var<storage, read> input: array<f32>;'
    );
    expect(compiled!.source).toContain(
      '@group(0) @binding(1) var<storage, read_write> result: array<f32>;'
    );
    expect(compiled!.source).toMatch(/@group\(0\) @binding\(100\) var<uniform> fp64arithmetic/);
    expect(compiled!.shaderLayout).toEqual({
      bindings: [
        {name: 'input', type: 'read-only-storage', group: 0, location: 0},
        {name: 'result', type: 'storage', group: 0, location: 1}
      ]
    });
    // The generated kernel follows the modules unchanged.
    const generated = compiled!.source.slice(compiled!.source.indexOf('override INVOCATION_COUNT'));
    expect(generated).toContain(props.body);
    expect(generated).toContain('if (index >= INVOCATION_COUNT) {');
    device.destroy();
  });

  test('fp64arithmetic alone also gets the integer variant', () => {
    const device = createNonAppleDevice();
    const graph = new GPUCommandGraph(device, {id: 'fp64-arithmetic-only'});
    const node = createWGSLKernelNode(graph, createFP64Props(graph, [fp64arithmetic]));
    const [compiled] = getGPUCommandNodeKernelSources([node]);
    expect(compiled!.source).toContain('fn fp64_two_sum_integer_bits(');
    expect(compiled!.source).not.toContain('fn sin_fp64(');
    device.destroy();
  });

  test('rejects defines that turn the integer variant off', () => {
    const device = createNonAppleDevice();
    const graph = new GPUCommandGraph(device, {id: 'fp64-classic'});
    const props = createFP64Props(graph, [fp64]);
    for (const value of [false, 0]) {
      expect(() =>
        createWGSLKernelNode(graph, {...props, defines: {LUMA_FP64_INTEGER_ARITHMETIC: value}})
      ).toThrow(/^fp64-kernel defines must leave LUMA_FP64_INTEGER_ARITHMETIC on/);
    }
    expect(() =>
      createWGSLKernelNode(graph, {...props, defines: {LUMA_FP64_INTEGER_ARITHMETIC: true}})
    ).not.toThrow();
    device.destroy();
  });

  test('rejects a body, declarations or module that reads the fp64arithmetic uniform', () => {
    const device = createNonAppleDevice();
    const graph = new GPUCommandGraph(device, {id: 'fp64-uniform-readers'});
    const props = createFP64Props(graph, [fp64]);
    const message = /^fp64-kernel body, declarations and modules must not read/;
    expect(() =>
      createWGSLKernelNode(graph, {...props, body: `${props.body}\n  _ = fp64arithmetic.ONE;`})
    ).toThrow(message);
    expect(() =>
      createWGSLKernelNode(graph, {
        ...props,
        declarations: 'fn zero() -> f32 { return fp64_runtime_zero(); }'
      })
    ).toThrow(message);
    const readingModule: ShaderModule = {
      name: 'fp64-reader',
      source: 'fn guarded(value: f32) -> f32 { return prevent_fp64_optimization(value); }',
      dependencies: [fp64arithmetic]
    };
    expect(() => createWGSLKernelNode(graph, {...props, modules: [readingModule]})).toThrow(
      message
    );
    device.destroy();
  });

  for (const [label, statements] of [
    ['whole-struct read', 'let uniforms = fp64arithmetic; _ = uniforms.ONE;'],
    ['comment before member access', '_ = fp64arithmetic /* comment */ .ONE;'],
    ['address reference', '_ = &fp64arithmetic;'],
    ['runtime zero bits', '_ = fp64_runtime_zero_bits();']
  ]) {
    for (const location of ['body', 'declarations', 'module']) {
      test(`rejects ${label} in ${location}`, () => {
        const device = createNonAppleDevice();
        const graph = new GPUCommandGraph(device, {id: 'fp64-uniform-reference'});
        const props = createFP64Props(graph, [fp64]);
        const helper = `fn readUniform() { ${statements} }`;
        const readingModule: ShaderModule = {
          name: 'fp64-reader',
          source: helper,
          dependencies: [fp64]
        };
        const readingProps = {
          ...props,
          ...(location === 'body'
            ? {body: `${props.body}\n${statements}`}
            : {body: `${props.body}\nreadUniform();`}),
          ...(location === 'declarations' ? {declarations: helper} : {}),
          ...(location === 'module' ? {modules: [readingModule]} : {})
        };
        expect(() => createWGSLKernelNode(graph, readingProps)).toThrow(
          /^fp64-kernel body, declarations and modules must not read.*does not bind it/
        );
        device.destroy();
      });
    }
  }

  test('accepts comments mentioning the uniform and readers, and longer identifiers', () => {
    const device = createNonAppleDevice();
    const graph = new GPUCommandGraph(device, {id: 'fp64-uniform-comments'});
    const props = createFP64Props(graph, [fp64]);
    const comments = `// fp64arithmetic.ONE fp64_runtime_zero_bits()\n
/* fp64arithmetic /* nested */ prevent_fp64_optimization(value) fp64_runtime_zero() */`;
    const commentingModule: ShaderModule = {
      name: 'fp64-comment',
      source: comments,
      dependencies: [fp64]
    };
    expect(() =>
      createWGSLKernelNode(graph, {
        ...props,
        body: `${props.body}\n${comments}\nlet fp64arithmeticCopy = 1.0; _ = fp64arithmeticCopy;`,
        declarations: `${comments}\nconst my_fp64arithmetic = 1.0;`,
        modules: [commentingModule]
      })
    ).not.toThrow();
    device.destroy();
  });

  test('checks caller module source even when it has the published fp64arithmetic name', () => {
    const device = createNonAppleDevice();
    const graph = new GPUCommandGraph(device, {id: 'fp64-caller-module'});
    const readingModule = {
      ...fp64arithmetic,
      source: `${fp64arithmetic.source}\nfn readUniform() { _ = &fp64arithmetic; }`
    };
    expect(() => createWGSLKernelNode(graph, createFP64Props(graph, [readingModule]))).toThrow(
      /^fp64-kernel body, declarations and modules must not read.*does not bind it/
    );
    device.destroy();
  });

  test('rejects a module binding the node would have to bind', () => {
    const device = createNonAppleDevice();
    const graph = new GPUCommandGraph(device, {id: 'module-uniform'});
    const uniformModule: ShaderModule = {
      name: 'scaleModule',
      source: `struct ScaleUniforms { factor: f32, };
@group(0) @binding(auto) var<uniform> scaleModule: ScaleUniforms;
fn scaled(value: f32) -> f32 { return value * scaleModule.factor; }`,
      uniformTypes: {factor: 'f32'}
    };
    expect(() =>
      createWGSLKernelNode(graph, {...createFP64Props(graph, [uniformModule]), body: ''})
    ).toThrow(/^fp64-kernel module scaleModule binding scaleModule needs a bound resource/);
    device.destroy();
  });

  test('rejects application bindings at or above location 100', () => {
    const device = createNonAppleDevice();
    Object.defineProperty(device, 'limits', {
      value: {...device.limits, maxStorageBuffersPerShaderStage: 128}
    });
    const graph = new GPUCommandGraph(device, {id: 'module-binding-range'});
    const view = createTransientView(graph, 'shared', 'float32', 4);
    const bindings = Array.from({length: 101}, (_, location) => ({
      name: `binding${location}`,
      view,
      type: 'f32' as const,
      access: 'read' as const
    }));
    const props = {
      id: 'many-bindings',
      operation: 'WGSLKernelTest',
      invocationCount: 1,
      body: '',
      modules: [fp64]
    };
    expect(() => createWGSLKernelNode(graph, {...props, bindings})).toThrow(
      'many-bindings storage and sampled-texture bindings with shader modules must be at most 100'
    );
    expect(() =>
      createWGSLKernelNode(graph, {...props, bindings: bindings.slice(0, 100)})
    ).not.toThrow();
    device.destroy();
  });
});

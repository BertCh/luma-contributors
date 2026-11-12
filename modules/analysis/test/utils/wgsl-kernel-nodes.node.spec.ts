// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import {expect, test} from 'vitest';
import {
  createPublishNode,
  createWGSLKernelNode,
  getAutomaticWGSLKernelWorkgroupSize
} from '../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {createNullWebGPUDevice, getGPUCommandNodeKernelSources} from './gpu-contributor-test-utils';

test('createWGSLKernelNode accepts an omitted workload variant', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device, {id: 'kernel-variant'});
  const output = createTransientView(graph, 'output', 'uint32', 4);

  const node = createWGSLKernelNode(graph, {
    id: 'no-variant',
    operation: 'WGSLKernelTest',
    bindings: [{name: 'output', view: output, type: 'u32', access: 'read_write'}],
    invocationCount: 4,
    body: 'output[outputOffset + index] = index;'
  });

  expect(node.workload && 'variant' in node.workload, 'no undefined variant key').toBe(false);
  expect(() => graph.add(node)).not.toThrow();
  device.destroy();
});

test('generated guarded kernels scale their default workgroup size', () => {
  expect(getAutomaticWGSLKernelWorkgroupSize(0)).toBe(32);
  expect(getAutomaticWGSLKernelWorkgroupSize(1)).toBe(32);
  expect(getAutomaticWGSLKernelWorkgroupSize(33)).toBe(64);
  expect(getAutomaticWGSLKernelWorkgroupSize(129)).toBe(256);
  expect(getAutomaticWGSLKernelWorkgroupSize(4096)).toBe(256);
  expect(() => getAutomaticWGSLKernelWorkgroupSize(-1)).toThrow(/non-negative safe integer/);
});

function createTextureKernelFixture() {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device, {id: 'kernel-texture'});
  const output = createTransientView(graph, 'output', 'uint32', 4);
  const source = graph.createTransientTexture({
    id: 'source-texture',
    format: 'rgba8unorm',
    width: 2,
    height: 2,
    usage: 0x4 /* TEXTURE_BINDING */
  } as never);
  const sourceView = graph.createTextureView(source);
  const props = {
    id: 'textured',
    operation: 'WGSLKernelTest',
    bindings: [{name: 'output', view: output, type: 'u32' as const, access: 'read_write' as const}],
    invocationCount: 4,
    body: 'output[outputOffset + index] = index;'
  };
  return {device, graph, output, sourceView, props};
}

test('createWGSLKernelNode declares textures after storage bindings', () => {
  const {device, graph, sourceView, props} = createTextureKernelFixture();
  const plainNode = createWGSLKernelNode(graph, props);
  const texturedNode = createWGSLKernelNode(graph, {
    ...props,
    id: 'textured-2',
    textures: [{name: 'sourceTexture', texture: sourceView, sampleType: 'float'}]
  });
  const [plain] = getGPUCommandNodeKernelSources([plainNode]);
  const [textured] = getGPUCommandNodeKernelSources([texturedNode]);

  expect(plain!.source).not.toContain('texture_');
  expect(textured!.source).toContain('@group(0) @binding(1) var sourceTexture: texture_2d<f32>;');
  expect(textured!.source).toContain('_ = sourceTexture;');
  expect(textured!.source).toContain('_ = &output;');
  // Everything outside the texture additions is unchanged.
  expect(
    textured!.source
      .replace('\n@group(0) @binding(1) var sourceTexture: texture_2d<f32>;', '')
      .replace('\n  _ = sourceTexture;', '')
  ).toBe(plain!.source);
  expect(texturedNode.resources).toContainEqual({texture: sourceView, usage: 'sampled'});
  expect(plainNode.resources?.some(resource => 'texture' in resource)).toBe(false);
  device.destroy();
});

test('createWGSLKernelNode rejects more sampled textures than the device allows', () => {
  const {device, graph, sourceView, props} = createTextureKernelFixture();
  Object.defineProperty(device, 'limits', {
    value: {...device.limits, maxSampledTexturesPerShaderStage: 0}
  });
  expect(() =>
    createWGSLKernelNode(graph, {
      ...props,
      textures: [{name: 'sourceTexture', texture: sourceView, sampleType: 'float'}]
    })
  ).toThrow(/sampled textures/);
  device.destroy();
});

test('createWGSLKernelNode rejects a binding above maxStorageBufferBindingSize at declaration', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device, {id: 'kernel-binding-size'});
  // A 6000 x 6000 float32 raster is 144 MB, above the 128 MiB default binding limit.
  const cellCount = 6000 * 6000;
  const large = createTransientView(graph, 'large', 'float32', cellCount);
  const props = {
    id: 'large-raster',
    operation: 'WGSLKernelTest',
    bindings: [{name: 'values', view: large, type: 'f32' as const, access: 'read_write' as const}],
    invocationCount: cellCount,
    body: 'values[valuesOffset + index] = 0.0;'
  };
  expect(() => createWGSLKernelNode(graph, props)).toThrow(
    /kernel large-raster binding values needs 144000000 bytes, above the device limit of 134217728/
  );
  // 128 MiB exactly fits.
  const limit = createTransientView(graph, 'limit', 'float32', 134217728 / 4);
  expect(() =>
    createWGSLKernelNode(graph, {
      ...props,
      bindings: [{name: 'values', view: limit, type: 'f32', access: 'read_write'}]
    })
  ).not.toThrow();
  device.destroy();
});

test('createPublishNode checks the device storage-buffer limit, not a fixed 8', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device, {id: 'publish-binding-limit'});
  const requiredCount = createTransientView(graph, 'required-count', 'uint32', 1);
  const output = {
    ids: createTransientView(graph, 'ids', 'uint32', 4),
    status: {
      count: createTransientView(graph, 'count', 'uint32', 1),
      overflow: createTransientView(graph, 'overflow', 'uint32', 1)
    }
  };
  // totalIn, countOut and overflowOut plus six extra counts: nine storage bindings.
  const extraCounts = Array.from({length: 6}, (_, extraIndex) =>
    createTransientView(graph, `extra-${extraIndex}`, 'uint32', 1)
  );
  const props = {id: 'publish', operation: 'WGSLKernelTest', requiredCount, output, extraCounts};
  expect(() => createPublishNode(graph, props)).toThrow(
    'WGSLKernelTest: kernel publish binds 9 storage buffers, more than the device limit of 8'
  );
  Object.defineProperty(device, 'limits', {
    value: {...device.limits, maxStorageBuffersPerShaderStage: 10}
  });
  expect(() => createPublishNode(graph, props)).not.toThrow();
  device.destroy();
});

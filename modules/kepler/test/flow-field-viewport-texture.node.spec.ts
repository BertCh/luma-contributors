// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, test, vi} from 'vitest';
import {
  PipelineLayout,
  type Device,
  type Framebuffer,
  type PipelineLayoutProps
} from '@luma.gl/core';
import {NullDevice} from '@luma.gl/core/test-utils';
import {WebGPUDevice} from '@luma.gl/core/webgpu';
import {FlowFieldViewportTexture} from '../../../examples/deck/kepler-compute/src/flow-field-viewport-texture';

/** Run the published WebGPU descriptor builder, recording its native call without requiring a GPU. */
function createDescriptorDevice() {
  const device: Device = new NullDevice({});
  const descriptors: GPURenderPipelineDescriptor[] = [];
  Object.defineProperties(device, {
    type: {value: 'webgpu'},
    preferredDepthFormat: {value: 'depth24plus'},
    handle: {
      value: {
        createRenderPipeline: (descriptor: GPURenderPipelineDescriptor) => {
          descriptors.push(descriptor);
          return {};
        }
      }
    },
    pushErrorScope: {value: () => {}},
    popErrorScope: {value: async () => {}}
  });
  class RecordingPipelineLayout extends PipelineLayout {
    readonly device = device;
    readonly handle = {};
  }
  Object.defineProperty(device, 'createPipelineLayout', {
    value: vi.fn((props: PipelineLayoutProps) => new RecordingPipelineLayout(device, props))
  });
  vi.spyOn(device, 'createRenderPipeline').mockImplementation(props =>
    WebGPUDevice.prototype.createRenderPipeline.call(device as WebGPUDevice, props)
  );
  const framebuffers = vi.spyOn(device, 'createFramebuffer');
  return {device, descriptors, framebuffers};
}

describe('flow field viewport crop attachments', () => {
  test('public luma infers depth even when a caller disables depth writes', () => {
    const {device, descriptors} = createDescriptorDevice();
    const vertex = device.createShader({stage: 'vertex', source: 'void main() {}'});
    const fragment = device.createShader({stage: 'fragment', source: 'void main() {}'});
    const pipeline = device.createRenderPipeline({
      vs: vertex,
      fs: fragment,
      shaderLayout: {attributes: [], bindings: []},
      colorAttachmentFormats: ['rgba32float'],
      parameters: {depthWriteEnabled: false, depthCompare: 'always'}
    });
    expect(descriptors[0].depthStencil?.format).toBe('depth24plus');
    pipeline.destroy();
    vertex.destroy();
    fragment.destroy();
    device.destroy();
  });

  test('crop pipeline matches its color-only single-sample framebuffer', () => {
    const {device, descriptors, framebuffers} = createDescriptorDevice();
    const crop = new FlowFieldViewportTexture(device, 24, 24);
    const framebuffer: Framebuffer = framebuffers.mock.results[0].value;
    expect(framebuffer.depthStencilAttachment).toBeNull();
    expect(framebuffer.props.colorAttachments).toEqual([crop.texture]);
    expect(crop.texture.samples).toBe(1);
    expect(descriptors[0].fragment?.targets).toEqual([{format: crop.texture.format}]);
    // This checks the native descriptor produced by published luma, not just the input props.
    expect(descriptors[0].depthStencil).toBeUndefined();
    expect(descriptors[0].multisample?.count).toBe(1);
    crop.destroy();
    device.destroy();
  });
});

describe('flow field crop state and construction rollback', () => {
  test('sets the full rectangular viewport and disables inherited blending', () => {
    const device = new NullDevice({});
    const createEncoder = device.createCommandEncoder.bind(device);
    const passes: import('@luma.gl/core').RenderPass[] = [];
    vi.spyOn(device, 'createCommandEncoder').mockImplementation(props => {
      const encoder = createEncoder(props);
      const beginPass = encoder.beginRenderPass.bind(encoder);
      vi.spyOn(encoder, 'beginRenderPass').mockImplementation(parameters => {
        const pass = beginPass(parameters);
        passes.push(pass);
        return pass;
      });
      return encoder;
    });
    const framebuffers = vi.spyOn(device, 'createFramebuffer');
    const pipelines = vi.spyOn(device, 'createRenderPipeline');
    const crop = new FlowFieldViewportTexture(device, 24, 16);
    const source = device.createTexture({width: 24, height: 16, format: 'rgba32float'});
    // NullDevice does not compile or execute shaders; exercise pass setup without claiming pixels.
    Object.defineProperty(pipelines.mock.results[0].value, 'isPending', {value: false});
    crop.update(source, [0.2, 0.2, 0.8, 0.8]);
    expect(passes).toHaveLength(1);
    const framebuffer = framebuffers.mock.results[0].value;
    expect([framebuffer.width, framebuffer.height]).toEqual([24, 16]);
    expect(passes[0].getParameters().viewport).toEqual([0, 0, 24, 16]);
    expect(pipelines.mock.calls[0][0].parameters?.blend).toBe(false);
    source.destroy();
    crop.destroy();
    device.destroy();
  });

  test.each([
    1, 2, 3, 4, 5, 6, 7
  ])('allocation failure %i destroys every earlier crop resource', failureIndex => {
    const device = new NullDevice({});
    const resources: {destroyed: boolean}[] = [];
    let allocationCount = 0;
    for (const method of [
      'createTexture',
      'createFramebuffer',
      'createBuffer',
      'createShader',
      'createRenderPipeline',
      'createVertexArray'
    ] as const) {
      const allocate = device[method];
      vi.spyOn(device, method).mockImplementation(((props: unknown) => {
        allocationCount++;
        if (allocationCount === failureIndex)
          throw new Error(`Injected allocation ${failureIndex}`);
        const resource = Reflect.apply(allocate, device, [props]);
        resources.push(resource);
        return resource;
      }) as never);
    }
    expect(() => new FlowFieldViewportTexture(device, 24, 16)).toThrow(
      `Injected allocation ${failureIndex}`
    );
    expect(resources).toHaveLength(failureIndex - 1);
    expect(resources.every(resource => resource.destroyed)).toBe(true);
    device.destroy();
  });
});

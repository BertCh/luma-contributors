// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {getPlatformInfo} from '../../execution/utils/get-platform-info';
import {Kernel, type KernelProps} from '../../execution/compute/kernel';
import {
  getBoundedDispatchLayout,
  getBoundedInvocationIndexOverrideSource,
  type GPUBoundedDispatchLayout
} from '../../gpu-core/gpu-dispatch-utils';
import {WGSLShaderAssembler, type ShaderModule} from '@luma.gl/shaders';

const ARROW_SHADER_ASSEMBLER = new WGSLShaderAssembler();

/** Assemble the existing Arrow compute source without engine shader-input state. */
export function createArrowComputeKernel(
  device: Device,
  props: KernelProps & {modules?: ShaderModule[]}
): Kernel {
  const {modules, ...kernelProps} = props;
  const assembled = ARROW_SHADER_ASSEMBLER.assembleWGSLShader({
    platformInfo: getPlatformInfo(device),
    source: props.source,
    modules,
    shaderStage: 'compute',
    computeEntryPoint: props.entryPoint,
    scanVertexAttributes: false
  });
  return new Kernel(device, {
    ...kernelProps,
    source: assembled.source,
    entryPoint: assembled.entryPoints?.compute || props.entryPoint
  });
}

/** Workgroup size of the one-thread-per-element Arrow upload and conversion kernels. */
export const ARROW_COMPUTE_WORKGROUP_SIZE = 64;

/** WGSL `override` declarations and index computation for bounded 3D dispatches. */
export const ARROW_COMPUTE_INVOCATION_INDEX = getBoundedInvocationIndexOverrideSource(
  ARROW_COMPUTE_WORKGROUP_SIZE
);

/** Plans a dispatch for `elementCount` elements that respects the device workgroup limit. */
export function getArrowComputeDispatchLayout(
  device: Device,
  operationName: string,
  elementCount: number
): GPUBoundedDispatchLayout {
  return getBoundedDispatchLayout(
    operationName,
    elementCount,
    ARROW_COMPUTE_WORKGROUP_SIZE,
    device.limits.maxComputeWorkgroupsPerDimension
  );
}

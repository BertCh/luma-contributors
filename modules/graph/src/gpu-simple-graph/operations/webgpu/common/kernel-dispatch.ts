// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import type {Kernel} from '../../../../execution/compute/kernel';
import {getGPUKernelOverrideDeclarations} from '../../../../gpu-core/gpu-kernel-node';
import type {GPUEvaluationScope} from '../../../operation/gpu-evaluation-scope';
import type {WebGPUDispatchLayout} from './dispatch';

const GPGPU_OPERATION_STATS = 'GPGPU Operation Counts';
const COMPUTATION_RUNS = 'Computation Runs';

/** Integer pipeline constants collected while generating a kernel. */
export type KernelConstants = Record<string, number>;

/** Declares each constant as an unsigned `override`; see {@link getGPUKernelOverrideDeclarations}. */
export function declareKernelConstants(constants: KernelConstants): string {
  return getGPUKernelOverrideDeclarations(constants);
}

/**
 * Records one dispatch of a handler kernel.
 *
 * With a {@link GPUEvaluationScope} the pass is appended to the scope's shared command encoder and
 * the kernel is released when the scope is submitted or cancelled, so a batch of operations costs
 * one submission. Without a scope the pass is submitted immediately and the kernel destroyed,
 * which is the behavior handlers had before scopes existed.
 */
export function dispatchHandlerKernel(props: {
  device: Device;
  scope?: GPUEvaluationScope;
  kernel: Kernel;
  bindings: Record<string, Buffer>;
  layout: WebGPUDispatchLayout;
}): void {
  const {device, scope, kernel, bindings, layout} = props;
  if (scope) scope.retain(() => kernel.destroy());
  const computePass = scope
    ? scope.commandEncoder.beginComputePass({})
    : device.beginComputePass({});
  device.statsManager.getStats(GPGPU_OPERATION_STATS).get(COMPUTATION_RUNS).incrementCount();
  kernel.dispatch(computePass, {bindings, x: layout.x, y: layout.y, z: layout.z});
  computePass.end();
  if (!scope) {
    device.submit();
    kernel.destroy();
  }
}

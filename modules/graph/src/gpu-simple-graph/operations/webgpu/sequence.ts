// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Kernel} from '../../../execution/compute/kernel';
import {OperationHandler} from '../../operation/operation';
import {
  getWebGPUDispatchLayout,
  getWebGPUDispatchOverrides,
  getWebGPUDispatchRowIndexOverride
} from './common/dispatch';
import {dispatchHandlerKernel} from './common/kernel-dispatch';

const WORKGROUP_SIZE = 64;

export const sequence: OperationHandler<{start: number; step: number}> = ({
  inputs,
  output,
  target,
  scope
}) => {
  const dispatchLayout = getWebGPUDispatchLayout(
    Math.ceil(output.length / WORKGROUP_SIZE),
    target.device.limits.maxComputeWorkgroupsPerDimension
  );
  // Length, layout, and sequence parameters are pipeline constants shared by every sequence.
  const source = `\
override ROW_COUNT: u32;
override RESULT_OFFSET: u32;
override RESULT_STRIDE: u32;
override SEQUENCE_START: i32;
override SEQUENCE_STEP: i32;
override DISPATCH_X: u32;
override DISPATCH_Y: u32;
@group(0) @binding(0) var<storage, read_write> result: array<i32>;

@compute @workgroup_size(${WORKGROUP_SIZE}) fn main(
  @builtin(workgroup_id) workgroupId: vec3<u32>,
  @builtin(local_invocation_id) localId: vec3<u32>
) {
  let rowIndex = ${getWebGPUDispatchRowIndexOverride(WORKGROUP_SIZE)};
  if (rowIndex >= ROW_COUNT) {
    return;
  }

  result[RESULT_OFFSET + rowIndex * RESULT_STRIDE] = SEQUENCE_START + i32(rowIndex) * SEQUENCE_STEP;
}
`;

  const kernel = new Kernel(target.device, {
    source,
    constants: {
      ROW_COUNT: output.length,
      RESULT_OFFSET: output.offset / output.ValueType.BYTES_PER_ELEMENT,
      RESULT_STRIDE: output.stride / output.ValueType.BYTES_PER_ELEMENT,
      SEQUENCE_START: inputs.start,
      SEQUENCE_STEP: inputs.step,
      ...getWebGPUDispatchOverrides(dispatchLayout)
    },
    shaderLayout: {
      bindings: [{name: 'result', type: 'storage' as const, group: 0, location: 0}]
    }
  });

  dispatchHandlerKernel({
    device: target.device,
    scope,
    kernel,
    bindings: {result: target},
    layout: dispatchLayout
  });
  return {success: true};
};

sequence.supportsEvaluationScope = true;

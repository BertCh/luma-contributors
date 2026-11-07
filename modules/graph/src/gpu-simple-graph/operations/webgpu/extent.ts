// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {Kernel} from '../../../execution/compute/kernel';
import {OperationHandler} from '../../operation/operation';
import type {GPUEvaluationScope} from '../../operation/gpu-evaluation-scope';
import {GPUDataEvaluator} from '../../operation/gpu-data-evaluator';
import {bufferPool} from '../../utils/buffer-pool';
import {
  getWebGPUDispatchLayout,
  getWebGPUDispatchOverrides,
  getWebGPUDispatchWorkgroupIndexOverride
} from './common/dispatch';
import {getWGSLType} from './common/helper';
import {
  declareKernelConstants,
  dispatchHandlerKernel,
  type KernelConstants
} from './common/kernel-dispatch';
import {
  getInputBinding,
  getOutputBinding,
  getOutputWriter,
  getSourceValuesAccessor,
  RANDOM_ACCESS_WORKGROUP_SIZE
} from './common/random-access-transform';

type ExtentInputMode = 'raw' | 'partial';

export const extent: OperationHandler<{sourceValues: GPUDataEvaluator}> = ({
  inputs,
  output,
  target,
  scope
}) => {
  const {sourceValues} = inputs;

  if (sourceValues.length === 0) {
    const value = new output.ValueType(output.length * output.size);
    target.write(value);
    return {success: true, value};
  }

  if (sourceValues.isConstant) {
    const values = sourceValues.value;
    if (!values) {
      throw new Error(`Constant input ${sourceValues} is missing CPU values`);
    }
    const result = new output.ValueType(output.length * output.size);
    for (let channelIndex = 0; channelIndex < output.length; channelIndex++) {
      const value = values[channelIndex];
      result[channelIndex * 2] = value;
      result[channelIndex * 2 + 1] = value;
    }
    target.write(result);
    return {success: true, value: result};
  }

  const intermediateBuffers: Buffer[] = [];
  let currentInput = sourceValues;
  let inputMode: ExtentInputMode = 'raw';
  let currentGroupCount = sourceValues.length;

  try {
    while (true) {
      const nextGroupCount = Math.ceil(currentGroupCount / RANDOM_ACCESS_WORKGROUP_SIZE);
      const nextOutputLength = output.length * nextGroupCount;
      const nextOutputBuffer =
        nextGroupCount === 1
          ? target
          : bufferPool.createOrReuse(target.device, nextOutputLength * output.stride);

      if (nextGroupCount > 1) {
        intermediateBuffers.push(nextOutputBuffer);
      }

      runExtentPass({
        scope,
        input: currentInput,
        inputMode,
        inputGroupCount: currentGroupCount,
        channelCount: output.length,
        outputType: output.type,
        outputBuffer: nextOutputBuffer,
        outputLength: nextOutputLength,
        outputStride: output.stride,
        outputOffset: output.offset
      });

      if (nextGroupCount === 1) {
        break;
      }

      currentInput = new GPUDataEvaluator({
        buffer: nextOutputBuffer,
        type: output.type,
        size: 2,
        length: nextOutputLength
      });
      inputMode = 'partial';
      currentGroupCount = nextGroupCount;
    }
    return {success: true};
  } finally {
    // Scoped passes read the intermediates when the shared encoder is submitted.
    const recycleIntermediates = () => {
      for (const buffer of intermediateBuffers) {
        bufferPool.recycle(buffer);
      }
    };
    if (scope) scope.retain(recycleIntermediates);
    else recycleIntermediates();
  }
};

extent.supportsEvaluationScope = true;

function runExtentPass({
  scope,
  input,
  inputMode,
  inputGroupCount,
  channelCount,
  outputType,
  outputBuffer,
  outputLength,
  outputStride,
  outputOffset
}: {
  scope?: GPUEvaluationScope;
  input: GPUDataEvaluator;
  inputMode: ExtentInputMode;
  inputGroupCount: number;
  channelCount: number;
  outputType: GPUDataEvaluator['type'];
  outputBuffer: Buffer;
  outputLength: number;
  outputStride: number;
  outputOffset: number;
}): void {
  const wgslType = getWGSLType(outputType);
  const dispatchLayout = getWebGPUDispatchLayout(
    outputLength,
    outputBuffer.device.limits.maxComputeWorkgroupsPerDimension
  );
  const outputTable = new GPUDataEvaluator({
    buffer: outputBuffer,
    type: outputType,
    size: 2,
    length: outputLength,
    stride: outputStride,
    offset: outputOffset
  });
  // Group, channel, and output counts, layouts, and dispatch shape are pipeline constants.
  const constants: KernelConstants = {
    OUTPUT_LENGTH: outputLength,
    CHANNEL_COUNT: channelCount,
    INPUT_GROUP_COUNT: inputGroupCount,
    ...getWebGPUDispatchOverrides(dispatchLayout)
  };
  const source = /* wgsl */ `
${input.isConstant ? '' : getInputBinding('sourceValues', input, 0)}
${getSourceValuesAccessor(input, outputType, constants)}
${getOutputBinding(outputTable, input.isConstant ? 0 : 1)}
${getOutputWriter(outputTable, constants)}
${getExtentPassFunction(inputMode, outputType)}

var<workgroup> sharedMin: array<${wgslType}, ${RANDOM_ACCESS_WORKGROUP_SIZE}>;
var<workgroup> sharedMax: array<${wgslType}, ${RANDOM_ACCESS_WORKGROUP_SIZE}>;

@compute @workgroup_size(${RANDOM_ACCESS_WORKGROUP_SIZE}) fn main(
  @builtin(workgroup_id) workgroupId: vec3<u32>,
  @builtin(local_invocation_id) localId: vec3<u32>
) {
  let outputRowIndex = ${getWebGPUDispatchWorkgroupIndexOverride()};
  if (outputRowIndex >= OUTPUT_LENGTH) {
    return;
  }

  let channelIndex = outputRowIndex % CHANNEL_COUNT;
  let outputGroupIndex = outputRowIndex / CHANNEL_COUNT;
  let inputGroupIndex = outputGroupIndex * ${RANDOM_ACCESS_WORKGROUP_SIZE}u + localId.x;

  let result = extent_pass(channelIndex, inputGroupIndex);
  sharedMin[localId.x] = result[0];
  sharedMax[localId.x] = result[1];
  workgroupBarrier();

  var stride = ${Math.floor(RANDOM_ACCESS_WORKGROUP_SIZE / 2)}u;
  loop {
    if (stride == 0u) {
      break;
    }
    if (localId.x < stride) {
      let compareIndex = localId.x + stride;
      if (sharedMin[compareIndex] < sharedMin[localId.x]) {
        sharedMin[localId.x] = sharedMin[compareIndex];
      }
      if (sharedMax[compareIndex] > sharedMax[localId.x]) {
        sharedMax[localId.x] = sharedMax[compareIndex];
      }
    }
    workgroupBarrier();
    stride = stride / 2u;
  }

  if (localId.x == 0u) {
    write_result(outputRowIndex, array<${wgslType}, 2>(sharedMin[0], sharedMax[0]));
  }
}
`;

  const kernel = new Kernel(outputBuffer.device, {
    source: `${declareKernelConstants(constants)}\n${source}`,
    constants,
    shaderLayout: {
      bindings: [
        ...(input.isConstant
          ? []
          : [{name: 'sourceValues', type: 'storage' as const, group: 0, location: 0}]),
        {
          name: 'result',
          type: 'storage' as const,
          group: 0,
          location: input.isConstant ? 0 : 1
        }
      ]
    }
  });

  const bindings: Record<string, Buffer> = {result: outputBuffer};
  if (!input.isConstant) {
    bindings['sourceValues'] = input.buffer;
  }

  dispatchHandlerKernel({
    device: outputBuffer.device,
    scope,
    kernel,
    bindings,
    layout: dispatchLayout
  });
}

function getExtentPassFunction(inputMode: ExtentInputMode, type: GPUDataEvaluator['type']): string {
  const wgslType = getWGSLType(type);
  const [minValue, maxValue] = getExtentInitialValues(type);

  if (inputMode === 'raw') {
    return `fn extent_pass(channelIndex: u32, inputGroupIndex: u32) -> array<${wgslType}, 2> {
  var result: array<${wgslType}, 2>;
  result[0] = ${minValue};
  result[1] = ${maxValue};

  if (inputGroupIndex < INPUT_GROUP_COUNT) {
    let value = read_source_values(inputGroupIndex);
    result[0] = value[channelIndex];
    result[1] = value[channelIndex];
  }

  return result;
}`;
  }

  return `fn extent_pass(channelIndex: u32, inputGroupIndex: u32) -> array<${wgslType}, 2> {
  var result: array<${wgslType}, 2>;
  result[0] = ${minValue};
  result[1] = ${maxValue};

  if (inputGroupIndex < INPUT_GROUP_COUNT) {
    let rowIndex = inputGroupIndex * CHANNEL_COUNT + channelIndex;
    let value = read_source_values(rowIndex);
    result[0] = value[0];
    result[1] = value[1];
  }

  return result;
}`;
}

function getExtentInitialValues(type: GPUDataEvaluator['type']): [string, string] {
  switch (type) {
    case 'uint32':
      return ['0xffffffffu', '0u'];

    case 'sint32':
      return ['2147483647', '-2147483648'];

    case 'float32':
      return ['3.402823e38', '-3.402823e38'];

    default:
      throw new Error(`Unsupported WebGPU extent type for ${type}`);
  }
}

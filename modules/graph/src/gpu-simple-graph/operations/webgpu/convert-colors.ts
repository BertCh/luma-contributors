// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Kernel} from '../../../execution/compute/kernel';
import {dispatchHandlerKernel} from './common/kernel-dispatch';
import {
  getBoundedDispatchLayout,
  getBoundedDispatchOverrides,
  getBoundedInvocationIndexOverrideSource
} from '../../../gpu-core/gpu-dispatch-utils';
import type {OperationHandler} from '../../operation/operation';
import type {ConvertColorsInputs} from '../convert-colors';

const WORKGROUP_SIZE = 64;

export const convertColors: OperationHandler<ConvertColorsInputs> = async ({
  inputs,
  output,
  target,
  scope
}) => {
  const {source, inputFormat} = inputs;
  const dispatchLayout = getBoundedDispatchLayout(
    'convertColors',
    output.length,
    WORKGROUP_SIZE,
    target.device.limits.maxComputeWorkgroupsPerDimension
  );
  const kernel = new Kernel(target.device, {
    source: getConvertColorsWGSL(inputFormat),
    // Sizes are pipeline overrides so one shader module serves every row count and view offset.
    constants: {
      SOURCE_OFFSET: source.offset,
      SOURCE_STRIDE: source.stride,
      ROW_COUNT: output.length,
      ...getBoundedDispatchOverrides(dispatchLayout)
    },
    shaderLayout: {
      bindings: [
        {name: 'source', type: 'storage' as const, group: 0, location: 0},
        {name: 'result', type: 'storage' as const, group: 0, location: 1}
      ]
    }
  });

  dispatchHandlerKernel({
    device: target.device,
    scope,
    kernel,
    bindings: {source: source.buffer, result: target},
    layout: dispatchLayout
  });
  return {success: true};
};

convertColors.supportsEvaluationScope = true;

function getConvertColorsWGSL(inputFormat: ConvertColorsInputs['inputFormat']): string {
  const invocationIndex = getBoundedInvocationIndexOverrideSource(WORKGROUP_SIZE);
  return /* wgsl */ `\
override SOURCE_OFFSET: u32;
override SOURCE_STRIDE: u32;
override ROW_COUNT: u32;
${invocationIndex.declarations}

@group(0) @binding(0) var<storage, read> source: array<u32>;
@group(0) @binding(1) var<storage, read_write> result: array<u32>;

// Treat the storage binding as raw 32-bit words so one shader can address packed Uint8, Float16,
// and Float32 rows with arbitrary supported strides. The float readers decode IEEE bit patterns;
// they do not numerically convert integer values to floats.

fn readByte(byteIndex: u32) -> u32 {
  let word = source[byteIndex / 4u];
  let shift = (byteIndex % 4u) * 8u;
  return (word >> shift) & 0xffu;
}

fn readUint8(byteIndex: u32) -> f32 {
  return f32(readByte(byteIndex)) / 255.0;
}

fn readFloat16Bits(byteIndex: u32) -> f32 {
  let word = source[byteIndex / 4u];
  let values = unpack2x16float(word);
  return select(values.x, values.y, byteIndex % 4u == 2u);
}

fn readFloat32Bits(byteIndex: u32) -> f32 {
  return bitcast<f32>(source[byteIndex / 4u]);
}

fn readColor(rowIndex: u32) -> vec4<f32> {
  let rowByteOffset = SOURCE_OFFSET + rowIndex * SOURCE_STRIDE;
${getReadColorWGSL(inputFormat)}
}

@compute @workgroup_size(${WORKGROUP_SIZE}) fn main(
  @builtin(workgroup_id) workgroupId: vec3<u32>,
  @builtin(local_invocation_index) localInvocationIndex: u32
) {
  ${invocationIndex.body}
  let rowIndex = index;
  if (rowIndex >= ROW_COUNT) {
    return;
  }

  result[rowIndex] = pack4x8unorm(readColor(rowIndex));
}
`;
}

function getReadColorWGSL(inputFormat: ConvertColorsInputs['inputFormat']): string {
  switch (inputFormat) {
    case 'uint8x3':
      return `  return vec4<f32>(
    readUint8(rowByteOffset),
    readUint8(rowByteOffset + 1u),
    readUint8(rowByteOffset + 2u),
    1.0
  );`;
    case 'uint8x4':
      return `  return vec4<f32>(
    readUint8(rowByteOffset),
    readUint8(rowByteOffset + 1u),
    readUint8(rowByteOffset + 2u),
    readUint8(rowByteOffset + 3u)
  );`;
    case 'float16x3':
      return `  return vec4<f32>(
    readFloat16Bits(rowByteOffset),
    readFloat16Bits(rowByteOffset + 2u),
    readFloat16Bits(rowByteOffset + 4u),
    1.0
  );`;
    case 'float16x4':
      return `  return vec4<f32>(
    readFloat16Bits(rowByteOffset),
    readFloat16Bits(rowByteOffset + 2u),
    readFloat16Bits(rowByteOffset + 4u),
    readFloat16Bits(rowByteOffset + 6u)
  );`;
    case 'float32x3':
      return `  return vec4<f32>(
    readFloat32Bits(rowByteOffset),
    readFloat32Bits(rowByteOffset + 4u),
    readFloat32Bits(rowByteOffset + 8u),
    1.0
  );`;
    case 'float32x4':
      return `  return vec4<f32>(
    readFloat32Bits(rowByteOffset),
    readFloat32Bits(rowByteOffset + 4u),
    readFloat32Bits(rowByteOffset + 8u),
    readFloat32Bits(rowByteOffset + 12u)
  );`;
    default: {
      const unreachable: never = inputFormat;
      throw new Error(`Unsupported color input format ${unreachable}`);
    }
  }
}

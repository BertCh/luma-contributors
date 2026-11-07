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
import {getGPUVectorFormatInfo} from '../../../gpu-data/gpu-vector-format';
import type {OperationHandler} from '../../operation/operation';
import type {CastDataInputs} from '../cast-data';

const WORKGROUP_SIZE = 64;

/** WebGPU raw-byte numeric conversion for compact and 32-bit fixed-width formats. */
export const castData: OperationHandler<CastDataInputs> = async ({
  inputs,
  output,
  target,
  scope
}) => {
  const outputWordCount = Math.ceil(output.byteLength / 4);
  const dispatchLayout = getBoundedDispatchLayout(
    'castData',
    outputWordCount,
    WORKGROUP_SIZE,
    target.device.limits.maxComputeWorkgroupsPerDimension
  );
  const kernel = new Kernel(target.device, {
    source: getCastDataWGSL(inputs),
    // Sizes are pipeline overrides so one shader module serves every row count and view offset.
    constants: {
      SOURCE_OFFSET: inputs.source.offset,
      SOURCE_STRIDE: inputs.source.stride,
      SCALAR_COUNT: output.length * getGPUVectorFormatInfo(inputs.inputFormat).components,
      ...getBoundedDispatchOverrides(dispatchLayout)
    },
    shaderLayout: {
      bindings: [
        {name: 'source', type: 'storage', group: 0, location: 0},
        {name: 'result', type: 'storage', group: 0, location: 1}
      ]
    }
  });
  if (outputWordCount > 0) {
    dispatchHandlerKernel({
      device: target.device,
      scope,
      kernel,
      bindings: {source: inputs.source.buffer, result: target},
      layout: dispatchLayout
    });
  } else if (scope) {
    scope.retain(() => kernel.destroy());
  } else {
    kernel.destroy();
  }
  return {success: true};
};

castData.supportsEvaluationScope = true;

function getCastDataWGSL(inputs: CastDataInputs): string {
  const invocationIndex = getBoundedInvocationIndexOverrideSource(WORKGROUP_SIZE);
  const inputInfo = getGPUVectorFormatInfo(inputs.inputFormat);
  const outputInfo = getGPUVectorFormatInfo(inputs.outputFormat);
  const componentCount = inputInfo.components;
  const outputComponentByteLength = outputInfo.elementByteLength / outputInfo.components;
  const scalarsPerWord = 4 / outputComponentByteLength;
  return /* wgsl */ `\
override SOURCE_OFFSET: u32;
override SOURCE_STRIDE: u32;
override SCALAR_COUNT: u32;
${invocationIndex.declarations}

@group(0) @binding(0) var<storage, read> source: array<u32>;
@group(0) @binding(1) var<storage, read_write> result: array<u32>;

fn readByte(byteIndex: u32) -> u32 {
  let word = source[byteIndex / 4u];
  return (word >> ((byteIndex % 4u) * 8u)) & 0xffu;
}

fn readUint16(byteIndex: u32) -> u32 {
  return readByte(byteIndex) | (readByte(byteIndex + 1u) << 8u);
}

fn readSourceValue(scalarIndex: u32) -> f32 {
  let rowIndex = scalarIndex / ${componentCount}u;
  let componentIndex = scalarIndex % ${componentCount}u;
  let byteIndex = SOURCE_OFFSET + rowIndex * SOURCE_STRIDE +
    componentIndex * ${inputInfo.elementByteLength / componentCount}u;
  ${getReadValueExpression(inputInfo.signedDataType, inputInfo.normalized)}
}

${getEncodeValueFunction(inputs.outputFormat)}

@compute @workgroup_size(${WORKGROUP_SIZE}) fn main(
  @builtin(workgroup_id) workgroupId: vec3<u32>,
  @builtin(local_invocation_index) localInvocationIndex: u32
) {
  ${invocationIndex.body}
  let wordIndex = index;
  let firstScalarIndex = wordIndex * ${scalarsPerWord}u;
  if (firstScalarIndex >= SCALAR_COUNT) {
    return;
  }
  ${getPackOutputWord(outputInfo.signedDataType, outputComponentByteLength)}
}
`;
}

function getReadValueExpression(type: string, normalized: boolean): string {
  switch (type) {
    case 'float32':
      return 'return bitcast<f32>(source[byteIndex / 4u]);';
    case 'float16':
      return `let values = unpack2x16float(source[byteIndex / 4u]);
  return select(values.x, values.y, byteIndex % 4u == 2u);`;
    case 'uint8':
      return normalized
        ? 'return f32(readByte(byteIndex)) / 255.0;'
        : 'return f32(readByte(byteIndex));';
    case 'sint8':
      return normalized
        ? 'return max(f32(i32(readByte(byteIndex) << 24u) >> 24) / 127.0, -1.0);'
        : 'return f32(i32(readByte(byteIndex) << 24u) >> 24);';
    case 'uint16':
      return normalized
        ? 'return f32(readUint16(byteIndex)) / 65535.0;'
        : 'return f32(readUint16(byteIndex));';
    case 'sint16':
      return normalized
        ? 'return max(f32(i32(readUint16(byteIndex) << 16u) >> 16) / 32767.0, -1.0);'
        : 'return f32(i32(readUint16(byteIndex) << 16u) >> 16);';
    case 'uint32':
      return normalized
        ? 'return f32(source[byteIndex / 4u]) / 4294967295.0;'
        : 'return f32(source[byteIndex / 4u]);';
    case 'sint32':
      return normalized
        ? 'return max(f32(bitcast<i32>(source[byteIndex / 4u])) / 2147483647.0, -1.0);'
        : 'return f32(bitcast<i32>(source[byteIndex / 4u]));';
    default:
      throw new Error(`castData WebGPU input does not support ${type}`);
  }
}

function getEncodeValueFunction(outputFormat: CastDataInputs['outputFormat']): string {
  const outputInfo = getGPUVectorFormatInfo(outputFormat);
  const {signedDataType, normalized} = outputInfo;
  if (signedDataType === 'float32' || signedDataType === 'float16') {
    return 'fn encodeValue(value: f32) -> f32 { return value; }';
  }
  const maximum = getIntegerMaximum(signedDataType);
  const minimum = signedDataType.startsWith('sint') ? -maximum : 0;
  const value = normalized
    ? `round(clamp(value, ${minimum < 0 ? '-1.0' : '0.0'}, 1.0) * ${maximum}.0)`
    : `round(clamp(value, ${minimum}.0, ${maximum}.0))`;
  return `fn encodeValue(value: f32) -> f32 { return ${value}; }`;
}

function getPackOutputWord(type: string, componentByteLength: number): string {
  if (componentByteLength === 4) {
    const encoded = 'encodeValue(readSourceValue(firstScalarIndex))';
    if (type === 'float32') return `result[wordIndex] = bitcast<u32>(${encoded});`;
    if (type === 'sint32') return `result[wordIndex] = bitcast<u32>(i32(${encoded}));`;
    return `result[wordIndex] = u32(${encoded});`;
  }
  if (componentByteLength === 2 && type === 'float16') {
    return `let second = select(0.0, readSourceValue(firstScalarIndex + 1u), firstScalarIndex + 1u < SCALAR_COUNT);
  result[wordIndex] = pack2x16float(vec2<f32>(
    encodeValue(readSourceValue(firstScalarIndex)),
    encodeValue(second)
  ));`;
  }
  const scalarSlots = 4 / componentByteLength;
  const bitWidth = componentByteLength * 8;
  const mask = bitWidth === 8 ? '0xffu' : '0xffffu';
  const values = Array.from({length: scalarSlots}, (_, slot) => {
    const scalarIndex = `firstScalarIndex + ${slot}u`;
    const encoded = type.startsWith('sint')
      ? `bitcast<u32>(i32(encodeValue(readSourceValue(${scalarIndex}))))`
      : `u32(encodeValue(readSourceValue(${scalarIndex})))`;
    return `(select(0u, (${encoded}) & ${mask}, ${scalarIndex} < SCALAR_COUNT) << ${slot * bitWidth}u)`;
  });
  return `result[wordIndex] = ${values.join(' |\n    ')};`;
}

function getIntegerMaximum(type: string): number {
  switch (type) {
    case 'sint8':
      return 127;
    case 'uint8':
      return 255;
    case 'sint16':
      return 32767;
    case 'uint16':
      return 65535;
    case 'sint32':
      return 2147483647;
    case 'uint32':
      return 4294967295;
    default:
      throw new Error(`castData WebGPU output does not support ${type}`);
  }
}

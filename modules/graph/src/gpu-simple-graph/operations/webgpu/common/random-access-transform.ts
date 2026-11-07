// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {SignedDataType} from '@luma.gl/core';
import {GPUDataEvaluator} from '../../../operation/gpu-data-evaluator';
import {getLiteralValue, getWGSLType, getZeroValue} from './helper';
import type {KernelConstants} from './kernel-dispatch';

export const RANDOM_ACCESS_WORKGROUP_SIZE = 64;

export function getInputBinding(name: string, input: GPUDataEvaluator, index: number): string {
  const inputType = getWGSLType(input.type);
  return `@group(0) @binding(${index}) var<storage, read> ${name}: array<${inputType}>;`;
}

/** Registers a view's element offset and stride as `${PREFIX}_OFFSET` and `${PREFIX}_STRIDE`. */
export function addViewConstants(
  constants: KernelConstants,
  prefix: string,
  view: GPUDataEvaluator
): void {
  constants[`${prefix}_OFFSET`] = view.offset / view.ValueType.BYTES_PER_ELEMENT;
  constants[`${prefix}_STRIDE`] = view.stride / view.ValueType.BYTES_PER_ELEMENT;
}

export function getTableAccessor(
  bindingName: string,
  input: GPUDataEvaluator,
  asType: SignedDataType,
  accessorName: string = bindingName,
  constants: KernelConstants
): string {
  const type = getWGSLType(asType);

  if (input.isConstant) {
    const values = input.value;
    if (!values) {
      throw new Error(`Constant input ${input} is missing CPU values`);
    }
    return `fn read_${accessorName}(_sourceIndex: u32) -> array<${type}, ${input.size}> {
  return array<${type}, ${input.size}>(${Array.from({length: input.size}, (_, index) =>
    getLiteralValue(type, values[index] ?? 0)
  ).join(', ')});
}`;
  }

  // Layout is a pipeline constant so one shader serves every view offset and stride.
  const prefix = accessorName.toUpperCase();
  addViewConstants(constants, prefix, input);
  const inputType = getWGSLType(input.type);
  const cast = inputType === type ? '' : `${type}`;

  return `fn read_${accessorName}(sourceIndex: u32) -> array<${type}, ${input.size}> {
  var value: array<${type}, ${input.size}>;
  let rowOffset = ${prefix}_OFFSET + sourceIndex * ${prefix}_STRIDE;
${Array.from({length: input.size}, (_, index) =>
  cast
    ? `  value[${index}] = ${cast}(${bindingName}[rowOffset + ${index}u]);`
    : `  value[${index}] = ${bindingName}[rowOffset + ${index}u];`
).join('\n')}
  return value;
}`;
}

export function getSourceValuesAccessor(
  input: GPUDataEvaluator,
  asType: SignedDataType,
  constants: KernelConstants
): string {
  return getTableAccessor('sourceValues', input, asType, 'source_values', constants);
}

export function getOutputBinding(output: GPUDataEvaluator, bindingIndex: number): string {
  const type = getWGSLType(output.type);
  return `@group(0) @binding(${bindingIndex}) var<storage, read_write> result: array<${type}>;`;
}

export function getOutputWriter(output: GPUDataEvaluator, constants: KernelConstants): string {
  addViewConstants(constants, 'RESULT', output);
  const type = getWGSLType(output.type);
  return `fn write_result(rowIndex: u32, value: array<${type}, ${output.size}>) {
  let rowOffset = RESULT_OFFSET + rowIndex * RESULT_STRIDE;
${Array.from({length: output.size}, (_, elementIndex) => `  result[rowOffset + ${elementIndex}u] = value[${elementIndex}];`).join('\n')}
}`;
}

export function getZeroResultFunction(type: SignedDataType, size: number): string {
  const zero = getZeroValue(type);
  return `fn zero_result() -> array<${getWGSLType(type)}, ${size}> {
  var result: array<${getWGSLType(type)}, ${size}>;
${Array.from({length: size}, (_, index) => `  result[${index}] = ${zero};`).join('\n')}
  return result;
}`;
}

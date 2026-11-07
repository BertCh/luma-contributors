// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {Kernel} from '../../../execution/compute/kernel';
import {OperationHandler} from '../../operation/operation';
import {GPUDataEvaluator} from '../../operation/gpu-data-evaluator';
import {getWebGPUDispatchLayout, getWebGPUDispatchRowIndexOverride} from './common/dispatch';
import {getLiteralValue, getWGSLType} from './common/helper';
import {
  declareKernelConstants,
  dispatchHandlerKernel,
  type KernelConstants
} from './common/kernel-dispatch';
import {
  addViewConstants,
  getInputBinding,
  getOutputBinding,
  getOutputWriter,
  getSourceValuesAccessor,
  getZeroResultFunction,
  RANDOM_ACCESS_WORKGROUP_SIZE
} from './common/random-access-transform';
import {getWebGPUDispatchOverrides} from './common/dispatch';

export const gather: OperationHandler<{
  ids: GPUDataEvaluator;
  sourceValues: GPUDataEvaluator;
}> = async ({inputs, output, target, scope}) => {
  const {ids, sourceValues} = inputs;
  const idsType = getWGSLType(ids.type);
  const bindings: {name: string; input: GPUDataEvaluator; index: number}[] = [];
  if (!ids.isConstant) {
    bindings.push({name: 'ids', input: ids, index: bindings.length});
  }
  if (!sourceValues.isConstant) {
    bindings.push({name: 'sourceValues', input: sourceValues, index: bindings.length});
  }
  const dispatchLayout = getWebGPUDispatchLayout(
    Math.ceil(output.length / RANDOM_ACCESS_WORKGROUP_SIZE),
    target.device.limits.maxComputeWorkgroupsPerDimension
  );

  // Row counts, layouts, and dispatch shape are pipeline constants, so one shader serves every size.
  const constants: KernelConstants = {
    ROW_COUNT: output.length,
    SOURCE_LENGTH: sourceValues.length,
    ...getWebGPUDispatchOverrides(dispatchLayout)
  };
  const source = /* wgsl */ `
${bindings.map(({name, input, index}) => getInputBinding(name, input, index)).join('\n')}
${getIdsAccessor(ids, idsType, constants)}
${getSourceValuesAccessor(sourceValues, output.type, constants)}
${getOutputBinding(output, bindings.length)}
${getOutputWriter(output, constants)}
${getZeroResultFunction(output.type, output.size)}
${getGatherFunction(ids.type, output.type, output.size)}

@compute @workgroup_size(${RANDOM_ACCESS_WORKGROUP_SIZE}) fn main(
  @builtin(workgroup_id) workgroupId: vec3<u32>,
  @builtin(local_invocation_id) localId: vec3<u32>
) {
  let rowIndex = ${getWebGPUDispatchRowIndexOverride(RANDOM_ACCESS_WORKGROUP_SIZE)};
  if (rowIndex >= ROW_COUNT) {
    return;
  }

  let idsValue = read_ids(rowIndex);
  let result = gather(idsValue);
  write_result(rowIndex, result);
}
`;

  const kernel = new Kernel(target.device, {
    source: `${declareKernelConstants(constants)}\n${source}`,
    constants,
    shaderLayout: {
      bindings: [
        ...bindings.map(({name, index}) => ({
          name,
          type: 'storage' as const,
          group: 0,
          location: index
        })),
        {name: 'result', type: 'storage' as const, group: 0, location: bindings.length}
      ]
    }
  });

  const computationBindings: Record<string, Buffer> = {};
  if (!ids.isConstant) {
    computationBindings['ids'] = ids.buffer;
  }
  if (!sourceValues.isConstant) {
    computationBindings['sourceValues'] = sourceValues.buffer;
  }
  computationBindings['result'] = target;

  dispatchHandlerKernel({
    device: target.device,
    scope,
    kernel,
    bindings: computationBindings,
    layout: dispatchLayout
  });
  return {success: true};
};

gather.supportsEvaluationScope = true;

function getIdsAccessor(input: GPUDataEvaluator, type: string, constants: KernelConstants): string {
  if (input.isConstant) {
    const values = input.value;
    if (!values) {
      throw new Error(`Constant input ${input} is missing CPU values`);
    }
    return `fn read_ids(_rowIndex: u32) -> ${type} {
  return ${getLiteralValue(type, values[0] ?? 0)};
}`;
  }

  addViewConstants(constants, 'IDS', input);
  return `fn read_ids(rowIndex: u32) -> ${type} {
  let rowOffset = IDS_OFFSET + rowIndex * IDS_STRIDE;
  return ids[rowOffset];
}`;
}

function getGatherFunction(
  idsType: GPUDataEvaluator['type'],
  outputType: GPUDataEvaluator['type'],
  outputSize: number
): string {
  const idsWGSLType = getWGSLType(idsType);
  const outputWGSLType = getWGSLType(outputType);
  const indexExpression =
    idsWGSLType === 'u32' ? 'i32(idsValue)' : idsWGSLType === 'i32' ? 'idsValue' : 'i32(idsValue)';

  return `fn gather(idsValue: ${idsWGSLType}) -> array<${outputWGSLType}, ${outputSize}> {
  let sourceIndex = ${indexExpression};
  if (sourceIndex < 0 || sourceIndex >= i32(SOURCE_LENGTH)) {
    return zero_result();
  }
  return read_source_values(u32(sourceIndex));
}`;
}

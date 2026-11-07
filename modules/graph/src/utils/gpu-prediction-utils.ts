// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GraphDataView} from '@luma-contributors/graph/gpu-core';
import {validateGraphOutputsDisjointFromInputs, validatePackedView} from './gpu-contributor-utils';

/** Per-query status: successful, missing/invalid input, rank failure, or exhausted geometric capacity. */
export const GPU_PREDICTION_STATUS = {success: 0, invalid: 1, singular: 2, capacity: 3} as const;

/** Validates packed graph ports without taking ownership. @internal */
export function validatePredictionViews(
  id: string,
  inputs: readonly GraphDataView[],
  outputs: readonly GraphDataView[]
): void {
  for (const view of [...inputs, ...outputs]) {
    validatePackedView(view, [view.format], id);
  }
  validateGraphOutputsDisjointFromInputs(id, outputs, inputs);
  if (new Set(outputs.map(view => view.buffer)).size !== outputs.length) {
    throw new Error(`${id} outputs must have distinct buffers`);
  }
}

/** Scalar storage bindings keep packed vec2/vec4 layouts and arbitrary component offsets. @internal */
export function getPackedPredictionAccessors(
  reads: Record<string, 2 | 4>,
  writes: readonly string[] = []
): string {
  return (
    Object.entries(reads)
      .map(
        ([name, count]) =>
          `fn read_${name}(row: u32) -> vec${count}<f32> { return vec${count}<f32>(${Array.from({length: count}, (_, component) => `${name}[${name}Offset + row * ${count}u + ${component}u]`).join(', ')}); }`
      )
      .join('\n') +
    writes
      .map(
        name =>
          `fn write_${name}(row: u32, value: vec4<f32>) { ${Array.from({length: 4}, (_, component) => `${name}[${name}Offset + row * 4u + ${component}u] = value[${component}u];`).join(' ')} }`
      )
      .join('\n')
  );
}

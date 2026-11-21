// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import type {GraphDataView} from '@luma.gl/graph/gpu-core';
import {getCoreDefaultBuffer} from '../../analysis-core/contracts/network-topology';
import type {GPUSpatialWeights} from '../spatial-weights/spatial-weights';
import {getGPUSpatialWeightsLossViews} from '../spatial-weights/gpu-chunked-spatial-weights';
import {validateDisjointWritableViews} from '../../analysis-core/kernels/graph-chunk-dispatch';
import {
  validateCapacity,
  validateSinglePackedView,
  validateGPUStorageBindingByteLength,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';

type Views = Readonly<Record<string, GraphDataView | undefined>>;

/** Accessibility accepts independent positive certificates as well as shared negative loss flags. @internal */
export function getAccessibilityWeightsLossChecks(id: string, weights: GPUSpatialWeights) {
  const complete = weights.status?.complete;
  if (complete) {
    validateSinglePackedView(complete, ['uint32'], `${id} weights.status.complete`);
    validateRequirement(
      id,
      complete.length === 1,
      `weights.status.complete length must equal 1; received ${complete.length}`
    );
  }
  // The shared reader currently requires a failure-polarity companion for complete. Collect its
  // ordinary flags without that metadata, then read the independent positive certificate itself.
  const negative = getGPUSpatialWeightsLossViews({
    ...weights,
    status: weights.status ? {...weights.status, complete: undefined} : undefined
  });
  return [
    ...negative.map(flag => ({flag, test: '!= 0u'})),
    ...(complete ? [{flag: complete, test: '== 0u'}] : [])
  ];
}

/** Validate logical separation globally, and aligned separation only for co-bound fields. @internal */
export function validateCatchmentViews(
  id: string,
  inputs: Views,
  outputs: Views,
  dispatches: readonly {reads: Views; writes: Views}[]
): void {
  const physical = (view: GraphDataView) =>
    view.buffer.defaultBuffer ? getCoreDefaultBuffer(view.buffer.defaultBuffer) : view.buffer;
  const entries = Object.entries(outputs).filter(
    (entry): entry is [string, GraphDataView] => entry[1] !== undefined
  );
  for (const [outputIndex, [outputName, output]] of entries.entries()) {
    if (output.length === 0) continue;
    const outputEnd =
      output.byteOffset + (output.length - 1) * output.byteStride + output.rowByteLength;
    for (const [name, other] of [...Object.entries(inputs), ...entries.slice(outputIndex + 1)]) {
      if (!other || other.length === 0 || physical(output) !== physical(other)) continue;
      const end = other.byteOffset + (other.length - 1) * other.byteStride + other.rowByteLength;
      validateRequirement(
        id,
        output.byteOffset >= end || other.byteOffset >= outputEnd,
        `${outputName} and ${name} logical byte ranges must not overlap`
      );
    }
  }
  for (const dispatch of dispatches) {
    validateDisjointWritableViews(id, dispatch.writes);
    for (const [outputName, output] of Object.entries(dispatch.writes))
      for (const [inputName, input] of Object.entries(dispatch.reads)) {
        validateDisjointWritableViews(id, {
          [`read.${inputName}`]: input,
          [`write.${outputName}`]: output
        });
      }
  }
}

/** Scratch needs one storage binding and one allocation; validate both before declaration. @internal */
export function validateCatchmentStorage(
  device: Device,
  id: string,
  name: string,
  bytes: number
): void {
  validateGPUStorageBindingByteLength(device, id, name, bytes, 'reduce row or group counts');
  validateCapacity(id, `${name} allocation byteLength`, bytes, 0, device.limits.maxBufferSize);
}

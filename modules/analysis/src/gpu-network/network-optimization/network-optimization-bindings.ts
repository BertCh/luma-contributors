// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import type {GPUCommandGraph, GraphDataView} from '@luma.gl/graph/gpu-core';
import {validateGraphChunkBindingRanges} from '../../analysis-core/kernels/graph-chunk-dispatch';
import {
  validateGraphOutputsDisjointFromInputs,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {getCoreDefaultBuffer} from '../../analysis-core/contracts/network-topology';

/** Checks every bound caller range and STORAGE usage without declaring resources. @internal */
export function validateNetworkOptimizationBindings<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  views: Readonly<Record<string, GraphDataView | undefined>>
): void {
  validateGraphChunkBindingRanges(graph, id, views);
  // A logical import must not promise STORAGE when its actual caller-owned buffer lacks it.
  for (const [name, view] of Object.entries(views)) {
    if (view?.buffer.defaultBuffer) {
      validateRequirement(
        id,
        (getCoreDefaultBuffer(view.buffer.defaultBuffer).usage & Buffer.STORAGE) !== 0,
        `${name} physical buffer usage must include Buffer.STORAGE`
      );
    }
  }
}

/** Disallows read-only/writable aliases even through separate handles of one physical buffer.
 * One physical allocation must be represented by one graph import when it can be written.
 * @internal */
export function validateNetworkOptimizationInputOutputBuffers(
  id: string,
  outputs: readonly (GraphDataView | undefined)[],
  inputs: readonly (GraphDataView | undefined)[]
): void {
  validateGraphOutputsDisjointFromInputs(id, outputs, inputs);
  const getAllocation = (view: GraphDataView) =>
    view.buffer.defaultBuffer ? getCoreDefaultBuffer(view.buffer.defaultBuffer) : view.buffer;
  const inputAllocations = new Set(inputs.filter(view => view !== undefined).map(getAllocation));
  validateRequirement(
    id,
    outputs.every(view => !view || !inputAllocations.has(getAllocation(view))),
    'output views must use separate physical buffers from inputs'
  );
}

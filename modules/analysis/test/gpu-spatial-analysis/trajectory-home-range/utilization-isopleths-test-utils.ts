// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, type GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {
  validateCapacity,
  validateGPUStorageBindingByteLength,
  validateRequirement
} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {createWGSLKernelNode} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';

// The 1/2-density fixture incurs intermediate float32 rounding at its mixed group and root;
// three parts in ten million allow that rounding while retaining an independent mass check.
export const NONUNIFORM_MASS_RELATIVE_TOLERANCE = 3e-7;

// The safety ceiling is the largest fixed boundary fixture, never a device-advertised capacity.
export const UTILIZATION_HEADLESS_MAXIMUM_CELL_COUNT = 2 ** 24 + 257;
// One workgroup beyond the removed 2^24-cell cap exercises a fourth prefix level at fixed cost.
export const UTILIZATION_HEADLESS_SCALABILITY_CELL_COUNT = 2 ** 24 + 256;

/** Rejects accidental device-sized allocations before a headless harness declares any buffers. */
export function validateUtilizationHeadlessCellCount(length: number): void {
  validateCapacity(
    'isopleth-headless',
    'cellCount',
    length,
    2,
    UTILIZATION_HEADLESS_MAXIMUM_CELL_COUNT
  );
}

/** Declares a parity-density fixture without allocating or uploading a row-sized CPU array. */
export function createNonuniformUtilizationDensity<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  length: number,
  byteOffset: number
) {
  validateGPUStorageBindingByteLength(
    graph.device,
    'isopleth-fixture',
    'density',
    (byteOffset % 256) + length * 4,
    'reduce the fixture length or alignment prefix'
  );
  validateRequirement(
    'isopleth-fixture',
    byteOffset + length * 4 <= graph.device.limits.maxBufferSize,
    'density buffer byte length must fit maxBufferSize'
  );
  const allocation = createTransientView(
    graph,
    'fixture-density',
    'float32',
    length + byteOffset / 4
  );
  const density = graph.createDataView(allocation.buffer, {format: 'float32', length, byteOffset});
  const node = createWGSLKernelNode(graph, {
    id: 'initialize-fixture-density',
    operation: 'InitializeUtilizationDensityFixture',
    invocationCount: length,
    bindings: [{name: 'density', view: density, type: 'f32', access: 'read_write'}],
    body: 'density[densityOffset + index] = select(1.0, 2.0, index % 2u == 0u);'
  });
  return {density, node};
}

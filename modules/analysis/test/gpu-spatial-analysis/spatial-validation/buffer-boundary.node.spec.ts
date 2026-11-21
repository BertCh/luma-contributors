// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {GPUSpatialCrossValidation} from '../../../src/gpu-spatial-analysis/spatial-validation/gpu-spatial-cross-validation';
import {computeSpatialCrossValidationOnCPU} from '../../../src/gpu-spatial-analysis/spatial-validation/cpu-spatial-cross-validation-operation';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {executeSpatialAuditKernel} from './spatial-design-audit.test-utils';
import {createWGSLKernelNode} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {
  getValidationBufferMembershipWGSL,
  isWithinValidationRadiusOnCPU
} from '../../../src/gpu-spatial-analysis/spatial-validation/validation-buffer-membership-wgsl';

it.each([
  [1, 0, 1, true],
  [1, 2 ** -27, 1, false],
  [3, 4, 5, true],
  [1, 0, 1 - 2 ** -30, false],
  [2 ** -149, 0, 0, false],
  [0, 0, 0, true],
  [2 ** -63, 2 ** -64, (17 / 16) * 2 ** -63, false],
  [-0, 0, 0, true]
] as const)('TWN-41 resolves exact membership at (%s,%s) with radius %s', (horizontal, vertical, radius, expected) => {
  const device = createNullWebGPUDevice(),
    graph = new GPUCommandGraph(device);
  try {
    const positions = createTransientView(graph, 'positions', 'float32', 2),
      result = createTransientView(graph, 'result', 'uint32', 1);
    const [kernel] = getGPUCommandNodeKernelSources([
      createWGSLKernelNode(graph, {
        id: 'radius-probe',
        operation: 'radius-probe',
        invocationCount: 1,
        bindings: [
          {name: 'positions', view: positions, type: 'f32', access: 'read'},
          {name: 'result', view: result, type: 'u32', access: 'read_write'}
        ],
        declarations: getValidationBufferMembershipWGSL(radius),
        body: 'result[resultOffset] = u32(isWithinValidationRadius(vec2<f32>(0.0), vec2<f32>(positions[positionsOffset], positions[positionsOffset + 1u]), 0.0));'
      })
    ]);
    const values = new Uint32Array(1);
    executeSpatialAuditKernel(
      kernel,
      {
        positions: Float32Array.of(horizontal, vertical),
        result: values
      },
      {flushSubnormalProducts: true, ...(radius === 0 ? {forbidCalls: ['validationSquare']} : {})}
    );
    // Exact Pythagorean/equality cases and strictly positive excess terms; no twin oracle.
    expect(Array.from(values)).toEqual([Number(expected)]);
    expect(isWithinValidationRadiusOnCPU(0, 0, horizontal, vertical, radius)).toBe(expected);
  } finally {
    device.destroy();
  }
});

it('TWN-05/TWN-41 keeps training rows strictly outside the inclusive stored-coordinate buffer', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends 'float32x2' | 'uint32'>(format: Format, length: number) =>
    createTransientView(graph, `boundary-${serial++}`, format, length);
  try {
    const contributor = new GPUSpatialCrossValidation({
      positions: view('float32x2', 2),
      blocks: view('uint32', 2),
      foldCount: 2,
      bufferDistance: 1,
      output: {
        folds: view('uint32', 2),
        training: view('uint32', 4),
        testing: view('uint32', 4),
        counts: view('uint32', 6),
        status: view('uint32', 2)
      }
    });
    const nodes = contributor.getCommandNodes(graph);
    const [masks] = getGPUCommandNodeKernelSources(
      nodes.filter(node => node.id.endsWith('-masks'))
    );
    const [counts] = getGPUCommandNodeKernelSources(
      nodes.filter(node => node.id.endsWith('-counts'))
    );
    const values = {
      positions: Float32Array.of(0, 0, 1, 2 ** -13),
      folds: Uint32Array.of(0, 1),
      training: new Uint32Array(4),
      testing: new Uint32Array(4),
      counts: new Uint32Array(6),
      status: new Uint32Array(2)
    };
    executeSpatialAuditKernel(masks, values);
    executeSpatialAuditKernel(counts, values);
    // Exact squared distance = 1 + 2^-26 > 1, although the displayed float32
    // distance is 1. Membership uses the stored coordinates, not that diagnostic.
    expect(Array.from(values.training)).toEqual([0, 1, 1, 0]);
    expect(Array.from(values.status)).toEqual([0, 0]);
    const cpu = computeSpatialCrossValidationOnCPU(
      {positions: values.positions, blocks: Uint32Array.of(1, 0)},
      {foldCount: 2, bufferDistance: 1}
    );
    expect(Array.from(cpu.training)).toEqual([0, 1, 1, 0]);
    expect(Array.from(cpu.status)).toEqual([0, 0]);
  } finally {
    device.destroy();
  }
});

// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import type {GPUCPUVector} from '@luma.gl/graph/gpu-core';
import {CPUNetworkFlowOperation} from '../../../src/gpu-network/network-optimization/cpu-network-flow-operation';
import {GPUNetworkFlowOperation} from '../../../src/gpu-network/network-optimization/gpu-network-flow-operation';
import {constructNetworkOperationContributor} from '../../utils/network-operation-test-utils';
import {
  getEmptyFlowExpectedRows,
  getEmptyFlowOperationProps
} from './network-flow-paths-x1.test-utils';

for (const solver of ['reference', 'parallel'] as const) {
  it(`network-recipes:X7 ${solver} GPU operation declares an empty edge-flow output`, () => {
    const operation = new GPUNetworkFlowOperation(getEmptyFlowOperationProps(solver));
    expect(operation.outputs['flow'].length).toBe(0);
    expect(() =>
      constructNetworkOperationContributor(operation, {
        edges: {format: 'uint32x2', length: 0},
        capacities: {format: 'uint32', length: 0},
        parameters: {format: 'uint32', length: 3}
      })
    ).not.toThrow();
  });
  it(`network-recipes:X7 ${solver} CPU operation publishes a complete empty decomposition`, async () => {
    const operation = new CPUNetworkFlowOperation(getEmptyFlowOperationProps(solver));
    const vector = (
      values: Uint32Array,
      format: 'uint32' | 'uint32x2' = 'uint32'
    ): GPUCPUVector => {
      const length = values.length / (format === 'uint32x2' ? 2 : 1);
      return {format, length, chunks: [{values, length}], pack: () => values};
    };
    const rows = await operation.execute({
      inputs: {
        edges: vector(new Uint32Array(0), 'uint32x2'),
        capacities: vector(new Uint32Array(0)),
        parameters: vector(Uint32Array.of(0, 1, 1))
      }
    });
    expect(rows).toEqual(getEmptyFlowExpectedRows());
  });
}

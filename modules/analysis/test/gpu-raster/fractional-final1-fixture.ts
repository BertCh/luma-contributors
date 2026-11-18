// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {GPUGraph, gpuGraph, type GPUGraphOperation} from '@luma.gl/graph/gpu-core';
import {CPUVector} from '@luma.gl/graph/gpu-data';
import {GPU_CONTRIBUTOR_OPERATION_LOWERING} from '../../src/analysis-core/operations/gpu-contributor-operation';
import {
  makeCPUVector,
  getCPUVectorRows,
  type OperationGraphInput
} from '../utils/mixed-execution-test-utils';

/** Mismatch and candidate-only loss each followed by a clean run of one compiled graph. */
export async function runPreparedCoverageRecovery(
  device: Device,
  operation: GPUGraphOperation,
  inputs: Record<string, OperationGraphInput>,
  outputs: string[]
) {
  const graph = new GPUGraph();
  for (const [name, input] of Object.entries(inputs))
    graph.addInput(name, gpuGraph.vector(input.format, {length: input.values.length}));
  graph.add(operation);
  const compiled = graph.compile(device, {
    outputs,
    outputLocations: Object.fromEntries(outputs.map(name => [name, 'cpu' as const])),
    lowerings: [GPU_CONTRIBUTOR_OPERATION_LOWERING]
  });
  try {
    const results = [];
    for (const incomplete of ['count-mismatch', 'clean', 'candidate-only', 'clean']) {
      const rows = {
        ...inputs,
        requiredCount: {
          format: 'uint32' as const,
          values: Uint32Array.of(
            inputs['count'].values[0] + Number(incomplete === 'count-mismatch')
          )
        },
        candidateOverflow: {
          format: 'uint32' as const,
          values: Uint32Array.of(Number(incomplete === 'candidate-only'))
        }
      };
      const result = await compiled.run({
        inputs: Object.fromEntries(
          Object.entries(rows).map(([name, value]) => [
            name,
            makeCPUVector(value.format, value.values)
          ])
        )
      });
      results.push({
        incomplete: incomplete !== 'clean',
        rows: Object.fromEntries(
          outputs.map(name => {
            const value: unknown = result[name];
            if (!(value instanceof CPUVector)) throw new Error(`${name} must return a CPU vector`);
            return [name, getCPUVectorRows(value)];
          })
        )
      });
    }
    return results;
  } finally {
    compiled.destroy();
    const owner = operation as {destroy?: () => void};
    owner.destroy?.();
  }
}

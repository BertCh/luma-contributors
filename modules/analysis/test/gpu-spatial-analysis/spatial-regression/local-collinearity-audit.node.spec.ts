// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {
  createTransientView,
  GPUCommandGraph,
  getViewBindingRange,
  type GraphBufferHandle
} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {WgslExec, WgslParser} from 'wgsl_reflect';
import {
  getWGSLKernelNodeRecord,
  recordWGSLKernelNodes
} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {GPULocalCollinearity} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-local-collinearity';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  LOCAL_COLLINEARITY_AUDIT_CASES,
  LOCAL_AUDIT_NEIGHBORS,
  LOCAL_AUDIT_WEIGHTS
} from './local-collinearity-audit-fixtures';

for (const fixture of LOCAL_COLLINEARITY_AUDIT_CASES) {
  it(`executes emitted local collinearity kernels against NumPy: transformed=${fixture.transformed}`, () => {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    recordWGSLKernelNodes(graph);
    const memory = new Map<GraphBufferHandle, ArrayBuffer>();
    let serial = 0;
    const port = <Format extends 'float32' | 'uint32'>(
      format: Format,
      values: readonly number[]
    ) => {
      const view = createTransientView(graph, `local-audit-${serial++}`, format, values.length);
      memory.set(
        view.buffer,
        (format === 'uint32' ? Uint32Array.from(values) : Float32Array.from(values)).buffer
      );
      return view;
    };
    const outputs = {
      varianceInflationFactors: port('float32', Array(18).fill(-999)),
      conditionNumbers: port('float32', Array(9).fill(-999)),
      designConditionNumbers: port('float32', Array(9).fill(-999)),
      varianceDecomposition: port('float32', Array(27).fill(-999)),
      status: port('uint32', Array(9).fill(999))
    };
    const offsets = [0];
    for (const neighbors of LOCAL_AUDIT_NEIGHBORS)
      offsets.push(offsets[offsets.length - 1] + neighbors.length);
    const loss = port('uint32', [0]);
    const nodes = new GPULocalCollinearity({
      predictorCount: 2,
      predictors: port('float32', fixture.predictors),
      mask: port('uint32', [1, 1, 1, 1, 1, 1, 1, 1, 0]),
      weights: {
        offsets: port('uint32', offsets),
        neighbors: port('uint32', LOCAL_AUDIT_NEIGHBORS.flat()),
        weights: port('float32', LOCAL_AUDIT_WEIGHTS.flat()),
        status: {candidateOverflow: loss}
      },
      output: outputs
    }).getCommandNodes(graph);
    try {
      for (const incomplete of [false, true, false]) {
        new Uint32Array(memory.get(loss.buffer)!)[0] = Number(incomplete);
        for (const node of nodes) {
          const record = getWGSLKernelNodeRecord(graph, node.id)!;
          const bindings = Object.fromEntries(
            record.bindings.map((binding, index) => {
              if (!memory.has(binding.view.buffer))
                memory.set(binding.view.buffer, new ArrayBuffer(binding.view.buffer.byteLength));
              const range = getViewBindingRange(binding.view);
              return [
                index,
                new Uint8Array(memory.get(binding.view.buffer)!, range.offset, range.size)
              ];
            })
          );
          // The interpreter aliases scalar array reads held by `let`, whereas WGSL snapshots
          // their values. Materialize only those snapshots before Givens/Jacobi overwrites.
          const source = record.source
            .replace(/^\s*_\s*=.*;$/gm, '')
            .replace(/@workgroup_size\((\d+)\)/, '@workgroup_size($1, 1, 1)')
            .replace(
              /let (held|incoming|x|y|vectorFirst|vectorSecond) = ([^;]+);/g,
              'let $1 = ($2) + 0.0;'
            );
          new WgslExec(new WgslParser().parse(source)).dispatchWorkgroups(
            'main',
            [...record.workgroups],
            {0: bindings},
            {constants: record.constants}
          );
        }
        const differences: Record<string, number> = {};
        for (const [name, view] of Object.entries(outputs)) {
          const actual =
            view.format === 'uint32'
              ? new Uint32Array(memory.get(view.buffer)!)
              : new Float32Array(memory.get(view.buffer)!);
          const expected = fixture[name as keyof typeof outputs];
          differences[name] = 0;
          for (let index = 0; index < expected.length; index++) {
            const value = incomplete ? (name === 'status' ? 1 : NaN) : expected[index];
            if (Number.isNaN(value)) expect(actual[index], `${name}[${index}]`).toBeNaN();
            else {
              expect(Number.isFinite(actual[index]), `${name}[${index}]`).toBe(true);
              differences[name] = Math.max(differences[name], Math.abs(actual[index] - value));
              expect(
                Math.abs(actual[index] - value),
                `${name}[${index}]: ${actual[index]} vs ${value}`
              ).toBeLessThanOrEqual(name === 'status' ? 0 : 3e-4 * Math.max(1, Math.abs(value)));
            }
          }
        }
        if (!incomplete)
          console.info('Emitted local collinearity maximum absolute differences', differences);
      }
    } finally {
      device.destroy();
    }
  });
}

it('preserves local collinearity shapes and bindings after caller mutation', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const props = {
    predictorCount: 2,
    predictors: createTransientView(graph, 'predictors', 'float32', 18),
    weights: {
      offsets: createTransientView(graph, 'offsets', 'uint32', 10),
      neighbors: createTransientView(graph, 'neighbors', 'uint32', 20),
      weights: createTransientView(graph, 'weights', 'float32', 20)
    },
    output: {
      varianceInflationFactors: createTransientView(graph, 'inflation', 'float32', 18),
      conditionNumbers: createTransientView(graph, 'condition', 'float32', 9),
      status: createTransientView(graph, 'status', 'uint32', 9)
    }
  };
  const contributor = new GPULocalCollinearity(props);
  const offsets = props.weights.offsets;
  const status = props.output.status;
  props.predictorCount = 1;
  props.weights.offsets = createTransientView(graph, 'changed-offsets', 'uint32', 2);
  props.output.status = createTransientView(graph, 'changed-status', 'uint32', 1);
  expect(contributor.props.predictorCount).toBe(2);
  expect(contributor.props.weights).toHaveProperty('offsets', offsets);
  expect(contributor.props.output.status).toBe(status);
  device.destroy();
});

it('declares suppression when neighborhood producers report lost candidates', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const nodes = new GPULocalCollinearity({
    predictorCount: 1,
    predictors: createTransientView(graph, 'predictors', 'float32', 8),
    weights: {
      offsets: createTransientView(graph, 'offsets', 'uint32', 9),
      neighbors: createTransientView(graph, 'neighbors', 'uint32', 8),
      weights: createTransientView(graph, 'weights', 'float32', 8),
      status: {candidateOverflow: createTransientView(graph, 'lost-candidates', 'uint32', 1)}
    },
    output: {
      varianceInflationFactors: createTransientView(graph, 'inflation', 'float32', 8),
      conditionNumbers: createTransientView(graph, 'condition', 'float32', 8),
      status: createTransientView(graph, 'status', 'uint32', 8)
    }
  }).getCommandNodes(graph);
  expect(nodes.some(node => node.id.includes('incomplete'))).toBe(true);
  device.destroy();
});

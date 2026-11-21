// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph, createTransientView, type GPUCPUVector} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {CPURadiationFlowModelOperation} from '../../../src/gpu-network/flow-aggregation/cpu-radiation-flow-model-operation';
import {
  GPURadiationFlowModelOperation,
  type RadiationFlowModelOperationProps
} from '../../../src/gpu-network/flow-aggregation/gpu-radiation-flow-model-operation';
import type {GPURadiationFlowModel} from '../../../src/gpu-network/flow-aggregation/gpu-radiation-flow-model';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {auditWGSLModule} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';

function getProps(): RadiationFlowModelOperationProps {
  return {
    id: 'radiation-evidence',
    zoneCount: 2,
    positions: 'positions',
    populations: 'populations',
    outflows: 'outflows',
    normalization: 'finite-size',
    spatialContext: {coordinateSpace: 'planar', metric: 'native', units: 'meters'},
    output: {flows: 'flows', invalidInput: 'invalid'}
  };
}
function getInputs(coordinate = 1): Record<string, GPUCPUVector> {
  const vector = (format: GPUCPUVector['format'], values: Float32Array): GPUCPUVector => {
    const length = values.length / (format === 'float32x2' ? 2 : 1);
    return {format, length, chunks: [{values, length}], pack: () => values};
  };
  return {
    positions: vector('float32x2', Float32Array.of(0, 0, coordinate, 0)),
    populations: vector('float32', Float32Array.of(5, 1)),
    outflows: vector('float32', Float32Array.of(12, 9))
  };
}
function mutateProps(props: RadiationFlowModelOperationProps) {
  props.positions = 'changed-positions';
  props.populations = 'changed-populations';
  props.outflows = 'changed-outflows';
  props.zoneCount = 3;
  props.normalization = 'none';
  props.spatialContext!.metric = 'none';
  props.output.flows = 'changed-flows';
  props.output.invalidInput = 'changed-invalid';
}

it.each([
  'none',
  'finite-size'
] as const)('DGN-05 declares, lowers and audits diagnostic shaders with captured options: %s', normalization => {
  const props = {...getProps(), normalization};
  const operation = new GPURadiationFlowModelOperation(props);
  expect(operation.outputs['invalidInput']).toEqual({
    value: 'invalid',
    format: 'uint32',
    length: 2
  });
  mutateProps(props);
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const views = {
    positions: createTransientView(graph, 'positions', 'float32x2', 2),
    populations: createTransientView(graph, 'populations', 'float32', 2),
    outflows: createTransientView(graph, 'outflows', 'float32', 2),
    flows: createTransientView(graph, 'flows', 'float32', 4),
    invalidInput: createTransientView(graph, 'invalid', 'uint32', 2)
  };
  try {
    expect(operation.inputs).toEqual({
      positions: 'positions',
      populations: 'populations',
      outflows: 'outflows'
    });
    const contributor = operation.createContributor(
      port => views[port as keyof typeof views] as never
    ) as GPURadiationFlowModel;
    expect(contributor.props.normalization).toBe(normalization);
    expect(contributor.props.spatialContext?.metric).toBe('native');
    expect(contributor.props.output.status?.invalidInput).toBe(views.invalidInput);
    const kernels = getGPUCommandNodeKernelSources(contributor.getCommandNodes(graph));
    expect(kernels.length).toBeGreaterThan(0);
    for (const kernel of kernels)
      expect.soft(auditWGSLModule(kernel.source, kernel.constants), kernel.id).toEqual([]);
  } finally {
    operation.destroy();
    device.destroy();
  }
});

it('DGN-05 CPU operation writes zero diagnostics for valid inputs and preserves zone rejection', async () => {
  const props = getProps();
  const operation = new CPURadiationFlowModelOperation(props);
  expect(operation.outputs['invalid']).toBeDefined();
  mutateProps(props);
  const result = await operation.execute({inputs: getInputs()});
  // Each origin has exactly one destination: finite-size radiation assigns its full outflow.
  expect(result['flows']).toEqual(Float32Array.of(0, 12, 9, 0));
  expect(result['invalid']).toEqual(Uint32Array.of(0, 0));
  expect(() => operation.execute({inputs: getInputs(NaN)})).toThrow(/positions.*zone 1/);
});

it('DGN-05 captures inherited operation getters once in both forms', () => {
  for (const Operation of [GPURadiationFlowModelOperation, CPURadiationFlowModelOperation]) {
    const original = getProps();
    const reads: Record<string, number> = {};
    const inherited = {};
    for (const field of Object.keys(original) as (keyof RadiationFlowModelOperationProps)[]) {
      Object.defineProperty(inherited, field, {
        get() {
          reads[field] = (reads[field] ?? 0) + 1;
          return original[field];
        }
      });
    }
    const operation = new Operation(Object.create(inherited));
    expect(Object.values(reads)).toEqual(new Array(Object.keys(original).length).fill(1));
    expect(operation.inputs).toEqual({
      positions: 'positions',
      populations: 'populations',
      outflows: 'outflows'
    });
    if (operation instanceof GPURadiationFlowModelOperation) operation.destroy();
  }
});

it('DGN-05 rejects unknown normalization values before lowering in both forms', () => {
  for (const Operation of [GPURadiationFlowModelOperation, CPURadiationFlowModelOperation]) {
    expect(() => new Operation({...getProps(), normalization: 'unknown' as never})).toThrow(
      /radiation-evidence.*normalization.*none.*finite-size/
    );
  }
});

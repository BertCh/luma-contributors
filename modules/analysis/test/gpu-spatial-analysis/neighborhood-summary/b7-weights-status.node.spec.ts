// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {recordWGSLKernelNodes} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {
  computeNeighborhoodSummaryOnCPU,
  CPUNeighborhoodSummaryOperation
} from '../../../src/gpu-spatial-analysis/neighborhood-summary/cpu-neighborhood-summary-operation';
import {GPUNeighborhoodSummaryOperation} from '../../../src/gpu-spatial-analysis/neighborhood-summary/gpu-neighborhood-summary-operation';
import {GPUNeighborhoodSummary} from '../../../src/gpu-spatial-analysis/neighborhood-summary/gpu-neighborhood-summary';
import {SPATIAL_OPERATION_FORMS} from '../../../src/gpu-spatial-analysis/ontology/all-entry-points/spatial-operation-forms';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {createOperationTestViewGetter} from '../../utils/statistics-dynamics-operation-test-utils';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';

const NAMED_PROPS = {
  id: 'summary-status',
  rowCount: 2,
  weights: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
  values: 'values',
  statistics: ['mean'] as const,
  output: {table: 'table'}
};
const UNSUPPORTED_LOSS_FIELDS = ['unresolvedCount', 'invalidCount', 'approximation'] as const;
const STOPPING_CASES = [
  {field: 'converged', port: 'weightsConverged', clear: 1, loss: 0},
  {field: 'iterationLimitReached', port: 'weightsIterationLimitReached', clear: 0, loss: 1}
] as const;
const CPU_INPUT = {
  offsets: Uint32Array.of(0, 1, 2),
  neighbors: Uint32Array.of(1, 0),
  weights: Float32Array.of(1, 1),
  values: Float32Array.of(10, 20)
};

function withDirectProps(run: (graph: GPUCommandGraph) => void): void {
  const device = createNullWebGPUDevice();
  try {
    run(new GPUCommandGraph(device));
  } finally {
    device.destroy();
  }
}

function getDirectProps(graph: GPUCommandGraph) {
  return {
    id: NAMED_PROPS.id,
    weights: {
      offsets: createTransientView(graph, 'offsets', 'uint32', 3),
      neighbors: createTransientView(graph, 'neighbors', 'uint32', 2),
      weights: createTransientView(graph, 'weights', 'float32', 2)
    },
    values: createTransientView(graph, 'values', 'float32', 2),
    statistics: NAMED_PROPS.statistics,
    output: {table: createTransientView(graph, 'table', 'float32', 2)}
  };
}

it.each(UNSUPPORTED_LOSS_FIELDS)('GPU summary rejects %s at matrix and chunk level', field => {
  withDirectProps(graph => {
    const props = getDirectProps(graph);
    const status = {[field]: createTransientView(graph, field, 'uint32', 1)};
    for (const weights of [
      {...props.weights, status},
      {...props.weights, chunks: [{weights: {status}}]}
    ]) {
      expect(() => new GPUNeighborhoodSummary({...props, weights})).toThrow(
        new RegExp(`summary-status.*weights.*status\\.${field}.*overflow and candidateOverflow`)
      );
    }
  });
});

it('GPU summary rejects a prototype getter for complete at construction', () => {
  withDirectProps(graph => {
    const props = getDirectProps(graph);
    const certificate = createTransientView(graph, 'complete', 'uint32', 1);
    class Status {
      get complete() {
        return certificate;
      }
    }
    expect(
      () =>
        new GPUNeighborhoodSummary({...props, weights: {...props.weights, status: new Status()}})
    ).toThrow(/summary-status.*weights.status.complete/);
  });
});

it.each(['own property', 'prototype getter'])('GPU summary checks chunks from %s', source => {
  withDirectProps(graph => {
    const props = getDirectProps(graph);
    const status = {unresolvedCount: createTransientView(graph, 'unresolved', 'uint32', 1)};
    const chunks = [{weights: {status}}];
    class Weights {
      readonly offsets = props.weights.offsets;
      readonly neighbors = props.weights.neighbors;
      readonly weights = props.weights.weights;
      get chunks() {
        return chunks;
      }
    }
    const weights = source === 'own property' ? {...props.weights, chunks} : new Weights();
    expect(() => new GPUNeighborhoodSummary({...props, weights})).toThrow(
      /weights\.chunks\[0\]\.weights\.status\.unresolvedCount/
    );
  });
});

it('GPU summary captures a supported prototype getter once and wires its loss node', () => {
  withDirectProps(graph => {
    const props = getDirectProps(graph);
    const loss = createTransientView(graph, 'candidate-loss', 'uint32', 1);
    let reads = 0;
    class Status {
      get candidateOverflow() {
        reads++;
        return loss;
      }
    }
    const summary = new GPUNeighborhoodSummary({
      ...props,
      weights: {...props.weights, status: new Status()}
    });
    expect(summary.props.weights.status?.candidateOverflow).toBe(loss);
    expect(summary.getCommandNodes(graph).map(node => node.id)).toContain(
      'summary-status-weights-loss-0'
    );
    expect(reads).toBe(1);
  });
});

for (const Operation of [GPUNeighborhoodSummaryOperation, CPUNeighborhoodSummaryOperation]) {
  it.each([
    'unresolvedCount',
    'overflow',
    'candidateOverflow',
    ...STOPPING_CASES.map(({field}) => field)
  ])(`${Operation.name} rejects child %s evidence before binding`, field => {
    const weights = {
      ...NAMED_PROPS.weights,
      chunks: [{weights: {status: {[field]: 'child-status'}}}]
    };
    expect(() => new Operation({...NAMED_PROPS, weights})).toThrow(
      new RegExp(`weights\\.chunks\\[0\\]\\.weights\\.status\\.${field}`)
    );
  });

  it.each([
    ...UNSUPPORTED_LOSS_FIELDS,
    'requiredCount'
  ])(`${Operation.name} rejects %s before binding inputs`, field => {
    const status = {overflow: undefined, [field]: 'producer-status'};
    expect(
      () => new Operation({...NAMED_PROPS, weights: {...NAMED_PROPS.weights, status}})
    ).toThrow(
      new RegExp(`summary-status.*weights\\.status\\.${field}.*overflow and candidateOverflow`)
    );
  });

  it(`${Operation.name} rejects a prototype getter for complete at construction`, () => {
    class Status {
      get complete() {
        return 'weights-complete';
      }
    }
    expect(
      () => new Operation({...NAMED_PROPS, weights: {...NAMED_PROPS.weights, status: new Status()}})
    ).toThrow(/summary-status.*weights.status.complete/);
  });

  it(`${Operation.name} captures a supported prototype getter once as an input`, () => {
    let reads = 0;
    class Status {
      get candidateOverflow() {
        reads++;
        return 'candidate-loss';
      }
    }
    const operation = new Operation({
      ...NAMED_PROPS,
      weights: {...NAMED_PROPS.weights, status: new Status()}
    });
    expect(operation.inputs['weightsCandidateOverflow']).toBe('candidate-loss');
    expect(reads).toBe(1);
  });
}

it.each(UNSUPPORTED_LOSS_FIELDS)('pure CPU summary rejects %s', field => {
  // Each row has one neighbor, so ignoring the supplied evidence would give means [8, 3].
  expect(() =>
    computeNeighborhoodSummaryOnCPU(
      {
        offsets: Uint32Array.of(0, 1, 2),
        neighbors: Uint32Array.of(1, 0),
        weights: Float32Array.of(1, 1),
        values: Float32Array.of(3, 8),
        status: {overflow: undefined, [field]: Uint32Array.of(1)}
      },
      {statistics: ['mean']}
    )
  ).toThrow(new RegExp(`computeNeighborhoodSummaryOnCPU.*weights\\.status\\.${field}`));
});

it.each([
  ['overflow', 'weightsOverflow'],
  ['candidateOverflow', 'weightsCandidateOverflow']
] as const)('catalog names the working pure CPU capacity flag status.%s', (field, retiredField) => {
  const forms =
    SPATIAL_OPERATION_FORMS[
      '@luma-contributors/analysis/gpu-spatial-analysis:GPUNeighborhoodSummary'
    ];
  const differences = forms?.arrays?.find(
    form => form.exportName === 'computeNeighborhoodSummaryOnCPU'
  )?.differences;
  const description = differences === 'unknown' ? undefined : differences?.join('\n');
  expect(description).toContain(`\`status.${field}\``);
  expect(description).not.toContain(`\`${retiredField}\``);
  const flag = Uint32Array.of(0);
  const input = {...CPU_INPUT, status: {overflow: undefined, [field]: flag}};
  expect(Array.from(computeNeighborhoodSummaryOnCPU(input, NAMED_PROPS).table)).toEqual([20, 10]);
  flag[0] = 1;
  expect(Array.from(computeNeighborhoodSummaryOnCPU(input, NAMED_PROPS).table)).toEqual([NaN, NaN]);
  expect(() =>
    computeNeighborhoodSummaryOnCPU({...CPU_INPUT, [retiredField]: flag}, NAMED_PROPS)
  ).toThrow(`use weights.status.${field}`);
});

it.each(STOPPING_CASES)('pure CPU summary reads matrix status.$field', ({field, clear, loss}) => {
  const stopping = Uint32Array.of(clear);
  const input = {...CPU_INPUT, status: {overflow: Uint32Array.of(0), [field]: stopping}};
  // Each row's only neighbor is the other row, giving means [20, 10] with clear status.
  for (const [value, expected] of [
    [clear, [20, 10]],
    [loss, [NaN, NaN]],
    [clear, [20, 10]]
  ] as const) {
    stopping[0] = value;
    expect(Array.from(computeNeighborhoodSummaryOnCPU(input, NAMED_PROPS).table)).toEqual(expected);
  }
});

it.each(STOPPING_CASES)('pure CPU summary rejects child status.$field', ({field, loss}) => {
  const status = {overflow: Uint32Array.of(0), [field]: Uint32Array.of(loss)};
  const input = {...CPU_INPUT, chunks: [{weights: {status}}]};
  expect(() => computeNeighborhoodSummaryOnCPU(input, NAMED_PROPS)).toThrow(
    new RegExp(`weights\\.chunks\\[0\\]\\.weights\\.status\\.${field}.*weights\\.status\\.${field}`)
  );
});

it.each(STOPPING_CASES)('both named summaries bind and apply matrix status.$field', ({
  field,
  port,
  clear,
  loss
}) => {
  withDirectProps(graph => {
    recordWGSLKernelNodes(graph);
    const props = {
      ...NAMED_PROPS,
      weights: {...NAMED_PROPS.weights, status: {overflow: undefined, [field]: 'stopping'}}
    };
    const gpu = new GPUNeighborhoodSummaryOperation(props);
    const cpu = new CPUNeighborhoodSummaryOperation(props);
    expect(gpu.inputs[port]).toBe('stopping');
    expect(cpu.inputs[port]).toBe('stopping');

    const getView = createOperationTestViewGetter(graph, {
      offsets: {format: 'uint32', length: 3},
      neighbors: {format: 'uint32', length: 2},
      weights: {format: 'float32', length: 2},
      values: {format: 'float32', length: 2},
      [port]: {format: 'uint32', length: 1},
      table: {format: 'float32', length: 2}
    });
    const nodes = gpu.createContributor(getView).getCommandNodes(graph);
    const memory = createCPUGraphMemory();
    for (const [input, values] of Object.entries(CPU_INPUT)) memory.write(getView(input), values);
    const rows = (values: Float32Array | Uint32Array) => ({
      format: values instanceof Float32Array ? ('float32' as const) : ('uint32' as const),
      length: values.length,
      chunks: [{values, length: values.length}],
      pack: () => values
    });
    for (const [value, expected] of [
      [clear, [20, 10]],
      [loss, [NaN, NaN]]
    ] as const) {
      const stopping = Uint32Array.of(value);
      memory.write(getView(port), stopping);
      executeCommandNodesOnCPU(nodes, memory);
      expect(Array.from(memory.readFloat32(getView('table')))).toEqual(expected);
      const result = cpu.execute({
        inputs: {
          offsets: rows(CPU_INPUT.offsets),
          neighbors: rows(CPU_INPUT.neighbors),
          weights: rows(CPU_INPUT.weights),
          values: rows(CPU_INPUT.values),
          [port]: rows(stopping)
        }
      }) as Record<string, Float32Array>;
      expect(Array.from(result['table'])).toEqual(expected);
    }
  });
});

// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {executePanelRound1NodesOnCPU} from './panel-round1-replay.test-utils';
import {computeSpatialPanelLagMaximumLikelihoodOnCPU} from '../../../src/gpu-spatial-analysis/spatial-regression/cpu-spatial-panel-lag-maximum-likelihood-operation';
import {CPUSpatialPanelErrorMaximumLikelihoodOperation} from '../../../src/gpu-spatial-analysis/spatial-regression/cpu-spatial-panel-error-maximum-likelihood-operation';
import {GPUSpatialPanelErrorMaximumLikelihoodOperation} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-panel-error-maximum-likelihood-operation';
import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import {GPUSpatialLagMaximumLikelihood} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-lag-maximum-likelihood';
import {GPUSpatialPanelErrorMaximumLikelihood} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-panel-error-maximum-likelihood';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {createPanelAuditInput} from './panel-audit.test-utils';

for (const scale of [1, 2 ** 25])
  it(`B1 keeps nonzero tiny slope proposals scale=${scale}`, () => {
    const response = Float32Array.from(
      {length: 12},
      (_, row) => (3 * row * row + 2 * row + 1) % 11
    );
    const predictors = Float32Array.from({length: 24}, (_, index) => {
      const row = Math.floor(index / 2),
        neighbor = row - (row % 4) + [1, 0, 3, 2][row % 4];
      const z = (row * row + row + 4) % 7;
      return index % 2 ? z : scale * (response[row] - 0.5 * response[neighbor] - 0.25 * z);
    });
    const actual = computeSpatialPanelLagMaximumLikelihoodOnCPU(
      {
        response,
        predictors,
        weights: {
          offsets: Uint32Array.of(0, 1, 2, 3, 4),
          neighbors: Uint32Array.of(1, 0, 3, 2),
          weights: Float32Array.of(1, 1, 1, 1)
        }
      },
      {periodCount: 3, predictorCount: 2, effects: 'unit'}
    );
    // Exact original relation y=x/scale+.25*z+.5*Wy, unchanged by predictor units.
    expect(actual.status[0], String(actual.summary[1])).toBe(2);
  });

it('B2 named panel error captures and consumes the optional positive certificate', async () => {
  const scene = createPanelAuditInput();
  const props = {
    entityCount: 4,
    periodCount: 4,
    predictorCount: 1,
    maximumPower: 3,
    predictors: 'predictors',
    response: 'response',
    weights: {
      offsets: 'offsets',
      neighbors: 'neighbors',
      weights: 'weights',
      status: {complete: 'complete'}
    },
    output: {table: 'table', summary: 'summary', status: {code: 'status'}},
    diagnostics: {tests: 'tests', status: 'diagnosticStatus'}
  };
  const gpu = new GPUSpatialPanelErrorMaximumLikelihoodOperation(props);
  expect(gpu.inputs['complete']).toBe('complete');
  const operation = new CPUSpatialPanelErrorMaximumLikelihoodOperation(props);
  for (const complete of [1, 0, 1]) {
    const values = {
      ...scene.weights,
      predictors: scene.predictors,
      response: scene.response,
      complete: Uint32Array.of(complete)
    };
    const result = await operation.execute({
      inputs: Object.fromEntries(
        Object.entries(values).map(([name, values]) => [
          name,
          {
            format: values instanceof Float32Array ? 'float32' : 'uint32',
            length: values.length,
            chunks: [{values, length: values.length}],
            pack: () => values
          }
        ])
      )
    });
    expect(Array.from(result['status'] as Uint32Array)).toEqual([complete ? 0 : 3]);
    expect(Array.from(result['diagnosticStatus'] as Uint32Array)).toEqual([complete ? 0 : 1]);
  }
});

it('B3 positive panel-lag certificate overrides accurate accessible counts', () => {
  const input = createPanelAuditInput();
  input.weights.offsets = Uint32Array.of(2, 4, 6, 8, 10);
  input.weights.neighbors = Uint32Array.of(0, 0, ...input.weights.neighbors);
  input.weights.weights = Float32Array.of(0, 0, ...input.weights.weights);
  for (const complete of [1, 0, 1]) {
    const actual = computeSpatialPanelLagMaximumLikelihoodOnCPU(
      {
        ...input,
        weights: {
          ...input.weights,
          status: {
            complete: Uint32Array.of(complete),
            count: Uint32Array.of(8),
            requiredCount: Uint32Array.of(8)
          }
        }
      },
      {periodCount: 4, predictorCount: 1, effects: 'random', hausman: true}
    );
    expect(actual.status[0]).toBe(complete ? 0 : 1);
    if (!complete) {
      expect(actual.hausman!.status[0]).toBe(1);
      expect(actual.hausman!.tests).toHaveLength(3);
      expect(Array.from(actual.hausman!.tests).every(Number.isNaN)).toBe(true);
    }
  }
});

it('B5 retains the previous 12-million-row binding boundary', () => {
  const device = createNullWebGPUDevice();
  try {
    Object.defineProperty(device.limits, 'maxStorageBufferBindingSize', {
      value: 128 * 1024 ** 2,
      configurable: true
    });
    const graph = new GPUCommandGraph(device);
    let serial = 0;
    const view = <Format extends 'float32' | 'uint32'>(format: Format, length: number) =>
      createTransientView(graph, `large-${serial++}`, format, length);
    const rows = 12000000;
    expect(() =>
      new GPUSpatialLagMaximumLikelihood({
        predictorCount: 1,
        maximumPower: 2,
        traceMode: 'stochastic',
        traceProbeCount: 1,
        predictors: view('float32', rows),
        response: view('float32', rows),
        weights: {
          offsets: view('uint32', rows + 1),
          neighbors: view('uint32', 1),
          weights: view('float32', 1)
        },
        output: {
          table: view('float32', 12),
          summary: view('float32', 22),
          status: {code: view('uint32', 1)}
        }
      }).getCommandNodes(graph)
    ).not.toThrow();
  } finally {
    device.destroy();
  }
});

it('S6 reports the actual malformed panel spectrum length before the cap', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    let serial = 0;
    const view = <Format extends 'float32' | 'uint32'>(format: Format, length: number) =>
      createTransientView(graph, `length-${serial++}`, format, length);
    expect(
      () =>
        new GPUSpatialPanelErrorMaximumLikelihood({
          periodCount: 513,
          predictorCount: 1,
          logDeterminant: 'exact',
          eigenvalues: view('float32', 3),
          predictors: view('float32', 1026),
          response: view('float32', 1026),
          weights: {
            offsets: view('uint32', 3),
            neighbors: view('uint32', 2),
            weights: view('float32', 2)
          },
          output: {
            table: view('float32', 12),
            summary: view('float32', 22),
            status: {code: view('uint32', 1)}
          }
        })
    ).toThrow(/eigenvalues length 3/);
  } finally {
    device.destroy();
  }
});

import {
  getWGSLKernelNodeRecord,
  recordWGSLKernelNodes
} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
it('S4 repeated spectra reduce scalar row certificates before the scalar likelihood search', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    recordWGSLKernelNodes(graph);
    let serial = 0;
    const view = <Format extends 'float32' | 'uint32'>(format: Format, length: number) =>
      createTransientView(graph, `scalar-${serial++}`, format, length);
    const nodes = new GPUSpatialPanelErrorMaximumLikelihood({
      periodCount: 513,
      predictorCount: 1,
      logDeterminant: 'exact',
      eigenvalues: view('float32', 2),
      predictors: view('float32', 1026),
      response: view('float32', 1026),
      weights: {
        offsets: view('uint32', 3),
        neighbors: view('uint32', 2),
        weights: view('float32', 2)
      },
      output: {
        table: view('float32', 12),
        summary: view('float32', 22),
        status: {code: view('uint32', 1)}
      }
    }).getCommandNodes(graph);
    const records = nodes
      .map(node => getWGSLKernelNodeRecord(graph, node.id))
      .filter(record => record !== undefined);
    expect(records.some(record => record.source.includes('scalarDifferenceSumItem'))).toBe(false);
    expect(records.some(record => record.id.includes('scalar-certificate-sum-tiles'))).toBe(true);
  } finally {
    device.destroy();
  }
});

import {createCPUGraphMemory} from '../segment-intersection/wgsl-graph-executor';
import {getPanelDiagnosticNodes} from '../../../src/gpu-spatial-analysis/spatial-regression/panel-diagnostics';
for (const constant of [0, 6])
  it(`B4 original lag cancellation and equal nonzero sums remain constant ${constant}`, () => {
    const device = createNullWebGPUDevice();
    try {
      const graph = new GPUCommandGraph(device),
        memory = createCPUGraphMemory();
      recordWGSLKernelNodes(graph);
      let serial = 0;
      const view = <Format extends 'float32' | 'uint32'>(format: Format, length: number) =>
        createTransientView(graph, `cancel-${serial++}`, format, length);
      const data = {
        response: Float32Array.of(3, -3, 1, 1, -1, -1, 2),
        predictors: Float32Array.of(0, 0, 1, -1, -1, 1, 0),
        offsets: constant
          ? Uint32Array.of(0, 2, 4, 5, 6, 7, 8, 9)
          : Uint32Array.of(0, 2, 4, 4, 4, 4, 4, 4),
        neighbors: constant
          ? Uint32Array.of(0, 1, 0, 1, 0, 0, 0, 0, 0)
          : Uint32Array.of(0, 1, 0, 1),
        weights: constant ? Float32Array.of(5, 3, 3, 1, 2, 2, 2, 2, 2) : Float32Array.of(5, 5, 1, 1)
      };
      const props = {
        predictorCount: 1,
        maximumPower: 2,
        predictors: view('float32', 7),
        response: view('float32', 7),
        weights: {
          offsets: view('uint32', 8),
          neighbors: view('uint32', data.neighbors.length),
          weights: view('float32', data.weights.length)
        },
        output: {
          table: view('float32', 12),
          summary: view('float32', 22),
          status: {code: view('uint32', 1)}
        }
      };
      memory.write(props.response, data.response);
      memory.write(props.predictors, data.predictors);
      for (const name of ['offsets', 'neighbors', 'weights'] as const)
        memory.write(props.weights[name], data[name]);
      const nodes = new GPUSpatialLagMaximumLikelihood(props)
        .getCommandNodes(graph)
        .filter(node =>
          /-(means-tiles|means-merge|lagged-response-tiles|row-sum-merge|constant-lag-witness|constant-lag-tiles|constant-lag-merge|lagged-response-shift)/.test(
            node.id
          )
        );
      executePanelRound1NodesOnCPU(nodes, memory);
      const shift = getWGSLKernelNodeRecord(
        graph,
        nodes.find(node => node.id.endsWith('-lagged-response-shift'))!.id
      )!;
      const lags = memory.readFloat32(
        shift.bindings.find(binding => binding.name === 'lagged')!.view
      );
      expect(lags).toHaveLength(7);
      expect(Number.isFinite(lags[0])).toBe(true);
      expect(Array.from(lags)).toEqual(new Array(7).fill(lags[0]));
      const witness = memory.readFloat32(
        shift.bindings.find(binding => binding.name === 'lagWitness')!.view
      );
      expect(Array.from(witness)).toEqual(new Array(7).fill(constant));
    } finally {
      device.destroy();
    }
  });

it('B6 precise projection preserves all six near-rank diagnostics in both period orders', () => {
  const expected = [
    2.4832931471755644, 0.11506111315877793, 0.1924145021982694, 0.6609146001913988,
    0.19241497912743916, 0.6609142062220347
  ];
  for (const predictorScale of [1, 2 ** 100])
    for (const reversed of [false, true]) {
      const device = createNullWebGPUDevice();
      try {
        const graph = new GPUCommandGraph(device),
          memory = createCPUGraphMemory();
        recordWGSLKernelNodes(graph);
        let serial = 0;
        const view = <Format extends 'float32' | 'uint32'>(format: Format, length: number) =>
          createTransientView(graph, `precise-${serial++}`, format, length);
        const data = createPanelAuditInput();
        data.predictors = Float32Array.from(
          {length: 16},
          (_, row) => (Math.floor(row / 4) + 1) * (row % 2 ? -1 : 1)
        );
        data.predictors[0] += 100 * 2 ** -23;
        data.predictors = Float32Array.from(data.predictors, value => value * predictorScale);
        if (reversed) {
          const reverse = (values: Float32Array) =>
            Float32Array.from(
              {length: 16},
              (_, row) => values[(3 - Math.floor(row / 4)) * 4 + (row % 4)]
            );
          data.predictors = reverse(data.predictors);
          data.response = reverse(data.response);
        }
        const props = {
          periodCount: 4,
          predictorCount: 1,
          predictors: view('float32', 16),
          response: view('float32', 16),
          weights: {
            offsets: view('uint32', 5),
            neighbors: view('uint32', 8),
            weights: view('float32', 8)
          },
          output: {
            table: view('float32', 12),
            summary: view('float32', 22),
            status: {code: view('uint32', 1)}
          },
          diagnostics: {tests: view('float32', 6), status: view('uint32', 1)}
        };
        memory.write(props.predictors, data.predictors);
        memory.write(props.response, data.response);
        for (const name of ['offsets', 'neighbors', 'weights'] as const)
          memory.write(props.weights[name], data.weights[name]);
        const all = getPanelDiagnosticNodes(
          graph,
          new GPUSpatialPanelErrorMaximumLikelihood(props)
        );
        const publisher = getWGSLKernelNodeRecord(
          graph,
          all.find(node => node.id.endsWith('-diagnostic-tests'))!.id
        )!;
        for (const name of ['fitStatus', 'projectionStatus'])
          memory.write(
            publisher.bindings.find(binding => binding.name === name)!.view,
            Uint32Array.of(0)
          );
        const summary = new Float32Array(22);
        summary[0] = 16;
        summary[3] = 9.354763019274625;
        memory.write(publisher.bindings.find(binding => binding.name === 'summary')!.view, summary);
        const nodes = all.filter(
          node =>
            node.id.endsWith('-diagnostic-lags') ||
            node.id.includes('-diagnostic-precise-') ||
            node.id.includes('-diagnostic-sums-') ||
            node.id.endsWith('-diagnostic-tests')
        );
        executePanelRound1NodesOnCPU(nodes, memory);
        expect(Array.from(memory.readUint32(props.diagnostics.status))).toEqual([0]);
        const actual = memory.readFloat32(props.diagnostics.tests);
        expect(actual).toHaveLength(6);
        for (let slot = 0; slot < 6; slot++) {
          expect(Number.isFinite(actual[slot])).toBe(true);
          expect(
            actual[slot],
            `reversed=${reversed}, predictorScale=${predictorScale}, slot=${slot}`
          ).toBeCloseTo(expected[slot], 3);
        }
      } finally {
        device.destroy();
      }
    }
});

import {fp64arithmetic} from '@luma.gl/shaders';
import {createWGSLKernelNode} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
it('B6 published double-single arithmetic retains the perturbation before projection', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device),
      memory = createCPUGraphMemory();
    const input = createTransientView(graph, 'ds-input', 'float32', 2),
      output = createTransientView(graph, 'ds-output', 'float32', 4);
    memory.write(input, Float32Array.of(1 + 100 * 2 ** -23, -1));
    const node = createWGSLKernelNode(graph, {
      id: 'ds-probe',
      operation: 'DSProbe',
      invocationCount: 1,
      modules: [fp64arithmetic],
      bindings: [
        {name: 'input', view: input, type: 'f32', access: 'read'},
        {name: 'output', view: output, type: 'f32', access: 'read_write'}
      ],
      body: 'let value=sub_fp64(vec2f(input[inputOffset+1u],0.0),vec2f(input[inputOffset],0.0));let total=sum_fp64(value,value);output[outputOffset]=value.x;output[outputOffset+1u]=value.y;output[outputOffset+2u]=total.x;output[outputOffset+3u]=total.y;'
    });
    executePanelRound1NodesOnCPU([node], memory);
    const actual = memory.readFloat32(output);
    expect(actual[0] + actual[1], JSON.stringify(Array.from(actual))).toBe(-2 - 100 * 2 ** -23);
    expect(actual[2] + actual[3]).toBe(-4 - 200 * 2 ** -23);
  } finally {
    device.destroy();
  }
});

it('B7 repetition is private composition metadata and explicit panel repetition props are rejected', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    let serial = 0;
    const view = <Format extends 'float32' | 'uint32'>(format: Format, length: number) =>
      createTransientView(graph, `private-spectrum-${serial++}`, format, length);
    const props = {
      periodCount: 3,
      predictorCount: 1,
      predictors: view('float32', 12),
      response: view('float32', 12),
      weights: {
        offsets: view('uint32', 5),
        neighbors: view('uint32', 4),
        weights: view('float32', 4)
      },
      output: {
        table: view('float32', 12),
        summary: view('float32', 22),
        status: {code: view('uint32', 1)}
      },
      exactSpectrumRepetitionCount: 0
    };
    expect(() => new GPUSpatialPanelErrorMaximumLikelihood(props)).toThrow(/not a panel prop/);
    const named = {
      entityCount: 4,
      periodCount: 3,
      predictorCount: 1,
      predictors: 'x',
      response: 'y',
      weights: {offsets: 'o', neighbors: 'n', weights: 'w'},
      output: {table: 'table', summary: 'summary', status: {code: 'code'}},
      exactSpectrumRepetitionCount: 0
    };
    expect(() => new GPUSpatialPanelErrorMaximumLikelihoodOperation(named)).toThrow(
      /not a panel prop/
    );
    expect(() => new CPUSpatialPanelErrorMaximumLikelihoodOperation(named)).toThrow(
      /not a panel prop/
    );
  } finally {
    device.destroy();
  }
});

import {
  PANEL_DIAGNOSTIC_SUM_COUNT,
  getPanelPreciseDiagnosticAllocationLengths
} from '../../../src/gpu-spatial-analysis/spatial-regression/panel-precise-diagnostics';
it('S2 diagnostic preflight and writer use one six-term layout', () => {
  expect(PANEL_DIAGNOSTIC_SUM_COUNT).toBe(6);
  const lengths = getPanelPreciseDiagnosticAllocationLengths(1048576, 1);
  expect(lengths['diagnostic precise lag pairs']).toBe(2097152);
  expect(lengths['diagnostic precise partials']).toBe(40960);
});

it('B4 constant-lag witness retains a unit between cancelling 2^100 products', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device),
      memory = createCPUGraphMemory();
    recordWGSLKernelNodes(graph);
    let serial = 0;
    const view = <Format extends 'float32' | 'uint32'>(format: Format, length: number) =>
      createTransientView(graph, `wide-cancel-${serial++}`, format, length);
    const props = {
      predictorCount: 1,
      maximumPower: 2,
      predictors: view('float32', 3),
      response: view('float32', 3),
      weights: {
        offsets: view('uint32', 4),
        neighbors: view('uint32', 5),
        weights: view('float32', 5)
      },
      output: {
        table: view('float32', 12),
        summary: view('float32', 22),
        status: {code: view('uint32', 1)}
      }
    };
    memory.write(props.response, Float32Array.of(1, 1, -1));
    memory.write(props.weights.offsets, Uint32Array.of(0, 3, 4, 5));
    memory.write(props.weights.neighbors, Uint32Array.of(0, 1, 2, 0, 0));
    memory.write(props.weights.weights, Float32Array.of(2 ** 100, 1, 2 ** 100, 1, 1));
    const nodes = new GPUSpatialLagMaximumLikelihood(props)
      .getCommandNodes(graph)
      .filter(node => node.id.includes('-constant-lag-witness'));
    executePanelRound1NodesOnCPU(nodes, memory);
    const record = getWGSLKernelNodeRecord(graph, nodes[0].id)!;
    const witness = record.bindings.find(binding => binding.name === 'witness')!.view;
    expect(Array.from(memory.readFloat32(witness))).toEqual([1, 1, 1]);
  } finally {
    device.destroy();
  }
});

it('B6 precise multivariate projection agrees with independent NumPy diagnostics', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device),
      memory = createCPUGraphMemory();
    recordWGSLKernelNodes(graph);
    let serial = 0;
    const view = <Format extends 'float32' | 'uint32'>(format: Format, length: number) =>
      createTransientView(graph, `multivariate-${serial++}`, format, length);
    const data = createPanelAuditInput();
    const predictors = Float32Array.from({length: 32}, (_, index) =>
      index % 2 ? ((5 * Math.floor(index / 2)) % 11) - 5 : data.predictors[index / 2]
    );
    const props = {
      periodCount: 4,
      predictorCount: 2,
      predictors: view('float32', 32),
      response: view('float32', 16),
      weights: {
        offsets: view('uint32', 5),
        neighbors: view('uint32', 8),
        weights: view('float32', 8)
      },
      output: {
        table: view('float32', 16),
        summary: view('float32', 22),
        status: {code: view('uint32', 1)}
      },
      diagnostics: {tests: view('float32', 6), status: view('uint32', 1)}
    };
    memory.write(props.predictors, predictors);
    memory.write(props.response, data.response);
    for (const name of ['offsets', 'neighbors', 'weights'] as const)
      memory.write(props.weights[name], data.weights[name]);
    const nodes = getPanelDiagnosticNodes(graph, new GPUSpatialPanelErrorMaximumLikelihood(props));
    const publisher = getWGSLKernelNodeRecord(
      graph,
      nodes.find(node => node.id.endsWith('-diagnostic-tests'))!.id
    )!;
    for (const name of ['fitStatus', 'projectionStatus'])
      memory.write(
        publisher.bindings.find(binding => binding.name === name)!.view,
        Uint32Array.of(0)
      );
    const summary = new Float32Array(22);
    summary[0] = 16;
    summary[3] = 2.1460930183159377;
    memory.write(publisher.bindings.find(binding => binding.name === 'summary')!.view, summary);
    executePanelRound1NodesOnCPU(
      nodes.filter(
        node =>
          node.id.endsWith('-diagnostic-lags') ||
          node.id.includes('-diagnostic-precise-') ||
          node.id.includes('-diagnostic-sums-') ||
          node.id.endsWith('-diagnostic-tests')
      ),
      memory
    );
    expect(Array.from(memory.readUint32(props.diagnostics.status))).toEqual([0]);
    const actual = memory.readFloat32(props.diagnostics.tests);
    const expected = [
      0.2683167337682054, 0.6044631629194325, 1.376923328707157, 0.24062591231296848,
      0.16417006507176315, 0.6853459010381062
    ];
    expect(actual).toHaveLength(6);
    for (let index = 0; index < 6; index++) {
      expect(Number.isFinite(actual[index])).toBe(true);
      expect(actual[index]).toBeCloseTo(expected[index], 4);
    }
  } finally {
    device.destroy();
  }
});

// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph, GraphDataView, createTransientView} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  GPUGeographicallyTemporallyWeightedRegression,
  GPUMultiscaleGeographicallyTemporallyWeightedRegression,
  getGPUGTWRBandwidthSearchParameterValues,
  type GPUGeographicallyTemporallyWeightedRegressionProps
} from '../../../src/gpu-spatial-analysis/spatial-regression';

import {createGoldenSectionIntervalNodes} from '../../../src/gpu-spatial-analysis/spatial-regression/gwr-golden-section-nodes';
import {
  createGTWRBandwidthSearchNodes,
  createMGTWRComponentEvaluationNodes
} from '../../../src/gpu-spatial-analysis/spatial-regression/gtwr-bandwidth-search';
import {
  createWGSLDispatchGate,
  createWGSLKernelNode,
  recordWGSLKernelNodes,
  getWGSLKernelNodeRecord
} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {getSerialAuditBody} from './local-regression-serial-audit';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';
import fixture from '../../oracles/fixtures/gtwr-bandwidth-search.json';

function createScene(rows = 16) {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends 'float32' | 'float32x2' | 'uint32'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `search-test-${serial++}`, format, length);
  const status = () => ({
    iterationCount: view('uint32', 1),
    converged: view('uint32', 1),
    iterationLimitReached: view('uint32', 1)
  });
  const props: GPUGeographicallyTemporallyWeightedRegressionProps = {
    id: 'gtwr-search',
    positions: view('float32x2', rows),
    times: view('float32', rows),
    temporalContext: {
      unit: 'second',
      epoch: 0n,
      encoding: 'float32-relative',
      axis: {kind: 'irregular'}
    },
    predictors: view('float32', rows),
    predictorCount: 1,
    response: view('float32', rows),
    parameters: view('float32', 4),
    adaptive: true,
    bandwidthSearch: {parameters: view('float32', 6), maximumIterations: 1},
    output: {
      coefficients: view('float32', rows * 2),
      localStatus: view('uint32', rows),
      fitted: view('float32', rows),
      hatDiagonal: view('float32', rows),
      summary: view('float32', 12),
      residuals: view('float32', rows),
      selectedParameters: view('float32', 2),
      status: status()
    }
  };
  return {device, graph, view, status, props};
}

it('packs float32 search bounds and rejects invalid ranges and tolerances', () => {
  expect(
    Array.from(
      getGPUGTWRBandwidthSearchParameterValues({bandwidthBounds: [3, 8], timeScaleBounds: [0, 4]})
    )
  ).toEqual([3, 8, 0, 4, Math.fround(1e-3), Math.fround(1e-3)]);
  for (const bandwidthBounds of [
    [0, 2],
    [4, 2],
    [1, Infinity],
    [1, 1e100]
  ] as const) {
    expect(() =>
      getGPUGTWRBandwidthSearchParameterValues({bandwidthBounds, timeScaleBounds: [0, 1]})
    ).toThrow('bounds');
  }
  for (const settings of [
    {timeScaleBounds: [-1, 2] as const},
    {timeScaleBounds: [2, 1] as const},
    {timeScaleBounds: [0, 1] as const, absoluteTolerance: 0},
    {timeScaleBounds: [0, 1] as const, relativeTolerance: -1}
  ]) {
    expect(() =>
      getGPUGTWRBandwidthSearchParameterValues({bandwidthBounds: [1, 2], ...settings})
    ).toThrow('bounds');
  }
});

it('validates budgets, spatial context, packed bounds, output lengths and status at construction', () => {
  const scene = createScene();
  try {
    const maximum = new GPUGeographicallyTemporallyWeightedRegression({
      ...scene.props,
      bandwidthSearch: {...scene.props.bandwidthSearch!, maximumIterations: 8}
    });
    expect(maximum.estimatedCommandNodeCount).toBeLessThanOrEqual(32768);
    // The independent command ceiling applies even within the allowed iteration range.
    const allocationSizes = () =>
      Object.fromEntries(
        Object.entries(scene.graph)
          .filter(([, value]) => value instanceof Map || Array.isArray(value))
          .map(([name, value]) => [name, value instanceof Map ? value.size : value.length])
      );
    const buffersBefore = allocationSizes();
    expect(
      () =>
        new GPUGeographicallyTemporallyWeightedRegression({
          ...scene.props,
          bandwidthSearch: {...scene.props.bandwidthSearch!, maximumIterations: 32}
        })
    ).toThrow(/estimated command node count.*32768/);
    expect(allocationSizes()).toEqual(buffersBefore);

    for (const maximumIterations of [0, 33, 1.5]) {
      expect(
        () =>
          new GPUGeographicallyTemporallyWeightedRegression({
            ...scene.props,
            bandwidthSearch: {...scene.props.bandwidthSearch!, maximumIterations}
          })
      ).toThrow('gtwr-search bandwidthSearch.maximumIterations');
    }
    expect(
      () =>
        new GPUGeographicallyTemporallyWeightedRegression({
          ...scene.props,
          bandwidthSearch: {parameters: scene.view('float32', 5)}
        })
    ).toThrow('at least 6');
    expect(
      () =>
        new GPUGeographicallyTemporallyWeightedRegression({
          ...scene.props,
          spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'great-circle'}
        })
    ).toThrow('planar');
    expect(
      () =>
        new GPUGeographicallyTemporallyWeightedRegression({
          ...scene.props,
          output: {...scene.props.output, selectedParameters: scene.view('float32', 1)}
        })
    ).toThrow('at least 2');
    expect(
      () =>
        new GPUGeographicallyTemporallyWeightedRegression({
          ...scene.props,
          bandwidthSearch: undefined
        })
    ).toThrow('requires bandwidthSearch');
    expect(
      () =>
        new GPUMultiscaleGeographicallyTemporallyWeightedRegression({
          ...scene.props,
          iterationCount: 32,
          bandwidthSearch: {parameters: scene.view('float32', 6), maximumIterations: 32},
          parameters: scene.view('float32', 6),
          output: {
            coefficients: scene.view('float32', 32),
            fitted: scene.view('float32', 16),
            residuals: scene.view('float32', 16),
            localStatus: scene.view('uint32', 16),
            scores: scene.view('float32', 32)
          }
        })
    ).toThrow('estimated command node count');
  } finally {
    scene.device.destroy();
  }
});

it('bounds actual command counts and gates every candidate data pass', () => {
  const scene = createScene();
  try {
    for (const maximumIterations of [1, 2, 3, 8]) {
      const contributor = new GPUGeographicallyTemporallyWeightedRegression({
        ...scene.props,
        id: `gtwr-${maximumIterations}`,
        bandwidthSearch: {...scene.props.bandwidthSearch!, maximumIterations}
      });
      const nodes = contributor.getCommandNodes(scene.graph);
      expect(nodes.length).toBe(contributor.estimatedCommandNodeCount);
      const candidates = nodes.filter(node =>
        /evaluate-\d+-(fit|publish-fit-values|response-anchor-bounds|statistics-(partial|fold)|variance-(partial|summary))$/.test(
          node.id
        )
      );
      expect(candidates.length).toBeGreaterThan(0);
      const fitNodes = candidates.filter(node => node.id.endsWith('-fit'));
      expect(candidates.length).toBe(fitNodes.length * 7);
      for (const fit of fitNodes) {
        const prefix = fit.id.slice(0, -4);
        for (const suffix of [
          'fit',
          'publish-fit-values',
          'response-anchor-bounds',
          'statistics-partial',
          'statistics-fold',
          'variance-partial',
          'variance-summary'
        ]) {
          expect(candidates.some(node => node.id === `${prefix}-${suffix}`)).toBe(true);
        }
      }
      for (const node of candidates) {
        expect(node.condition, node.id).toMatchObject({source: 'gpu', mode: 'indirect'});
      }
    }
    const contributor = new GPUMultiscaleGeographicallyTemporallyWeightedRegression({
      ...scene.props,
      id: 'mgtwr',
      iterationCount: 2,
      parameters: scene.view('float32', 6),
      output: {
        coefficients: scene.view('float32', 32),
        fitted: scene.view('float32', 16),
        residuals: scene.view('float32', 16),
        localStatus: scene.view('uint32', 16),
        scores: scene.view('float32', 2),
        selectedParameters: scene.view('float32', 6),
        status: scene.status()
      }
    });
    expect(contributor.getCommandNodes(scene.graph).length).toBe(
      contributor.estimatedCommandNodeCount
    );
  } finally {
    scene.device.destroy();
  }
});

function replayIntegerSearch(lower: number, scores: number[]) {
  const scene = createScene();
  try {
    const memory = createCPUGraphMemory();
    const bounds = scene.view('float32', 6);
    const objectives = scene.view('float32', scores.length);
    memory.write(bounds, [lower, lower + scores.length - 1, 1, 1, 0.01, 0.005]);
    memory.write(objectives, scores);
    const search = createGoldenSectionIntervalNodes(scene.graph, {
      id: 'integer-replay',
      bounds,
      axis: 0,
      integer: true,
      maximumIterations: 12,
      evaluate: (value, score, enabled, id) => [
        createWGSLKernelNode(scene.graph, {
          id,
          operation: 'IntegerSearchOracle',
          invocationCount: 1,
          bindings: [
            {name: 'value', view: value, type: 'f32', access: 'read'},
            {name: 'objectives', view: objectives, type: 'f32', access: 'read'},
            {name: 'enabled', view: enabled, type: 'f32', access: 'read'},
            {name: 'score', view: score, type: 'f32', access: 'read_write'}
          ],
          body: `if (enabled[enabledOffset + 5u] < 0.5) { return; }
  score[scoreOffset] = objectives[objectivesOffset + u32(value[valueOffset]) - ${lower}u];
  score[scoreOffset + 1u] = 0.0;
  score[scoreOffset + 2u] = 1.0;
  score[scoreOffset + 3u] = 0.0;`
        })
      ]
    });
    executeCommandNodesOnCPU(search.nodes, memory);
    return memory.readFloat32(search.result);
  } finally {
    scene.device.destroy();
  }
}

it('S2 each candidate owns correctly shaped summary gates across the 4096-row boundary', () => {
  for (const rows of [4096, 4097]) {
    const scene = createScene(rows);
    recordWGSLKernelNodes(scene.graph);
    try {
      const contributor = new GPUGeographicallyTemporallyWeightedRegression({
        ...scene.props,
        output: {...scene.props.output, selectedParameters: undefined}
      });
      const nodes = contributor.getCommandNodes(scene.graph);
      expect(nodes.length).toBe(rows === 4096 ? 223 : 240);
      expect(nodes.length).toBe(contributor.estimatedCommandNodeCount);
      const memory = createCPUGraphMemory();
      const candidateBuffers = new Set();
      for (const fit of nodes.filter(node => /evaluate-\d+-fit$/.test(node.id))) {
        const prefix = fit.id.slice(0, -4);
        const arm = nodes.find(node => node.id === `${prefix}-summary-arm`)!;
        const armRecord = getWGSLKernelNodeRecord(scene.graph, arm.id)!;
        const parent = armRecord.bindings.find(binding => binding.name === 'parent')!.view;
        const dispatch = armRecord.bindings.find(binding => binding.name === 'dispatch')!.view;
        expect(fit.condition).toMatchObject({buffer: parent.buffer, byteOffset: parent.byteOffset});
        expect(candidateBuffers.has(dispatch.buffer)).toBe(false);
        candidateBuffers.add(dispatch.buffer);
        const reductions = nodes.filter(
          node =>
            node.id.startsWith(`${prefix}-`) &&
            /-(bounds-blocks|bounds|partial|fold|summary)$/.test(node.id)
        );
        expect(reductions).toHaveLength(rows === 4096 ? 5 : 6);
        memory.write(parent, new Uint32Array([1, 1, 1, 1, 1, 1]));
        executeCommandNodesOnCPU([arm], memory);
        const triples = memory.readUint32(dispatch);
        reductions.forEach((node, slot) => {
          expect(node.condition).toMatchObject({
            source: 'gpu',
            mode: 'indirect',
            buffer: dispatch.buffer,
            byteOffset: dispatch.byteOffset + slot * 12
          });
          const record = getWGSLKernelNodeRecord(scene.graph, node.id)!;
          expect([...triples.slice(slot * 3, slot * 3 + 3)]).toEqual([...record.workgroups]);
        });
        memory.write(parent, new Uint32Array(6));
        executeCommandNodesOnCPU([arm], memory);
        for (let slot = 0; slot < reductions.length; slot++)
          expect(memory.readUint32(dispatch)[slot * 3]).toBe(0);
      }
    } finally {
      scene.device.destroy();
    }
  }
});

it('executes rounded integer search kernels without discarding the NumPy optimum 18', () => {
  const result = replayIntegerSearch(15, fixture.roundedCandidates.aicc);
  expect(result[0]).toBe(fixture.roundedCandidates.optimum);
  expect(result[3]).toBe(1);
  expect(result[6]).toBe(0);
});

it('resolves every optimum in narrow integer brackets, including adjacent-count score ties', () => {
  for (const lower of [15, 16]) {
    for (const width of [0, 1, 2, 3, 4, 7, 15]) {
      for (let optimum = 0; optimum <= width; optimum++) {
        const result = replayIntegerSearch(
          lower,
          Array.from({length: width + 1}, (_, index) => (index - optimum) ** 2)
        );
        expect(result[0], `bounds [${lower}, ${lower + width}], optimum ${lower + optimum}`).toBe(
          lower + optimum
        );
        expect(result[3]).toBe(1);
        expect(result[6]).toBe(0);
      }
    }
  }
  expect(replayIntegerSearch(15, [4, 2, 0, 0, 2])[0]).toBe(17);
});

it('executes division-first MGTWR component scoring without an overflowing intermediate product', () => {
  const scene = createScene();
  try {
    const memory = createCPUGraphMemory();
    const points = scene.view('float32', 3);
    const predictors = scene.view('float32', 1);
    const partial = scene.view('float32', 1);
    const parameters = scene.view('float32', 4);
    memory.write(points, [0, 0, 0]);
    memory.write(predictors, [1e14]);
    memory.write(partial, [1e12]);
    memory.write(parameters, [22, 1.3, 0, 0]);
    const gate = createWGSLDispatchGate(scene.graph, {
      id: 'overflow-gate',
      operation: 'Test',
      shapes: [[1], [1]]
    });
    memory.write(gate.view, new Uint32Array([1, 1, 1, 1, 1, 1]));
    const evaluation = createMGTWRComponentEvaluationNodes(scene.graph, {
      id: 'overflow-evaluation',
      rows: 1,
      predictorCount: 1,
      component: 1,
      points,
      predictors,
      partial,
      parameters,
      adaptive: false,
      bisquare: true,
      gate
    });
    executeCommandNodesOnCPU([evaluation.nodes[0]], memory);
    const scratch = evaluation.nodes[0].resources?.find(
      resource => resource.usage === 'storage-read-write'
    );
    if (!scratch || !('buffer' in scratch) || !(scratch.buffer instanceof GraphDataView)) {
      throw new Error('component scoring scratch view missing');
    }
    const fitted = memory.readFloat32(scratch.buffer)[0];
    // One observation, no intercept: beta = response / predictor, fitted = response.
    expect(Number.isFinite(fitted)).toBe(true);
    expect(fitted / 1e12).toBeCloseTo(1, 6);
  } finally {
    scene.device.destroy();
  }
});

it('W5-54 intercept search evaluator preserves a constant million-unit partial response', () => {
  const scene = createScene();
  recordWGSLKernelNodes(scene.graph);
  try {
    const memory = createCPUGraphMemory();
    const points = scene.view('float32', 42);
    const partial = scene.view('float32', 14);
    const parameters = scene.view('float32', 4);
    memory.write(
      points,
      Array.from({length: 14}, (_, row) => [row < 6 ? 0 : 2, 0, row < 6 ? 0 : 3]).flat()
    );
    memory.write(partial, Array(14).fill(1000000));
    memory.write(parameters, [5, 1.25, 0, 0]);
    const gate = createWGSLDispatchGate(scene.graph, {
      id: 'intercept-search-gate',
      operation: 'Test',
      shapes: [[14], [1]]
    });
    memory.write(gate.view, new Uint32Array([1, 1, 1, 1, 1, 1]));
    const evaluation = createMGTWRComponentEvaluationNodes(scene.graph, {
      id: 'intercept-search',
      rows: 14,
      predictorCount: 1,
      component: 0,
      points,
      predictors: scene.view('float32', 14),
      partial,
      parameters,
      adaptive: false,
      bisquare: false,
      gate
    });
    executeCommandNodesOnCPU([evaluation.nodes[0]], memory);
    const record = getWGSLKernelNodeRecord(scene.graph, evaluation.nodes[0].id)!;
    const scratch = record.bindings.find(binding => binding.name === 'scratch')!.view;
    const actual = memory.readFloat32(scratch);
    for (let row = 0; row < 14; row++) expect(actual[row * (scratch.length / 14)]).toBe(1000000);
  } finally {
    scene.device.destroy();
  }
});

for (const scenario of ['mixed', 'rounded'] as const) {
  it(`B3 B4 intercept search evaluator retains ${scenario} partial-response evidence`, () => {
    const scene = createScene();
    recordWGSLKernelNodes(scene.graph);
    try {
      const memory = createCPUGraphMemory();
      const points = scene.view('float32', 24);
      const partial = scene.view('float32', 8);
      const parameters = scene.view('float32', 4);
      memory.write(
        points,
        Array.from({length: 8}, (_, row) => [
          scenario === 'rounded' && row >= 4 ? 2 : 0,
          0,
          0
        ]).flat()
      );
      memory.write(
        partial,
        scenario === 'mixed'
          ? [33554432, -33554432, 1, 1, 1, 1, 1, 1]
          : [1e6, 1e6, 1e6, 1e6, 1e6 + 1 / 16, 1e6 + 1 / 16, 1e6 + 1 / 16, 1e6 + 1 / 16]
      );
      memory.write(parameters, [1, 0, 0, 0]);
      const gate = createWGSLDispatchGate(scene.graph, {
        id: `review-${scenario}-gate`,
        operation: 'Test',
        shapes: [[8], [1]]
      });
      memory.write(gate.view, new Uint32Array([1, 1, 1, 1, 1, 1]));
      const evaluation = createMGTWRComponentEvaluationNodes(scene.graph, {
        id: `review-${scenario}`,
        rows: 8,
        predictorCount: 1,
        component: 0,
        points,
        predictors: scene.view('float32', 8),
        partial,
        parameters,
        adaptive: false,
        bisquare: false,
        gate
      });
      executeCommandNodesOnCPU(evaluation.nodes, memory);
      if (scenario === 'mixed') {
        const fit = getWGSLKernelNodeRecord(scene.graph, evaluation.nodes[0].id)!;
        const scratch = fit.bindings.find(binding => binding.name === 'scratch')!.view;
        const values = memory.readFloat32(scratch);
        const stride = scratch.length / 8;
        for (let row = 0; row < 8; row++) expect(values[row * stride]).toBeCloseTo(0.75, 6);
      } else {
        // r=(1/16)*exp(-2)/(1+exp(-2)); RSS=8*r*r; tr(S)=2/(1+exp(-2)).
        const summary = memory.readFloat32(evaluation.summary);
        expect(Number.isFinite(summary[2])).toBe(true);
        expect(summary[2]).toBeCloseTo(-45.26422157, 4);
      }
    } finally {
      scene.device.destroy();
    }
  });
}

for (const model of ['component', 'gtwr'] as const) {
  it(`preserves a finite large-response ${model} AICc`, () => {
    const scene = createScene();
    try {
      const memory = createCPUGraphMemory();
      const points = scene.view('float32', 24);
      const positions = scene.view('float32x2', 8);
      const times = scene.view('float32', 8);
      const predictors = scene.view('float32', 8);
      const response = scene.view('float32', 8);
      const bounds = scene.view('float32', 6);
      const selected = scene.view('float32', 2);
      const status = scene.status();
      memory.write(points, fixture.largeResponse.points);
      memory.write(positions, fixture.largeResponse.positions);
      memory.write(times, new Float32Array(8));
      memory.write(predictors, fixture.largeResponse.predictors);
      memory.write(response, fixture.largeResponse.response);
      memory.write(bounds, [100, 100, 0, 0, 0.01, 0.005]);
      let summary = scene.view('float32', 12);
      if (model === 'component') {
        const nodes = createGTWRBandwidthSearchNodes(scene.graph, {
          id: 'large-component-search',
          rows: 8,
          adaptive: false,
          search: {parameters: bounds, maximumIterations: 1},
          parameters: selected,
          status,
          evaluate: (parameters, gate, id) => {
            const evaluation = createMGTWRComponentEvaluationNodes(scene.graph, {
              id,
              rows: 8,
              predictorCount: 1,
              component: 0,
              points,
              predictors,
              partial: response,
              parameters,
              adaptive: false,
              bisquare: false,
              gate
            });
            summary = evaluation.summary;
            return evaluation;
          }
        });
        executeCommandNodesOnCPU(nodes, memory);
      } else {
        recordWGSLKernelNodes(scene.graph);
        const parameters = scene.view('float32', 4);
        memory.write(parameters, [100, 0, 0, 0]);
        const fitted = scene.view('float32', 8);
        const hatDiagonal = scene.view('float32', 8);
        const localStatus = scene.view('uint32', 8);
        const residuals = scene.view('float32', 8);
        memory.write(fitted, fixture.largeResponse.expected.gtwr.fitted);
        memory.write(hatDiagonal, fixture.largeResponse.expected.gtwr.hat);
        memory.write(localStatus, new Uint32Array(8));
        // Independent float64 QR oracle values, before either prediction or residual is
        // rounded to float32. Never subtract the buffer's rounded fitted values.
        memory.write(
          residuals,
          fixture.largeResponse.response.map(
            (value, row) => value - fixture.largeResponse.expected.gtwr.fitted[row]
          )
        );
        const contributor = new GPUGeographicallyTemporallyWeightedRegression({
          ...scene.props,
          positions,
          times,
          predictors,
          response,
          parameters,
          adaptive: false,
          bandwidthSearch: undefined,
          output: {
            coefficients: scene.view('float32', 16),
            fitted,
            hatDiagonal,
            localStatus,
            residuals,
            summary
          }
        });
        const summaryNodes = contributor
          .getCommandNodes(scene.graph)
          .filter(node =>
            /-(response-anchor-bounds|statistics-(partial|fold)|variance-(partial|summary))$/.test(
              node.id
            )
          )
          .map(node => {
            const record = getWGSLKernelNodeRecord(scene.graph, node.id)!;
            return createWGSLKernelNode(scene.graph, {
              id: `${node.id}-serial-audit`,
              operation: record.operation,
              invocationCount: 1,
              bindings: record.bindings,
              declarations: record.declarations,
              body: getSerialAuditBody(record)
            });
          });
        expect(summaryNodes).toHaveLength(5);
        executeCommandNodesOnCPU(summaryNodes, memory);
      }
      expect(memory.readFloat32(summary)[2]).toBeCloseTo(
        fixture.largeResponse.expected[model].aicc,
        3
      );
      if (model === 'component') {
        expect(Array.from(memory.readFloat32(selected))).toEqual([100, 0]);
        expect(memory.readUint32(status.converged)[0]).toBe(1);
        expect(memory.readUint32(status.iterationLimitReached)[0]).toBe(0);
      }
    } finally {
      scene.device.destroy();
    }
  });
}

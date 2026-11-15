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
  createWGSLKernelNode
} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';
import fixture from '../../oracles/fixtures/gtwr-bandwidth-search.json';

function createScene() {
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
    positions: view('float32x2', 16),
    times: view('float32', 16),
    temporalContext: {
      unit: 'second',
      epoch: 0n,
      encoding: 'float32-relative',
      axis: {kind: 'irregular'}
    },
    predictors: view('float32', 16),
    predictorCount: 1,
    response: view('float32', 16),
    parameters: view('float32', 4),
    adaptive: true,
    bandwidthSearch: {parameters: view('float32', 6), maximumIterations: 1},
    output: {
      coefficients: view('float32', 32),
      localStatus: view('uint32', 16),
      fitted: view('float32', 16),
      hatDiagonal: view('float32', 16),
      summary: view('float32', 12),
      residuals: view('float32', 16),
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
      bandwidthSearch: {...scene.props.bandwidthSearch!, maximumIterations: 32}
    });
    expect(maximum.estimatedCommandNodeCount).toBeLessThanOrEqual(32768);

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
    for (const maximumIterations of [1, 2, 3]) {
      const contributor = new GPUGeographicallyTemporallyWeightedRegression({
        ...scene.props,
        id: `gtwr-${maximumIterations}`,
        bandwidthSearch: {...scene.props.bandwidthSearch!, maximumIterations}
      });
      const nodes = contributor.getCommandNodes(scene.graph);
      expect(nodes.length).toBe(contributor.estimatedCommandNodeCount);
      const candidates = nodes.filter(node => /evaluate-\d+-(points|fit|summary)$/.test(node.id));
      expect(candidates.length).toBeGreaterThan(0);
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
        const parameters = scene.view('float32', 4);
        memory.write(parameters, [100, 0, 0, 0]);
        const fitted = scene.view('float32', 8);
        const hatDiagonal = scene.view('float32', 8);
        const localStatus = scene.view('uint32', 8);
        memory.write(fitted, fixture.largeResponse.expected.gtwr.fitted);
        memory.write(hatDiagonal, fixture.largeResponse.expected.gtwr.hat);
        memory.write(localStatus, new Uint32Array(8));
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
            summary
          }
        });
        // Isolate the production summary from the Cholesky kernels, whose pointer-to-array
        // assignments are not supported by the WGSL interpreter. The headless case runs the fit.
        const summaryNode = contributor
          .getCommandNodes(scene.graph)
          .find(node => node.id.endsWith('-summary'));
        if (!summaryNode) {
          throw new Error('GTWR summary node missing');
        }
        executeCommandNodesOnCPU([summaryNode], memory);
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

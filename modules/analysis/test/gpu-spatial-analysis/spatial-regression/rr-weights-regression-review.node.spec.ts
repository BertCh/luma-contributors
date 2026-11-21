// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPUSpatialWeightsTraces} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-weights-traces';
import {GPUSpatialErrorGM} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-error-gm';
import {GPUSpatialTwoStageLeastSquares} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-two-stage-least-squares';
import {GPUSpatialLagMaximumLikelihood} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-lag-maximum-likelihood';
import {GPUSpatialErrorMaximumLikelihood} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-error-maximum-likelihood';
import {
  getGPUSpatialTwoStageLeastSquaresDiagnostics,
  getGPUSpatialTwoStageLeastSquaresModelResult
} from '../../../src/gpu-spatial-analysis/spatial-regression/spatial-two-stage-least-squares-model-result';
import {
  getGPUSpatialMaximumLikelihoodDiagnostics,
  getGPUSpatialLagMaximumLikelihoodModelResult
} from '../../../src/gpu-spatial-analysis/spatial-regression/spatial-maximum-likelihood-model-result';
import {computeSpatialErrorGMOnCPU} from '../../../src/gpu-spatial-analysis/spatial-regression/cpu-spatial-error-gm-operation';
import {computeSpatialTwoStageLeastSquaresOnCPU} from '../../../src/gpu-spatial-analysis/spatial-regression/cpu-spatial-two-stage-least-squares-operation';
import {REVIEW_REGRESSION_FIXTURE} from './rr-weights-regression-review-fixture';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';

for (const scope of ['whole', 'root', 'child'] as const) {
  for (const hasCount of [false, true]) {
    it(`R1 ${scope} requiredCount suppresses a truncated graph with count=${hasCount}`, () => {
      const device = createNullWebGPUDevice();
      const graph = new GPUCommandGraph(device);
      const memory = createCPUGraphMemory();
      let serial = 0;
      const input = <Format extends 'float32' | 'uint32'>(
        format: Format,
        values: Float32Array | Uint32Array
      ) => {
        const view = createTransientView(graph, `count-${serial++}`, format, values.length);
        memory.write(view, values);
        return view;
      };
      const capacity = scope === 'child' ? 2 : 3;
      const requiredCount = input('uint32', Uint32Array.of(capacity));
      const count = input('uint32', Uint32Array.of(capacity));
      const status = {requiredCount, ...(hasCount ? {count} : {})};
      const createWeights = (offsets: number[], neighbors: number[]) => ({
        offsets: input('uint32', Uint32Array.from(offsets)),
        neighbors: input('uint32', Uint32Array.from(neighbors)),
        weights: input(
          'float32',
          Float32Array.from(neighbors, () => 1)
        )
      });
      const weights =
        scope === 'whole'
          ? {...createWeights([0, 1, 2, 3], [1, 0, 1]), status}
          : {
              rowCount: 3,
              ...(scope === 'root' ? {status} : {}),
              chunks: [
                {rowStart: 0, rowCount: 1, weights: createWeights([0, 1], [1])},
                {
                  rowStart: 1,
                  rowCount: 2,
                  weights: {
                    ...createWeights([0, 1, 2], [0, 1]),
                    ...(scope === 'child' ? {status} : {})
                  }
                }
              ]
            };
      const powerTraces = input('float32', Float32Array.of(77, 98, 99));
      const powerSums = input('float32', Float32Array.of(77, 98, 99));
      const infinityNorm = input('float32', Float32Array.of(77, 98, 99));
      try {
        const nodes = new GPUSpatialWeightsTraces({
          weights,
          maximumPower: 1,
          mode: 'exact',
          output: {powerTraces, powerSums, infinityNorm}
        }).getCommandNodes(graph);
        // Every row has one unit link: ||W||inf=1, tr(W)=0 and 1'W1=3.
        // required > retained (or capacity when count is absent) proves missing topology.
        for (const missing of [false, true, false]) {
          memory.write(requiredCount, Uint32Array.of(capacity + Number(missing)));
          executeCommandNodesOnCPU(nodes, memory);
          expect(Array.from(memory.readFloat32(infinityNorm))).toEqual([missing ? NaN : 1, 98, 99]);
          expect(Array.from(memory.readFloat32(powerTraces))).toEqual([missing ? NaN : 0, 98, 99]);
          expect(Array.from(memory.readFloat32(powerSums))).toEqual([missing ? NaN : 3, 98, 99]);
        }
        if (hasCount) {
          // A supplied count can prove loss even when the requirement fits physical capacity.
          memory.write(count, Uint32Array.of(capacity - 1));
          executeCommandNodesOnCPU(nodes, memory);
          expect(Array.from(memory.readFloat32(infinityNorm))).toEqual([NaN, 98, 99]);
        }
      } finally {
        device.destroy();
      }
    });
  }
}

for (const Contributor of [
  GPUSpatialErrorGM,
  GPUSpatialTwoStageLeastSquares,
  GPUSpatialWeightsTraces,
  GPUSpatialLagMaximumLikelihood,
  GPUSpatialErrorMaximumLikelihood
]) {
  it(`R1 ${Contributor.name} suppresses count-proven loss even when the requirement fits capacity`, () => {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    const memory = createCPUGraphMemory();
    let serial = 0;
    const view = <Format extends 'float32' | 'uint32'>(format: Format, length: number) =>
      createTransientView(graph, `model-count-${serial++}`, format, length);
    const count = view('uint32', 1);
    const requiredCount = view('uint32', 1);
    const table = view('float32', 12);
    const infinityNorm = view('float32', 1);
    const code = view('uint32', 1);
    try {
      const props = {
        weights: {
          offsets: view('uint32', 9),
          neighbors: view('uint32', 8),
          weights: view('float32', 8),
          status: {count, requiredCount}
        },
        predictors: view('float32', 8),
        response: view('float32', 8),
        predictorCount: 1,
        maximumPower: 2,
        mode: 'stochastic' as const,
        probeCount: 1,
        traceMode: 'stochastic' as const,
        traceProbeCount: 1,
        estimator: 'kelejian-prucha' as const,
        output: {
          table,
          summary: view('float32', 22),
          infinityNorm,
          powerTraces: view('float32', 2),
          powerSums: view('float32', 2),
          status: {code}
        }
      };
      const allNodes = new Contributor(props).getCommandNodes(graph);
      const destination =
        Contributor === GPUSpatialWeightsTraces ? props.output.powerTraces : table;
      const writers = allNodes.flatMap((node, index) =>
        !/suppress-incomplete/.test(node.id) &&
        node.resources?.some(
          resource =>
            'buffer' in resource &&
            resource.buffer === destination &&
            resource.usage === 'storage-read-write'
        )
          ? [index]
          : []
      );
      expect(writers.length).toBeGreaterThan(0);
      // R6: a normal finish writer after suppression would make the result definite again.
      expect(
        allNodes.findIndex(
          node =>
            /suppress-incomplete/.test(node.id) &&
            node.resources?.some(
              resource =>
                'buffer' in resource &&
                resource.buffer === destination &&
                resource.usage === 'storage-read-write'
            )
        )
      ).toBeGreaterThan(Math.max(...writers));
      const nodes = allNodes.filter(node => /weights-loss|suppress-incomplete/.test(node.id));
      memory.write(count, Uint32Array.of(4));
      memory.write(requiredCount, Uint32Array.of(5));
      memory.write(table, new Float32Array(12).fill(7));
      memory.write(infinityNorm, Float32Array.of(7));
      memory.write(code, Uint32Array.of(0));
      executeCommandNodesOnCPU(nodes, memory);
      if (Contributor === GPUSpatialWeightsTraces)
        expect(Array.from(memory.readFloat32(infinityNorm))).toEqual([NaN]);
      else {
        expect(Array.from(memory.readFloat32(table))).toEqual(new Array(12).fill(NaN));
        expect(Array.from(memory.readUint32(code))).toEqual([3]);
      }
      expect(
        () =>
          new Contributor({
            ...props,
            weights: {...props.weights, status: {requiredCount: view('float32', 1) as never}}
          })
      ).toThrow(/weights.*status/);
    } finally {
      device.destroy();
    }
  });
}

for (const failure of ['short', 'wrong-format'] as const) {
  it(`R3 rejects ${failure} trace-error output before graph mutation and permits a same-ID retry`, () => {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    let serial = 0;
    const view = <Format extends 'float32' | 'uint32'>(format: Format, length: number) =>
      createTransientView(graph, `retry-view-${serial++}`, format, length);
    const props = {
      id: 'retry',
      weights: {
        offsets: view('uint32', 9),
        neighbors: view('uint32', 8),
        weights: view('float32', 8)
      },
      predictors: view('float32', 8),
      response: view('float32', 8),
      predictorCount: 1,
      maximumPower: 2,
      traceMode: 'exact' as const,
      output: {
        table: view('float32', 12),
        summary: view('float32', 22),
        status: {code: view('uint32', 1)},
        powerTraceStandardErrors:
          failure === 'short' ? view('float32', 1) : (view('uint32', 2) as never)
      }
    };
    try {
      // A rejection at either stage must leave the graph usable for the corrected declaration.
      expect(() => graph.add(new GPUSpatialErrorMaximumLikelihood(props))).toThrow(
        /powerTraceStandardErrors/
      );
      expect(() =>
        graph.add(
          new GPUSpatialErrorMaximumLikelihood({
            ...props,
            output: {...props.output, powerTraceStandardErrors: view('float32', 2)}
          })
        )
      ).not.toThrow();
      expect(() => new GPUSpatialErrorMaximumLikelihood(props)).toThrow(
        /retry.*powerTraceStandardErrors/
      );
    } finally {
      device.destroy();
    }
  });
}

for (const value of [0, 1, NaN]) {
  it(`R4 diagnostic flags preserve ${String(value)} availability`, () => {
    const twoStageSummary = new Float32Array(16).fill(value);
    const likelihoodSummary = new Float32Array(22).fill(value);
    const expected = Number.isNaN(value) ? null : value === 1;
    const twoStage = getGPUSpatialTwoStageLeastSquaresDiagnostics(twoStageSummary);
    expect(twoStage.rhoOutsideAdmissibleRange).toBe(expected);
    const likelihood = getGPUSpatialMaximumLikelihoodDiagnostics(likelihoodSummary);
    for (const name of [
      'approximation',
      'converged',
      'atBoundary',
      'seriesLimited',
      'varianceDefined'
    ] as const)
      expect(likelihood[name], name).toBe(expected);
    if (Number.isNaN(value)) {
      const readback = {table: new Float32Array(12).fill(NaN), status: Uint32Array.of(3)};
      expect(
        getGPUSpatialTwoStageLeastSquaresModelResult({...readback, summary: twoStageSummary}).status
          .code
      ).toBe(3);
      expect(
        getGPUSpatialLagMaximumLikelihoodModelResult({...readback, summary: likelihoodSummary})
          .status.code
      ).toBe(3);
    }
  });
}

it('R5 traces ignore wider undeclared code and converged status views through loss and recovery', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const memory = createCPUGraphMemory();
  let serial = 0;
  const input = <Format extends 'float32' | 'uint32'>(
    format: Format,
    values: Float32Array | Uint32Array
  ) => {
    const view = createTransientView(graph, `wide-status-${serial++}`, format, values.length);
    memory.write(view, values);
    return view;
  };
  const candidateOverflow = input('uint32', Uint32Array.of(0));
  const code = input('uint32', Uint32Array.of(91));
  const converged = input('uint32', Uint32Array.of(92));
  const approximation = input('uint32', Uint32Array.of(77));
  const widerStatus = {approximation, code, converged};
  try {
    const nodes = new GPUSpatialWeightsTraces({
      weights: {
        offsets: input('uint32', Uint32Array.of(0, 1, 3, 4)),
        neighbors: input('uint32', Uint32Array.of(1, 0, 2, 1)),
        weights: input('float32', Float32Array.of(1, 1, 1, 1)),
        status: {candidateOverflow}
      },
      maximumPower: 1,
      mode: 'exact',
      output: {
        powerTraces: input('float32', Float32Array.of(77)),
        powerSums: input('float32', Float32Array.of(77)),
        status: widerStatus
      }
    }).getCommandNodes(graph);
    for (const loss of [0, 1, 0]) {
      memory.write(candidateOverflow, Uint32Array.of(loss));
      executeCommandNodesOnCPU(nodes, memory);
      expect(Array.from(memory.readUint32(code))).toEqual([91]);
      expect(Array.from(memory.readUint32(converged))).toEqual([92]);
      expect(Array.from(memory.readUint32(approximation))).toEqual([0]);
    }
  } finally {
    device.destroy();
  }
});

for (const kind of ['error', 'twoStage'] as const) {
  it(`R6 ${kind} clean CPU fit matches independent NumPy coefficients and residuals`, () => {
    const fixture = REVIEW_REGRESSION_FIXTURE;
    const result =
      kind === 'error'
        ? computeSpatialErrorGMOnCPU(fixture, {predictorCount: 1, estimator: 'kelejian-prucha'})
        : computeSpatialTwoStageLeastSquaresOnCPU(fixture, {predictorCount: 1});
    expect(Array.from(result.status)).toEqual([0]);
    const coefficients =
      kind === 'error' ? fixture.errorCoefficients : fixture.twoStageCoefficients;
    const residuals = kind === 'error' ? fixture.errorResiduals : fixture.twoStageResiduals;
    for (const [index, expected] of coefficients.entries()) {
      expect(Number.isFinite(result.table[index * 4])).toBe(true);
      expect(Math.abs(result.table[index * 4] - expected)).toBeLessThanOrEqual(2e-5);
    }
    for (const [index, expected] of residuals.entries()) {
      expect(Number.isFinite(result.residuals[index])).toBe(true);
      expect(Math.abs(result.residuals[index] - expected)).toBeLessThanOrEqual(2e-5);
    }
  });
}

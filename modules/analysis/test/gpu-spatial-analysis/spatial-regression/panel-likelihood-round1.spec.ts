// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {createRegressionTestContext} from './math-review-regression-test-utils';
import {readFloat32, readUint32, submitGraph} from '../../utils/gpu-contributor-test-utils';
import {
  createPanelAuditInput,
  expectPanelAuditRows,
  PANEL_AUDIT_EXPECTED
} from './panel-audit.test-utils';
import {GPUSpatialPanelLagMaximumLikelihood} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-panel-lag-maximum-likelihood';
import {GPUSpatialPanelErrorMaximumLikelihoodOperation} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-panel-error-maximum-likelihood-operation';
import {CPUSpatialPanelErrorMaximumLikelihoodOperation} from '../../../src/gpu-spatial-analysis/spatial-regression/cpu-spatial-panel-error-maximum-likelihood-operation';
import {GPUSpatialPanelErrorMaximumLikelihood} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-panel-error-maximum-likelihood';
import {GPUSpatialLagMaximumLikelihood} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-lag-maximum-likelihood';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {getPanelDiagnosticNodes} from '../../../src/gpu-spatial-analysis/spatial-regression/panel-diagnostics';
import {
  createWGSLKernelNode,
  recordWGSLKernelNodes,
  getWGSLKernelNodeRecord
} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';

for (const scale of [1, 2 ** 25])
  it(`B1 GPU two-predictor exact pole scale=${scale}`, async () => {
    const context = await createRegressionTestContext();
    try {
      const response = Float32Array.from(
        {length: 12},
        (_, row) => (3 * row * row + 2 * row + 1) % 11
      );
      const predictors = Float32Array.from({length: 24}, (_, index) => {
        const row = Math.floor(index / 2),
          neighbor = row - (row % 4) + [1, 0, 3, 2][row % 4],
          other = (row * row + row + 4) % 7;
        return index % 2
          ? other
          : scale * (response[row] - 0.5 * response[neighbor] - 0.25 * other);
      });
      const coefficients = context.output('float32', 5),
        status = context.output('uint32', 1);
      context.graph.add(
        new GPUSpatialPanelLagMaximumLikelihood({
          periodCount: 3,
          predictorCount: 2,
          effects: 'unit',
          response: context.input('float32', response),
          predictors: context.input('float32', predictors),
          weights: {
            offsets: context.input('uint32', [0, 1, 2, 3, 4]),
            neighbors: context.input('uint32', [1, 0, 3, 2]),
            weights: context.input('float32', [1, 1, 1, 1])
          },
          output: {
            coefficients: coefficients.view,
            covariance: context.output('float32', 16).view,
            summary: context.output('float32', 6).view,
            fitted: context.output('float32', 12).view,
            residuals: context.output('float32', 12).view,
            status: status.view
          }
        })
      );
      context.run();
      expect(Array.from(await readUint32(status.buffer, 1))).toEqual([2]);
      const actual = await readFloat32(coefficients.buffer, 5);
      expect(actual).toHaveLength(5);
      expect(actual.every(Number.isNaN)).toBe(true);
    } finally {
      context.destroy();
    }
  });

for (const Operation of [
  GPUSpatialPanelErrorMaximumLikelihoodOperation,
  CPUSpatialPanelErrorMaximumLikelihoodOperation
])
  it(`B2 named certificate recovery ${Operation.name}`, async () => {
    const context = await createRegressionTestContext();
    try {
      const input = createPanelAuditInput();
      const operation = new Operation({
        entityCount: 4,
        periodCount: 4,
        predictorCount: 1,
        maximumPower: 8,
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
      });
      for (const complete of [1, 0, 1]) {
        const actual = await runOperationGraph(context.device, {
          inputs: {
            predictors: {format: 'float32', values: input.predictors},
            response: {format: 'float32', values: input.response},
            offsets: {format: 'uint32', values: input.weights.offsets},
            neighbors: {format: 'uint32', values: input.weights.neighbors},
            weights: {format: 'float32', values: input.weights.weights},
            complete: {format: 'uint32', values: Uint32Array.of(complete)}
          },
          operations: [operation],
          outputs: ['table', 'status', 'tests', 'diagnosticStatus']
        });
        expect(Array.from(actual['status'])).toEqual([complete ? 0 : 3]);
        expect(Array.from(actual['diagnosticStatus'])).toEqual([complete ? 0 : 1]);
        expect(actual['table']).toHaveLength(12);
        expect(actual['tests']).toHaveLength(6);
        if (complete)
          expectPanelAuditRows(
            actual['table'],
            [
              NaN,
              NaN,
              NaN,
              NaN,
              1.4543553721878475,
              0.16217494003990773,
              8.96781815877325,
              3.024458354593923e-19,
              -0.27636131663120095,
              0.22663902994434895,
              -1.2193897789761157,
              0.22269628745043568
            ],
            5e-4
          );
        else {
          expect(Array.from(actual['table']).every(Number.isNaN)).toBe(true);
          expect(Array.from(actual['tests']).every(Number.isNaN)).toBe(true);
        }
      }
    } finally {
      context.destroy();
    }
  });

for (const route of ['random', 'unit', 'tiled'] as const)
  it(`B3 GPU panel-lag certificate and accessible span ${route}`, async () => {
    const context = await createRegressionTestContext();
    try {
      const input = createPanelAuditInput();
      const complete = context.output('uint32', 1),
        status = context.output('uint32', 1),
        coefficients = context.output('float32', 4),
        hausman = context.output('float32', 3),
        hausmanStatus = context.output('uint32', 1);
      context.graph.add(
        new GPUSpatialPanelLagMaximumLikelihood({
          periodCount: 4,
          predictorCount: 1,
          effects: route === 'random' ? 'random' : 'unit',
          fitMode: route === 'tiled' ? 'tiled' : 'reference',
          eigenvalues: route === 'tiled' ? context.input('float32', [-1, 0, 0, 1]) : undefined,
          predictors: context.input('float32', input.predictors),
          response: context.input('float32', input.response),
          weights: {
            offsets: context.input('uint32', [2, 4, 6, 8, 10]),
            neighbors: context.input('uint32', [0, 0, ...input.weights.neighbors]),
            weights: context.input('float32', [0, 0, ...input.weights.weights]),
            status: {
              complete: complete.view,
              count: context.input('uint32', [8]),
              requiredCount: context.input('uint32', [8])
            }
          },
          output: {
            coefficients: coefficients.view,
            covariance: context.output('float32', 9).view,
            summary: context.output('float32', 6).view,
            fitted: context.output('float32', 16).view,
            residuals: context.output('float32', 16).view,
            status: status.view
          },
          hausman:
            route === 'random' ? {tests: hausman.view, status: hausmanStatus.view} : undefined
        })
      );
      const compiled = context.graph.compile();
      try {
        for (const certificate of [1, 0, 1]) {
          complete.buffer.write(Uint32Array.of(certificate));
          submitGraph(context.device, compiled, undefined);
          expect(Array.from(await readUint32(status.buffer, 1))).toEqual([certificate ? 0 : 1]);
          const actual = await readFloat32(coefficients.buffer, 4);
          expect(actual).toHaveLength(4);
          if (certificate)
            expectPanelAuditRows(
              actual,
              PANEL_AUDIT_EXPECTED[route === 'random' ? 'random' : 'unit'].coefficients,
              5e-4
            );
          else expect(actual.every(Number.isNaN)).toBe(true);
          if (route === 'random' && !certificate) {
            expect(Array.from(await readUint32(hausmanStatus.buffer, 1))).toEqual([1]);
            const values = await readFloat32(hausman.buffer, 3);
            expect(values).toHaveLength(3);
            expect(values.every(Number.isNaN)).toBe(true);
          }
        }
      } finally {
        compiled.destroy();
      }
    } finally {
      context.destroy();
    }
  });

for (const constant of [0, 6])
  it(`B4 GPU cancellation and equal nonzero original lag ${constant}`, async () => {
    const context = await createRegressionTestContext();
    try {
      const table = context.output('float32', 12),
        summary = context.output('float32', 22),
        status = context.output('uint32', 1);
      const offsets = constant ? [0, 2, 4, 5, 6, 7, 8, 9] : [0, 2, 4, 4, 4, 4, 4, 4],
        neighbors = constant ? [0, 1, 0, 1, 0, 0, 0, 0, 0] : [0, 1, 0, 1],
        weights = constant ? [5, 3, 3, 1, 2, 2, 2, 2, 2] : [5, 5, 1, 1];
      context.graph.add(
        new GPUSpatialLagMaximumLikelihood({
          predictorCount: 1,
          maximumPower: 8,
          predictors: context.input('float32', [0, 0, 1, -1, -1, 1, 0]),
          response: context.input('float32', [3, -3, 1, 1, -1, -1, 2]),
          weights: {
            offsets: context.input('uint32', offsets),
            neighbors: context.input('uint32', neighbors),
            weights: context.input('float32', weights)
          },
          output: {table: table.view, summary: summary.view, status: {code: status.view}}
        })
      );
      context.run();
      expect(Array.from(await readUint32(status.buffer, 1))).toEqual([0]);
      const actual = await readFloat32(table.buffer, 12),
        values = await readFloat32(summary.buffer, 22);
      expect(actual).toHaveLength(12);
      expect(values).toHaveLength(22);
      expect(actual[4]).toBe(0);
      expect(values[6]).toBeNaN();
    } finally {
      context.destroy();
    }
  });

for (const first of [true, false])
  it(`B6 GPU million-row centered variation outlier first=${first}`, async () => {
    const context = await createRegressionTestContext();
    try {
      recordWGSLKernelNodes(context.graph);
      const rows = 1048576;
      const contributor = new GPUSpatialPanelErrorMaximumLikelihood({
        periodCount: rows / 4,
        predictorCount: 1,
        maximumPower: 2,
        traceMode: 'stochastic',
        traceProbeCount: 1,
        predictors: context.input('float32', new Float32Array(rows)),
        response: context.input('float32', new Float32Array(rows)),
        weights: {
          offsets: context.input('uint32', [0, 1, 2, 3, 4]),
          neighbors: context.input('uint32', [1, 0, 3, 2]),
          weights: context.input('float32', [1, 1, 1, 1])
        },
        output: {
          table: context.output('float32', 12).view,
          summary: context.output('float32', 22).view,
          status: {code: context.output('uint32', 1).view}
        },
        diagnostics: {
          tests: context.output('float32', 6).view,
          status: context.output('uint32', 1).view
        }
      });
      const nodes = getPanelDiagnosticNodes(context.graph, contributor),
        tiles = nodes.find(node => node.id.endsWith('-diagnostic-sums-tiles'))!,
        record = getWGSLKernelNodeRecord(context.graph, tiles.id)!,
        lags = record.bindings.find(binding => binding.name === 'lags')!.view;
      context.graph.add(
        createWGSLKernelNode(context.graph, {
          id: 'seed-centered-variation',
          operation: 'PanelReview',
          invocationCount: rows,
          bindings: [{name: 'lags', view: lags, type: 'f32', access: 'read_write'}],
          body: `let value=select(0.0,0.1,index==${first ? 0 : rows - 1}u)-0.1/${rows}.0;lags[lagsOffset+index*4u+2u]=value;`
        })
      );
      context.graph.add(nodes.filter(node => node.id.includes('-diagnostic-sums-')));
      const publisher = getWGSLKernelNodeRecord(
          context.graph,
          nodes.find(node => node.id.endsWith('-diagnostic-tests'))!.id
        )!,
        sums = publisher.bindings.find(binding => binding.name === 'sums')!.view,
        output = context.output('float32', 1);
      context.graph.add(
        createWGSLKernelNode(context.graph, {
          id: 'read-variation',
          operation: 'PanelReview',
          invocationCount: 1,
          bindings: [
            {name: 'sums', view: sums, type: 'f32', access: 'read'},
            {name: 'output', view: output.view, type: 'f32', access: 'read_write'}
          ],
          body: 'output[outputOffset]=sums[sumsOffset+4u];'
        })
      );
      context.run();
      const actual = (await readFloat32(output.buffer, 1))[0];
      const value = Math.fround(0.1),
        expected = value * value * (1 - 1 / rows);
      expect(Number.isFinite(actual)).toBe(true);
      expect(actual).toBeGreaterThan(0);
      expect(Math.abs(actual - expected)).toBeLessThan(2e-8);
    } finally {
      context.destroy();
    }
  });

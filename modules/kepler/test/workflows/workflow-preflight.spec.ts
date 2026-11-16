// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test, vi} from 'vitest';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {mountLadderLens} from '@luma-contributors/lenses/computation-views';
import {requireTestDevice} from '@luma-contributors/analysis/testing';
import {
  compareWorkflowPreflightMeasurement,
  createWorkflowPreflightMeasurementChains,
  estimateWorkflowPreflight
} from '../../src/workflows/workflow-preflight';
import {makeWorkflowComputationRecord} from '../../src/workflows/workflow-computation-record';
import {getWorkflowOperationRegistry} from '../../src/workflows/workflow-operations';
import {WorkflowRunner} from '../../src/workflows/workflow-runner';
import type {WorkflowOperationRegistry} from '../../src/workflows/workflow-types';
import {makeFixtureHost} from './workflow-fixtures';

function createRegressionChains(): ReturnType<typeof createWorkflowPreflightMeasurementChains> {
  const limited = createWorkflowPreflightMeasurementChains()[0];
  limited.document.id = 'limited-point-measures';
  limited.document.steps.splice(1, 0, {
    id: 'sort',
    operationId: 'table-sort-rows',
    position: {x: 100, y: 0},
    inputs: {rows: {stepId: 'source', output: 'data'}},
    parameters: {
      column: {from: 'static', value: 'feature_id'},
      limit: {from: 'static', value: 1}
    }
  });
  limited.document.steps[2].inputs['features'] = {stepId: 'sort', output: 'sorted'};
  const emptyCases = [false, true].map(populated => {
    const chain = createWorkflowPreflightMeasurementChains()[0];
    chain.document.id = populated ? 'partially-empty-point-measures' : 'empty-point-measures';
    chain.document.steps[0].parameters['text'] = {
      from: 'static',
      value: JSON.stringify({
        type: 'FeatureCollection',
        features: [
          {
            type: 'Feature',
            properties: {},
            geometry: populated
              ? {type: 'Point', coordinates: [0, 0]}
              : {type: 'MultiPoint', coordinates: []}
          },
          {type: 'Feature', properties: {}, geometry: {type: 'MultiPoint', coordinates: []}}
        ]
      })
    };
    chain.sourceShapes = {};
    return chain;
  });
  return [limited, ...emptyCases];
}

test('KE-WF-07: three chains and empty/truncated regressions meet scratch/node tolerance and open in computation views', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const baseRegistry = getWorkflowOperationRegistry();
  // Measure the actual dispatched graphs through public compiled statistics. Scratch excludes
  // imported input/output buffers and staging. No estimate is fed into these observations.
  let currentStep = '';
  let measurements: Array<{
    stepId: string;
    scratchBytes: number;
    physicalBytes: number;
    nodes: number;
  }> = [];
  const compile = GPUCommandGraph.prototype.compile;
  const observer = vi.spyOn(GPUCommandGraph.prototype, 'compile').mockImplementation(function (
    this: GPUCommandGraph,
    ...parameters: Parameters<typeof compile>
  ) {
    const compiled = compile.apply(this, parameters);
    measurements.push({
      stepId: currentStep,
      scratchBytes: compiled.stats.logicalTransientBytes,
      physicalBytes: compiled.stats.physicalTransientResourceBytes,
      nodes: compiled.stats.nodeOrder.length
    });
    return compiled;
  });
  const operations = baseRegistry.list().map(operation => ({
    ...operation,
    async run(...parameters: Parameters<typeof operation.run>) {
      currentStep = parameters[0].stepId;
      try {
        return await operation.run(...parameters);
      } finally {
        currentStep = '';
      }
    }
  }));
  const registry: WorkflowOperationRegistry = {
    list: () => operations,
    get: id => operations.find(operation => operation.id === id)
  };
  const reports: unknown[] = [];
  try {
    for (const chain of [
      ...createWorkflowPreflightMeasurementChains(),
      ...createRegressionChains()
    ]) {
      measurements = [];
      const estimate = estimateWorkflowPreflight(chain.document, registry, {
        ...chain,
        profileSources: true
      });
      const runner = new WorkflowRunner({
        registry,
        host: {...makeFixtureHost(), getDevice: () => device}
      });
      const container = document.createElement('div');
      document.body.append(container);
      try {
        runner.setDocument(chain.document);
        const run = await runner.run();
        const measured = {
          peakScratchBytes: Math.max(0, ...measurements.map(entry => entry.physicalBytes)),
          commandNodeCount: measurements.reduce((total, entry) => total + entry.nodes, 0)
        };
        const comparison = compareWorkflowPreflightMeasurement(estimate, measured);
        const report = {chain: chain.document.id, run, estimate, measurements, comparison};
        reports.push(report);
        if (run.status !== 'done' || !comparison.passed)
          throw new Error(
            `KE-WF-07 scratch/node qualification failed:\n${JSON.stringify(report, null, 2)}`
          );
        for (const step of run.steps) {
          const operation = registry.get(step.operationId)!;
          if (operation.origin === 'catalog') {
            expect(step.engine, JSON.stringify(report, null, 2)).toBe('gpu');
            expect(
              step.dispatchedContributors?.length,
              JSON.stringify(report, null, 2)
            ).toBeGreaterThan(0);
          }
          const stepMeasurements = measurements.filter(entry => entry.stepId === step.stepId);
          const stepEstimate = estimate.steps.find(entry => entry.stepId === step.stepId)!;
          expect(
            Math.max(0, ...stepMeasurements.map(entry => entry.scratchBytes)),
            JSON.stringify(report, null, 2)
          ).toBe(stepEstimate.scratchBytes);
          expect(
            stepMeasurements.reduce((total, entry) => total + entry.nodes, 0),
            JSON.stringify(report, null, 2)
          ).toBe(stepEstimate.commandNodeCount);
        }
        const lens = mountLadderLens(container, {
          record: makeWorkflowComputationRecord(run, registry),
          rung: 'table',
          bind: chain.document.id,
          urlState: false
        });
        try {
          expect(container.textContent, JSON.stringify(report, null, 2)).toContain(
            'geometry-measures'
          );
          lens.dispatch({type: 'bind', axis: 'node', value: 'measures-0'});
          expect(container.textContent).toContain('CPU tables between steps');
          expect(container.textContent).toContain('runner wall time');
        } finally {
          lens.dispose();
        }
      } finally {
        container.remove();
        runner.destroy();
      }
    }
    expect(reports).toHaveLength(6);
  } finally {
    observer.mockRestore();
  }
}, 120000);

// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createGPUComputationRecord,
  type GPUComputationRecord,
  type GPUComputationEdge,
  type GPUComputationNode
} from '@luma-contributors/analysis/computation-record';
import type {WorkflowOperationRegistry, WorkflowRunRecord} from './workflow-types';

/**
 * Recorded workflow provenance in the existing computation views. This is an intent graph of
 * actual step invocations, not a compiled GPU graph. Durations are runner wall time, including
 * readback/host work, never GPU timestamps or CPU encode measurements. Only the run record is
 * read: later edits/results cannot rewrite an older trace.
 */
export function makeWorkflowComputationRecord(
  run: WorkflowRunRecord,
  registry: WorkflowOperationRegistry
): GPUComputationRecord {
  const recordedIds = new Set(run.steps.map(step => step.stepId));
  const externalIds = new Set(
    run.steps
      .flatMap(step => step.inputs.map(input => input.stepId))
      .filter(stepId => !recordedIds.has(stepId))
  );
  const edges: GPUComputationEdge[] = [];
  const nodes: GPUComputationNode[] = [...externalIds].sort().map(stepId => ({
    id: stepId,
    label: stepId,
    namespace: 'external-inputs',
    kind: 'operation',
    reads: [],
    writes: [],
    attrs: {
      external: true,
      evidence: 'Referenced by this run; producer invocation was not recorded.'
    }
  }));
  for (const step of run.steps) {
    const operation = registry.get(step.operationId);
    nodes.push({
      id: step.stepId,
      label: operation?.title ?? step.operationId,
      namespace: 'steps',
      kind: 'operation',
      op: step.operationId,
      reads: [],
      writes: [],
      attrs: {
        status: step.status,
        engine: step.engine ?? 'none',
        cacheHit: step.cacheHit,
        contentKey: step.key,
        durationMilliseconds: step.durationMilliseconds,
        timingScope: 'runner wall time, not GPU timestamps',
        dispatchedContributors: (step.dispatchedContributors ?? []).join(', '),
        inputs: JSON.stringify(step.inputs),
        outputRows: JSON.stringify(step.outputRows ?? {}),
        materialization:
          'CPU tables between steps (decoded bands for rasters); GPU outputs read back before handoff',
        ...(step.fallbackReason ? {fallbackReason: step.fallbackReason} : {}),
        ...(step.error ? {errorCode: step.error.code, error: step.error.message} : {})
      }
    });
    for (const input of step.inputs) {
      if (!edges.some(edge => edge.from === input.stepId && edge.to === step.stepId))
        edges.push({from: input.stepId, to: step.stepId, kind: 'dataflow'});
    }
  }
  return createGPUComputationRecord({
    id: run.workflowId,
    runId: run.runId,
    createdAt: new Date(run.startedAt).toISOString(),
    adapter: {backend: 'workflow', description: 'Step-level provenance; adapter was not recorded'},
    domains: ['kepler-workflow'],
    graphs: [
      {
        id: 'workflow',
        form: 'intent',
        domain: 'kepler-workflow',
        label: `${run.workflowId} · ${run.runId}`,
        nodes,
        edges,
        attrs: {
          status: run.status,
          durationMilliseconds: run.durationMilliseconds,
          scope: 'Actual workflow steps and CPU materialisation boundaries; no kernel captures'
        }
      }
    ],
    workflow: {
      id: run.workflowId,
      invocations: [{graph: 'workflow', label: run.runId, reason: run.status}]
    }
  });
}

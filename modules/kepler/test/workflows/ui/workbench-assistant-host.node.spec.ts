// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, test} from 'vitest';
import {
  readWorkbenchAssistantCall,
  withWorkbenchAssistant
} from '../../../../../examples/deck/kepler-compute/src/workflow/workbench-assistant-host';
import type {AssistantHost} from '../../../../../examples/deck/kepler-compute/src/assistant-host';
import type {
  WorkbenchDraft,
  WorkbenchStore
} from '../../../../../examples/deck/kepler-compute/src/workflow/workbench-store';
import {makeFixtureDocument, makeFixtureRegistry} from '../workflow-fixtures';

function makeStore(): {store: WorkbenchStore; drafts: WorkbenchDraft[]} {
  const drafts: WorkbenchDraft[] = [];
  const store = {
    registry: makeFixtureRegistry(),
    getState: () => ({workflows: []}),
    proposeDraft: (draft: WorkbenchDraft) => drafts.push(draft)
  } as unknown as WorkbenchStore;
  return {store, drafts};
}

function makeHost(calls: unknown[]): AssistantHost {
  return {
    getContext: () => ({}) as ReturnType<AssistantHost['getContext']>,
    runOperation: async (_approved, operation) => {
      calls.push(operation);
      return {accepted: false, code: 'unknown-operation', reason: 'wrapped host'};
    }
  };
}

describe('workbench assistant host', () => {
  test('reads envelope and flat tool calls and ignores other operations', () => {
    expect(
      readWorkbenchAssistantCall({toolName: 'describeAnalysisOperation', args: {operationId: 'x'}})
    ).toEqual({name: 'describeAnalysisOperation', arguments: {operationId: 'x'}});
    expect(readWorkbenchAssistantCall({name: 'listAnalysisOperations'})).toEqual({
      name: 'listAnalysisOperations',
      arguments: {}
    });
    expect(readWorkbenchAssistantCall({type: 'listAnalysisOperations', query: 'scale'})).toEqual({
      name: 'listAnalysisOperations',
      arguments: {query: 'scale'}
    });
    expect(readWorkbenchAssistantCall({type: 'setFilter', field: 'value'})).toBeNull();
    expect(readWorkbenchAssistantCall('defineWorkflow')).toBeNull();
  });

  test('a draft is proposed without approval and nothing reaches the wrapped host', async () => {
    const {store, drafts} = makeStore();
    const calls: unknown[] = [];
    const host = withWorkbenchAssistant(makeHost(calls), () => store);
    const result = await host.runOperation(false, {
      toolName: 'defineWorkflow',
      args: {
        title: 'Scaled points',
        steps: [
          {id: 'source', operationId: 'fixture-source'},
          {id: 'scale', operationId: 'fixture-scale', inputs: {points: 'source'}}
        ]
      }
    });
    expect(result.accepted).toBe(true);
    expect(drafts).toHaveLength(1);
    expect(drafts[0].source).toBe('assistant');
    expect(drafts[0].document.steps.map(step => step.operationId)).toEqual([
      'fixture-source',
      'fixture-scale'
    ]);
    expect(calls).toEqual([]);
  });

  test('running a saved workflow is refused without approval', async () => {
    const calls: unknown[] = [];
    const runs: unknown[] = [];
    const document = makeFixtureDocument();
    let activeWorkflowId: string | null = null;
    const store = {
      registry: makeFixtureRegistry(),
      getState: () => ({
        workflows: [document],
        activeWorkflowId,
        document: activeWorkflowId ? document : null,
        runner: null
      }),
      getRunner: () => null,
      setNotice: () => undefined,
      setActiveWorkflow: (workflowId: string | null) => {
        activeWorkflowId = workflowId;
      },
      run: async (options: unknown) => {
        runs.push(options);
        return null;
      }
    } as unknown as WorkbenchStore;
    const host = withWorkbenchAssistant(makeHost(calls), () => store);
    const call = {toolName: 'runWorkflow', args: {workflowId: document.id}};
    expect(await host.runOperation(false, call)).toMatchObject({
      accepted: false,
      code: 'not-approved'
    });
    expect(runs).toEqual([]);
    expect(await host.runOperation(true, call)).toMatchObject({accepted: true});
    expect(runs).toHaveLength(1);
    expect(calls).toEqual([]);
  });

  test('a malformed draft is rejected and other operations pass through', async () => {
    const {store, drafts} = makeStore();
    const calls: unknown[] = [];
    const host = withWorkbenchAssistant(makeHost(calls), () => store);
    const rejected = await host.runOperation(true, {toolName: 'defineWorkflow', args: {steps: 3}});
    expect(rejected).toMatchObject({accepted: false, code: 'malformed'});
    expect(drafts).toEqual([]);
    const passed = await host.runOperation(true, {type: 'setFilter'});
    expect(passed).toMatchObject({accepted: false, reason: 'wrapped host'});
    expect(calls).toEqual([{type: 'setFilter'}]);
  });
});

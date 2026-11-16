// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {AssistantHost} from '../assistant-host';
import type {AssistantRouteResult} from '../assistant-operation-router';
import {runWorkbenchAssistantTool} from './workbench-assistant';
import type {WorkbenchStore} from './workbench-store';

const WORKFLOW_TOOL_NAMES = new Set([
  'listAnalysisOperations',
  'describeAnalysisOperation',
  'defineWorkflow',
  'listWorkflows',
  'describeWorkflow',
  'setWorkflowInputs',
  'runWorkflow',
  'listAnalysisTemplates',
  'describeAnalysisTemplate'
]);

const CAPABILITY_BY_STATUS: Readonly<Record<string, string>> = {
  'awaiting-approval': 'workflow draft (awaiting your approval in the workbench)',
  applied: 'workflow inputs changed',
  started: 'workflow run started'
};

const ARGUMENT_KEYS = ['args', 'arguments', 'input', 'parameters'] as const;

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

/**
 * Reads a workflow tool call from the shapes the assistant router accepts: an envelope
 * `{toolName | tool | name, args | arguments | input | parameters}` or a flat `{type, ...args}`.
 * Null when the operation is not a workflow tool.
 */
export function readWorkbenchAssistantCall(
  operation: unknown
): {name: string; arguments: unknown} | null {
  if (!isRecord(operation)) return null;
  for (const key of ['toolName', 'tool', 'name'] as const) {
    const name = operation[key];
    if (typeof name === 'string' && WORKFLOW_TOOL_NAMES.has(name)) {
      const argumentKey = ARGUMENT_KEYS.find(candidate => candidate in operation);
      return {name, arguments: argumentKey ? operation[argumentKey] : {}};
    }
  }
  const type = operation['type'];
  if (typeof type === 'string' && WORKFLOW_TOOL_NAMES.has(type)) {
    const {type: _type, ...rest} = operation;
    return {name: type, arguments: rest};
  }
  return null;
}

/**
 * Adds the workflow tools to an assistant host. Listing and describing read state, and
 * `defineWorkflow` only shows a draft that the user accepts or discards in the workbench, so those
 * need no approval. `setWorkflowInputs` and `runWorkflow` change or run a saved workflow and are
 * refused as `not-approved` unless the user approved that exact call, like every other assistant
 * operation. Operations that are not workflow tools go to the wrapped host unchanged.
 */
export function withWorkbenchAssistant(
  host: AssistantHost,
  getStore: () => WorkbenchStore
): AssistantHost {
  return {
    getContext: () => host.getContext(),
    async runOperation(approved, operation): Promise<AssistantRouteResult> {
      const call = readWorkbenchAssistantCall(operation);
      if (!call) return host.runOperation(approved, operation);
      const result = runWorkbenchAssistantTool(getStore(), call, {approved});
      const status = isRecord(result) ? result['status'] : undefined;
      if (status === 'error')
        return {
          accepted: false,
          code:
            (result as Record<string, unknown>)['code'] === 'unknown-operation'
              ? 'unknown-target'
              : 'malformed',
          reason: String((result as Record<string, unknown>)['message'])
        };
      if (status === 'needs-approval')
        return {
          accepted: false,
          code: 'not-approved',
          reason:
            typeof (result as Record<string, unknown>)['message'] === 'string'
              ? String((result as Record<string, unknown>)['message'])
              : call.name === 'runWorkflow'
                ? 'Running a workflow needs explicit user approval of this exact call.'
                : 'Changing workflow inputs needs explicit user approval of this exact call.'
        };
      return {
        accepted: true,
        route: {
          kind: 'host-capability',
          capability:
            (typeof status === 'string' && CAPABILITY_BY_STATUS[status]) ||
            'workflow information lookup',
          request: {tool: call.name, result}
        }
      };
    }
  };
}

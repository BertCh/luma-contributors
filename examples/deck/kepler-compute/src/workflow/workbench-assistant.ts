// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  describeWorkflowOperationForAssistant,
  describeWorkflowTemplateForAssistant,
  listWorkflowTemplatesForAssistant,
  parseWorkflowTemplateForAssistant,
  prepareWorkflowDraftForAssistant,
  validateWorkflowForAssistant,
  getNewWorkflowAssistantWarnings,
  serializeWorkflowDocument,
  describeWorkflowForAssistant,
  getWorkflowAssistantTools,
  listWorkflowOperationsForAssistant,
  listWorkflowsForAssistant,
  parseWorkflowInputChanges,
  parseWorkflowRunRequest,
  planWorkflow,
  parseWorkflowDraft,
  WorkflowError,
  type WorkflowDocument,
  type KeplerJSONValue
} from '@luma-contributors/kepler';
import type {WorkbenchStore} from './workbench-store';
import {
  createWorkbenchAnalysisDocument,
  getWorkbenchAnalysisProfiles
} from './workbench-analysis-launch';

/** Tool definitions advertised by the host, including approval-gated edits and runs. */
export function getWorkbenchAssistantTools(): ReturnType<typeof getWorkflowAssistantTools> {
  return getWorkflowAssistantTools();
}

/** Proposals stay drafts; editing and running saved workflows require explicit approval. */
export function runWorkbenchAssistantTool(
  store: WorkbenchStore,
  call: {name: string; arguments: unknown},
  options?: {approved?: boolean}
): KeplerJSONValue {
  try {
    switch (call.name) {
      case 'listAnalysisOperations':
        return listWorkflowOperationsForAssistant(store.registry, call.arguments);
      case 'describeAnalysisOperation':
        return describeWorkflowOperationForAssistant(store.registry, call.arguments);
      case 'listAnalysisTemplates':
        return listWorkflowTemplatesForAssistant(call.arguments);
      case 'describeAnalysisTemplate':
        return describeWorkflowTemplateForAssistant(store.registry, call.arguments);
      case 'listWorkflows':
        return listWorkflowsForAssistant(store.getState().workflows, store.registry);
      case 'describeWorkflow': {
        const state = store.getState();
        const description = describeWorkflowForAssistant(
          state.workflows,
          store.registry,
          call.arguments
        );
        if (
          description &&
          typeof description === 'object' &&
          !Array.isArray(description) &&
          'id' in description &&
          description['id'] === state.activeWorkflowId
        ) {
          return describeWorkflowForAssistant(
            state.workflows,
            store.registry,
            call.arguments,
            state.runner
          );
        }
        return description;
      }
      case 'setWorkflowInputs': {
        const {workflowId, document, changes} = parseWorkflowInputChanges(
          store.getState().workflows,
          call.arguments
        );
        if (!options?.approved)
          return {status: 'needs-approval', workflowTitle: document.title, changes};
        store.setActiveWorkflow(workflowId);
        requireActiveWorkflow(store, workflowId, document.title, 'input changes could be applied');
        let appliedChanges = changes;
        let editFailure: unknown;
        if (
          !store.edit('assistant-inputs', current => {
            if (current.id !== workflowId) {
              throw new WorkflowError(
                'operation-failed',
                `${document.title}: Workflow activation changed before input changes could be applied. Try again.`
              );
            }
            try {
              const patch = parseWorkflowInputChanges([current], {
                workflowId,
                inputs: Object.fromEntries(changes.map(change => [change.inputId, change.next]))
              });
              appliedChanges = patch.changes;
              return patch.document;
            } catch (error) {
              editFailure = error;
              throw error;
            }
          })
        ) {
          if (editFailure) throw editFailure;
          throw new WorkflowError(
            'operation-failed',
            store.getState().notice?.message ??
              `${document.title}: Workflow inputs could not be applied.`
          );
        }
        return {status: 'applied', changes: appliedChanges};
      }
      case 'runWorkflow': {
        const state = store.getState();
        const {workflowId, target} = parseWorkflowRunRequest(state.workflows, call.arguments);
        const document = state.workflows.find(workflow => workflow.id === workflowId)!;
        if (!options?.approved) {
          const issues = validateWorkflowForAssistant(document, {
            registry: store.registry,
            host: store.host,
            target
          });
          store.recordAssistantReview?.(document, issues, target);
          return {
            status: 'needs-approval',
            workflowTitle: document.title,
            steps: countWorkflowRunSteps(store, document, target),
            cachedResultsKnown: state.activeWorkflowId === workflowId,
            issues: issues.map(({severity, code, message}) => ({severity, code, message})),
            message: [
              'Running a workflow needs explicit user approval of this exact call.',
              ...issues.filter(issue => issue.severity === 'warning').map(issue => issue.message)
            ].join('\n')
          };
        }
        store.setActiveWorkflow(workflowId);
        requireActiveWorkflow(store, workflowId, document.title, 'the run could start');
        const current = store.getState().document!;
        if (
          JSON.stringify(serializeWorkflowDocument(current)) !==
          JSON.stringify(serializeWorkflowDocument(document))
        )
          throw new WorkflowError(
            'malformed',
            `${document.title}: The approved workflow changed before the run could start. Review the current whole plan again.`
          );
        const issues = validateWorkflowForAssistant(current, {
          registry: store.registry,
          host: store.host,
          target
        });
        if (issues.some(issue => issue.severity === 'error'))
          throw new WorkflowError(
            'malformed',
            issues
              .filter(issue => issue.severity === 'error')
              .map(issue => issue.message)
              .join('\n')
          );
        const reviewedIssues = store.getReviewedAssistantIssues?.(current, target);
        if (!reviewedIssues) {
          return {
            status: 'needs-approval',
            workflowTitle: current.title,
            message:
              'The workflow or execution target changed, or has not been reviewed. Request runWorkflow again and review the current whole plan before approving.'
          };
        }
        const newWarnings = getNewWorkflowAssistantWarnings(issues, reviewedIssues);
        if (newWarnings.length) {
          const message = `Review the updated warnings with runWorkflow before approving this exact call again.\n${newWarnings.map(issue => issue.message).join('\n')}`;
          store.setNotice({kind: 'info', message});
          return {
            status: 'needs-approval',
            workflowTitle: document.title,
            message,
            issues: issues.map(({severity, code, message}) => ({severity, code, message}))
          };
        }
        void store.run({target}).catch(error =>
          store.setNotice({
            kind: 'error',
            message:
              error instanceof Error ? error.message : `${document.title}: Workflow run failed.`
          })
        );
        return {
          status: 'started',
          workflowTitle: document.title,
          issues: issues.map(({severity, code, message}) => ({severity, code, message}))
        };
      }
      case 'defineWorkflow': {
        const usesTemplate =
          call.arguments !== null &&
          typeof call.arguments === 'object' &&
          Object.hasOwn(call.arguments, 'templateId');
        const document = usesTemplate
          ? (() => {
              const {template, mapping} = parseWorkflowTemplateForAssistant(
                call.arguments,
                getWorkbenchAnalysisProfiles(store)
              );
              return createWorkbenchAnalysisDocument(template, store, mapping);
            })()
          : parseWorkflowDraft(call.arguments, {
              registry: store.registry,
              takenIds: store.getState().workflows.map(document => document.id),
              preserveExplicitParameters: true
            }).document;
        const draft = prepareWorkflowDraftForAssistant(document, {
          registry: store.registry,
          host: store.host
        });
        store.proposeDraft({...draft, source: 'assistant'});
        return {
          status: 'awaiting-approval',
          workflowTitle: draft.document.title,
          stepCount: draft.document.steps.length,
          issues: draft.issues.map(({severity, code, message}) => ({severity, code, message})),
          note: 'The workflow is shown to the user as a draft. It has not been added or run.'
        };
      }
      default:
        return {
          status: 'error',
          code: 'malformed',
          message: `Assistant tool "${call.name}" is not available.`
        };
    }
  } catch (error) {
    if (error instanceof WorkflowError) {
      return {status: 'error', code: error.code, message: error.message};
    }
    return {
      status: 'error',
      code: 'operation-failed',
      message: error instanceof Error ? error.message : 'The assistant tool could not complete.'
    };
  }
}

function requireActiveWorkflow(
  store: WorkbenchStore,
  workflowId: string,
  title: string,
  action: string
): void {
  const state = store.getState();
  if (state.activeWorkflowId !== workflowId || state.document?.id !== workflowId) {
    throw new WorkflowError(
      'operation-failed',
      `${title}: Workflow activation changed before ${action}. Try again.`
    );
  }
}

function countWorkflowRunSteps(
  store: WorkbenchStore,
  document: WorkflowDocument,
  target?: string
): number {
  const snapshot = store.getState();
  const runner = snapshot.activeWorkflowId === document.id ? snapshot.runner : null;
  const observedKinds = Object.fromEntries(
    Object.entries(runner?.steps ?? {}).map(([stepId, state]) => [
      stepId,
      Object.fromEntries(
        Object.entries(state.result?.outputs ?? {}).map(([output, data]) => [output, data.kind])
      )
    ])
  );
  const plan = planWorkflow(document, {registry: store.registry, host: store.host, observedKinds});
  if (target && plan.steps[target].status !== 'ready') return 0;
  const included = new Set<string>();
  function includeStep(stepId: string): void {
    if (included.has(stepId)) return;
    included.add(stepId);
    for (const upstream of plan.steps[stepId].upstream) includeStep(upstream);
  }
  if (target) includeStep(target);
  else for (const stepId of plan.order) included.add(stepId);
  return plan.order.filter(stepId => {
    const planned = plan.steps[stepId];
    const current = runner?.steps[stepId];
    return (
      included.has(stepId) &&
      planned.status === 'ready' &&
      (current?.result?.key !== planned.key || current?.status === 'error')
    );
  }).length;
}

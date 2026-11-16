// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {useEffect, useMemo, useRef, useState} from 'react';
import styled from 'styled-components';
import {
  estimateWorkflowPreflight,
  makeWorkflowComputationRecord,
  type WorkflowRunRecord,
  type WorkflowOperationRegistry
} from '@luma-contributors/kepler';
import {mountLadderLens} from '@luma-contributors/lenses/computation-views';
import {WORKBENCH_COLORS, WORKBENCH_STATUS, WORKBENCH_TEST_IDS} from './workbench-contract';
import type {WorkbenchStore} from './workbench-store';
import {WorkbenchButton, WorkbenchMuted} from './workbench-styles';
import {useWorkbenchState} from './use-workbench';

const RunLogStrip = styled.div`
  display: flex;
  flex-direction: column;
  flex: 0 1 auto;
  min-height: 27px;
  max-height: 50%;
  border-top: 1px solid ${WORKBENCH_COLORS.border};
`;
const RunLogHeader = styled.div`
  display: flex;
  flex: none;
  min-height: 26px;
  & > button:first-child {
    flex: 1;
    min-width: 0;
    justify-content: flex-start;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  & > button + button { flex: none; padding: 0 6px; }
`;
const RunList = styled.div`
  flex: 1 1 auto;
  min-height: 0;
  max-height: 160px;
  overflow: auto;
  padding: 4px 8px;
  & summary { cursor: pointer; min-height: 24px; }
  & summary:focus-visible { outline: 2px solid ${WORKBENCH_COLORS.focus}; }
  & ul { margin: 4px 0; padding-left: 20px; }
  & li { margin-bottom: 5px; }
`;

export function formatWorkbenchDuration(milliseconds: number): string {
  return milliseconds < 1000
    ? `${Math.round(milliseconds).toLocaleString()} ms`
    : `${(milliseconds / 1000).toLocaleString(undefined, {maximumFractionDigits: 1})} s`;
}

export function formatWorkbenchRunLine(run: WorkflowRunRecord): string {
  // Runner ids encode their cumulative sequence. Other producers may use opaque identities.
  const prefix = `${run.workflowId}-`;
  const suffix = run.runId.startsWith(prefix) ? run.runId.slice(prefix.length) : '';
  const sequence = /^[1-9]\d*$/.test(suffix) ? Number(suffix) : NaN;
  const identifier = Number.isSafeInteger(sequence) ? sequence.toLocaleString() : run.runId;
  const cached = run.steps.filter(step => step.cacheHit).length;
  return `Run ${identifier} · ${run.status} · ${run.steps.length.toLocaleString()} steps · ${formatWorkbenchDuration(run.durationMilliseconds)} · ${cached.toLocaleString()} cached`;
}

/** Uses the existing computation inspector without inventing GPU timings or kernel captures. */
export function WorkbenchRunTrace({
  run,
  registry
}: {
  run: WorkflowRunRecord;
  registry: WorkflowOperationRegistry;
}): React.JSX.Element {
  const [open, setOpen] = useState(false);
  const container = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!open || !container.current) return;
    const lens = mountLadderLens(container.current, {
      record: makeWorkflowComputationRecord(run, registry),
      rung: 'table',
      urlState: false,
      bind: `workflow-${run.runId}`,
      caption: 'Workflow step provenance; durations are wall time.'
    });
    return () => lens.dispose();
  }, [open, run, registry]);
  return (
    <div>
      <WorkbenchButton aria-expanded={open} onClick={() => setOpen(!open)}>
        {open ? 'Close computation views' : 'Open computation views'}
      </WorkbenchButton>
      {open && (
        <>
          <WorkbenchMuted>
            Step trace with CPU table boundaries. GPU kernel timings were not recorded.
          </WorkbenchMuted>
          <div ref={container} data-testid="workflow-computation-views" />
        </>
      )}
    </div>
  );
}

/** Run records of the active workflow. */
export function WorkbenchRunLog({store}: {store: WorkbenchStore}): React.JSX.Element | null {
  const state = useWorkbenchState(store);
  const preflight = useMemo(
    () =>
      state.document
        ? estimateWorkflowPreflight(state.document, store.registry, {
            state: state.runner,
            profileSources: true,
            host: store.host
          })
        : undefined,
    [state.document, state.runner, store.registry, store.host]
  );
  const labels = useMemo(
    () =>
      new Map(
        state.document?.steps.map(step => [
          step.id,
          step.label ?? store.registry.get(step.operationId)?.title ?? step.operationId
        ])
      ),
    [state.document, store.registry]
  );
  const runs = useMemo(
    () =>
      state.runLog.map(run => ({
        run,
        line: formatWorkbenchRunLine(run)
      })),
    [state.runLog]
  );
  return (
    <RunLogStrip>
      <RunLogHeader>
        <WorkbenchButton
          data-testid={WORKBENCH_TEST_IDS.runLogToggle}
          aria-expanded={state.dock.logOpen}
          aria-controls={WORKBENCH_TEST_IDS.runLog}
          title={runs[0]?.line ?? 'No runs yet'}
          onClick={() => store.setDock({logOpen: !state.dock.logOpen})}
        >
          {state.dock.logOpen ? '▾' : '▸'} {runs[0]?.line ?? 'No runs yet'}
        </WorkbenchButton>
        {preflight && (
          <WorkbenchMuted title="Cold-run logical GPU scratch only; inputs, outputs, staging and host heap excluded.">
            Scratch:{' '}
            {preflight.peakScratchBytes === null
              ? 'unknown'
              : `${preflight.peakScratchBytes.toLocaleString()} B`}
            {' · '}Nodes: {preflight.commandNodeCount ?? 'unknown'}
          </WorkbenchMuted>
        )}
        {runs[0] && (
          <WorkbenchButton
            data-testid={WORKBENCH_TEST_IDS.runReport}
            title="Report for the latest run"
            onClick={() => store.setDialog('import-export')}
          >
            Report
          </WorkbenchButton>
        )}
      </RunLogHeader>
      {state.dock.logOpen && (
        <RunList
          id={WORKBENCH_TEST_IDS.runLog}
          data-testid={WORKBENCH_TEST_IDS.runLog}
          role="region"
          aria-label="Run log"
          tabIndex={0}
        >
          {runs.length === 0 && (
            <WorkbenchMuted>No runs yet. Run the workflow to see its steps.</WorkbenchMuted>
          )}
          {runs.map(({run, line}) => (
            <details key={run.runId}>
              <summary data-status={run.status}>{line}</summary>
              <WorkbenchRunTrace run={run} registry={store.registry} />
              <ul>
                {run.steps.map((step, index) => (
                  <li key={`${step.stepId}:${index}`} data-status={step.status}>
                    <WorkbenchButton onClick={() => store.select([step.stepId])}>
                      {labels.get(step.stepId) ??
                        store.registry.get(step.operationId)?.title ??
                        step.operationId}
                    </WorkbenchButton>
                    {' · '}
                    {step.status === 'done' || step.status === 'error'
                      ? WORKBENCH_STATUS[step.status].label
                      : step.status}
                    {' · '}
                    {step.engine === 'gpu'
                      ? 'GPU'
                      : step.engine === 'cpu-reference'
                        ? 'CPU'
                        : step.engine === 'host'
                          ? 'Host'
                          : 'No engine'}
                    {' · '}
                    {step.cacheHit ? 'Cached' : formatWorkbenchDuration(step.durationMilliseconds)}
                    {Object.entries(step.outputRows ?? {}).map(([output, rows]) => (
                      <span key={output}>
                        {' · '}
                        {store.registry
                          .get(step.operationId)
                          ?.outputs.find(port => port.name === output)?.label ?? output}
                        : {rows.toLocaleString()} rows
                      </span>
                    ))}
                    {step.fallbackReason && <div>{step.fallbackReason}</div>}
                    {step.error && <div>{step.error.message}</div>}
                  </li>
                ))}
              </ul>
            </details>
          ))}
        </RunList>
      )}
    </RunLogStrip>
  );
}

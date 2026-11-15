// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {useEffect, useId, useMemo, useRef, useState, useSyncExternalStore} from 'react';
import type {ReactNode} from 'react';
import styled from 'styled-components';
import {
  describeWorkflow,
  exportWorkflowJSON,
  exportWorkflowTypeScript,
  importWorkflowJSON,
  instantiateWorkflowTemplate,
  listWorkflowTemplates,
  makeWorkflowFromCatalogHistory,
  makeWorkflowRunReport,
  makeWorkflowDot,
  planWorkflow,
  WORKFLOW_JSON_LIMITS,
  type WorkflowIssue,
  type WorkflowPortKind,
  type WorkflowStepStatus
} from '@luma-contributors/kepler';
import {getCatalogSampleInputs} from '../analysis-catalog-input';
import {listCatalogInvocations, subscribeCatalogInvocations} from '../catalog-invocation-store';
import {
  WORKBENCH_CATEGORY_LABELS,
  WORKBENCH_COLORS,
  WORKBENCH_DOCK,
  WORKBENCH_PORT_COLORS,
  WORKBENCH_PORT_LABELS,
  WORKBENCH_SHORTCUTS,
  WORKBENCH_STATUS,
  WORKBENCH_TEST_IDS
} from './workbench-contract';
import type {WorkbenchDraft, WorkbenchState, WorkbenchStore} from './workbench-store';
import {
  WorkbenchButton,
  WorkbenchField,
  WorkbenchIconButton,
  WorkbenchPortSwatch,
  WorkbenchSection,
  WorkbenchStatusBadge,
  WorkbenchSurface,
  WorkbenchTable,
  WorkbenchTabs
} from './workbench-styles';
import {useWorkbenchState} from './use-workbench';

const Scrim = styled.div`
  position: fixed;
  inset: 0;
  z-index: 1000;
  display: grid;
  place-items: center;
  background: ${WORKBENCH_COLORS.surfaceSunken}cc;
`;
const ModalPanel = styled(WorkbenchSurface)`
  width: min(880px, calc(100vw - 32px));
  max-height: 80vh;
  overflow: auto;
  border: 1px solid ${WORKBENCH_COLORS.borderStrong};
  border-radius: 4px;
  & summary { cursor: pointer; min-height: 24px; }
  & summary:focus-visible { outline: 2px solid ${WORKBENCH_COLORS.focus}; }
  & input[type='checkbox'] { min-width: 24px; min-height: 24px; }
`;
const Header = styled.div`
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  padding: 10px 12px;
  border-bottom: 1px solid ${WORKBENCH_COLORS.border};
  & h2 { margin: 0; font-size: 14px; }
`;
const Actions = styled.div`
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  align-items: center;
`;
const Cards = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
  gap: 8px;
`;
const Card = styled.article`
  padding: 10px;
  border: 1px solid ${WORKBENCH_COLORS.border};
  & h4 { margin: 0; font-size: 12px; }
`;
const Outline = styled.pre`
  max-height: 220px;
  overflow: auto;
  white-space: pre-wrap;
  overflow-wrap: anywhere;
  font: 11px/1.5 monospace;
`;
const TextArea = styled.textarea`
  min-height: 260px;
  resize: vertical;
  font-family: monospace !important;
`;
const DraftPanel = styled(WorkbenchSurface)<{$bottom: number}>`
  position: fixed;
  z-index: 900;
  right: 12px;
  bottom: ${({$bottom}) => $bottom + 12}px;
  width: min(460px, calc(100vw - 24px));
  max-height: min(60vh, 520px);
  overflow: auto;
  border: 1px solid ${WORKBENCH_COLORS.borderStrong};
`;

function readErrorMessage(error: unknown): string {
  return error instanceof Error ? error.message : 'The workflow could not be prepared.';
}

/** Formats only the platform modifier; shortcut order remains the shared contract's order. */
export function formatWorkbenchShortcut(keys: string, platform: string): string {
  return keys
    .split('+')
    .map(key => (key === 'Mod' ? (/mac|iphone|ipad/i.test(platform) ? '⌘' : 'Ctrl') : key))
    .join('+');
}

/** Stable severity ordering preserves the planner's order within each severity. */
export function sortWorkbenchDraftIssues(issues: readonly WorkflowIssue[]): WorkflowIssue[] {
  const ranks = {error: 0, warning: 1, info: 2};
  return [...issues].sort((first, second) => ranks[first.severity] - ranks[second.severity]);
}

function Modal({
  title,
  close,
  children
}: {
  title: string;
  close: () => void;
  children: ReactNode;
}): React.JSX.Element {
  const headingId = useId();
  const panel = useRef<HTMLDivElement>(null);
  const closeReference = useRef(close);
  closeReference.current = close;
  useEffect(() => {
    const element = panel.current;
    if (!element) return;
    const owner = element.ownerDocument;
    const opener = owner.activeElement;
    const controls = () =>
      Array.from(
        element.querySelectorAll<HTMLElement>(
          'button:not(:disabled), [href], input:not(:disabled), select:not(:disabled), textarea:not(:disabled), summary, [tabindex="0"]'
        )
      ).filter(
        control =>
          control.tabIndex >= 0 &&
          !control.hidden &&
          !control.closest('[hidden]') &&
          !control.closest('details:not([open]) :not(summary)')
      );
    const focusFirst = () => (controls()[0] ?? element).focus();
    focusFirst();
    const keydown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        event.stopPropagation();
        event.preventDefault();
        closeReference.current();
      } else if (event.key === 'Tab') {
        const focusable = controls();
        const index = focusable.indexOf(owner.activeElement as HTMLElement);
        if (
          !focusable.length ||
          (event.shiftKey ? index <= 0 : index === focusable.length - 1 || index < 0)
        ) {
          event.preventDefault();
          (event.shiftKey ? (focusable.at(-1) ?? element) : (focusable[0] ?? element)).focus();
        }
      }
    };
    const focusin = (event: FocusEvent) => {
      if (event.target instanceof Node && !element.contains(event.target)) focusFirst();
    };
    owner.addEventListener('keydown', keydown, true);
    owner.addEventListener('focusin', focusin);
    return () => {
      owner.removeEventListener('keydown', keydown, true);
      owner.removeEventListener('focusin', focusin);
      if (opener instanceof HTMLElement && opener.isConnected) opener.focus();
    };
  }, []);
  return (
    <Scrim>
      <ModalPanel
        ref={panel}
        role="dialog"
        aria-modal="true"
        aria-labelledby={headingId}
        tabIndex={-1}
      >
        <Header>
          <h2 id={headingId}>{title}</h2>
          <WorkbenchButton onClick={close}>Close</WorkbenchButton>
        </Header>
        {children}
      </ModalPanel>
    </Scrim>
  );
}

function Templates({store}: {store: WorkbenchStore}): React.JSX.Element {
  const state = useWorkbenchState(store);
  const [error, setError] = useState('');
  const [deleteTemplateId, setDeleteTemplateId] = useState<string | null>(null);
  const userTemplates = useMemo(
    () =>
      state.templates.map(document => ({
        document,
        outline: describeWorkflow(document, store.registry)
      })),
    [state.templates, store.registry]
  );
  function insertTemplate(document: Parameters<WorkbenchStore['insertWorkflow']>[0]): void {
    if (store.insertWorkflow(document)) store.setDialog(null);
    else setError(store.getState().notice?.message ?? 'The steps could not be inserted.');
  }
  const groups = useMemo(() => {
    const templates = listWorkflowTemplates();
    return [...new Set(templates.map(template => template.category))].map(category => ({
      category,
      templates: templates
        .filter(template => template.category === category)
        .map(template => ({
          ...template,
          outline: describeWorkflow(template.document, store.registry)
        }))
    }));
  }, [store.registry]);
  const categoryLabels = {
    points: WORKBENCH_PORT_LABELS.point,
    network: WORKBENCH_CATEGORY_LABELS.network,
    polygons: WORKBENCH_PORT_LABELS.polygon,
    statistics: WORKBENCH_CATEGORY_LABELS.analysis,
    glue: WORKBENCH_CATEGORY_LABELS.table
  };
  return (
    <div data-testid={WORKBENCH_TEST_IDS.templates}>
      {error && <p role="alert">{error}</p>}
      <WorkbenchSection>
        <h3>Your templates</h3>
        <WorkbenchButton
          data-testid={WORKBENCH_TEST_IDS.saveTemplate}
          disabled={!state.document?.steps.length}
          onClick={() => {
            try {
              if (!store.saveTemplate())
                setError(store.getState().notice?.message ?? 'The template could not be saved.');
            } catch (failure) {
              setError(readErrorMessage(failure));
            }
          }}
        >
          Save current workflow as a template
        </WorkbenchButton>
        {!state.document?.steps.length && <p>Select a workflow with steps to save a template.</p>}
        <p>Adds these steps to the current workflow so you can connect them to your own.</p>
        {!userTemplates.length && <p>No saved templates.</p>}
        <Cards>
          {userTemplates.map(({document, outline}) => (
            <Card key={document.id}>
              <h4>{document.title}</h4>
              <p>{document.steps.length.toLocaleString()} steps</p>
              <details>
                <summary aria-label={`View outline: ${document.title}`}>View outline</summary>
                <Outline>{outline}</Outline>
              </details>
              <Actions>
                <WorkbenchButton
                  data-testid={WORKBENCH_TEST_IDS.userTemplate(document.id)}
                  aria-label={`Use template: ${document.title}`}
                  onClick={() => {
                    try {
                      if (store.useTemplate(document.id)) store.setDialog(null);
                      else
                        setError(
                          store.getState().notice?.message ?? 'The template could not be used.'
                        );
                    } catch (failure) {
                      setError(readErrorMessage(failure));
                    }
                  }}
                >
                  Use template
                </WorkbenchButton>
                <WorkbenchButton
                  data-testid={WORKBENCH_TEST_IDS.insertWorkflow(document.id)}
                  aria-label={`Insert into current workflow: ${document.title}`}
                  disabled={!state.document}
                  onClick={() => {
                    try {
                      insertTemplate(document);
                    } catch (failure) {
                      setError(readErrorMessage(failure));
                    }
                  }}
                >
                  Insert into current workflow
                </WorkbenchButton>
                <WorkbenchIconButton
                  data-testid={WORKBENCH_TEST_IDS.userTemplateDelete(document.id)}
                  aria-label={`Delete template ${document.title}`}
                  title={`Delete template ${document.title}`}
                  onClick={() => setDeleteTemplateId(document.id)}
                >
                  <svg
                    aria-hidden="true"
                    viewBox="0 0 16 16"
                    width="12"
                    height="12"
                    fill="none"
                    stroke="currentColor"
                  >
                    <path d="M3 4h10M6 4V2h4v2M4 4l1 10h6l1-10M7 6v6M9 6v6" />
                  </svg>
                </WorkbenchIconButton>
              </Actions>
              {deleteTemplateId === document.id && (
                <div role="group" aria-label={`Delete template ${document.title}`}>
                  <p>Delete this template?</p>
                  <Actions>
                    <WorkbenchButton
                      $variant="danger"
                      aria-label={`Delete template: ${document.title}`}
                      onClick={() => {
                        try {
                          store.deleteTemplate(document.id);
                          setDeleteTemplateId(null);
                        } catch (failure) {
                          setError(readErrorMessage(failure));
                        }
                      }}
                    >
                      Delete
                    </WorkbenchButton>
                    <WorkbenchButton
                      aria-label={`Cancel deleting template: ${document.title}`}
                      onClick={() => setDeleteTemplateId(null)}
                    >
                      Cancel
                    </WorkbenchButton>
                  </Actions>
                </div>
              )}
            </Card>
          ))}
        </Cards>
      </WorkbenchSection>
      {groups.map(group => (
        <WorkbenchSection key={group.category}>
          <h3>{categoryLabels[group.category]}</h3>
          <Cards>
            {group.templates.map(template => (
              <Card key={template.id}>
                <h4>{template.title}</h4>
                <p>{template.summary}</p>
                <p>{template.document.steps.length.toLocaleString()} steps</p>
                <details>
                  <summary aria-label={`View outline: ${template.title}`}>View outline</summary>
                  <Outline>{template.outline}</Outline>
                </details>
                <WorkbenchButton
                  data-testid={WORKBENCH_TEST_IDS.template(template.id)}
                  aria-label={`Use template: ${template.title}`}
                  onClick={() => {
                    try {
                      const document = instantiateWorkflowTemplate(
                        template.id,
                        store.getState().workflows.map(workflow => workflow.id)
                      );
                      if (store.addWorkflow(document)) store.setDialog(null);
                      else
                        setError(
                          store.getState().notice?.message ?? 'The workflow could not be added.'
                        );
                    } catch (failure) {
                      setError(readErrorMessage(failure));
                    }
                  }}
                >
                  Use template
                </WorkbenchButton>
                <WorkbenchButton
                  data-testid={WORKBENCH_TEST_IDS.insertWorkflow(template.id)}
                  aria-label={`Insert into current workflow: ${template.title}`}
                  disabled={!state.document}
                  onClick={() => {
                    try {
                      insertTemplate(
                        instantiateWorkflowTemplate(
                          template.id,
                          store.getState().workflows.map(workflow => workflow.id)
                        )
                      );
                    } catch (failure) {
                      setError(readErrorMessage(failure));
                    }
                  }}
                >
                  Insert into current workflow
                </WorkbenchButton>
              </Card>
            ))}
          </Cards>
        </WorkbenchSection>
      ))}
    </div>
  );
}

export type WorkbenchTransferTab = 'json' | 'typescript' | 'import' | 'history' | 'report' | 'dot';
const TRANSFER_TABS: readonly {id: WorkbenchTransferTab; label: string}[] = [
  {id: 'json', label: 'Export JSON'},
  {id: 'typescript', label: 'Export TypeScript'},
  {id: 'import', label: 'Import'},
  {id: 'history', label: 'From catalog history'},
  {id: 'report', label: 'Run report'},
  {id: 'dot', label: 'Graph (DOT)'}
];

function downloadWorkflow(text: string, filename: string, type: string): void {
  const address = URL.createObjectURL(new Blob([text], {type}));
  const anchor = document.createElement('a');
  try {
    anchor.href = address;
    anchor.download = filename;
    document.body.append(anchor);
    anchor.click();
  } finally {
    anchor.remove();
    // Let the browser start reading the object URL before releasing it.
    setTimeout(() => URL.revokeObjectURL(address), 0);
  }
}

/** Exported for static rendering of each transient tab without a browser. */
export function WorkbenchTransfer({
  store,
  initialTab = 'json'
}: {
  store: WorkbenchStore;
  initialTab?: WorkbenchTransferTab;
}): React.JSX.Element {
  const state = useWorkbenchState(store);
  const [tab, setTab] = useState<WorkbenchTransferTab>(initialTab);
  const [text, setText] = useState('');
  const [error, setError] = useState('');
  const [confirmation, setConfirmation] = useState('');
  const fileInput = useRef<HTMLInputElement>(null);
  const fileReadGeneration = useRef(0);
  const copyGeneration = useRef(0);
  const prefix = useId();
  const exported = useMemo(() => {
    if (!state.document || tab === 'import' || tab === 'history') return {text: '', error: ''};
    if (tab === 'report' && (!state.runner || !state.runLog[0])) return {text: '', error: ''};
    try {
      return {
        text:
          tab === 'json'
            ? exportWorkflowJSON(state.document)
            : tab === 'typescript'
              ? exportWorkflowTypeScript(state.document, store.registry)
              : tab === 'dot'
                ? makeWorkflowDot(state.document, store.registry, state.runner)
                : makeWorkflowRunReport(
                    state.document,
                    store.registry,
                    state.runner!,
                    state.runLog[0]
                  ),
        error: ''
      };
    } catch (failure) {
      return {text: '', error: readErrorMessage(failure)};
    }
  }, [state.document, store.registry, state.runner, state.runLog, tab]);
  const exportAvailable = Boolean(
    state.document && (tab !== 'report' || (state.runner && state.runLog[0]))
  );
  const exportLabel =
    tab === 'json'
      ? 'Workflow JSON'
      : tab === 'typescript'
        ? 'Workflow TypeScript'
        : tab === 'report'
          ? 'Run report'
          : 'Graph (DOT)';
  const currentExport = useRef(exported);
  const confirmationExport = useRef(exported);
  currentExport.current = exported;
  useEffect(
    () => () => {
      fileReadGeneration.current++;
      copyGeneration.current++;
    },
    []
  );
  function changeTab(next: WorkbenchTransferTab): void {
    fileReadGeneration.current++;
    copyGeneration.current++;
    setTab(next);
    setError('');
    setConfirmation('');
  }
  return (
    <div data-testid={WORKBENCH_TEST_IDS.importExport}>
      <WorkbenchTabs aria-label="Import and export">
        {TRANSFER_TABS.map((item, index) => (
          <button
            key={item.id}
            type="button"
            role="tab"
            id={`${prefix}-tab-${item.id}`}
            aria-controls={`${prefix}-panel-${item.id}`}
            aria-selected={tab === item.id}
            tabIndex={tab === item.id ? 0 : -1}
            data-testid={
              item.id === 'json'
                ? WORKBENCH_TEST_IDS.exportJSON
                : item.id === 'typescript'
                  ? WORKBENCH_TEST_IDS.exportTypeScript
                  : item.id === 'report'
                    ? WORKBENCH_TEST_IDS.runReport
                    : item.id === 'dot'
                      ? WORKBENCH_TEST_IDS.graphDot
                      : undefined
            }
            onClick={() => changeTab(item.id)}
            onKeyDown={event => {
              const next =
                event.key === 'ArrowRight'
                  ? (index + 1) % TRANSFER_TABS.length
                  : event.key === 'ArrowLeft'
                    ? (index + TRANSFER_TABS.length - 1) % TRANSFER_TABS.length
                    : event.key === 'Home'
                      ? 0
                      : event.key === 'End'
                        ? TRANSFER_TABS.length - 1
                        : null;
              if (next !== null) {
                event.preventDefault();
                changeTab(TRANSFER_TABS[next].id);
                event.currentTarget.parentElement
                  ?.querySelectorAll<HTMLButtonElement>('[role="tab"]')
                  [next]?.focus();
              }
            }}
          >
            {item.label}
          </button>
        ))}
      </WorkbenchTabs>
      {TRANSFER_TABS.filter(item => item.id !== tab).map(item => (
        <div
          key={item.id}
          role="tabpanel"
          id={`${prefix}-panel-${item.id}`}
          aria-labelledby={`${prefix}-tab-${item.id}`}
          hidden
        />
      ))}
      <WorkbenchSection
        role="tabpanel"
        id={`${prefix}-panel-${tab}`}
        aria-labelledby={`${prefix}-tab-${tab}`}
      >
        {(tab === 'json' || tab === 'typescript' || tab === 'report' || tab === 'dot') && (
          <>
            {!state.document && <p>Select or create a workflow to export it.</p>}
            {tab === 'report' && (!state.runner || !state.runLog[0]) && (
              <p>Run the workflow to get a report.</p>
            )}
            {exported.error && <p role="alert">{exported.error}</p>}
            <WorkbenchField>
              <label htmlFor={`${prefix}-export-text`}>{exportLabel}</label>
              <TextArea
                id={`${prefix}-export-text`}
                data-testid={
                  tab === 'report'
                    ? WORKBENCH_TEST_IDS.runReportText
                    : WORKBENCH_TEST_IDS.exportText
                }
                readOnly
                aria-label={exportLabel}
                disabled={!exportAvailable}
                value={exported.text}
              />
            </WorkbenchField>
            <Actions>
              <WorkbenchButton
                disabled={!exportAvailable || Boolean(exported.error)}
                onClick={async () => {
                  const generation = ++copyGeneration.current;
                  const confirm = (message: string) => {
                    if (generation !== copyGeneration.current || currentExport.current !== exported)
                      return;
                    confirmationExport.current = exported;
                    setConfirmation(message);
                  };
                  try {
                    if (!globalThis.navigator?.clipboard?.writeText) {
                      confirm(
                        'Clipboard unavailable. Select the export text and copy it manually.'
                      );
                      return;
                    }
                    await navigator.clipboard.writeText(exported.text);
                    confirm('Copied');
                  } catch {
                    confirm('Could not copy. Select the export text and copy it manually.');
                  }
                }}
              >
                Copy
              </WorkbenchButton>
              <WorkbenchButton
                disabled={!exportAvailable || Boolean(exported.error)}
                onClick={() => {
                  if (!state.document) return;
                  try {
                    downloadWorkflow(
                      exported.text,
                      tab === 'report'
                        ? `${state.document.id}-run-${state.runLog[0].runId.match(/-(\d+)$/)?.[1] ?? state.runLog.length}.md`
                        : tab === 'dot'
                          ? `${state.document.id}.dot`
                          : `${state.document.id}.workflow.${tab === 'json' ? 'json' : 'ts'}`,
                      tab === 'json'
                        ? 'application/json'
                        : tab === 'typescript'
                          ? 'text/typescript'
                          : tab === 'report'
                            ? 'text/markdown'
                            : 'text/vnd.graphviz'
                    );
                  } catch (failure) {
                    setError(readErrorMessage(failure));
                  }
                }}
              >
                Download
              </WorkbenchButton>
            </Actions>
            <p aria-live="polite">{confirmationExport.current === exported ? confirmation : ''}</p>
          </>
        )}
        {tab === 'import' && (
          <>
            <WorkbenchField>
              <label htmlFor={`${prefix}-import-text`}>Workflow JSON to import</label>
              <TextArea
                id={`${prefix}-import-text`}
                data-testid={WORKBENCH_TEST_IDS.importText}
                aria-label="Workflow JSON to import"
                value={text}
                aria-invalid={Boolean(error)}
                onChange={event => {
                  fileReadGeneration.current++;
                  setText(event.target.value);
                  setError('');
                }}
              />
            </WorkbenchField>
            <input
              ref={fileInput}
              type="file"
              accept=".json,application/json"
              aria-label="Choose workflow JSON file"
              hidden
              onChange={async event => {
                const generation = ++fileReadGeneration.current;
                const file = event.target.files?.[0];
                event.target.value = '';
                if (!file) return;
                try {
                  if (file.size > WORKFLOW_JSON_LIMITS.maximumFormattedCharacters * 4) {
                    setError(
                      `Workflow JSON: File size must be at most ${(WORKFLOW_JSON_LIMITS.maximumFormattedCharacters * 4).toLocaleString()} bytes.`
                    );
                    return;
                  }
                  const fileText = await file.text();
                  if (generation !== fileReadGeneration.current) return;
                  setText(fileText);
                  setError('');
                } catch (failure) {
                  if (generation === fileReadGeneration.current)
                    setError(readErrorMessage(failure));
                }
              }}
            />
            <Actions>
              <WorkbenchButton
                aria-label="Choose file: Workflow JSON"
                onClick={() => fileInput.current?.click()}
              >
                Choose file
              </WorkbenchButton>
              <WorkbenchButton
                data-testid={WORKBENCH_TEST_IDS.importApply}
                $variant="primary"
                disabled={!text.trim()}
                onClick={() => {
                  try {
                    if (store.addWorkflow(importWorkflowJSON(text))) store.setDialog(null);
                    else
                      setError(
                        store.getState().notice?.message ?? 'The workflow could not be added.'
                      );
                  } catch (failure) {
                    setError(readErrorMessage(failure));
                  }
                }}
              >
                Import
              </WorkbenchButton>
            </Actions>
          </>
        )}
        {tab === 'history' && <HistoryPromotion store={store} />}
        {error && <p role="alert">{error}</p>}
      </WorkbenchSection>
    </div>
  );
}

function HistoryPromotion({store}: {store: WorkbenchStore}): React.JSX.Element {
  const records = useSyncExternalStore(
    subscribeCatalogInvocations,
    listCatalogInvocations,
    listCatalogInvocations
  );
  const prefix = useId();
  const [excludedIds, setExcludedIds] = useState<readonly string[]>([]);
  const [error, setError] = useState('');
  const selected = useMemo(
    () => records.filter(record => !excludedIds.includes(record.id)),
    [records, excludedIds]
  );
  return (
    <div data-testid={WORKBENCH_TEST_IDS.promoteHistory}>
      <p>
        Runs saved in the Contributor catalog panel can become a workflow. Review the draft before
        adding it.
      </p>
      {!records.length && <p>No saved catalog runs.</p>}
      {records.map(record => (
        <WorkbenchField key={record.id}>
          <label htmlFor={`${prefix}-${record.id}`}>
            <input
              id={`${prefix}-${record.id}`}
              type="checkbox"
              aria-label={`Include catalog run ${record.id}: ${record.operationId}`}
              checked={!excludedIds.includes(record.id)}
              onChange={event =>
                setExcludedIds(previous =>
                  event.target.checked
                    ? previous.filter(id => id !== record.id)
                    : [...previous, record.id]
                )
              }
            />
            {record.id} · {record.operationId}
          </label>
        </WorkbenchField>
      ))}
      <WorkbenchButton
        disabled={!selected.length}
        onClick={() => {
          try {
            const document = makeWorkflowFromCatalogHistory(selected, {
              registry: store.registry,
              getSampleText: (operationId, inputName) =>
                getCatalogSampleInputs(operationId)[inputName],
              takenIds: store.getState().workflows.map(workflow => workflow.id)
            });
            const plan = planWorkflow(document, {registry: store.registry, host: store.host});
            const issues = [
              ...plan.issues,
              ...plan.order.flatMap(stepId => plan.steps[stepId].issues)
            ];
            store.proposeDraft({
              document,
              issues,
              outline: describeWorkflow(document, store.registry),
              source: 'history'
            });
            store.setDialog(null);
          } catch (failure) {
            setError(readErrorMessage(failure));
          }
        }}
      >
        Create draft
      </WorkbenchButton>
      {error && <p role="alert">{error}</p>}
    </div>
  );
}

function Shortcuts(): React.JSX.Element {
  const platform = typeof navigator === 'undefined' ? '' : navigator.platform;
  return (
    <WorkbenchSection>
      <p>Shortcuts apply while focus is in the dock and outside text fields.</p>
      <WorkbenchTable aria-label="Keyboard shortcuts">
        <thead>
          <tr>
            <th>Keys</th>
            <th>Action</th>
          </tr>
        </thead>
        <tbody>
          {WORKBENCH_SHORTCUTS.map(shortcut => (
            <tr key={shortcut.action}>
              <td>
                <kbd>{formatWorkbenchShortcut(shortcut.keys, platform)}</kbd>
              </td>
              <td>{shortcut.label}</td>
            </tr>
          ))}
        </tbody>
      </WorkbenchTable>
      <h3>Ports</h3>
      <Actions>
        {(Object.keys(WORKBENCH_PORT_COLORS) as WorkflowPortKind[]).map(kind => (
          <span key={kind}>
            <WorkbenchPortSwatch $kind={kind} /> {WORKBENCH_PORT_LABELS[kind]}
          </span>
        ))}
      </Actions>
      <h3>Status</h3>
      <Actions>
        {(Object.keys(WORKBENCH_STATUS) as WorkflowStepStatus[]).map(status => (
          <WorkbenchStatusBadge key={status} $status={status} data-status={status}>
            {WORKBENCH_STATUS[status].label}
          </WorkbenchStatusBadge>
        ))}
      </Actions>
    </WorkbenchSection>
  );
}

function DraftApproval({
  draft,
  dock,
  store
}: {
  draft: WorkbenchDraft;
  dock: WorkbenchState['dock'];
  store: WorkbenchStore;
}): React.JSX.Element {
  const issues = useMemo(() => sortWorkbenchDraftIssues(draft.issues), [draft.issues]);
  const sourceLabels = {
    assistant: 'Proposed by the assistant',
    history: 'From catalog history',
    import: 'Imported'
  };
  return (
    <DraftPanel
      role="region"
      aria-label="Proposed workflow"
      data-testid={WORKBENCH_TEST_IDS.draft}
      $bottom={dock.open ? dock.height : WORKBENCH_DOCK.collapsedHeight}
    >
      <Header>
        <h2>{draft.document.title}</h2>
      </Header>
      <WorkbenchSection>
        <p>{sourceLabels[draft.source]}</p>
        <Outline>{draft.outline}</Outline>
        <div aria-live="polite">
          {issues.some(issue => issue.severity === 'error') && (
            <p>This workflow has problems and may not run as proposed.</p>
          )}
          {issues.length > 0 && (
            <ul>
              {issues.map((issue, index) => (
                <li
                  key={`${issue.code}-${index}`}
                  data-status={issue.severity === 'error' ? 'error' : undefined}
                >
                  {issue.severity === 'error'
                    ? 'Error'
                    : issue.severity === 'warning'
                      ? 'Warning'
                      : 'Info'}
                  : {issue.message}
                </li>
              ))}
            </ul>
          )}
        </div>
        <Actions>
          <WorkbenchButton
            $variant="primary"
            data-testid={WORKBENCH_TEST_IDS.draftAccept}
            onClick={() => store.acceptDraft()}
          >
            Add workflow
          </WorkbenchButton>
          <WorkbenchButton
            data-testid={WORKBENCH_TEST_IDS.draftReject}
            onClick={() => store.rejectDraft()}
          >
            Discard
          </WorkbenchButton>
        </Actions>
      </WorkbenchSection>
    </DraftPanel>
  );
}

/** Template gallery, import and export, shortcut help and the draft approval banner. */
export function WorkbenchDialogs({store}: {store: WorkbenchStore}): React.JSX.Element | null {
  const state = useWorkbenchState(store);
  if (!state.dialog && !state.draft) return null;
  const titles = {
    templates: 'Workflow templates',
    'import-export': 'Import and export',
    shortcuts: 'Keyboard shortcuts'
  };
  return (
    <>
      {state.dialog && (
        <Modal key={state.dialog} title={titles[state.dialog]} close={() => store.setDialog(null)}>
          {state.dialog === 'templates' ? (
            <Templates store={store} />
          ) : state.dialog === 'import-export' ? (
            <WorkbenchTransfer store={store} initialTab={state.runLog.length ? 'report' : 'json'} />
          ) : (
            <Shortcuts />
          )}
        </Modal>
      )}
      {state.draft && <DraftApproval draft={state.draft} dock={state.dock} store={store} />}
    </>
  );
}

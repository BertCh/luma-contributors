// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createContext, useContext, useEffect, useMemo, useRef, useState} from 'react';
import styled from 'styled-components';
import {layoutWorkflow, renameWorkflow, updateWorkflowSettings} from '@luma-contributors/kepler';
import {WORKBENCH_COLORS, WORKBENCH_DOCK, WORKBENCH_TEST_IDS} from './workbench-contract';
import {formatWorkbenchShortcut} from './workbench-shortcuts';
import type {WorkbenchState, WorkbenchStore} from './workbench-store';
import {
  WorkbenchButton,
  WorkbenchIconButton,
  WorkbenchSeparator,
  WorkbenchSpacer,
  WorkbenchToolbarRow
} from './workbench-styles';
import {useWorkbenchState} from './use-workbench';

const NARROW_TOOLBAR_WIDTH = 1000;

/** Routes toolbar paste through the dock's live canvas API when the toolbar is docked. */
export const WorkbenchPasteContext = createContext<(() => void) | null>(null);

/** The dock owns restore memory so closing its toolbar does not discard the previous height. */
export const WorkbenchDockHeightContext = createContext<{
  previousHeight: number | null;
  setPreviousHeight(height: number | null): void;
} | null>(null);

const Toolbar = styled(WorkbenchToolbarRow)`
  position: relative;
  flex: none;
  flex-wrap: nowrap;
  gap: 1px;
  padding: 4px;
  overflow: visible;
  & > button, & > span { flex-shrink: 0; }
`;
const WorkflowSelect = styled.select`
  /* Override the surface's shared select defaults so flex cannot collapse the native control. */
  && {
    flex: 0 1 180px;
    min-width: 120px;
    width: 180px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
`;
const WorkflowTitle = styled.input`
  flex: 0 1 180px;
  min-width: 0;
  width: 180px;
`;
const EngineSelect = styled.select`
  && {
    flex: none;
    width: 68px;
    min-width: 68px;
    max-width: 68px;
    padding: 5px 3px;
  }
`;
const TextButton = styled(WorkbenchButton)<{$width: number}>`
  flex: 0 0 ${({$width}) => $width}px;
  width: ${({$width}) => $width}px;
  padding: 0 2px;
`;
const AutoRunLabel = styled.label`
  display: flex;
  align-items: center;
  gap: 3px;
  flex: 0 0 64px;
  min-height: 26px;
  white-space: nowrap;
  & input { margin: 0; }
`;
const GpuIndicator = styled.span`
  display: flex;
  align-items: center;
  gap: 4px;
  flex: 0 0 50px;
  white-space: nowrap;
`;
const GpuDot = styled.span<{$ready: boolean}>`
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: ${({$ready}) => ($ready ? WORKBENCH_COLORS.success : WORKBENCH_COLORS.textMuted)};
`;
const DeleteConfirmation = styled.div`
  position: absolute;
  top: 100%;
  left: 6px;
  z-index: 1;
  display: flex;
  align-items: center;
  gap: 6px;
  max-width: calc(100% - 12px);
  padding: 6px;
  border: 1px solid ${WORKBENCH_COLORS.border};
  background: ${WORKBENCH_COLORS.surfaceRaised};
  & > span { min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
`;

const ICON_PATHS = {
  new: 'M10 3v14M3 10h14',
  duplicate: 'M7 7h10v10H7zM3 13V3h10',
  copy: 'M7 7h10v10H7zM3 13V3h10',
  paste: 'M7 4H4v13h12V4h-3M7 2h6v4H7zM7 9h6M7 12h6',
  saveTemplate: 'M5 2h10v16l-5-3-5 3zM8 6h4M10 4v4',
  delete: 'M3 5h14M7 5V3h6v2M5 5l1 12h8l1-12M8 8v6M12 8v6',
  undo: 'M7 4L3 8l4 4M3 8h8a5 5 0 0 1 0 10',
  redo: 'M13 4l4 4-4 4M17 8H9a5 5 0 0 0 0 10',
  layout: 'M2 3h6v5H2zM12 3h6v5h-6zM7 13h6v5H7zM5 8v3h10V8M10 11v2',
  exchange: 'M6 3v14M3 6l3-3 3 3M14 17V3M11 14l3 3 3-3',
  shortcuts: 'M2 5h16v11H2zM5 8h1M9 8h1M13 8h1M5 11h1M9 11h1M13 11h1M6 14h8',
  maximise: 'M7 3H3v4M13 3h4v4M17 13v4h-4M7 17H3v-4',
  restore: 'M7 7h10v10H7zM3 13V3h10',
  collapse: 'M4 7l6 6 6-6',
  expand: 'M4 13l6-6 6 6',
  rename: 'M3 14l1 3 3-1L17 6l-3-3zM12 5l3 3',
  close: 'M4 4l12 12M16 4L4 16'
} as const;

export function WorkbenchToolbarIcon({name}: {name: keyof typeof ICON_PATHS}): React.JSX.Element {
  return (
    <svg
      viewBox="0 0 20 20"
      width="14"
      height="14"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.5"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      focusable="false"
    >
      <path d={ICON_PATHS[name]} />
    </svg>
  );
}

/** The toolbar responds to its dock width, independent of Kepler's side-panel state. */
export function isWorkbenchToolbarNarrow(width: number): boolean {
  return width < NARROW_TOOLBAR_WIDTH;
}

export function getWorkbenchHeightToggle(
  height: number,
  previousHeight: number | null,
  windowHeight: number
): {height: number; previousHeight: number | null} {
  const maximum = Math.max(0, Math.floor(windowHeight * WORKBENCH_DOCK.maximumHeightFraction));
  if (previousHeight === null) return {height: maximum, previousHeight: height};
  return {
    height: Math.min(maximum, Math.max(WORKBENCH_DOCK.minimumHeight, previousHeight)),
    previousHeight: null
  };
}

export function getWorkbenchPlatform(): 'mac' | 'other' {
  return typeof navigator !== 'undefined' && /Mac|iPhone|iPad/.test(navigator.platform)
    ? 'mac'
    : 'other';
}

export function getWorkbenchRunControls(state: WorkbenchState): {
  count: number;
  canRun: boolean;
  target: string | null;
} {
  const runnable = new Set(['ready', 'stale', 'error', 'done']);
  const steps = state.document?.steps ?? [];
  const count = steps.filter(step => {
    const status = state.runner?.steps[step.id]?.status;
    return status && status !== 'done' && runnable.has(status);
  }).length;
  const selected = state.selectedStepIds.length === 1 ? state.selectedStepIds[0] : undefined;
  const target =
    selected &&
    steps.some(step => step.id === selected) &&
    runnable.has(state.runner?.steps[selected]?.status ?? '')
      ? selected
      : null;
  return {
    count,
    canRun: Boolean(state.document && count && !state.runner?.running),
    target: state.runner?.running ? null : target
  };
}

/** Workflow selector, run controls, undo and redo, layout, templates, import and export. */
export function WorkbenchToolbar({store}: {store: WorkbenchStore}): React.JSX.Element | null {
  const state = useWorkbenchState(store);
  const pasteAtViewportCenter = useContext(WorkbenchPasteContext);
  const {document, runner} = state;
  const controls = useMemo(() => getWorkbenchRunControls(state), [state]);
  const [width, setWidth] = useState(900);
  const toolbar = useRef<HTMLDivElement>(null);
  const titleField = useRef<HTMLInputElement>(null);
  const renameButton = useRef<HTMLButtonElement>(null);
  const workflowSelect = useRef<HTMLSelectElement>(null);
  const deleteButton = useRef<HTMLButtonElement>(null);
  const confirmButton = useRef<HTMLButtonElement>(null);
  const cancelButton = useRef<HTMLButtonElement>(null);
  const cancelBlur = useRef(false);
  const narrow = isWorkbenchToolbarNarrow(width);
  const heightMemory = useContext(WorkbenchDockHeightContext);
  const [localPreviousHeight, setLocalPreviousHeight] = useState<number | null>(null);
  const previousHeight = heightMemory ? heightMemory.previousHeight : localPreviousHeight;
  const setPreviousHeight = heightMemory?.setPreviousHeight ?? setLocalPreviousHeight;
  const [renamingWorkflowId, setRenamingWorkflowId] = useState<string | null>(null);
  const [titleInput, setTitleInput] = useState<{workflowId: string; value: string} | null>(null);
  const [deleteWorkflowId, setDeleteWorkflowId] = useState<string | null>(null);
  const renaming = Boolean(document && renamingWorkflowId === document.id);
  const deleting = Boolean(document && deleteWorkflowId === document.id);
  const title = titleInput?.workflowId === document?.id ? titleInput?.value : document?.title;
  useEffect(() => {
    const element = toolbar.current;
    if (!element) return;
    const measure = () => setWidth(element.getBoundingClientRect().width);
    measure();
    if (typeof ResizeObserver !== 'undefined') {
      const observer = new ResizeObserver(measure);
      observer.observe(element);
      return () => observer.disconnect();
    }
    window.addEventListener('resize', measure);
    return () => window.removeEventListener('resize', measure);
  }, []);
  useEffect(() => {
    if (renaming) titleField.current?.focus();
  }, [renaming]);
  useEffect(() => {
    if (deleting) confirmButton.current?.focus();
  }, [deleting]);
  function commitTitle() {
    if (!cancelBlur.current && document && titleInput?.workflowId === document.id) {
      store.edit(`title:${document.id}`, current =>
        renameWorkflow(current, titleInput.value, current.description)
      );
    }
    cancelBlur.current = false;
    setTitleInput(null);
    setRenamingWorkflowId(null);
  }
  function closeDelete() {
    setDeleteWorkflowId(null);
    deleteButton.current?.focus();
  }
  const platform = getWorkbenchPlatform();
  const hasGpu = Boolean(store.host.getDevice());
  return (
    <Toolbar
      ref={toolbar}
      data-testid={WORKBENCH_TEST_IDS.toolbar}
      data-layout={narrow ? 'narrow' : 'wide'}
      role="toolbar"
      aria-label="Workflow"
    >
      {(!narrow || !renaming) && (
        <WorkflowSelect
          ref={workflowSelect}
          data-testid={WORKBENCH_TEST_IDS.workflowSelect}
          aria-label="Workflow"
          title={document?.title ?? 'Workflow'}
          disabled={!document}
          value={state.activeWorkflowId ?? ''}
          onChange={event => store.setActiveWorkflow(event.target.value)}
        >
          {!document && <option value="">No workflow</option>}
          {state.workflows.map(workflow => (
            <option key={workflow.id} value={workflow.id}>
              {workflow.title}
            </option>
          ))}
        </WorkflowSelect>
      )}
      {(!narrow || renaming) && (
        <WorkflowTitle
          ref={titleField}
          data-testid={WORKBENCH_TEST_IDS.workflowTitle}
          aria-label="Workflow title"
          title="Workflow title"
          disabled={!document}
          value={title ?? ''}
          onBlur={commitTitle}
          onChange={event =>
            document && setTitleInput({workflowId: document.id, value: event.target.value})
          }
          onKeyDown={event => {
            if (event.key === 'Enter' || event.key === 'Escape') {
              event.preventDefault();
              event.stopPropagation();
              cancelBlur.current = event.key === 'Escape';
              event.currentTarget.blur();
              (narrow ? renameButton.current : workflowSelect.current)?.focus();
            }
          }}
        />
      )}
      {narrow && (
        <WorkbenchIconButton
          ref={renameButton}
          aria-label="Rename workflow"
          title="Rename workflow"
          disabled={!document}
          aria-pressed={renaming}
          onClick={() => {
            if (document) {
              setTitleInput({workflowId: document.id, value: document.title});
              setRenamingWorkflowId(document.id);
            }
          }}
        >
          <WorkbenchToolbarIcon name="rename" />
        </WorkbenchIconButton>
      )}
      <WorkbenchIconButton
        data-testid={WORKBENCH_TEST_IDS.workflowNew}
        aria-label="New workflow"
        title="New workflow"
        onClick={() => store.createWorkflow()}
      >
        <WorkbenchToolbarIcon name="new" />
      </WorkbenchIconButton>
      <WorkbenchIconButton
        data-testid={WORKBENCH_TEST_IDS.workflowDuplicate}
        aria-label="Duplicate workflow"
        title="Duplicate workflow"
        disabled={!document}
        onClick={() => document && store.duplicateWorkflow(document.id)}
      >
        <WorkbenchToolbarIcon name="duplicate" />
      </WorkbenchIconButton>
      <WorkbenchIconButton
        ref={deleteButton}
        data-testid={WORKBENCH_TEST_IDS.workflowDelete}
        aria-label="Delete workflow"
        title="Delete workflow"
        disabled={!document}
        aria-expanded={deleting}
        onClick={() => document && setDeleteWorkflowId(document.id)}
      >
        <WorkbenchToolbarIcon name="delete" />
      </WorkbenchIconButton>
      {document && deleting && (
        <DeleteConfirmation
          role="dialog"
          aria-label={`Delete "${document.title}"?`}
          onKeyDown={event => {
            event.stopPropagation();
            if (event.key === 'Escape') {
              event.preventDefault();
              closeDelete();
            }
            if (event.key === 'Tab') {
              event.preventDefault();
              (event.target === confirmButton.current
                ? cancelButton.current
                : confirmButton.current
              )?.focus();
            }
          }}
        >
          <span title={document.title}>Delete "{document.title}"?</span>
          <WorkbenchButton
            ref={confirmButton}
            $variant="danger"
            aria-label={`Confirm delete "${document.title}"`}
            onClick={() => {
              store.deleteWorkflow(document.id);
              setDeleteWorkflowId(null);
            }}
          >
            Confirm
          </WorkbenchButton>
          <WorkbenchButton ref={cancelButton} onClick={closeDelete}>
            Cancel
          </WorkbenchButton>
        </DeleteConfirmation>
      )}
      <WorkbenchSeparator />
      <TextButton
        $width={54}
        data-testid={WORKBENCH_TEST_IDS.runAll}
        $variant="primary"
        disabled={!controls.canRun}
        title={`Run (${formatWorkbenchShortcut('Shift+Enter', platform)})`}
        onClick={() => {
          void store.run();
        }}
      >
        Run ({controls.count.toLocaleString()})
      </TextButton>
      <TextButton
        $width={70}
        data-testid={WORKBENCH_TEST_IDS.runToHere}
        disabled={!controls.target}
        title={`Run to here (${formatWorkbenchShortcut('Enter', platform)})`}
        onClick={() => {
          if (controls.target) void store.run({target: controls.target});
        }}
      >
        Run to here
      </TextButton>
      {runner?.running && (
        <TextButton
          $width={44}
          data-testid={WORKBENCH_TEST_IDS.cancel}
          title="Cancel (Esc)"
          onClick={() => store.cancel()}
        >
          Cancel
        </TextButton>
      )}
      <WorkbenchSeparator />
      <WorkbenchIconButton
        data-testid={WORKBENCH_TEST_IDS.undo}
        disabled={!document || !state.canUndo}
        aria-label="Undo"
        title={`Undo (${formatWorkbenchShortcut('Mod+Z', platform)})`}
        onClick={() => store.undo()}
      >
        <WorkbenchToolbarIcon name="undo" />
      </WorkbenchIconButton>
      <WorkbenchIconButton
        data-testid={WORKBENCH_TEST_IDS.redo}
        disabled={!document || !state.canRedo}
        aria-label="Redo"
        title={`Redo (${formatWorkbenchShortcut('Shift+Mod+Z', platform)})`}
        onClick={() => store.redo()}
      >
        <WorkbenchToolbarIcon name="redo" />
      </WorkbenchIconButton>
      <WorkbenchIconButton
        data-testid={WORKBENCH_TEST_IDS.copySteps}
        aria-label="Copy steps"
        title={`Copy steps (${formatWorkbenchShortcut('Mod+C', platform)})`}
        disabled={!document || state.selectedStepIds.length === 0}
        onClick={() => store.copySelection()}
      >
        <WorkbenchToolbarIcon name="copy" />
      </WorkbenchIconButton>
      <WorkbenchIconButton
        data-testid={WORKBENCH_TEST_IDS.pasteSteps}
        aria-label="Paste steps"
        title={`Paste ${state.clipboardStepCount.toLocaleString()} ${state.clipboardStepCount === 1 ? 'step' : 'steps'}`}
        disabled={!document || state.clipboardStepCount === 0}
        onClick={pasteAtViewportCenter ?? (() => store.paste())}
      >
        <WorkbenchToolbarIcon name="paste" />
      </WorkbenchIconButton>
      <TextButton
        $width={54}
        data-testid={WORKBENCH_TEST_IDS.addStep}
        disabled={!document}
        title="Add step (N)"
        onClick={() => store.openPalette()}
      >
        Add step
      </TextButton>
      <WorkbenchIconButton
        data-testid={WORKBENCH_TEST_IDS.layout}
        disabled={!document}
        aria-label="Lay out"
        title="Lay out (L)"
        onClick={() => store.edit('layout', current => layoutWorkflow(current))}
      >
        <WorkbenchToolbarIcon name="layout" />
      </WorkbenchIconButton>
      <WorkbenchSeparator />
      <WorkbenchIconButton
        data-testid={WORKBENCH_TEST_IDS.saveTemplate}
        aria-label="Save as template"
        title="Save as template"
        disabled={!document || document.steps.length === 0}
        onClick={() => document && store.saveTemplate(document.id)}
      >
        <WorkbenchToolbarIcon name="saveTemplate" />
      </WorkbenchIconButton>
      <TextButton
        $width={64}
        data-testid={WORKBENCH_TEST_IDS.templates}
        onClick={() => store.setDialog('templates')}
      >
        Templates
      </TextButton>
      <WorkbenchIconButton
        data-testid={WORKBENCH_TEST_IDS.importExport}
        aria-label="Import and export"
        title="Import and export"
        onClick={() => store.setDialog('import-export')}
      >
        <WorkbenchToolbarIcon name="exchange" />
      </WorkbenchIconButton>
      <WorkbenchIconButton
        disabled={!document}
        aria-label="Shortcuts"
        title="Shortcuts"
        onClick={() => store.setDialog('shortcuts')}
      >
        <WorkbenchToolbarIcon name="shortcuts" />
      </WorkbenchIconButton>
      <WorkbenchSpacer />
      <EngineSelect
        data-testid={WORKBENCH_TEST_IDS.engine}
        aria-label="Engine"
        title="Engine"
        disabled={!document}
        value={document?.settings.engine ?? 'auto'}
        onChange={event =>
          store.edit('settings:engine', current =>
            updateWorkflowSettings(current, {engine: event.target.value === 'cpu' ? 'cpu' : 'auto'})
          )
        }
      >
        <option value="auto">GPU</option>
        <option value="cpu">CPU</option>
      </EngineSelect>
      <AutoRunLabel>
        <input
          data-testid={WORKBENCH_TEST_IDS.autoRun}
          type="checkbox"
          disabled={!document}
          checked={document?.settings.autoRun ?? false}
          onChange={event =>
            store.edit('settings:auto-run', current =>
              updateWorkflowSettings(current, {autoRun: event.target.checked})
            )
          }
        />
        Auto-run
      </AutoRunLabel>
      <GpuIndicator title={hasGpu ? 'GPU ready' : 'No GPU device available'}>
        <GpuDot $ready={hasGpu} aria-hidden="true" />
        {hasGpu ? 'GPU' : 'No GPU'}
      </GpuIndicator>
      <WorkbenchIconButton
        aria-label={previousHeight === null ? 'Maximise workbench' : 'Restore workbench'}
        title={previousHeight === null ? 'Maximise workbench' : 'Restore workbench'}
        disabled={!document}
        aria-pressed={previousHeight !== null}
        onClick={() => {
          const next = getWorkbenchHeightToggle(
            state.dock.height,
            previousHeight,
            window.innerHeight
          );
          setPreviousHeight(next.previousHeight);
          store.setDock({height: next.height});
        }}
      >
        <WorkbenchToolbarIcon name={previousHeight === null ? 'maximise' : 'restore'} />
      </WorkbenchIconButton>
      <WorkbenchIconButton
        data-testid={WORKBENCH_TEST_IDS.dockToggle}
        aria-label="Collapse workbench"
        title="Collapse workbench"
        disabled={!document}
        onClick={() => store.setDock({open: false})}
      >
        <WorkbenchToolbarIcon name="collapse" />
      </WorkbenchIconButton>
    </Toolbar>
  );
}

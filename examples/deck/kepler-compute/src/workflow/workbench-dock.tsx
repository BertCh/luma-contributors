// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {useCallback, useEffect, useMemo, useRef, useState, type KeyboardEvent} from 'react';
import styled, {css} from 'styled-components';
import {
  duplicateWorkflowSteps,
  layoutWorkflow,
  removeWorkflowSteps,
  updateWorkflowStep,
  type WorkflowStepStatus,
  type WorkflowOperationRegistry
} from '@luma-contributors/kepler';
import {WorkbenchCanvas, type WorkbenchCanvasApi} from './workbench-canvas';
import {
  WORKBENCH_COLORS,
  WORKBENCH_DOCK,
  WORKBENCH_STATUS,
  WORKBENCH_TEST_IDS,
  type WorkbenchShortcutAction
} from './workbench-contract';
import {WorkbenchInspector} from './workbench-inspector';
import {WorkbenchPalette} from './workbench-palette';
import {WorkbenchRunLog, formatWorkbenchDuration} from './workbench-run-log';
import {isEditableTarget, matchWorkbenchShortcut} from './workbench-shortcuts';
import type {WorkbenchState, WorkbenchStore} from './workbench-store';
import {
  WorkbenchButton,
  WorkbenchIconButton,
  WorkbenchNoticeBar,
  WorkbenchSpacer,
  WorkbenchSurface,
  WorkbenchToolbarRow
} from './workbench-styles';
import {
  WorkbenchToolbar,
  WorkbenchToolbarIcon,
  WorkbenchDockHeightContext,
  WorkbenchPasteContext,
  getWorkbenchPlatform,
  getWorkbenchRunControls
} from './workbench-toolbar';
import {WorkbenchStoreContext, useWorkbenchState} from './use-workbench';

// Open, the dock spans the window below Kepler, which the host shortens by the dock height so the
// time widget, map controls and side panel stay above it. Collapsed, it is a pill at the top left
// of the map, clear of the time widget and the scale bar.
const Dock = styled(WorkbenchSurface)<{$height: number; $open: boolean}>`
  position: fixed;
  left: 0;
  right: 0;
  bottom: 0;
  z-index: 50;
  height: ${({$height}) => $height}px;
  display: flex;
  flex-direction: column;
  border-top: 1px solid ${WORKBENCH_COLORS.border};
  box-shadow: 0 -6px 20px ${WORKBENCH_COLORS.surfaceSunken}99;
  ${({$open}) =>
    !$open &&
    css`
    left: calc(var(--workbench-dock-left, 0px) + 48px);
    right: auto;
    top: 10px;
    bottom: auto;
    width: max-content;
    max-width: calc(100vw - var(--workbench-dock-left, 0px) - 120px);
    border: 1px solid ${WORKBENCH_COLORS.border};
    border-radius: 18px;
    overflow: hidden;
  `}
`;
const CollapsedBar = styled(WorkbenchToolbarRow)`
  height: 100%;
  min-height: 0;
  flex: none;
  padding: 4px 6px;
  border: 0;
  background: transparent;
  overflow: hidden;
  & > button { flex: none; }
`;
const CollapsedTitle = styled.span`
  min-width: 0;
  max-width: 28ch;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
`;
const CollapsedSummary = styled.span`
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
`;
const ResizeHandle = styled.div`
  height: 24px;
  flex: none;
  cursor: ns-resize;
  touch-action: none;
  border-bottom: 1px solid ${WORKBENCH_COLORS.border};
  &:focus-visible { outline: 2px solid ${WORKBENCH_COLORS.focus}; outline-offset: -2px; }
`;
const Content = styled.div`
  display: flex;
  flex: 1;
  min-height: 24px;
  position: relative;
`;
const CanvasSlot = styled.div`
  flex: 1;
  min-width: 0;
`;
const InspectorSlot = styled.div`
  width: ${WORKBENCH_DOCK.inspectorWidth}px;
  flex: none;
  overflow: auto;
  border-left: 1px solid ${WORKBENCH_COLORS.border};
`;
const LiveRegion = styled.div`
  position: absolute;
  width: 1px;
  height: 1px;
  overflow: hidden;
  clip-path: inset(50%);
`;

export function clampWorkbenchHeight(height: number, windowHeight: number): number {
  const maximum = Math.max(0, Math.floor(windowHeight * WORKBENCH_DOCK.maximumHeightFraction));
  return Math.max(Math.min(WORKBENCH_DOCK.minimumHeight, maximum), Math.min(maximum, height));
}

export function shouldHandleTabAsAddStep({
  targetInsideCanvas,
  canvasPointerFocused
}: {
  targetInsideCanvas: boolean;
  canvasPointerFocused: boolean;
}): boolean {
  return targetInsideCanvas && canvasPointerFocused;
}

export function subscribeWorkbenchPaletteClose(store: WorkbenchStore, onClose: () => void) {
  let wasOpen = store.getState().palette.open;
  return store.subscribe(() => {
    const open = store.getState().palette.open;
    const closed = wasOpen && !open;
    wasOpen = open;
    if (closed) onClose();
  });
}

export function truncateWorkbenchTitle(title: string): string {
  const characters = Array.from(title);
  return characters.length > 28 ? `${characters.slice(0, 27).join('')}…` : title;
}

export function formatWorkbenchSummary(state: WorkbenchState): string {
  if (!state.document) return 'No workflow';
  if (state.document.steps.length === 0) return 'No steps';
  const counts = new Map<WorkflowStepStatus, number>();
  for (const step of state.document.steps) {
    const status =
      state.runner?.steps[step.id]?.status ?? (step.disabled ? 'disabled' : 'not-configured');
    counts.set(status, (counts.get(status) ?? 0) + 1);
  }
  return (Object.keys(WORKBENCH_STATUS) as WorkflowStepStatus[])
    .filter(status => counts.has(status))
    .map(
      status =>
        `${counts.get(status)!.toLocaleString()} ${WORKBENCH_STATUS[status].label.toLowerCase()}`
    )
    .join(' · ');
}

export function formatWorkbenchAnnouncement(
  state: WorkbenchState,
  registry?: WorkflowOperationRegistry
): string {
  if (state.runner?.running) {
    const step = state.document?.steps.find(
      candidate => candidate.id === state.runner?.runningStepId
    );
    return step
      ? `Running ${step.label ?? registry?.get(step.operationId)?.title ?? step.operationId}.`
      : 'Run started. Running workflow.';
  }
  const run = state.runner?.lastRun ?? state.runLog[0];
  if (!run) return '';
  if (run.status === 'error')
    return `Run failed: ${run.steps.find(step => step.error)?.error?.message ?? 'Review the run log.'}`;
  if (run.status === 'cancelled') return 'Run cancelled.';
  return `Run finished: ${run.steps.filter(step => step.status === 'done').length.toLocaleString()} steps done in ${formatWorkbenchDuration(run.durationMilliseconds)}`;
}

/** Dispatch is separate from the DOM listener so all actions can be checked in node specs. */
export function applyWorkbenchShortcut(
  action: WorkbenchShortcutAction,
  store: WorkbenchStore,
  canvasApi: WorkbenchCanvasApi | null
): void {
  const state = store.getState();
  const selected = state.selectedStepIds;
  if (action === 'cancel-or-close') {
    if (state.dialog) store.setDialog(null);
    else if (state.palette.open) store.closePalette();
    else if (state.runner?.running) store.cancel();
    return;
  }
  if (!state.document) return;
  switch (action) {
    case 'add-step':
      store.openPalette({position: canvasApi?.getViewportCenter()});
      break;
    case 'run-to-here': {
      const {target} = getWorkbenchRunControls(state);
      if (target) void store.run({target});
      break;
    }
    case 'run-all':
      if (getWorkbenchRunControls(state).canRun) void store.run();
      break;
    case 'toggle-disabled':
    case 'toggle-display':
      store.edit(`${action}:${selected.join(':')}`, document =>
        selected.reduce((current, stepId) => {
          const step = current.steps.find(candidate => candidate.id === stepId);
          if (!step) return current;
          if (action === 'toggle-disabled')
            return updateWorkflowStep(current, step.id, {disabled: !step.disabled});
          const output = store.registry.get(step.operationId)?.outputs[0]?.name;
          if (!output) return current;
          const display = step.display?.includes(output)
            ? step.display.filter(name => name !== output)
            : [...(step.display ?? []), output];
          return updateWorkflowStep(current, step.id, {display});
        }, document)
      );
      break;
    case 'frame':
      canvasApi?.frame(selected.length ? selected : undefined);
      break;
    case 'layout':
      store.edit('layout', document => layoutWorkflow(document));
      break;
    case 'delete':
      store.edit(`delete:${selected.join(':')}`, document =>
        removeWorkflowSteps(document, selected)
      );
      break;
    case 'duplicate': {
      let duplicated: string[] = [];
      if (
        store.edit(`duplicate:${selected.join(':')}`, document => {
          const result = duplicateWorkflowSteps(document, selected);
          duplicated = result.stepIds;
          return result.document;
        })
      )
        store.select(duplicated);
      break;
    }
    case 'copy':
      store.copySelection();
      break;
    case 'cut':
      store.cutSelection();
      break;
    case 'paste':
      store.paste(canvasApi?.getViewportCenter());
      break;
    case 'undo':
      if (state.canUndo) store.undo();
      break;
    case 'redo':
      if (state.canRedo) store.redo();
      break;
    case 'select-all':
      store.select(state.document.steps.map(step => step.id));
      break;
  }
}

/** Bottom dock: collapsed summary bar or toolbar, canvas, inspector and run log. */
export function WorkbenchDock({store}: {store: WorkbenchStore}): React.JSX.Element | null {
  const state = useWorkbenchState(store);
  const [previousHeight, setPreviousHeight] = useState<number | null>(null);
  const canvasApi = useRef<WorkbenchCanvasApi | null>(null);
  const canvasPointerFocused = useRef(false);
  const setCanvasApi = useCallback((api: WorkbenchCanvasApi | null) => {
    canvasApi.current = api;
  }, []);
  const pasteAtViewportCenter = useCallback(() => {
    applyWorkbenchShortcut('paste', store, canvasApi.current);
  }, [store]);
  const drag = useRef<{pointerId: number; startY: number; height: number} | null>(null);
  const summary = useMemo(() => formatWorkbenchSummary(state), [state.document, state.runner]);
  const announcement = useMemo(
    () => formatWorkbenchAnnouncement(state, store.registry),
    [state.document, state.runner, state.runLog, store.registry]
  );
  const controls = getWorkbenchRunControls(state);
  useEffect(
    () =>
      subscribeWorkbenchPaletteClose(store, () => {
        canvasPointerFocused.current = false;
      }),
    [store]
  );
  useEffect(() => {
    function resize() {
      const height = store.getState().dock.height;
      const clamped = clampWorkbenchHeight(height, window.innerHeight);
      if (height !== clamped) store.setDock({height: clamped});
    }
    resize();
    window.addEventListener('resize', resize);
    return () => window.removeEventListener('resize', resize);
  }, [store]);
  function handleKeyDown(event: KeyboardEvent<HTMLDivElement>) {
    if (!dispatchShortcut(event) && event.key === 'Tab') canvasPointerFocused.current = false;
  }
  function dispatchShortcut(event: KeyboardEvent<HTMLDivElement>): boolean {
    if (event.defaultPrevented || (event.key === 'Tab' && event.shiftKey)) return false;
    if (state.dialog && event.key !== 'Escape') return false;
    if (isEditableTarget(event.target) && !state.dialog) return false;
    // Native controls keep Enter, Space and Tab so the toolbar and run log remain navigable.
    const target = event.target as HTMLElement;
    if (
      ['Enter', 'Tab', ' '].includes(event.key) &&
      target.closest('button, a[href], summary, [role="button"], [role="tab"], [role="separator"]')
    )
      return false;
    // Keyboard focus must be able to leave the canvas for the inspector.
    if (
      event.key === 'Tab' &&
      !shouldHandleTabAsAddStep({
        targetInsideCanvas: Boolean(target.closest(`[data-testid="${WORKBENCH_TEST_IDS.canvas}"]`)),
        canvasPointerFocused: canvasPointerFocused.current
      })
    )
      return false;
    const action = matchWorkbenchShortcut(event, getWorkbenchPlatform());
    if (!action) return false;
    event.preventDefault();
    applyWorkbenchShortcut(action, store, canvasApi.current);
    return true;
  }
  return (
    <WorkbenchStoreContext.Provider value={store}>
      <Dock
        data-testid={WORKBENCH_TEST_IDS.dock}
        role="region"
        aria-label="Analysis workbench"
        tabIndex={-1}
        $open={state.dock.open}
        $height={state.dock.open ? state.dock.height : WORKBENCH_DOCK.collapsedHeight}
        onKeyDown={handleKeyDown}
        onPointerDownCapture={event => {
          if ((event.target as Element).closest(`[data-testid="${WORKBENCH_TEST_IDS.canvas}"]`))
            canvasPointerFocused.current = true;
        }}
        onBlurCapture={event => {
          const canvas = (event.target as Element).closest(
            `[data-testid="${WORKBENCH_TEST_IDS.canvas}"]`
          );
          if (canvas && !canvas.contains(event.relatedTarget as Node | null))
            canvasPointerFocused.current = false;
        }}
      >
        <LiveRegion aria-live="polite" aria-atomic="true">
          {announcement}
        </LiveRegion>
        {!state.dock.open ? (
          <CollapsedBar>
            <WorkbenchIconButton
              data-testid={WORKBENCH_TEST_IDS.dockToggle}
              aria-label="Expand workbench"
              title="Expand workbench"
              onClick={() => store.setDock({open: true})}
            >
              <WorkbenchToolbarIcon name="expand" />
            </WorkbenchIconButton>
            <CollapsedTitle title={state.document?.title ?? 'Analysis workbench'}>
              {truncateWorkbenchTitle(state.document?.title ?? 'Analysis workbench')}
            </CollapsedTitle>
            <CollapsedSummary data-testid={WORKBENCH_TEST_IDS.dockSummary} title={summary}>
              {summary}
            </CollapsedSummary>
            {!state.document && (
              <WorkbenchIconButton
                data-testid={WORKBENCH_TEST_IDS.workflowNew}
                aria-label="New workflow"
                title="New workflow"
                onClick={() => store.createWorkflow()}
              >
                <WorkbenchToolbarIcon name="new" />
              </WorkbenchIconButton>
            )}
            <WorkbenchButton
              data-testid={WORKBENCH_TEST_IDS.runAll}
              $variant="primary"
              disabled={!controls.canRun}
              onClick={() => {
                void store.run();
              }}
            >
              Run
            </WorkbenchButton>
            <WorkbenchButton
              data-testid={WORKBENCH_TEST_IDS.cancel}
              disabled={!state.runner?.running}
              onClick={() => store.cancel()}
            >
              Cancel
            </WorkbenchButton>
          </CollapsedBar>
        ) : (
          <>
            <ResizeHandle
              data-testid={WORKBENCH_TEST_IDS.dockResize}
              role="separator"
              aria-label="Workbench height"
              aria-orientation="horizontal"
              aria-valuenow={state.dock.height}
              aria-valuemin={
                typeof window === 'undefined'
                  ? WORKBENCH_DOCK.minimumHeight
                  : Math.min(
                      WORKBENCH_DOCK.minimumHeight,
                      Math.floor(window.innerHeight * WORKBENCH_DOCK.maximumHeightFraction)
                    )
              }
              aria-valuemax={
                typeof window === 'undefined'
                  ? undefined
                  : Math.floor(window.innerHeight * WORKBENCH_DOCK.maximumHeightFraction)
              }
              tabIndex={0}
              onKeyDown={event => {
                if (event.key !== 'ArrowUp' && event.key !== 'ArrowDown') return;
                event.preventDefault();
                event.stopPropagation();
                store.setDock({
                  height: clampWorkbenchHeight(
                    state.dock.height + (event.key === 'ArrowUp' ? 24 : -24),
                    window.innerHeight
                  )
                });
              }}
              onPointerDown={event => {
                if (event.button !== 0) return;
                event.preventDefault();
                event.currentTarget.focus();
                event.currentTarget.setPointerCapture(event.pointerId);
                drag.current = {
                  pointerId: event.pointerId,
                  startY: event.clientY,
                  height: state.dock.height
                };
              }}
              onPointerMove={event => {
                const current = drag.current;
                if (current?.pointerId === event.pointerId)
                  store.setDock({
                    height: clampWorkbenchHeight(
                      current.height + current.startY - event.clientY,
                      window.innerHeight
                    )
                  });
              }}
              onPointerUp={event => {
                if (drag.current?.pointerId === event.pointerId) {
                  drag.current = null;
                  event.currentTarget.releasePointerCapture(event.pointerId);
                }
              }}
              onPointerCancel={() => {
                drag.current = null;
              }}
              onLostPointerCapture={() => {
                drag.current = null;
              }}
            />
            <WorkbenchDockHeightContext.Provider value={{previousHeight, setPreviousHeight}}>
              <WorkbenchPasteContext.Provider value={pasteAtViewportCenter}>
                <WorkbenchToolbar store={store} />
              </WorkbenchPasteContext.Provider>
            </WorkbenchDockHeightContext.Provider>
            {state.notice && (
              <WorkbenchNoticeBar
                data-testid={WORKBENCH_TEST_IDS.notice}
                $kind={state.notice.kind}
                role="status"
              >
                <span>{state.notice.message}</span>
                <WorkbenchSpacer />
                <WorkbenchIconButton
                  aria-label="Dismiss"
                  title="Dismiss"
                  onClick={() => store.setNotice(null)}
                >
                  <WorkbenchToolbarIcon name="close" />
                </WorkbenchIconButton>
              </WorkbenchNoticeBar>
            )}
            <Content>
              <CanvasSlot>
                <WorkbenchCanvas store={store} onApi={setCanvasApi} />
              </CanvasSlot>
              <InspectorSlot>
                <WorkbenchInspector store={store} />
              </InspectorSlot>
              <WorkbenchPalette store={store} />
            </Content>
            <WorkbenchRunLog store={store} />
          </>
        )}
      </Dock>
    </WorkbenchStoreContext.Provider>
  );
}
